// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package workflow

import (
	"errors"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// This file holds end-to-end output-validation tests that drive the
// full scheduler pipeline, complementing the per-component unit tests
// in base_node_test.go, function_node_test.go, tool_node_test.go and
// scheduler_test.go. The scenarios use realistic struct types (nested
// structs and enum-like string fields) rather than bare int/string to
// mirror how real workflows declare their schemas.

// priority is an enum-like string used in the integration schemas.
type priority string

const (
	priorityLow  priority = "low"
	priorityHigh priority = "high"
)

// ticket is a realistic nested struct used as a node output payload.
type ticket struct {
	ID       int      `json:"id"`
	Title    string   `json:"title"`
	Priority priority `json:"priority"`
	Assignee assignee `json:"assignee"`
	Labels   []string `json:"labels"`
}

type assignee struct {
	Name  string `json:"name"`
	Email string `json:"email"`
}

// runForOutput drives the workflow to completion and returns the last
// non-nil Event.Output observed and the first error.
func runForOutput(t *testing.T, w *Workflow, ctx agent.InvocationContext) (any, error) {
	t.Helper()
	var last any
	for ev, err := range w.Run(ctx) {
		if err != nil {
			return nil, err
		}
		if ev.Output != nil {
			last = ev.Output
		}
	}
	return last, nil
}

// contentOutputNode yields a single event whose Event.Output is a
// *genai.Content carrying the given parts. Used to exercise the
// scheduler-driven Content fallback path.
type contentOutputNode struct {
	BaseNode
	content *genai.Content
}

func (n *contentOutputNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx.InvocationID())
		ev.Output = n.content
		yield(ev, nil)
	}
}

// eventPassthroughNode yields an event whose Output is itself a
// *session.Event — a framework control value the scheduler must never
// schema-validate.
type eventPassthroughNode struct {
	BaseNode
	inner *session.Event
}

func (n *eventPassthroughNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx.InvocationID())
		ev.Output = n.inner
		yield(ev, nil)
	}
}

// TestIntegration_FunctionNode_StructOutput verifies that a
// FunctionNode returning a struct matching its inferred output schema
// has that value forwarded by the scheduler unchanged.
func TestIntegration_FunctionNode_StructOutput(t *testing.T) {
	want := ticket{
		ID:       7,
		Title:    "Fix login",
		Priority: priorityHigh,
		Assignee: assignee{Name: "Ada", Email: "ada@example.com"},
		Labels:   []string{"bug", "auth"},
	}
	node := NewFunctionNode("make_ticket",
		func(ctx agent.Context, _ any) (ticket, error) {
			return want, nil
		}, defaultNodeConfig)

	w := mustNew(t, Chain(Start, node))
	got, err := runForOutput(t, w, newSeededMockCtx(t))
	if err != nil {
		t.Fatalf("workflow failed: %v", err)
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("output mismatch (-want +got):\n%s", diff)
	}
}

// TestIntegration_FunctionNode_OutputSchemaMismatch verifies that a
// FunctionNode whose output violates an explicit override schema
// surfaces a validation error through the scheduler and ends the run.
func TestIntegration_FunctionNode_OutputSchemaMismatch(t *testing.T) {
	type wantShape struct {
		ID int `json:"id"`
	}
	oschema, err := jsonschema.For[wantShape](nil)
	if err != nil {
		t.Fatalf("jsonschema.For failed: %v", err)
	}
	// The function returns {"id": "not-an-int"} which violates the
	// override schema (id must be an integer).
	ischema, err := jsonschema.For[struct{}](nil)
	if err != nil {
		t.Fatalf("jsonschema.For failed: %v", err)
	}
	node, err := NewFunctionNodeWithSchema[any, map[string]any]("bad",
		func(ctx agent.Context, _ any) (map[string]any, error) {
			return map[string]any{"id": "not-an-int"}, nil
		}, ischema, oschema, defaultNodeConfig)
	if err != nil {
		t.Fatalf("NewFunctionNodeWithSchema failed: %v", err)
	}

	w := mustNew(t, Chain(Start, node))
	if _, err := runForOutput(t, w, newSeededMockCtx(t)); err == nil {
		t.Fatal("expected validation error from scheduler, got nil")
	}
}

// TestIntegration_ContentFallback drives the *genai.Content fallback
// through the scheduler: a node yields raw model Content and the
// scheduler projects it onto the node's declared output schema.
func TestIntegration_ContentFallback(t *testing.T) {
	t.Run("json_content_parsed_against_object_schema", func(t *testing.T) {
		schema := resolveTestSchema[assignee](t)
		node := &contentOutputNode{
			BaseNode: NewBaseNodeWithSchemas("c", "", defaultNodeConfig, nil, schema),
			content: &genai.Content{Parts: []*genai.Part{
				{Text: `{"name":"Ada",`},
				{Text: `"email":"ada@example.com"}`},
			}},
		}
		w := mustNew(t, Chain(Start, node))
		got, err := runForOutput(t, w, newSeededMockCtx(t))
		if err != nil {
			t.Fatalf("workflow failed: %v", err)
		}
		gotMap, ok := got.(map[string]any)
		if !ok {
			t.Fatalf("expected map[string]any, got %T", got)
		}
		if gotMap["name"] != "Ada" || gotMap["email"] != "ada@example.com" {
			t.Errorf("got %v, want name=Ada email=ada@example.com", gotMap)
		}
	})

	t.Run("string_schema_returns_text", func(t *testing.T) {
		schema := resolveTestSchema[string](t)
		node := &contentOutputNode{
			BaseNode: NewBaseNodeWithSchemas("c", "", defaultNodeConfig, nil, schema),
			content:  &genai.Content{Parts: []*genai.Part{{Text: "plain "}, {Text: "answer"}}},
		}
		w := mustNew(t, Chain(Start, node))
		got, err := runForOutput(t, w, newSeededMockCtx(t))
		if err != nil {
			t.Fatalf("workflow failed: %v", err)
		}
		if got != "plain answer" {
			t.Errorf("got %v, want %q", got, "plain answer")
		}
	})

	t.Run("non_json_against_object_schema_errors", func(t *testing.T) {
		schema := resolveTestSchema[assignee](t)
		node := &contentOutputNode{
			BaseNode: NewBaseNodeWithSchemas("c", "", defaultNodeConfig, nil, schema),
			content:  &genai.Content{Parts: []*genai.Part{{Text: "this is not json"}}},
		}
		w := mustNew(t, Chain(Start, node))
		if _, err := runForOutput(t, w, newSeededMockCtx(t)); err == nil {
			t.Fatal("expected validation error, got nil")
		}
	})
}

// TestIntegration_PassthroughTypes verifies that framework control
// values routed through Event.Output are forwarded unchanged even when
// the node declares an output schema.
func TestIntegration_PassthroughTypes(t *testing.T) {
	schema := resolveTestSchema[assignee](t)
	inner := &session.Event{Author: "inner"}
	node := &eventPassthroughNode{
		BaseNode: NewBaseNodeWithSchemas("p", "", defaultNodeConfig, nil, schema),
		inner:    inner,
	}
	w := mustNew(t, Chain(Start, node))
	got, err := runForOutput(t, w, newSeededMockCtx(t))
	if err != nil {
		t.Fatalf("workflow failed: %v", err)
	}
	if got != any(inner) {
		t.Errorf("expected the *session.Event to pass through unchanged, got %T", got)
	}
}

// TestIntegration_EndToEndChain runs a multi-step workflow where every
// step declares schemas and the validated value of each node flows as
// the next node's input. Verifies the consumer only ever sees
// validated values.
func TestIntegration_EndToEndChain(t *testing.T) {
	type rawOrder struct {
		Item string `json:"item"`
		Qty  int    `json:"qty"`
	}
	type pricedOrder struct {
		Item  string  `json:"item"`
		Qty   int     `json:"qty"`
		Total float64 `json:"total"`
	}

	// seed -> price -> terminal. Each node returns a typed struct,
	// validated by the scheduler against the inferred output schema.
	seed := NewFunctionNode("seed",
		func(ctx agent.Context, _ any) (rawOrder, error) {
			return rawOrder{Item: "widget", Qty: 3}, nil
		}, defaultNodeConfig)
	price := NewFunctionNode("price",
		func(ctx agent.Context, in rawOrder) (pricedOrder, error) {
			return pricedOrder{Item: in.Item, Qty: in.Qty, Total: float64(in.Qty) * 2.5}, nil
		}, defaultNodeConfig)

	w := mustNew(t, Chain(Start, seed, price))
	got, err := runForOutput(t, w, newSeededMockCtx(t))
	if err != nil {
		t.Fatalf("workflow failed: %v", err)
	}
	want := pricedOrder{Item: "widget", Qty: 3, Total: 7.5}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("terminal output mismatch (-want +got):\n%s", diff)
	}
}

// TestIntegration_TerminalOutputConstraint verifies the runtime
// single-terminal-output invariant end-to-end: a single
// output-producing terminal succeeds, while several output-producing
// terminals fail with ErrMultipleTerminalOutputs.
func TestIntegration_TerminalOutputConstraint(t *testing.T) {
	t.Run("single_terminal_ok", func(t *testing.T) {
		a := NewFunctionNode("a",
			func(ctx agent.Context, _ any) (ticket, error) {
				return ticket{ID: 1, Title: "t", Priority: priorityLow}, nil
			}, defaultNodeConfig)
		w := mustNew(t, Chain(Start, a))
		if _, err := runForOutput(t, w, newSeededMockCtx(t)); err != nil {
			t.Fatalf("expected success, got %v", err)
		}
	})

	t.Run("multiple_terminals_error", func(t *testing.T) {
		a := NewFunctionNode("a",
			func(ctx agent.Context, _ any) (string, error) { return "A", nil },
			defaultNodeConfig)
		b := NewFunctionNode("b",
			func(ctx agent.Context, _ any) (string, error) { return "B", nil },
			defaultNodeConfig)
		w := mustNew(t, []Edge{{From: Start, To: a}, {From: Start, To: b}})
		_, err := runForOutput(t, w, newSeededMockCtx(t))
		if !errors.Is(err, ErrMultipleTerminalOutputs) {
			t.Fatalf("got %v, want ErrMultipleTerminalOutputs", err)
		}
	})
}

// TestIntegration_ToolNode_ResultUnwrap verifies that a tool returning
// a scalar (wrapped by FunctionTool as {"result": X}) is unwrapped to X
// by ToolNode.ValidateOutput when the node's schema expects the scalar,
// driven through the scheduler.
func TestIntegration_ToolNode_ResultUnwrap(t *testing.T) {
	type Input struct {
		Name string `json:"name"`
	}
	greet, err := functiontool.New(functiontool.Config{Name: "greet"},
		func(ctx tool.Context, in Input) (string, error) {
			return "Hello " + in.Name, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New failed: %v", err)
	}
	// Output type string -> FunctionTool wraps as {"result": "Hello X"};
	// ToolNode.ValidateOutput unwraps it back to the string.
	toolNode, err := NewToolNodeTyped[Input, string](greet, defaultNodeConfig)
	if err != nil {
		t.Fatalf("NewToolNodeTyped failed: %v", err)
	}
	seed := NewFunctionNode("seed",
		func(ctx agent.Context, _ any) (Input, error) {
			return Input{Name: "world"}, nil
		}, defaultNodeConfig)

	w := mustNew(t, Chain(Start, seed, toolNode))
	got, err := runForOutput(t, w, newSeededMockCtx(t))
	if err != nil {
		t.Fatalf("workflow failed: %v", err)
	}
	if got != "Hello world" {
		t.Errorf("got %v, want %q (result key should be unwrapped)", got, "Hello world")
	}
}

// TestIntegration_ToolNode_DirectOutput verifies the default validation
// path: a tool returning a value that matches the schema directly is
// forwarded as-is, with no {"result": X} unwrap.
func TestIntegration_ToolNode_DirectOutput(t *testing.T) {
	type Input struct {
		Name string `json:"name"`
	}
	type Output struct {
		Greeting string `json:"greeting"`
	}
	greet, err := functiontool.New(functiontool.Config{Name: "greet"},
		func(ctx tool.Context, in Input) (Output, error) {
			return Output{Greeting: "Hi " + in.Name}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New failed: %v", err)
	}
	toolNode, err := NewToolNodeTyped[Input, Output](greet, defaultNodeConfig)
	if err != nil {
		t.Fatalf("NewToolNodeTyped failed: %v", err)
	}
	seed := NewFunctionNode("seed",
		func(ctx agent.Context, _ any) (Input, error) {
			return Input{Name: "Ada"}, nil
		}, defaultNodeConfig)

	w := mustNew(t, Chain(Start, seed, toolNode))
	got, err := runForOutput(t, w, newSeededMockCtx(t))
	if err != nil {
		t.Fatalf("workflow failed: %v", err)
	}
	gotMap, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("expected map[string]any, got %T", got)
	}
	if gotMap["greeting"] != "Hi Ada" {
		t.Errorf("got %v, want greeting=Hi Ada", gotMap)
	}
}

// TestIntegration_JoinNode_WithOutputSchema verifies that a JoinNode
// declaring an output schema matching the aggregated map[string]any of
// its predecessors validates successfully through the scheduler.
func TestIntegration_JoinNode_WithOutputSchema(t *testing.T) {
	a := NewFunctionNode("a",
		func(ctx agent.Context, _ any) (string, error) { return "A", nil },
		defaultNodeConfig)
	b := NewFunctionNode("b",
		func(ctx agent.Context, _ any) (string, error) { return "B", nil },
		defaultNodeConfig)

	// The aggregated input/output is map[string]any{"a": "A", "b": "B"}.
	// A permissive object schema (no required properties, string-valued)
	// matches it.
	joinSchema, err := jsonschema.For[map[string]string](nil)
	if err != nil {
		t.Fatalf("jsonschema.For failed: %v", err)
	}
	joinResolved, err := joinSchema.Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve failed: %v", err)
	}
	join := &JoinNode{BaseNode: NewBaseNodeWithSchemas("join", "", defaultNodeConfig, nil, joinResolved)}

	w := mustNew(t, []Edge{
		{From: Start, To: a},
		{From: Start, To: b},
		{From: a, To: join},
		{From: b, To: join},
	})
	got, err := runForOutput(t, w, newSeededMockCtx(t))
	if err != nil {
		t.Fatalf("workflow failed: %v", err)
	}
	gotMap, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("expected aggregated map[string]any, got %T", got)
	}
	if gotMap["a"] != "A" || gotMap["b"] != "B" {
		t.Errorf("got %v, want a=A b=B", gotMap)
	}
}

// TestIntegration_AgentNode_BoundaryValidation verifies the
// workflow-boundary layer: an AgentNode whose wrapped agent emits a
// struct output has that output validated against the AgentNode's
// declared output schema by the scheduler.
func TestIntegration_AgentNode_BoundaryValidation(t *testing.T) {
	type Output struct {
		Result string `json:"result"`
	}
	myAgent, err := agent.New(agent.Config{
		Name: "agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx.InvocationID())
				ev.Output = map[string]any{"result": "ok"}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New failed: %v", err)
	}
	node, err := NewAgentNodeTyped[any, Output](myAgent, defaultNodeConfig)
	if err != nil {
		t.Fatalf("NewAgentNodeTyped failed: %v", err)
	}

	w := mustNew(t, Chain(Start, node))
	got, err := runForOutput(t, w, newSeededMockCtx(t))
	if err != nil {
		t.Fatalf("workflow failed: %v", err)
	}
	gotMap, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("expected map[string]any, got %T", got)
	}
	if gotMap["result"] != "ok" {
		t.Errorf("got %v, want result=ok", gotMap)
	}
}
