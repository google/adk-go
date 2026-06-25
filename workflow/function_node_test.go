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
	"context"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

func TestNewFunctionNodeWithSchema(t *testing.T) {
	type Input struct {
		Value string `json:"value"`
	}
	type Output struct {
		Result string `json:"result"`
	}

	tests := []struct {
		name         string
		nodeName     string
		fn           func(_ context.Context, _ agent.Context, input Input) (map[string]any, error)
		inputSchema  *jsonschema.Schema
		outputSchema *jsonschema.Schema
		input        any
		wantOutput   map[string]any
		wantErr      bool
		errSubstr    string
	}{
		{
			name:     "Success",
			nodeName: "upper",
			fn: func(_ context.Context, _ agent.Context, input Input) (map[string]any, error) {
				return map[string]any{"result": strings.ToUpper(input.Value)}, nil
			},
			inputSchema:  mustSchema[Input](t),
			outputSchema: mustSchema[Output](t),
			input:        Input{Value: "hello"},
			wantOutput:   map[string]any{"result": "HELLO"},
			wantErr:      false,
		},
		{
			name:     "NilInput",
			nodeName: "nil_test",
			fn: func(_ context.Context, _ agent.Context, input Input) (map[string]any, error) {
				if input.Value == "" {
					return map[string]any{"result": "zero"}, nil
				}
				return map[string]any{"result": "not-zero"}, nil
			},
			inputSchema:  mustSchema[Input](t),
			outputSchema: mustSchema[Output](t),
			input:        nil,
			wantOutput:   map[string]any{"result": "zero"},
			wantErr:      false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node, err := NewFunctionNodeWithSchema[Input, map[string]any](tc.nodeName, tc.fn, tc.inputSchema, tc.outputSchema, defaultNodeConfig)
			if err != nil {
				t.Fatalf("NewFunctionNodeWithSchema failed: %v", err)
			}

			mockCtx := &MockInvocationContext{sess: nil}
			exCtx := agent.NewNodeContext(t.Context(), mockCtx, nil)
			events := node.Run(t.Context(), exCtx, tc.input)

			count := 0
			for ev, err := range events {
				if err != nil {
					if !tc.wantErr {
						t.Fatalf("unexpected error: %v", err)
					}
					if tc.errSubstr != "" && !strings.Contains(err.Error(), tc.errSubstr) {
						t.Errorf("expected error containing %q, got %v", tc.errSubstr, err)
					}
					return // Expected error handled
				}
				count++
				if tc.wantErr {
					t.Fatal("expected error, got nil")
				}

				if diff := cmp.Diff(tc.wantOutput, ev.Output); diff != "" {
					t.Errorf("output mismatch (-want +got):\n%s", diff)
				}
			}

			if !tc.wantErr && count != 1 {
				t.Errorf("expected 1 event, got %d", count)
			}
		})
	}
}

// TestFunctionNode_RunDoesNotValidate verifies Run yields the raw output
// unchanged even when it violates the output schema: validation is the
// scheduler's job (ValidateOutput), not Run's.
func TestFunctionNode_RunDoesNotValidate(t *testing.T) {
	type Input struct {
		Value string `json:"value"`
	}
	type TargetOutput struct {
		Result int `json:"result"`
	}

	raw := map[string]any{"result": "not-an-int"} // violates TargetOutput
	fn := func(_ context.Context, _ agent.Context, input Input) (map[string]any, error) {
		return raw, nil
	}
	node, err := NewFunctionNodeWithSchema[Input, map[string]any](
		"test", fn, mustSchema[Input](t), mustSchema[TargetOutput](t), defaultNodeConfig)
	if err != nil {
		t.Fatalf("NewFunctionNodeWithSchema failed: %v", err)
	}

	mockCtx := &MockInvocationContext{sess: nil}
	exCtx := agent.NewNodeContext(t.Context(), mockCtx, nil)
	count := 0
	for ev, err := range node.Run(t.Context(), exCtx, Input{Value: "hello"}) {
		if err != nil {
			t.Fatalf("Run returned unexpected error: %v", err)
		}
		if diff := cmp.Diff(raw, ev.Output); diff != "" {
			t.Errorf("Run output mismatch (-want +got):\n%s", diff)
		}
		count++
	}
	if count != 1 {
		t.Fatalf("expected 1 event from Run, got %d", count)
	}
}

// TestFunctionNode_ValidateOutput covers the scheduler-side output
// validation contract: a schema-conforming value passes through, a
// schema mismatch errors, and a nil schema accepts anything.
func TestFunctionNode_ValidateOutput(t *testing.T) {
	type Input struct {
		Value string `json:"value"`
	}
	type TargetOutput struct {
		Result int `json:"result"`
	}

	fn := func(_ context.Context, _ agent.Context, input Input) (map[string]any, error) {
		return nil, nil // body unused: ValidateOutput is exercised directly
	}
	schemaNode, err := NewFunctionNodeWithSchema[Input, map[string]any](
		"test", fn, mustSchema[Input](t), mustSchema[TargetOutput](t), defaultNodeConfig)
	if err != nil {
		t.Fatalf("NewFunctionNodeWithSchema failed: %v", err)
	}
	nilSchemaNode := NewFunctionNode[Input, map[string]any]("test_nil", fn, defaultNodeConfig)

	tests := []struct {
		name    string
		node    *FunctionNode
		output  any
		want    any
		wantErr bool
	}{
		{
			name:   "direct_valid_passes_through",
			node:   schemaNode,
			output: map[string]any{"result": 1},
			want:   map[string]any{"result": 1},
		},
		{
			name:    "schema_mismatch_fails",
			node:    schemaNode,
			output:  map[string]any{"result": "not-an-int"},
			wantErr: true,
		},
		{
			name:   "nil_schema_passes_through",
			node:   nilSchemaNode,
			output: map[string]any{"anything": 1},
			want:   map[string]any{"anything": 1},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.node.ValidateOutput(tc.output)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("ValidateOutput: expected error, got nil (out=%v)", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateOutput: unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func mustSchema[T any](t *testing.T) *jsonschema.Schema {
	t.Helper()
	s, err := jsonschema.For[T](nil)
	if err != nil {
		t.Fatalf("jsonschema.For failed: %v", err)
	}
	return s
}

func TestFunctionNodeDirectEventPropagation(t *testing.T) {
	fn := func(_ context.Context, ctx agent.Context, input string) (*session.Event, error) {
		ev := session.NewEvent(ctx.InvocationID())
		ev.Output = input + " processed"
		ev.Routes = []string{"CUSTOM_ROUTE"}
		return ev, nil
	}

	node := NewFunctionNode[string, *session.Event]("event_proc", fn, defaultNodeConfig)
	mockCtx := &MockInvocationContext{sess: nil}
	exCtx := agent.NewNodeContext(t.Context(), mockCtx, nil)

	events := node.Run(t.Context(), exCtx, "hello")

	var yieldedEvents []*session.Event
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error running node: %v", err)
		}
		yieldedEvents = append(yieldedEvents, ev)
	}

	if len(yieldedEvents) != 1 {
		t.Fatalf("expected 1 event, got %d", len(yieldedEvents))
	}

	ev := yieldedEvents[0]
	if ev.Output != "hello processed" {
		t.Errorf("expected Output 'hello processed', got %v", ev.Output)
	}
	if len(ev.Routes) != 1 || ev.Routes[0] != "CUSTOM_ROUTE" {
		t.Errorf("expected Routes ['CUSTOM_ROUTE'], got %v", ev.Routes)
	}
}

func TestNewFunctionNodeFromState(t *testing.T) {
	type TwoFieldParams struct {
		Foo string `state:"foo_key"`
		Bar int
	}

	type NodeInputParams struct {
		PredecessorOutput string `state:"node_input"`
		OtherVal          int    `state:"other_val"`
	}

	type OutputStruct struct {
		Message string
		Code    int
	}

	tests := []struct {
		name                string
		setupFn             func() (*FunctionNode, error)
		stateData           map[string]any
		input               any
		wantOutput          any
		wantErr             bool
		errSubstr           string
		wantStateFieldNames []string
	}{
		{
			name: "Success",
			setupFn: func() (*FunctionNode, error) {
				return NewFunctionNodeFromState("test1", func(_ context.Context, _ agent.InvocationContext, p TwoFieldParams) (string, error) {
					return fmt.Sprintf("%s-%d", p.Foo, p.Bar), nil
				}, defaultNodeConfig)
			},
			stateData: map[string]any{
				"foo_key": "hello",
				"Bar":     42,
			},
			input:               nil,
			wantOutput:          "hello-42",
			wantStateFieldNames: []string{"foo_key", "Bar"},
		},
		{
			name: "Missing_state_key",
			setupFn: func() (*FunctionNode, error) {
				return NewFunctionNodeFromState("test2", func(_ context.Context, _ agent.InvocationContext, p TwoFieldParams) (string, error) {
					return "", nil
				}, defaultNodeConfig)
			},
			stateData: map[string]any{
				"foo_key": "hello",
				// Bar missing
			},
			wantErr:             true,
			errSubstr:           "missing state value for required field",
			wantStateFieldNames: []string{"foo_key", "Bar"},
		},
		{
			name: "Type_mismatch",
			setupFn: func() (*FunctionNode, error) {
				return NewFunctionNodeFromState("test3", func(_ context.Context, _ agent.InvocationContext, p TwoFieldParams) (string, error) {
					return "", nil
				}, defaultNodeConfig)
			},
			stateData: map[string]any{
				"foo_key": "hello",
				"Bar":     "not-an-int",
			},
			wantErr:             true,
			errSubstr:           "failed to convert state value to field",
			wantStateFieldNames: []string{"foo_key", "Bar"},
		},
		{
			name: "Node_input_success",
			setupFn: func() (*FunctionNode, error) {
				return NewFunctionNodeFromState("test4", func(_ context.Context, _ agent.InvocationContext, p NodeInputParams) (string, error) {
					return fmt.Sprintf("input:%s,other:%d", p.PredecessorOutput, p.OtherVal), nil
				}, defaultNodeConfig)
			},
			stateData: map[string]any{
				"other_val": 100,
			},
			input:               "from_pred",
			wantOutput:          "input:from_pred,other:100",
			wantStateFieldNames: []string{"other_val"},
		},
		{
			name: "Node_input_type_mismatch",
			setupFn: func() (*FunctionNode, error) {
				return NewFunctionNodeFromState("test5", func(_ context.Context, _ agent.InvocationContext, p NodeInputParams) (string, error) {
					return "", nil
				}, defaultNodeConfig)
			},
			stateData: map[string]any{
				"other_val": 100,
			},
			input:               123, // should be string
			wantErr:             true,
			errSubstr:           "invalid input type for node_input",
			wantStateFieldNames: []string{"other_val"},
		},
		{
			name: "Struct_output_success",
			setupFn: func() (*FunctionNode, error) {
				return NewFunctionNodeFromState("test6", func(_ context.Context, _ agent.InvocationContext, p TwoFieldParams) (OutputStruct, error) {
					return OutputStruct{
						Message: p.Foo,
						Code:    p.Bar,
					}, nil
				}, defaultNodeConfig)
			},
			stateData: map[string]any{
				"foo_key": "hello",
				"Bar":     42,
			},
			input:               nil,
			wantOutput:          OutputStruct{Message: "hello", Code: 42},
			wantStateFieldNames: []string{"foo_key", "Bar"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node, err := tc.setupFn()
			if err != nil {
				t.Fatalf("setupFn failed: %v", err)
			}

			if diff := cmp.Diff(tc.wantStateFieldNames, node.StateFieldNames()); diff != "" {
				t.Errorf("StateFieldNames mismatch (-want +got):\n%s", diff)
			}

			mockSess := &mockSessionForTest{
				state: &mockStateForTest{data: tc.stateData},
			}
			mockCtx := &MockInvocationContext{sess: mockSess}
			exCtx := agent.NewNodeContext(t.Context(), mockCtx, nil)

			events := node.Run(t.Context(), exCtx, tc.input)

			var lastErr error
			var output any
			for ev, err := range events {
				if err != nil {
					lastErr = err
					break
				}
				output = ev.Output
			}

			if tc.wantErr {
				if lastErr == nil {
					t.Fatalf("expected error containing %q, got nil", tc.errSubstr)
				}
				if tc.errSubstr != "" && !strings.Contains(lastErr.Error(), tc.errSubstr) {
					t.Errorf("expected error containing %q, got %v", tc.errSubstr, lastErr)
				}
			} else {
				if lastErr != nil {
					t.Fatalf("unexpected error: %v", lastErr)
				}
				if diff := cmp.Diff(tc.wantOutput, output); diff != "" {
					t.Errorf("output mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

type mockStateForTest struct {
	data map[string]any
}

func (m *mockStateForTest) Get(key string) (any, error) {
	if val, ok := m.data[key]; ok {
		return val, nil
	}
	return nil, session.ErrStateKeyNotExist
}

func (m *mockStateForTest) Set(key string, val any) error {
	m.data[key] = val
	return nil
}

func (m *mockStateForTest) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range m.data {
			if !yield(k, v) {
				return
			}
		}
	}
}

type mockSessionForTest struct {
	session.Session
	state session.State
}

func (m *mockSessionForTest) State() session.State {
	return m.state
}

// A body that emits a RequestInput and returns ErrNodeInterrupted
// pauses the workflow: the prompt is forwarded and the successor does
// not run.
func TestEmittingFunctionNode_HitlPausesAndForwardsRequest(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	var downstreamRan atomic.Bool
	asker := NewEmittingFunctionNode("asker", func(_ context.Context, ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		if err := emit(NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID: "human_review",
			Message:     "Please approve",
			Payload:     "the draft",
		})); err != nil {
			return nil, err
		}
		return nil, ErrNodeInterrupted
	}, defaultNodeConfig)
	downstream := newHitlNode("downstream", func(_ agent.Context, _ any, _ func(*session.Event, error) bool) {
		downstreamRan.Store(true)
	})

	w := mustNew(t, []Edge{
		{From: Start, To: asker},
		{From: asker, To: downstream},
	})

	events := drain(t, w.Run(t.Context(), mockCtx))

	if downstreamRan.Load() {
		t.Error("downstream node ran; HITL pause must suppress successor scheduling")
	}
	req, count := findRequestedInputEvent(events)
	if count != 1 {
		t.Fatalf("expected exactly 1 RequestedInput event, got %d", count)
	}
	if got, want := req.RequestedInput.InterruptID, "human_review"; got != want {
		t.Errorf("InterruptID = %q, want %q", got, want)
	}
	if got, want := req.RequestedInput.Message, "Please approve"; got != want {
		t.Errorf("Message = %q, want %q", got, want)
	}
	if got, want := req.RequestedInput.Payload, "the draft"; got != want {
		t.Errorf("Payload = %v, want %q", got, want)
	}
}

// When the body emits a RequestInput without an InterruptID, the
// engine assigns one so the resume can still be correlated.
func TestEmittingFunctionNode_AutoGeneratesInterruptID(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	asker := NewEmittingFunctionNode("asker", func(_ context.Context, ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		if err := emit(NewRequestInputEvent(ctx, session.RequestInput{Message: "approve?"})); err != nil {
			return nil, err
		}
		return nil, ErrNodeInterrupted
	}, defaultNodeConfig)
	w := mustNew(t, []Edge{{From: Start, To: asker}})

	events := drain(t, w.Run(t.Context(), mockCtx))
	req, count := findRequestedInputEvent(events)
	if count != 1 {
		t.Fatalf("expected 1 RequestedInput event, got %d", count)
	}
	if req.RequestedInput.InterruptID == "" {
		t.Error("InterruptID is empty; engine must auto-generate when caller omits it")
	}
}

// Two RequestInputs emitted in a single activation both register on
// the parked node, so either response can resume it.
func TestEmittingFunctionNode_MultipleRequestsPark(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	asker := NewEmittingFunctionNode("asker", func(_ context.Context, ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		if err := emit(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "first"})); err != nil {
			return nil, err
		}
		if err := emit(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "second"})); err != nil {
			return nil, err
		}
		return nil, ErrNodeInterrupted
	}, defaultNodeConfig)

	w := mustNew(t, []Edge{{From: Start, To: asker}})
	events := drain(t, w.Run(t.Context(), mockCtx))

	got := map[string]bool{}
	for _, ev := range events {
		for _, id := range ev.LongRunningToolIDs {
			got[id] = true
		}
	}
	if !got["first"] || !got["second"] {
		t.Errorf("long-running interrupts = %v, want both first and second", got)
	}
}

// A non-sentinel error returned after emitting a RequestInput fails
// the node; the recorded request must not turn the failure into a
// clean pause.
func TestEmittingFunctionNode_ErrorAfterRequestFails(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	wantErr := errors.New("downstream of request")
	asker := NewEmittingFunctionNode("asker", func(_ context.Context, ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		if err := emit(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "ignored"})); err != nil {
			return nil, err
		}
		return nil, wantErr
	}, defaultNodeConfig)
	w := mustNew(t, []Edge{{From: Start, To: asker}})

	gotErr := drainErr(t, w.Run(t.Context(), mockCtx))
	if !errors.Is(gotErr, wantErr) {
		t.Errorf("Run error = %v, want %v (failure must take precedence over the recorded request)", gotErr, wantErr)
	}
}

// A body can emit intermediate content-only events and still return a
// terminal output: both events surface and the successor runs (no
// pause). Intermediate events leave Output unset to respect the
// scheduler's single-output-per-node invariant.
func TestEmittingFunctionNode_EmitProgressBeforeOutput(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	var downstreamRan atomic.Bool
	worker := NewEmittingFunctionNode("worker", func(_ context.Context, ctx agent.Context, _ any, emit func(*session.Event) error) (string, error) {
		progress := session.NewEvent(ctx.InvocationID())
		progress.Content = &genai.Content{Parts: []*genai.Part{{Text: "tick"}}}
		if err := emit(progress); err != nil {
			return "", err
		}
		return "done", nil
	}, defaultNodeConfig)
	downstream := newHitlNode("downstream", func(_ agent.Context, _ any, _ func(*session.Event, error) bool) {
		downstreamRan.Store(true)
	})

	w := mustNew(t, []Edge{
		{From: Start, To: worker},
		{From: worker, To: downstream},
	})
	events := drain(t, w.Run(t.Context(), mockCtx))

	if !downstreamRan.Load() {
		t.Error("downstream did not run; emit without ErrNodeInterrupted must not pause")
	}
	var sawProgress, sawDone bool
	for _, ev := range events {
		if ev == nil {
			continue
		}
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p != nil && p.Text == "tick" {
					sawProgress = true
				}
			}
		}
		if s, ok := ev.Output.(string); ok && s == "done" {
			sawDone = true
		}
	}
	if !sawProgress {
		t.Error(`did not observe progress event with Content text "tick"`)
	}
	if !sawDone {
		t.Error(`did not observe terminal event with Output="done"`)
	}
}

// Input schema constraints (e.g. maxLength) must be enforced even when
// the input already arrives as type IN, i.e. on the type-assertion-hit
// path that skips ConvertToWithJSONSchema.
func TestFunctionNode_ValidatesInputOnAssertionHitPath(t *testing.T) {
	type Input struct {
		Name string `json:"name"`
	}

	maxLen := 3
	inputSchema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"name": {Type: "string", MaxLength: &maxLen},
		},
	}

	fn := func(_ context.Context, _ agent.Context, in Input) (string, error) { return in.Name, nil }
	emittingFn := func(_ context.Context, _ agent.Context, in Input, _ func(*session.Event) error) (any, error) {
		return in.Name, nil
	}

	runOnce := func(t *testing.T, node *FunctionNode, input Input) error {
		t.Helper()
		exCtx := agent.NewNodeContext(t.Context(), newMockCtx(t), nil)
		for _, err := range node.Run(t.Context(), exCtx, input) {
			if err != nil {
				return err
			}
		}
		return nil
	}

	t.Run("plain", func(t *testing.T) {
		node, err := NewFunctionNodeWithSchema[Input, string]("greet", fn, inputSchema, nil, defaultNodeConfig)
		if err != nil {
			t.Fatalf("NewFunctionNodeWithSchema: %v", err)
		}
		if err := runOnce(t, node, Input{Name: "ok"}); err != nil {
			t.Errorf("valid input rejected: %v", err)
		}
		err = runOnce(t, node, Input{Name: "too-long"})
		if err == nil || !strings.Contains(err.Error(), "validation failed for input") {
			t.Errorf("over-length input: err = %v, want validation failure", err)
		}
	})

	t.Run("emitting", func(t *testing.T) {
		node, err := NewEmittingFunctionNodeWithSchema[Input, any]("greet", emittingFn, inputSchema, nil, defaultNodeConfig)
		if err != nil {
			t.Fatalf("NewEmittingFunctionNodeWithSchema: %v", err)
		}
		if err := runOnce(t, node, Input{Name: "ok"}); err != nil {
			t.Errorf("valid input rejected: %v", err)
		}
		err = runOnce(t, node, Input{Name: "too-long"})
		if err == nil || !strings.Contains(err.Error(), "validation failed for input") {
			t.Errorf("over-length input: err = %v, want validation failure", err)
		}
	})
}

// runFunctionNodeOnce drives a FunctionNode for a single input and
// returns the sole emitted event.
func runFunctionNodeOnce[OUT any](t *testing.T, fn func(ctx context.Context, invCleanCtx agent.Context, input any) (OUT, error), input any) *session.Event {
	t.Helper()
	node := NewFunctionNode[any, OUT]("n", fn, defaultNodeConfig)
	exCtx := agent.NewNodeContext(t.Context(), &MockInvocationContext{sess: nil}, nil)

	var got *session.Event
	count := 0
	for ev, err := range node.Run(t.Context(), exCtx, input) {
		if err != nil {
			t.Fatalf("FunctionNode.Run: %v", err)
		}
		got = ev
		count++
	}
	if count != 1 {
		t.Fatalf("FunctionNode.Run emitted %d events, want 1", count)
	}
	return got
}

// TestFunctionNode_ContentOutputGoesToContent asserts that a
// *genai.Content (or genai.Content) return populates event.Content, not
// event.Output. Mirrors adk-python _function_node.py, which maps a
// Content return to Event(content=...).
func TestFunctionNode_ContentOutputGoesToContent(t *testing.T) {
	t.Parallel()

	want := &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{genai.NewPartFromText("hello from content")},
	}

	t.Run("pointer *genai.Content", func(t *testing.T) {
		t.Parallel()
		ev := runFunctionNodeOnce(t, func(_ context.Context, _ agent.Context, _ any) (*genai.Content, error) {
			return want, nil
		}, nil)

		if ev.Output != nil {
			t.Errorf("event.Output = %v, want nil; Content must not go to Output", ev.Output)
		}
		if diff := cmp.Diff(want, ev.Content); diff != "" {
			t.Errorf("event.Content mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("value genai.Content", func(t *testing.T) {
		t.Parallel()
		ev := runFunctionNodeOnce(t, func(_ context.Context, _ agent.Context, _ any) (genai.Content, error) {
			return *want, nil
		}, nil)

		if ev.Output != nil {
			t.Errorf("event.Output = %v, want nil; Content must not go to Output", ev.Output)
		}
		if diff := cmp.Diff(want, ev.Content); diff != "" {
			t.Errorf("event.Content mismatch (-want +got):\n%s", diff)
		}
	})
}

// TestFunctionNode_NonContentOutputGoesToOutput pins the complementary
// case: a non-Content return value populates event.Output and leaves
// event.Content nil.
func TestFunctionNode_NonContentOutputGoesToOutput(t *testing.T) {
	t.Parallel()

	want := map[string]any{"result": "ok"}
	ev := runFunctionNodeOnce(t, func(_ context.Context, _ agent.Context, _ any) (map[string]any, error) {
		return want, nil
	}, nil)

	if ev.Content != nil {
		t.Errorf("event.Content = %v, want nil; non-Content output must not set Content", ev.Content)
	}
	if diff := cmp.Diff(want, ev.Output); diff != "" {
		t.Errorf("event.Output mismatch (-want +got):\n%s", diff)
	}
}
