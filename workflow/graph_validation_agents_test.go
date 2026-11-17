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

package workflow_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/workflow"
)

// gcvAgentNode wraps a new LlmAgent with the given mode in a graph node.
func gcvAgentNode(t *testing.T, name string, mode llmagent.Mode) workflow.Node {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: name, Mode: mode})
	if err != nil {
		t.Fatalf("llmagent.New(%q, %q): %v", name, mode, err)
	}
	n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(%q): %v", name, err)
	}
	return n
}

func gcvPassThrough(_ agent.Context, in any) (any, error) { return in, nil }

// gcvFunctionNode returns a node with no schemas.
func gcvFunctionNode(name string) workflow.Node {
	return workflow.NewFunctionNode(name, gcvPassThrough, workflow.NodeConfig{})
}

// gcvTitledFunctionNode returns a node whose input or output schema is an
// object schema with the given title; an empty title means no schema.
func gcvTitledFunctionNode(t *testing.T, name, inTitle, outTitle string) workflow.Node {
	t.Helper()
	titled := func(title string) *jsonschema.Schema {
		if title == "" {
			return nil
		}
		return &jsonschema.Schema{Type: "object", Title: title, Properties: map[string]*jsonschema.Schema{"v": {Type: "string"}}}
	}
	n, err := workflow.NewFunctionNodeWithSchema(name, gcvPassThrough, titled(inTitle), titled(outTitle), workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewFunctionNodeWithSchema(%q): %v", name, err)
	}
	return n
}

type gcvUndeclaredParams struct {
	Bar string `state:"bar"`
}

func gcvFnUndeclared(agent.InvocationContext, gcvUndeclaredParams) (string, error) {
	return "ok", nil
}

// gcvStateMismatch returns a node that reads the state field "bar" and a
// state schema that declares only "Foo".
func gcvStateMismatch(t *testing.T) (workflow.Node, *jsonschema.Resolved) {
	t.Helper()
	n, err := workflow.NewFunctionNodeFromState("s", gcvFnUndeclared, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewFunctionNodeFromState: %v", err)
	}
	schema, err := (&jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"Foo": {Type: "string"}},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	return n, schema
}

// Of two task-mode agent nodes, the first mentioned is reported, on every
// call.
func TestNew_TaskModeReportDeterministic(t *testing.T) {
	n2 := gcvAgentNode(t, "t2", llmagent.ModeTask)
	n1 := gcvAgentNode(t, "t1", llmagent.ModeTask)
	edges := []workflow.Edge{{From: workflow.Start, To: n2}, {From: n2, To: n1}}

	for i := range 50 {
		_, err := workflow.New("w", edges)

		if err == nil || !strings.Contains(err.Error(), `Agent "t2" has mode='task'`) {
			t.Fatalf("call %d: New() error = %v, want it to contain %q", i, err, `Agent "t2" has mode='task'`)
		}
	}
}

// The chat-wiring rule (P10) comes after the schema rule (P9) and before the
// state-schema rule (P11); the task-mode rule (G1) comes after both P6 and
// P11.
func TestNew_RuleOrderWithAgents(t *testing.T) {
	t.Run("R8 schema mismatch before chat wiring", func(t *testing.T) {
		a := gcvTitledFunctionNode(t, "A", "", "X")
		b := gcvTitledFunctionNode(t, "B", "Y", "")
		chat := gcvAgentNode(t, "chatter", llmagent.ModeChat)

		_, err := workflow.New("w", workflow.Chain(workflow.Start, a, b, chat))

		gcvWantFirstDefect(t, err, "schema mismatch on edge A -> B", "mode='chat'")
	})

	t.Run("R9 chat wiring before the state schema", func(t *testing.T) {
		s, schema := gcvStateMismatch(t)
		chat := gcvAgentNode(t, "chatter", llmagent.ModeChat)

		// p declares no schemas, so no edge here is compared by the schema
		// rule.
		_, err := workflow.New("w", workflow.Chain(workflow.Start, s, gcvFunctionNode("p"), chat), workflow.WithStateSchema(schema))

		gcvWantFirstDefect(t, err, "has mode='chat'", "references state field")
	})

	t.Run("R10 state schema before task mode", func(t *testing.T) {
		s, schema := gcvStateMismatch(t)
		task := gcvAgentNode(t, "doer", llmagent.ModeTask)
		edges := []workflow.Edge{{From: workflow.Start, To: task}, {From: workflow.Start, To: s}}

		_, err := workflow.New("w", edges, workflow.WithStateSchema(schema))

		gcvWantFirstDefect(t, err, "references state field", "mode='task'")
	})

	t.Run("R11 duplicate edge before task mode", func(t *testing.T) {
		a, b := gcvFunctionNode("A"), gcvFunctionNode("B")
		task := gcvAgentNode(t, "doer", llmagent.ModeTask)
		edges := []workflow.Edge{
			{From: workflow.Start, To: a},
			{From: a, To: b},
			{From: a, To: b, Route: workflow.StringRoute("x")},
			{From: b, To: task},
		}

		_, err := workflow.New("w", edges)

		gcvWantFirstDefect(t, err, `duplicate edge: from "A" to "B"`, "mode='task'")
	})
}

// gcvWantFirstDefect fails the test unless err reports the first defect and
// not the second.
func gcvWantFirstDefect(t *testing.T, err error, first, second string) {
	t.Helper()
	if err == nil {
		t.Fatalf("New() error = nil, want an error containing %q", first)
	}
	if !strings.Contains(err.Error(), first) {
		t.Errorf("New() error = %q, want it to contain %q", err, first)
	}
	if strings.Contains(err.Error(), second) {
		t.Errorf("New() error = %q, want no %q (that rule comes later)", err, second)
	}
}

func TestNew_AgentModeErrorsMatchUmbrella(t *testing.T) {
	t.Run("chat agent after a node other than Start", func(t *testing.T) {
		pred := gcvFunctionNode("pred")
		chat := gcvAgentNode(t, "chatter", llmagent.ModeChat)

		_, err := workflow.New("w", workflow.Chain(workflow.Start, pred, chat))

		if !errors.Is(err, workflow.ErrGraphValidation) {
			t.Fatalf("New() error = %v, want an error matching ErrGraphValidation", err)
		}
		if !strings.Contains(err.Error(), "mode='chat'") {
			t.Errorf("New() error = %q, want it to contain %q", err, "mode='chat'")
		}
	})

	t.Run("task agent as a graph node", func(t *testing.T) {
		task := gcvAgentNode(t, "doer", llmagent.ModeTask)

		_, err := workflow.New("w", []workflow.Edge{{From: workflow.Start, To: task}})

		if !errors.Is(err, workflow.ErrGraphValidation) {
			t.Fatalf("New() error = %v, want an error matching ErrGraphValidation", err)
		}
		if !strings.Contains(err.Error(), "mode='task'") {
			t.Errorf("New() error = %q, want it to contain %q", err, "mode='task'")
		}
	})
}
