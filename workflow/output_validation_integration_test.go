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
	"reflect"
	"strings"
	"testing"

	"google.golang.org/genai"
)

// End-to-end output-validation tests: they drive the engine via
// Workflow.Run and assert the scheduler enforces each node's
// OutputSchema across edges, joins, and nested workflows — not by
// calling ValidateOutput directly (that is unit-tested in
// base_node_test.go / tool_node_test.go).
//
// adk-go validates output per node (the schema lives on the node),
// so these mirror the scenarios in adk-python's
// tests/unittests/workflow/test_workflow_schema.py adapted to a
// per-node schema rather than a workflow-level output_schema (the
// workflow-level schema is the follow-up tracked by b/516381930).

// TestOutputValidation_ValidatedOutputPropagatesDownstream verifies a
// schema-validated producer's output is forwarded to its successor as
// that successor's input.
func TestOutputValidation_ValidatedOutputPropagatesDownstream(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	schema := resolveTestSchema[testSchemaInput](t)
	producer := newSchemaValidatedNode("producer", schema, map[string]any{"value": "hello"})

	var seen any
	consumer := inputRecorder("consumer", &seen)

	w := mustNew(t, Chain(Start, producer, consumer))
	drain(t, w.Run(mockCtx))

	got, ok := seen.(map[string]any)
	if !ok {
		t.Fatalf("consumer input = %T %#v, want map[string]any", seen, seen)
	}
	if got["value"] != "hello" {
		t.Errorf("consumer input[value] = %v, want %q", got["value"], "hello")
	}
}

// TestOutputValidation_InvalidOutputShortCircuitsChain verifies a
// producer whose output violates its schema fails the run with
// ErrNodeFailed and the downstream consumer never executes.
func TestOutputValidation_InvalidOutputShortCircuitsChain(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	schema := resolveTestSchema[testSchemaInput](t)
	producer := newSchemaValidatedNode("producer", schema, map[string]any{"value": 123})

	var seen any
	consumer := inputRecorder("consumer", &seen)

	w := mustNew(t, Chain(Start, producer, consumer))
	gotErr := drainErr(t, w.Run(mockCtx))

	if !errors.Is(gotErr, ErrNodeFailed) {
		t.Errorf("err = %v, want errors.Is(err, ErrNodeFailed)", gotErr)
	}
	if wantSubstr := `output validation failed for node "producer"`; !strings.Contains(gotErr.Error(), wantSubstr) {
		t.Errorf("error = %q, want substring %q", gotErr.Error(), wantSubstr)
	}
	if seen != nil {
		t.Errorf("consumer ran with input %#v, want it skipped after producer failure", seen)
	}
}

// TestOutputValidation_ContentJSONParsedAgainstSchema verifies the
// scheduler projects a *genai.Content carrying JSON text onto the
// node's object schema, stamping the parsed value onto Event.Output.
func TestOutputValidation_ContentJSONParsedAgainstSchema(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	schema := resolveTestSchema[testSchemaInput](t)
	content := &genai.Content{Parts: []*genai.Part{{Text: `{"value":"hi"}`}}}
	n := newSchemaValidatedNode("n", schema, content)

	w := mustNew(t, []Edge{{From: Start, To: n}})
	events := drain(t, w.Run(mockCtx))

	var out any
	for _, ev := range events {
		if ev.Output != nil {
			out = ev.Output
		}
	}
	got, ok := out.(map[string]any)
	if !ok {
		t.Fatalf("Output = %T %#v, want parsed map[string]any", out, out)
	}
	if got["value"] != "hi" {
		t.Errorf("Output[value] = %v, want %q", got["value"], "hi")
	}
}

// TestOutputValidation_ListSchemaEnforcedAtRuntime verifies array-root
// output schemas are enforced by the engine: a conforming list passes,
// a list with a non-conforming element fails with ErrNodeFailed.
func TestOutputValidation_ListSchemaEnforcedAtRuntime(t *testing.T) {
	schema := resolveTestSchema[[]testSchemaInput](t)

	t.Run("valid list passes", func(t *testing.T) {
		mockCtx := newSeededMockCtx(t)
		out := []any{
			map[string]any{"value": "a"},
			map[string]any{"value": "b"},
		}
		n := newSchemaValidatedNode("n", schema, out)
		w := mustNew(t, []Edge{{From: Start, To: n}})

		events := drain(t, w.Run(mockCtx))
		var got any
		for _, ev := range events {
			if ev.Output != nil {
				got = ev.Output
			}
		}
		if !reflect.DeepEqual(got, out) {
			t.Errorf("Output = %#v, want %#v", got, out)
		}
	})

	t.Run("non-conforming element fails", func(t *testing.T) {
		mockCtx := newSeededMockCtx(t)
		out := []any{map[string]any{"value": 123}}
		n := newSchemaValidatedNode("n", schema, out)
		w := mustNew(t, []Edge{{From: Start, To: n}})

		gotErr := drainErr(t, w.Run(mockCtx))
		if !errors.Is(gotErr, ErrNodeFailed) {
			t.Errorf("err = %v, want errors.Is(err, ErrNodeFailed)", gotErr)
		}
	})
}

// TestOutputValidation_FanOutJoinValidatesEachBranch verifies that
// when two schema-validated branches fan out and fan back into a
// JoinNode, both outputs are validated concurrently and the join
// observes the aggregated values.
func TestOutputValidation_FanOutJoinValidatesEachBranch(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	strSchema := resolveTestSchema[string](t)
	branchA := newSchemaValidatedNode("branchA", strSchema, "A-result")
	branchB := newSchemaValidatedNode("branchB", strSchema, "B-result")
	join := NewJoinNode("join")

	var seen any
	consumer := inputRecorder("consumer", &seen)

	w := mustNew(t, []Edge{
		{From: Start, To: branchA},
		{From: Start, To: branchB},
		{From: branchA, To: join},
		{From: branchB, To: join},
		{From: join, To: consumer},
	})
	drain(t, w.Run(mockCtx))

	got, ok := seen.(map[string]any)
	if !ok {
		t.Fatalf("consumer input = %T %#v, want map[string]any", seen, seen)
	}
	want := map[string]any{"branchA": "A-result", "branchB": "B-result"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("join aggregate = %#v, want %#v", got, want)
	}
}

// TestOutputValidation_NestedWorkflowTerminalSchema verifies a nested
// workflow validates its terminal node's output against that node's
// schema: a valid terminal output surfaces to the parent consumer, an
// invalid one propagates as ErrNodeFailed to the parent run.
func TestOutputValidation_NestedWorkflowTerminalSchema(t *testing.T) {
	schema := resolveTestSchema[testSchemaInput](t)

	t.Run("valid terminal surfaces to parent", func(t *testing.T) {
		mockCtx := newSeededMockCtx(t)
		innerTerminal := newSchemaValidatedNode("inner", schema, map[string]any{"value": "ok"})
		wfNode, err := NewWorkflowNode("sub", []Edge{{From: Start, To: innerTerminal}})
		if err != nil {
			t.Fatalf("NewWorkflowNode: %v", err)
		}

		var seen any
		consumer := inputRecorder("consumer", &seen)
		w := mustNew(t, Chain(Start, wfNode, consumer))
		drain(t, w.Run(mockCtx))

		got, ok := seen.(map[string]any)
		if !ok {
			t.Fatalf("consumer input = %T %#v, want map[string]any", seen, seen)
		}
		if got["value"] != "ok" {
			t.Errorf("consumer input[value] = %v, want %q", got["value"], "ok")
		}
	})

	t.Run("invalid terminal propagates error", func(t *testing.T) {
		mockCtx := newSeededMockCtx(t)
		innerTerminal := newSchemaValidatedNode("inner", schema, map[string]any{"value": 123})
		wfNode, err := NewWorkflowNode("sub", []Edge{{From: Start, To: innerTerminal}})
		if err != nil {
			t.Fatalf("NewWorkflowNode: %v", err)
		}

		var seen any
		consumer := inputRecorder("consumer", &seen)
		w := mustNew(t, Chain(Start, wfNode, consumer))
		gotErr := drainErr(t, w.Run(mockCtx))

		if !errors.Is(gotErr, ErrNodeFailed) {
			t.Errorf("err = %v, want errors.Is(err, ErrNodeFailed)", gotErr)
		}
		if seen != nil {
			t.Errorf("consumer ran with input %#v, want it skipped", seen)
		}
	})
}
