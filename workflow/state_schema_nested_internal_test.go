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
	"maps"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// nestedNode returns a node that runs inner as a nested workflow whose own
// state schema declares only inner_key, an integer. No exported constructor
// gives a nested workflow a state schema of its own, so the node is built the
// way NewWorkflowNode builds it.
func nestedNode(t *testing.T, name string, inner Node) *WorkflowNode {
	t.Helper()
	innerSchema, err := (&jsonschema.Schema{
		Title:      "InnerSchema",
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"inner_key": {Type: "integer"}},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve(inner schema) error = %v", err)
	}
	wf, err := New(name, []Edge{{From: Start, To: inner}}, WithStateSchema(innerSchema))
	if err != nil {
		t.Fatalf("New(%s) error = %v", name, err)
	}
	return &WorkflowNode{BaseNode: NewBaseNode(name, "", NodeConfig{}), subWorkflow: wf}
}

// runWithStore runs wf on a mock context, appends every non-partial event to
// the session as the runner does, and returns the stored state and the first
// run error.
func runWithStore(t *testing.T, wf *Workflow) (map[string]any, error) {
	t.Helper()
	mock, svc := newSessionMockCtx(t)
	var runErr error
	for ev, err := range wf.Run(mock) {
		if err != nil {
			if runErr == nil {
				runErr = err
			}
			continue
		}
		if ev.Partial {
			continue
		}
		if err := svc.AppendEvent(t.Context(), mock.sess, ev); err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}
	}
	got, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: mock.sess.ID()})
	if err != nil {
		t.Fatalf("session Get() error = %v", err)
	}
	return maps.Collect(got.Session.State().All()), runErr
}

func TestStateSchema_NestedWorkflowOwnSchema(t *testing.T) {
	writer := NewFunctionNode("writer", func(ctx agent.Context, _ any) (string, error) {
		return "written", ctx.State().Set("inner_key", 1)
	}, NodeConfig{})
	// The outer schema declares counter and not inner_key.
	wf, err := New("wf", []Edge{{From: Start, To: nestedNode(t, "inner", writer)}}, WithStateSchema(counterSchema(t)))
	if err != nil {
		t.Fatalf("New(outer) error = %v", err)
	}

	stored, runErr := runWithStore(t, wf)
	if runErr != nil {
		t.Fatalf("Run error = %v, want nil", runErr)
	}

	if stored["inner_key"] != 1 {
		t.Errorf("stored inner_key = %v, want 1", stored["inner_key"])
	}
}

// An inner node's write that no event carried (it wrote after its last yield)
// goes out on the inner runner's trailing event. The enclosing runner does not
// check that event against its own schema.
func TestStateSchema_NestedWorkflowTrailingWriteNotRechecked(t *testing.T) {
	writer := newFaninNode("writer", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		if err := ctx.State().Set("inner_key", 1); err != nil {
			yield(nil, err)
		}
	})
	wf, err := New("wf", []Edge{{From: Start, To: nestedNode(t, "inner", writer)}}, WithStateSchema(counterSchema(t)))
	if err != nil {
		t.Fatalf("New(outer) error = %v", err)
	}

	stored, runErr := runWithStore(t, wf)

	if runErr != nil {
		t.Errorf("run error = %v, want nil", runErr)
	}
	if stored["inner_key"] != 1 {
		t.Errorf("stored inner_key = %v, want 1", stored["inner_key"])
	}
}

// A parallel worker over a nested workflow with its own schema: when an item
// fails, the earlier item's write, which the inner schema accepts, is stored
// and the run reports the item's error. The enclosing runner does not check
// the worker's failure-path state event against its own schema.
func TestParallelWorker_NestedSchemaFailureEventNotRechecked(t *testing.T) {
	errBoom := errors.New("boom")
	item := NewFunctionNode("item", func(ctx agent.Context, in int) (int, error) {
		if in == 1 {
			return 0, errBoom
		}
		return in, ctx.State().Set("inner_key", 1)
	}, NodeConfig{})
	// One item at a time, so item 0 has finished before item 1 fails.
	pw, err := NewParallelWorker("pw", nestedNode(t, "inner", item), 1, NodeConfig{})
	if err != nil {
		t.Fatalf("NewParallelWorker() error = %v", err)
	}
	seed := NewFunctionNode("seed", func(agent.Context, any) ([]any, error) { return []any{0, 1}, nil }, NodeConfig{})
	wf, err := New("wf", Chain(Start, seed, pw), WithStateSchema(counterSchema(t)))
	if err != nil {
		t.Fatalf("New(outer) error = %v", err)
	}

	stored, runErr := runWithStore(t, wf)

	if !errors.Is(runErr, errBoom) {
		t.Errorf("run error = %v, want errBoom", runErr)
	}
	if se, ok := errors.AsType[*StateSchemaError](runErr); ok {
		t.Errorf("run error is a StateSchemaError for key %q, want the item's error only", se.Key)
	}
	if stored["inner_key"] != 1 {
		t.Errorf("stored inner_key = %v, want 1", stored["inner_key"])
	}
}
