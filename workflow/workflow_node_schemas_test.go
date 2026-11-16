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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
)

func TestNewWorkflowNodeWithSchemas_OutputSchema(t *testing.T) {
	in := resolveSchemaJSON(t, rootInputSchema)
	out := resolveSchemaJSON(t, rootOutputSchema)

	// newParent returns a workflow Start -> inner -> after, where inner
	// is a nested workflow whose only node produce outputs innerOutput.
	newParent := func(t *testing.T, innerOutput any, inSchema, outSchema *jsonschema.Resolved, log *inputLog) *Workflow {
		t.Helper()
		inner, err := NewWorkflowNodeWithSchemas("inner", []Edge{{From: Start, To: fixedNode("produce", innerOutput, log)}}, inSchema, outSchema)
		if err != nil {
			t.Fatalf("NewWorkflowNodeWithSchemas: %v", err)
		}
		w, err := New("outer", []Edge{{From: Start, To: inner}, {From: inner, To: echoNode("after", log)}})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		return w
	}

	t.Run("accessors", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			in, out *jsonschema.Resolved
		}{
			{name: "output_only", out: out},
			{name: "input_only", in: in},
		} {
			t.Run(tc.name, func(t *testing.T) {
				n, err := NewWorkflowNodeWithSchemas("inner", []Edge{{From: Start, To: echoNode("echo", &inputLog{})}}, tc.in, tc.out)
				if err != nil {
					t.Fatalf("NewWorkflowNodeWithSchemas: %v", err)
				}
				if n.InputSchema() != tc.in || n.OutputSchema() != tc.out {
					t.Error("InputSchema and OutputSchema do not return the constructor's arguments")
				}
			})
		}
	})

	t.Run("invalid_output_stops_parent", func(t *testing.T) {
		log := &inputLog{}
		w := newParent(t, map[string]any{"age": 10}, nil, out, log)

		_, err := collect(w.Run(newMockCtx(t)))

		if !errors.Is(err, ErrNodeFailed) {
			t.Fatalf("Run error = %v, want one wrapping ErrNodeFailed", err)
		}
		if got := log.recorded()["after"]; len(got) != 0 {
			t.Errorf("after ran with %v, want no run", got)
		}
	})

	t.Run("valid_output_fills_default", func(t *testing.T) {
		log := &inputLog{}
		w := newParent(t, map[string]any{"name": "done"}, nil, out, log)

		if _, err := collect(w.Run(newMockCtx(t))); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if diff := cmp.Diff([]any{map[string]any{"name": "done", "tag": "ok"}}, log.recorded()["after"]); diff != "" {
			t.Errorf("after inputs (-want +got):\n%s", diff)
		}
	})

	t.Run("invalid_input_skips_inner_nodes", func(t *testing.T) {
		log := &inputLog{}
		w := newParent(t, map[string]any{"name": "done"}, in, nil, log)

		_, err := collect(w.Run(userText(newMockCtx(t), `{"wrong": 1}`)))

		if !errors.Is(err, ErrInputValidation) {
			t.Fatalf("Run error = %v, want one wrapping ErrInputValidation", err)
		}
		if got := log.recorded(); len(got) != 0 {
			t.Errorf("nodes ran with inputs %v, want none", got)
		}
	})
}
