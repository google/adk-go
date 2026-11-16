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

package workflowagent

import (
	"encoding/json"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/workflow"
)

// recordingNode returns a node that appends each input it receives to
// *inputs and outputs output, or its input when output is nil.
func recordingNode(mu *sync.Mutex, inputs *[]any, output any) workflow.Node {
	return workflow.NewFunctionNode("node", func(_ agent.Context, in any) (any, error) {
		mu.Lock()
		*inputs = append(*inputs, in)
		mu.Unlock()
		if output == nil {
			return in, nil
		}
		return output, nil
	}, workflow.NodeConfig{})
}

func resolveJSONSchema(t *testing.T, src string) *jsonschema.Resolved {
	t.Helper()
	var s jsonschema.Schema
	if err := json.Unmarshal([]byte(src), &s); err != nil {
		t.Fatalf("unmarshal schema: %v", err)
	}
	r, err := s.Resolve(nil)
	if err != nil {
		t.Fatalf("resolve schema: %v", err)
	}
	return r
}

func TestWorkflowAgent_InputAndOutputSchemas(t *testing.T) {
	in := resolveJSONSchema(t, `{
		"type": "object",
		"required": ["x"],
		"properties": {
			"x": {"type": "string"},
			"in_def": {"type": "string", "default": "d1"}
		}
	}`)
	out := resolveJSONSchema(t, `{
		"type": "object",
		"required": ["x", "in_def"],
		"properties": {
			"x": {"type": "string"},
			"in_def": {"type": "string"},
			"out_def": {"type": "string", "default": "d2"}
		}
	}`)

	tests := []struct {
		name       string
		message    string
		output     any // nil: the node outputs its input
		wantInputs []any
		wantErr    error
	}{
		{
			name:       "valid_input_reaches_node_with_defaults",
			message:    `{"x": "hi"}`,
			wantInputs: []any{map[string]any{"x": "hi", "in_def": "d1"}},
		},
		{
			name:    "invalid_input_fails_before_any_node",
			message: `{"x": 123}`,
			wantErr: workflow.ErrInputValidation,
		},
		{
			name:       "invalid_terminal_output_fails_run",
			message:    `{"x": "hi"}`,
			output:     map[string]any{"x": 123},
			wantInputs: []any{map[string]any{"x": "hi", "in_def": "d1"}},
			wantErr:    workflow.ErrNodeFailed,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var inputs []any
			a, err := New(Config{
				Name:         "test_workflow",
				Edges:        workflow.Chain(workflow.Start, recordingNode(&mu, &inputs, tc.output)),
				InputSchema:  in,
				OutputSchema: out,
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			sess := newFakeSession()
			msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: tc.message}}}

			drainAgent(t, sess, a.Run(newMockCtx(sess, a, msg)), tc.wantErr)

			mu.Lock()
			defer mu.Unlock()
			if diff := cmp.Diff(tc.wantInputs, inputs); diff != "" {
				t.Errorf("node inputs (-want +got):\n%s", diff)
			}
		})
	}
}
