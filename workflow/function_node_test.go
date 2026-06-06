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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/agent"
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
		fn           func(ctx agent.InvocationContext, input Input) (map[string]any, error)
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
			fn: func(ctx agent.InvocationContext, input Input) (map[string]any, error) {
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
			fn: func(ctx agent.InvocationContext, input Input) (map[string]any, error) {
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
			events := node.Run(nodeCtx(mockCtx), tc.input)

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

// TestFunctionNode_ValidateOutput verifies that output schema validation
// is enforced through the node-level ValidateOutput contract (invoked
// scheduler-side), not inline inside Run. Run itself passes the output
// through unchanged; ValidateOutput surfaces the schema mismatch.
func TestFunctionNode_ValidateOutput(t *testing.T) {
	type Input struct {
		Value string `json:"value"`
	}
	type TargetOutput struct {
		Result int `json:"result"`
	}

	fn := func(ctx agent.InvocationContext, input Input) (map[string]any, error) {
		return map[string]any{"result": "not-an-int"}, nil
	}
	node, err := NewFunctionNodeWithSchema[Input, map[string]any](
		"test", fn, mustSchema[Input](t), mustSchema[TargetOutput](t), defaultNodeConfig)
	if err != nil {
		t.Fatalf("NewFunctionNodeWithSchema failed: %v", err)
	}

	// Run no longer validates: it yields the raw output without error.
	mockCtx := &MockInvocationContext{sess: nil}
	var got any
	count := 0
	for ev, err := range node.Run(nodeCtx(mockCtx), Input{Value: "hello"}) {
		if err != nil {
			t.Fatalf("Run returned unexpected error: %v", err)
		}
		got = ev.Output
		count++
	}
	if count != 1 {
		t.Fatalf("expected 1 event from Run, got %d", count)
	}

	// ValidateOutput (the scheduler-side contract) rejects the mismatch.
	if _, err := node.ValidateOutput(got); err == nil {
		t.Fatalf("ValidateOutput: expected validation error, got nil")
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
