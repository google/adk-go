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

package workflowinternal_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/internal/workflowinternal"
	"google.golang.org/adk/model"
)

var sampleOutputSchema = &genai.Schema{
	Type: genai.TypeObject,
	Properties: map[string]*genai.Schema{
		"result": {Type: genai.TypeString},
		"count":  {Type: genai.TypeInteger},
	},
	Required: []string{"result", "count"},
}

func makeTaskAgent(t *testing.T, outputSchema *genai.Schema) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{
		Name:         "test_agent",
		OutputSchema: outputSchema,
	})
	if err != nil {
		t.Fatalf("llmagent.New failed: %v", err)
	}
	return a
}

func newFinishTaskTool(t *testing.T, outputSchema *genai.Schema) *workflowinternal.FinishTaskTool {
	t.Helper()
	tt, err := workflowinternal.NewFinishTaskTool(makeTaskAgent(t, outputSchema))
	if err != nil {
		t.Fatalf("NewFinishTaskTool failed: %v", err)
	}
	ft, ok := tt.(*workflowinternal.FinishTaskTool)
	if !ok {
		t.Fatalf("NewFinishTaskTool returned %T, want *finishTaskTool", tt)
	}
	return ft
}

func TestNewFinishTaskTool(t *testing.T) {
	tests := []struct {
		name             string
		outputSchema     *genai.Schema
		wantHasOutputDoc bool
	}{
		{
			name:             "without output schema",
			outputSchema:     nil,
			wantHasOutputDoc: false,
		},
		{
			name:             "with output schema",
			outputSchema:     sampleOutputSchema,
			wantHasOutputDoc: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ft := newFinishTaskTool(t, tc.outputSchema)
			if got, want := ft.Name(), workflowinternal.FinishTaskToolName; got != want {
				t.Errorf("Name() = %q, want %q", got, want)
			}
			if !strings.Contains(ft.Description(), "Signal that this agent has completed") {
				t.Errorf("Description() = %q, want substring %q",
					ft.Description(), "Signal that this agent has completed")
			}
			if got, want := strings.Contains(ft.Description(), "output data"), tc.wantHasOutputDoc; got != want {
				t.Errorf("Description() contains %q = %v, want %v",
					"output data", got, want)
			}
		})
	}
}

func TestFinishTaskTool_Declaration(t *testing.T) {
	tests := []struct {
		name           string
		outputSchema   *genai.Schema
		wantProperties []string
	}{
		{
			name:           "without output schema",
			outputSchema:   nil,
			wantProperties: []string{"result"},
		},
		{
			name:           "with object output schema",
			outputSchema:   sampleOutputSchema,
			wantProperties: []string{"result", "count"},
		},
		{
			name:           "wrapped string output schema",
			outputSchema:   &genai.Schema{Type: genai.TypeString},
			wantProperties: []string{"result"},
		},
		{
			name:           "wrapped integer output schema",
			outputSchema:   &genai.Schema{Type: genai.TypeInteger},
			wantProperties: []string{"result"},
		},
		{
			name:           "wrapped boolean output schema",
			outputSchema:   &genai.Schema{Type: genai.TypeBoolean},
			wantProperties: []string{"result"},
		},
		{
			name:           "wrapped number output schema",
			outputSchema:   &genai.Schema{Type: genai.TypeNumber},
			wantProperties: []string{"result"},
		},
		{
			name: "wrapped array of strings output schema",
			outputSchema: &genai.Schema{
				Type:  genai.TypeArray,
				Items: &genai.Schema{Type: genai.TypeString},
			},
			wantProperties: []string{"result"},
		},
		{
			name: "wrapped array of integers output schema",
			outputSchema: &genai.Schema{
				Type:  genai.TypeArray,
				Items: &genai.Schema{Type: genai.TypeInteger},
			},
			wantProperties: []string{"result"},
		},
		{
			name: "wrapped array of objects output schema",
			outputSchema: &genai.Schema{
				Type:  genai.TypeArray,
				Items: sampleOutputSchema,
			},
			wantProperties: []string{"result"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ft := newFinishTaskTool(t, tc.outputSchema)
			decl := ft.Declaration()
			if decl == nil {
				t.Fatal("Declaration() = nil, want non-nil")
			}
			if got, want := decl.Name, workflowinternal.FinishTaskToolName; got != want {
				t.Errorf("Declaration().Name = %q, want %q", got, want)
			}
			if decl.Parameters == nil {
				t.Fatal("Declaration().Parameters = nil, want non-nil")
			}
			if got, want := decl.Parameters.Type, genai.TypeObject; got != want {
				t.Errorf("Declaration().Parameters.Type = %q, want %q", got, want)
			}
			for _, p := range tc.wantProperties {
				if _, ok := decl.Parameters.Properties[p]; !ok {
					t.Errorf("Declaration().Parameters.Properties is missing %q, got keys %v",
						p, propertyKeys(decl.Parameters.Properties))
				}
			}
		})
	}
}

func TestFinishTaskTool_WrapperKey(t *testing.T) {
	tests := []struct {
		name           string
		outputSchema   *genai.Schema
		wantWrapperKey string
	}{
		{
			name:           "default (no user schema)",
			outputSchema:   nil,
			wantWrapperKey: "",
		},
		{
			name:           "object",
			outputSchema:   sampleOutputSchema,
			wantWrapperKey: "",
		},
		{
			name:           "string",
			outputSchema:   &genai.Schema{Type: genai.TypeString},
			wantWrapperKey: "result",
		},
		{
			name:           "integer",
			outputSchema:   &genai.Schema{Type: genai.TypeInteger},
			wantWrapperKey: "result",
		},
		{
			name:           "boolean",
			outputSchema:   &genai.Schema{Type: genai.TypeBoolean},
			wantWrapperKey: "result",
		},
		{
			name:           "number",
			outputSchema:   &genai.Schema{Type: genai.TypeNumber},
			wantWrapperKey: "result",
		},
		{
			name: "array of strings",
			outputSchema: &genai.Schema{
				Type:  genai.TypeArray,
				Items: &genai.Schema{Type: genai.TypeString},
			},
			wantWrapperKey: "result",
		},
		{
			name: "array of integers",
			outputSchema: &genai.Schema{
				Type:  genai.TypeArray,
				Items: &genai.Schema{Type: genai.TypeInteger},
			},
			wantWrapperKey: "result",
		},
		{
			name: "array of objects",
			outputSchema: &genai.Schema{
				Type:  genai.TypeArray,
				Items: sampleOutputSchema,
			},
			wantWrapperKey: "result",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ft := newFinishTaskTool(t, tc.outputSchema)
			if got, want := ft.WrapperKey(), tc.wantWrapperKey; got != want {
				t.Errorf("wrapperKey = %q, want %q", got, want)
			}
		})
	}
}

func TestFinishTaskTool_Run(t *testing.T) {
	wantSuccess := map[string]any{"result": workflowinternal.FinishTaskSuccessResult}

	tests := []struct {
		name         string
		outputSchema *genai.Schema
		args         any
		want         map[string]any
		wantErr      bool
		// wantErrSubstr, when non-empty, asserts the returned tool result has
		// an `error` key containing the substring. wantErr must be false.
		wantErrSubstr []string
	}{
		{
			name:         "default schema, valid args",
			outputSchema: nil,
			args:         map[string]any{"result": "done"},
			want:         wantSuccess,
		},
		{
			name:         "object schema, valid args",
			outputSchema: sampleOutputSchema,
			args:         map[string]any{"result": "success", "count": 42.0},
			want:         wantSuccess,
		},
		{
			name:         "string wrapper, valid args",
			outputSchema: &genai.Schema{Type: genai.TypeString},
			args:         map[string]any{"result": "hello"},
			want:         wantSuccess,
		},
		{
			name:         "integer wrapper, valid args",
			outputSchema: &genai.Schema{Type: genai.TypeInteger},
			args:         map[string]any{"result": 42.0},
			want:         wantSuccess,
		},
		{
			name:         "boolean wrapper, valid args",
			outputSchema: &genai.Schema{Type: genai.TypeBoolean},
			args:         map[string]any{"result": true},
			want:         wantSuccess,
		},
		{
			name:         "number wrapper, valid args",
			outputSchema: &genai.Schema{Type: genai.TypeNumber},
			args:         map[string]any{"result": 3.14},
			want:         wantSuccess,
		},
		{
			name: "array of strings wrapper, valid args",
			outputSchema: &genai.Schema{
				Type:  genai.TypeArray,
				Items: &genai.Schema{Type: genai.TypeString},
			},
			args: map[string]any{"result": []any{"a", "b", "c"}},
			want: wantSuccess,
		},
		{
			name: "array of integers wrapper, valid args",
			outputSchema: &genai.Schema{
				Type:  genai.TypeArray,
				Items: &genai.Schema{Type: genai.TypeInteger},
			},
			args: map[string]any{"result": []any{1.0, 2.0, 3.0}},
			want: wantSuccess,
		},
		{
			name: "array of strings wrapper, empty args",
			outputSchema: &genai.Schema{
				Type:  genai.TypeArray,
				Items: &genai.Schema{Type: genai.TypeString},
			},
			args: map[string]any{"result": []any{}},
			want: wantSuccess,
		},
		{
			name:          "validation error: missing required field",
			outputSchema:  sampleOutputSchema,
			args:          map[string]any{"result": "success"},
			wantErrSubstr: []string{"finish_task", "validation errors", "count"},
		},
		{
			name:          "validation error: wrong type",
			outputSchema:  sampleOutputSchema,
			args:          map[string]any{"result": "success", "count": "not_an_int"},
			wantErrSubstr: []string{"finish_task", "validation errors"},
		},
		{
			name:    "nil args",
			args:    nil,
			wantErr: true,
		},
		{
			name:    "non-map args",
			args:    "not a map",
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ft := newFinishTaskTool(t, tc.outputSchema)
			got, err := ft.Run(nil, tc.args)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("Run() error = nil, want non-nil; result = %v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("Run() unexpected error: %v", err)
			}

			if len(tc.wantErrSubstr) > 0 {
				gotErr, ok := got["error"].(string)
				if !ok {
					t.Fatalf("Run() result missing %q key or wrong type, got %v",
						"error", got)
				}
				for _, sub := range tc.wantErrSubstr {
					if !strings.Contains(gotErr, sub) {
						t.Errorf("Run() error %q does not contain substring %q",
							gotErr, sub)
					}
				}
				return
			}

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFinishTaskTool_ProcessRequest(t *testing.T) {
	ft := newFinishTaskTool(t, nil)
	req := &model.LLMRequest{}

	if err := ft.ProcessRequest(nil, req); err != nil {
		t.Fatalf("ProcessRequest() failed: %v", err)
	}

	// The tool should be registered on the request.
	if _, ok := req.Tools[workflowinternal.FinishTaskToolName]; !ok {
		t.Errorf("req.Tools does not include %q; got keys %v",
			workflowinternal.FinishTaskToolName, toolKeys(req.Tools))
	}

	// The function declaration should also be present.
	decls := utils.FunctionDecls(req.Config)
	if !slices.ContainsFunc(decls, func(d *genai.FunctionDeclaration) bool {
		return d.Name == workflowinternal.FinishTaskToolName
	}) {
		t.Errorf("req.Config function declarations do not include %q; got %v",
			workflowinternal.FinishTaskToolName, decls)
	}

	// The finish_task instruction should have been appended to the system
	// instruction.
	instructions := utils.TextParts(req.Config.SystemInstruction)
	wantSubstrings := []string{
		"finish_task",
		"Do NOT call 'finish_task' prematurely",
		"call 'finish_task' by itself with",
	}
	if !slices.ContainsFunc(instructions, func(s string) bool {
		for _, sub := range wantSubstrings {
			if !strings.Contains(s, sub) {
				return false
			}
		}
		return true
	}) {
		t.Errorf("system instruction does not contain finish_task guidance phrases\n"+
			"  want all substrings: %q\n"+
			"  got instructions: %v",
			wantSubstrings, instructions)
	}
}

func TestFinishTaskToolConstants(t *testing.T) {
	if got, want := workflowinternal.FinishTaskToolName, "finish_task"; got != want {
		t.Errorf("FinishTaskToolName = %q, want %q", got, want)
	}
	if got, want := workflowinternal.FinishTaskSuccessResult, "Task completed."; got != want {
		t.Errorf("FinishTaskSuccessResult = %q, want %q", got, want)
	}
}

// propertyKeys returns the sorted keys of a schema properties map, for
// stable error messages.
func propertyKeys(m map[string]*genai.Schema) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

// toolKeys returns the sorted keys of req.Tools for stable error messages.
func toolKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}
