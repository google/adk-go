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
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/workflowinternal"
)

// intMapSchema is a map[string]int output schema, which a *genai.Schema
// cannot express.
func intMapSchema() *jsonschema.Schema {
	return &jsonschema.Schema{Type: "object", AdditionalProperties: &jsonschema.Schema{Type: "integer"}}
}

func newJSONSchemaFinishTaskTool(t *testing.T, s *jsonschema.Schema) *workflowinternal.FinishTaskTool {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: "test_agent", Mode: llmagent.ModeTask, OutputJsonSchema: s})
	if err != nil {
		t.Fatalf("llmagent.New failed: %v", err)
	}
	tt, err := workflowinternal.NewFinishTaskTool(a)
	if err != nil {
		t.Fatalf("NewFinishTaskTool failed: %v", err)
	}
	ft, ok := tt.(*workflowinternal.FinishTaskTool)
	if !ok {
		t.Fatalf("NewFinishTaskTool returned %T, want *workflowinternal.FinishTaskTool", tt)
	}
	return ft
}

// jsonOf returns v marshaled and unmarshaled into generic values, so that a
// test compares JSON content and not key order.
func jsonOf(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	return out
}

// TestFinishTask_JSONSchemaValidArgsFinishTheTask fails if finish_task ignores
// the agent's OutputJsonSchema, declaring the default "result" schema and
// rejecting a correct map answer.
func TestFinishTask_JSONSchemaValidArgsFinishTheTask(t *testing.T) {
	ft := newJSONSchemaFinishTaskTool(t, intMapSchema())

	got, err := ft.Run(nil, map[string]any{"a": float64(1)})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if diff := cmp.Diff(map[string]any{"result": "Task completed."}, got); diff != "" {
		t.Errorf("Run() mismatch (-want +got):\n%s", diff)
	}

	decl := ft.Declaration()
	if decl.Parameters != nil {
		t.Errorf("Declaration().Parameters = %v, want nil", decl.Parameters)
	}
	wantParams := map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "integer"}}
	if diff := cmp.Diff(wantParams, jsonOf(t, decl.ParametersJsonSchema)); diff != "" {
		t.Errorf("Declaration().ParametersJsonSchema mismatch (-want +got):\n%s", diff)
	}
	if !strings.HasSuffix(decl.Description, " Pass the required output data in the parameters.") {
		t.Errorf("Declaration().Description = %q, want the output-data suffix", decl.Description)
	}
}

// TestFinishTask_JSONSchemaInvalidArgsReturnFeedback fails if finish_task does
// not validate the answer against the OutputJsonSchema, so an answer of the
// wrong type finishes the task.
func TestFinishTask_JSONSchemaInvalidArgsReturnFeedback(t *testing.T) {
	ft := newJSONSchemaFinishTaskTool(t, intMapSchema())

	got, err := ft.Run(nil, map[string]any{"a": "x"})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	msg, _ := got["error"].(string)
	if !strings.Contains(msg, "validation errors") {
		t.Errorf("Run() = %v, want an error entry with validation errors", got)
	}
}

// TestFinishTask_JSONSchemaNonObjectRootIsWrapped fails if a non-object
// OutputJsonSchema is declared as the parameters themselves, which a function
// declaration cannot accept, or if the call is not validated under the
// wrapper key.
func TestFinishTask_JSONSchemaNonObjectRootIsWrapped(t *testing.T) {
	ft := newJSONSchemaFinishTaskTool(t, &jsonschema.Schema{Type: "array", Items: &jsonschema.Schema{Type: "string"}})

	if got := ft.WrapperKey(); got != "result" {
		t.Errorf("WrapperKey() = %q, want %q", got, "result")
	}
	wantParams := map[string]any{
		"type":       "object",
		"properties": map[string]any{"result": map[string]any{"type": "array", "items": map[string]any{"type": "string"}}},
		"required":   []any{"result"},
	}
	if diff := cmp.Diff(wantParams, jsonOf(t, ft.Declaration().ParametersJsonSchema)); diff != "" {
		t.Errorf("Declaration().ParametersJsonSchema mismatch (-want +got):\n%s", diff)
	}
	got, err := ft.Run(nil, map[string]any{"result": []any{"x", "y"}})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if diff := cmp.Diff(map[string]any{"result": "Task completed."}, got); diff != "" {
		t.Errorf("Run() mismatch (-want +got):\n%s", diff)
	}
}

// TestTaskAgentTool_JSONSchemaSetsResponseSchema fails if the task agent tool
// treats an agent with only OutputJsonSchema as having no output schema and
// declares a STRING response.
func TestTaskAgentTool_JSONSchemaSetsResponseSchema(t *testing.T) {
	a, err := llmagent.New(llmagent.Config{
		Name:             taskAgentName,
		Description:      taskAgentDesc,
		Model:            &fakeLLM{backend: genai.BackendVertexAI},
		OutputJsonSchema: intMapSchema(),
	})
	if err != nil {
		t.Fatalf("llmagent.New failed: %v", err)
	}

	got := newTaskAgentTool(t, a).Declaration().ResponseJsonSchema

	if diff := cmp.Diff(&genai.Schema{Type: "OBJECT"}, got); diff != "" {
		t.Errorf("Declaration().ResponseJsonSchema mismatch (-want +got):\n%s", diff)
	}
}

// TestFinishTask_JSONSchemaNullResultReturnsFeedback fails if finish_task
// accepts a null result for a nullable non-object OutputJsonSchema. The
// unwrapped null becomes the task's output, which reads as "the task did not
// finish", so the delegation would never close; the genai branch refuses it
// for the same reason.
func TestFinishTask_JSONSchemaNullResultReturnsFeedback(t *testing.T) {
	tests := []struct {
		name   string
		schema *jsonschema.Schema
	}{
		{name: "type list", schema: &jsonschema.Schema{Types: []string{"null", "string"}}},
		{name: "anyOf", schema: &jsonschema.Schema{AnyOf: []*jsonschema.Schema{{Type: "string"}, {Type: "null"}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ft := newJSONSchemaFinishTaskTool(t, tt.schema)

			gotNull, err := ft.Run(nil, map[string]any{"result": nil})
			if err != nil {
				t.Fatalf("Run(null) error = %v", err)
			}
			gotValue, err := ft.Run(nil, map[string]any{"result": "done"})
			if err != nil {
				t.Fatalf("Run(value) error = %v", err)
			}

			if msg, _ := gotNull["error"].(string); !strings.Contains(msg, "validation errors") {
				t.Errorf("Run(null) = %v, want an error entry with validation errors", gotNull)
			}
			if diff := cmp.Diff(map[string]any{"result": "Task completed."}, gotValue); diff != "" {
				t.Errorf("Run(value) mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
