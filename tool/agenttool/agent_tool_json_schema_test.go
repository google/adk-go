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

package agenttool_test

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/tool/agenttool"
)

// TestAgentTool_Run_OutputJSONSchema fails if the agent tool returns the
// sub-agent's reply as text under "result" when the sub-agent has an
// OutputJsonSchema, keeps a code fence around the JSON, returns a non-object
// value without the "result" key, or accepts a reply that is not JSON.
func TestAgentTool_Run_OutputJSONSchema(t *testing.T) {
	tests := []struct {
		name    string
		reply   string
		want    map[string]any
		wantErr string
	}{
		{name: "object", reply: `{"a": 1}`, want: map[string]any{"a": float64(1)}},
		{name: "fenced object", reply: "```json\n{\"a\": 1}\n```", want: map[string]any{"a": float64(1)}},
		{name: "array", reply: `[1, 2]`, want: map[string]any{"result": []any{float64(1), float64(2)}}},
		{name: "not JSON", reply: "not json", wantErr: "json_agent"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sub, err := llmagent.New(llmagent.Config{
				Name:             "json_agent",
				Model:            &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText(tt.reply, genai.RoleModel)}},
				OutputJsonSchema: &jsonschema.Schema{Type: "object", AdditionalProperties: &jsonschema.Schema{Type: "integer"}},
			})
			if err != nil {
				t.Fatalf("llmagent.New failed: %v", err)
			}
			toolImpl, ok := agenttool.New(sub, nil).(toolinternal.FunctionTool)
			if !ok {
				t.Fatal("agentTool does not implement FunctionTool")
			}

			got, err := toolImpl.Run(createToolContext(t, sub), map[string]any{"request": "go"})

			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("Run() error = %v, want an error containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Run() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
