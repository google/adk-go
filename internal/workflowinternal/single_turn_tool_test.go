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
	"errors"
	"iter"
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
	"google.golang.org/adk/session"
)

const (
	agentName = "math_agent"
	agentDesc = "Solves math problems."
)

var sampleInputSchema = &genai.Schema{
	Type: "OBJECT",
	Properties: map[string]*genai.Schema{
		"is_magic": {Type: "BOOLEAN"},
		"name":     {Type: "STRING"},
	},
	Required: []string{"is_magic", "name"},
}

var defaultDecl = &genai.FunctionDeclaration{
	Name:        agentName,
	Description: agentDesc,
	Parameters: &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"request": {Type: "STRING"},
		},
		Required: []string{"request"},
	},
}

func TestSingleTurnTool_Metadata(t *testing.T) {
	a := newLLMAgent(t, agentName, agentDesc, nil)
	st := newSingleTurnTool(t, a)

	if got, want := st.Name(), agentName; got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
	if got, want := st.Description(), agentDesc; got != want {
		t.Errorf("Description() = %q, want %q", got, want)
	}
	if st.IsLongRunning() {
		t.Errorf("IsLongRunning() = true, want false")
	}
}

func TestSingleTurnTool_Declaration(t *testing.T) {
	tests := []struct {
		name  string
		agent agent.Agent
		want  *genai.FunctionDeclaration
	}{
		{
			name:  "no input schema falls back to {request: STRING}",
			agent: newLLMAgent(t, agentName, agentDesc, nil),
			want:  defaultDecl,
		},
		{
			name:  "uses wrapped agent's input schema as is",
			agent: newLLMAgent(t, agentName, agentDesc, sampleInputSchema),
			want: &genai.FunctionDeclaration{
				Name:        agentName,
				Description: agentDesc,
				Parameters:  sampleInputSchema,
			},
		},
		{
			name: "composite agent recurses into first sub-agent's schema",
			agent: newCompositeAgent(t, "parent_no_schema", "Outer composite.",
				newLLMAgent(t, "child_with_schema", "Inner agent.", sampleInputSchema)),
			want: &genai.FunctionDeclaration{
				Name:        "parent_no_schema",
				Description: "Outer composite.",
				Parameters:  sampleInputSchema,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := newSingleTurnTool(t, tc.agent).Declaration()
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("Declaration() diff (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSingleTurnTool_Run_LLMRecoverableFailures(t *testing.T) {
	tests := []struct {
		name        string
		inputSchema *genai.Schema
		args        any
		wantSubstr  []string
	}{
		{
			name:       "args is not a map",
			args:       "not a map",
			wantSubstr: []string{"map[string]any"},
		},
		{
			name:        "schema: extra field rejected",
			inputSchema: sampleInputSchema,
			args:        map[string]any{"is_magic": true, "name": "x", "name_invalid": "y"},
			wantSubstr:  []string{"argument validation failed", agentName, "name_invalid"},
		},
		{
			name:        "schema: missing required field",
			inputSchema: sampleInputSchema,
			args:        map[string]any{"is_magic": true},
			wantSubstr:  []string{"argument validation failed", agentName, "name"},
		},
		{
			name:        "schema: wrong type for boolean field",
			inputSchema: sampleInputSchema,
			args:        map[string]any{"is_magic": "not a bool", "name": "x"},
			wantSubstr:  []string{"argument validation failed", agentName, "is_magic"},
		},
		{
			name:        "schema: wrong type for string field",
			inputSchema: sampleInputSchema,
			args:        map[string]any{"is_magic": true, "name": 42.0},
			wantSubstr:  []string{"argument validation failed", agentName, "name"},
		},
		{
			name:       "default schema: empty args missing required \"request\"",
			args:       map[string]any{},
			wantSubstr: []string{"argument validation failed", "request"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			st := newSingleTurnTool(t, newLLMAgent(t, agentName, agentDesc, tc.inputSchema))

			got, err := st.Run(nil, tc.args)
			if err != nil {
				t.Fatalf("Run(%v) err = %v, want nil (LLM-recoverable failure surfaces via result[\"error\"])",
					tc.args, err)
			}
			errStr := unpackErrorResult(t, got)
			for _, sub := range tc.wantSubstr {
				if !strings.Contains(errStr, sub) {
					t.Errorf("result[\"error\"] = %q, want substring %q", errStr, sub)
				}
			}
		})
	}
}

func TestSingleTurnTool_ProcessRequest_RegistersTool(t *testing.T) {
	st := newSingleTurnTool(t, newLLMAgent(t, agentName, agentDesc, nil))

	req := &model.LLMRequest{}
	if err := st.ProcessRequest(nil, req); err != nil {
		t.Fatalf("ProcessRequest() err = %v", err)
	}

	if _, ok := req.Tools[agentName]; !ok {
		t.Errorf("req.Tools missing %q; got keys %v", agentName, sortedKeys(req.Tools))
	}
	decls := utils.FunctionDecls(req.Config)
	if !slices.ContainsFunc(decls, func(d *genai.FunctionDeclaration) bool { return d.Name == agentName }) {
		var names []string
		for _, d := range decls {
			if d != nil {
				names = append(names, d.Name)
			}
		}
		t.Errorf("req.Config function declarations missing %q; got %v", agentName, names)
	}
}

func newSingleTurnTool(t *testing.T, a agent.Agent) *workflowinternal.SingleTurnTool {
	t.Helper()
	tt, err := workflowinternal.NewSingleTurnTool(a)
	if err != nil {
		t.Fatalf("NewSingleTurnTool: %v", err)
	}
	st, ok := tt.(*workflowinternal.SingleTurnTool)
	if !ok {
		t.Fatalf("NewSingleTurnTool returned %T, want *workflowinternal.SingleTurnTool", tt)
	}
	return st
}

func newLLMAgent(t *testing.T, name, description string, inputSchema *genai.Schema) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{
		Name:        name,
		Description: description,
		InputSchema: inputSchema,
	})
	if err != nil {
		t.Fatalf("llmagent.New(%q): %v", name, err)
	}
	return a
}

func newCompositeAgent(t *testing.T, name, description string, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name:        name,
		Description: description,
		SubAgents:   subAgents,
		Run: func(_ agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, errors.New("custom agent Run should not be invoked in unit tests"))
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%q): %v", name, err)
	}
	return a
}

func unpackErrorResult(t *testing.T, result map[string]any) string {
	t.Helper()
	if result == nil {
		t.Fatalf("Run result = nil, want map with \"error\" key")
	}
	raw, ok := result["error"]
	if !ok {
		t.Fatalf("Run result missing \"error\" key; got keys %v", sortedKeys(result))
	}
	switch v := raw.(type) {
	case error:
		return v.Error()
	case string:
		return v
	default:
		t.Fatalf("result[\"error\"] has unexpected type %T (%v)", raw, raw)
		return ""
	}
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}
