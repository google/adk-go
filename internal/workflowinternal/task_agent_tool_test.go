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
	"context"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/workflowinternal"
	"google.golang.org/adk/model"
)

const (
	taskAgentName = "task_agent"
	taskAgentDesc = "Solves a delegated task."
	taskSuffix    = "IMPORTANT: This tool delegates execution to a specialized agent. Do NOT call this tool in parallel with any other tools."
)

var taskAgentOutputSchema = &genai.Schema{
	Type: "OBJECT",
	Properties: map[string]*genai.Schema{
		"answer": {Type: "STRING"},
	},
	Required: []string{"answer"},
}

var defaultTaskRequestParams = &genai.Schema{
	Type: "OBJECT",
	Properties: map[string]*genai.Schema{
		"request": {
			Type:        "STRING",
			Description: "Detailed instructions or context for the task sub-agent.",
		},
	},
	Required: []string{"request"},
}

func TestTaskAgentTool_Metadata(t *testing.T) {
	a := newLLMAgent(t, taskAgentName, taskAgentDesc, nil)
	tt := newTaskAgentTool(t, a)

	if got, want := tt.Name(), taskAgentName; got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
	if got, want := tt.Description(), taskAgentDesc; got != want {
		t.Errorf("Description() = %q, want %q", got, want)
	}
	if tt.IsLongRunning() {
		t.Errorf("IsLongRunning() = true, want false")
	}
	if !tt.DefersResponse() {
		t.Errorf("DefersResponse() = false, want true")
	}
	result, err := tt.Run(nil, nil)
	if err != nil {
		t.Errorf("Run() returned err = %v, want nil", err)
	}
	if result != nil {
		t.Errorf("Run() = %v, want nil (Run must be a no-op so the "+
			"DefersResponse gate skips auto-FR build)", result)
	}
}

func TestTaskAgentTool_Declaration(t *testing.T) {
	tests := []struct {
		name  string
		agent agent.Agent
		want  *genai.FunctionDeclaration
	}{
		{
			name:  "no input schema falls back to default {request: STRING}",
			agent: newLLMAgent(t, taskAgentName, taskAgentDesc, nil),
			want: &genai.FunctionDeclaration{
				Name:        taskAgentName,
				Description: taskAgentDesc + "\n" + taskSuffix,
				Parameters:  defaultTaskRequestParams,
				// Model is nil → IsGeminiAPIVariant returns false →
				// Vertex branch fires; no output schema → STRING.
				ResponseJsonSchema: &genai.Schema{Type: "STRING"},
			},
		},
		{
			name:  "uses wrapped agent's input schema as is",
			agent: newLLMAgent(t, taskAgentName, taskAgentDesc, sampleInputSchema),
			want: &genai.FunctionDeclaration{
				Name:               taskAgentName,
				Description:        taskAgentDesc + "\n" + taskSuffix,
				Parameters:         sampleInputSchema,
				ResponseJsonSchema: &genai.Schema{Type: "STRING"},
			},
		},
		{
			name: "composite agent recurses into first sub-agent for input schema",
			agent: newCompositeAgent(t, "parent_no_schema", "Outer composite.",
				newLLMAgent(t, "child_with_schema", "Inner agent.", sampleInputSchema)),
			want: &genai.FunctionDeclaration{
				Name:        "parent_no_schema",
				Description: "Outer composite." + "\n" + taskSuffix,
				Parameters:  sampleInputSchema,
			},
		},
		{
			name:  "empty wrapped-agent description is trimmed (no leading newline)",
			agent: newLLMAgent(t, taskAgentName, "", nil),
			want: &genai.FunctionDeclaration{
				Name:               taskAgentName,
				Description:        taskSuffix,
				Parameters:         defaultTaskRequestParams,
				ResponseJsonSchema: &genai.Schema{Type: "STRING"},
			},
		},
		{
			name:  "wrapped-agent description with trailing whitespace is trimmed",
			agent: newLLMAgent(t, taskAgentName, "Solves problems.  ", nil),
			want: &genai.FunctionDeclaration{
				Name:               taskAgentName,
				Description:        "Solves problems.  \n" + taskSuffix,
				Parameters:         defaultTaskRequestParams,
				ResponseJsonSchema: &genai.Schema{Type: "STRING"},
			},
		},
		{
			name:  "Gemini API variant suppresses ResponseJsonSchema",
			agent: newLLMAgentWithModel(t, taskAgentName, taskAgentDesc, nil, nil, &fakeLLM{backend: genai.BackendGeminiAPI}),
			want: &genai.FunctionDeclaration{
				Name:               taskAgentName,
				Description:        taskAgentDesc + "\n" + taskSuffix,
				Parameters:         defaultTaskRequestParams,
				ResponseJsonSchema: nil,
			},
		},
		{
			name:  "Vertex AI variant sets ResponseJsonSchema to STRING when no output schema",
			agent: newLLMAgentWithModel(t, taskAgentName, taskAgentDesc, nil, nil, &fakeLLM{backend: genai.BackendVertexAI}),
			want: &genai.FunctionDeclaration{
				Name:               taskAgentName,
				Description:        taskAgentDesc + "\n" + taskSuffix,
				Parameters:         defaultTaskRequestParams,
				ResponseJsonSchema: &genai.Schema{Type: "STRING"},
			},
		},
		{
			name:  "Vertex AI variant sets ResponseJsonSchema to OBJECT when output schema is present",
			agent: newLLMAgentWithModel(t, taskAgentName, taskAgentDesc, nil, taskAgentOutputSchema, &fakeLLM{backend: genai.BackendVertexAI}),
			want: &genai.FunctionDeclaration{
				Name:               taskAgentName,
				Description:        taskAgentDesc + "\n" + taskSuffix,
				Parameters:         defaultTaskRequestParams,
				ResponseJsonSchema: &genai.Schema{Type: "OBJECT"},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := newTaskAgentTool(t, tc.agent).Declaration()
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("Declaration() diff (-want +got):\n%s", diff)
			}
		})
	}
}

func newTaskAgentTool(t *testing.T, a agent.Agent) *workflowinternal.TaskAgentTool {
	t.Helper()
	tt, err := workflowinternal.NewTaskAgentTool(a)
	if err != nil {
		t.Fatalf("NewTaskAgentTool: %v", err)
	}
	st, ok := tt.(*workflowinternal.TaskAgentTool)
	if !ok {
		t.Fatalf("NewTaskAgentTool returned %T, want *workflowinternal.TaskAgentTool", tt)
	}
	return st
}

func newLLMAgentWithModel(t *testing.T, name, description string, inputSchema, outputSchema *genai.Schema, llm model.LLM) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{
		Name:         name,
		Description:  description,
		InputSchema:  inputSchema,
		OutputSchema: outputSchema,
		Model:        llm,
	})
	if err != nil {
		t.Fatalf("llmagent.New(%q): %v", name, err)
	}
	return a
}

type fakeLLM struct {
	backend genai.Backend
}

func (f *fakeLLM) Name() string { return "fake-model" }

func (f *fakeLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {}
}

func (f *fakeLLM) GetGoogleLLMVariant() genai.Backend { return f.backend }

var _ model.LLM = (*fakeLLM)(nil)
