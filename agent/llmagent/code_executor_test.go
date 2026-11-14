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

package llmagent_test

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/codeexecutor"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

const disableModelIDCheckEnv = "ADK_DISABLE_GEMINI_MODEL_ID_CHECK"

// namedModel is a scriptedModel whose Name, which the flow sends as the
// request's model name, is set by the test.
type namedModel struct {
	*scriptedModel
	name string
}

func (m namedModel) Name() string { return m.name }

func newNamedModel(name string, responses ...*model.LLMResponse) namedModel {
	return namedModel{scriptedModel: &scriptedModel{responses: responses}, name: name}
}

// embeddingBuiltIn is a user type that embeds *codeexecutor.BuiltInExecutor,
// as a Python subclass of BuiltInCodeExecutor would.
type embeddingBuiltIn struct {
	*codeexecutor.BuiltInExecutor
}

// localExecutor is a CodeExecutor that is not a codeexecutor.BuiltIn.
type localExecutor struct{}

func (localExecutor) Settings() codeexecutor.Settings { return codeexecutor.DefaultSettings() }

func (localExecutor) ExecuteCode(context.Context, agent.InvocationContext, *codeexecutor.Input) (*codeexecutor.Result, error) {
	return &codeexecutor.Result{Stdout: "ran"}, nil
}

func newTestFuncTool(t *testing.T) tool.Tool {
	t.Helper()
	type args struct {
		X string `json:"x"`
	}
	ft, err := functiontool.New(functiontool.Config{Name: "test_func", Description: "a test function"},
		func(agent.Context, args) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}
	return ft
}

var codeExecutionEntry = &genai.Tool{CodeExecution: &genai.ToolCodeExecution{}}

func TestLLMAgent_BuiltInCodeExecutorRequestHookOrderAndConfigCopy(t *testing.T) {
	tests := []struct {
		name     string
		executor codeexecutor.CodeExecutor
	}{
		{name: "built_in_executor", executor: codeexecutor.NewBuiltInExecutor()},
		{name: "struct_embedding_built_in_executor", executor: embeddingBuiltIn{codeexecutor.NewBuiltInExecutor()}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			llm := newNamedModel("gemini-2.5-flash", textResponse("done"))
			// Length 0 passes llmagent.New's check on GenerateContentConfig.Tools;
			// the spare capacity shows whether the hook appends into this slice.
			agentCfg := &genai.GenerateContentConfig{Tools: make([]*genai.Tool, 0, 4)}
			a := newPlannerAgent(t, llmagent.Config{
				Name:                  "test_agent",
				Model:                 llm,
				GenerateContentConfig: agentCfg,
				Tools:                 []tool.Tool{newTestFuncTool(t)},
				CodeExecutor:          tc.executor,
			})

			run := runPlannerAgent(t, a)

			if run.err != nil {
				t.Fatalf("run error: %v", run.err)
			}
			reqs := llm.recorded()
			if len(reqs) != 1 {
				t.Fatalf("model calls = %d, want 1", len(reqs))
			}
			tools := reqs[0].Config.Tools
			if len(tools) != 2 {
				t.Fatalf("request has %d tool entries, want 2: %+v", len(tools), tools)
			}
			if diff := cmp.Diff(codeExecutionEntry, tools[0]); diff != "" {
				t.Errorf("request tool entry 0 mismatch (-want +got):\n%s", diff)
			}
			var gotNames []string
			for _, fd := range tools[1].FunctionDeclarations {
				gotNames = append(gotNames, fd.Name)
			}
			if diff := cmp.Diff([]string{"test_func"}, gotNames); diff != "" {
				t.Errorf("request tool entry 1 declaration names mismatch (-want +got):\n%s", diff)
			}
			if tools[1].CodeExecution != nil {
				t.Errorf("request tool entry 1 CodeExecution = %+v, want nil", tools[1].CodeExecution)
			}
			if len(agentCfg.Tools) != 0 {
				t.Errorf("agent GenerateContentConfig.Tools length after run = %d, want 0", len(agentCfg.Tools))
			}
			for i, tl := range agentCfg.Tools[:cap(agentCfg.Tools)] {
				if tl != nil {
					t.Errorf("agent GenerateContentConfig.Tools spare slot %d after run = %+v, want nil", i, tl)
				}
			}
		})
	}
}

func TestLLMAgent_BuiltInCodeExecutorModelCheckAndEnvBypass(t *testing.T) {
	t.Run("refuses_name", func(t *testing.T) {
		tests := []struct {
			name    string
			model   string
			wantErr string
		}{
			{
				name:    "upper_case_gemini",
				model:   "Gemini-2.5-flash",
				wantErr: "Gemini code execution tool is not supported for model Gemini-2.5-flash",
			},
			{
				name:    "empty",
				model:   "",
				wantErr: "Gemini code execution tool is not supported for model ",
			},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				llm := newNamedModel(tc.model, textResponse("done"))
				a := newPlannerAgent(t, llmagent.Config{
					Name:         "test_agent",
					Model:        llm,
					CodeExecutor: codeexecutor.NewBuiltInExecutor(),
				})

				run := runPlannerAgent(t, a)

				if run.err == nil || run.err.Error() != tc.wantErr {
					t.Errorf("run error = %v, want %q", run.err, tc.wantErr)
				}
				if got := len(llm.recorded()); got != 0 {
					t.Errorf("model calls = %d, want 0", got)
				}
			})
		}
	})

	t.Run("env_switch_read_on_every_run", func(t *testing.T) {
		llm := newNamedModel("claude-3-sonnet", textResponse("done"), textResponse("done"))
		a := newPlannerAgent(t, llmagent.Config{
			Name:         "test_agent",
			Model:        llm,
			CodeExecutor: codeexecutor.NewBuiltInExecutor(),
		})

		t.Setenv(disableModelIDCheckEnv, "true")
		run := runPlannerAgent(t, a)

		if run.err != nil {
			t.Fatalf("run with %s=true: error %v, want nil", disableModelIDCheckEnv, run.err)
		}
		reqs := llm.recorded()
		if len(reqs) != 1 {
			t.Fatalf("model calls with %s=true = %d, want 1", disableModelIDCheckEnv, len(reqs))
		}
		if diff := cmp.Diff([]*genai.Tool{codeExecutionEntry}, reqs[0].Config.Tools); diff != "" {
			t.Errorf("request tools with %s=true mismatch (-want +got):\n%s", disableModelIDCheckEnv, diff)
		}

		t.Setenv(disableModelIDCheckEnv, "false")
		run = runPlannerAgent(t, a)

		wantErr := "Gemini code execution tool is not supported for model claude-3-sonnet"
		if run.err == nil || run.err.Error() != wantErr {
			t.Errorf("run with %s=false: error %v, want %q", disableModelIDCheckEnv, run.err, wantErr)
		}
		if got := len(llm.recorded()); got != 1 {
			t.Errorf("model calls after the run with %s=false = %d, want 1", disableModelIDCheckEnv, got)
		}
	})
}

func TestLLMAgent_NonBuiltInOrNilCodeExecutorSkipsBuiltInHook(t *testing.T) {
	tests := []struct {
		name     string
		executor codeexecutor.CodeExecutor
	}{
		{name: "nil_executor", executor: nil},
		{name: "executor_that_is_not_built_in", executor: localExecutor{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			llm := newNamedModel("claude-3-sonnet", textResponse("done"))
			a := newPlannerAgent(t, llmagent.Config{
				Name:         "test_agent",
				Model:        llm,
				CodeExecutor: tc.executor,
			})

			run := runPlannerAgent(t, a)

			if run.err != nil {
				t.Fatalf("run error: %v", run.err)
			}
			reqs := llm.recorded()
			if len(reqs) != 1 {
				t.Fatalf("model calls = %d, want 1", len(reqs))
			}
			if got := reqs[0].Config.Tools; len(got) != 0 {
				t.Errorf("request tools = %+v, want none", got)
			}
		})
	}
}
