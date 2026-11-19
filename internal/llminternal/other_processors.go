// Copyright 2025 Google LLC
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

package llminternal

import (
	"fmt"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/codeexecutor"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func nlPlanningRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/_nl_plnning.py)
	return func(yield func(*session.Event, error) bool) {}
}

// codeExecutionRequestProcessor prepares req for the code executor
// resolveCodeExecutor returns.
// For a codeexecutor.BuiltIn it appends the model's code-execution tool as its
// own entry of req.Config.Tools, creating the config when it is nil, or fails
// when the request's model name is not a Gemini name and
// ADK_DISABLE_GEMINI_MODEL_ID_CHECK is not set (adk-python
// code_executors/built_in_code_executor.py, process_llm_request). For any
// other executor it runs codeExecutionPreProcess, yielding its events, and
// stops at its first error. Then, for every executor, it replaces the code
// execution parts of req.Contents with fenced text (adk-python
// flows/llm_flows/extensions/_code_execution.py), without editing the
// contents it replaces. It leaves req unchanged for a nil executor.
func codeExecutionRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		exec := resolveCodeExecutor(ctx)
		if exec == nil {
			return
		}
		if _, ok := exec.(codeexecutor.BuiltIn); ok {
			ok := googlellm.IsGeminiModelExact(req.Model) || googlellm.IsGeminiModelIDCheckDisabled()
			if !ok {
				yield(nil, fmt.Errorf("Gemini code execution tool is not supported for model %s", req.Model)) //nolint:staticcheck // ST1005: the message text is pinned by adk-python and by the conformance test
				return
			}
			if req.Config == nil {
				req.Config = &genai.GenerateContentConfig{}
			}
			req.Config.Tools = append(req.Config.Tools, &genai.Tool{CodeExecution: &genai.ToolCodeExecution{}})
		} else {
			for ev, err := range codeExecutionPreProcess(ctx, req, exec) {
				if err != nil {
					yield(nil, err)
					return
				}
				if !yield(ev, nil) {
					return
				}
			}
		}
		convertCodeExecutionHistory(req.Contents, exec.Settings())
	}
}

func nlPlanningResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) iter.Seq2[*session.Event, error] {
	// TODO: implement (adk-python src/google/adk/_nl_planning.py)
	return func(yield func(*session.Event, error) bool) {}
}

// codeExecutionResponseProcessor runs the code the model wrote and emits the
// code, result and image events; see codeExecutionPostProcess. req is not
// used.
func codeExecutionResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) iter.Seq2[*session.Event, error] {
	return codeExecutionPostProcess(ctx, resp)
}

// resolveCodeExecutor returns the code executor the flow uses for the agent
// in ctx in this invocation (adk-python _resolve_code_executor): the agent's
// own, or nil when the agent is not an LLM agent. When the run config has
// SupportCFC, the agent has no parent agent and its executor is not a
// codeexecutor.BuiltIn (a nil executor is not), it returns a new
// *codeexecutor.BuiltInExecutor instead. The agent is never changed, so the
// built-in executor lasts one invocation. A nil run config, as for an agent
// run without a runner, means no CFC.
//
// "No parent" is the static link agent.New records, not the runner's parent
// map: an agent that another agent lists as a sub-agent has a parent also when
// a runner has it as its root.
func resolveCodeExecutor(ctx agent.InvocationContext) codeexecutor.CodeExecutor {
	if asLLMAgent(ctx.Agent()) == nil {
		return nil
	}
	exec := getCodeExecutor(ctx)
	if rc := ctx.RunConfig(); rc == nil || !rc.SupportCFC {
		return exec
	}
	if _, ok := exec.(codeexecutor.BuiltIn); ok {
		return exec
	}
	if inner := innerAgent(ctx.Agent()); inner != nil {
		if parent, _ := agentinternal.ParentOf(agentinternal.Reveal(inner)); parent != nil {
			return exec
		}
	}
	return codeexecutor.NewBuiltInExecutor()
}

// getCodeExecutor returns the code executor of the agent in ctx, or nil when
// the agent is not an LLM agent or has no code executor.
func getCodeExecutor(ctx agent.InvocationContext) codeexecutor.CodeExecutor {
	llmAgent := asLLMAgent(ctx.Agent())
	if llmAgent == nil {
		return nil
	}
	return llmAgent.internal().CodeExecutor
}
