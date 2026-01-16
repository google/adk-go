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
	"iter"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func identityRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/identity.py)
	return func(yield func(*session.Event, error) bool) {}
}

func nlPlanningRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/_nl_plnning.py)
	return func(yield func(*session.Event, error) bool) {}
}

func codeExecutionRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/_code_execution.py)
	return func(yield func(*session.Event, error) bool) {}
}

func authPreprocessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	// TODO: implement (adk-python src/google/adk/auth/auth_preprocessor.py)
	return func(yield func(*session.Event, error) bool) {}
}

func nlPlanningResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error {
	// TODO: implement (adk-python src/google/adk/_nl_planning.py)
	return nil
}

func codeExecutionResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error {
	// TODO: implement (adk-python src/google/adk_code_execution.py)
	return nil
}
