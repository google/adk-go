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

package agent

import (
	"context"

	"google.golang.org/adk/types"
)

func identityRequestProcessor(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/identity.py)
	return nil
}

func nlPlanningRequestProcessor(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/_nl_plnning.py)
	return nil
}

func codeExecutionRequestProcessor(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/_code_execution.py)
	return nil
}

func authPreprocesssor(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/auth/auth_preprocessor.py)
	return nil
}

func nlPlanningResponseProcessor(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest, resp *types.LLMResponse) error {
	// TODO: implement (adk-python src/google/adk/_nl_planning.py)
	return nil
}

func codeExecutionResponseProcessor(ctx context.Context, parentCtx *types.InvocationContext, req *types.LLMRequest, resp *types.LLMResponse) error {
	// TODO: implement (adk-python src/google/adk_code_execution.py)
	return nil
}
