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
	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/llminternal/googlellm"
	"google.golang.org/adk/model"
)

// The Gemini API (non-Vertex) backend does not support the display_name parameter for file uploads,
// so it must be removed to prevent request failures.
func removeDisplayNameIfExists(ctx agent.InvocationContext, req *model.LLMRequest) error {
	if req.Contents == nil {
		return nil
	}
	for _, content := range req.Contents {
		if content.Parts == nil {
			continue
		}
		if googlellm.GetGoogleLLMVariant() == googlellm.GoogleLLMVariantGeminiAPI {
			for _, part := range content.Parts {
				if part.InlineData != nil {
					part.InlineData.DisplayName = ""
				}
				if part.FileData != nil {
					part.FileData.DisplayName = ""
				}
			}
		}
	}
	return nil
}
