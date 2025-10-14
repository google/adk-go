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

package googlellm

import (
	"os"
	"slices"
)

const (
	// For using credentials from Google Vertex AI
	GoogleLLMVariantVertexAI = "VERTEX_AI"
	// For using API Key from Google AI Studio
	GoogleLLMVariantGeminiAPI = "GEMINI_API"
)

// GetGoogleLLMVariant returns the Google LLM variant to use.
// see https://google.github.io/adk-docs/get-started/quickstart/#set-up-the-model
func GetGoogleLLMVariant() string {
	useVertexAI, _ := os.LookupEnv("GOOGLE_GENAI_USE_VERTEXAI")
	if slices.Contains([]string{"1", "true"}, useVertexAI) {
		return GoogleLLMVariantVertexAI
	}
	return GoogleLLMVariantGeminiAPI
}
