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
	"regexp"
	"slices"
	"strconv"
	"strings"
)

const (
	// For using credentials from Google Vertex AI
	GoogleLLMVariantVertexAI = "VERTEX_AI"
	// For using API Key from Google AI Studio
	GoogleLLMVariantGeminiAPI = "GEMINI_API"
)

var geminiModelVersionRegex = regexp.MustCompile(`^gemini-(\d+(\.\d+)?)`)

// GetGoogleLLMVariant returns the Google LLM variant to use.
// see https://google.github.io/adk-docs/get-started/quickstart/#set-up-the-model
func GetGoogleLLMVariant() string {
	useVertexAI, _ := os.LookupEnv("GOOGLE_GENAI_USE_VERTEXAI")
	if slices.Contains([]string{"1", "true"}, useVertexAI) {
		return GoogleLLMVariantVertexAI
	}
	return GoogleLLMVariantGeminiAPI
}

// IsVertexVariant returns true if the variant is Vertex AI.
func IsVertexVariant() bool {
	return GetGoogleLLMVariant() == GoogleLLMVariantVertexAI
}

// IsGeminiVariant returns true if the variant is Gemini API.
func IsGeminiVariant() bool {
	return GetGoogleLLMVariant() == GoogleLLMVariantGeminiAPI
}

// IsGeminiModel returns true if the model is a Gemini model.
func IsGeminiModel(model string) bool {
	return strings.HasPrefix(extractModelName(model), "gemini-")
}

// IsGemini2OrAbove returns true if the model is a Gemini 2.0 or above.
func IsGemini2OrAbove(model string) bool {
	model = extractModelName(model)
	// extract the model version from model name - e.g. turn gemini-2.5-flash or gemini-2.5-flash-lite into 2.5
	matches := geminiModelVersionRegex.FindStringSubmatch(model)
	if len(matches) < 2 {
		return false
	}
	version, err := strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return false
	}
	return version >= 2.0
}

// CanGeminiModelUseOutputSchemaWithTools returns true if the model is a Gemini model and the variant is Vertex AI and the model is a Gemini 2.x+ .
func CanGeminiModelUseOutputSchemaWithTools(model string) bool {
	return IsGeminiModel(model) && IsVertexVariant() && IsGemini2OrAbove(model)
}

func extractModelName(model string) string {
	modelstring := model[strings.LastIndex(model, "/")+1:]
	return strings.ToLower(modelstring)
}
