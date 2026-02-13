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
	"regexp"
	"strconv"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

var geminiModelVersionRegex = regexp.MustCompile(`^gemini-(\d+(\.\d+)?)`)

// GetGoogleLLMVariant returns the Google LLM variant used (GeminiAPI or VertexAI).
func GetGoogleLLMVariant(llm model.LLM) genai.Backend {
	i, ok := llm.(GoogleLLM)
	if !ok {
		return genai.BackendUnspecified
	}
	return i.GetGoogleLLMVariant()
}

// GoogleLLM is an interface which allows to distinguish between Vertex AI and Gemini API models.
type GoogleLLM interface {
	GetGoogleLLMVariant() genai.Backend
}

// IsGeminiModel returns true if the model is a Gemini model.
func IsGeminiModel(model string) bool {
	return strings.HasPrefix(extractModelName(model), "gemini-")
}

// IsGemini25OrLower returns true if the model is a Gemini 2.5 or less.
// These models do not support output schema with tools natively, so we need to use a processor to handle it.
func IsGemini25OrLower(model string) bool {
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
	return version <= 2.5
}

// IsGeminiAPIVariant returns true if the model is a Gemini API model (not Vertex AI).
func IsGeminiAPIVariant(llm model.LLM) bool {
	return GetGoogleLLMVariant(llm) == genai.BackendGeminiAPI
}

// NeedsOutputSchemaProcessor returns true if the Gemini model doesn't support output schema with tools natively and requires a processor to handle it.
// Only Gemini 2.5 models and lower and only in Gemini API don't support natively, so we enable the processor for them.
func NeedsOutputSchemaProcessor(llm model.LLM) bool {
	if llm == nil {
		return false
	}
	return IsGeminiModel(llm.Name()) && IsGeminiAPIVariant(llm) && IsGemini25OrLower(llm.Name())
}

func extractModelName(model string) string {
	modelstring := model[strings.LastIndex(model, "/")+1:]
	return strings.ToLower(modelstring)
}
