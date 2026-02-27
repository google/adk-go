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
	"context"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

func TestIsGemini25OrLower(t *testing.T) {
	testCases := []struct {
		model string
		want  bool
	}{
		{"gemini-1.5-pro", true},
		{"gemini-2.0-flash", true},
		{"gemini-2.5-flash-lite", true},
		{"gemini-2.0-flash-exp", true},
		{"gemini-1.0-pro", true},
		{"projects/p/locations/l/models/gemini-2.0-flash", true},
		{"models/gemini-1.5-pro", true},
		{"not-a-gemini-model", false},
		{"gemini-2", true},
		{"gemini-3.0", false},
		{"gemini-3-pro", false},
	}

	for _, tc := range testCases {
		got := IsGemini25OrLower(tc.model)
		if got != tc.want {
			t.Errorf("IsGemini25OrLower(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestIsGeminiModel(t *testing.T) {
	testCases := []struct {
		model string
		want  bool
	}{
		{"gemini-1.5-pro", true},
		{"models/gemini-2.0-flash", true},
		{"claud-3.5-sonnet", false},
	}

	for _, tc := range testCases {
		got := IsGeminiModel(tc.model)
		if got != tc.want {
			t.Errorf("IsGeminiModel(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestNeedsOutputSchemaProcessor(t *testing.T) {
	testCases := []struct {
		name string
		llm  model.LLM
		want bool
	}{
		// Gemini models on Vertex AI have native support
		{"Gemini2.0_Vertex", &mockGoogleLLM{nameVal: "gemini-2.0-flash", variant: genai.BackendVertexAI}, false},
		{"Gemini3.0_Vertex", &mockGoogleLLM{nameVal: "gemini-3.0", variant: genai.BackendVertexAI}, false},
		// Gemini <= 2.5 on Gemini API need the processor
		{"Gemini2.0_GeminiAPI", &mockGoogleLLM{nameVal: "gemini-2.0-flash", variant: genai.BackendGeminiAPI}, true},
		// Gemini >= 3.0 on Gemini API have native support
		{"Gemini3.0_GeminiAPI", &mockGoogleLLM{nameVal: "gemini-3.0", variant: genai.BackendGeminiAPI}, false},
		// Unspecified backend defaults to no processor (conservative)
		{"CustomGemini2", &mockGoogleLLM{nameVal: "gemini-2.0-hack", variant: genai.BackendUnspecified}, false},
		{"CustomGemini3", &mockGoogleLLM{nameVal: "gemini-3.0-hack", variant: genai.BackendUnspecified}, false},
		// Non-Gemini models (Bedrock/Claude/GPT) always need the processor
		{"Claude_Sonnet", &simpleLLM{name: "claude-3-5-sonnet"}, true},
		{"Claude_Opus", &simpleLLM{name: "anthropic.claude-v2"}, true},
		{"Bedrock_Claude", &simpleLLM{name: "bedrock/anthropic.claude-3-sonnet"}, true},
		{"GPT4", &simpleLLM{name: "gpt-4"}, true},
		{"CustomModel", &simpleLLM{name: "my-custom-model"}, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := NeedsOutputSchemaProcessor(tc.llm)
			if got != tc.want {
				t.Errorf("NeedsOutputSchemaProcessor(%q) = %v, want %v", tc.llm.Name(), got, tc.want)
			}
		})
	}
}

type mockGoogleLLM struct {
	model.LLM
	variant genai.Backend
	nameVal string
}

func (m *mockGoogleLLM) GetGoogleLLMVariant() genai.Backend {
	return m.variant
}

func (m *mockGoogleLLM) Name() string {
	return m.nameVal
}

var _ GoogleLLM = (*mockGoogleLLM)(nil)

// simpleLLM is a minimal LLM mock for non-Google LLMs (e.g., Bedrock, OpenAI)
// that don't implement the GoogleLLM interface.
type simpleLLM struct {
	name string
}

func (m *simpleLLM) Name() string {
	return m.name
}

func (m *simpleLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return nil
}

var _ model.LLM = (*simpleLLM)(nil)
