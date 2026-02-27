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
		name    string
		model   string
		variant genai.Backend
		want    bool
	}{
		{"Gemini2.0_Vertex", "gemini-2.0-flash", genai.BackendVertexAI, false},
		{"Gemini2.0_GeminiAPI", "gemini-2.0-flash", genai.BackendGeminiAPI, true},
		{"NonGemini_Vertex", "not-a-gemini", genai.BackendVertexAI, false},
		{"Gemini3.0_GeminiAPI", "gemini-3.0", genai.BackendGeminiAPI, false},
		{"Gemini3.0_Vertex", "gemini-3.0", genai.BackendVertexAI, false},
		{"CustomGemini2", "gemini-2.0-hack", genai.BackendUnspecified, false},
		{"CustomGemini3", "gemini-3.0-hack", genai.BackendUnspecified, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := NeedsOutputSchemaProcessor(&mockGoogleLLM{
				variant: tc.variant,
				nameVal: tc.model,
			})
			if got != tc.want {
				t.Errorf("NeedsOutputSchemaProcessor(%q) = %v, want %v", tc.model, got, tc.want)
			}
		})
	}
}

type mockGoogleLLM struct {
	model.LLM
	variant genai.Backend
	nameVal string
}

var _ GoogleLLM = (*mockGoogleLLM)(nil)

func (m *mockGoogleLLM) GetGoogleLLMVariant() genai.Backend {
	return m.variant
}

func (m *mockGoogleLLM) Name() string {
	return m.nameVal
}
