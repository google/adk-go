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

	"google.golang.org/adk/v2/model"
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

func TestIsGemini1(t *testing.T) {
	testCases := []struct {
		model string
		want  bool
	}{
		{"gemini-1", true},
		{"gemini-1.5", true},
		{"gemini-1.5-flash", true},
		{"gemini-1.5-pro", true},
		{"gemini-10", false},
		{"gemini-10-flash", false},
		{"gemini-2.0-flash", false},
		{"gemini-2.5-flash-lite", false},
		{"gemini-2.0-flash-exp", false},
		{"gemini-1.0-pro", true},
		{"projects/p/locations/l/models/gemini-2.0-flash", false},
		{"projects/p/locations/l/models/gemini-1.0-flash", true},
		{"projects/p/locations/l/models/gemini-10-flash", false},
		{"models/gemini-1.5-pro", true},
		{"not-a-gemini-model", false},
		{"gemini-2", false},
		{"gemini-3.0", false},
		{"gemini-3-pro", false},
	}

	for _, tc := range testCases {
		got := IsGemini1Model(tc.model)
		if got != tc.want {
			t.Errorf("IsGemini1Model(%q) = %v, want %v", tc.model, got, tc.want)
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

func TestIsGemini3xLive(t *testing.T) {
	testCases := []struct {
		model string
		want  bool
	}{
		{"gemini-3.1-flash-live-preview", true},
		{"gemini-3.8-live", true},
		{"models/gemini-3.8-live", true},
		{"projects/p/locations/l/models/gemini-3.1-flash-live-preview", true},
		{"gemini-3.5-flash-live-preview", true},
		{"models/gemini-3.5-live-translate-preview", false},
		{"gemini-3.5-live-translate", false},
		{"gemini-3.1-pro-preview", false},
		{"gemini-2.0-flash-live-001", false},
	}

	for _, tc := range testCases {
		got := IsGemini3xLive(tc.model)
		if got != tc.want {
			t.Errorf("IsGemini3xLive(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

// TestGeminiOutputSchemaAndTools pins adk-python's
// gemini_output_schema_and_tools (models/_capabilities.py): any Gemini model
// name on Vertex AI, with no version check, and nothing else.
func TestGeminiOutputSchemaAndTools(t *testing.T) {
	testCases := []struct {
		name    string
		model   string
		variant genai.Backend
		want    bool
	}{
		{"Gemini2.0_Vertex", "gemini-2.0-flash", genai.BackendVertexAI, true},
		{"Gemini3.0_Vertex", "gemini-3.0", genai.BackendVertexAI, true},
		{"Gemini1.5_Vertex", "gemini-1.5-pro", genai.BackendVertexAI, true},
		{"Gemini3.0_GeminiAPI", "gemini-3.0", genai.BackendGeminiAPI, false},
		{"Gemini2.0_Unspecified", "gemini-2.0-flash", genai.BackendUnspecified, false},
		{"NonGemini_Vertex", "not-a-gemini", genai.BackendVertexAI, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := GeminiOutputSchemaAndTools(&mockGoogleLLM{
				variant: tc.variant,
				nameVal: tc.model,
			})
			if got != tc.want {
				t.Errorf("GeminiOutputSchemaAndTools(%q) = %v, want %v", tc.model, got, tc.want)
			}
		})
	}
	if GeminiOutputSchemaAndTools(nil) {
		t.Error("GeminiOutputSchemaAndTools(nil) = true, want false")
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
