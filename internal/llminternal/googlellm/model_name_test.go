// Copyright 2026 Google LLC
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
	"testing"
)

// The want values are the results of adk-python v2.12.0's is_gemini_model
// (utils/model_name_utils.py) for the same names.
func TestIsGeminiModelExact(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		// Plain names: every gemini- version, case-sensitive prefix only.
		{model: "gemini-1.0-pro", want: true},
		{model: "gemini-1.5-pro", want: true},
		{model: "gemini-2.5-flash", want: true},
		{model: "gemini-3-pro", want: true},
		{model: "gemini-early-exp", want: true},
		{model: "gemini-2.5-PRO", want: true},
		{model: "", want: false},
		{model: "claude-3-sonnet", want: false},
		{model: "Gemini-2.5-flash", want: false},
		{model: "Gemini-2.5-pro", want: false},
		{model: "GEMINI-2.5-pro", want: false},
		{model: "gemini", want: false},
		{model: "gemini_1_5_flash", want: false},
		{model: "my-gemini-model", want: false},

		// "models/" prefix: the remainder is the name, with no further step.
		{model: "models/gemini-2.5-pro", want: true},
		{model: "models/Gemini-2.5-pro", want: false},
		{model: "models/google/gemini-2.5-flash", want: false},
		{model: "models/gemini/gemini-2.5-flash", want: false},

		// Vertex AI paths.
		{model: "projects/265104255505/locations/us-central1/publishers/google/models/gemini-2.0-flash-001", want: true},
		{model: "projects/p/locations/l/publishers/google/models/gemini-2.5-flash", want: true},
		{model: "projects/p/locations/l/publishers/google/models/gemini-2.5-flash\n", want: true},
		{model: "projects/265104255505/locations/us-central1/publishers/gemini/models/claude-3-sonnet", want: false},
		// Malformed "projects/" paths are not reduced.
		{model: "projects/a/b/locations/l/publishers/p/models/gemini-2.5-flash", want: false},
		{model: "projects/123/locations/us-central1/models/gemini-2.5-flash", want: false},
		{model: "projects/123/publishers/google/models/gemini-2.5-flash", want: false},
		{model: "projects/123/locations/us-central1/publishers/google/gemini-2.5-flash", want: false},
		{model: "projects/bad/gemini-2.5-flash", want: false},

		// Apigee paths: at most two segments after "apigee/" are cut.
		{model: "apigee/gemini-2.5-flash", want: true},
		{model: "apigee/v1/gemini-2.5-flash", want: true},
		{model: "apigee/vertex_ai/v1beta/gemini-2.5-flash", want: true},
		{model: "apigee/a/b/c/gemini-2.5-flash", want: false},
		{model: "apigee//gemini-2.5-flash", want: false},

		// Provider prefixes: only a last segment starting with gemini- counts.
		{model: "gemini/gemini-2.5-flash", want: true},
		{model: "vertex_ai/gemini-2.5-flash", want: true},
		{model: "openrouter/google/gemini-2.5-pro:online", want: true},
		{model: "openrouter/anthropic/claude-sonnet-4", want: false},
		// Otherwise the whole name is kept.
		{model: "gemini-2.5-flash/tuned", want: true},
	}
	for _, tt := range tests {
		if got := IsGeminiModelExact(tt.model); got != tt.want {
			t.Errorf("IsGeminiModelExact(%q) = %v, want %v", tt.model, got, tt.want)
		}
	}
}

// The want values are the results of adk-python v2.12.0's is_env_enabled
// (utils/env_utils.py) for the same values.
func TestIsGeminiModelIDCheckDisabled(t *testing.T) {
	const envVar = "ADK_DISABLE_GEMINI_MODEL_ID_CHECK"
	tests := []struct {
		value string
		want  bool
	}{
		{value: "true", want: true},
		{value: "TRUE", want: true},
		{value: "TrUe", want: true},
		{value: "1", want: true},
		{value: "false", want: false},
		{value: "FALSE", want: false},
		{value: "0", want: false},
		{value: "yes", want: false},
		{value: "on", want: false},
		{value: "", want: false},
		{value: " true", want: false},
		{value: "true ", want: false},
		{value: "t", want: false},
		{value: "01", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.value, func(t *testing.T) {
			t.Setenv(envVar, tt.value)
			if got := IsGeminiModelIDCheckDisabled(); got != tt.want {
				t.Errorf("IsGeminiModelIDCheckDisabled() with %s=%q = %v, want %v", envVar, tt.value, got, tt.want)
			}
		})
	}

	t.Run("unset", func(t *testing.T) {
		t.Setenv(envVar, "")
		if err := os.Unsetenv(envVar); err != nil {
			t.Fatalf("os.Unsetenv(%q): %v", envVar, err)
		}
		if got := IsGeminiModelIDCheckDisabled(); got {
			t.Errorf("IsGeminiModelIDCheckDisabled() with %s unset = true, want false", envVar)
		}
	})

	t.Run("read on every call", func(t *testing.T) {
		t.Setenv(envVar, "true")
		if got := IsGeminiModelIDCheckDisabled(); !got {
			t.Fatalf("IsGeminiModelIDCheckDisabled() with %s=true = false, want true", envVar)
		}
		t.Setenv(envVar, "false")
		if got := IsGeminiModelIDCheckDisabled(); got {
			t.Errorf("IsGeminiModelIDCheckDisabled() after changing %s to false = true, want false", envVar)
		}
	})
}
