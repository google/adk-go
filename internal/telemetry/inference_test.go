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

package telemetry

import (
	"testing"

	"google.golang.org/genai"
)

// Cases from https://github.com/google/adk-python/blob/ed030cbf431ef6c12a2b3a75dcc7aff08378cb82/tests/unittests/telemetry/test_spans.py#L2208-L2302
func TestProviderName(t *testing.T) {
	for _, tc := range []struct {
		model   string
		backend genai.Backend
		want    any
	}{
		{"claude-sonnet-4-5", genai.BackendVertexAI, "anthropic"},
		{"claude-3-5-haiku-latest", genai.BackendUnspecified, "anthropic"},
		{"Claude-3-Opus", genai.BackendUnspecified, "anthropic"},
		{"anthropic/claude-sonnet-4-5", genai.BackendUnspecified, "anthropic"},
		{"projects/p/locations/l/publishers/anthropic/models/claude-sonnet-4-5", genai.BackendVertexAI, "anthropic"},
		{"openai/gpt-4o", genai.BackendUnspecified, "openai"},
		{"OpenAI/gpt-4o", genai.BackendUnspecified, "openai"},
		{"projects/p/locations/l/publishers/meta/models/llama-3", genai.BackendVertexAI, "gcp.vertex_ai"},
		{"projects/p/locations/l/endpoints/123456", genai.BackendVertexAI, "gcp.vertex_ai"},
		{"tunedModels/my-tuned-model", genai.BackendGeminiAPI, "gcp.gemini"},
		{"gemini-2.0-flash", genai.BackendVertexAI, "gcp.vertex_ai"},
		{"gemini/gemini-2.0-flash", genai.BackendGeminiAPI, "gcp.gemini"},
		{"models/gemini-2.0-flash", genai.BackendGeminiAPI, "gcp.gemini"},
		{"gemini-2.0-flash/001", genai.BackendVertexAI, "gcp.vertex_ai"},
		{"models/claude-3-haiku", genai.BackendGeminiAPI, "anthropic"},
		{"apigee/openai/gpt-4o", genai.BackendUnspecified, nil},
		{"some-model", genai.BackendVertexAI, "gcp.vertex_ai"},
		{"/foo", genai.BackendVertexAI, "gcp.vertex_ai"},
		// ADK Python guesses from the environment here.
		{"gemini-2.0-flash", genai.BackendUnspecified, nil},
		{"some-model", genai.BackendUnspecified, nil},
		{"", genai.BackendUnspecified, nil},
	} {
		t.Run(tc.model+"/"+tc.backend.String(), func(t *testing.T) {
			var got any
			if kv, ok := providerName(tc.model, tc.backend); ok {
				got = kv.Value.AsString()
			}
			if got != tc.want {
				t.Errorf("providerName(%q, %v) = %v, want %v", tc.model, tc.backend, got, tc.want)
			}
		})
	}
}
