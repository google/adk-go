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
	"fmt"
	"testing"

	"google.golang.org/genai"
)

// Cases from https://github.com/google/adk-python/blob/ed030cbf431ef6c12a2b3a75dcc7aff08378cb82/tests/unittests/telemetry/test_spans.py#L2208-L2302
func TestProviderName(t *testing.T) {
	for _, tc := range []struct {
		model   string
		backend genai.Backend
		want    string
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
		{"apigee/openai/gpt-4o", genai.BackendUnspecified, "openai"},
		{"apigee/vertex_ai/gemini-2.5-flash", genai.BackendUnspecified, "gcp.vertex_ai"},
		{"apigee/vertex_ai/gemini-2.5-flash", genai.BackendGeminiAPI, "gcp.vertex_ai"},
		{"apigee/gemini/gemini-2.5-flash", genai.BackendUnspecified, "gcp.gemini"},
		{"apigee/vertex_ai/v1beta/gemini-2.5-flash", genai.BackendUnspecified, "gcp.vertex_ai"},
		{"apigee/vertex_ai/claude-sonnet-4-5", genai.BackendUnspecified, "anthropic"},
		{"apigee/v1/gemini-2.5-flash", genai.BackendVertexAI, "gcp.vertex_ai"},
		{"gemini-2.0-flash", genai.BackendEnterprise, "gcp.vertex_ai"},
		{"gemini-2.0-flash", genai.Backend(99), "unknown"},
		{"some-model", genai.BackendVertexAI, "gcp.vertex_ai"},
		{"/foo", genai.BackendVertexAI, "gcp.vertex_ai"},
		{"gemini-2.0-flash", genai.BackendUnspecified, "unknown"},
		{"some-model", genai.BackendUnspecified, "unknown"},
		{"", genai.BackendUnspecified, "unknown"},
		{"apigee/gemini-2.5-flash", genai.BackendUnspecified, "unknown"},
		{"apigee/v1/gemini-2.5-flash", genai.BackendUnspecified, "unknown"},
	} {
		t.Run(fmt.Sprintf("%s/%d", tc.model, tc.backend), func(t *testing.T) {
			if got := providerName("", tc.model, tc.backend).Value.AsString(); got != tc.want {
				t.Errorf("providerName(%q, %v) = %q, want %q", tc.model, tc.backend, got, tc.want)
			}
		})
	}
}

func TestProviderName_Reported(t *testing.T) {
	for _, tc := range []struct {
		model   string
		backend genai.Backend
	}{
		{"claude-sonnet-4-5", genai.BackendUnspecified},
		{"gemini-2.0-flash", genai.BackendGeminiAPI},
		{"apigee/openai/gpt-4o", genai.BackendUnspecified},
	} {
		if got := providerName("gcp.vertex_ai", tc.model, tc.backend).Value.AsString(); got != "gcp.vertex_ai" {
			t.Errorf("providerName(%q, %q, %v) = %q, want the reported gcp.vertex_ai", "gcp.vertex_ai", tc.model, tc.backend, got)
		}
	}
}
