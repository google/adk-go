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
	"os"
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
		{"apigee/openai/gpt-4o", genai.BackendUnspecified, "gcp.gemini"},
		{"some-model", genai.BackendVertexAI, "gcp.vertex_ai"},
		{"/foo", genai.BackendVertexAI, "gcp.vertex_ai"},
		// Guessed from the environment, which names no backend.
		{"gemini-2.0-flash", genai.BackendUnspecified, "gcp.gemini"},
		{"some-model", genai.BackendUnspecified, "gcp.gemini"},
		{"", genai.BackendUnspecified, "gcp.gemini"},
	} {
		t.Run(tc.model+"/"+tc.backend.String(), func(t *testing.T) {
			unsetEnvForTesting(t, "GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_USE_VERTEXAI")
			if got := providerName(tc.model, tc.backend).Value.AsString(); got != tc.want {
				t.Errorf("providerName(%q, %v) = %q, want %q", tc.model, tc.backend, got, tc.want)
			}
		})
	}
}

func TestProviderNameFromEnvironment(t *testing.T) {
	const (
		enterprise = "GOOGLE_GENAI_USE_ENTERPRISE"
		vertexAI   = "GOOGLE_GENAI_USE_VERTEXAI"
	)
	for _, tc := range []struct {
		name    string
		model   string
		backend genai.Backend
		// env holds the variables to set; the others are unset.
		env  map[string]string
		want string
	}{
		{"claude", "claude-sonnet-4-5", genai.BackendUnspecified, map[string]string{enterprise: "true"}, "anthropic"},
		{"provider prefix", "openai/gpt-4o", genai.BackendUnspecified, map[string]string{enterprise: "true"}, "openai"},
		{"known backend", "gemini-2.0-flash", genai.BackendGeminiAPI, map[string]string{enterprise: "true"}, "gcp.gemini"},
		{"nothing set", "gemini-2.0-flash", genai.BackendUnspecified, nil, "gcp.gemini"},
		{"enterprise true", "some-model", genai.BackendUnspecified, map[string]string{enterprise: "true"}, "gcp.vertex_ai"},
		{"enterprise TRUE", "some-model", genai.BackendUnspecified, map[string]string{enterprise: "TRUE"}, "gcp.vertex_ai"},
		{"enterprise 1", "", genai.BackendUnspecified, map[string]string{enterprise: "1"}, "gcp.vertex_ai"},
		{"enterprise not trimmed", "some-model", genai.BackendUnspecified, map[string]string{enterprise: " true"}, "gcp.gemini"},
		{"enterprise yes", "some-model", genai.BackendUnspecified, map[string]string{enterprise: "yes"}, "gcp.gemini"},
		{"enterprise empty wins", "some-model", genai.BackendUnspecified, map[string]string{enterprise: "", vertexAI: "true"}, "gcp.gemini"},
		{"enterprise false wins", "some-model", genai.BackendUnspecified, map[string]string{enterprise: "false", vertexAI: "true"}, "gcp.gemini"},
		{"vertex ai true", "some-model", genai.BackendUnspecified, map[string]string{vertexAI: "True"}, "gcp.vertex_ai"},
		{"vertex ai 0", "some-model", genai.BackendUnspecified, map[string]string{vertexAI: "0"}, "gcp.gemini"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			unsetEnvForTesting(t, enterprise, vertexAI)
			for name, v := range tc.env {
				t.Setenv(name, v)
			}
			got := providerName(tc.model, tc.backend)
			if got.Key != genAIProviderName || got.Value.AsString() != tc.want {
				t.Errorf("providerName(%q, %v) = %v=%q, want %s=%q", tc.model, tc.backend, got.Key, got.Value.AsString(), genAIProviderName, tc.want)
			}
		})
	}
}

// unsetEnvForTesting unsets the given environment variables until the test
// ends.
func unsetEnvForTesting(t *testing.T, names ...string) {
	t.Helper()
	for _, name := range names {
		t.Setenv(name, "")
		if err := os.Unsetenv(name); err != nil {
			t.Fatal(err)
		}
	}
}
