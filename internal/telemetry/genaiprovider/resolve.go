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

package genaiprovider

import (
	"regexp"
	"strings"

	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"
)

// Resolve is the provider a model call reports: the one the model reports,
// else named by the model name, else by the provider segment of an Apigee
// path, else by the call's backend, else unknown.
func Resolve(reported, modelName string, backend genai.Backend) string {
	if reported != "" {
		return reported
	}
	name := bareModelName(modelName)
	if !strings.HasPrefix(name, "gemini-") {
		if anthropicModel.MatchString(name) {
			return "anthropic"
		}
		if provider, _, ok := strings.Cut(name, "/"); ok && provider != "" && !resourceCollections[strings.ToLower(provider)] {
			return strings.ToLower(provider)
		}
	}
	if rest, ok := strings.CutPrefix(modelName, "apigee/"); ok {
		if segment, _, ok := strings.Cut(rest, "/"); ok && apigeeProviders[segment] != "" {
			return apigeeProviders[segment]
		}
	}
	if backend == genai.BackendEnterprise {
		backend = genai.BackendVertexAI
	}
	if provider, ok := ForBackend(backend); ok {
		return provider
	}
	return "unknown"
}

// ForBackend is the provider that serves a Google backend, and whether it
// can be named. Naming the wrong provider is worse than naming none.
func ForBackend(backend genai.Backend) (string, bool) {
	switch backend {
	case genai.BackendVertexAI:
		return semconv.GenAISystemGCPVertexAI.Value.AsString(), true
	case genai.BackendGeminiAPI:
		return semconv.GenAISystemGCPGemini.Value.AsString(), true
	}
	return "", false
}

var (
	anthropicModel = regexp.MustCompile(`(?i)^claude[-.]`)

	// Vertex AI and Apigee model paths.
	modelPaths = []*regexp.Regexp{
		regexp.MustCompile(`^projects/[^/]+/locations/[^/]+/publishers/[^/]+/models/(.+)$`),
		regexp.MustCompile(`^apigee/(?:[^/]+/)?(?:[^/]+/)?(.+)$`),
	}

	apigeeProviders = map[string]string{
		"vertex_ai": "gcp.vertex_ai",
		"gemini":    "gcp.gemini",
		"openai":    "openai",
	}

	// Leading path segments that are not providers.
	resourceCollections = map[string]bool{
		"endpoints":   true,
		"locations":   true,
		"models":      true,
		"projects":    true,
		"publishers":  true,
		"tunedmodels": true,
	}
)

// bareModelName strips the Vertex AI and Apigee path prefixes from a model
// name. A malformed "projects/" path is left as is: [Resolve] names the same
// provider either way.
func bareModelName(model string) string {
	for _, p := range modelPaths {
		if m := p.FindStringSubmatch(model); m != nil {
			return m[1]
		}
	}
	if rest, ok := strings.CutPrefix(model, "models/"); ok {
		return rest
	}
	if i := strings.LastIndex(model, "/"); i >= 0 && strings.HasPrefix(model[i+1:], "gemini-") {
		return model[i+1:]
	}
	return model
}
