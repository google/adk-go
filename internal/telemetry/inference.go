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
	"regexp"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/genai"
)

// genAIProviderName supersedes gen_ai.system in semantic conventions >1.36.
var genAIProviderName = attribute.Key("gen_ai.provider.name")

// providerName is the provider a model call's span, log record and metrics
// report: named by the model, else by the backend, else by the backend the
// environment selects. adk-python skips the client's backend and guesses from
// the environment; Go knows the backend, so it uses it.
func providerName(modelName string, backend genai.Backend) attribute.KeyValue {
	name := bareModelName(modelName)
	if !strings.HasPrefix(name, "gemini-") {
		if anthropicModel.MatchString(name) {
			return genAIProviderName.String("anthropic")
		}
		if provider, _, ok := strings.Cut(name, "/"); ok && provider != "" && !resourceCollections[strings.ToLower(provider)] {
			return genAIProviderName.String(strings.ToLower(provider))
		}
	}
	if backend == genai.BackendUnspecified {
		backend = environmentBackend()
	}
	sys, _ := GenAISystemAttr(backend)
	return genAIProviderName.String(sys.Value.AsString())
}

// environmentBackend returns the backend the environment selects for a client
// that names none.
func environmentBackend() genai.Backend {
	for _, name := range []string{"GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_USE_VERTEXAI"} {
		if v, ok := os.LookupEnv(name); ok {
			if v := strings.ToLower(v); v == "1" || v == "true" {
				return genai.BackendVertexAI
			}
			return genai.BackendGeminiAPI
		}
	}
	return genai.BackendGeminiAPI
}

var (
	anthropicModel = regexp.MustCompile(`(?i)^claude[-.]`)

	// Vertex AI and Apigee model paths.
	modelPaths = []*regexp.Regexp{
		regexp.MustCompile(`^projects/[^/]+/locations/[^/]+/publishers/[^/]+/models/(.+)$`),
		regexp.MustCompile(`^apigee/(?:[^/]+/)?(?:[^/]+/)?(.+)$`),
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

// bareModelName mirrors ADK Python: https://github.com/google/adk-python/blob/ed030cbf431ef6c12a2b3a75dcc7aff08378cb82/src/google/adk/utils/model_name_utils.py#L49-L91
// Its malformed "projects/" path rule is left out: [providerName] names the same provider either way.
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
