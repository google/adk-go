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

package gemini

import (
	"context"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/version"
	"google.golang.org/adk/v2/model"
)

// registeredPatterns are the model names this package registers with
// model.Register. They are adk-python's Gemini.supported_models
// (models/google_llm.py), anchored because Python matches the whole name and
// model.NewLLM matches any part of it. Matching is case-sensitive, as in
// Python.
var registeredPatterns = []string{
	`^gemini-.*$`,
	// Gemma 4 and later run natively as Gemini models.
	`^gemma-4.*$`,
	`^model-optimizer-.*$`,
	// A fine-tuned model on a Vertex AI endpoint.
	`^projects/.+/locations/.+/endpoints/.+$`,
	// The full Vertex AI resource name of a Gemini model.
	`^projects/.+/locations/.+/publishers/google/models/gemini.+$`,
}

func init() {
	for _, pattern := range registeredPatterns {
		model.Register(pattern, newRegisteredModel)
	}
}

// newRegisteredModel is the factory registered for registeredPatterns. Unlike
// NewModel it needs no credentials: the client is created from the
// environment on first use, as adk-python's Gemini.api_client is, so resolving
// a name always succeeds and a missing credential is reported by each model
// call until the client can be created.
func newRegisteredModel(ctx context.Context, name string) (model.LLM, error) {
	// The client may be created long after ctx has ended; it only needs the
	// context's values.
	clientCtx := context.WithoutCancel(ctx)
	return &geminiModel{
		name:               name,
		versionHeaderValue: version.TrackingHeader(),
		newClient: func() (*genai.Client, error) {
			return newGenAIClient(clientCtx, nil)
		},
	}, nil
}

// IsGeminiConnector reports whether m is ADK's Gemini connector, whether it
// was built by NewModel or resolved through the model registry.
//
// Caller provides: any model.LLM, nil included.
// Guarantees: true only for a model built by this package; false for nil and
// for every other model.
// On failure: N/A.
// Zero value: N/A.
// Concurrency: safe for concurrent use.
func IsGeminiConnector(m model.LLM) bool {
	g, ok := m.(*geminiModel)
	return ok && g != nil
}
