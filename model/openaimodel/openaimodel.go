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

package openaimodel

import (
	"context"
	"fmt"
	"net/http"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openaimodel/internal/completions"
	"google.golang.org/adk/v2/model/openaimodel/internal/responses"
)

// API selects the OpenAI HTTP API a model talks to. The zero value selects
// [APIResponses].
type API string

const (
	// APIResponses is OpenAI's Responses API, POST /v1/responses.
	APIResponses API = "responses"

	// APIChatCompletions is OpenAI's Chat Completions API, POST
	// /v1/chat/completions, which is the surface most OpenAI-compatible
	// third-party providers implement.
	APIChatCompletions API = "chat_completions"
)

// ClientConfig configures the OpenAI client. Mirrors model/gemini, which takes
// *genai.ClientConfig. Empty APIKey/BaseURL fall back to the OPENAI_API_KEY /
// OPENAI_BASE_URL env vars (handled by openai-go's default options).
type ClientConfig struct {
	APIKey     string
	BaseURL    string       // for OpenAI-compatible endpoints
	HTTPClient *http.Client // optional; e.g. for tests

	// Options is an escape hatch for advanced openai-go request options,
	// appended after the options derived from the fields above.
	Options []option.RequestOption

	// API selects which OpenAI HTTP API to talk to. The zero value is
	// [APIResponses], so a configuration written before this field existed
	// keeps the endpoint it already used.
	API API
}

// FinishMessageKey is the [model.LLMResponse.CustomMetadata] key under which a
// turn that ended badly but still carries an answer reports the provider's own
// account of why — a content filter's "content_filter", an incomplete reason
// this package does not map, or a server error message. Its FinishReason says
// the turn was cut short; this says what the provider called it.
//
// A turn left with nothing to read reports the same wording in ErrorCode and
// ErrorMessage instead, which callers treat as a failed turn.
//
// It reaches in-process callers, session storage and A2A metadata, but not
// REST: server/adkrest maps events field by field and omits CustomMetadata, so
// an ADK Web consumer sees the FinishReason alone.
const FinishMessageKey = "openai_finish_message"

// NewModel constructs a model talking to the API named by cfg.
// The context is unused but kept for signature parity with other model constructors (e.g., gemini.NewModel).
func NewModel(_ context.Context, modelName string, cfg *ClientConfig) (model.LLM, error) {
	if modelName == "" {
		return nil, ErrModelNameRequired
	}
	if cfg == nil {
		cfg = &ClientConfig{}
	}
	var opts []option.RequestOption
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
	}
	opts = append(opts, cfg.Options...)
	client := openai.NewClient(opts...)
	switch cfg.API {
	case "", APIResponses:
		return responses.New(&client, modelName), nil
	case APIChatCompletions:
		return completions.New(&client, modelName), nil
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedAPI, cfg.API)
	}
}
