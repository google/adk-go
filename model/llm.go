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

// Package model defines the interfaces and data structures for interacting with LLMs.
package model

import (
	"context"
	"iter"

	"google.golang.org/genai"
)

// LLM provides the access to the underlying LLM.
type LLM interface {
	Name() string
	GenerateContent(ctx context.Context, req *LLMRequest, stream bool) iter.Seq2[*LLMResponse, error]
}

// Capabilities holds the resolved capabilities of an [LLM] instance.
//
// Each field reports whether the model supports a specific optional feature.
// The zero value (Capabilities{}) indicates that none of the optional
// capabilities are supported. Capabilities is a value type and is safe for
// concurrent use.
type Capabilities struct {
	// OutputSchemaAndTools reports whether the model can accept a native
	// output schema (GenerateContentConfig.ResponseSchema) and tools in the
	// same request.
	//
	// When false and an agent has both an OutputSchema and tools (or
	// toolsets), ADK omits ResponseSchema and ResponseMIMEType from the
	// request config and uses the set_model_response tool workaround instead.
	OutputSchemaAndTools bool
}

// CapabilitiesProvider is an optional interface that an [LLM] may implement to
// declare its [Capabilities].
//
// When an [LLM] implements CapabilitiesProvider, ADK uses the returned
// [Capabilities] directly instead of inferring support from the model name or
// backend variant. When an [LLM] does not implement CapabilitiesProvider, ADK
// falls back to granting OutputSchemaAndTools only to Gemini models on the
// Vertex AI backend.
//
// Implementations of Capabilities must be safe for concurrent use and must not
// mutate the model or have observable side effects.
type CapabilitiesProvider interface {
	Capabilities() Capabilities
}

// LLMRequest is the raw LLM request.
type LLMRequest struct {
	Model    string
	Contents []*genai.Content
	Config   *genai.GenerateContentConfig

	Tools map[string]any `json:"-"`

	// CacheConfig turns on context caching for this request when non-nil. The
	// flow sets it from the runner's ContextCache config, and a model that
	// does not support caching ignores it. See [ContextCacheConfig].
	CacheConfig *ContextCacheConfig
	// CacheMetadata is the cache state recorded on the agent's latest model
	// response, which the model checks before reusing or replacing that
	// cache. Nil on the agent's first request, or when caching is off.
	CacheMetadata *CacheMetadata
	// PreviousPromptTokenCount is the prompt token count the provider
	// reported for the agent's previous request, which sizes a new cache. Zero
	// when no earlier count is known, and no cache is created then.
	PreviousPromptTokenCount int32
}

// LLMResponse is the raw LLM response.
// It provides the first candidate response from the model if available.
type LLMResponse struct {
	Content             *genai.Content                              `json:"content,omitempty"`
	CitationMetadata    *genai.CitationMetadata                     `json:"citationMetadata,omitempty"`
	GroundingMetadata   *genai.GroundingMetadata                    `json:"groundingMetadata,omitempty"`
	UsageMetadata       *genai.GenerateContentResponseUsageMetadata `json:"usageMetadata,omitempty"`
	CustomMetadata      map[string]any                              `json:"customMetadata,omitempty"`
	LogprobsResult      *genai.LogprobsResult                       `json:"logprobsResult,omitempty"`
	InputTranscription  *genai.Transcription                        `json:"inputTranscription,omitempty"`
	OutputTranscription *genai.Transcription                        `json:"outputTranscription,omitempty"`
	ModelVersion        string                                      `json:"modelVersion,omitempty"`
	// Partial indicates whether the content is part of a unfinished content stream.
	// Only used for streaming mode and when the content is plain text.
	// The Runner fully processes only the final non-partial event, partial
	// events are simply forwarded downstream (eg. to UI for display).
	Partial bool `json:"partial,omitempty"`
	// PartialSet reports whether Partial was set explicitly on this response.
	//
	// adk-python has no such field: its LlmResponse.partial is an
	// Optional[bool], so an unset partial (None) and an explicit False differ
	// there. Partial here is a plain bool whose zero value is false, so a
	// response that leaves Partial unset cannot otherwise be told apart from
	// one that sets Partial: false. An AfterModelCallback replacement with
	// Partial: false and PartialSet: true for a partial model response
	// therefore produces a non-partial event, instead of inheriting
	// Partial: true from the response it replaces.
	//
	// When Partial is true, the response is always treated as explicitly
	// partial regardless of PartialSet. The zero value (Partial: false,
	// PartialSet: false) means Partial was left unset. PartialSet is never
	// written to JSON.
	PartialSet bool `json:"-"`
	// Indicates whether the response from the model is complete.
	// Only used for streaming mode.
	TurnComplete bool `json:"turnComplete,omitempty"`
	// Flag indicating that LLM was interrupted when generating the content.
	// Usually it is due to user interruption during a bidi streaming.
	Interrupted             bool               `json:"interrupted,omitempty"`
	SessionResumptionHandle string             `json:"sessionResumptionHandle,omitempty"`
	ErrorCode               string             `json:"errorCode,omitempty"`
	ErrorMessage            string             `json:"errorMessage,omitempty"`
	FinishReason            genai.FinishReason `json:"finishReason,omitempty"`
	AvgLogprobs             float64            `json:"avgLogprobs,omitempty"`
	// CacheMetadata is the context cache state of the request that produced
	// this response, set when the request had a CacheConfig. Only complete
	// responses carry it, not partial streaming chunks.
	CacheMetadata *CacheMetadata `json:"cacheMetadata,omitempty"`
}
