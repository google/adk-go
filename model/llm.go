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

// LLM provides access to an underlying language model.
type LLM interface {
	Name() string
	// Capabilities reports the resolved capabilities of this model instance.
	// Implementations must be safe for concurrent use and must not mutate the
	// model or have observable side effects.
	Capabilities() Capabilities
	GenerateContent(ctx context.Context, req *LLMRequest, stream bool) iter.Seq2[*LLMResponse, error]
}

// LiveConnector is an optional capability of an [LLM] that serves bidirectional
// live sessions.
type LiveConnector interface {
	// Connect opens a session using req.Config and req.LiveConnectConfig.
	// The flow sends req.Contents separately with SendHistory. Implementations
	// must treat the request and its fields as read-only. The context may be
	// canceled during setup if the live session closes. On success Connect
	// returns a non-nil connection, which the caller closes.
	Connect(ctx context.Context, req *LLMRequest) (LiveConnection, error)
}

// LiveConnection is an open live session with a model.
//
// The flow serializes calls to the Send methods and may iterate Receive
// concurrently with them. Close may run concurrently with either and must
// unblock any pending sends or receives.
type LiveConnection interface {
	// SendHistory sends earlier turns immediately after connecting. The model
	// should reply if the final turn is the user's.
	SendHistory(ctx context.Context, history []*genai.Content) error
	// SendContent sends a user turn or answers the model's tool calls. Tool
	// response content must contain only function response parts.
	SendContent(ctx context.Context, content *genai.Content) error
	// SendRealtime accepts live input such as *genai.Blob, *genai.ActivityStart,
	// or *genai.ActivityEnd. It returns an error for unsupported input types.
	SendRealtime(ctx context.Context, input any) error
	// Receive is called once per connection. Returning without an error ends
	// the session cleanly. A resumable transport error such as io.EOF allows
	// the flow to reconnect, using the latest SessionResumptionHandle if set.
	// The flow may retain yielded responses, so implementations must leave
	// their contents intact after yielding.
	Receive(ctx context.Context) iter.Seq2[*LLMResponse, error]
	Close() error
}

// Capabilities holds the resolved capabilities of an [LLM] instance.
//
// Each field reports whether the model supports a specific optional feature.
// The zero value (Capabilities{}) indicates that none of the optional
// capabilities are supported. Capabilities is a value type and is safe for
// concurrent use.
type Capabilities struct {
	// AcceptsOutputSchemaAndTools reports whether the model accepts a native
	// output schema (GenerateContentConfig.ResponseSchema) together with tools
	// in the same request.
	//
	// When false and an agent configures both an output schema and tools (or
	// toolsets), ADK omits ResponseSchema and ResponseMIMEType from the
	// request config and instead instructs the model to return its structured
	// answer through a synthetic function tool. When an agent has no tools,
	// ADK still sends the output schema natively.
	AcceptsOutputSchemaAndTools bool
}

// LLMRequest is the raw LLM request.
type LLMRequest struct {
	Model    string
	Contents []*genai.Content
	Config   *genai.GenerateContentConfig

	// LiveConnectConfig holds settings from the live invocation. It is nil for
	// non-live requests.
	LiveConnectConfig *genai.LiveConnectConfig `json:",omitempty"`

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
