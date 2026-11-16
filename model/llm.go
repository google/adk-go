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
	"slices"

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

	// ClientFunctionCallIDs reports whether the model accepts the function
	// call IDs ADK generates (prefixed "adk-") on the function calls and
	// function responses it receives, and pairs tool calls with their
	// results by ID.
	//
	// When true, ADK keeps those IDs in the requests it sends to the model.
	// When false, ADK removes them from the request; IDs issued by the
	// model provider are always kept, and the session events keep every ID.
	//
	// A model that does not implement [CapabilitiesProvider] is treated as
	// false, so a type that only embeds [LLM] and implements no provider
	// gets the IDs removed. A custom [LLM] that pairs calls by ID implements
	// CapabilitiesProvider and sets this field. A model that wraps another
	// returns the inner model's Capabilities, copied, with only the fields
	// it changes overridden, so the wrapper keeps the IDs exactly when the
	// inner model does.
	ClientFunctionCallIDs bool
}

// CapabilitiesProvider is an optional interface that an [LLM] may implement to
// declare its [Capabilities].
//
// When an [LLM] implements CapabilitiesProvider, ADK uses the returned
// [Capabilities] directly instead of inferring support from the model name or
// backend variant: the returned value replaces the name-based default of
// OutputSchemaAndTools, even when it leaves that field false. When an [LLM]
// does not implement CapabilitiesProvider, ADK falls back to granting
// OutputSchemaAndTools only to Gemini models on the Vertex AI backend, and
// treats ClientFunctionCallIDs as false.
//
// A model that wraps another [LLM] implements CapabilitiesProvider by
// returning the inner model's Capabilities, copied, with only the fields the
// wrapper changes overridden, so the wrapper keeps ADK's function call IDs
// exactly when the inner model does. A type that only embeds [LLM] does not
// get the inner model's Capabilities method, because [LLM] does not declare
// it: ADK treats such a type as a model with no provider, removes ADK's
// function call IDs from its requests and applies the name-based default.
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

	// Tools maps a tool name to the tool that answers a function call of that
	// name. Tools add themselves while the LLM flow prepares the request,
	// usually through toolutils.PackTool. When preparation ends, the flow
	// deletes every entry whose Declaration method returns nil, so the model
	// and the callbacks see only declared tools. An entry added later, for
	// example by a before-model callback, stays in Tools; if it has no
	// declaration, a function call to its name gets the tool-not-found result.
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
	Content          *genai.Content          `json:"content,omitempty"`
	CitationMetadata *genai.CitationMetadata `json:"citationMetadata,omitempty"`
	// GroundingMetadata is the grounding metadata of the model's first
	// candidate. The Gemini model turns a candidate that has no content parts
	// and a finish reason other than STOP into an error response (ErrorCode
	// set), and such a response carries no grounding metadata.
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
	Interrupted             bool   `json:"interrupted,omitempty"`
	SessionResumptionHandle string `json:"sessionResumptionHandle,omitempty"`
	// LiveSessionID is the identifier the Live API gave the connection that
	// produced this response. It is opaque: compare it for equality, never
	// parse it. The empty string means the response came from no live
	// connection, or from a connection the server gave no identifier.
	//
	// In a live run, every response a live connection returns carries the
	// connection's identifier, and so does the model event built from it,
	// except an event for an input or output transcription or for a session
	// resumption update that has no content and neither a turn-complete nor
	// an interrupted signal, which carries none. After a reconnection,
	// responses carry the identifier of the new connection.
	//
	// The function response event built for the function calls of a live
	// model event carries that model event's identifier, whether it answers
	// one call or merges the responses of several, and carries "" when the
	// model event has none. A run without a live connection builds its
	// function response events without one, whatever the model returned.
	//
	// A response that an after-model callback replaced does not carry the id
	// unless the replacement sets it. A plugin's replacement event replaces
	// the identifier only when it sets one; otherwise the original's stays.
	//
	// The in-memory, Redis and database session services and the REST event
	// keep the field. The Vertex AI session service keeps it in the event's
	// raw_event, since its SessionEvent has no column for it.
	LiveSessionID string             `json:"liveSessionId,omitempty"`
	ErrorCode     string             `json:"errorCode,omitempty"`
	ErrorMessage  string             `json:"errorMessage,omitempty"`
	FinishReason  genai.FinishReason `json:"finishReason,omitempty"`
	AvgLogprobs   float64            `json:"avgLogprobs,omitempty"`
	// CacheMetadata is the context cache state of the request that produced
	// this response, set when the request had a CacheConfig. Only complete
	// responses carry it, not partial streaming chunks.
	CacheMetadata *CacheMetadata `json:"cacheMetadata,omitempty"`
}

// FunctionCalls returns the FunctionCall of each part of r.Content that has
// one, in part order. It returns nil when r is nil, r.Content is nil, or no
// part holds a function call. Nil entries in r.Content.Parts are skipped.
// The returned pointers are the ones stored in r.Content; they are not copies.
func (r *LLMResponse) FunctionCalls() []*genai.FunctionCall {
	if r == nil || r.Content == nil {
		return nil
	}
	var calls []*genai.FunctionCall
	for _, p := range r.Content.Parts {
		if p != nil && p.FunctionCall != nil {
			calls = append(calls, p.FunctionCall)
		}
	}
	return calls
}

// FunctionResponses returns the FunctionResponse of each part of r.Content
// that has one, in part order, under the same rules as FunctionCalls.
func (r *LLMResponse) FunctionResponses() []*genai.FunctionResponse {
	if r == nil || r.Content == nil {
		return nil
	}
	var responses []*genai.FunctionResponse
	for _, p := range r.Content.Parts {
		if p != nil && p.FunctionResponse != nil {
			responses = append(responses, p.FunctionResponse)
		}
	}
	return responses
}

// HasTrailingCodeExecutionResult reports whether the last non-nil part of
// r.Content holds a CodeExecutionResult. It returns false when r is nil,
// r.Content is nil, or no non-nil part exists.
func (r *LLMResponse) HasTrailingCodeExecutionResult() bool {
	if r == nil || r.Content == nil {
		return false
	}
	// A nil entry after a code execution result does not hide it: the
	// trailing part is the last non-nil one.
	for _, p := range slices.Backward(r.Content.Parts) {
		if p != nil {
			return p.CodeExecutionResult != nil
		}
	}
	return false
}
