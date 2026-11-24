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
//
// An implementation may also implement [OutputSchemaTool]. Returning
// true asks agents to use set_model_response for structured output, including
// when no other tools are configured. Task-mode agents still use finish_task.
// Implementations without this method retain the model-name-based defaults.
type LLM interface {
	Name() string
	GenerateContent(ctx context.Context, req *LLMRequest, stream bool) iter.Seq2[*LLMResponse, error]
}

// OutputSchemaTool is implemented by models that deliver a declared output
// schema as a set_model_response function tool instead of a native response
// format; see ClientConfig.UseOutputSchemaTool in model/openaimodel. Wrappers
// holding an inner LLM must forward UseOutputSchemaTool, otherwise the agent
// silently falls back to the model-name-based defaults and native
// response_format.
type OutputSchemaTool interface {
	UseOutputSchemaTool() bool
}

// LLMRequest is the raw LLM request.
type LLMRequest struct {
	Model    string
	Contents []*genai.Content
	Config   *genai.GenerateContentConfig

	Tools map[string]any `json:"-"`
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
}
