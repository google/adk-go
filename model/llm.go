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
	Content           *genai.Content
	CitationMetadata  *genai.CitationMetadata
	GroundingMetadata *genai.GroundingMetadata
	UsageMetadata     *genai.GenerateContentResponseUsageMetadata
	CustomMetadata    map[string]any
	LogprobsResult    *genai.LogprobsResult
	// Partial indicates whether the content is part of a unfinished content stream.
	// Only used for streaming mode and when the content is plain text.
	// The Runner fully processes only the final non-partial event, partial
	// events are simply forwarded downstream (eg. to UI for display).
	Partial bool
	// Indicates whether the response from the model is complete.
	// Only used for streaming mode.
	TurnComplete bool
	// Flag indicating that LLM was interrupted when generating the content.
	// Usually it is due to user interruption during a bidi streaming.
	Interrupted  bool
	ErrorCode    string
	ErrorMessage string
	FinishReason genai.FinishReason
	AvgLogprobs  float64
}
