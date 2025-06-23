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

package adk

import (
	"context"
	"encoding/json"
	"iter"
	"strings"

	"google.golang.org/genai"
)

// Model is the LLM Model.
type Model interface {
	Name() string
	GenerateContent(ctx context.Context, req *LLMRequest, stream bool) LLMResponseStream
}

// LLMRequest is the input to LLMModel's generate functions.
// This allows passing in tools, output schema, and system instructions
// to the model.
type LLMRequest struct {
	Model          Model
	Contents       []*genai.Content
	GenerateConfig *genai.GenerateContentConfig

	Tools map[string]Tool

	// TODO: Can't we use genai's types?

	// Corresponds to adk-python's LLMRequest
	// TODO(jbd): Add other fields.
}

func (r *LLMRequest) AppendInstructions(instructions ...string) {
	if len(instructions) == 0 {
		return
	}
	inst := strings.Join(instructions, "\n\n")
	if current := r.GenerateConfig.SystemInstruction; current != nil && len(current.Parts) > 0 && current.Parts[0].Text != "" {
		r.GenerateConfig.SystemInstruction = genai.NewContentFromText(current.Parts[0].Text+"\n\n"+inst, "")
	} else {
		r.GenerateConfig.SystemInstruction = genai.NewContentFromText(inst, "")
	}
}

func (r *LLMRequest) AppendTools(tools ...Tool) {
	panic("unimplemented")
}

func (r *LLMRequest) String() string {
	b, _ := json.MarshalIndent(r, "", " ")
	return string(b)
}

// LLMResponseStream is the output of LLMModel's generate functions.
type LLMResponseStream iter.Seq2[*LLMResponse, error]

// LLMResponse provides the first candidate response from the model if available.
type LLMResponse struct {
	Content           *genai.Content           `json:"content,omitempty"`
	GroundingMetadata *genai.GroundingMetadata `json:"grounding_metadata,omitempty"`
	// Partial indicates whether the content is part of a unfinished content stream.
	// Only used for streaming mode and when the content is plain text.
	Partial bool `json:"partial,omitempty"`
	// Indicates whether the response from the model is complete.
	// Only used for streaming mode.
	TurnComplete bool `json:"turn_complete,omitempty"`
	// Flag indicating that LLM was interrupted when generating the content.
	// Usually it is due to user interruption during a bidi streaming.
	Interrupted  bool   `json:"interrupted,omitempty"`
	ErrorCode    int    `json:"error_code,omitempty"`
	ErrorMessage string `json:"error_message,omitempty"`
}

func (r *LLMResponse) String() string {
	b, _ := json.MarshalIndent(r, "", " ")
	return string(b)
}
