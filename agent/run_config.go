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

package agent

import "google.golang.org/genai"

// StreamingMode defines the streaming mode for agent execution.
type StreamingMode string

const (
	// StreamingModeNone indicates no streaming.
	StreamingModeNone StreamingMode = "none"
	// StreamingModeSSE enables server-sent events streaming, one-way, where
	// LLM response parts are streamed immediately as they are generated.
	StreamingModeSSE StreamingMode = "sse"
	// StreamingModeBidi enables bidirectional streaming.
	StreamingModeBidi StreamingMode = "bidi"
)

// RunConfig controls runtime behavior of an agent.
type RunConfig struct {
	// StreamingMode defines the streaming mode for an agent.
	StreamingMode StreamingMode
	// If true, ADK runner will save each part of the user input that is a blob
	// (e.g., images, files) as an artifact.
	SaveInputBlobsAsArtifacts bool
	// Optional. The requested modalities of the response. Represents the set of
	// modalities that the model can return. Defaults to AUDIO if not specified.
	ResponseModalities []genai.Modality `json:"responseModalities,omitempty"`

	// Optional. The speech generation configuration.
	SpeechConfig *genai.SpeechConfig `json:"speechConfig,omitempty"`
}
