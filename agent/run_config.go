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

import (
	"time"

	"google.golang.org/genai"
)

// StreamingMode defines the streaming mode for agent execution.
type StreamingMode string

const (
	// StreamingModeNone indicates no streaming.
	StreamingModeNone StreamingMode = "none"
	// StreamingModeSSE enables server-sent events streaming, one-way, where
	// LLM response parts are streamed immediately as they are generated.
	StreamingModeSSE StreamingMode = "sse"
	// StreamingModeBidi enables bidirectional live streaming for real-time
	// voice/audio interactions.
	StreamingModeBidi StreamingMode = "bidi"
)

// RunConfig controls runtime behavior of an agent.
type RunConfig struct {
	// StreamingMode defines the streaming mode for an agent.
	StreamingMode StreamingMode
	// If true, ADK runner will save each part of the user input that is a blob
	// (e.g., images, files) as an artifact.
	SaveInputBlobsAsArtifacts bool
	// Model overrides the model name for this run. When non-empty, the live
	// connect call uses this model instead of the agent's base model. This
	// allows per-session model selection (e.g. switching between flash and pro
	// voice models) without rebuilding the agent tree.
	Model string
	// Live-specific configuration
	ResponseModalities       []genai.Modality
	SpeechConfig             *genai.SpeechConfig
	InputAudioTranscription  bool
	OutputAudioTranscription bool
	ToolCoalesceWindow       time.Duration // default 150ms if zero
	LiveBufferSize           int           // default 100 if zero
	// SessionResumption configures session resumption for live sessions,
	// allowing reconnection with preserved state.
	SessionResumption *genai.SessionResumptionConfig
	// Generation parameters — applied to the live session config when set.
	ThinkingConfig  *genai.ThinkingConfig
	Temperature     *float32
	TopP            *float32
	TopK            *float32
	MaxOutputTokens *int32

	// Live session capabilities (require genai v1.51.0).
	RealtimeInputConfig      *genai.RealtimeInputConfig
	Proactivity              *genai.ProactivityConfig
	EnableAffectiveDialog    *bool
	ContextWindowCompression *genai.ContextWindowCompressionConfig
}
