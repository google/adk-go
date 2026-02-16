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
	ResponseModalities []genai.Modality
	// Optional. Configures the realtime input behavior in BidiGenerateContent.
	RealtimeInputConfig *genai.RealtimeInputConfig
	// Optional. The speech generation configuration.
	SpeechConfig *genai.SpeechConfig
	// Optional. The transcription of the input aligns with the input audio language.
	InputAudioTranscription *genai.AudioTranscriptionConfig
	// Optional. The transcription of the output aligns with the language code
	// specified for the output audio.
	OutputAudioTranscription *genai.AudioTranscriptionConfig
	// Enable automatic reconnection and resumption of live sessions in case of transient network issues or interruptions.
	SessionResumption *genai.SessionResumptionConfig
	// Optional. Configures context window compression mechanism.
	// If included, server will compress context window to fit into given length.
	ContextWindowCompression *genai.ContextWindowCompressionConfig
	// Optional. Configures the proactivity of the model. This allows the model to respond
	// proactively to the input and to ignore irrelevant input.
	Proactivity *genai.ProactivityConfig
	// Optional. Configures the explicit VAD signal. If enabled, the client will send
	// vad_signal to indicate the start and end of speech. This allows the server
	// to process the audio more efficiently.
	ExplicitVADSignal bool
	// This parameter controls whether audio and video streams are persisted to ADK's session and
	// artifact services for debugging, compliance, and quality assurance purposes.
	SaveLiveBlob bool
	// This parameter caps the total number of LLM invocations allowed per invocation context,
	// providing protection against runaway costs and infinite agent loops.
	MaxLLMCalls int
	// Attach metadata to invocation events
	CustomMetadata map[string]any
	// Enable compositional function calling
	SupportCFC bool
	// Optional. If enabled, the model will detect emotions and adapt its responses accordingly.
	EnableAffectiveDialog bool
}
