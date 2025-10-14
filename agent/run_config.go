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

type StreamingMode string

const (
	StreamingModeNone StreamingMode = "none"
	StreamingModeSSE  StreamingMode = "sse"
	StreamingModeBidi StreamingMode = "bidi"
)

// RunConfig controls runtime behavior.
type RunConfig struct {
	// Speech configuration for the live agent.
	SpeechConfig *genai.SpeechConfig
	// Output transcription for live agents with audio response.
	OutputAudioTranscriptionConfig *genai.AudioTranscriptionConfig
	// The output modalities. If not set, it defaults to AUDIO.
	ResponseModalities []string
	// Streaming mode, None or StreamingMode.SSE or StreamingMode.BIDI.
	StreamingMode StreamingMode
	// Whether or not to save the input blobs as artifacts
	SaveInputBlobsAsArtifacts bool

	// Whether to support CFC (Compositional Function Calling). Only applicable for
	// StreamingModeSSE. If it's true. the LIVE API will be invoked since only LIVE
	// API supports CFC.
	//
	// .. warning::
	//      This feature is **experimental** and its API or behavior may change
	//     in future releases.
	SupportCFC bool

	// A limit on the total number of llm calls for a given run.
	//
	// Valid Values:
	//  - More than 0 and less than sys.maxsize: The bound on the number of llm
	//    calls is enforced, if the value is set in this range.
	//  - Less than or equal to 0: This allows for unbounded number of llm calls.
	MaxLLMCalls int
}
