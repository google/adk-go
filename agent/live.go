// Copyright 2026 Google LLC
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
	"google.golang.org/genai"
)

// LiveSession manages the bidirectional stream for a live session.
type LiveSession interface {
	Send(req LiveRequest) error
	Close() error
}

// LiveRequest represents an incoming client event for a live session.
type LiveRequest struct {
	// RealtimeInput can be *genai.Blob, *genai.ActivityStart, or *genai.ActivityEnd.
	RealtimeInput any

	// Content represents standard text or multimodal content from the user.
	// Can also represent a reply to a tool call if it contains a FunctionResponse part.
	Content *genai.Content
}

// LiveRunConfig contains options for configuring a live session.
type LiveRunConfig struct {
	ResponseModalities       []genai.Modality
	SpeechConfig             *genai.SpeechConfig
	InputAudioTranscription  *genai.AudioTranscriptionConfig
	OutputAudioTranscription *genai.AudioTranscriptionConfig
	RealtimeInputConfig     *genai.RealtimeInputConfig
	EnableAffectiveDialog    bool
	Proactivity             *genai.ProactivityConfig
	SessionResumption       *genai.SessionResumptionConfig
	SaveLiveBlob            bool
	MaxLLMCalls             int
}
