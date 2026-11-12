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
	RealtimeInputConfig      *genai.RealtimeInputConfig
	EnableAffectiveDialog    bool
	Proactivity              *genai.ProactivityConfig
	SessionResumption        *genai.SessionResumptionConfig
	SaveLiveBlob             bool
	MaxLLMCalls              int
	// CustomMetadata holds caller-supplied key-value metadata for this live
	// invocation.
	//
	// The runner merges CustomMetadata (after dropping ADK-reserved keys, as
	// on RunConfig.CustomMetadata) into every event yielded by RunLive's event
	// stream — including a BeforeRunCallback early-exit reply — before
	// OnEventCallback plugins run and before non-partial events are persisted.
	// An event's own value wins on a key conflict. User content events
	// appended through LiveSession.Send do not carry CustomMetadata.
	//
	// Nil or empty leaves every event's CustomMetadata unchanged. The runner
	// copies the non-reserved entries when RunLive is called and merges that
	// copy into events: the caller's map is never mutated by ADK, and changes
	// to it after RunLive is called do not reach the events. The copy is
	// shallow, so values must not be mutated concurrently while the session
	// runs.
	//
	// A plugin that edits the event and returns nil keeps its edits; one that
	// returns an event, the same or a new one, gets the run's metadata merged
	// into it, its own keys winning.
	CustomMetadata map[string]any
	// GetSessionConfig controls which events are fetched when the runner
	// loads the session from the session service at the start of RunLive.
	// Nil or a zero GetSessionConfig loads all session events.
	GetSessionConfig *GetSessionConfig
}
