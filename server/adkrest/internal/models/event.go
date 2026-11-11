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

package models

import (
	"encoding/json"
	"fmt"
	"math"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// EventActions represent a data model for session.EventActions
type EventActions struct {
	StateDelta                 stateValues                                  `json:"stateDelta"`
	ArtifactDelta              map[string]int64                             `json:"artifactDelta"`
	Escalate                   bool                                         `json:"escalate,omitempty"`
	SkipSummarization          bool                                         `json:"skipSummarization,omitempty"`
	TransferToAgent            string                                       `json:"transferToAgent,omitempty"`
	RequestedToolConfirmations map[string]toolconfirmation.ToolConfirmation `json:"requestedToolConfirmations,omitempty"`
	RequestedAuthConfigs       authConfigs                                  `json:"requestedAuthConfigs,omitempty"`
	Compaction                 *session.EventCompaction                     `json:"compaction,omitempty"`
	AgentState                 map[string]any                               `json:"agentState,omitempty"`
	EndOfAgent                 bool                                         `json:"endOfAgent,omitempty"`
}

// Event represents a single event in a session.
type Event struct {
	ID                  string                                      `json:"id"`
	Timestamp           float64                                     `json:"timestamp"`
	InvocationID        string                                      `json:"invocationId"`
	Branch              string                                      `json:"branch,omitempty"`
	Author              string                                      `json:"author"`
	Partial             bool                                        `json:"partial,omitempty"`
	LongRunningToolIDs  []string                                    `json:"longRunningToolIds,omitempty"`
	Content             *genai.Content                              `json:"content"`
	GroundingMetadata   *genai.GroundingMetadata                    `json:"groundingMetadata"`
	UsageMetadata       *genai.GenerateContentResponseUsageMetadata `json:"usageMetadata"`
	TurnComplete        bool                                        `json:"turnComplete,omitempty"`
	Interrupted         bool                                        `json:"interrupted,omitempty"`
	ErrorCode           string                                      `json:"errorCode,omitempty"`
	ErrorMessage        string                                      `json:"errorMessage,omitempty"`
	AvgLogprobs         float64                                     `json:"avgLogprobs,omitempty"`
	FinishReason        genai.FinishReason                          `json:"finishReason,omitempty"`
	ModelVersion        string                                      `json:"modelVersion,omitempty"`
	InputTranscription  *genai.Transcription                        `json:"inputTranscription,omitempty"`
	OutputTranscription *genai.Transcription                        `json:"outputTranscription,omitempty"`
	Actions             EventActions                                `json:"actions"`
	Output              any                                         `json:"output,omitempty"`
	NodeInfo            *session.NodeInfo                           `json:"nodeInfo,omitempty"`
	IsolationScope      string                                      `json:"isolationScope,omitempty"`
	Routes              []string                                    `json:"routes,omitempty"`
	RequestedInput      *session.RequestInput                       `json:"requestedInput,omitempty"`
}

// The calendar bounds for an event timestamp, in epoch microseconds: the first
// and last instants of years 1 and 9999. They match the bounds the
// session.Event JSON decoder applies to a numeric timestamp, which
// TestToSessionEventTimestampBoundsMatchSessionDecoder pins.
const (
	minTimestampMicros = -62135596800 * 1e6
	maxTimestampMicros = 253402300799 * 1e6
)

// timestampFromEpochSeconds converts a REST event timestamp to a time.Time.
// Zero means the client sent none and maps to the zero time.Time.
//
// A value outside years 1 to 9999 is an error rather than an instant. It is
// most often epoch milliseconds sent as seconds (JavaScript's Date.now()), or
// a number too large for the int64 conversion, which overflows into a negative
// time. Session services store such an instant without complaint, but some
// cannot read it back, so the session it lands in can no longer be fetched or
// listed.
func timestampFromEpochSeconds(secs float64) (time.Time, error) {
	if secs == 0 {
		return time.Time{}, nil
	}
	micros := math.Round(secs * 1e6)
	if math.IsNaN(micros) || micros < minTimestampMicros || micros > maxTimestampMicros {
		return time.Time{}, fmt.Errorf("event timestamp %v is not a plausible date: want epoch seconds between years 1 and 9999", secs)
	}
	return time.UnixMicro(int64(micros)).UTC(), nil
}

// ToSessionEvent maps Event data struct to session.Event. It returns an error
// when the event's timestamp is not a plausible date; see
// timestampFromEpochSeconds.
func ToSessionEvent(event Event) (*session.Event, error) {
	ts, err := timestampFromEpochSeconds(event.Timestamp)
	if err != nil {
		return nil, err
	}
	return &session.Event{
		ID:                 event.ID,
		Timestamp:          ts,
		InvocationID:       event.InvocationID,
		Branch:             event.Branch,
		Author:             event.Author,
		LongRunningToolIDs: event.LongRunningToolIDs,
		Output:             event.Output,
		NodeInfo:           event.NodeInfo,
		IsolationScope:     event.IsolationScope,
		Routes:             event.Routes,
		RequestedInput:     event.RequestedInput,
		LLMResponse: model.LLMResponse{
			AvgLogprobs:         event.AvgLogprobs,
			Content:             event.Content,
			GroundingMetadata:   event.GroundingMetadata,
			UsageMetadata:       event.UsageMetadata,
			Partial:             event.Partial,
			TurnComplete:        event.TurnComplete,
			Interrupted:         event.Interrupted,
			ErrorCode:           event.ErrorCode,
			ErrorMessage:        event.ErrorMessage,
			FinishReason:        event.FinishReason,
			ModelVersion:        event.ModelVersion,
			InputTranscription:  event.InputTranscription,
			OutputTranscription: event.OutputTranscription,
		},
		Actions: session.EventActions{
			StateDelta:                 event.Actions.StateDelta,
			ArtifactDelta:              event.Actions.ArtifactDelta,
			Escalate:                   event.Actions.Escalate,
			SkipSummarization:          event.Actions.SkipSummarization,
			TransferToAgent:            event.Actions.TransferToAgent,
			RequestedToolConfirmations: event.Actions.RequestedToolConfirmations,
			// Actions.Compaction is deliberately not mapped inbound. A
			// compaction record tells prompt assembly to drop a span of history
			// and substitute content in its place, so honoring one from a
			// request body would let a client erase a conversation and inject
			// text into it as a model turn. Only the runner writes these.
			// FromSessionEvent still returns them, so a client can read
			// summaries it did not author.
			// Actions.RequestedAuthConfigs is not mapped inbound either. Only a
			// tool writes a credential request, and one taken from a request body
			// would claim a sign-in no tool asked for, which compaction then
			// holds open as an unanswered call.
			// Actions.AgentState and Actions.EndOfAgent are not mapped inbound.
			// They tell a resumed workflow where to continue, so one taken from
			// a request body could skip a sub-agent such as an approval step.
		},
	}, nil
}

// FromSessionEvent maps session.Event to Event data struct
func FromSessionEvent(event session.Event) Event {
	var ts float64
	if !event.Timestamp.IsZero() {
		ts = float64(event.Timestamp.UnixMicro()) / 1e6
	}
	return Event{
		ID:                  event.ID,
		Timestamp:           ts,
		InvocationID:        event.InvocationID,
		Branch:              event.Branch,
		Author:              event.Author,
		Partial:             event.Partial,
		LongRunningToolIDs:  event.LongRunningToolIDs,
		Output:              event.Output,
		NodeInfo:            event.NodeInfo,
		IsolationScope:      event.IsolationScope,
		Routes:              event.Routes,
		RequestedInput:      event.RequestedInput,
		AvgLogprobs:         event.LLMResponse.AvgLogprobs,
		Content:             event.LLMResponse.Content,
		GroundingMetadata:   event.LLMResponse.GroundingMetadata,
		UsageMetadata:       event.LLMResponse.UsageMetadata,
		TurnComplete:        event.LLMResponse.TurnComplete,
		Interrupted:         event.LLMResponse.Interrupted,
		ErrorCode:           event.LLMResponse.ErrorCode,
		ErrorMessage:        event.LLMResponse.ErrorMessage,
		FinishReason:        event.LLMResponse.FinishReason,
		ModelVersion:        event.LLMResponse.ModelVersion,
		InputTranscription:  event.LLMResponse.InputTranscription,
		OutputTranscription: event.LLMResponse.OutputTranscription,
		Actions: EventActions{
			StateDelta:                 event.Actions.StateDelta,
			ArtifactDelta:              event.Actions.ArtifactDelta,
			Escalate:                   event.Actions.Escalate,
			SkipSummarization:          event.Actions.SkipSummarization,
			TransferToAgent:            event.Actions.TransferToAgent,
			RequestedToolConfirmations: event.Actions.RequestedToolConfirmations,
			RequestedAuthConfigs:       event.Actions.RequestedAuthConfigs,
			Compaction:                 event.Actions.Compaction,
			AgentState:                 event.Actions.AgentState,
			EndOfAgent:                 event.Actions.EndOfAgent,
		},
	}
}

func (e Event) MarshalJSON() ([]byte, error) {
	// Define Proxy structs to override specific JSON tags.
	// These embed the original types to inherit all other fields automatically.

	// ProxyFunctionCall overrides 'Args' to remove 'omitempty'.
	type ProxyFunctionCall struct {
		*genai.FunctionCall
		Args map[string]any `json:"args"` // Tag changed: omitempty removed
	}

	// ProxyPart overrides 'FunctionCall' to use ProxyFunctionCall.
	type ProxyPart struct {
		*genai.Part
		FunctionCall *ProxyFunctionCall `json:"functionCall,omitempty"`
	}

	// ProxyContent overrides 'Parts' to use ProxyPart.
	type ProxyContent struct {
		*genai.Content
		Parts []*ProxyPart `json:"parts,omitempty"`
	}

	// Create an Alias of Event to prevent infinite recursion during Marshal.
	type EventAlias Event

	// Create a temporary struct that mimics Event but uses ProxyContent.
	aux := &struct {
		EventAlias
		Content *ProxyContent `json:"content"`
	}{
		EventAlias: EventAlias(e),
	}

	// Reconstruct the Content hierarchy if it exists.
	if e.Content != nil {
		aux.Content = &ProxyContent{
			Content: e.Content,
			Parts:   make([]*ProxyPart, 0, len(e.Content.Parts)),
		}

		for _, part := range e.Content.Parts {
			// A nil part (e.g. a null in a stored request body) has nothing to
			// wrap; drop it from the output so REST clients — including the
			// bundled web UI, which reads fields off every part it renders —
			// never see a null part.
			if part == nil {
				continue
			}
			part, err := withoutCredentialSecrets(part)
			if err != nil {
				return nil, err
			}
			// Wrap the original part
			proxyPart := &ProxyPart{Part: part}

			// If this part is a FunctionCall, wrap it to enforce Args visibility
			if part.FunctionCall != nil {
				// Ensure args is at least an empty map (not nil) so it marshals to {}
				args := part.FunctionCall.Args
				if args == nil {
					args = make(map[string]any)
				}

				proxyPart.FunctionCall = &ProxyFunctionCall{
					FunctionCall: part.FunctionCall,
					Args:         args,
				}
			}
			aux.Content.Parts = append(aux.Content.Parts, proxyPart)
		}
	}

	return json.Marshal(aux)
}
