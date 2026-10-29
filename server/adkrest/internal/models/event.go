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
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// EventFunctionCall overrides genai.FunctionCall.Args so an empty argument map
// marshals as {} instead of being omitted by omitempty.
type EventFunctionCall struct {
	genai.FunctionCall
	Args map[string]any `json:"args"`
}

// EventPart overrides genai.Part.FunctionCall to use EventFunctionCall.
type EventPart struct {
	genai.Part
	FunctionCall *EventFunctionCall `json:"functionCall,omitempty"`
}

// EventContent overrides genai.Content.Parts to use EventPart.
type EventContent struct {
	genai.Content
	Parts []EventPart `json:"parts,omitempty"`
}

// EventActions represent a data model for session.EventActions
type EventActions struct {
	StateDelta                 map[string]any                               `json:"stateDelta"`
	ArtifactDelta              map[string]int64                             `json:"artifactDelta"`
	Escalate                   bool                                         `json:"escalate,omitempty"`
	SkipSummarization          bool                                         `json:"skipSummarization,omitempty"`
	TransferToAgent            string                                       `json:"transferToAgent,omitempty"`
	RequestedToolConfirmations map[string]toolconfirmation.ToolConfirmation `json:"requestedToolConfirmations,omitempty"`
	Compaction                 *session.EventCompaction                     `json:"compaction,omitempty"`
}

// Event represents a single event in a session.
type Event struct {
	ID                  string                                      `json:"id"`
	InvocationID        string                                      `json:"invocationId"`
	Branch              string                                      `json:"branch,omitempty"`
	Author              string                                      `json:"author"`
	Partial             bool                                        `json:"partial,omitempty"`
	LongRunningToolIDs  []string                                    `json:"longRunningToolIds,omitempty"`
	Content             *EventContent                               `json:"content"`
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

// ToSessionEvent maps Event data struct to session.Event
func ToSessionEvent(event Event) *session.Event {
	return &session.Event{
		ID:                 event.ID,
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
			Content:             toGenAIContent(event.Content),
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
		},
	}
}

// FromSessionEvent maps session.Event to Event data struct
func FromSessionEvent(event session.Event) Event {
	return Event{
		ID:                  event.ID,
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
		Content:             fromGenAIContent(event.LLMResponse.Content),
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
			Compaction:                 event.Actions.Compaction,
		},
	}
}

func fromGenAIContent(c *genai.Content) *EventContent {
	if c == nil {
		return nil
	}
	ec := &EventContent{Content: *c}
	ec.Content.Parts = nil
	if len(c.Parts) > 0 {
		ec.Parts = make([]EventPart, len(c.Parts))
		for i, part := range c.Parts {
			if part == nil {
				continue
			}
			ep := EventPart{Part: *part}
			ep.Part.FunctionCall = nil
			if part.FunctionCall != nil {
				args := part.FunctionCall.Args
				if args == nil {
					args = make(map[string]any)
				}
				fc := *part.FunctionCall
				fc.Args = nil
				ep.FunctionCall = &EventFunctionCall{
					FunctionCall: fc,
					Args:         args,
				}
			}
			ec.Parts[i] = ep
		}
	}
	return ec
}

func toGenAIContent(ec *EventContent) *genai.Content {
	if ec == nil {
		return nil
	}
	c := ec.Content
	if len(ec.Parts) > 0 {
		c.Parts = make([]*genai.Part, len(ec.Parts))
		for i := range ec.Parts {
			p := ec.Parts[i].Part
			if fc := ec.Parts[i].FunctionCall; fc != nil {
				f := fc.FunctionCall
				f.Args = fc.Args
				p.FunctionCall = &f
			}
			c.Parts[i] = &p
		}
	}
	return &c
}
