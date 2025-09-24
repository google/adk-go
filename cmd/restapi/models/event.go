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
	"time"

	"google.golang.org/adk/llm"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// Event represents a single event in a session.
type Event struct {
	ID                 string                   `json:"id"`
	Time               int64                    `json:"time"`
	InvocationID       string                   `json:"invocationId"`
	Branch             string                   `json:"branch"`
	Author             string                   `json:"author"`
	Partial            bool                     `json:"partial"`
	LongRunningToolIDs []string                 `json:"longRunningToolIds"`
	Content            *genai.Content           `json:"content"`
	GroundingMetadata  *genai.GroundingMetadata `json:"groundingMetadata"`
	TurnComplete       bool                     `json:"turnComplete"`
	Interrupted        bool                     `json:"interrupted"`
	ErrorCode          int                      `json:"errorCode"`
	ErrorMessage       string                   `json:"errorMessage"`
}

func ToSessionEvent(event Event) *session.Event {
	return &session.Event{
		ID:                 event.ID,
		Time:               time.Unix(event.Time, 0),
		InvocationID:       event.InvocationID,
		Branch:             event.Branch,
		Author:             event.Author,
		Partial:            event.Partial,
		LongRunningToolIDs: event.LongRunningToolIDs,
		LLMResponse: &llm.Response{
			Content:           event.Content,
			GroundingMetadata: event.GroundingMetadata,
			Partial:           event.Partial,
			TurnComplete:      event.TurnComplete,
			Interrupted:       event.Interrupted,
			ErrorCode:         event.ErrorCode,
			ErrorMessage:      event.ErrorMessage,
		},
	}
}

func FromSessionEvent(event session.Event) Event {
	return Event{
		ID:                 event.ID,
		Time:               event.Time.Unix(),
		InvocationID:       event.InvocationID,
		Branch:             event.Branch,
		Author:             event.Author,
		Partial:            event.Partial,
		LongRunningToolIDs: event.LongRunningToolIDs,
		Content:            event.LLMResponse.Content,
		GroundingMetadata:  event.LLMResponse.GroundingMetadata,
		TurnComplete:       event.LLMResponse.TurnComplete,
		Interrupted:        event.LLMResponse.Interrupted,
		ErrorCode:          event.LLMResponse.ErrorCode,
		ErrorMessage:       event.LLMResponse.ErrorMessage,
	}
}
