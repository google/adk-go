package models

import (
	"time"

	"google.golang.org/adk/llm"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type LLMResponse struct {
	Content           *genai.Content           `json:"content"`
	GroundingMetadata *genai.GroundingMetadata `json:"groundingMetadata"`
	Partial           bool                     `json:"partial"`
	TurnComplete      bool                     `json:"turnComplete"`
	Interrupted       bool                     `json:"interrupted"`
	ErrorCode         int                      `json:"errorCode"`
	ErrorMessage      string                   `json:"errorMessage"`
}

type Event struct {
	ID                 string      `json:"id"`
	Time               time.Time   `json:"time"`
	InvocationID       string      `json:"invocationId"`
	Branch             string      `json:"branch"`
	Author             string      `json:"author"`
	Partial            bool        `json:"partial"`
	LongRunningToolIDs []string    `json:"longRunningToolIds"`
	LLMResponse        LLMResponse `json:"llmResponse"`
}

func ToSessionEvent(event Event) *session.Event {
	return &session.Event{
		ID:                 event.ID,
		Time:               event.Time,
		InvocationID:       event.InvocationID,
		Branch:             event.Branch,
		Author:             event.Author,
		Partial:            event.Partial,
		LongRunningToolIDs: event.LongRunningToolIDs,
		LLMResponse: &llm.Response{
			Content:           event.LLMResponse.Content,
			GroundingMetadata: event.LLMResponse.GroundingMetadata,
			Partial:           event.LLMResponse.Partial,
			TurnComplete:      event.LLMResponse.TurnComplete,
			Interrupted:       event.LLMResponse.Interrupted,
			ErrorCode:         event.LLMResponse.ErrorCode,
			ErrorMessage:      event.LLMResponse.ErrorMessage,
		},
	}
}

func FromSessionEvent(event session.Event) Event {
	return Event{
		ID:                 event.ID,
		Time:               event.Time,
		InvocationID:       event.InvocationID,
		Branch:             event.Branch,
		Author:             event.Author,
		Partial:            event.Partial,
		LongRunningToolIDs: event.LongRunningToolIDs,
		LLMResponse: LLMResponse{
			Content:           event.LLMResponse.Content,
			GroundingMetadata: event.LLMResponse.GroundingMetadata,
			Partial:           event.LLMResponse.Partial,
			TurnComplete:      event.LLMResponse.TurnComplete,
			Interrupted:       event.LLMResponse.Interrupted,
			ErrorCode:         event.LLMResponse.ErrorCode,
			ErrorMessage:      event.LLMResponse.ErrorMessage,
		},
	}
}
