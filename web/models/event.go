package models

import (
	"time"

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
	ID                 string    `json:"id"`
	Time               time.Time `json:"time"`
	InvocationID       string    `json:"invocationId"`
	Branch             string    `json:"branch"`
	Author             string    `json:"author"`
	Partial            bool      `json:"partial"`
	LongRunningToolIDs []string  `json:"longRunningToolIds"`
}
