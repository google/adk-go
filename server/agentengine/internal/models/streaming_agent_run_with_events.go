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

package models

import (
	"google.golang.org/genai"

	"google.golang.org/adk/session"
)

// StreamingAgentRunWithEventsRequest is the JSON-encoded payload for the
// streaming_agent_run_with_events method.
type StreamingAgentRunWithEventsRequest struct {
	ClassMethod string                           `json:"class_method"`
	Input       StreamingAgentRunWithEventsInput `json:"input"`
}

// StreamingAgentRunWithEventsInput wraps the actual request payload as JSON.
type StreamingAgentRunWithEventsInput struct {
	RequestJSON string `json:"request_json"`
}

// StreamingAgentRunWithEventsRunRequest is the request decoded from
// StreamingAgentRunWithEventsInput.RequestJSON.
type StreamingAgentRunWithEventsRunRequest struct {
	UserID    string        `json:"user_id"`
	SessionID string        `json:"session_id"`
	Message   genai.Content `json:"message"`
}

// StreamingAgentRunWithEventsResponse is the response envelope expected by
// Gemini Enterprise for streaming_agent_run_with_events.
type StreamingAgentRunWithEventsResponse struct {
	Events    []*session.Event `json:"events,omitempty"`
	Artifacts []any            `json:"artifacts,omitempty"`
	SessionID string           `json:"session_id,omitempty"`
}
