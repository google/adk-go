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

	"google.golang.org/adk/session"
)

// StreamQueryRequest is a struct representing JSON-encoded payload to async_stream_query method with dedicated Input.
type StreamQueryRequest struct {
	ClassMethod string           `json:"class_method"`
	Input       StreamQueryInput `json:"input"`
}

// StreamQueryInput is the actual Input for async_stream_query method.
type StreamQueryInput struct {
	UserID    string        `json:"user_id"`
	SessionID string        `json:"session_id"`
	Message   genai.Content `json:"message"`
}

// StreamQueryResponse defines the content of event data for async_stream_query method.
// It is returned as one line with JSON-encoded StreamQuerySSEResponse
// Please mind that errors are also returned by this method
type StreamQuerySSEResponse session.Event
