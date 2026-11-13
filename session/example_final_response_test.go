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

package session_test

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// An event with an ErrorCode is a final response even when its content holds
// a function response or ends with a code execution result, as an event built
// from an A2A error reply can. It is not final while it is partial or when it
// holds a function call.
func ExampleEvent_IsFinalResponse() {
	toolResult := genai.NewContentFromFunctionResponse("lookup", map[string]any{"error": "timeout"}, genai.RoleUser)
	call := genai.NewContentFromFunctionCall("lookup", map[string]any{"id": "42"}, genai.RoleModel)
	events := []struct {
		name  string
		event *session.Event
	}{
		{"tool result", &session.Event{LLMResponse: model.LLMResponse{Content: toolResult}}},
		{"error with tool result", &session.Event{LLMResponse: model.LLMResponse{Content: toolResult, ErrorCode: "TOOL_FAILED"}}},
		{"error, partial", &session.Event{LLMResponse: model.LLMResponse{ErrorCode: "SAFETY", Partial: true}}},
		{"error with function call", &session.Event{LLMResponse: model.LLMResponse{Content: call, ErrorCode: "SAFETY"}}},
		{"error message only", &session.Event{LLMResponse: model.LLMResponse{Content: toolResult, ErrorMessage: "timeout"}}},
	}
	for _, e := range events {
		fmt.Printf("%s: %v\n", e.name, e.event.IsFinalResponse())
	}
	// Output:
	// tool result: false
	// error with tool result: true
	// error, partial: false
	// error with function call: false
	// error message only: false
}
