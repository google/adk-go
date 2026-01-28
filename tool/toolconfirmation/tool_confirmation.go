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

// Package toolconfirmation provides structures and utilities for handling
// Human-in-the-Loop tool execution confirmations within the ADK.
package toolconfirmation

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/internal/converters"
)

// RequestConfirmationFunctionCallName defines the specific name for the FunctionCall event
// emitted by ADK when a Human-in-the-Loop confirmation is required.
//
// The 'args' of this FunctionCall include:
//   - "toolConfirmation": A toolConfirmation with the hint.
//   - "originalFunctionCall": The original FunctionCall (including its name and arguments) that the agent intended to execute.
//
// Client applications or frontends interacting with the ADK-powered agent must:
// 1. Listen for events containing a FunctionCall with this name.
// 2. Extract the details of the 'originalFunctionCall' from the arguments.
// 3. Present a clear confirmation prompt to the human user, explaining the action and potential consequences.
// 4. Capture the user's decision (e.g., true for yes/approve, false for no/deny).
// 5. Send a FunctionResponse message back to the ADK. This FunctionResponse MUST:
//   - Have the same 'id' as the received "adk_request_confirmation" FunctionCall.
//   - Have the name set to "adk_request_confirmation".
//   - Include a response payload, typically a map like {"confirmed": bool}.
//
// Based on the boolean value in "confirmed", the ADK will either proceed to execute
// the 'originalFunctionCall' or block it and return an error.
const RequestConfirmationFunctionCallName = "adk_request_confirmation"

// ToolConfirmation represents the state and details of a user confirmation request
// for a tool execution.
type ToolConfirmation struct {
	// Hint is the message provided to the user to explain why the confirmation
	// is needed and what action is being confirmed.
	Hint string

	// Confirmed indicates the user's decision.
	// true if the user approved the action, false if they denied it.
	// The state before the user has responded is typically handled outside
	// this struct (e.g., by the absence of a result or a pending status).
	Confirmed bool

	// Payload contains any additional data or context related to the confirmation request.
	// The structure of the Payload is application-specific.
	Payload any
}

func OriginalCallFrom(functionCall *genai.FunctionCall) (*genai.FunctionCall, error) {
	if functionCall == nil || functionCall.Args == nil {
		return nil, fmt.Errorf("functionCall or its arguments cannot be nil")
	}
	const key = "originalFunctionCall"

	val, exists := functionCall.Args[key]
	if !exists {
		return nil, fmt.Errorf("required argument %q is missing from call with ID %s", key, functionCall.ID)
	}

	originalCall, ok := val.(*genai.FunctionCall)
	if ok {
		return originalCall, nil
	}

	// Check for type correctness
	// This helps debug if the LLM sent a stringified JSON instead of an object
	originalCallRaw, ok := val.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("argument %q has invalid type: expected JSON object (map[string]any) or *genai.FunctionCall, got %T", key, val)
	}

	originalFunctionCall, err := converters.FromMapStructure[genai.FunctionCall](originalCallRaw)
	if err != nil {
		return nil, fmt.Errorf("failed to decode %q structure for call ID %s: %w", key, functionCall.ID, err)
	}

	return originalFunctionCall, nil
}
