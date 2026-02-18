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

package llminternal

import (
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// generateRequestConfirmationEvent creates a new Event containing
// adk_request_confirmation function calls based on the requested confirmations.
// NOTE: The trigger for this in ADK Go is usually a tool.Context.RequestConfirmation call,
// not parsing a function_response_event like in the Python example.
// This function assumes you have a list of confirmations to process.
func generateRequestConfirmationEvent(
	invocationContext agent.InvocationContext,
	functionCallEvent *session.Event,
	functionResponseEvent *session.Event,
) *session.Event {
	if functionResponseEvent == nil || len(functionResponseEvent.Actions.RequestedToolConfirmations) == 0 {
		return nil
	}
	if functionCallEvent == nil || functionCallEvent.Content == nil {
		return nil
	}

	parts := []*genai.Part{}
	longRunningToolIDs := []string{}
	functionCalls := make(map[string]*genai.FunctionCall, len(functionCallEvent.Content.Parts))
	for _, call := range utils.FunctionCalls(functionCallEvent.Content) {
		functionCalls[call.ID] = call
	}

	for funcID, confirmation := range functionResponseEvent.Actions.RequestedToolConfirmations {
		originalFunctionCall, ok := functionCalls[funcID]
		if !ok || originalFunctionCall == nil {
			continue
		}

		// Prepare arguments for the adk_request_confirmation call
		args := map[string]any{
			"originalFunctionCall": originalFunctionCall,
			"toolConfirmation":     confirmation,
		}

		requestConfirmationFC := &genai.FunctionCall{
			ID:   utils.GenerateFunctionCallID(),
			Name: toolconfirmation.FunctionCallName,
			Args: args,
		}

		parts = append(parts, &genai.Part{
			FunctionCall: requestConfirmationFC,
		})
		longRunningToolIDs = append(longRunningToolIDs, requestConfirmationFC.ID)
	}

	if len(parts) == 0 {
		return nil
	}

	return &session.Event{
		InvocationID: invocationContext.InvocationID(),
		Author:       invocationContext.Agent().Name(),
		Branch:       invocationContext.Branch(),
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Parts: parts,
				Role:  genai.RoleModel,
			},
		},
		Timestamp:          time.Now(),
		LongRunningToolIDs: longRunningToolIDs,
		Actions:            session.EventActions{},
	}
}
