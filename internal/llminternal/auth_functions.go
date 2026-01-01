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
	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/auth"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

const afFunctionCallIDPrefix = "adk-"

// generateFunctionCallID creates a unique function call ID.
// This matches Python's generate_client_function_call_id() with AF_FUNCTION_CALL_ID_PREFIX = 'adk-'
func generateFunctionCallID() string {
	return afFunctionCallIDPrefix + uuid.NewString()
}

// GenerateAuthEvent creates an event with adk_request_credential function calls
// from the RequestedAuthConfigs in the function response event.
// This matches Python ADK's generate_auth_event in flows/llm_flows/functions.py.
func GenerateAuthEvent(ctx agent.InvocationContext, fnResponseEvent *session.Event) *session.Event {
	if fnResponseEvent == nil || len(fnResponseEvent.Actions.RequestedAuthConfigs) == 0 {
		return nil
	}

	var parts []*genai.Part
	var longRunningToolIDs []string

	for functionCallID, authConfig := range fnResponseEvent.Actions.RequestedAuthConfigs {
		// Create args map matching Python's AuthToolArguments.model_dump()
		// Note: We preserve *auth.AuthConfig pointer since this is in-memory,
		// matching Python's behavior where objects are passed by reference.
		argsMap := map[string]any{
			"function_call_id": functionCallID,
			"auth_config":      authConfig, // Keep as *auth.AuthConfig pointer
		}

		// Create the adk_request_credential function call
		requestEucFunctionCall := &genai.FunctionCall{
			Name: auth.RequestEUCFunctionCallName,
			Args: argsMap,
		}

		// Generate a unique ID for this function call
		requestEucFunctionCall.ID = generateFunctionCallID()
		longRunningToolIDs = append(longRunningToolIDs, requestEucFunctionCall.ID)

		parts = append(parts, &genai.Part{
			FunctionCall: requestEucFunctionCall,
		})
	}

	// Determine the role from the original event
	role := "model"
	if fnResponseEvent.Content != nil && fnResponseEvent.Content.Role != "" {
		role = fnResponseEvent.Content.Role
	}

	// Create the auth event
	authEvent := session.NewEvent(ctx.InvocationID())
	authEvent.Author = ctx.Agent().Name()
	authEvent.Branch = ctx.Branch()
	authEvent.LLMResponse = model.LLMResponse{
		Content: &genai.Content{
			Role:  role,
			Parts: parts,
		},
	}
	authEvent.LongRunningToolIDs = longRunningToolIDs

	return authEvent
}
