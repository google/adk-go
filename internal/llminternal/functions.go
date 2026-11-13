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
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolauthinternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// generateRequestConfirmationEvent creates a new Event containing
// adk_request_confirmation function calls based on the requested confirmations.
// NOTE: The trigger for this in ADK Go is usually a agent.Context.RequestConfirmation call,
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

	// Emit confirmations in the order their function calls appear in the model
	// response, mirroring adk-python. Iterating Content.Parts (an ordered slice)
	// rather than ranging RequestedToolConfirmations (a map, whose iteration
	// order Go randomizes) keeps the emitted order deterministic and aligned
	// with execution flow.
	for _, originalPart := range functionCallEvent.Content.Parts {
		if originalPart.FunctionCall == nil {
			continue
		}
		confirmation, ok := functionResponseEvent.Actions.RequestedToolConfirmations[originalPart.FunctionCall.ID]
		if !ok {
			continue
		}

		// Prepare arguments for the adk_request_confirmation call
		args := map[string]any{
			"originalFunctionCall": originalPart.FunctionCall,
			"toolConfirmation":     confirmation,
		}

		requestConfirmationFC := &genai.FunctionCall{
			ID:   utils.GenerateFunctionCallID(invocationContext),
			Name: toolconfirmation.FunctionCallName,
			Args: args,
		}

		parts = append(parts, &genai.Part{
			FunctionCall:     requestConfirmationFC,
			ThoughtSignature: originalPart.ThoughtSignature,
		})
		longRunningToolIDs = append(longRunningToolIDs, requestConfirmationFC.ID)
	}

	if len(parts) == 0 {
		return nil
	}

	ev := session.NewEvent(invocationContext, invocationContext.InvocationID())
	ev.Author = invocationContext.Agent().Name()
	ev.Branch = invocationContext.Branch()
	ev.LLMResponse = model.LLMResponse{
		Content: &genai.Content{
			Parts: parts,
			Role:  genai.RoleModel,
		},
	}
	ev.LongRunningToolIDs = longRunningToolIDs
	return ev
}

// generateAuthEvent returns the event that asks the client for the
// credentials the tools requested in functionResponseEvent, with one
// adk_request_credential call for each distinct credential key, or nil when
// no tool requested one.
//
// adk-go has no tool-context request_credential, so the flow turns a tool's
// request into an auth request here. Each requested config in
// functionResponseEvent is replaced by its auth request, which carries none
// of the secrets the tool was configured with, so they are neither stored in
// the session nor sent to the client. On error functionResponseEvent is left
// unchanged.
func generateAuthEvent(ctx agent.InvocationContext, functionCallEvent, functionResponseEvent *session.Event) (*session.Event, error) {
	if functionResponseEvent == nil || len(functionResponseEvent.Actions.RequestedAuthConfigs) == 0 {
		return nil, nil
	}
	requested := functionResponseEvent.Actions.RequestedAuthConfigs

	// Visit the requests in the order of the model's function calls, then any
	// others in sorted order, since map iteration order is random.
	var ids []string
	if functionCallEvent != nil {
		for _, fc := range utils.FunctionCalls(functionCallEvent.Content) {
			if _, ok := requested[fc.ID]; ok && !slices.Contains(ids, fc.ID) {
				ids = append(ids, fc.ID)
			}
		}
	}
	for _, fc := range utils.FunctionResponses(functionResponseEvent.Content) {
		if _, ok := requested[fc.ID]; ok && !slices.Contains(ids, fc.ID) {
			ids = append(ids, fc.ID)
		}
	}
	for _, id := range slices.Sorted(maps.Keys(requested)) {
		if !slices.Contains(ids, id) {
			ids = append(ids, id)
		}
	}

	generated := make(map[string]toolauth.Config, len(requested))
	seenKeys := make(map[string]bool)
	var parts []*genai.Part
	var longRunningToolIDs []string
	for _, id := range ids {
		cfg := requested[id]
		req, err := toolauthinternal.GenerateRequest(&cfg)
		if err != nil {
			return nil, fmt.Errorf("auth request for function call %q: %w", id, err)
		}
		generated[id] = *req
		// GenerateRequest sets CredentialKey, so this is the key.
		if seenKeys[req.CredentialKey] {
			continue
		}
		seenKeys[req.CredentialKey] = true
		args, err := toolArgumentsMap(toolauth.ToolArguments{FunctionCallID: id, AuthConfig: *req})
		if err != nil {
			return nil, fmt.Errorf("auth request for function call %q: %w", id, err)
		}
		fc := &genai.FunctionCall{
			ID:   utils.GenerateFunctionCallID(ctx),
			Name: toolauth.RequestCredentialFunctionCallName,
			Args: args,
		}
		parts = append(parts, &genai.Part{FunctionCall: fc})
		longRunningToolIDs = append(longRunningToolIDs, fc.ID)
	}
	functionResponseEvent.Actions.RequestedAuthConfigs = generated

	var role string
	if functionResponseEvent.Content != nil {
		role = functionResponseEvent.Content.Role
	}
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = ctx.Agent().Name()
	ev.Branch = ctx.Branch()
	ev.LLMResponse = model.LLMResponse{
		Content: &genai.Content{Parts: parts, Role: role},
	}
	ev.LongRunningToolIDs = longRunningToolIDs
	return ev, nil
}

// toolArgumentsMap returns args as the JSON object an adk_request_credential
// call carries.
func toolArgumentsMap(args toolauth.ToolArguments) (map[string]any, error) {
	b, err := json.Marshal(args)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	return m, nil
}
