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
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/converters"
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

// generateAuthEvent returns the event that asks the client for the credentials
// the tools in functionResponseEvent requested, or nil if they requested none.
// It makes one adk_request_credential call per credential key, for the first
// tool that asked for it in the order of the responses, which is the order the
// calls were made in.
func generateAuthEvent(ctx agent.InvocationContext, functionResponseEvent *session.Event) (*session.Event, error) {
	requests := functionResponseEvent.Actions.RequestedAuthConfigs
	if len(requests) == 0 {
		return nil, nil
	}
	var parts []*genai.Part
	var longRunningToolIDs []string
	seenKeys := map[string]bool{}
	for _, fr := range utils.FunctionResponses(functionResponseEvent.Content) {
		cfg, ok := requests[fr.ID]
		if !ok || seenKeys[cfg.CredentialKey] {
			continue
		}
		// A request with no key is never merged with another, as in adk-python.
		if cfg.CredentialKey != "" {
			seenKeys[cfg.CredentialKey] = true
		}
		args, err := converters.ToMapStructure(toolauth.ToolArguments{FunctionCallID: fr.ID, AuthConfig: cfg})
		if err != nil {
			return nil, fmt.Errorf("encode %s args: %w", toolauth.RequestCredentialFunctionCallName, err)
		}
		call := &genai.FunctionCall{
			ID:   utils.GenerateFunctionCallID(ctx),
			Name: toolauth.RequestCredentialFunctionCallName,
			Args: args,
		}
		parts = append(parts, &genai.Part{FunctionCall: call})
		longRunningToolIDs = append(longRunningToolIDs, call.ID)
	}
	if len(parts) == 0 {
		return nil, nil
	}
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = ctx.Agent().Name()
	ev.Branch = ctx.Branch()
	// adk-python gives the request the role of the tool responses, which is user.
	ev.LLMResponse = model.LLMResponse{Content: &genai.Content{Parts: parts, Role: genai.RoleUser}}
	ev.LongRunningToolIDs = longRunningToolIDs
	return ev, nil
}

// yieldWithAuthRequest yields ev and then, when tools in ev asked for
// credentials, the adk_request_credential call for them. It reports whether the
// caller should go on.
//
// As in adk-python, asking the user to sign in ends the invocation, so the flow
// calls neither its model nor a transfer target again.
func yieldWithAuthRequest(ctx agent.InvocationContext, ev *session.Event, yield func(*session.Event, error) bool) bool {
	authEvent, err := generateAuthEvent(ctx, ev)
	if err != nil {
		yield(nil, err)
		return false
	}
	if !yield(ev, nil) {
		return false
	}
	if authEvent == nil {
		return true
	}
	ctx.EndInvocation()
	return yield(authEvent, nil)
}
