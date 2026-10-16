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
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/authconsent"
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

// generateRequestCredentialEvent creates an Event of adk_request_credential
// function calls from the interactive OAuth consent requests a tool raised via
// agent.Context.RequestCredential. It is the credential twin of
// [generateRequestConfirmationEvent]: it marks each emitted call long-running so
// the run pauses until the client returns the user's consent.
//
// The arguments differ from the confirmation event's, and deliberately so. The
// confirmation call wraps the whole originalFunctionCall; adk-python's
// AuthToolArguments carries only the paused call's id under "functionCallId",
// alongside the "authConfig" the client reads the consent URL from. Matching it
// is what lets an existing ADK client drive a Go agent's consent flow.
func generateRequestCredentialEvent(
	invocationContext agent.InvocationContext,
	functionCallEvent *session.Event,
	functionResponseEvent *session.Event,
) *session.Event {
	if functionResponseEvent == nil || len(functionResponseEvent.Actions.RequestedCredentials) == 0 {
		return nil
	}
	if functionCallEvent == nil || functionCallEvent.Content == nil {
		return nil
	}

	parts := []*genai.Part{}
	longRunningToolIDs := []string{}

	// Iterate Content.Parts (an ordered slice) rather than ranging
	// RequestedCredentials (a map, whose iteration order Go randomizes), so the
	// emitted order is deterministic. Same reason as the confirmation twin.
	for _, originalPart := range functionCallEvent.Content.Parts {
		if originalPart.FunctionCall == nil {
			continue
		}
		cfg, ok := functionResponseEvent.Actions.RequestedCredentials[originalPart.FunctionCall.ID]
		if !ok {
			continue
		}

		args, err := authconsent.WireArgs(originalPart.FunctionCall.ID, cfg)
		if err != nil {
			// A consent request that cannot be encoded cannot be asked for. Drop
			// this one rather than emitting a call no client can read; the tool's
			// own error already reports the pause.
			continue
		}

		requestCredentialFC := &genai.FunctionCall{
			ID:   utils.GenerateFunctionCallID(invocationContext),
			Name: authconsent.FunctionCallName,
			Args: args,
		}

		parts = append(parts, &genai.Part{
			FunctionCall:     requestCredentialFC,
			ThoughtSignature: originalPart.ThoughtSignature,
		})
		longRunningToolIDs = append(longRunningToolIDs, requestCredentialFC.ID)
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
