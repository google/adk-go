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
	"strings"
	"unicode"

	"github.com/mitchellh/mapstructure"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/auth"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func identityRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/identity.py)
	return nil
}

func nlPlanningRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/_nl_plnning.py)
	return nil
}

func codeExecutionRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/_code_execution.py)
	return nil
}

// AuthPreprocessorResult contains the result of auth preprocessing.
// It tells the Flow whether tools need to be re-executed.
type AuthPreprocessorResult struct {
	// ToolIdsToResume contains the IDs of function calls that should be re-executed.
	ToolIdsToResume map[string]bool
	// CredentialsStored indicates if any credentials were stored.
	CredentialsStored bool
	// OriginalEvent is the event containing the original function calls to resume.
	OriginalEvent *session.Event
}

// CurrentAuthPreprocessorResult holds the result of the last auth preprocessing.
// This is set by authPreprocessor and read by Flow.runOneStep.
var CurrentAuthPreprocessorResult *AuthPreprocessorResult

func authPreprocessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	// Reset the result
	CurrentAuthPreprocessorResult = nil

	// This implements Python ADK's auth_preprocessor logic exactly.
	// It checks SESSION EVENTS (not userContent) for auth responses.
	// This is crucial - checking session events means we won't re-process
	// the same auth response on every runOneStep iteration.

	events := ctx.Session().Events()
	if events.Len() == 0 {
		return nil
	}

	// Find the last event with non-None content (Python lines 54-60)
	var lastEventWithContent *session.Event
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)
		if event.Content != nil {
			lastEventWithContent = event
			break
		}
	}

	// Check if the last event with content is authored by user (Python lines 62-64)
	if lastEventWithContent == nil || lastEventWithContent.Author != "user" {
		return nil
	}

	// Get function responses from the event (Python lines 66-68)
	var functionResponses []*genai.FunctionResponse
	for _, part := range lastEventWithContent.Content.Parts {
		if part.FunctionResponse != nil {
			functionResponses = append(functionResponses, part.FunctionResponse)
		}
	}
	if len(functionResponses) == 0 {
		return nil
	}

	// Collect request_euc function call IDs and store credentials (Python lines 70-80)
	requestEucFunctionCallIDs := make(map[string]bool)
	for _, funcResponse := range functionResponses {
		if funcResponse.Name != auth.RequestEUCFunctionCallName {
			continue
		}
		// Found the function call response for the system long running request euc function call
		requestEucFunctionCallIDs[funcResponse.ID] = true

		// Parse and store the credential
		if funcResponse.Response != nil {
			if authConfigData, ok := funcResponse.Response["auth_config"]; ok {
				authConfig, err := parseAuthConfigFromMap(authConfigData)
				if err != nil {
					continue
				}
				// Store the credential in session state
				if authConfig.CredentialKey != "" && authConfig.ExchangedAuthCredential != nil {
					key := session.KeyPrefixTemp + authConfig.CredentialKey
					if err := ctx.Session().State().Set(key, authConfig.ExchangedAuthCredential); err != nil {
						return fmt.Errorf("failed to store auth credential: %w", err)
					}
				}
			}
		}
	}

	if len(requestEucFunctionCallIDs) == 0 {
		return nil
	}

	// Now find the original tool calls that need to be resumed.
	// Python lines 85-130: Search backwards for adk_request_credential function calls,
	// then find the original tool calls that triggered them.

	result := &AuthPreprocessorResult{
		ToolIdsToResume: make(map[string]bool),
	}

	for i := events.Len() - 2; i >= 0; i-- {
		event := events.At(i)
		if event.Content == nil {
			continue
		}

		// Look for adk_request_credential function calls in this event (Python lines 87-101)
		var functionCalls []*genai.FunctionCall
		for _, part := range event.Content.Parts {
			if part.FunctionCall != nil {
				functionCalls = append(functionCalls, part.FunctionCall)
			}
		}
		if len(functionCalls) == 0 {
			continue
		}

		toolsToResume := make(map[string]bool)
		for _, fc := range functionCalls {
			if !requestEucFunctionCallIDs[fc.ID] {
				continue
			}
			// Extract function_call_id from args (the original tool that requested auth)
			if args := fc.Args; args != nil {
				if fcID, ok := args["function_call_id"].(string); ok {
					toolsToResume[fcID] = true
				}
			}
		}

		if len(toolsToResume) == 0 {
			continue
		}

		// Found the system long running request euc function call
		// Now looking for original function call that requests euc (Python lines 103-129)
		for j := i - 1; j >= 0; j-- {
			originalEvent := events.At(j)
			if originalEvent.Content == nil {
				continue
			}

			var originalFunctionCalls []*genai.FunctionCall
			for _, part := range originalEvent.Content.Parts {
				if part.FunctionCall != nil {
					originalFunctionCalls = append(originalFunctionCalls, part.FunctionCall)
				}
			}
			if len(originalFunctionCalls) == 0 {
				continue
			}

			// Check if any function call matches our tools_to_resume
			hasMatch := false
			for _, fc := range originalFunctionCalls {
				if toolsToResume[fc.ID] {
					hasMatch = true
					break
				}
			}

			if hasMatch {
				// Found the original event containing function calls to resume
				result.ToolIdsToResume = toolsToResume
				result.OriginalEvent = originalEvent
				result.CredentialsStored = true
				CurrentAuthPreprocessorResult = result
				return nil
			}
		}
		return nil
	}

	return nil
}

// parseAuthConfigFromMap converts any map-like auth_config payload into auth.AuthConfig.
func parseAuthConfigFromMap(data any) (*auth.AuthConfig, error) {
	var config auth.AuthConfig
	if err := decodeSnakeCompatibleMap(data, &config, "auth_config"); err != nil {
		return nil, err
	}
	return &config, nil
}

// parseAuthCredentialFromMap converts any map-like auth credential payload into auth.AuthCredential.
func parseAuthCredentialFromMap(data any) (*auth.AuthCredential, error) {
	var cred auth.AuthCredential
	if err := decodeSnakeCompatibleMap(data, &cred, "credential"); err != nil {
		return nil, err
	}
	return &cred, nil
}

func decodeSnakeCompatibleMap(data any, target any, kind string) error {
	dataMap, ok := data.(map[string]any)
	if !ok {
		return fmt.Errorf("%s is not a map", kind)
	}
	decoder, err := mapstructure.NewDecoder(&mapstructure.DecoderConfig{
		TagName:          "json",
		Result:           target,
		WeaklyTypedInput: true,
		MatchName: func(mapKey, fieldName string) bool {
			return canonicalFieldName(mapKey) == canonicalFieldName(fieldName)
		},
	})
	if err != nil {
		return fmt.Errorf("failed to build decoder: %w", err)
	}
	if err := decoder.Decode(dataMap); err != nil {
		return fmt.Errorf("failed to decode %s: %w", kind, err)
	}
	return nil
}

func canonicalFieldName(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if r == '_' || r == '-' {
			continue
		}
		b.WriteRune(unicode.ToLower(r))
	}
	return b.String()
}

func nlPlanningResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error {
	// TODO: implement (adk-python src/google/adk/_nl_planning.py)
	return nil
}

func codeExecutionResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error {
	// TODO: implement (adk-python src/google/adk_code_execution.py)
	return nil
}
