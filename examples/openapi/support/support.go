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

// Package support contains helper routines for OpenAPI-based examples.
package support

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/mitchellh/mapstructure"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/auth"
	"google.golang.org/adk/examples/openapi/oauth2handler"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// RunInteractive launches an interactive CLI loop that forwards user input to
// the provided runner and handles OAuth requests emitted via tools.
func RunInteractive(
	ctx context.Context,
	appName string,
	userID string,
	intro string,
	r *runner.Runner,
	sessionService session.Service,
	oauth2Handler *oauth2handler.Handler,
) error {
	if appName == "" {
		appName = "openapi_example"
	}
	if userID == "" {
		userID = "user123"
	}
	if intro == "" {
		intro = "Agent Assistant (type 'quit' to exit)"
	}

	scanner := bufio.NewScanner(os.Stdin)
	sess, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}

	fmt.Println(intro)

	for {
		fmt.Print("User -> ")
		if !scanner.Scan() {
			break
		}

		userInput := strings.TrimSpace(scanner.Text())
		if userInput == "" {
			continue
		}
		if userInput == "quit" || userInput == "exit" {
			break
		}

		msg := &genai.Content{
			Parts: []*genai.Part{{Text: userInput}},
			Role:  "user",
		}

		if err := runAgentWithAuth(ctx, r, sess.Session, msg, oauth2Handler); err != nil {
			fmt.Printf("Error: %v\n\n", err)
		}

		fmt.Println()
	}

	return nil
}

func runAgentWithAuth(
	ctx context.Context,
	r *runner.Runner,
	sess session.Session,
	msg *genai.Content,
	oauth2Handler *oauth2handler.Handler,
) error {
	var pendingAuthCalls []*genai.FunctionCall

	for event, err := range r.Run(ctx, sess.UserID(), sess.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			return err
		}

		if event.Content != nil {
			for _, part := range event.Content.Parts {
				if part.FunctionCall != nil && part.FunctionCall.Name == auth.RequestEUCFunctionCallName {
					pendingAuthCalls = append(pendingAuthCalls, part.FunctionCall)
				}
			}
		}

		if event.Content != nil {
			for _, part := range event.Content.Parts {
				if part.Text != "" && !event.LLMResponse.Partial {
					fmt.Printf("Agent -> %s\n", part.Text)
				}
			}
		}
	}

	if len(pendingAuthCalls) == 0 {
		return nil
	}

	fmt.Println("\nOAuth2 authorization required.")

	var authResponseParts []*genai.Part
	for _, fc := range pendingAuthCalls {
		authConfig := parseAuthConfigFromFunctionCall(fc)
		if authConfig == nil {
			continue
		}

		fmt.Printf("Processing credential request: %s\n", authConfig.CredentialKey)

		authCred, err := oauth2Handler.HandleAuthRequest(ctx, authConfig)
		if err != nil {
			fmt.Printf("Authorization failed: %v\n", err)
			continue
		}

		authResponseParts = append(authResponseParts, &genai.Part{
			FunctionResponse: &genai.FunctionResponse{
				ID:   fc.ID,
				Name: fc.Name,
				Response: map[string]any{
					"auth_config": map[string]any{
						"credential_key":            authConfig.CredentialKey,
						"exchanged_auth_credential": authCred,
					},
				},
			},
		})
	}

	if len(authResponseParts) == 0 {
		return nil
	}

	return runAgentWithAuth(
		ctx,
		r,
		sess,
		&genai.Content{
			Role:  "tool",
			Parts: authResponseParts,
		},
		oauth2Handler,
	)
}

// parseAuthConfigFromFunctionCall converts an adk_request_credential call into
// an AuthConfig structure expected by the auth manager.
func parseAuthConfigFromFunctionCall(fc *genai.FunctionCall) *auth.AuthConfig {
	if fc == nil || fc.Args == nil {
		return nil
	}

	rawConfig, ok := fc.Args["auth_config"]
	if !ok || rawConfig == nil {
		return nil
	}

	switch cfg := rawConfig.(type) {
	case *auth.AuthConfig:
		return cfg.Copy()
	case auth.AuthConfig:
		copyCfg := cfg
		return &copyCfg
	case map[string]any:
		var decoded auth.AuthConfig
		decoder, err := mapstructure.NewDecoder(&mapstructure.DecoderConfig{
			TagName:          "json",
			Result:           &decoded,
			WeaklyTypedInput: true,
		})
		if err != nil {
			return nil
		}
		if err := decoder.Decode(cfg); err != nil {
			return nil
		}
		return &decoded
	default:
		return nil
	}
}
