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

// Package authflow lets a tool ask the end user for a credential, such as an
// OAuth2 sign-in, and read what the user's answer produced. It is
// experimental: it may change or be removed in a later version.
//
// A tool calls [GetAuthResponse] first. When that returns nil, the tool calls
// [RequestCredential] and returns a result saying it is waiting. ADK then
// sends the client an adk_request_credential call carrying the sign-in link
// and ends the invocation. The client signs the user in and answers that call
// in a new message; ADK stores the answer and runs the tool call again, and
// this time [GetAuthResponse] returns the credential:
//
//	cred, err := authflow.GetAuthResponse(ctx, cfg)
//	if err != nil {
//		return nil, err
//	}
//	if cred == nil {
//		if err := authflow.RequestCredential(ctx, cfg); err != nil {
//			return nil, err
//		}
//		return map[string]any{"status": "waiting for the user to sign in"}, nil
//	}
//	// Use cred.OAuth2.AccessToken.
//
// These are adk-python's ToolContext.request_credential and
// ToolContext.get_auth_response. They are functions rather than methods on
// [agent.Context] because a new method on that interface would break every
// other implementation of it.
package authflow

import (
	"errors"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/authhandler"
	"google.golang.org/adk/v2/tool/toolauth"
)

// RequestCredential asks the end user for the credential cfg describes, on
// behalf of the tool call ctx belongs to. It records the request on the tool
// call's response; the request carries the sign-in link for OAuth2 and OpenID
// Connect, and none of the secrets cfg holds. ctx must be a tool's context.
func RequestCredential(ctx agent.Context, cfg *toolauth.Config) error {
	id := ctx.FunctionCallID()
	if id == "" {
		return errors.New("authflow: RequestCredential needs the context of a tool call")
	}
	req, err := authhandler.GenerateAuthRequest(cfg)
	if err != nil {
		return err
	}
	actions := ctx.Actions()
	if actions.RequestedAuthConfigs == nil {
		actions.RequestedAuthConfigs = map[string]toolauth.Config{}
	}
	actions.RequestedAuthConfigs[id] = *req
	return nil
}

// GetAuthResponse returns the credential the user's answer to an earlier
// [RequestCredential] produced for cfg, or nil when there is none yet. An
// OAuth2 authorization code is exchanged for tokens here, with cfg's client
// secret, and the tokens replace the code in session state.
//
// ADK keeps the answer under a "temp:" key, so it lasts for the rest of the
// invocation. A tool that needs the credential in a later invocation, such as
// one that goes on to ask for a second credential, stores it in state under
// cfg's key ([toolauth.Config.Key]), where GetAuthResponse looks next.
//
// The exchanged tokens are written through ctx's state, so they also appear
// in the state delta of the tool call's response event, where plugins and code
// reading the runner's events see them. No session service persists "temp:"
// keys, and the REST server strips credential secrets from the events it
// returns.
func GetAuthResponse(ctx agent.Context, cfg *toolauth.Config) (*toolauth.Credential, error) {
	return authhandler.GetAuthResponse(ctx, ctx.State(), cfg)
}
