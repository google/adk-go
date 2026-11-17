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

// Package authtool builds tools that get a credential before they run, and
// the credential manager that gets it.
//
// [New] wraps a Go function in a tool, as [functiontool.New] does, and gives
// the function the credential its [toolauth.Config] describes. Before each
// call the tool asks a [CredentialManager] for the credential. When there is
// none yet, such as before the user has signed in with OAuth2, the tool asks
// the user for it and returns a result saying it is waiting, without running
// the function. ADK then sends the client an adk_request_credential call;
// when the client answers it, ADK runs the tool call again and the function
// gets the credential:
//
//	type args struct {
//		City string `json:"city"`
//	}
//	weather, err := authtool.New(authtool.Config{
//		Config: functiontool.Config{Name: "get_weather", Description: "Returns the weather in a city."},
//		AuthConfig: &toolauth.Config{
//			AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
//			RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: key},
//		},
//	}, func(ctx agent.Context, a args, cred *toolauth.Credential, authErr error) (map[string]any, error) {
//		// Send the request with cred.APIKey.
//	})
//
// [NewBaseTool] does the same for a tool that is not built from a typed Go
// function, and [CredentialManager] can be used on its own by a tool that
// gets its credential itself.
//
// A [CredentialManager] loads the credential from the runner's credential
// service, reads the user's answer to an earlier request with
// [authflow.GetAuthResponse], exchanges an OAuth2 or OpenID Connect
// credential for tokens, refreshes tokens that are about to expire, and saves
// the result in the credential service. Calls for the same app, user and
// credential key run one at a time, in every manager of the process.
//
// These are adk-python's AuthenticatedFunctionTool, BaseAuthenticatedTool,
// CredentialManager and its OAuth2 credential exchanger, which adk-python
// marks experimental. Custom auth schemes and their providers, OAuth2
// endpoint discovery from an issuer URL and service-account credentials are
// not supported: the manager returns an error that names them.
//
// EXPERIMENTAL: This package is experimental; its API and behavior may change or be removed in a later version.
package authtool
