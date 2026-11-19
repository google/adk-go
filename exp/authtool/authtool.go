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

package authtool

import (
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// Config represents configuration for an authenticated tool.
//
// The embedded [functiontool.Config] names and describes the tool as it does
// for [functiontool.New].
//
// AuthConfig describes the credential the tool needs. When it is nil or its
// AuthScheme is zero, the tool runs without authentication and the function
// receives a nil credential. The tool reads AuthConfig and never writes to
// it, so one AuthConfig can be shared by many tools and users; the caller
// must not change it after building the tool.
//
// ResponseForAuthRequired is the tool's result when no credential is ready
// and the tool has asked the user for one: a string, which the tool returns
// as {"result": s}, or a map[string]any, returned as it is. When it is nil,
// an empty string or an empty map, the result is
// {"result": "Pending User Authorization."}. Any other type is refused when
// the tool is built.
//
// This is the configuration of adk-python's AuthenticatedFunctionTool
// (tools/authenticated_function_tool.py), which adk-python marks
// experimental.
//
// EXPERIMENTAL: Config is experimental; its API and behavior may change or be removed in a later version.
type Config struct {
	functiontool.Config
	AuthConfig              *toolauth.Config
	ResponseForAuthRequired any
}

// Func is a Go function wrapped in an authenticated tool.
//
// cred receives the procured credential, or nil when the tool has no
// AuthConfig. A tool with an AuthConfig calls the function only once it has a
// credential. authErr receives a soft exchange or refresh error that left cred
// unexchanged or stale; the call goes on with that credential, as in
// adk-python, and the function decides whether to use it, report authErr or
// fail. authErr is nil when the credential is ready.
//
// This is the function adk-python's AuthenticatedFunctionTool calls with its
// credential argument (tools/authenticated_function_tool.py), which adk-python
// marks experimental.
//
// EXPERIMENTAL: Func is experimental; its API and behavior may change or be removed in a later version.
type Func[TArgs, TResults any] func(ctx agent.Context, args TArgs, cred *toolauth.Credential, authErr error) (TResults, error)

// New creates an authenticated function tool.
//
// TArgs and TResults follow the rules of [functiontool.New], and the tool's
// declaration is the one [functiontool.New] would build for them: the
// credential is not one of its parameters. New returns (nil, err) when
// [functiontool.New] refuses cfg or handler's types, when
// ResponseForAuthRequired has another type than those [Config] names, or when
// AuthConfig's AuthScheme is not zero but has no Type.
//
// On each call with an AuthConfig, the tool first gets the credential from
// its [CredentialManager] ([CredentialManager.GetAuthCredential]), before the
// arguments are checked. When that returns no credential, the tool asks the
// user for one ([CredentialManager.RequestCredential]) and returns the
// auth-required result of [Config] without calling handler. An error from
// the manager, from the request, or from handler is returned unchanged.
//
// The tool is safe for concurrent use. Each call's credential reaches only
// that call's handler.
//
// This is adk-python's AuthenticatedFunctionTool
// (tools/authenticated_function_tool.py), which adk-python marks
// experimental.
//
// EXPERIMENTAL: New is experimental; its API and behavior may change or be removed in a later version.
func New[TArgs, TResults any](cfg Config, handler Func[TArgs, TResults]) (tool.Tool, error) {
	inner, err := functiontool.New(cfg.Config, func(ctx agent.Context, args TArgs) (TResults, error) {
		cred, authErr := callAuthFrom(ctx)
		return handler(ctx, args, cred, authErr)
	})
	if err != nil {
		return nil, err
	}
	it, ok := inner.(innerTool)
	if !ok {
		return nil, fmt.Errorf("authtool: functiontool.New returned %T, which lacks a function tool's methods", inner)
	}
	return newAuthTool(it, cfg)
}
