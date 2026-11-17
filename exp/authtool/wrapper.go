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
	"context"
	"fmt"
	"maps"
	"reflect"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/adkcontext"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// defaultAuthRequired is adk-python's answer while the user has not
// authorized (tools/authenticated_function_tool.py:91).
const defaultAuthRequired = "Pending User Authorization."

// innerTool is the tool an authTool wraps: one built by functiontool.New, or
// a baseTool. It has every method the LLM flow calls on a function tool.
type innerTool interface {
	Name() string
	Description() string
	IsLongRunning() bool
	Declaration() *genai.FunctionDeclaration
	ProcessRequest(ctx agent.Context, req *model.LLMRequest) error
	Run(ctx agent.Context, args any) (map[string]any, error)
	RunRaw(ctx agent.Context, args any) (any, error)
	ResponseScheduling() genai.FunctionResponseScheduling
	RunsSequentially() bool
}

// authTool gets the call's credential, then runs the inner tool with a
// context that carries it (see callContext). It holds no per-call state, so
// one authTool serves concurrent calls of many users.
type authTool struct {
	inner innerTool
	// manager is nil when the tool has no auth scheme; the inner tool then
	// runs with a nil credential.
	manager      *CredentialManager
	authRequired map[string]any
}

// newAuthTool wraps inner. It fails when cfg names a scheme the manager
// cannot be built for, or an auth-required response that is neither a string
// nor a map[string]any.
func newAuthTool(inner innerTool, cfg Config) (tool.Tool, error) {
	authRequired, err := authRequiredResult(cfg.ResponseForAuthRequired)
	if err != nil {
		return nil, err
	}
	t := &authTool{inner: inner, authRequired: authRequired}
	if cfg.AuthConfig == nil || reflect.ValueOf(cfg.AuthConfig.AuthScheme).IsZero() {
		return t, nil
	}
	m, err := NewCredentialManager(cfg.AuthConfig, nil)
	if err != nil {
		return nil, fmt.Errorf("authtool: tool %q: %w", inner.Name(), err)
	}
	t.manager = m
	return t, nil
}

// authRequiredResult is the result for response, which adk-python returns as
// "response or default" and the flow wraps in {"result": ...} when it is not
// a dict: so an empty string or map gives the default.
func authRequiredResult(response any) (map[string]any, error) {
	switch r := response.(type) {
	case nil:
	case string:
		if r != "" {
			return map[string]any{"result": r}, nil
		}
	case map[string]any:
		if len(r) > 0 {
			return r, nil
		}
	default:
		return nil, fmt.Errorf("authtool: ResponseForAuthRequired must be a string or a map[string]any, got %T", response)
	}
	return map[string]any{"result": defaultAuthRequired}, nil
}

func (t *authTool) Name() string { return t.inner.Name() }

func (t *authTool) Description() string { return t.inner.Description() }

func (t *authTool) IsLongRunning() bool { return t.inner.IsLongRunning() }

func (t *authTool) Declaration() *genai.FunctionDeclaration { return t.inner.Declaration() }

func (t *authTool) ResponseScheduling() genai.FunctionResponseScheduling {
	return t.inner.ResponseScheduling()
}

func (t *authTool) RunsSequentially() bool { return t.inner.RunsSequentially() }

// ProcessRequest declares the inner tool to the model. The inner tool also
// registers itself in req.Tools, the table the flow calls tools from, so t
// takes its place there, as tool.WithConfirmation's wrapper does; otherwise
// the flow would run the inner tool without a credential.
func (t *authTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if err := t.inner.ProcessRequest(ctx, req); err != nil {
		return err
	}
	req.Tools[t.Name()] = t
	return nil
}

// Run gets the credential and runs the inner tool's Run with it.
func (t *authTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	callCtx, early, err := t.authorize(ctx)
	if err != nil || early != nil {
		return early, err
	}
	return t.inner.Run(callCtx, args)
}

// RunRaw gets the credential and runs the inner tool's RunRaw with it.
func (t *authTool) RunRaw(ctx agent.Context, args any) (any, error) {
	callCtx, early, err := t.authorize(ctx)
	if err != nil {
		return nil, err
	}
	if early != nil {
		return early, nil
	}
	return t.inner.RunRaw(callCtx, args)
}

// authorize returns the context the inner tool runs with. When no credential
// is ready it asks the user for one and returns early, the auth-required
// result, instead. err is a failure of the manager or of the request.
func (t *authTool) authorize(ctx agent.Context) (callCtx agent.Context, early map[string]any, err error) {
	if t.manager == nil {
		return withCallAuth(ctx, &callAuth{}), nil, nil
	}
	cred, authErr := t.manager.GetAuthCredential(ctx)
	if cred == nil {
		if authErr != nil {
			return nil, nil, authErr
		}
		if err := t.manager.RequestCredential(ctx); err != nil {
			return nil, nil, err
		}
		return nil, maps.Clone(t.authRequired), nil
	}
	return withCallAuth(ctx, &callAuth{cred: cred, err: authErr}), nil, nil
}

// callAuthKey is the context key of a call's callAuth.
type callAuthKey struct{}

// callAuth is what one call's function receives besides its arguments.
type callAuth struct {
	cred *toolauth.Credential
	err  error
}

// withCallAuth returns ctx carrying auth under callAuthKey. For a tool
// context ADK built, it derives one with WithDelta, which keeps the call's
// function call ID and event actions, so agent.MergeArtifactDelta and
// agent.RenderUIWidget record into the call's function response event as
// they do in a plain function tool. A context whose WithDelta does not keep
// the value, such as one of the caller's own, gets callContext instead.
func withCallAuth(ctx agent.Context, auth *callAuth) agent.Context {
	v := context.WithValue(ctx, callAuthKey{}, auth)
	derived := ctx.WithDelta(&agent.CommonContextDelta{
		InvocationContextDelta: &agent.InvocationContextDelta{Context: &v},
	})
	if derived != nil && derived.Value(callAuthKey{}) == auth {
		return derived
	}
	return &callContext{Context: ctx, auth: auth}
}

// callContext is the context the inner tool runs with when ctx's WithDelta
// does not carry the call's callAuth: ctx, carrying it under callAuthKey. It
// is built as the flow's own tool context decorator (cancelledToolContext in
// internal/llminternal) is: the identity marker, because Value reaches the
// tool context, and RenderUIWidget, which the embedding would otherwise hide.
type callContext struct {
	agent.Context
	adkcontext.Marker
	auth *callAuth
}

// RenderUIWidget implements [agent.UIWidgetRenderer] for the tool context it
// embeds.
func (c *callContext) RenderUIWidget(widget session.UIWidget) error {
	return agent.RenderUIWidget(c.Context, widget)
}

func (c *callContext) Value(key any) any {
	// Fails closed rather than dereferencing, as the identity procedure can
	// ask from inside http.RoundTripper, where net/http does not recover.
	if c == nil || c.Context == nil {
		return nil
	}
	if key == (callAuthKey{}) {
		return c.auth
	}
	return c.Context.Value(key)
}

var _ adkcontext.Source = (*callContext)(nil)

// callAuthFrom returns the credential and error authorize gave the call of
// ctx, or nils when ctx carries none.
func callAuthFrom(ctx agent.Context) (*toolauth.Credential, error) {
	a, ok := ctx.Value(callAuthKey{}).(*callAuth)
	if !ok {
		return nil, nil
	}
	return a.cred, a.err
}
