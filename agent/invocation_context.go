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

package agent

import (
	"context"
	"errors"

	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// ErrOutsideToolCall is returned by the action methods on a Context
// when invoked from a non-tool call site (e.g. an agent callback or
// instruction provider). Pollable accessors return zero values in
// the same situation; only mutating actions surface the misuse as
// an error.
//
// Callers that want to handle this case explicitly can match with
// errors.Is(err, agent.ErrOutsideToolCall).
var ErrOutsideToolCall = errors.New("operation requires a tool-call context; called from a non-tool site")

// InvocationContextParams holds the data used to construct a new
// InvocationContext via NewInvocationContext.
type InvocationContextParams struct {
	Artifacts Artifacts
	Memory    Memory
	Session   session.Session

	Branch string
	Agent  Agent

	UserContent   *genai.Content
	RunConfig     *RunConfig
	EndInvocation bool
	InvocationID  string
}

// NewInvocationContext returns a new InvocationContext built from the
// given Go context and params. If params.InvocationID is empty, a new
// unique identifier of the form "e-<uuid>" is generated.
func NewInvocationContext(ctx context.Context, params InvocationContextParams) InvocationContext {
	if params.InvocationID == "" {
		params.InvocationID = "e-" + uuid.NewString()
	}
	return &invocationContextImpl{
		Context: ctx,
		params:  params,
	}
}

// invocationContextImpl is the canonical, in-process implementation of
// InvocationContext. It is unexported because callers should depend on
// the InvocationContext interface and construct values via
// NewInvocationContext.
type invocationContextImpl struct {
	context.Context

	params InvocationContextParams
}

func (c *invocationContextImpl) Artifacts() Artifacts {
	return c.params.Artifacts
}

func (c *invocationContextImpl) Agent() Agent {
	return c.params.Agent
}

func (c *invocationContextImpl) Branch() string {
	return c.params.Branch
}

func (c *invocationContextImpl) InvocationID() string {
	return c.params.InvocationID
}

func (c *invocationContextImpl) Memory() Memory {
	return c.params.Memory
}

func (c *invocationContextImpl) Session() session.Session {
	return c.params.Session
}

func (c *invocationContextImpl) UserContent() *genai.Content {
	return c.params.UserContent
}

func (c *invocationContextImpl) RunConfig() *RunConfig {
	return c.params.RunConfig
}

func (c *invocationContextImpl) EndInvocation() {
	c.params.EndInvocation = true
}

func (c *invocationContextImpl) Ended() bool {
	return c.params.EndInvocation
}

func (c *invocationContextImpl) WithContext(ctx context.Context) InvocationContext {
	newCtx := *c
	newCtx.Context = ctx
	return &newCtx
}

// WithAgent returns a copy of c with the Agent param overridden. The
// embedded context.Context and all other params are shared with the
// receiver. See Context.WithAgent for the contract.
func (c *invocationContextImpl) WithAgent(a Agent) Context {
	newCtx := *c
	newCtx.params.Agent = a
	return &newCtx
}

// AgentName returns the name of the active agent, or "" if no agent is set.
func (c *invocationContextImpl) AgentName() string {
	if c.params.Agent == nil {
		return ""
	}
	return c.params.Agent.Name()
}

// UserID returns the session's user ID, or "" if no session is set.
func (c *invocationContextImpl) UserID() string {
	if c.params.Session == nil {
		return ""
	}
	return c.params.Session.UserID()
}

// AppName returns the session's app name, or "" if no session is set.
func (c *invocationContextImpl) AppName() string {
	if c.params.Session == nil {
		return ""
	}
	return c.params.Session.AppName()
}

// SessionID returns the session ID, or "" if no session is set.
func (c *invocationContextImpl) SessionID() string {
	if c.params.Session == nil {
		return ""
	}
	return c.params.Session.ID()
}

// State returns the session's read-write state. Returns nil when no
// session is set on the context.
func (c *invocationContextImpl) State() session.State {
	if c.params.Session == nil {
		return nil
	}
	return c.params.Session.State()
}

// ReadonlyState returns a read-only view of the session state.
// Returns nil when no session is set.
func (c *invocationContextImpl) ReadonlyState() session.ReadonlyState {
	if c.params.Session == nil {
		return nil
	}
	return c.params.Session.State()
}

// SearchMemory delegates to the configured Memory service. Returns an
// error when no Memory service is configured.
func (c *invocationContextImpl) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	if c.params.Memory == nil {
		return nil, errors.New("no Memory service configured on this Context")
	}
	return c.params.Memory.SearchMemory(ctx, query)
}

// FunctionCallID returns "" because a bare InvocationContext is not a
// tool-call site. A wrapping tool.Context overrides this.
func (c *invocationContextImpl) FunctionCallID() string { return "" }

// Actions returns nil because a bare InvocationContext is not a
// tool-call site. A wrapping tool.Context overrides this.
func (c *invocationContextImpl) Actions() *session.EventActions { return nil }

// ToolConfirmation returns nil because a bare InvocationContext is not
// a tool-call site. A wrapping tool.Context overrides this.
func (c *invocationContextImpl) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }

// RequestConfirmation returns ErrOutsideToolCall because
// human-in-the-loop confirmation is meaningful only inside a tool
// call. A wrapping tool.Context overrides this.
func (c *invocationContextImpl) RequestConfirmation(_ string, _ any) error {
	return ErrOutsideToolCall
}

var _ Context = (*invocationContextImpl)(nil)
