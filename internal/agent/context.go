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

	"google.golang.org/adk/artifact"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// ContextBase carries information necessary in constructing
// [agent.ReadonlyContext] and [agent.CallbackContext].
type ContextBase interface {
	Context() context.Context
	AgentName() string
	InvocationID() string
	Branch() string
	Session() session.Session
	UserContent() *genai.Content

	Artifacts() artifact.Artifacts
}

// readonlyContext is an implementation of [agent.ReadonlyContext].
type readonlyContext struct {
	context.Context
	wrapped ContextBase
}

// callbackContext is an implementation of [agent.CallbackContext].
type callbackContext struct {
	ReadonlyContext
	wrapped ContextBase
	actions *session.EventActions
}

// NewReadonlyContext returns a new [agent.ReadonlyContext]
// created with the information from [ContextBase].
func NewReadonlyContext(b ContextBase) ReadonlyContext {
	return &readonlyContext{
		Context: b.Context(),
		wrapped: b,
	}
}

// NewCallbackContext returns a new [agent.CallbackContext]
// created with the information from [ContextBase] and the given [session.EventActions].
func NewCallbackContext(b ContextBase, actions *session.EventActions) CallbackContext {
	rCtx := NewReadonlyContext(b)
	return &callbackContext{
		ReadonlyContext: rCtx,
		wrapped:         b,
		actions:         actions,
	}
}

// ReadonlyContext is [agent.ReadonlyContext].
// Type assignability is tested from the agent package.
type ReadonlyContext interface {
	context.Context

	UserContent() *genai.Content
	InvocationID() string
	AgentName() string
	ReadonlyState() session.ReadonlyState

	UserID() string
	AppName() string
	SessionID() string
	Branch() string
}

// CallbackContext is [agent.CallbackContext].
// Type assignability is tested from the agent package.
type CallbackContext interface {
	ReadonlyContext

	Artifacts() artifact.Artifacts
	State() session.State
	Actions() *session.EventActions
}

func (c *readonlyContext) AppName() string {
	return c.wrapped.Session().AppName()
}

func (c *readonlyContext) Branch() string {
	return c.wrapped.Branch()
}

func (c *readonlyContext) SessionID() string {
	return c.wrapped.Session().ID()
}

func (c *readonlyContext) UserID() string {
	return c.wrapped.Session().UserID()
}

func (c *readonlyContext) AgentName() string {
	return c.wrapped.AgentName()
}

func (c *readonlyContext) ReadonlyState() session.ReadonlyState {
	return c.wrapped.Session().State()
}

func (c *readonlyContext) InvocationID() string {
	return c.wrapped.InvocationID()
}

func (c *readonlyContext) UserContent() *genai.Content {
	return c.wrapped.UserContent()
}

func (c *callbackContext) Artifacts() artifact.Artifacts {
	return c.wrapped.Artifacts()
}

func (c *callbackContext) State() session.State {
	return c.wrapped.Session().State()
}

func (c *callbackContext) Actions() *session.EventActions {
	return c.actions
}
