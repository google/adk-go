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
	"iter"

	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type callbackContext struct {
	ReadonlyContext
	artifacts         *internalArtifacts
	invocationContext InvocationContext
	actions           *session.EventActions
}

func (c *callbackContext) Artifacts() Artifacts {
	return c.artifacts
}

func (c *callbackContext) State() session.State {
	return &callbackContextState{ctx: c}
}

func (c *callbackContext) AgentName() string {
	return c.invocationContext.Agent().Name()
}

func (c *callbackContext) ReadonlyState() session.ReadonlyState {
	return c.invocationContext.Session().State()
}

func (c *callbackContext) InvocationID() string {
	return c.invocationContext.InvocationID()
}

func (c *callbackContext) UserContent() *genai.Content {
	return c.invocationContext.UserContent()
}

// AppName implements CallbackContext.
func (c *callbackContext) AppName() string {
	return c.invocationContext.Session().AppName()
}

// Branch implements CallbackContext.
func (c *callbackContext) Branch() string {
	return c.invocationContext.Branch()
}

// SessionID implements CallbackContext.
func (c *callbackContext) SessionID() string {
	return c.invocationContext.Session().ID()
}

// UserID implements CallbackContext.
func (c *callbackContext) UserID() string {
	return c.invocationContext.Session().UserID()
}

var _ CallbackContext = (*callbackContext)(nil)

// GetEventActions returns the internal EventActions for framework use
// This is needed for internal framework functionality
func (c *callbackContext) GetEventActions() *session.EventActions {
	return c.actions
}

type callbackContextState struct {
	ctx *callbackContext
}

func (c *callbackContextState) Get(key string) (any, error) {
	if c.ctx.actions != nil && c.ctx.actions.StateDelta != nil {
		if val, ok := c.ctx.actions.StateDelta[key]; ok {
			return val, nil
		}
	}
	return c.ctx.invocationContext.Session().State().Get(key)
}

func (c *callbackContextState) Set(key string, val any) error {
	if c.ctx.actions != nil && c.ctx.actions.StateDelta != nil {
		c.ctx.actions.StateDelta[key] = val
	}
	return c.ctx.invocationContext.Session().State().Set(key, val)
}

func (c *callbackContextState) All() iter.Seq2[string, any] {
	return c.ctx.invocationContext.Session().State().All()
}

// NewCallbackContext creates a CallbackContext with default empty state delta
func NewCallbackContext(ctx InvocationContext) CallbackContext {
	return NewCallbackContextWithDelta(ctx, make(map[string]any))
}

// NewCallbackContextWithDelta creates a CallbackContext with the specified state delta
func NewCallbackContextWithDelta(ctx InvocationContext, stateDelta map[string]any) CallbackContext {
	rCtx := NewReadonlyContext(ctx)
	eventActions := &session.EventActions{StateDelta: stateDelta}
	return &callbackContext{
		ReadonlyContext:   rCtx,
		invocationContext: ctx,
		actions:           eventActions,
		artifacts: &internalArtifacts{
			Artifacts:    ctx.Artifacts(),
			eventActions: eventActions,
		},
	}
}
