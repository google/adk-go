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
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/artifact"
	"google.golang.org/adk/session"
)

// NewCallbackContext returns a CallbackContext backed by the given invocation
// context. The returned context is suitable for passing to user agent / model /
// tool callbacks.
//
// If trackArtifactDeltas is true, the returned context's Artifacts().Save(...)
// wrapper records each saved artifact's version into the underlying
// EventActions.ArtifactDelta so the resulting Event reflects the saves.
//
// stateDelta and artifactDelta may be nil; if non-nil they are used as the
// initial backing maps for the EventActions on this context (allowing callers
// that aggregate deltas across multiple callbacks to share storage).
//
// The returned concrete type carries the underlying *session.EventActions; use
// EventActionsOf to retrieve it.
func NewCallbackContext(ic InvocationContext, trackArtifactDeltas bool, stateDelta map[string]any, artifactDelta map[string]int64) CallbackContext {
	if stateDelta == nil {
		stateDelta = make(map[string]any)
	}
	if artifactDelta == nil {
		artifactDelta = make(map[string]int64)
	}
	actions := &session.EventActions{StateDelta: stateDelta, ArtifactDelta: artifactDelta}
	cc := &callbackContext{
		Context:           ic,
		invocationContext: ic,
		actions:           actions,
	}
	if trackArtifactDeltas {
		cc.artifacts = &trackedArtifacts{Artifacts: ic.Artifacts(), actions: actions}
	} else {
		cc.artifacts = ic.Artifacts()
	}
	return cc
}

// EventActionsOf returns the *session.EventActions backing a CallbackContext
// produced by NewCallbackContext, or nil if the context was not produced by
// NewCallbackContext.
func EventActionsOf(c CallbackContext) *session.EventActions {
	cc, ok := c.(*callbackContext)
	if !ok {
		return nil
	}
	return cc.actions
}

// callbackContext is the single concrete implementation of CallbackContext.
//
// Its Artifacts() return value may optionally wrap the underlying Artifacts to
// record saved artifact versions into the EventActions.ArtifactDelta, depending
// on the trackArtifactDeltas flag passed to NewCallbackContext.
type callbackContext struct {
	context.Context
	invocationContext InvocationContext
	artifacts         Artifacts
	actions           *session.EventActions
}

func (c *callbackContext) AgentName() string {
	return c.invocationContext.Agent().Name()
}

func (c *callbackContext) ReadonlyState() session.ReadonlyState {
	return c.invocationContext.Session().State()
}

func (c *callbackContext) State() session.State {
	return &callbackContextState{ctx: c}
}

func (c *callbackContext) Artifacts() Artifacts {
	return c.artifacts
}

func (c *callbackContext) InvocationID() string {
	return c.invocationContext.InvocationID()
}

func (c *callbackContext) UserContent() *genai.Content {
	return c.invocationContext.UserContent()
}

func (c *callbackContext) AppName() string {
	return c.invocationContext.Session().AppName()
}

func (c *callbackContext) Branch() string {
	return c.invocationContext.Branch()
}

func (c *callbackContext) SessionID() string {
	return c.invocationContext.Session().ID()
}

func (c *callbackContext) UserID() string {
	return c.invocationContext.Session().UserID()
}

var _ CallbackContext = (*callbackContext)(nil)

// callbackContextState is a session.State implementation backed by the
// callback context's EventActions.StateDelta and the underlying session state.
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

// trackedArtifacts wraps an Artifacts to record each successful Save into the
// supplied EventActions.ArtifactDelta.
type trackedArtifacts struct {
	Artifacts
	actions *session.EventActions
}

func (a *trackedArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	resp, err := a.Artifacts.Save(ctx, name, data)
	if err != nil {
		return resp, err
	}
	if a.actions != nil {
		if a.actions.ArtifactDelta == nil {
			a.actions.ArtifactDelta = make(map[string]int64)
		}
		// TODO: RWLock, check the version stored is newer in case multiple tools save the same file.
		a.actions.ArtifactDelta[name] = resp.Version
	}
	return resp, nil
}
