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

// internalArtifacts wraps an Artifacts service so that Save also
// records the new version into the supplied EventActions.ArtifactDelta.
type internalArtifacts struct {
	Artifacts
	eventActions *session.EventActions
}

// Save persists the artifact and records its version in the event's
// ArtifactDelta.
func (ia *internalArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	resp, err := ia.Artifacts.Save(ctx, name, data)
	if err != nil {
		return resp, err
	}
	if ia.eventActions != nil {
		if ia.eventActions.ArtifactDelta == nil {
			ia.eventActions.ArtifactDelta = make(map[string]int64)
		}
		// TODO: RWLock, check the version stored is newer in case multiple tools save the same file.
		ia.eventActions.ArtifactDelta[name] = resp.Version
	}
	return resp, nil
}

// NewCallbackContext returns a CallbackContext with fresh state and
// artifact delta maps. Used by callback dispatchers in the agent
// package.
func NewCallbackContext(ctx InvocationContext) CallbackContext {
	return newCallbackContext(ctx, make(map[string]any), make(map[string]int64))
}

// NewCallbackContextWithDelta returns a CallbackContext that uses the
// supplied delta maps directly (rather than allocating fresh ones).
// Used when the caller already holds the delta maps and needs them
// shared with downstream code.
func NewCallbackContextWithDelta(ctx InvocationContext, stateDelta map[string]any, artifactDelta map[string]int64) CallbackContext {
	return newCallbackContext(ctx, stateDelta, artifactDelta)
}

func newCallbackContext(ctx InvocationContext, stateDelta map[string]any, artifactDelta map[string]int64) *callbackContext {
	rCtx := NewReadonlyContext(ctx)
	eventActions := &session.EventActions{StateDelta: stateDelta, ArtifactDelta: artifactDelta}
	return &callbackContext{
		ReadonlyContext: rCtx,
		invocationCtx:   ctx,
		eventActions:    eventActions,
		artifacts: &internalArtifacts{
			Artifacts:    ctx.Artifacts(),
			eventActions: eventActions,
		},
	}
}

// callbackContext is the canonical implementation of CallbackContext.
// Construct via NewCallbackContext or NewCallbackContextWithDelta.
type callbackContext struct {
	ReadonlyContext
	artifacts     *internalArtifacts
	invocationCtx InvocationContext
	eventActions  *session.EventActions
}

func (c *callbackContext) Artifacts() Artifacts {
	return c.artifacts
}

func (c *callbackContext) AgentName() string {
	return c.invocationCtx.Agent().Name()
}

func (c *callbackContext) ReadonlyState() session.ReadonlyState {
	return c.invocationCtx.Session().State()
}

func (c *callbackContext) State() session.State {
	return &callbackContextState{ctx: c}
}

func (c *callbackContext) InvocationID() string {
	return c.invocationCtx.InvocationID()
}

func (c *callbackContext) UserContent() *genai.Content {
	return c.invocationCtx.UserContent()
}

var _ CallbackContext = (*callbackContext)(nil)

type callbackContextState struct {
	ctx *callbackContext
}

func (c *callbackContextState) Get(key string) (any, error) {
	if c.ctx.eventActions != nil && c.ctx.eventActions.StateDelta != nil {
		if val, ok := c.ctx.eventActions.StateDelta[key]; ok {
			return val, nil
		}
	}
	return c.ctx.invocationCtx.Session().State().Get(key)
}

func (c *callbackContextState) Set(key string, val any) error {
	if c.ctx.eventActions != nil && c.ctx.eventActions.StateDelta != nil {
		c.ctx.eventActions.StateDelta[key] = val
	}
	return c.ctx.invocationCtx.Session().State().Set(key, val)
}

func (c *callbackContextState) All() iter.Seq2[string, any] {
	return c.ctx.invocationCtx.Session().State().All()
}
