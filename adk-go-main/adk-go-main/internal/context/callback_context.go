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

package context

import (
	"context"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/session"
)

type internalArtifacts struct {
	agent.Artifacts
	eventActions *session.EventActions
}

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

func NewCallbackContext(ctx agent.InvocationContext) agent.CallbackContext {
	return newCallbackContext(ctx, make(map[string]any))
}

func NewCallbackContextWithDelta(ctx agent.InvocationContext, stateDelta map[string]any) agent.CallbackContext {
	return newCallbackContext(ctx, stateDelta)
}

func newCallbackContext(ctx agent.InvocationContext, stateDelta map[string]any) *callbackContext {
	rCtx := NewReadonlyContext(ctx)
	eventActions := &session.EventActions{StateDelta: stateDelta}
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

// TODO: unify with agent.callbackContext

type callbackContext struct {
	agent.ReadonlyContext
	artifacts     *internalArtifacts
	invocationCtx agent.InvocationContext
	eventActions  *session.EventActions
}

func (c *callbackContext) Artifacts() agent.Artifacts {
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
