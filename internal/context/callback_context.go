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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type internalArtifacts struct {
	agent.Artifacts
	ctx *callbackContext
}

func (ia *internalArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	resp, err := ia.Artifacts.Save(ctx, name, data)
	if err != nil {
		return resp, err
	}
	if ia.ctx.eventActions != nil {
		if ia.ctx.eventActions.ArtifactDelta == nil {
			ia.ctx.eventActions.ArtifactDelta = make(map[string]int64)
		}
		// TODO: RWLock, check the version stored is newer in case multiple tools save the same file.
		ia.ctx.eventActions.ArtifactDelta[name] = resp.Version
	}
	return resp, err
}

func NewCallbackContext(ctx agent.InvocationContext) agent.CallbackContext {
	return newCallbackContext(ctx)
}

func newCallbackContext(ctx agent.InvocationContext) *callbackContext {
	rCtx := NewReadonlyContext(ctx)
	eventActions := &session.EventActions{}
	cbCtx := &callbackContext{
		ReadonlyContext: rCtx,
		invocationCtx:   ctx,
		eventActions:    eventActions,
	}
	cbCtx.artifacts = &internalArtifacts{
		Artifacts: ctx.Artifacts(),
		ctx:       cbCtx,
	}
	return cbCtx
}

// TODO: unify with agent.callbackContext

type callbackContext struct {
	agent.ReadonlyContext
	artifacts     agent.Artifacts
	invocationCtx agent.InvocationContext
	eventActions  *session.EventActions
}

func (c *callbackContext) Artifacts() agent.Artifacts {
	return c.artifacts
}

func (c *callbackContext) AgentName() string {
	return c.invocationCtx.Agent().Name()
}

func (c *callbackContext) Actions() *session.EventActions {
	return c.eventActions
}

func (c *callbackContext) ReadonlyState() session.ReadonlyState {
	return c.invocationCtx.Session().State()
}

func (c *callbackContext) State() session.State {
	return c.invocationCtx.Session().State()
}

func (c *callbackContext) InvocationID() string {
	return c.invocationCtx.InvocationID()
}

func (c *callbackContext) UserContent() *genai.Content {
	return c.invocationCtx.UserContent()
}
