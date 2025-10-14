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
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func NewCallbackContext(ctx agent.InvocationContext) agent.CallbackContext {
	return newCallbackContext(ctx)
}

func newCallbackContext(ctx agent.InvocationContext) *callbackContext {
	rCtx := NewReadonlyContext(ctx)
	return &callbackContext{
		ReadonlyContext: rCtx,
		invocationCtx:   ctx,
		eventActions:    &session.EventActions{},
	}
}

// TODO: unify with agent.callbackContext

type callbackContext struct {
	agent.ReadonlyContext
	invocationCtx agent.InvocationContext
	eventActions  *session.EventActions
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

func (c *callbackContext) Artifacts() agent.Artifacts {
	return c.invocationCtx.Artifacts()
}

func (c *callbackContext) InvocationID() string {
	return c.invocationCtx.InvocationID()
}

func (c *callbackContext) UserContent() *genai.Content {
	return c.invocationCtx.UserContent()
}
