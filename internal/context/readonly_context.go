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
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func NewReadonlyContext(ctx agent.InvocationContext) agent.ReadonlyContext {
	return &readonlyContext{
		Context:           ctx,
		invocationContext: ctx,
	}
}

type readonlyContext struct {
	context.Context
	invocationContext agent.InvocationContext
}

// AppName implements agent.ReadonlyContext.
func (c *readonlyContext) AppName() string {
	return c.invocationContext.Session().AppName()
}

// Branch implements agent.ReadonlyContext.
func (c *readonlyContext) Branch() string {
	return c.invocationContext.Branch()
}

// SessionID implements agent.ReadonlyContext.
func (c *readonlyContext) SessionID() string {
	return c.invocationContext.Session().ID()
}

// UserID implements agent.ReadonlyContext.
func (c *readonlyContext) UserID() string {
	return c.invocationContext.Session().UserID()
}

func (c *readonlyContext) AgentName() string {
	return c.invocationContext.Agent().Name()
}

func (c *readonlyContext) ReadonlyState() session.ReadonlyState {
	return c.invocationContext.Session().State()
}

func (c *readonlyContext) InvocationID() string {
	return c.invocationContext.InvocationID()
}

func (c *readonlyContext) UserContent() *genai.Content {
	return c.invocationContext.UserContent()
}
