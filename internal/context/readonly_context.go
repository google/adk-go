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

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

func NewReadonlyContext(ctx agent.InvocationContext) agent.ReadonlyContext {
	return &ReadonlyContext{
		Context:           ctx,
		InvocationContext: ctx,
	}
}

type ReadonlyContext struct {
	context.Context
	InvocationContext agent.InvocationContext
}

// AppName implements agent.ReadonlyContext.
func (c *ReadonlyContext) AppName() string {
	return c.InvocationContext.Session().AppName()
}

// Branch implements agent.ReadonlyContext.
func (c *ReadonlyContext) Branch() string {
	return c.InvocationContext.Branch()
}

// SessionID implements agent.ReadonlyContext.
func (c *ReadonlyContext) SessionID() string {
	return c.InvocationContext.Session().ID()
}

// UserID implements agent.ReadonlyContext.
func (c *ReadonlyContext) UserID() string {
	return c.InvocationContext.Session().UserID()
}

func (c *ReadonlyContext) AgentName() string {
	return c.InvocationContext.Agent().Name()
}

func (c *ReadonlyContext) ReadonlyState() session.ReadonlyState {
	return c.InvocationContext.Session().State()
}

func (c *ReadonlyContext) InvocationID() string {
	return c.InvocationContext.InvocationID()
}

func (c *ReadonlyContext) UserContent() *genai.Content {
	return c.InvocationContext.UserContent()
}
