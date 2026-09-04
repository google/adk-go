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

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/adkcontext"
	"google.golang.org/adk/v2/session"
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

// Value implements context.Context. It answers the ADK identity key for the
// invocation this context speaks for; every other key delegates to the embedded
// context, preserving existing behavior.
//
// The identity key has to be answered here rather than left to the embedded
// context, even though the two are the same invocation. Forwarding asks that
// invocation directly, and one implemented outside the agent package cannot
// override a key it cannot name, so its embedded parent replies — with the
// enclosing invocation's user, whose credential would then be minted for a call
// they never made. agent.Promote applies the one identity procedure instead,
// which reads the invocation's own session and falls back only to a context type
// agent itself owns. It returns the argument unchanged when it is already one of
// those, so the copy is made only where it is needed.
func (c *ReadonlyContext) Value(key any) any {
	if c == nil {
		return nil
	}
	if key == adkcontext.IdentityKey {
		if c.InvocationContext == nil {
			return nil
		}
		return agent.Promote(c.InvocationContext).Value(key)
	}
	if c.Context == nil {
		return nil
	}
	return c.Context.Value(key)
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
