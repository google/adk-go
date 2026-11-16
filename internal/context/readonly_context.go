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
// invocation this context speaks for. Every other key delegates to the embedded
// context, preserving existing behavior.
//
// This type answers the key without carrying [adkcontext.Marker], unlike every
// other type that does. That is safe only because it is not an
// [agent.InvocationContext], so it can never sit in the field the identity
// procedure READS rather than asks — give it those methods and it would need the
// marker too.
//
// The identity key has to be answered here rather than left to the embedded
// context, even though the two are the same invocation. Forwarding asks that
// invocation directly, and one implemented outside the agent package cannot
// override a key it cannot name, so its embedded parent replies — with the
// enclosing invocation's user, whose credential would then be minted for a call
// they never made. agent.Promote applies the one identity procedure instead,
// which reads the invocation's own session and falls back only to a context type
// agent itself owns. It returns the argument unchanged only when that argument is
// already the type the procedure lives on; for every other invocation, this one
// included, it allocates a wrapper per lookup.
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

// AgentName implements agent.ReadonlyContext. It asks the invocation context
// for its own AgentName when it has one, so that an invocation context that
// answers AgentName but returns nil from Agent still has a name; otherwise it
// names the invocation's agent.
func (c *ReadonlyContext) AgentName() string {
	if n, ok := c.InvocationContext.(interface{ AgentName() string }); ok {
		return n.AgentName()
	}
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

// RunConfig returns the run config of the invocation c reads from, unchanged
// and possibly nil. It does not make c an [agent.InvocationContext]; it lets
// agent.CustomMetadata reach the run config from an instruction provider or
// toolset when no runner stored a filtered snapshot.
func (c *ReadonlyContext) RunConfig() *agent.RunConfig {
	return c.InvocationContext.RunConfig()
}
