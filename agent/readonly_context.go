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

	"google.golang.org/genai"

	"google.golang.org/adk/session"
)

// NewReadonlyContext returns a ReadonlyContext that delegates every
// read to the given InvocationContext. The returned value embeds the
// invocation's Go context.Context, so it satisfies context.Context as
// well as agent.ReadonlyContext.
func NewReadonlyContext(ctx InvocationContext) ReadonlyContext {
	return &readonlyContextImpl{
		Context:           ctx,
		invocationContext: ctx,
	}
}

// readonlyContextImpl is the canonical, in-process implementation of
// ReadonlyContext. It is unexported because callers should depend on
// the ReadonlyContext interface and construct values via
// NewReadonlyContext.
type readonlyContextImpl struct {
	context.Context

	invocationContext InvocationContext
}

func (c *readonlyContextImpl) AppName() string {
	return c.invocationContext.Session().AppName()
}

func (c *readonlyContextImpl) Branch() string {
	return c.invocationContext.Branch()
}

func (c *readonlyContextImpl) SessionID() string {
	return c.invocationContext.Session().ID()
}

func (c *readonlyContextImpl) UserID() string {
	return c.invocationContext.Session().UserID()
}

func (c *readonlyContextImpl) AgentName() string {
	return c.invocationContext.Agent().Name()
}

func (c *readonlyContextImpl) ReadonlyState() session.ReadonlyState {
	return c.invocationContext.Session().State()
}

func (c *readonlyContextImpl) InvocationID() string {
	return c.invocationContext.InvocationID()
}

func (c *readonlyContextImpl) UserContent() *genai.Content {
	return c.invocationContext.UserContent()
}

var _ ReadonlyContext = (*readonlyContextImpl)(nil)
