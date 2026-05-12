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

// NewReadonlyContext returns a ReadonlyContext that delegates all
// reads to the supplied InvocationContext.
func NewReadonlyContext(ctx InvocationContext) ReadonlyContext {
	return &readonlyContext{
		Context:           ctx,
		InvocationContext: ctx,
	}
}

// readonlyContext is the canonical implementation of ReadonlyContext.
// Construct via NewReadonlyContext.
type readonlyContext struct {
	context.Context
	InvocationContext InvocationContext
}

// AppName implements ReadonlyContext.
func (c *readonlyContext) AppName() string {
	return c.InvocationContext.Session().AppName()
}

// Branch implements ReadonlyContext.
func (c *readonlyContext) Branch() string {
	return c.InvocationContext.Branch()
}

// SessionID implements ReadonlyContext.
func (c *readonlyContext) SessionID() string {
	return c.InvocationContext.Session().ID()
}

// UserID implements ReadonlyContext.
func (c *readonlyContext) UserID() string {
	return c.InvocationContext.Session().UserID()
}

func (c *readonlyContext) AgentName() string {
	return c.InvocationContext.Agent().Name()
}

func (c *readonlyContext) ReadonlyState() session.ReadonlyState {
	return c.InvocationContext.Session().State()
}

func (c *readonlyContext) InvocationID() string {
	return c.InvocationContext.InvocationID()
}

func (c *readonlyContext) UserContent() *genai.Content {
	return c.InvocationContext.UserContent()
}

var _ ReadonlyContext = (*readonlyContext)(nil)
