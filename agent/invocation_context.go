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

	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/session"
)

// InvocationContextParams gathers everything NewInvocationContext needs.
type InvocationContextParams struct {
	Artifacts Artifacts
	Memory    Memory
	Session   session.Session

	Branch string
	Agent  Agent

	UserContent   *genai.Content
	RunConfig     *RunConfig
	EndInvocation bool
	InvocationID  string
}

// NewInvocationContext returns a fresh InvocationContext for one
// agent invocation. Used by runner and any code constructing an
// invocation outside the framework's own dispatch path.
func NewInvocationContext(ctx context.Context, params InvocationContextParams) InvocationContext {
	if params.InvocationID == "" {
		params.InvocationID = "e-" + uuid.NewString()
	}
	return &invocationContext{
		Context: ctx,
		params:  params,
	}
}

// invocationContext is the canonical implementation of InvocationContext.
// Construct via NewInvocationContext.
type invocationContext struct {
	context.Context

	params InvocationContextParams
}

func (c *invocationContext) Artifacts() Artifacts {
	return c.params.Artifacts
}

func (c *invocationContext) Agent() Agent {
	return c.params.Agent
}

func (c *invocationContext) Branch() string {
	return c.params.Branch
}

func (c *invocationContext) InvocationID() string {
	return c.params.InvocationID
}

func (c *invocationContext) Memory() Memory {
	return c.params.Memory
}

func (c *invocationContext) Session() session.Session {
	return c.params.Session
}

func (c *invocationContext) UserContent() *genai.Content {
	return c.params.UserContent
}

func (c *invocationContext) RunConfig() *RunConfig {
	return c.params.RunConfig
}

func (c *invocationContext) EndInvocation() {
	c.params.EndInvocation = true
}

func (c *invocationContext) Ended() bool {
	return c.params.EndInvocation
}

func (c *invocationContext) WithContext(ctx context.Context) InvocationContext {
	newCtx := *c
	newCtx.Context = ctx
	return &newCtx
}

var _ InvocationContext = (*invocationContext)(nil)
