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

// InvocationContextParams holds the data used to construct a new
// InvocationContext via NewInvocationContext.
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

// NewInvocationContext returns a new InvocationContext built from the
// given Go context and params. If params.InvocationID is empty, a new
// unique identifier of the form "e-<uuid>" is generated.
func NewInvocationContext(ctx context.Context, params InvocationContextParams) InvocationContext {
	if params.InvocationID == "" {
		params.InvocationID = "e-" + uuid.NewString()
	}
	return &invocationContextImpl{
		Context: ctx,
		params:  params,
	}
}

// invocationContextImpl is the canonical, in-process implementation of
// InvocationContext. It is unexported because callers should depend on
// the InvocationContext interface and construct values via
// NewInvocationContext.
type invocationContextImpl struct {
	context.Context

	params InvocationContextParams
}

func (c *invocationContextImpl) Artifacts() Artifacts {
	return c.params.Artifacts
}

func (c *invocationContextImpl) Agent() Agent {
	return c.params.Agent
}

func (c *invocationContextImpl) Branch() string {
	return c.params.Branch
}

func (c *invocationContextImpl) InvocationID() string {
	return c.params.InvocationID
}

func (c *invocationContextImpl) Memory() Memory {
	return c.params.Memory
}

func (c *invocationContextImpl) Session() session.Session {
	return c.params.Session
}

func (c *invocationContextImpl) UserContent() *genai.Content {
	return c.params.UserContent
}

func (c *invocationContextImpl) RunConfig() *RunConfig {
	return c.params.RunConfig
}

func (c *invocationContextImpl) EndInvocation() {
	c.params.EndInvocation = true
}

func (c *invocationContextImpl) Ended() bool {
	return c.params.EndInvocation
}

func (c *invocationContextImpl) WithContext(ctx context.Context) InvocationContext {
	newCtx := *c
	newCtx.Context = ctx
	return &newCtx
}

// WithAgent returns a copy of c with the Agent param overridden. The
// embedded context.Context and all other params are shared with the
// receiver. See InvocationContext.WithAgent for the contract.
func (c *invocationContextImpl) WithAgent(a Agent) InvocationContext {
	newCtx := *c
	newCtx.params.Agent = a
	return &newCtx
}

var _ InvocationContext = (*invocationContextImpl)(nil)
