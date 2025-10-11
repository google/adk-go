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

	"github.com/google/uuid"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// InvocationContextParams is the parameter for NewInvocationContext.
type InvocationContextParams struct {
	Artifacts artifact.Artifacts
	Memory    memory.Memory
	Session   session.Session

	Branch string
	Agent  agent.Agent

	UserContent *genai.Content
	RunConfig   *agent.RunConfig
}

// NewInvocationContext constructs a new agent.InvocationContext.
func NewInvocationContext(ctx context.Context, params InvocationContextParams) agent.InvocationContext {
	return &invocationContext{
		Context:      ctx,
		params:       params,
		invocationID: "e-" + uuid.NewString(),
	}
}

type invocationContext struct {
	context.Context

	params       InvocationContextParams
	invocationID string
}

func (c *invocationContext) Artifacts() artifact.Artifacts {
	return c.params.Artifacts
}

func (c *invocationContext) Agent() agent.Agent {
	return c.params.Agent
}

func (c *invocationContext) Branch() string {
	return c.params.Branch
}

func (c *invocationContext) InvocationID() string {
	return c.invocationID
}

func (c *invocationContext) Memory() memory.Memory {
	return c.params.Memory
}

func (c *invocationContext) Session() session.Session {
	return c.params.Session
}

func (c *invocationContext) UserContent() *genai.Content {
	return c.params.UserContent
}

func (c *invocationContext) RunConfig() *agent.RunConfig {
	return c.params.RunConfig
}

// TODO: implement endInvocation
func (c *invocationContext) EndInvocation() {
}

// TODO: implement endInvocation
func (c *invocationContext) Ended() bool {
	return false
}
