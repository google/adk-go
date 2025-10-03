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
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type InvocationContextParams struct {
	Artifacts agent.Artifacts
	Memory    agent.Memory
	Session   session.Session

	Branch string
	Agent  agent.Agent

	UserContent *genai.Content
	RunConfig   *agent.RunConfig
}

func NewInvocationContext(ctx context.Context, params InvocationContextParams) agent.InvocationContext {
	return &InvocationContext{
		Context: ctx,

		artifacts: params.Artifacts,
		memory:    params.Memory,
		session:   params.Session,

		invocationID: "e-" + uuid.NewString(),
		branch:       params.Branch,
		agent:        params.Agent,

		userContent: params.UserContent,
		runConfig:   params.RunConfig,
	}
}

type InvocationContext struct {
	context.Context

	artifacts agent.Artifacts
	memory    agent.Memory
	session   session.Session

	invocationID string
	branch       string
	agent        agent.Agent

	userContent *genai.Content
	runConfig   *agent.RunConfig
}

func (c *InvocationContext) Artifacts() agent.Artifacts {
	return c.artifacts
}

func (c *InvocationContext) Agent() agent.Agent {
	return c.agent
}

func (c *InvocationContext) Branch() string {
	return c.branch
}

func (c *InvocationContext) InvocationID() string {
	return c.invocationID
}

func (c *InvocationContext) Memory() agent.Memory {
	return c.memory
}

func (c *InvocationContext) Session() session.Session {
	return c.session
}

func (c *InvocationContext) UserContent() *genai.Content {
	return c.userContent
}

func (c *InvocationContext) RunConfig() *agent.RunConfig {
	return c.runConfig
}

// TODO: implement endInvocation
func (c *InvocationContext) EndInvocation() {
}
