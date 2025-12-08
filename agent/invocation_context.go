// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
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

	"google.golang.org/adk/artifact"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// InvocationContextParams holds parameters for creating an InvocationContext.
// This matches the pattern used in internal/context for consistency.
type InvocationContextParams struct {
	Artifacts     Artifacts
	Memory        Memory
	Session       session.Session
	Agent         Agent
	InvocationID  string
	Branch        string
	UserContent   *genai.Content
	RunConfig     *RunConfig
	EndInvocation bool
}

// NewInvocationContextFromParams creates an InvocationContext from parameters.
// This provides the same interface as internal/context.NewInvocationContext.
func NewInvocationContextFromParams(ctx context.Context, params InvocationContextParams) InvocationContext {
	invocationID := params.InvocationID
	if invocationID == "" {
		invocationID = "e-" + uuid.NewString()
	}
	return &invocationContext{
		Context:       ctx,
		agent:         params.Agent,
		artifacts:     params.Artifacts,
		memory:        params.Memory,
		session:       params.Session,
		invocationID:  invocationID,
		branch:        params.Branch,
		userContent:   params.UserContent,
		runConfig:     params.RunConfig,
		endInvocation: params.EndInvocation,
	}
}

// NewWrappedInvocationContext creates an InvocationContext that wraps an existing InvocationContext.
func NewWrappedInvocationContext(existingCtx InvocationContext, newAgent Agent) InvocationContext {
	return &invocationContext{
		Context:       existingCtx,
		agent:         newAgent,
		artifacts:     existingCtx.Artifacts(),
		memory:        existingCtx.Memory(),
		session:       existingCtx.Session(),
		invocationID:  existingCtx.InvocationID(),
		branch:        existingCtx.Branch(),
		userContent:   existingCtx.UserContent(),
		runConfig:     existingCtx.RunConfig(),
		endInvocation: existingCtx.Ended(),
	}
}

type invocationContext struct {
	context.Context

	agent     Agent
	artifacts Artifacts
	memory    Memory
	session   session.Session

	invocationID  string
	branch        string
	userContent   *genai.Content
	runConfig     *RunConfig
	endInvocation bool
}

func (c *invocationContext) Agent() Agent {
	return c.agent
}

func (c *invocationContext) Artifacts() Artifacts {
	return c.artifacts
}

func (c *invocationContext) Memory() Memory {
	return c.memory
}

func (c *invocationContext) Session() session.Session {
	return c.session
}

func (c *invocationContext) InvocationID() string {
	return c.invocationID
}

func (c *invocationContext) Branch() string {
	return c.branch
}

func (c *invocationContext) UserContent() *genai.Content {
	return c.userContent
}

func (c *invocationContext) RunConfig() *RunConfig {
	return c.runConfig
}

func (c *invocationContext) EndInvocation() {
	c.endInvocation = true
}

func (c *invocationContext) Ended() bool {
	return c.endInvocation
}

// internalArtifacts wraps the Artifacts interface to track artifact saves in event actions
type internalArtifacts struct {
	Artifacts
	eventActions *session.EventActions
}

func (ia *internalArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	resp, err := ia.Artifacts.Save(ctx, name, data)
	if err != nil {
		return resp, err
	}
	if ia.eventActions != nil {
		if ia.eventActions.ArtifactDelta == nil {
			ia.eventActions.ArtifactDelta = make(map[string]int64)
		}
		// TODO: RWLock, check the version stored is newer in case multiple tools save the same file.
		ia.eventActions.ArtifactDelta[name] = resp.Version
	}
	return resp, nil
}
