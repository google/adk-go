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

package toolinternal

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	contextinternal "google.golang.org/adk/internal/context"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/toolconfirmation"
)

const artifactServiceNotConfiguredMsg = "artifact service is not configured: please configure ArtifactService in the runner"

type internalArtifacts struct {
	agent.Artifacts
	eventActions *session.EventActions
}

func (ia *internalArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	if ia.Artifacts == nil {
		return nil, fmt.Errorf(artifactServiceNotConfiguredMsg)
	}
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

func (ia *internalArtifacts) List(ctx context.Context) (*artifact.ListResponse, error) {
	if ia.Artifacts == nil {
		return nil, fmt.Errorf(artifactServiceNotConfiguredMsg)
	}
	return ia.Artifacts.List(ctx)
}

func (ia *internalArtifacts) Load(ctx context.Context, name string) (*artifact.LoadResponse, error) {
	if ia.Artifacts == nil {
		return nil, fmt.Errorf(artifactServiceNotConfiguredMsg)
	}
	return ia.Artifacts.Load(ctx, name)
}

func (ia *internalArtifacts) LoadVersion(ctx context.Context, name string, version int) (*artifact.LoadResponse, error) {
	if ia.Artifacts == nil {
		return nil, fmt.Errorf(artifactServiceNotConfiguredMsg)
	}
	return ia.Artifacts.LoadVersion(ctx, name, version)
}

func NewToolContext(ctx agent.InvocationContext, functionCallID string, actions *session.EventActions, confirmation *toolconfirmation.ToolConfirmation) tool.Context {
	if functionCallID == "" {
		functionCallID = uuid.NewString()
	}
	if actions == nil {
		actions = &session.EventActions{StateDelta: make(map[string]any)}
	}
	if actions.StateDelta == nil {
		actions.StateDelta = make(map[string]any)
	}
	cbCtx := contextinternal.NewCallbackContextWithDelta(ctx, actions.StateDelta)

	return &toolContext{
		CallbackContext:   cbCtx,
		invocationContext: ctx,
		functionCallID:    functionCallID,
		eventActions:      actions,
		artifacts: &internalArtifacts{
			Artifacts:    ctx.Artifacts(),
			eventActions: actions,
		},
		toolConfirmation: confirmation,
	}
}

type toolContext struct {
	agent.CallbackContext
	invocationContext agent.InvocationContext
	functionCallID    string
	eventActions      *session.EventActions
	artifacts         *internalArtifacts
	toolConfirmation  *toolconfirmation.ToolConfirmation
}

func (c *toolContext) Artifacts() agent.Artifacts {
	return c.artifacts
}

func (c *toolContext) FunctionCallID() string {
	return c.functionCallID
}

func (c *toolContext) Actions() *session.EventActions {
	return c.eventActions
}

func (c *toolContext) AgentName() string {
	return c.invocationContext.Agent().Name()
}

func (c *toolContext) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	if c.invocationContext.Memory() == nil {
		return nil, fmt.Errorf("memory service is not set")
	}
	return c.invocationContext.Memory().Search(ctx, query)
}

func (c *toolContext) ToolConfirmation() *toolconfirmation.ToolConfirmation {
	return c.toolConfirmation
}

func (c *toolContext) RequestConfirmation(hint string, payload any) error {
	if c.functionCallID == "" {
		return fmt.Errorf("error function call id not set when requesting confirmation for tool")
	}
	if c.eventActions.RequestedToolConfirmations == nil {
		c.eventActions.RequestedToolConfirmations = make(map[string]toolconfirmation.ToolConfirmation)
	}
	c.eventActions.RequestedToolConfirmations[c.functionCallID] = toolconfirmation.ToolConfirmation{
		Hint:      hint,
		Confirmed: false,
		Payload:   payload,
	}
	// SkipSummarization stops the agent loop after this tool call. Without it,
	// the function response event becomes lastEvent and IsFinalResponse() returns
	// false (hasFunctionResponses == true), causing the loop to continue.
	// This matches the behavior of the built-in RequireConfirmation path in
	// functiontool (function.go).
	c.eventActions.SkipSummarization = true
	return nil
}
