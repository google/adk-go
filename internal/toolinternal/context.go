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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/toolconfirmation"
)

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
	if actions.ArtifactDelta == nil {
		actions.ArtifactDelta = make(map[string]int64)
	}
	// NewCallbackContextWithDelta already wraps Artifacts with delta
	// tracking against the supplied EventActions, so toolContext just
	// inherits Artifacts() from the embedded CallbackContext.
	cbCtx := agent.NewCallbackContextWithDelta(ctx, actions.StateDelta, actions.ArtifactDelta)

	return &toolContext{
		CallbackContext:   cbCtx,
		invocationContext: ctx,
		functionCallID:    functionCallID,
		eventActions:      actions,
		toolConfirmation:  confirmation,
	}
}

type toolContext struct {
	agent.CallbackContext
	invocationContext agent.InvocationContext
	functionCallID    string
	eventActions      *session.EventActions
	toolConfirmation  *toolconfirmation.ToolConfirmation
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
	return c.invocationContext.Memory().SearchMemory(ctx, query)
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
