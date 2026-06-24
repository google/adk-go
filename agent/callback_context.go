// Copyright 2026 Google LLC
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
	"fmt"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/platform"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// NewCallbackContext returns CallbackContext initialized with provided actions.
// actions may be nil; if so, a new session.EventActions is created with empty StateDelta and ArtifactDelta
func NewCallbackContext(ic InvocationContext, actions *session.EventActions) CallbackContext {
	actions = prepareEventActions(actions)
	cc := &callbackContext{
		Context:           ic,
		invocationContext: ic,
		actions:           actions,
		artifacts:         ic.Artifacts(),
	}
	return cc
}

// NewCallbackContextWithArtifactTracking returns CallbackContext initialized with provided actions.
// the returned context's Artifacts().Save(...) wrapper records each saved artifact's version into the underlying
// EventActions.ArtifactDelta so the resulting Event reflects the saves.
// actions may be nil; if so, a new session.EventActions is created with empty StateDelta and ArtifactDelta
func NewCallbackContextWithArtifactTracking(ic InvocationContext, actions *session.EventActions) CallbackContext {
	actions = prepareEventActions(actions)
	var artifacts Artifacts
	if ic.Artifacts() != nil {
		artifacts = &trackedArtifacts{Artifacts: ic.Artifacts(), actions: actions}
	}
	cc := &callbackContext{
		Context:           ic,
		invocationContext: ic,
		actions:           actions,
		artifacts:         artifacts,
	}
	return cc
}

// NewToolContext constructs a ToolContext for a tool execution.
//
// If functionCallID is empty a new UUID is generated. If actions is nil a
// fresh session.EventActions with empty StateDelta and ArtifactDelta is
// allocated; missing sub-maps are populated. The returned ToolContext is
// backed by the same *callbackContext implementation used for CallbackContext,
// so all callback-context semantics (state delta tracking, artifact delta
// tracking, etc.) apply, plus the tool-specific extensions on ToolContext.
func NewToolContext(ic InvocationContext, functionCallID string, actions *session.EventActions, confirmation *toolconfirmation.ToolConfirmation) ToolContext {
	if functionCallID == "" {
		functionCallID = platform.NewUUID(ic)
	}
	actions = prepareEventActions(actions)
	var artifacts Artifacts
	if ic.Artifacts() != nil {
		artifacts = &trackedArtifacts{Artifacts: ic.Artifacts(), actions: actions}
	}
	return &callbackContext{
		Context:           ic,
		invocationContext: ic,
		actions:           actions,
		artifacts:         artifacts,
		functionCallID:    functionCallID,
		toolConfirmation:  confirmation,
	}
}

func prepareEventActions(actions *session.EventActions) *session.EventActions {
	if actions == nil {
		return &session.EventActions{StateDelta: make(map[string]any), ArtifactDelta: make(map[string]int64)}
	}
	// create missing maps if needed
	if actions.StateDelta == nil {
		actions.StateDelta = make(map[string]any)
	}
	if actions.ArtifactDelta == nil {
		actions.ArtifactDelta = make(map[string]int64)
	}
	return actions
}

// callbackContext is the single concrete implementation of CallbackContext
// (and, when constructed via NewToolContext, of ToolContext as well). The
// tool-specific methods (FunctionCallID, Actions, SearchMemory,
// ToolConfirmation, RequestConfirmation) are always present on the concrete
// type; they are only meaningful when the context is used as a ToolContext.
type callbackContext struct {
	context.Context
	invocationContext InvocationContext
	artifacts         Artifacts
	actions           *session.EventActions

	// Fields below are only populated by NewToolContext.
	functionCallID   string
	toolConfirmation *toolconfirmation.ToolConfirmation
}

func (c *callbackContext) AgentName() string {
	return c.invocationContext.Agent().Name()
}

func (c *callbackContext) ReadonlyState() session.ReadonlyState {
	return c.invocationContext.Session().State()
}

func (c *callbackContext) State() session.State {
	return &callbackContextState{ctx: c}
}

func (c *callbackContext) Artifacts() Artifacts {
	return c.artifacts
}

func (c *callbackContext) InvocationID() string {
	return c.invocationContext.InvocationID()
}

func (c *callbackContext) UserContent() *genai.Content {
	return c.invocationContext.UserContent()
}

func (c *callbackContext) AppName() string {
	return c.invocationContext.Session().AppName()
}

func (c *callbackContext) Branch() string {
	return c.invocationContext.Branch()
}

func (c *callbackContext) SessionID() string {
	return c.invocationContext.Session().ID()
}

func (c *callbackContext) UserID() string {
	return c.invocationContext.Session().UserID()
}

var (
	_ CallbackContext = (*callbackContext)(nil)
	_ ToolContext     = (*callbackContext)(nil)
)

// --- ToolContext extensions ----------------------------------------------
//
// The methods below are always present on *callbackContext but only
// meaningful when the context was constructed via NewToolContext (i.e.
// when functionCallID is set).

// FunctionCallID returns the function call identifier associated with the
// current tool execution, or "" if this context was not constructed for a
// tool call.
func (c *callbackContext) FunctionCallID() string {
	return c.functionCallID
}

// Actions returns the EventActions for the current event. Tools can mutate
// the returned value to influence the agent loop (e.g. state deltas, agent
// transfers).
func (c *callbackContext) Actions() *session.EventActions {
	return c.actions
}

// SearchMemory performs a semantic search on the agent's memory.
func (c *callbackContext) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	if c.invocationContext.Memory() == nil {
		return nil, fmt.Errorf("memory service is not set")
	}
	return c.invocationContext.Memory().SearchMemory(ctx, query)
}

// ToolConfirmation returns the Human-in-the-Loop confirmation handle for the
// current tool execution, or nil if no confirmation is currently associated
// with the call.
func (c *callbackContext) ToolConfirmation() *toolconfirmation.ToolConfirmation {
	return c.toolConfirmation
}

// RequestConfirmation initiates the Human-in-the-Loop (HITL) approval flow
// for the current tool call. It records a pending confirmation in the
// underlying EventActions and sets SkipSummarization so the agent loop halts
// until the user responds.
func (c *callbackContext) RequestConfirmation(hint string, payload any) error {
	if c.functionCallID == "" {
		return fmt.Errorf("error function call id not set when requesting confirmation for tool")
	}
	if c.actions.RequestedToolConfirmations == nil {
		c.actions.RequestedToolConfirmations = make(map[string]toolconfirmation.ToolConfirmation)
	}
	c.actions.RequestedToolConfirmations[c.functionCallID] = toolconfirmation.ToolConfirmation{
		Hint:      hint,
		Confirmed: false,
		Payload:   payload,
	}
	// SkipSummarization stops the agent loop after this tool call. Without it,
	// the function response event becomes lastEvent and IsFinalResponse() returns
	// false (hasFunctionResponses == true), causing the loop to continue.
	c.actions.SkipSummarization = true
	return nil
}

// callbackContextState is a session.State implementation backed by the
// callback context's EventActions.StateDelta and the underlying session state.
type callbackContextState struct {
	ctx *callbackContext
}

func (c *callbackContextState) Get(key string) (any, error) {
	if c.ctx.actions != nil && c.ctx.actions.StateDelta != nil {
		if val, ok := c.ctx.actions.StateDelta[key]; ok {
			return val, nil
		}
	}
	return c.ctx.invocationContext.Session().State().Get(key)
}

func (c *callbackContextState) Set(key string, val any) error {
	if c.ctx.actions != nil && c.ctx.actions.StateDelta != nil {
		c.ctx.actions.StateDelta[key] = val
	}
	return c.ctx.invocationContext.Session().State().Set(key, val)
}

func (c *callbackContextState) All() iter.Seq2[string, any] {
	return c.ctx.invocationContext.Session().State().All()
}

// trackedArtifacts wraps an Artifacts to record each successful Save into the
// supplied EventActions.ArtifactDelta.
type trackedArtifacts struct {
	Artifacts
	actions *session.EventActions
}

func (a *trackedArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	resp, err := a.Artifacts.Save(ctx, name, data)
	if err != nil {
		return resp, err
	}
	if a.actions != nil {
		if a.actions.ArtifactDelta == nil {
			a.actions.ArtifactDelta = make(map[string]int64)
		}
		// TODO: RWLock, check the version stored is newer in case multiple tools save the same file.
		a.actions.ArtifactDelta[name] = resp.Version
	}
	return resp, nil
}
