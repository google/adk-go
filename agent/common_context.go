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

	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

func NewNodeContext(parent InvocationContext, resumeInputs map[string]any) Context {
	return &commonContext{
		Context:           parent,
		invocationContext: parent,
		resumeInputs:      resumeInputs,
	}
}

func NewDynamicNodeContext(parent Context, path, runID string, sub DynamicSubScheduler, outputForAncestors []string) Context {
	var inherited map[string]any
	if p, ok := parent.(*commonContext); ok {
		inherited = p.resumeInputs
	}
	return &commonContext{
		Context:            parent,
		invocationContext:  parent,
		resumeInputs:       inherited,
		path:               path,
		runID:              runID,
		subScheduler:       sub,
		outputForAncestors: outputForAncestors,
	}
}

// NewCallbackContext returns CallbackContext initialized with provided actions.
// actions may be nil; if so, a new session.EventActions is created with empty StateDelta and ArtifactDelta
func NewCallbackContext(ic InvocationContext, actions *session.EventActions) CallbackContext {
	actions = prepareEventActions(actions)
	cc := &commonContext{
		Context:           ic,
		invocationContext: ic,
		actions:           actions,
		artifacts:         ic.Artifacts(),
	}
	// wrap the callbackContext in order to log information about someone using ToolContext-related methods for CallbackContext
	wrapper := &callbackContextWrapper{
		context: cc,
	}
	return wrapper
}

// NewCallbackContextWithArtifactTracking returns CallbackContext initialized with provided actions.
// the returned context's Artifacts().Save(...) wrapper records each saved artifact's version into the underlying
// EventActions.ArtifactDelta so the resulting Event reflects the saves.
// actions may be nil; if so, a new session.EventActions is created with empty StateDelta and ArtifactDelta
func NewCallbackContextWithArtifactTracking(ic InvocationContext, actions *session.EventActions) CallbackContext {
	actions = prepareEventActions(actions)
	cc := &commonContext{
		Context:           ic,
		invocationContext: ic,
		actions:           actions,
		artifacts:         &trackedArtifacts{Artifacts: ic.Artifacts(), actions: actions},
	}
	// wrap the callbackContext in order to log information about someone using ToolContext-related methods for CallbackContext
	wrapper := &callbackContextWrapper{
		context: cc,
	}
	return wrapper
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
		functionCallID = uuid.NewString()
	}
	actions = prepareEventActions(actions)
	return &commonContext{
		Context:           ic,
		invocationContext: ic,
		actions:           actions,
		artifacts:         &trackedArtifacts{Artifacts: ic.Artifacts(), actions: actions},
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

// commonContext is the single concrete implementation of CallbackContext
// (and, when constructed via NewToolContext, of ToolContext as well). The
// tool-specific methods (FunctionCallID, Actions, SearchMemory,
// ToolConfirmation, RequestConfirmation) are always present on the concrete
// type; they are only meaningful when the context is used as a ToolContext.
type commonContext struct {
	context.Context
	invocationContext InvocationContext
	artifacts         Artifacts
	actions           *session.EventActions

	// Fields below are only populated by NewToolContext.
	functionCallID   string
	toolConfirmation *toolconfirmation.ToolConfirmation

	// Fields below are used by NodeContext
	// resumeInputs are keyed by InterruptID. Nil on fresh activations
	// and on handoff resume.
	resumeInputs map[string]any

	// path and runID are populated for dynamic children, empty for
	// top-level static activations.
	path  string
	runID string

	// subScheduler is non-nil only when this context belongs to a
	// dynamic-node activation; RunNode uses it to schedule children.
	subScheduler DynamicSubScheduler

	// outputForAncestors are the delegating-ancestor paths carried
	// into this activation when it runs as a WithUseAsOutput child;
	// its dynamic sub-scheduler reads them to stamp OutputFor.
	outputForAncestors []string
}

func (c *commonContext) SubScheduler() DynamicSubScheduler {
	return c.subScheduler
}

// Path implements [Context].
func (c *commonContext) Path() string {
	return c.path
}

// RunID implements [Context].
func (c *commonContext) RunID() string {
	return c.runID
}

// withBranch returns ctx wrapped with branch as its Branch().
// Implemented as a small adapter that overrides only Branch() and
// delegates the rest of the interface to the embedded ctx.
func withBranch(ctx Context, branch string) Context {
	if ctx.Branch() == branch {
		return ctx
	}
	return &branchOverride{Context: ctx, branch: branch}
}

// branchOverride wraps an InvocationContext and overrides Branch().
// All other interface methods delegate to the embedded value.
//
// WithContext is overridden so the branch survives a subsequent
// context-cancellation wrap. Without this, a caller that does
// ctx.WithContext(cancelCtx) would get an InvocationContext whose
// Branch() returns the inner ctx's branch (empty), silently
// losing the override.
type branchOverride struct {
	Context
	branch string
}

func (b *branchOverride) Branch() string {
	return b.branch
}

// WithBranch implements [Context].
func (c *commonContext) WithBranch(branch string) Context {
	ctx := withBranch(c, branch)
	return &commonContext{
		Context:           ctx,
		invocationContext: ctx,
		resumeInputs:      c.resumeInputs,
		path:              c.path,
		runID:             c.runID,
		subScheduler:      c.subScheduler,
	}
}

// Agent implements [InvocationContext].
func (c *commonContext) Agent() Agent {
	return c.invocationContext.Agent()
}

// EndInvocation implements [InvocationContext].
func (c *commonContext) EndInvocation() {
	c.invocationContext.EndInvocation()
}

// Ended implements [InvocationContext].
func (c *commonContext) Ended() bool {
	return c.invocationContext.Ended()
}

// IsolationScope implements [InvocationContext].
func (c *commonContext) IsolationScope() string {
	return c.invocationContext.IsolationScope()
}

// Memory implements [InvocationContext].
func (c *commonContext) Memory() Memory {
	return c.invocationContext.Memory()
}

// ResumedInput implements [InvocationContext].
func (c *commonContext) ResumedInput(interruptID string) (any, bool) {
	return c.invocationContext.ResumedInput(interruptID)
}

// RunConfig implements [InvocationContext].
func (c *commonContext) RunConfig() *RunConfig {
	return c.invocationContext.RunConfig()
}

// Session implements [InvocationContext].
func (c *commonContext) Session() session.Session {
	return c.invocationContext.Session()
}

// WithContext implements [InvocationContext].
func (c *commonContext) WithContext(ctx context.Context) InvocationContext {
	panic("Should not be used")
	// newCtx := c.invocationContext.WithContext(ctx)
	// return &commonContext{
	// 	Context:           newCtx,
	// 	invocationContext: newCtx,
	// 	artifacts:         c.artifacts,
	// 	actions:           c.actions,
	// 	functionCallID:    c.functionCallID,
	// 	toolConfirmation:  c.toolConfirmation,
	// }
}

func (c *commonContext) WithAgentContext(ctx context.Context) Context {
	var ic InvocationContext
	if c, ok := ctx.(InvocationContext); ok {
		ic = c
	} else {
		ic = &invocationContext{
			Context: ctx,
		}
	}

	//TODO: other fields???
	// newCtx := agent.NewNodeContext(ctx, nil)
	return &commonContext{
		Context:           ic,
		invocationContext: ic,
		artifacts:         c.artifacts,
		actions:           c.actions,
		functionCallID:    c.functionCallID,
		toolConfirmation:  c.toolConfirmation,
	}
}

func (c *commonContext) AgentName() string {
	return c.invocationContext.Agent().Name()
}

func (c *commonContext) ReadonlyState() session.ReadonlyState {
	return c.invocationContext.Session().State()
}

func (c *commonContext) State() session.State {
	return &callbackContextState{ctx: c}
}

func (c *commonContext) Artifacts() Artifacts {
	return c.artifacts
}

func (c *commonContext) InvocationID() string {
	return c.invocationContext.InvocationID()
}

func (c *commonContext) UserContent() *genai.Content {
	return c.invocationContext.UserContent()
}

func (c *commonContext) AppName() string {
	return c.invocationContext.Session().AppName()
}

func (c *commonContext) Branch() string {
	return c.invocationContext.Branch()
}

func (c *commonContext) SessionID() string {
	return c.invocationContext.Session().ID()
}

func (c *commonContext) UserID() string {
	return c.invocationContext.Session().UserID()
}

var (
	_ Context           = (*commonContext)(nil)
	_ CallbackContext   = (*commonContext)(nil)
	_ ToolContext       = (*commonContext)(nil)
	_ InvocationContext = (*commonContext)(nil)
	_ ReadonlyContext   = (*commonContext)(nil)
)

// --- ToolContext extensions ----------------------------------------------
//
// The methods below are always present on *callbackContext but only
// meaningful when the context was constructed via NewToolContext (i.e.
// when functionCallID is set).

// FunctionCallID returns the function call identifier associated with the
// current tool execution, or "" if this context was not constructed for a
// tool call.
func (c *commonContext) FunctionCallID() string {
	return c.functionCallID
}

// Actions returns the EventActions for the current event. Tools can mutate
// the returned value to influence the agent loop (e.g. state deltas, agent
// transfers).
func (c *commonContext) Actions() *session.EventActions {
	return c.actions
}

// SearchMemory performs a semantic search on the agent's memory.
func (c *commonContext) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	if c.invocationContext.Memory() == nil {
		return nil, fmt.Errorf("memory service is not set")
	}
	return c.invocationContext.Memory().SearchMemory(ctx, query)
}

// ToolConfirmation returns the Human-in-the-Loop confirmation handle for the
// current tool execution, or nil if no confirmation is currently associated
// with the call.
func (c *commonContext) ToolConfirmation() *toolconfirmation.ToolConfirmation {
	return c.toolConfirmation
}

// RequestConfirmation initiates the Human-in-the-Loop (HITL) approval flow
// for the current tool call. It records a pending confirmation in the
// underlying EventActions and sets SkipSummarization so the agent loop halts
// until the user responds.
func (c *commonContext) RequestConfirmation(hint string, payload any) error {
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

func (c *commonContext) SetInvocationContext(ic InvocationContext) {
	c.invocationContext = ic
	c.Context = ic
}

func (c *commonContext) InvocationContext() InvocationContext {
	return c.invocationContext
}

// callbackContextState is a session.State implementation backed by the
// callback context's EventActions.StateDelta and the underlying session state.
type callbackContextState struct {
	ctx *commonContext
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
