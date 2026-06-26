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
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/platform"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// In general CommonContext should not be wrapped with contexts not providing agent.Context.
// It allows to copy&modify context instead of building chains.

// Promotes Context from non-commonContext to commonContext
func Promote(parent InvocationContext) Context {
	if c, ok := parent.(*commonContext); ok {
		return c
	}
	return &commonContext{
		Context:           parent,
		invocationContext: parent,
	}
}

func PromoteWithDelta(ctx InvocationContext, delta *CommonContextDelta) Context {
	c := Promote(ctx)
	return c.WithDelta(delta)
}

// NewContext returns a full Context backed by parent, with no callback,
// tool, or node specializations. Use it wherever a plain run context is
// needed (e.g. running an agent).
func NewContext(parent InvocationContext) Context {
	if _, ok := parent.(*commonContext); ok {
		panic("Should not happen")
	}
	return &commonContext{
		Context:           parent,
		invocationContext: parent,
	}
}

// NewNodeContext returns a Context carrying per-node resume inputs.
func NewNodeContext(parent InvocationContext, resumeInputs map[string]any) Context {
	if c, ok := parent.(Context); ok {
		// apply delta
		res := c.WithDelta(&CommonContextDelta{
			ResumeInputs: &resumeInputs,
		})
		return res
	}

	c := NewContext(parent).(*commonContext)
	c.resumeInputs = resumeInputs
	return c
}

// NewCallbackContext returns a callback context initialized with provided actions.
// actions may be nil; if so, a new session.EventActions is created with empty StateDelta and ArtifactDelta
func NewCallbackContext(ic InvocationContext, actions *session.EventActions) Context {
	actions = prepareEventActions(actions)
	cc := &commonContext{
		Context:           ic,
		invocationContext: ic,
		actions:           actions,
		artifacts:         ic.Artifacts(),
	}
	// wrap the commonContext in order to log information about someone using tool-context methods on a callback context
	wrapper := &callbackContextWrapper{
		context: cc,
	}
	return wrapper
}

// NewCallbackContextWithArtifactTracking returns a callback context initialized with provided actions.
// the returned context's Artifacts().Save(...) wrapper records each saved artifact's version into the underlying
// EventActions.ArtifactDelta so the resulting Event reflects the saves.
// actions may be nil; if so, a new session.EventActions is created with empty StateDelta and ArtifactDelta
func NewCallbackContextWithArtifactTracking(ic InvocationContext, actions *session.EventActions) Context {
	actions = prepareEventActions(actions)
	cc := &commonContext{
		Context:           ic,
		invocationContext: ic,
		actions:           actions,
		artifacts:         &trackedArtifacts{Artifacts: ic.Artifacts(), actions: actions},
	}
	// wrap the commonContext in order to log information about someone using tool-context methods on a callback context
	wrapper := &callbackContextWrapper{
		context: cc,
	}
	return wrapper
}

// NewToolContext constructs a tool context for a tool execution.
//
// If functionCallID is empty a new UUID is generated. If actions is nil a
// fresh session.EventActions with empty StateDelta and ArtifactDelta is
// allocated; missing sub-maps are populated. The returned context is
// backed by the same *commonContext implementation used for a callback context,
// so all callback-context semantics (state delta tracking, artifact delta
// tracking, etc.) apply, plus the tool-specific extensions.
func NewToolContext(ic InvocationContext, functionCallID string, actions *session.EventActions, confirmation *toolconfirmation.ToolConfirmation) Context {
	var res commonContext
	ctx, ok := ic.(*commonContext)
	if ok {
		// copy fields
		res = *ctx
	} else {
		res = commonContext{
			Context:           ic,
			invocationContext: ic,
		}
	}

	if functionCallID == "" {
		functionCallID = platform.NewUUID(ic)
	}
	actions = prepareEventActions(actions)

	res.actions = actions
	res.functionCallID = functionCallID
	res.toolConfirmation = confirmation
	res.artifacts = &trackedArtifacts{Artifacts: ic.Artifacts(), actions: actions}

	wrapper := &toolContextWrapper{
		context: &res,
	}

	return wrapper
}

func NewCleanToolContext(ctx Context, functionCallID string, actions *session.EventActions, confirmation *toolconfirmation.ToolConfirmation) (Context, error) {
	c, ok := ctx.(*commonContext)
	if !ok {
		return nil, fmt.Errorf("Context is not commonContext, but %T", ctx)
	}
	res := c.newCleanToolContext(functionCallID, actions, confirmation)
	return res, nil
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

// commonContext is the single concrete implementation of Context for static and dynamic Nodes
// Callbacks and Tools
type commonContext struct {
	context.Context
	invocationContext InvocationContext
	artifacts         Artifacts
	actions           *session.EventActions

	// Fields below are only populated by NewToolContext.
	functionCallID   string
	toolConfirmation *toolconfirmation.ToolConfirmation

	// Fields below are used by node contexts.
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

// SubScheduler returns the sub-scheduler RunNode uses to schedule
// children, or nil outside a dynamic-node activation. The struct field
// is the fast path for a freshly built dynamic-node context (and takes
// precedence); the embedded context value is the fallback that survives
// context re-wrapping by agents and the LLM flow.
func (c *commonContext) SubScheduler() DynamicSubScheduler {
	if c.subScheduler != nil {
		return c.subScheduler
	}
	if cc, ok := c.Context.(interface{ SubScheduler() DynamicSubScheduler }); ok {
		return cc.SubScheduler()
	}
	return nil
}

func (c *commonContext) newCleanToolContext(functionCallID string, actions *session.EventActions, confirmation *toolconfirmation.ToolConfirmation) Context {
	ic := &invocationContext{
		session:      c.Session(),
		invocationID: c.InvocationID(),
	}
	res := &commonContext{
		invocationContext: ic,
		actions:           actions,
		functionCallID:    functionCallID,
		toolConfirmation:  confirmation,
		subScheduler:      c.subScheduler,
	}
	return res
}

// Path implements [Context].
func (c *commonContext) Path() string {
	if c.path != "" {
		return c.path
	}
	// Fallback: when this commonContext wraps another ADK Context (e.g. via NewToolContext
	// wrapping a branchOverride), c.path is empty. Asserting against interface{ Path() string }
	// delegates to c.Context. Without this delegation, reading Path() would fail whenever
	// context is wrapped by adapters like branchOverride.
	if p, ok := c.Context.(interface{ Path() string }); ok {
		return p.Path()
	}
	return ""
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
		Context:            ctx,
		invocationContext:  ctx,
		resumeInputs:       c.resumeInputs,
		path:               c.path,
		runID:              c.runID,
		subScheduler:       c.subScheduler,
		artifacts:          c.artifacts,
		actions:            c.actions,
		functionCallID:     c.functionCallID,
		toolConfirmation:   c.toolConfirmation,
		outputForAncestors: c.outputForAncestors,
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
	if c.resumeInputs != nil {
		if v, ok := c.resumeInputs[interruptID]; ok {
			return v, true
		}
	}
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
	return c.WithAgentContext(ctx)
}

// WithAgentTimeout creates a new context as a shallow copy, adding timeout to the top of the underlying context.Context.
func (c *commonContext) WithAgentTimeout(timeout time.Duration) (Context, context.CancelFunc) {
	// copy & modify
	res := *c
	newC, cancelFunc := context.WithTimeout(res.Context, timeout)
	res.Context = newC
	return &res, cancelFunc
}

// WithAgentCancel creates a new context as a shallow copy, adding cancellation to the top of the underlying context.Context.
func (c *commonContext) WithAgentCancel() (Context, context.CancelFunc) {
	// copy & modify
	res := *c
	newC, cancelFunc := context.WithCancel(res.Context)
	res.Context = newC
	return &res, cancelFunc
}

// WithAgentContext creates a new context as a shallow copy setting the internal contexts to ctx.
// If the ctx is InvocationContext, the underlying invocationContext is set to ctx.
func (c *commonContext) WithAgentContext(ctx context.Context) Context {
	res := *c
	if c, ok := ctx.(InvocationContext); ok {
		res.Context = c
		res.invocationContext = c
	} else {
		res.Context = ctx
	}
	return &res
}

// OutputForAncestors implements [Context].
func (c *commonContext) OutputForAncestors() []string {
	if c.outputForAncestors != nil {
		return c.outputForAncestors
	}
	// Fallback: when this commonContext wraps another ADK Context (e.g. via NewToolContext
	// wrapping a branchOverride), c.outputForAncestors is nil. Asserting against
	// interface{ OutputForAncestors() []string } delegates to c.Context. Without this delegation,
	// reading OutputForAncestors() would fail whenever context is wrapped by adapters like branchOverride.
	if p, ok := c.Context.(interface{ OutputForAncestors() []string }); ok {
		return p.OutputForAncestors()
	}
	return nil
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
	if c.artifacts != nil {
		return c.artifacts
	}
	return c.invocationContext.Artifacts()
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
	_ InvocationContext = (*commonContext)(nil)
	_ ReadonlyContext   = (*commonContext)(nil)
)

// --- Tool-context extensions ----------------------------------------------
//
// The methods below are always present on *commonContext but only
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
	if c.ctx.invocationContext == nil {
		return nil, fmt.Errorf("cannot get key %q from state: invocation context is nil", key)
	}
	s := c.ctx.invocationContext.Session()
	if s == nil {
		return nil, fmt.Errorf("cannot get key %q from state: session is nil", key)
	}
	state := s.State()
	if state == nil {
		return nil, fmt.Errorf("cannot get key %q from state: state is nil", key)
	}
	return c.ctx.invocationContext.Session().State().Get(key)
}

func (c *callbackContextState) Set(key string, val any) error {
	if c.ctx.actions != nil && c.ctx.actions.StateDelta != nil {
		c.ctx.actions.StateDelta[key] = val
	}
	if c.ctx.invocationContext == nil {
		return fmt.Errorf("cannot set key %q to state: invocation context is nil", key)
	}
	s := c.ctx.invocationContext.Session()
	if s == nil {
		return fmt.Errorf("cannot set key %q to state: session is nil", key)
	}
	state := s.State()
	if state == nil {
		return fmt.Errorf("cannot set key %q to state: state is nil", key)
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
