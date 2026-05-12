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
	"fmt"
	"iter"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/artifact"
	agentinternal "google.golang.org/adk/internal/agent"
	"google.golang.org/adk/internal/plugininternal/plugincontext"
	"google.golang.org/adk/internal/telemetry"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// Agent is the base interface which all agents must implement.
//
// Agents are created with ADK constructors to ensure correct
// init & configuration.
// The constructors are available in this package and its subpackages.
// For example: llmagent.New, workflow agents, remote agent or
// agent.New.
// NOTE: in future releases we will allow just implementing this interface.
// For now agent.New is a correct solution to create custom agents.
type Agent interface {
	Name() string
	Description() string
	Run(InvocationContext) iter.Seq2[*session.Event, error]
	SubAgents() []Agent
	FindAgent(name string) Agent
	FindSubAgent(name string) Agent

	internal() *agent
}

// New creates an Agent with a custom logic defined by Run function.
func New(cfg Config) (Agent, error) {
	subAgentSet := make(map[Agent]bool)
	for _, subAgent := range cfg.SubAgents {
		if _, ok := subAgentSet[subAgent]; ok {
			return nil, fmt.Errorf("error creating agent: subagent %q appears multiple times in subAgents", subAgent.Name())
		}
		subAgentSet[subAgent] = true
	}
	return &agent{
		name:                 cfg.Name,
		description:          cfg.Description,
		subAgents:            cfg.SubAgents,
		beforeAgentCallbacks: cfg.BeforeAgentCallbacks,
		run:                  cfg.Run,
		afterAgentCallbacks:  cfg.AfterAgentCallbacks,
		State: agentinternal.State{
			AgentType: agentinternal.TypeCustomAgent,
		},
	}, nil
}

// Config is the configuration for creating a new Agent.
type Config struct {
	// Name must be a non-empty string, unique within the agent tree.
	// Agent name cannot be "user", since it's reserved for end-user's input.
	Name string
	// Description of the agent's capability.
	//
	// LLM uses this to determine whether to delegate control to the agent.
	// One-line description is enough and preferred.
	Description string
	// SubAgents are the child agents that this agent can delegate tasks to.
	// ADK will automatically set a parent of each sub-agent to this agent to
	// allow agent transferring across the tree.
	SubAgents []Agent

	// BeforeAgentCallbacks is a list of callbacks that are called sequentially
	// before the agent starts its run.
	//
	// If any callback returns non-nil content or error, then the agent run and
	// the remaining callbacks will be skipped, and a new event will be created
	// from the content or error of that callback.
	BeforeAgentCallbacks []BeforeAgentCallback
	// Run is the function that defines the agent's behavior.
	Run func(InvocationContext) iter.Seq2[*session.Event, error]
	// AfterAgentCallbacks is a list of callbacks that are called sequentially
	// after the agent has completed its run.
	//
	// If any callback returns non-nil content or error, then a new event will be
	// created from the content or error of that callback and the remaining
	// callbacks will be skipped.
	AfterAgentCallbacks []AfterAgentCallback
}

// Artifacts interface provides methods to work with artifacts of the current
// session.
type Artifacts interface {
	Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error)
	List(context.Context) (*artifact.ListResponse, error)
	Load(ctx context.Context, name string) (*artifact.LoadResponse, error)
	LoadVersion(ctx context.Context, name string, version int) (*artifact.LoadResponse, error)
}

// Memory interface provides methods to access agent memory across the
// sessions of the current user_id.
type Memory interface {
	AddSessionToMemory(context.Context, session.Session) error
	SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error)
}

// BeforeAgentCallback is a function that is called before the agent starts
// its run.
// If it returns non-nil content or error, the agent run will be skipped and a
// new event will be created.
type BeforeAgentCallback func(CallbackContext) (*genai.Content, error)

// AfterAgentCallback is a function that is called after the agent has completed
// its run.
// If it returns non-nil content or error, a new event will be created.
//
// The callback will be skipped also if EndInvocation was called before or
// BeforeAgentCallbacks returned non-nil results.
type AfterAgentCallback func(CallbackContext) (*genai.Content, error)

type agent struct {
	agentinternal.State

	name, description string
	subAgents         []Agent

	beforeAgentCallbacks []BeforeAgentCallback
	run                  func(InvocationContext) iter.Seq2[*session.Event, error]
	afterAgentCallbacks  []AfterAgentCallback
}

func (a *agent) Name() string {
	return a.name
}

func (a *agent) Description() string {
	return a.description
}

func (a *agent) SubAgents() []Agent {
	return a.subAgents
}

func (a *agent) Run(ctx InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		spanCtx, span := telemetry.StartInvokeAgentSpan(ctx, a, ctx.Session().ID(), ctx.InvocationID())
		yield, endSpan := telemetry.WrapYield(span, yield, func(span trace.Span, event *session.Event, err error) {
			telemetry.TraceAgentResult(span, telemetry.TraceAgentResultParams{
				ResponseEvent: event,
				Error:         err,
			})
		})
		defer endSpan()
		// TODO: verify&update the setup here. Should we branch etc.
		ctx := &invocationContext{
			Context:   ctx.WithContext(spanCtx),
			agent:     a,
			artifacts: ctx.Artifacts(),
			memory:    ctx.Memory(),
			session:   ctx.Session(),

			invocationID:  ctx.InvocationID(),
			branch:        ctx.Branch(),
			userContent:   ctx.UserContent(),
			runConfig:     ctx.RunConfig(),
			endInvocation: ctx.Ended(),
		}
		event, err := runBeforeAgentCallbacks(ctx)
		if event != nil || err != nil {
			if !yield(event, err) {
				return
			}
		}

		if ctx.Ended() {
			return
		}

		for event, err := range a.run(ctx) {
			if event != nil && event.Author == "" {
				event.Author = getAuthorForEvent(ctx, event)
			}
			if !yield(event, err) {
				return
			}
		}

		if ctx.Ended() {
			return
		}

		event, err = runAfterAgentCallbacks(ctx)
		if event != nil || err != nil {
			yield(event, err)
		}
	}
}

func (a *agent) internal() *agent {
	return a
}

func (a *agent) FindAgent(name string) Agent {
	if a.Name() == name {
		return a
	}
	return a.FindSubAgent(name)
}

func (a *agent) FindSubAgent(name string) Agent {
	for _, subAgent := range a.SubAgents() {
		if result := subAgent.FindAgent(name); result != nil {
			return result
		}
	}
	return nil
}

func getAuthorForEvent(ctx InvocationContext, event *session.Event) string {
	if event.LLMResponse.Content != nil && event.LLMResponse.Content.Role == genai.RoleUser {
		return genai.RoleUser
	}

	return ctx.Agent().Name()
}

// runBeforeAgentCallbacks checks if any beforeAgentCallback returns non-nil content
// then it skips agent run and returns callback result.
func runBeforeAgentCallbacks(ctx InvocationContext) (*session.Event, error) {
	agent := ctx.Agent()
	pluginManager := pluginManagerFromContext(ctx)

	callbackCtx := newCallbackContextImpl(ctx, make(map[string]any), make(map[string]int64))

	if pluginManager != nil {
		content, err := pluginManager.RunBeforeAgentCallback(callbackCtx)
		if err != nil {
			return nil, fmt.Errorf("failed to run plugin before agent callback: %w", err)
		}
		if content != nil {
			event := session.NewEvent(ctx.InvocationID())
			event.LLMResponse = model.LLMResponse{
				Content: content,
			}
			event.Author = agent.Name()
			event.Branch = ctx.Branch()
			event.Actions = *callbackCtx.actions
			ctx.EndInvocation()
			return event, nil
		}
	}

	for _, callback := range ctx.Agent().internal().beforeAgentCallbacks {
		content, err := callback(callbackCtx)
		if err != nil {
			return nil, fmt.Errorf("failed to run before agent callback: %w", err)
		}
		if content == nil {
			continue
		}

		event := session.NewEvent(ctx.InvocationID())
		event.LLMResponse = model.LLMResponse{
			Content: content,
		}
		event.Author = agent.Name()
		event.Branch = ctx.Branch()
		event.Actions = *callbackCtx.actions
		ctx.EndInvocation()
		return event, nil
	}

	// check if has delta create event with it
	if len(callbackCtx.actions.StateDelta) > 0 {
		event := session.NewEvent(ctx.InvocationID())
		event.Author = agent.Name()
		event.Branch = ctx.Branch()
		event.Actions = *callbackCtx.actions
		return event, nil
	}

	return nil, nil
}

// runAfterAgentCallbacks checks if any afterAgentCallback returns non-nil content or a state modification
// then it create a new event with the new content and state delta.
func runAfterAgentCallbacks(ctx InvocationContext) (*session.Event, error) {
	agent := ctx.Agent()
	pluginManager := pluginManagerFromContext(ctx)

	callbackCtx := newCallbackContextImpl(ctx, make(map[string]any), make(map[string]int64))

	if pluginManager != nil {
		content, err := pluginManager.RunAfterAgentCallback(callbackCtx)
		if err != nil {
			return nil, fmt.Errorf("failed to run plugin after agent callback: %w", err)
		}
		if content != nil {
			event := session.NewEvent(ctx.InvocationID())
			event.LLMResponse = model.LLMResponse{
				Content: content,
			}
			event.Author = agent.Name()
			event.Branch = ctx.Branch()
			event.Actions = *callbackCtx.actions
			return event, nil
		}
	}

	for _, callback := range agent.internal().afterAgentCallbacks {
		newContent, err := callback(callbackCtx)
		if err != nil {
			return nil, fmt.Errorf("failed to run after agent callback: %w", err)
		}
		if newContent == nil {
			continue
		}

		event := session.NewEvent(ctx.InvocationID())
		event.LLMResponse = model.LLMResponse{
			Content: newContent,
		}
		event.Author = agent.Name()
		event.Branch = ctx.Branch()
		event.Actions = *callbackCtx.actions
		// TODO set context invocation ended
		// ctx.invocationEnded = true
		return event, nil
	}

	// check if has delta create event with it
	if len(callbackCtx.actions.StateDelta) > 0 {
		event := session.NewEvent(ctx.InvocationID())
		event.Author = agent.Name()
		event.Branch = ctx.Branch()
		event.Actions = *callbackCtx.actions
		return event, nil
	}
	return nil, nil
}

// TODO: unify with internal/context.callbackContext

type callbackContext struct {
	context.Context
	invocationContext InvocationContext
	actions           *session.EventActions
	artifacts         *internalArtifacts
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

// AppName implements CallbackContext.
func (c *callbackContext) AppName() string {
	return c.invocationContext.Session().AppName()
}

// Branch implements CallbackContext.
func (c *callbackContext) Branch() string {
	return c.invocationContext.Branch()
}

// SessionID implements CallbackContext.
func (c *callbackContext) SessionID() string {
	return c.invocationContext.Session().ID()
}

// UserID implements CallbackContext.
func (c *callbackContext) UserID() string {
	return c.invocationContext.Session().UserID()
}

var _ CallbackContext = (*callbackContext)(nil)

// internalArtifacts wraps an Artifacts service so that Save also
// records the new version into the supplied EventActions.ArtifactDelta.
type internalArtifacts struct {
	Artifacts
	eventActions *session.EventActions
}

// Save persists the artifact and records its version in the event's
// ArtifactDelta.
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

// NewCallbackContext returns a CallbackContext with fresh state and
// artifact delta maps. Used by callback dispatchers in the agent
// package.
func NewCallbackContext(ctx InvocationContext) CallbackContext {
	return newCallbackContextImpl(ctx, make(map[string]any), make(map[string]int64))
}

// NewCallbackContextWithDelta returns a CallbackContext that uses the
// supplied delta maps directly (rather than allocating fresh ones).
// Used when the caller already holds the delta maps and needs them
// shared with downstream code.
func NewCallbackContextWithDelta(ctx InvocationContext, stateDelta map[string]any, artifactDelta map[string]int64) CallbackContext {
	return newCallbackContextImpl(ctx, stateDelta, artifactDelta)
}

func newCallbackContextImpl(ctx InvocationContext, stateDelta map[string]any, artifactDelta map[string]int64) *callbackContext {
	eventActions := &session.EventActions{StateDelta: stateDelta, ArtifactDelta: artifactDelta}
	return &callbackContext{
		Context:           ctx,
		invocationContext: ctx,
		actions:           eventActions,
		artifacts: &internalArtifacts{
			Artifacts:    ctx.Artifacts(),
			eventActions: eventActions,
		},
	}
}

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

func (c *invocationContext) WithContext(ctx context.Context) InvocationContext {
	newCtx := *c
	newCtx.Context = ctx
	return &newCtx
}

func pluginManagerFromContext(ctx context.Context) pluginManager {
	a := ctx.Value(plugincontext.PluginManagerCtxKey)
	m, ok := a.(pluginManager)
	if !ok {
		return nil
	}
	return m
}

type pluginManager interface {
	RunBeforeAgentCallback(cctx CallbackContext) (*genai.Content, error)
	RunAfterAgentCallback(cctx CallbackContext) (*genai.Content, error)
}

var _ InvocationContext = (*invocationContext)(nil)

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
		Context:       ctx,
		agent:         params.Agent,
		artifacts:     params.Artifacts,
		memory:        params.Memory,
		session:       params.Session,
		invocationID:  params.InvocationID,
		branch:        params.Branch,
		userContent:   params.UserContent,
		runConfig:     params.RunConfig,
		endInvocation: params.EndInvocation,
	}
}

// NewReadonlyContext returns a ReadonlyContext that delegates all
// reads to the supplied InvocationContext.
func NewReadonlyContext(ctx InvocationContext) ReadonlyContext {
	return &readonlyContext{
		Context:           ctx,
		InvocationContext: ctx,
	}
}

// readonlyContext is the canonical implementation of ReadonlyContext.
// Construct via NewReadonlyContext.
type readonlyContext struct {
	context.Context
	InvocationContext InvocationContext
}

func (c *readonlyContext) AppName() string   { return c.InvocationContext.Session().AppName() }
func (c *readonlyContext) Branch() string    { return c.InvocationContext.Branch() }
func (c *readonlyContext) SessionID() string { return c.InvocationContext.Session().ID() }
func (c *readonlyContext) UserID() string    { return c.InvocationContext.Session().UserID() }
func (c *readonlyContext) AgentName() string { return c.InvocationContext.Agent().Name() }
func (c *readonlyContext) ReadonlyState() session.ReadonlyState {
	return c.InvocationContext.Session().State()
}
func (c *readonlyContext) InvocationID() string        { return c.InvocationContext.InvocationID() }
func (c *readonlyContext) UserContent() *genai.Content { return c.InvocationContext.UserContent() }

var _ ReadonlyContext = (*readonlyContext)(nil)

// InvocationOf returns the underlying InvocationContext for ReadonlyContexts
// produced by NewReadonlyContext. Returns nil for any other ReadonlyContext
// implementation. Used by code that needs the wider invocation surface
// (e.g. instruction template rendering).
func InvocationOf(rc ReadonlyContext) InvocationContext {
	impl, ok := rc.(*readonlyContext)
	if !ok {
		return nil
	}
	return impl.InvocationContext
}
