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
type BeforeAgentCallback func(Context) (*genai.Content, error)

// AfterAgentCallback is a function that is called after the agent has completed
// its run.
// If it returns non-nil content or error, a new event will be created.
//
// The callback will be skipped also if EndInvocation was called before or
// BeforeAgentCallbacks returned non-nil results.
type AfterAgentCallback func(Context) (*genai.Content, error)

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
		spanCtx, span := telemetry.StartNodeSpan(ctx, ctx, telemetry.OperationAgent{Agent: a})
		yield, endSpan := telemetry.WrapYield(span, yield, func(span trace.Span, event *session.Event, err error) {
			telemetry.TraceAgentResult(span, telemetry.TraceAgentResultParams{
				ResponseEvent: event,
				Error:         err,
			})
		})
		defer endSpan()
		// TODO: verify&update the setup here. Should we branch etc.

		// create a node context based on spanCtx and ctx
		// case 1: ctx is Context
		// case 2: ctx is InvocationContext and is not Context

		var aa Agent = a
		var newCtx context.Context = ctx.WithContext(spanCtx)

		icDelta := &InvocationContextDelta{Agent: &aa, Context: &newCtx}

		// TODO(kdroste): consider new Promote func converting InvocationContext to Context
		var nodeCtx Context
		if parentCC, ok := ctx.(Context); ok {
			nodeCtx = parentCC.WithDelta(
				&CommonContextDelta{
					InvocationContextDelta: icDelta,
				})
			// nc := NewNodeContext(ic, nil)
			// nodeCtx = NewDynamicNodeContext(nc, parentCC.Path(), parentCC.RunID(), parentCC.SubScheduler(), parentCC.OutputForAncestors())
		} else {
			ic := ctx.WithICDelta(icDelta)
			nodeCtx = NewContext(ic)
		}
		// ic := &invocationContext{
		// 	Context:   ctx.WithContext(spanCtx),
		// 	agent:     a,
		// 	artifacts: ctx.Artifacts(),
		// 	memory:    ctx.Memory(),
		// 	session:   ctx.Session(),

		// 	invocationID:   ctx.InvocationID(),
		// 	branch:         ctx.Branch(),
		// 	isolationScope: ctx.IsolationScope(),
		// 	userContent:    ctx.UserContent(),
		// 	runConfig:      ctx.RunConfig(),
		// 	endInvocation:  ctx.Ended(),
		// }

		// var nodeCtx Context
		// if parentCC, ok := ctx.(Context); ok {
		// 	nc := NewNodeContext(ic, nil)
		// 	nodeCtx = NewDynamicNodeContext(nc, parentCC.Path(), parentCC.RunID(), parentCC.SubScheduler(), parentCC.OutputForAncestors())
		// } else {
		// 	nodeCtx = NewNodeContext(ic, nil)
		// }

		event, err := runBeforeAgentCallbacks(nodeCtx)
		if event != nil || err != nil {
			if !yield(event, err) {
				return
			}
		}

		if nodeCtx.Ended() {
			return
		}

		for event, err := range a.run(nodeCtx) {
			if event != nil && event.Author == "" {
				event.Author = getAuthorForEvent(nodeCtx, event)
			}
			if !yield(event, err) {
				return
			}
		}

		if nodeCtx.Ended() {
			return
		}

		event, err = runAfterAgentCallbacks(nodeCtx)
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

func getAuthorForEvent(ctx Context, event *session.Event) string {
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

	actions := &session.EventActions{StateDelta: make(map[string]any), ArtifactDelta: make(map[string]int64)}
	callbackCtx := NewCallbackContext(ctx, actions)

	if pluginManager != nil {
		content, err := pluginManager.RunBeforeAgentCallback(callbackCtx)
		if err != nil {
			return nil, fmt.Errorf("failed to run plugin before agent callback: %w", err)
		}
		if content != nil {
			event := session.NewEvent(ctx, ctx.InvocationID())
			event.LLMResponse = model.LLMResponse{
				Content: content,
			}
			event.Author = agent.Name()
			event.Branch = ctx.Branch()
			event.Actions = *actions
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

		event := session.NewEvent(ctx, ctx.InvocationID())
		event.LLMResponse = model.LLMResponse{
			Content: content,
		}
		event.Author = agent.Name()
		event.Branch = ctx.Branch()
		event.Actions = *actions
		ctx.EndInvocation()
		return event, nil
	}

	// check if has delta create event with it
	if len(actions.StateDelta) > 0 {
		event := session.NewEvent(ctx, ctx.InvocationID())
		event.Author = agent.Name()
		event.Branch = ctx.Branch()
		event.Actions = *actions
		return event, nil
	}

	return nil, nil
}

// runAfterAgentCallbacks checks if any afterAgentCallback returns non-nil content or a state modification
// then it create a new event with the new content and state delta.
func runAfterAgentCallbacks(ctx InvocationContext) (*session.Event, error) {
	agent := ctx.Agent()
	pluginManager := pluginManagerFromContext(ctx)

	actions := &session.EventActions{StateDelta: make(map[string]any), ArtifactDelta: make(map[string]int64)}
	callbackCtx := NewCallbackContext(ctx, actions)

	if pluginManager != nil {
		content, err := pluginManager.RunAfterAgentCallback(callbackCtx)
		if err != nil {
			return nil, fmt.Errorf("failed to run plugin after agent callback: %w", err)
		}
		if content != nil {
			event := session.NewEvent(ctx, ctx.InvocationID())
			event.LLMResponse = model.LLMResponse{
				Content: content,
			}
			event.Author = agent.Name()
			event.Branch = ctx.Branch()
			event.Actions = *actions
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

		event := session.NewEvent(ctx, ctx.InvocationID())
		event.LLMResponse = model.LLMResponse{
			Content: newContent,
		}
		event.Author = agent.Name()
		event.Branch = ctx.Branch()
		event.Actions = *actions
		// TODO set context invocation ended
		// ctx.invocationEnded = true
		return event, nil
	}

	// check if has delta create event with it
	if len(actions.StateDelta) > 0 {
		event := session.NewEvent(ctx, ctx.InvocationID())
		event.Author = agent.Name()
		event.Branch = ctx.Branch()
		event.Actions = *actions
		return event, nil
	}
	return nil, nil
}

type invocationContext struct {
	context.Context

	agent     Agent
	artifacts Artifacts
	memory    Memory
	session   session.Session

	invocationID   string
	branch         string
	isolationScope string
	userContent    *genai.Content
	runConfig      *RunConfig
	endInvocation  bool
}

// Apply implements [InvocationContext].
func (c *invocationContext) WithICDelta(d *InvocationContextDelta) InvocationContext {
	if d == nil {
		return c
	}
	res := *c
	if d.UserContent != nil {
		res.userContent = *d.UserContent
	}
	return &res
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

func (c *invocationContext) IsolationScope() string {
	return c.isolationScope
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

// ResumedInput always returns (nil, false) for the base
// invocation context. Implementations that carry a resume payload
// override this method.
func (c *invocationContext) ResumedInput(string) (any, bool) { return nil, false }

func pluginManagerFromContext(ctx context.Context) pluginManager {
	a := ctx.Value(plugincontext.PluginManagerCtxKey)
	m, ok := a.(pluginManager)
	if !ok {
		return nil
	}
	return m
}

type pluginManager interface {
	RunBeforeAgentCallback(cctx Context) (*genai.Content, error)
	RunAfterAgentCallback(cctx Context) (*genai.Content, error)
}

var _ InvocationContext = (*invocationContext)(nil)
