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

	"google.golang.org/adk/artifact"
	agentinternal "google.golang.org/adk/internal/agent"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type Agent interface {
	Name() string
	Description() string
	Run(InvocationContext) iter.Seq2[*session.Event, error]
	SubAgents() []Agent

	internal() *agent
}

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

type Config struct {
	Name        string
	Description string
	SubAgents   []Agent

	BeforeAgentCallbacks []BeforeAgentCallback
	Run                  func(InvocationContext) iter.Seq2[*session.Event, error]
	AfterAgentCallbacks  []AfterAgentCallback
}

type Artifacts interface {
	Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error)
	List(context.Context) (*artifact.ListResponse, error)
	Load(ctx context.Context, name string) (*artifact.LoadResponse, error)
	LoadVersion(ctx context.Context, name string, version int) (*artifact.LoadResponse, error)
}

type Memory interface {
	AddSession(context.Context, session.Session) error
	Search(ctx context.Context, query string) (*memory.SearchResponse, error)
}

type BeforeAgentCallback func(CallbackContext) (*genai.Content, error)
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
		// TODO: verify&update the setup here. Should we branch etc.
		ctx := &invocationContext{
			Context:   ctx,
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

var _ Agent = (*agent)(nil)

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

	callbackCtx := &callbackContext{
		Context:           ctx,
		invocationContext: ctx,
		actions:           &session.EventActions{StateDelta: make(map[string]any)},
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

	callbackCtx := &callbackContext{
		Context:           ctx,
		invocationContext: ctx,
		actions:           &session.EventActions{StateDelta: make(map[string]any)},
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
	return c.invocationContext.Artifacts()
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
