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

	"google.golang.org/adk/llm"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type Agent interface {
	Name() string
	Description() string
	// TODO: verify if the interface would have "Run(Context) error" and agent will call agent.Context.Report(Event)
	Run(Context) iter.Seq2[*session.Event, error]
	Parent() Agent
	SubAgents() []Agent
	// TODO: verify if we should add unexported methods to ensure only this package can implement this interface.
	// TODO: maybe opact struct?

	internal() *agent
}

func New(cfg Config) (Agent, error) {
	a := &agent{
		name:        cfg.Name,
		description: cfg.Description,
		subAgents:   cfg.SubAgents,
		beforeAgent: cfg.BeforeAgent,
		run:         cfg.Run,
		afterAgent:  cfg.AfterAgent,
	}

	a.self = a
	if cfg.Self != nil {
		a.self = cfg.Self
	}

	for _, subAgent := range cfg.SubAgents {
		sa := subAgent.internal()

		if sa.parent != nil {
			return nil, fmt.Errorf("subAgent %v already has a parent %v", subAgent.Name(), sa.parent.Name())
		}

		sa.parent = a.self
	}

	return a, nil
}

type Config struct {
	Name        string
	Description string
	SubAgents   []Agent

	BeforeAgent []Callback
	// TODO: verify if the interface would have "Run(Context) error" and agent will call agent.Context.Report(Event)
	Run func(Context) iter.Seq2[*session.Event, error]
	// TODO: after agent callback should take: ctx, actual_resp, actual_err. So the callback can inspect and decide what to return.
	AfterAgent []Callback

	// TODO: this is the field for internal usage
	Self Agent
}

type Context interface {
	context.Context

	UserContent() *genai.Content
	InvocationID() string
	Branch() string
	Agent() Agent

	Session() session.Session
	Artifacts() Artifacts

	Report(*session.Event)

	End()
	Ended() bool
}

type Artifacts interface {
	Save(name string, data genai.Part) error
	Load(name string) (genai.Part, error)
	LoadVersion(name string, version int) (genai.Part, error)
}

type Callback func(Context) (*genai.Content, error)

type agent struct {
	name, description string
	subAgents         []Agent

	parent Agent
	self   Agent

	beforeAgent []Callback
	run         func(Context) iter.Seq2[*session.Event, error]
	afterAgent  []Callback
}

func (a *agent) Name() string {
	return a.name
}

func (a *agent) Description() string {
	return a.description
}

func (a *agent) Parent() Agent {
	return a.parent
}

func (a *agent) SubAgents() []Agent {
	return a.subAgents
}

func (a *agent) Run(ctx Context) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// TODO: verify&update the setup here. Should we branch etc.
		ctx := NewContext(ctx, a.self, ctx.UserContent(), ctx.Session(), ctx.Branch())

		event, err := runBeforeAgentCallbacks(ctx)
		if event != nil || err != nil {
			yield(event, err)
			return
		}

		for event, err := range a.run(ctx) {
			if event != nil && event.Author == "" {
				event.Author = getAuthorForEvent(ctx, event)
			}

			event, err := runAfterAgentCallbacks(ctx, event, err)
			if !yield(event, err) {
				return
			}
		}
	}
}

func (a *agent) internal() *agent {
	return a
}

var _ Agent = (*agent)(nil)

func getAuthorForEvent(ctx Context, event *session.Event) string {
	if event.LLMResponse != nil && event.LLMResponse.Content != nil && event.LLMResponse.Content.Role == genai.RoleUser {
		return genai.RoleUser
	}

	return ctx.Agent().Name()
}

// runBeforeAgentCallbacks checks if any beforeAgentCallback returns non-nil content
// then it skips agent run and returns callback result.
func runBeforeAgentCallbacks(ctx Context) (*session.Event, error) {
	agent := ctx.Agent()
	for _, callback := range ctx.Agent().internal().beforeAgent {
		content, err := callback(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to run before agent callback: %w", err)
		}
		if content == nil {
			continue
		}

		event := session.NewEvent(ctx.InvocationID())
		event.LLMResponse = &llm.Response{
			Content: content,
		}
		event.Author = agent.Name()
		event.Branch = ctx.Branch()
		// TODO: how to set it. Should it be a part of Context?
		// event.Actions = callbackContext.EventActions

		// TODO: set ictx.end_invocation

		return event, nil
	}

	return nil, nil
}

// runAfterAgentCallbacks checks if any afterAgentCallback returns non-nil content
// then it replaces the event content with a value from the callback.
func runAfterAgentCallbacks(ctx Context, agentEvent *session.Event, agentError error) (*session.Event, error) {
	agent := ctx.Agent()
	for _, callback := range agent.internal().afterAgent {
		// TODO: after agent callback should take: ctx, actual_resp, actual_err. So the callback can inspect and decide what to return.
		newContent, err := callback(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to run after agent callback: %w", err)
		}
		if newContent == nil {
			continue
		}

		agentEvent.LLMResponse.Content = newContent
		return agentEvent, nil
	}

	return agentEvent, agentError
}
