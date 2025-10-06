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

	agentinternal "google.golang.org/adk/internal/agent"
	"google.golang.org/adk/memoryservice"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type Agent interface {
	Name() string
	Description() string
	Run(Context) iter.Seq2[*session.Event, error]
	SubAgents() []Agent

	internal() *agent
}

func New(cfg Config) (Agent, error) {
	return &agent{
		name:        cfg.Name,
		description: cfg.Description,
		subAgents:   cfg.SubAgents,
		beforeAgent: cfg.BeforeAgent,
		run:         cfg.Run,
		afterAgent:  cfg.AfterAgent,
		State: agentinternal.State{
			AgentType: agentinternal.TypeCustomAgent,
		},
	}, nil
}

type Config struct {
	Name        string
	Description string
	SubAgents   []Agent

	BeforeAgent []BeforeAgentCallback
	Run         func(Context) iter.Seq2[*session.Event, error]
	AfterAgent  []AfterAgentCallback
}

type Context interface {
	context.Context

	UserContent() *genai.Content
	InvocationID() string
	Branch() string
	Agent() Agent

	Session() session.Session
	Artifacts() Artifacts
	Memory() Memory

	End()
	Ended() bool
}

type Artifacts interface {
	Save(name string, data genai.Part) error
	Load(name string) (genai.Part, error)
	LoadVersion(name string, version int) (genai.Part, error)
	List() ([]string, error)
}

type Memory interface {
	AddSession(session session.Session) error
	Search(query string) ([]memoryservice.MemoryEntry, error)
}

type BeforeAgentCallback func(Context) (*genai.Content, error)
type AfterAgentCallback func(Context, *session.Event, error) (*genai.Content, error)

type agent struct {
	agentinternal.State

	name, description string
	subAgents         []Agent

	beforeAgent []BeforeAgentCallback
	run         func(Context) iter.Seq2[*session.Event, error]
	afterAgent  []AfterAgentCallback
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

func (a *agent) Run(ctx Context) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// TODO: verify&update the setup here. Should we branch etc.
		ctx := NewContext(ctx, a, ctx.UserContent(), ctx.Artifacts(), ctx.Session(), ctx.Memory(), ctx.Branch())

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
		event.LLMResponse = &model.LLMResponse{
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
		newContent, err := callback(ctx, agentEvent, agentError)
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
