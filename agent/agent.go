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
	"iter"

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

	setParent(parent Agent)
}

type Builder struct {
	Name        string
	Description string
	SubAgents   []Agent

	BeforeAgent []Callback
	// TODO: verify if the interface would have "Run(Context) error" and agent will call agent.Context.Report(Event)
	Run        func(Context) iter.Seq2[*session.Event, error]
	AfterAgent []Callback
}

func (b Builder) Agent() Agent {
	a := &agent{
		name:        b.Name,
		description: b.Description,
		subAgents:   b.SubAgents,
		beforeAgent: b.BeforeAgent,
		run:         b.Run,
		afterAgent:  b.AfterAgent,
	}

	for _, sub := range b.SubAgents {
		sub.setParent(a)
	}

	return a
}

type Context interface {
	context.Context

	UserContent() *genai.Content
	InvocationID() string
	Branch() string
	AgentName() string

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
	return a.run(ctx)
}

func (a *agent) setParent(parent Agent) {
	a.parent = parent
}

var _ Agent = (*agent)(nil)
