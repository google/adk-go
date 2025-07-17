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

	"github.com/google/adk-go"
)

// Option configures the Base agent.
type Option interface {
	apply2Base(*BaseAgent) error
}

type optionFunc func(*BaseAgent) error

func (o optionFunc) apply2Base(b *BaseAgent) error { return o(b) }

// WithName sets the agent name.
func WithName(name string) Option {
	return optionFunc(func(b *BaseAgent) error {
		b.name = name
		return nil
	})
}

// WithDescription sets the agent description.
func WithDescription(desc string) Option {
	return optionFunc(func(b *BaseAgent) error {
		b.description = desc
		return nil
	})
}

func WithSubAgents(agents ...adk.Agent) Option {
	return optionFunc(func(b *BaseAgent) error {
		return b.AddSubAgents(agents...)
	})
}

// BaseAgent is the base agent implementation that offers
// shared, default agent behavior. All agents should extend this
// base agent by embedding.

// For example,
//
//		type MyCustomAgent struct {
//		   *BaseAgent
//		   ....
//		}
//
//		func NewMyCustomAgent(name string, opts ...agent.Option) *MyCustomAgent {
//		   agent := &MyCustomAgent{}
//		   agent.BaseAgent = agent.NewBaseAgent(name, agent, opts...)}
//		   agent.Self = a  // This allows methods implemented in BaseAgent can access *MyCustomAgent.
//	        ...
//		   return agent
//		}
type BaseAgent struct {
	name        string
	description string
	parentAgent adk.Agent
	subAgents   []adk.Agent

	self adk.Agent
}

// NewBaseAgent returns a BaseAgent that can be the base of the implementation agent.
func NewBaseAgent(name string, implementation adk.Agent, opts ...Option) *BaseAgent {
	if implementation == nil {
		panic("implementation is nil")
	}
	b := &BaseAgent{name: name, self: implementation}
	for _, opt := range opts {
		if err := opt.apply2Base(b); err != nil {
			panic(err) // TODO: what do we do with error.
		}
	}
	return b
}

var _ adk.Agent = (*BaseAgent)(nil)

func (a *BaseAgent) Name() string           { return a.name }
func (a *BaseAgent) Description() string    { return a.description }
func (a *BaseAgent) Parent() adk.Agent      { return a.parentAgent }
func (a *BaseAgent) SubAgents() []adk.Agent { return a.subAgents }
func (a *BaseAgent) Run(ctx context.Context, parentCtx *adk.InvocationContext) iter.Seq2[*adk.Event, error] {
	panic("unimplemented")
}

// TODO: Should we export it as Base() and include it in the interface?
// That will allows custom agents to wrap its BaseAgent instead of embedding.
func (a *BaseAgent) _base_() *BaseAgent { return a }

func (a *BaseAgent) findSubAgent(name string) (adk.Agent, bool) {
	for _, s := range a.subAgents {
		if s.Name() == name {
			return a, true
		}
	}
	return nil, false
}

// AddSubAgents adds the agents to the subagent list.
func (a *BaseAgent) AddSubAgents(agents ...adk.Agent) error {
	for _, subagent := range agents {
		// O(n^2) search, but n is small enough.
		if _, found := a.findSubAgent(subagent.Name()); found {
			return fmt.Errorf("cannot register multiple agents with the same name: %q", subagent.Name())
		}

		a.subAgents = append(a.subAgents, subagent)

		if s, ok := subagent.(interface{ _base_() *BaseAgent }); ok {
			if base := s._base_(); base != nil {
				if base.parentAgent != nil {
					return fmt.Errorf("Agent(%q) is already a subagent of Agent(%q)", base.Name(), base.parentAgent.Name())
				}
				base.parentAgent = a.self
			}
		}
	}
	return nil
}
