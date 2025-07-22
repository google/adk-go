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
	"fmt"

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
/*
	type MyCustomAgent struct {
	   *BaseAgent
	   ....
	}

	func NewMyCustomAgent(name string, opts ...agent.Option) (*MyCustomAgent, error) {
	   agent := &MyCustomAgent{}
	   base, err := agent.NewBaseAgent(name, agent, opts...)}
	   if err != nil { return nil, err }
	   agent.BaseAgent = base
	    ...
	   return agent, nil
	}
*/
type BaseAgent struct {
	name        string
	description string
	parentAgent adk.Agent
	subAgents   []adk.Agent

	self adk.Agent
}

// NewBaseAgent returns a BaseAgent that can be the base of the implementation agent.
func NewBaseAgent(name string, implementation adk.Agent, opts ...Option) (*BaseAgent, error) {
	if implementation == nil {
		return nil, fmt.Errorf("implementation is nil")
	}
	b := &BaseAgent{name: name, self: implementation}
	for _, opt := range opts {
		if err := opt.apply2Base(b); err != nil {
			return nil, err
		}
	}
	return b, nil
}

func (a *BaseAgent) Name() string           { return a.name }
func (a *BaseAgent) Description() string    { return a.description }
func (a *BaseAgent) Parent() adk.Agent      { return a.parentAgent }
func (a *BaseAgent) SubAgents() []adk.Agent { return a.subAgents }

// TODO: Should we export it as Base() and include it in the interface?
// That will allows custom agents to wrap its BaseAgent instead of embedding.
func (a *BaseAgent) _base_() *BaseAgent { return a }

func asBaseAgent(a adk.Agent) *BaseAgent {
	if b, ok := a.(interface{ _base_() *BaseAgent }); ok {
		return b._base_()
	}
	return nil
}

// AddSubAgents adds the agents to the subagent list.
func (a *BaseAgent) AddSubAgents(agents ...adk.Agent) error {
	names := map[string]bool{}
	for _, subagent := range a.subAgents {
		names[subagent.Name()] = true
	}
	// run sanity check (no duplicate name, no multiple parents)
	for _, subagent := range agents {
		name := subagent.Name()
		if names[name] {
			return fmt.Errorf("multiple subagents with the same name (%q) are not allowed", name)
		}
		if parent := subagent.Parent(); parent != nil {
			return fmt.Errorf("agent %q already has parent %q", name, parent.Name())
		}
		names[name] = true
	}

	// mutate.
	for _, subagent := range agents {
		a.subAgents = append(a.subAgents, subagent)
		if base := asBaseAgent(subagent); base != nil {
			base.parentAgent = a.self
		}
		// TODO: error if subagent is not a base agent?
	}
	return nil
}
