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

	"github.com/google/adk-go"
)

// Option configures the Base agent.
type Option interface {
	apply2Base(*AgentBase)
}

type optionFunc func(*AgentBase)

func (o optionFunc) apply2Base(b *AgentBase) { o(b) }

// WithName sets the agent name.
func WithName(name string) Option {
	return optionFunc(func(b *AgentBase) { b.name = name })
}

// WithDescription sets the agent description.
func WithDescription(desc string) Option {
	return optionFunc(func(b *AgentBase) { b.description = desc })
}

// AgentBase is the base agent implementation that offers
// shared, default agent behavior. All agents should extend this
// base agent and configure the [AgentBase.Self] from the constructor
// [NewAgentBase].

// For example,
//
//		type MyCustomAgent struct {
//		   *AgentBase
//		   ....
//		}
//
//		func NewMyCustomAgent(name string, opts ...agent.Option) *MyCustomAgent {
//		   a := &MyCustomAgent{AgentBase: agent.NewAgentBase(name, opts...)}
//		   a.Self = a  // This allows methods implemented in AgentBase can access *MyCustomAgent.
//	        ...
//		   return a
//		}
type AgentBase struct {
	name        string
	description string
	parentAgent adk.Agent
	subAgents   []adk.Agent

	Self adk.Agent
}

func NewAgentBase(name string, opts ...Option) *AgentBase {
	b := &AgentBase{name: name}
	for _, opt := range opts {
		opt.apply2Base(b)
	}
	return b
}

var _ adk.Agent = (*AgentBase)(nil)

func (a *AgentBase) Name() string           { return a.name }
func (a *AgentBase) Description() string    { return a.description }
func (a *AgentBase) Parent() adk.Agent      { return a.parentAgent }
func (a *AgentBase) SubAgents() []adk.Agent { return a.subAgents }
func (a *AgentBase) Run(ctx context.Context, parentCtx *adk.InvocationContext) iter.Seq2[*adk.Event, error] {
	panic("unimplemented")
}

func (a *AgentBase) _base_() *AgentBase { return a }

// AddSubAgents adds the agents to the subagent list.
func (a *AgentBase) AddSubAgents(agents ...adk.Agent) {
	for _, subagent := range agents {
		a.subAgents = append(a.subAgents, subagent)

		if s, ok := subagent.(interface{ _base_() *AgentBase }); ok {
			if base := s._base_(); base != nil {
				base.parentAgent = a.Self
			}
		}
	}
}
