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
	"sync"

	"google.golang.org/adk/v2/session"
)

// holds Agent internal state
type Agent interface {
	internal() *State
}

type State struct {
	AgentType Type
	Config    any

	// OnBeforeAgentCallbackEvent, if set, is called with the event produced by
	// the before-agent callbacks, before it is yielded. It lets an agent type
	// give a short-circuit response the same handling as one from its own run.
	// When it returns an error, the error is yielded in place of the event.
	OnBeforeAgentCallbackEvent func(*session.Event) error

	// Parent is the agent whose agent.New listed this agent in its
	// Config.SubAgents. It is the parent's inner agent (the value agent.New
	// built), set on the sub-agent's inner state, and nil for an agent that no
	// agent.New call has listed. An agent listed by two parents holds the one
	// whose link was published last.
	//
	// linkMu guards Parent and Owner: Link writes both, and code that can run
	// while another goroutine constructs an agent reads them through ParentOf
	// and OwnerOf. Link publishes a parent only after its constructor has
	// finished, so a reader that sees a parent sees it completely built.
	Parent Agent

	// Owner is the value a constructor returned when that value wraps this
	// inner agent (llmagent.New sets it to the *llmAgent). Nil when the inner
	// agent is itself what the constructor returned. Guarded by linkMu; see
	// Parent.
	Owner any
}

// linkMu guards the Parent and Owner fields of every State. One lock for
// all agents keeps a parent's Owner and its children's Parent links
// consistent with each other, and the links are written only while agents
// are constructed.
var linkMu sync.Mutex

// Link sets the Owner of parent's state to owner and records parent as the
// Parent of each state in children, in one critical section. A constructor
// calls it as its last step: a goroutine that then reads the link through
// ParentOf also sees owner and every write the constructor made before Link.
func Link(parent Agent, owner any, children []*State) {
	linkMu.Lock()
	defer linkMu.Unlock()
	Reveal(parent).Owner = owner
	for _, child := range children {
		child.Parent = parent
	}
}

// ParentOf returns the Parent of s and that parent's Owner, or nil and nil
// when s has no parent.
func ParentOf(s *State) (parent Agent, owner any) {
	linkMu.Lock()
	defer linkMu.Unlock()
	if s.Parent == nil {
		return nil, nil
	}
	return s.Parent, Reveal(s.Parent).Owner
}

// OwnerOf returns the Owner of s.
func OwnerOf(s *State) any {
	linkMu.Lock()
	defer linkMu.Unlock()
	return s.Owner
}

// Package agent sets these functions in its init, because this package
// cannot import it. A function variable in an internal package keeps them
// reachable only from this module without adding API to package agent.
var (
	// NewUnlinked builds an agent from cfg, an agent.Config, with the
	// validation of agent.New, but does not link the agent to its
	// sub-agents. Calling link publishes the agent as their parent, with
	// owner as its Owner, through Link. A constructor that wraps the agent
	// calls link once its own value is complete; a failed constructor does
	// not call it and leaves the sub-agents as they were.
	NewUnlinked func(cfg any) (a Agent, link func(owner any), err error)

	// InnerState returns the state of the inner agent of a, a non-nil
	// agent.Agent: the agent agent.New built, reached through an LLM agent
	// or any type that embeds an agent.Agent.
	InnerState func(a any) *State
)

type Type string

const (
	TypeLLMAgent        Type = "LLMAgent"
	TypeLoopAgent       Type = "LoopAgent"
	TypeSequentialAgent Type = "SequentialAgent"
	TypeParallelAgent   Type = "ParallelAgent"
	TypeWorkflowAgent   Type = "WorkflowAgent"
	TypeCustomAgent     Type = "CustomAgent"
	TypeRemoteAgent     Type = "RemoteAgent"
)

func (s *State) internal() *State { return s }

func Reveal(a Agent) *State { return a.internal() }
