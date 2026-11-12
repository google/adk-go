// Copyright 2026 Google LLC
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

// Package agentstate holds the checkpoints workflow agents save during a
// resumable invocation, so a sequence, loop or parallel agent that stopped on a
// long-running call continues where it was once the call is answered. It is the
// Go counterpart of adk-python's InvocationContext.agent_states and
// end_of_agents.
//
// A Store is only present in a resumable run (runner.Config.Resumable). Every
// helper here is a no-op without one, so a run that is not resumable behaves
// as it did before.
package agentstate

import (
	"context"
	"maps"
	"slices"
	"sync"
	"sync/atomic"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// Store holds the latest checkpoint of each agent in one invocation, by agent
// name. Parallel sub-agents share it, so it is safe for concurrent use.
type Store struct {
	mu     sync.Mutex
	states map[string]map[string]any
	ended  map[string]bool
}

// New returns an empty Store.
func New() *Store {
	return &Store{states: map[string]map[string]any{}, ended: map[string]bool{}}
}

// Populate loads the checkpoints that events of invocationID carry, oldest
// first, as adk-python's populate_invocation_agent_states does. root is the
// agent tree, used to reset an agent's sub-agents when it saves a checkpoint,
// the same way Checkpoint does while the invocation runs.
func (s *Store) Populate(events session.Events, invocationID string, root agent.Agent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for ev := range events.All() {
		if ev == nil || ev.InvocationID != invocationID {
			continue
		}
		switch {
		case ev.Actions.EndOfAgent:
			s.ended[ev.Author] = true
			delete(s.states, ev.Author)
		case ev.Actions.AgentState != nil:
			if a := root.FindAgent(ev.Author); a != nil {
				s.clearSubAgents(a)
			}
			s.states[ev.Author] = maps.Clone(ev.Actions.AgentState)
			s.ended[ev.Author] = false
		case ev.Author != "user" && ev.LLMResponse.Content != nil && s.states[ev.Author] == nil && !s.ended[ev.Author]:
			// An agent that saves no checkpoint of its own, such as an LLM
			// agent, still ran: an empty checkpoint marks it as one to resume.
			// Content after an agent's end, such as an after-agent callback's,
			// does not reopen it; a parent's next checkpoint does.
			s.states[ev.Author] = map[string]any{}
			s.ended[ev.Author] = false
		}
	}
}

// clearSubAgents forgets the checkpoints of a's sub-agents, at every depth.
// s.mu must be held.
func (s *Store) clearSubAgents(a agent.Agent) {
	for _, sub := range a.SubAgents() {
		delete(s.states, sub.Name())
		delete(s.ended, sub.Name())
		s.clearSubAgents(sub)
	}
}

// Load returns a copy of the checkpoint saved for name, and whether there is
// one.
func (s *Store) Load(name string) (map[string]any, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, ok := s.states[name]
	return maps.Clone(state), ok
}

// Ended reports whether name finished its run in this invocation.
func (s *Store) Ended(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ended[name]
}

// Resumes reports whether name stopped partway through this invocation and
// should be run again to continue it.
func (s *Store) Resumes(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.states[name]
	return ok && !s.ended[name]
}

// Checkpoint saves state for ctx's agent, which is about to start a sub-agent
// afresh, and returns the event that records it. Whatever the agent's
// sub-agents recorded before then, such as finishing on an earlier pass of a
// loop, no longer applies, so it is forgotten, as adk-python's
// reset_sub_agent_states does for a loop's next pass. When the store is
// rebuilt from events, the sub-agent's own later events reset it the same
// way.
func (s *Store) Checkpoint(ctx agent.InvocationContext, state map[string]any) *session.Event {
	name := ctx.Agent().Name()
	s.mu.Lock()
	s.clearSubAgents(ctx.Agent())
	s.states[name] = maps.Clone(state)
	s.ended[name] = false
	s.mu.Unlock()
	ev := newEvent(ctx, name)
	ev.Actions.AgentState = state
	return ev
}

// End records that the agent named name, run under ctx, has finished, and
// returns the event that records it.
func (s *Store) End(ctx agent.InvocationContext, name string) *session.Event {
	s.mu.Lock()
	s.ended[name] = true
	delete(s.states, name)
	s.mu.Unlock()
	ev := newEvent(ctx, name)
	ev.Actions.EndOfAgent = true
	return ev
}

func newEvent(ctx agent.InvocationContext, author string) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = author
	ev.Branch = ctx.Branch()
	return ev
}

type storeKey struct{}

// ToContext returns ctx carrying s.
func ToContext(ctx context.Context, s *Store) context.Context {
	return context.WithValue(ctx, storeKey{}, s)
}

// FromContext returns the Store of a resumable run, or nil.
func FromContext(ctx context.Context) *Store {
	s, _ := ctx.Value(storeKey{}).(*Store)
	return s
}

// Scope records whether a sub-agent run paused the invocation. A pause marks
// the scope it happens in and every scope above it, but not a sibling's, so
// the other branches of a parallel agent still finish, as in adk-python.
type Scope struct {
	paused atomic.Bool
	parent *Scope
}

type scopeKey struct{}

// NewScope returns ctx with a new Scope nested in ctx's, for running one
// sub-agent.
func NewScope(ctx agent.InvocationContext) (agent.InvocationContext, *Scope) {
	parent, _ := ctx.Value(scopeKey{}).(*Scope)
	scope := &Scope{parent: parent}
	return ctx.WithContext(context.WithValue(ctx, scopeKey{}, scope)), scope
}

// Paused reports whether the invocation paused within s.
func (s *Scope) Paused() bool {
	return s.paused.Load()
}

// MarkPaused records that the invocation paused within ctx's scope.
func MarkPaused(ctx context.Context) {
	for s, _ := ctx.Value(scopeKey{}).(*Scope); s != nil; s = s.parent {
		s.paused.Store(true)
	}
}

// Pauses reports whether ev holds a long-running call, which pauses the
// invocation until the call is answered, as adk-python's
// should_pause_invocation does.
func Pauses(ev *session.Event) bool {
	if ev == nil || ev.LLMResponse.Content == nil {
		return false
	}
	for _, part := range ev.LLMResponse.Content.Parts {
		if part != nil && part.FunctionCall != nil && slices.Contains(ev.LongRunningToolIDs, part.FunctionCall.ID) {
			return true
		}
	}
	return false
}

// Index returns the position among subAgents of the one named in state[key],
// or 0 when there is none, as when the sub-agent was removed since the
// checkpoint.
func Index(subAgents []agent.Agent, state map[string]any, key string) int {
	name, _ := state[key].(string)
	for i, sub := range subAgents {
		if sub.Name() == name {
			return i
		}
	}
	return 0
}

// Int returns state[key] as an int. A count read back from a session that
// stores JSON is a float64.
func Int(state map[string]any, key string) int {
	switch v := state[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	}
	return 0
}

// AwaitsAnswer reports whether ctx's agent, on ctx's branch, made a
// long-running call earlier in this invocation that the user has not answered
// yet. Run again on a later turn, such an agent has to keep waiting rather than
// call the model as if its calls were settled: as a parallel sibling of the
// agent whose call was answered, or after only some of the calls it made in
// one turn were answered.
func AwaitsAnswer(ctx agent.InvocationContext) bool {
	self := ctx.Agent().Name()
	var calls []string
	answered := map[string]bool{}
	for ev := range ctx.Session().Events().All() {
		if ev.InvocationID != ctx.InvocationID() || ev.LLMResponse.Content == nil {
			continue
		}
		for _, part := range ev.LLMResponse.Content.Parts {
			if part == nil {
				continue
			}
			if ev.Author == "user" && part.FunctionResponse != nil {
				answered[part.FunctionResponse.ID] = true
			}
			if ev.Author == self && ev.Branch == ctx.Branch() && part.FunctionCall != nil && slices.Contains(ev.LongRunningToolIDs, part.FunctionCall.ID) {
				calls = append(calls, part.FunctionCall.ID)
			}
		}
	}
	return slices.ContainsFunc(calls, func(id string) bool { return !answered[id] })
}

// TransferToResume returns the agent that ctx's agent, an LLM agent run again
// on a later turn, transferred to before the invocation paused, as
// adk-python's LlmAgent._get_subagent_to_resume does. That agent continues
// the invocation, so ctx's agent does not ask its model again. It returns ""
// when the agent should run as usual, as when the user answered the agent's
// own call.
func TransferToResume(ctx agent.InvocationContext) string {
	self := ctx.Agent().Name()
	var events []*session.Event
	// The latest user message of the invocation, the one being answered. It
	// is written on the root's branch, so it is looked for on every branch.
	var answer *session.Event
	for ev := range ctx.Session().Events().All() {
		if ev.InvocationID != ctx.InvocationID() {
			continue
		}
		if ev.Author == "user" {
			answer = ev
			continue
		}
		if ev.Branch == ctx.Branch() {
			events = append(events, ev)
		}
	}
	// Whether the answer is this agent's is settled before any earlier
	// transfer: an agent that transferred on a loop's earlier pass and then
	// made its own call keeps the answer.
	if answer != nil && answer.LLMResponse.Content != nil {
		for _, part := range answer.LLMResponse.Content.Parts {
			if part != nil && part.FunctionResponse != nil && callAuthor(events, part.FunctionResponse.ID) == self {
				return ""
			}
		}
	}
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].Author == self && events[i].Actions.TransferToAgent != "" {
			return events[i].Actions.TransferToAgent
		}
	}
	return ""
}

// callAuthor returns the author of the latest event in events with a call
// with the given id, or "".
func callAuthor(events []*session.Event, id string) string {
	for i := len(events) - 1; i >= 0; i-- {
		if content := events[i].LLMResponse.Content; content != nil {
			for _, part := range content.Parts {
				if part != nil && part.FunctionCall != nil && part.FunctionCall.ID == id {
					return events[i].Author
				}
			}
		}
	}
	return ""
}
