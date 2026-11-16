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

package runconfig

import (
	"maps"
	"sync"
)

// agentRuns is RunConfig's per-invocation record of the agents that ended and
// the restorable states of agents that have one, keyed by agent name. It ports
// adk-python's InvocationContext.end_of_agents and agent_states
// (agents/invocation_context.py). The zero value is empty and ready to use.
type agentRuns struct {
	mu     sync.Mutex
	states map[string]map[string]any // agent name -> stored state; never mutated in place, only replaced or deleted
	ended  map[string]bool           // agent name -> true once ended; false once reopened by a seeded state
}

// SeedAgentRuns fills the store with the states and ended marks the runner
// rebuilt from the session when it resumes an invocation. Call it once, before
// c is shared (before ToContext). It copies both maps and each state map, so
// the caller may keep using its own. A nil map seeds nothing, and a nil c does
// nothing.
func (c *RunConfig) SeedAgentRuns(states map[string]map[string]any, ended map[string]bool) {
	if c == nil {
		return
	}
	c.runs.mu.Lock()
	defer c.runs.mu.Unlock()
	if states != nil {
		c.runs.states = make(map[string]map[string]any, len(states))
		for name, st := range states {
			c.runs.states[name] = maps.Clone(st)
		}
	}
	if ended != nil {
		c.runs.ended = maps.Clone(ended)
	}
}

// AgentState returns the stored state of the agent named name and whether
// there is one. The map is shared with the store and must not be written. A nil
// c has no state.
func (c *RunConfig) AgentState(name string) (map[string]any, bool) {
	if c == nil {
		return nil, false
	}
	c.runs.mu.Lock()
	defer c.runs.mu.Unlock()
	st, ok := c.runs.states[name]
	return st, ok
}

// AgentEnded reports whether the agent named name has ended in this
// invocation: it was seeded as ended, or RecordAgentEnd marked it and no
// ResetAgentRuns cleared it since. A nil c reports false.
func (c *RunConfig) AgentEnded(name string) bool {
	if c == nil {
		return false
	}
	c.runs.mu.Lock()
	defer c.runs.mu.Unlock()
	return c.runs.ended[name]
}

// RecordAgentEnd marks the agent named name ended and drops its stored state.
// It does nothing unless c.Resumable, so a non-resumable run never records an
// end. It is idempotent, and a nil c does nothing.
func (c *RunConfig) RecordAgentEnd(name string) {
	if c == nil || !c.Resumable {
		return
	}
	c.runs.mu.Lock()
	defer c.runs.mu.Unlock()
	if c.runs.ended == nil {
		c.runs.ended = map[string]bool{}
	}
	c.runs.ended[name] = true
	delete(c.runs.states, name)
}

// ResetAgentRuns drops the stored state and the end mark of every agent in
// names, so that each starts afresh the next time it runs. A LoopAgent calls it
// for its descendants between passes. A nil c does nothing.
func (c *RunConfig) ResetAgentRuns(names []string) {
	if c == nil {
		return
	}
	c.runs.mu.Lock()
	defer c.runs.mu.Unlock()
	for _, name := range names {
		delete(c.runs.states, name)
		delete(c.runs.ended, name)
	}
}
