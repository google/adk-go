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

package runner

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agenttree"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/tool"
)

// toolsetCloseTimeout bounds one toolset's Close. A var so a test can
// shorten it rather than wait ten seconds, as defaultSummarizerTimeout is.
var toolsetCloseTimeout = 10 * time.Second

// Close releases the resources the runner's agent tree holds, then shuts down
// the runner's plugins.
//
// Close calls Close on every toolset that implements [io.Closer] and is
// reachable from the runner's root: the root agent or node, the toolsets of
// each LLM agent, their sub-agents, the agents a workflow graph runs
// (including nested workflows and parallel workers) and the child agent of an
// agent tool, whether an LLM agent or a workflow ToolNode holds the tool.
// Each distinct toolset is closed once, one after another, and has ten
// seconds to return. An agent's own toolsets are closed before those
// of the agents it leads to; beyond that the order is unspecified. After the
// last toolset, Close closes the runner's plugins.
//
// Every toolset and plugin is attempted. A toolset whose Close fails, panics
// or does not return in ten seconds never stops the next toolset or the
// plugins. Close returns nil when everything closed without error, and
// otherwise [errors.Join] of one error per failed toolset, in the order the
// toolsets were closed, followed by the plugin manager's error. A toolset
// error reads `close toolset "<name>": <err>` and wraps err. A timeout
// satisfies errors.Is(err, context.DeadlineExceeded). A panic is recovered
// and reported as `close toolset "<name>": panic of type <T>`; the panic
// value is not included.
//
// Close takes no context, so *Runner is an [io.Closer] and "defer r.Close()"
// is the way to shut a runner down. Nothing Close calls can be canceled, so
// a started close always finishes with the remaining toolsets and the
// plugins. Close writes no field of the Runner and may be called more than
// once and concurrently with itself, Run and RunLive. After Close, Run still
// works; an operation on a toolset that was closed fails with that toolset's
// own error (for the MCP toolset, mcp.ErrConnectionClosed).
//
// Toolsets that a workflow hides behind agent.New{Run: w.Run}, and agents a
// dynamic node creates while it runs, are not reached.
func (r *Runner) Close() error {
	var errs []error
	for _, ts := range r.closableToolsets() {
		if err := closeToolset(ts); err != nil {
			errs = append(errs, err)
		}
	}
	if err := r.pluginManager.Close(); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

// closableToolsets returns the toolsets reachable from the runner's root that
// implement io.Closer, once per identity (see toolsetWalk.toolset), in the
// order the walk first reaches them.
func (r *Runner) closableToolsets() []tool.Toolset {
	w := &toolsetWalk{agents: map[agent.Agent]struct{}{}, keys: map[any]struct{}{}}
	if r.rootNode != nil {
		for _, a := range agenttree.NodeAgents(r.rootNode) {
			w.agent(a)
		}
		for _, a := range agenttree.NodeToolAgents(r.rootNode) {
			w.agent(a)
		}
	} else {
		w.agent(r.rootAgent)
	}
	return w.found
}

// toolsetWalk collects the closable toolsets of an agent tree, depth first.
type toolsetWalk struct {
	agents map[agent.Agent]struct{} // agents already walked
	keys   map[any]struct{}         // identity keys of toolsets already seen
	found  []tool.Toolset
}

// agent walks a's own toolsets, the agents behind its agent tools, then the
// agents it leads to, then the agents behind the agent tools its workflow
// graph's ToolNodes run. An agent is walked once even when several paths lead
// to it.
func (w *toolsetWalk) agent(a agent.Agent) {
	if _, ok := w.agents[a]; ok {
		return
	}
	w.agents[a] = struct{}{}
	if llmAgent, ok := a.(llminternal.Agent); ok {
		state := llminternal.Reveal(llmAgent)
		for _, ts := range state.Toolsets {
			w.toolset(ts)
		}
		for _, t := range state.Tools {
			if wrapper, ok := t.(toolinternal.AgentWrapper); ok {
				w.agent(wrapper.WrappedAgent())
			}
		}
	}
	for _, child := range agenttree.Children(a) {
		w.agent(child)
	}
	for _, child := range agenttree.ToolAgents(a) {
		w.agent(child)
	}
}

// toolset records ts when it implements io.Closer, unless a closer with the
// same identity was recorded. The identity is that of the innermost toolset
// ts wraps, so two wrappers of one toolset count as one; the first-seen outer
// closer is the one closed, and it passes Close through. A toolset that is not
// a closer leaves no identity behind, so a wrapper without Close does not hide
// the toolset it wraps. A toolset that is not comparable has no identity and
// is recorded every time.
func (w *toolsetWalk) toolset(ts tool.Toolset) {
	if _, ok := ts.(io.Closer); !ok {
		return
	}
	inner := ts
	for {
		u, ok := inner.(toolinternal.ToolsetUnwrapper)
		if !ok {
			break
		}
		inner = u.Unwrap()
	}
	if key, ok := identityKey(inner); ok {
		if _, seen := w.keys[key]; seen {
			return
		}
		w.keys[key] = struct{}{}
	}
	w.found = append(w.found, ts)
}

// identityKey returns v as a map key when its dynamic value is comparable,
// and ok == false otherwise, so that using the key never panics.
// reflect.Value.Comparable also rejects a comparable struct whose interface
// field holds a value that is not comparable.
func identityKey(v any) (key any, ok bool) {
	if v == nil || !reflect.ValueOf(v).Comparable() {
		return nil, false
	}
	return v, true
}

// closeToolset calls Close on ts, which must implement io.Closer, in a
// goroutine of its own, and waits for it at most toolsetCloseTimeout. A panic
// in Close is recovered and reported by its type only, because the value may
// hold user data. A Close that does not return in time is abandoned: its
// goroutine ends when Close returns, since the channel it sends on is
// buffered.
func closeToolset(ts tool.Toolset) error {
	timeout := toolsetCloseTimeout
	done := make(chan error, 1)
	go func() {
		defer func() {
			if p := recover(); p != nil {
				done <- fmt.Errorf("panic of type %T", p)
			}
		}()
		done <- ts.(io.Closer).Close()
	}()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case err := <-done:
		if err != nil {
			return fmt.Errorf("close toolset %q: %w", ts.Name(), err)
		}
		return nil
	case <-timer.C:
		return fmt.Errorf("close toolset %q: no result after %v: %w", ts.Name(), timeout, context.DeadlineExceeded)
	}
}
