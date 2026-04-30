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

package workflow

import (
	"errors"
	"fmt"
	"iter"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow/internal/engine"
)

// runSequential drives the workflow synchronously: ready nodes run in
// queue order, each on the calling goroutine. The single-flight execution
// model fits Phase 2 (no parallelism, no dynamic nodes, no resume).
//
// Phase 3 introduces per-node goroutines, the semaphore-based concurrency
// cap, and JoinNode fan-in. Phases 4-5 add resume, dynamic nodes, and HITL.
//
// The function returns an iter.Seq2 that, when consumed, drives the
// workflow forward and yields each emitted event in turn.
func (w *Workflow) runSequential(
	ic agent.InvocationContext,
	input any,
	parentPath string,
) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		path := engine.JoinPath(parentPath, w.Name(), 1)
		state := newRunState(w.graph, input, path)

		for !state.done() {
			ref := state.popReady()
			if ref == nil {
				break
			}
			node, ok := w.graph.nodes[ref.name]
			if !ok {
				yield(nil, fmt.Errorf("workflow %q: node %q missing from graph", w.Name(), ref.name))
				return
			}

			ctx := &NodeContext{
				InvocationContext: ic,
				nodePath:          engine.JoinPath(path, node.Name(), ref.runID),
				runID:             fmt.Sprintf("%d", ref.runID),
				actions: &session.EventActions{
					StateDelta:    map[string]any{},
					ArtifactDelta: map[string]int64{},
				},
			}
			em := newCollectingEmitter(ctx, ic, node)

			err := node.RunImpl(ctx, ref.input, em)

			// Forward every event the node emitted, even on error, so callers
			// can observe partial progress.
			for _, ev := range em.events {
				if !yield(ev, nil) {
					return
				}
			}
			if err != nil {
				yield(nil, fmt.Errorf("workflow %q: node %q: %w", w.Name(), node.Name(), err))
				return
			}

			// Use the last emitted Output value as the input to successors.
			// Multiple outputs from a streaming node funnel only the final
			// value forward in Phase 2; richer fan-out lands in Phase 3.
			var nextInput any
			if len(em.outputs) > 0 {
				nextInput = em.outputs[len(em.outputs)-1]
			}
			state.completeAndQueueSuccessors(ref.name, nextInput)
		}
	}
}

// runState is the orchestrator's working memory: a queue of ready node
// refs, plus a counter for per-name run IDs.
type runState struct {
	graph   *workflowGraph
	input   any
	queue   []readyRef
	runIDs  map[string]int
	visited map[string]bool
}

type readyRef struct {
	name  string
	runID int
	input any
}

func newRunState(g *workflowGraph, input any, _ string) *runState {
	s := &runState{
		graph:   g,
		input:   input,
		runIDs:  map[string]int{},
		visited: map[string]bool{},
	}
	// Seed initial ready set: every node directly downstream of START.
	for _, name := range g.startNames {
		s.runIDs[name]++
		s.queue = append(s.queue, readyRef{name: name, runID: s.runIDs[name], input: input})
	}
	return s
}

func (s *runState) done() bool { return len(s.queue) == 0 }

func (s *runState) popReady() *readyRef {
	if len(s.queue) == 0 {
		return nil
	}
	ref := s.queue[0]
	s.queue = s.queue[1:]
	return &ref
}

// completeAndQueueSuccessors records the output and enqueues every
// successor edge that fires (in Phase 2 every edge fires unconditionally).
func (s *runState) completeAndQueueSuccessors(name string, output any) {
	s.visited[name] = true
	for _, e := range s.graph.out[name] {
		// Phase 2: ignore Routes. Phase 3 adds route matching.
		_ = e.routes
		s.runIDs[e.to]++
		s.queue = append(s.queue, readyRef{name: e.to, runID: s.runIDs[e.to], input: output})
	}
}

// collectingEmitter is the in-memory EventEmitter implementation. It
// records every call so the orchestrator can replay them onto the
// workflow's iter.Seq2.
//
// Each em.Output(value) becomes a *session.Event whose
// Actions.NodeInfo.Output is set, so resume (Phase 4) can reconstruct
// node state by scanning events.
type collectingEmitter struct {
	ctx     *NodeContext
	ic      agent.InvocationContext
	node    Node
	events  []*session.Event
	outputs []any
}

func newCollectingEmitter(ctx *NodeContext, ic agent.InvocationContext, node Node) *collectingEmitter {
	return &collectingEmitter{ctx: ctx, ic: ic, node: node}
}

// Event forwards a pre-built event verbatim, enriching NodeInfo with the
// node's path so resume scans can find it.
func (e *collectingEmitter) Event(ev *session.Event) error {
	if ev == nil {
		return errors.New("collectingEmitter: nil event")
	}
	e.attachNodeInfo(ev, false)
	e.events = append(e.events, ev)
	return nil
}

// Output produces an event whose Content is empty and whose
// NodeInfo.Output carries the value. Mirrors adk-python's behavior of
// yielding "raw output" through a generated Event.
func (e *collectingEmitter) Output(v any) error {
	ev := session.NewEvent(e.ic.InvocationID())
	ev.Author = e.node.Name()
	ev.Branch = e.ic.Branch()
	ev.LLMResponse = model.LLMResponse{}
	ev.Actions.StateDelta = e.ctx.Actions().StateDelta
	ev.Actions.ArtifactDelta = e.ctx.Actions().ArtifactDelta
	e.attachNodeInfo(ev, false)
	ev.Actions.NodeInfo.Output = v
	e.events = append(e.events, ev)
	e.outputs = append(e.outputs, v)
	return nil
}

// RequestInput is a placeholder in Phase 2. The full HITL implementation
// (event encoding, LongRunningToolIDs population, resume hook) lands in
// Phase 5. For now the emitter records the request so tests can verify it
// was raised.
func (e *collectingEmitter) RequestInput(r RequestInput) error {
	ev := session.NewEvent(e.ic.InvocationID())
	ev.Author = e.node.Name()
	ev.Branch = e.ic.Branch()
	e.attachNodeInfo(ev, true)
	ev.Actions.NodeInfo.InterruptID = r.InterruptID
	e.events = append(e.events, ev)
	return nil
}

// StateDelta merges the supplied delta into the NodeContext's actions.
func (e *collectingEmitter) StateDelta(delta map[string]any) error {
	for k, v := range delta {
		e.ctx.Actions().StateDelta[k] = v
	}
	return nil
}

// ArtifactDelta merges the supplied delta into the NodeContext's actions.
func (e *collectingEmitter) ArtifactDelta(delta map[string]int64) error {
	for k, v := range delta {
		e.ctx.Actions().ArtifactDelta[k] = v
	}
	return nil
}

// attachNodeInfo ensures ev.Actions.NodeInfo is populated with the node's
// hierarchical path and (when relevant) the interrupt marker.
func (e *collectingEmitter) attachNodeInfo(ev *session.Event, interrupt bool) {
	if ev.Actions.NodeInfo == nil {
		ev.Actions.NodeInfo = &session.NodeInfo{Path: e.ctx.NodePath(), Interrupt: interrupt}
	} else {
		ev.Actions.NodeInfo.Path = e.ctx.NodePath()
		if interrupt {
			ev.Actions.NodeInfo.Interrupt = true
		}
	}
}
