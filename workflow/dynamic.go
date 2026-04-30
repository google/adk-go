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
	"strings"
	"sync"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow/internal/engine"
)

// ErrNodeInterrupted is returned by NodeContext.RunNode when the dynamic
// child raised a HITL interrupt. The interrupt IDs propagate up via the
// parent NodeContext's actions; resume populates them with user responses.
var ErrNodeInterrupted = errors.New("workflow: dynamic node interrupted")

// RunNodeOpts customizes a dynamic-node invocation.
type RunNodeOpts struct {
	// Name overrides the default node name when forming the dynamic node's
	// path. Use this to disambiguate multiple invocations of the same node
	// type within one parent. Defaults to node.Name().
	Name string

	// RunID overrides the per-name run counter. Mostly useful for tests;
	// the scheduler picks a fresh counter per invocation otherwise.
	RunID int
}

// dynamicScheduler tracks dynamic-node runs scoped to a single parent
// NodeContext. It is created on demand the first time RunNode is called on
// a given context.
type dynamicScheduler struct {
	mu      sync.Mutex
	runs    map[string]*dynamicRun
	counter map[string]int
}

type dynamicRun struct {
	output any
	err    error
	// interruptIDs that this run is waiting on (Phase 5b: HITL).
	interruptIDs map[string]struct{}
	completed    bool
}

func newDynamicScheduler() *dynamicScheduler {
	return &dynamicScheduler{
		runs:    map[string]*dynamicRun{},
		counter: map[string]int{},
	}
}

// RunNode invokes a child node from inside a parent node's RunImpl.
// Synchronous: blocks the calling goroutine until the child completes.
// Events emitted by the child are forwarded into the parent's emitter (so
// they reach the workflow's iter.Seq2 in order).
//
// Dedup: the scheduler scans session events for prior runs at the
// resolved node_path. If a completed prior run is found, its output is
// returned immediately without executing the node. Mirrors adk-python's
// DynamicNodeScheduler at workflow/_dynamic_node_scheduler.py:111-253.
func (c *NodeContext) RunNode(node Node, input any, opts ...RunNodeOpts) (any, error) {
	if node == nil {
		return nil, errors.New("workflow: RunNode called with nil node")
	}

	var opt RunNodeOpts
	if len(opts) > 0 {
		opt = opts[0]
	}
	name := opt.Name
	if name == "" {
		name = node.Name()
	}

	if c.scheduler == nil {
		c.scheduler = newDynamicScheduler()
	}
	sch := c.scheduler

	// Resolve runID. If caller specified one, honor it; otherwise pick the
	// next per-name counter.
	sch.mu.Lock()
	runID := opt.RunID
	if runID == 0 {
		sch.counter[name]++
		runID = sch.counter[name]
	}
	path := engine.JoinPath(c.NodePath(), name, runID)

	if existing, ok := sch.runs[path]; ok {
		sch.mu.Unlock()
		if !existing.completed {
			// Should not happen with synchronous scheduling; signal a bug.
			return nil, fmt.Errorf("workflow: dynamic node %q already in flight", path)
		}
		return existing.output, existing.err
	}

	// Lazy event scan: did this exact node_path complete in a prior
	// invocation? If so, return cached output without re-running (unless
	// the node is RerunOnResume).
	if cached, ok := scanPriorDynamicRun(c.InvocationContext, path); ok {
		if !node.Spec().RerunOnResume {
			run := &dynamicRun{output: cached, completed: true}
			sch.runs[path] = run
			sch.mu.Unlock()
			return cached, nil
		}
	}

	run := &dynamicRun{interruptIDs: map[string]struct{}{}}
	sch.runs[path] = run
	sch.mu.Unlock()

	// Execute on the calling goroutine. Events flow into a per-call
	// emitter that forwards into the parent's actions accumulator.
	em := newCollectingEmitter(node, c.InvocationContext, runID, path)
	child := &NodeContext{
		InvocationContext: c.InvocationContext,
		nodePath:          path,
		runID:             fmt.Sprintf("%d", runID),
		actions: &session.EventActions{
			StateDelta:    map[string]any{},
			ArtifactDelta: map[string]int64{},
		},
	}
	em.ctx = child
	err := node.RunImpl(child, input, em)

	sch.mu.Lock()
	defer sch.mu.Unlock()
	run.completed = true
	if err != nil {
		run.err = err
		return nil, err
	}
	if len(em.outputs) > 0 {
		run.output = em.outputs[len(em.outputs)-1]
	}
	// Forward emitted events through the parent's emitter chain.
	if c.parentEmitter != nil {
		for _, ev := range em.events {
			if e := c.parentEmitter.Event(ev); e != nil {
				return nil, e
			}
		}
	}
	return run.output, nil
}

// scanPriorDynamicRun checks session events for a completed run at
// nodePath. Returns (output, true) if found.
func scanPriorDynamicRun(ic agent.InvocationContext, nodePath string) (any, bool) {
	if ic == nil {
		return nil, false
	}
	sess := ic.Session()
	if sess == nil {
		return nil, false
	}
	evs := sess.Events()
	if evs == nil {
		return nil, false
	}
	for ev := range evs.All() {
		if ev == nil || ev.Actions.NodeInfo == nil {
			continue
		}
		if ev.Actions.NodeInfo.Path == nodePath && ev.Actions.NodeInfo.Output != nil {
			return ev.Actions.NodeInfo.Output, true
		}
	}
	return nil, false
}

// validateNodePath is a guard for path arithmetic in RunNode tests.
func validateNodePath(p string) error {
	if !strings.Contains(p, "@") {
		return fmt.Errorf("workflow: malformed node path %q (missing @)", p)
	}
	return nil
}
