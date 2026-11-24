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
	"context"
	"errors"
	"sync"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

var (
	// errEmitAfterReturn is returned by a node's emit called after the
	// node body returned.
	errEmitAfterReturn = errors.New("workflow: emit called after the node body returned")

	// errDetachedUndeliverable fails a detached RunNode child that no
	// workflow can take any more: its workflow has finished, or its node
	// ran outside one.
	errDetachedUndeliverable = errors.New("workflow: no running workflow to take a detached node")

	// errDetachedInterrupted is the workflow's error when a detached
	// child requests human input. Mirrors the RuntimeError of adk-python's
	// Workflow._surface_detached_dynamic_outcome.
	errDetachedInterrupted = errors.New("workflow: a dynamic node started without waiting for RunNode requested human input, but a detached node cannot be resumed; wait for RunNode instead")
)

// detachedRuns tracks the RunNode children that a dynamic node body
// started and did not wait for (detached children), for one workflow
// scheduler, so the workflow waits for them before it finishes.
//
// A child is detached when the body that started it returns while the
// child is still running, or when its RunNode is called only after
// that. From then on its events no longer go through the node's
// iterator, which has ended, but straight to the scheduler's queue, as
// adk-python's detached child, an asyncio task, enqueues its events on
// the invocation (workflow/_node_runner.py). The scheduler yields them,
// keeps running until every detached child has finished, and fails the
// workflow when one failed or requested human input, as adk-python's
// Workflow._run_loop does (workflow/_workflow.py:405-458 at v2.12.0).
//
// Two differences from adk-python are deliberate or inherent. A
// detached child's failure fails the workflow whenever it happens;
// adk-python inspects only the children still running when the graph
// ends, as it cannot tell a finished detached run from an awaited one
// (a limitation its tests pin). And a detached child runs under its
// parent node's context, where an asyncio task outlives its parent:
// whatever ends that context ends the child too, such as the parent's
// NodeConfig.Timeout, which fails the workflow, or a ParallelWorker
// finishing the item that started it, which cancels the child.
//
// detachedRuns is safe for concurrent use.
type detachedRuns struct {
	queue chan<- queueItem

	mu sync.Mutex
	// pending counts registered children whose outcome the consumer has
	// not handled; sending counts events sent but not yet handled. The
	// consumer reads the queue until it closes the tracker, which it
	// does only when both are zero, so a send to the queue never blocks
	// forever.
	pending int
	sending int
	closed  bool
}

// detachedEventItem carries one event of a detached child to the
// consumer. processed is the same handshake as eventItem's.
type detachedEventItem struct {
	ev        *session.Event
	branch    string
	processed chan struct{}
}

func (detachedEventItem) isQueueItem() {}

// detachedDoneItem reports a detached child's outcome. Exactly one is
// sent for each successful register.
type detachedDoneItem struct {
	childPath string
	err       error
}

func (detachedDoneItem) isQueueItem() {}

type detachedRunsKey struct{}

// withDetachedRuns returns ctx carrying d, for the dynamic nodes and
// emitting function nodes that run under it.
func withDetachedRuns(ctx agent.Context, d *detachedRuns) agent.Context {
	base := context.WithValue(ctx, detachedRunsKey{}, d)
	return ctx.WithDelta(&agent.CommonContextDelta{
		InvocationContextDelta: &agent.InvocationContextDelta{Context: &base},
	})
}

// detachedRunsFrom returns the tracker of the workflow that ctx runs
// under, or nil outside a workflow.
func detachedRunsFrom(ctx context.Context) *detachedRuns {
	d, _ := ctx.Value(detachedRunsKey{}).(*detachedRuns)
	return d
}

// register counts a detached child, which must later call finish once.
// It reports false, and counts nothing, once the workflow has finished.
func (d *detachedRuns) register() bool {
	if d == nil {
		return false
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return false
	}
	d.pending++
	return true
}

// finish reports the outcome of a registered child.
func (d *detachedRuns) finish(childPath string, err error) {
	d.queue <- detachedDoneItem{childPath: childPath, err: err}
}

// send delivers ev, an event of a detached child, to the consumer, and
// waits until the consumer has handled it. branch is the branch of the
// node activation that started the child.
func (d *detachedRuns) send(ev *session.Event, branch string) error {
	if d == nil {
		return errDetachedUndeliverable
	}
	d.mu.Lock()
	if d.closed {
		d.mu.Unlock()
		return errDetachedUndeliverable
	}
	d.sending++
	d.mu.Unlock()

	var processed chan struct{}
	if ev != nil && !ev.LLMResponse.Partial {
		processed = make(chan struct{})
	}
	d.queue <- detachedEventItem{ev: ev, branch: branch, processed: processed}
	if processed != nil {
		<-processed
	}
	return nil
}

// handled records, on the consumer, that it has handled an item: an
// event when childDone is false, a child's outcome when it is true.
func (d *detachedRuns) handled(childDone bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if childDone {
		d.pending--
	} else {
		d.sending--
	}
}

// closeIfIdle closes the tracker when no child is pending and no event
// is on its way, and reports whether it is closed. Once closed, register
// and send fail with errDetachedUndeliverable.
func (d *detachedRuns) closeIfIdle() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.pending == 0 && d.sending == 0 {
		d.closed = true
	}
	return d.closed
}

// detachedOutcome returns the error a detached child's outcome gives
// its workflow, or nil. A child that waits for output without
// producing it is not an error: adk-python checks only a detached
// child's error and interrupt IDs. Nor is a canceled child, which
// adk-python skips (task.cancelled()): when the workflow canceled it,
// the workflow already has its own error, and when its parent node's
// context was canceled, for example by a ParallelWorker finishing the
// item that started it, nothing failed.
func detachedOutcome(it detachedDoneItem) error {
	switch {
	case it.err == nil, errors.Is(it.err, ErrNodeWaitingForOutput), errors.Is(it.err, context.Canceled):
		return nil
	case errors.Is(it.err, ErrNodeInterrupted):
		return &NodeRunError{ChildPath: it.childPath, Cause: errDetachedInterrupted}
	default:
		return it.err
	}
}
