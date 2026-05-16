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
	"fmt"
	"sync"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// defaultEventQueueCapacity bounds the buffered channel between
// node-runner goroutines and the consumer. A small fixed capacity
// keeps backpressure tight without serialising producers.
const defaultEventQueueCapacity = 16

var (
	// ErrMultipleOutputs is returned when a node yields more than
	// one event whose Actions.StateDelta carries an "output" key.
	// A node activation may emit at most one output value.
	ErrMultipleOutputs = errors.New("workflow: node produced multiple events with output values; only one event per execution can carry output")

	// ErrMultipleRoutingEvents is returned when a node yields more
	// than one event whose Routes field is set. A node activation
	// may emit at most one routing decision.
	ErrMultipleRoutingEvents = errors.New("workflow: node produced multiple events with route tags; only one event per execution can specify routes")

	// ErrMultipleInputRequests is returned when a node yields more
	// than one event whose RequestedInput field is set. A node
	// activation may issue at most one human-input request.
	ErrMultipleInputRequests = errors.New("workflow: node produced multiple events with RequestedInput; only one human-input request per execution is allowed")
)

// scheduler drives a single Workflow.Run invocation. It owns the
// per-node task table, the event channel, lifecycle counters, and
// the parent invocation context — none of which survive across
// processes. The persistable view of the same run (node statuses,
// inputs, triggers) lives on the embedded *RunState.
//
// Concurrency model: producer-consumer over eventQueue.
//
//   - Producers are the per-node goroutines started by scheduleNode.
//     They only send to eventQueue (events and a final completion);
//     they never read from it and never touch any other scheduler
//     field.
//   - The consumer is the single goroutine running scheduler.run
//     (the caller of Workflow.Run). It is the only reader of
//     eventQueue and the only mutator of runsByName, runCancels,
//     state.Nodes, and per-node accumulators.
//
// Because producers and the consumer share only eventQueue (a
// channel — already safe for concurrent use), the consumer-only
// fields below need no mutex.
type scheduler struct {
	state       *RunState       // persisted lifecycle state
	graph       *graph          // adjacency, terminal lookup
	nodesByName map[string]Node // built once at construction; lets handleCompletion resolve a completion's name back to its Node in O(1)

	// Per-node accumulators, created when a node is scheduled and
	// deleted on its completion. Owned by the consumer goroutine.
	runsByName map[string]*nodeRun

	// Per-node cancel funcs. Owned by the consumer goroutine.
	runCancels map[string]context.CancelFunc

	// eventQueue carries events and completions from producer
	// goroutines to the consumer.
	eventQueue chan queueItem
	wg         sync.WaitGroup

	parentCtx agent.InvocationContext
}

// nodeRun is the per-node consumer-side accumulator. It buffers the
// node's routing event (if any) and its single output between the
// time those events arrive and the time the node's completion
// arrives, at which point successors are scheduled.
//
// The single-output and single-routing-event constraints are
// enforced here: a second event carrying either kind sets err
// without overwriting the first value, and the consumer surfaces
// the error at completion.
type nodeRun struct {
	routingEvent *session.Event        // at most one; multiple is an error
	output       any                   // single StateDelta["output"]; nil if hasOutput is false
	hasOutput    bool                  // distinguishes "no output yet" from "output was nil"
	inputRequest *session.RequestInput // at most one human-input request; multiple is an error
	err          error                 // set on duplicate output, duplicate routing event, or duplicate input request
}

// recordErr stores err as the accumulator's first error. Subsequent
// calls are no-ops, preserving the first failure for handleCompletion
// to surface.
func (nr *nodeRun) recordErr(err error) {
	if nr.err == nil {
		nr.err = err
	}
}

// setRoutingEvent stores ev as the node's single routing event. A
// second call records ErrMultipleRoutingEvents instead of overwriting.
func (nr *nodeRun) setRoutingEvent(ev *session.Event, nodeName string) {
	if nr.routingEvent != nil {
		nr.recordErr(fmt.Errorf("%w: node %q", ErrMultipleRoutingEvents, nodeName))
		return
	}
	nr.routingEvent = ev
}

// setInputRequest stores req as the node's single in-flight
// human-input request. A second call records
// ErrMultipleInputRequests instead of overwriting; the consumer
// surfaces the error at completion and the node ends up
// NodeFailed (the waiting branch is gated on nr.err == nil so a
// node that requested twice does not silently park).
func (nr *nodeRun) setInputRequest(req *session.RequestInput, nodeName string) {
	if nr.inputRequest != nil {
		nr.recordErr(fmt.Errorf("%w: node %q", ErrMultipleInputRequests, nodeName))
		return
	}
	nr.inputRequest = req
}

// setOutput stores out as the node's single output value. A second
// call records ErrMultipleOutputs instead of overwriting.
func (nr *nodeRun) setOutput(out any, nodeName string) {
	if nr.hasOutput {
		nr.recordErr(fmt.Errorf("%w: node %q", ErrMultipleOutputs, nodeName))
		return
	}
	nr.output = out
	nr.hasOutput = true
}

// queueItem is sealed: only types in this package can satisfy it.
// The unexported sentinel method enforces the seal at compile time.
type queueItem interface{ isQueueItem() }

// eventItem carries one event from a node-runner goroutine to the
// consumer. nodeName is required so the consumer can correlate the
// event with the right nodeRun without relying on channel-FIFO-
// per-task semantics (which Go channels do not provide).
type eventItem struct {
	nodeName string
	ev       *session.Event
}

func (eventItem) isQueueItem() {}

// completionItem signals that a node-runner goroutine has finished.
// err is nil on success; non-nil errors are classified by the
// consumer via errors.Is (currently: context.Canceled,
// context.DeadlineExceeded, anything else → NodeFailed).
type completionItem struct {
	nodeName string
	err      error
}

func (completionItem) isQueueItem() {}

// newScheduler returns an initialised scheduler ready for the
// consumer to drive. The caller is responsible for seeding the
// initial trigger (typically Start with the user input).
func newScheduler(parent agent.InvocationContext, g *graph) *scheduler {
	return &scheduler{
		state:       NewRunState(),
		graph:       g,
		nodesByName: buildNodesByName(g),
		runsByName:  map[string]*nodeRun{},
		runCancels:  map[string]context.CancelFunc{},
		eventQueue:  make(chan queueItem, defaultEventQueueCapacity),
		parentCtx:   parent,
	}
}

// buildNodesByName walks the graph's edges and returns the name→Node
// lookup. Lets handleCompletion resolve a completion's node name to
// its instance in O(1) instead of scanning the full table.
func buildNodesByName(g *graph) map[string]Node {
	nodesByName := map[string]Node{}
	for n, edges := range g.successors {
		nodesByName[n.Name()] = n
		for _, e := range edges {
			nodesByName[e.To.Name()] = e.To
		}
	}
	return nodesByName
}

// scheduleNode launches a per-node goroutine for n with the given
// input. The node's lifecycle status transitions to NodeRunning, and
// the node is registered in runsByName and runCancels. The goroutine
// wrapper is responsible for pushing exactly one completionItem when
// it returns (success, error, panic, or cancellation).
//
// scheduleNode runs only on the consumer goroutine.
func (s *scheduler) scheduleNode(n Node, input any, triggeredBy string) {
	s.scheduleResumedNode(n, input, triggeredBy, nil)
}

// scheduleResumedNode is like scheduleNode but additionally
// injects resumeInputs into the per-node context, so re-entry
// nodes can read the user-supplied response payload via
// ctx.ResumedInput(interruptID). resumeInputs is keyed by
// InterruptID; nil disables re-entry semantics and yields the same
// behaviour as scheduleNode.
//
// scheduleResumedNode runs only on the consumer goroutine.
func (s *scheduler) scheduleResumedNode(n Node, input any, triggeredBy string, resumeInputs map[string]any) {
	name := n.Name()

	// Per-node context: WithTimeout when Config().Timeout > 0,
	// WithCancel otherwise. Either way it inherits from parentCtx,
	// so an ambient deadline on the workflow invocation still
	// applies.
	cfg := n.Config()
	var (
		nodeCtx context.Context
		cancel  context.CancelFunc
	)
	if cfg.Timeout > 0 {
		nodeCtx, cancel = context.WithTimeout(s.parentCtx, cfg.Timeout)
	} else {
		nodeCtx, cancel = context.WithCancel(s.parentCtx)
	}
	perNodeCtx := newNodeContext(s.parentCtx.WithContext(nodeCtx), triggeredBy, resumeInputs)

	ns := s.state.EnsureNode(name)
	ns.Status = NodeRunning
	ns.Input = input
	ns.TriggeredBy = triggeredBy
	s.runsByName[name] = &nodeRun{}

	s.runCancels[name] = cancel
	s.wg.Add(1)

	go runNode(s.eventQueue, &s.wg, name, n, perNodeCtx, input)
}

// runNode is the per-node goroutine wrapper. It drives the node's
// iter.Seq2, pushes events into the queue, and ends with exactly
// one completionItem. A panic in the node body is recovered and
// reported as a completion error so the consumer never deadlocks
// waiting for a vanished goroutine.
//
// Event sends select on ctx.Done(): if the scheduler has cancelled
// this node, an in-progress send to a full eventQueue does not
// deadlock — the goroutine drops the pending event and proceeds to
// completion. The completion send is unconditional because the
// consumer's runsByName bookkeeping relies on it.
func runNode(
	out chan<- queueItem,
	wg *sync.WaitGroup,
	name string,
	n Node,
	ctx agent.InvocationContext,
	input any,
) {
	defer wg.Done()

	// completion holds the final completionItem. It is sent in the
	// outer defer so panic recovery, normal exit, and cancellation
	// all funnel through the same send path.
	completion := completionItem{nodeName: name}
	defer func() {
		if r := recover(); r != nil {
			completion.err = fmt.Errorf("node %q panicked: %v", name, r)
		}
		out <- completion
	}()

	for ev, err := range n.Run(ctx, input) {
		if err != nil {
			completion.err = err
			return
		}
		select {
		case out <- eventItem{nodeName: name, ev: ev}:
		case <-ctx.Done():
			completion.err = ctx.Err()
			return
		}
	}
	// If the node's iter returned cleanly but the context was
	// cancelled or its deadline elapsed, surface that as the
	// completion error: the node likely returned because it observed
	// ctx.Done(), and the consumer needs to classify it.
	if ctxErr := ctx.Err(); ctxErr != nil {
		completion.err = ctxErr
	}
}

// cancelAll cancels every running task. Idempotent: cancelled
// goroutines may still push events that already left the producer
// before observing ctx.Done(); the consumer continues draining
// until runsByName is empty.
//
// cancelAll runs only on the consumer goroutine.
func (s *scheduler) cancelAll() {
	for _, cancel := range s.runCancels {
		cancel()
	}
}

// run is the single-consumer loop. It drains the eventQueue, applies
// state-side effects, yields events to the caller, and schedules
// successor nodes when a node completes. Returns when all running
// tasks have signalled completion.
//
// On non-nil yield-return-false (caller broke from the range loop)
// or on a non-retryable node error, run cancels all in-flight
// tasks and continues draining until runsByName is empty, then
// surfaces the original error (if any) via yield.
//
// run runs on the caller's goroutine (the goroutine that called
// Workflow.Run); it is the only mutator of state.Nodes and the
// node-side accumulators.
func (s *scheduler) run(yield func(*session.Event, error) bool) {
	var pendingErr error // first non-nil node error; surfaced after drain
	draining := false    // true once cancelAll has run; remaining queue items are drained without yielding or scheduling new successors

	for len(s.runsByName) > 0 {
		item := <-s.eventQueue
		switch it := item.(type) {
		case eventItem:
			s.handleEvent(it)
			if !draining {
				if !yield(it.ev, nil) {
					draining = true
					s.cancelAll()
				}
			}
		case completionItem:
			err := s.handleCompletion(it, !draining)
			if err != nil && pendingErr == nil {
				pendingErr = err
				if !draining {
					draining = true
					s.cancelAll()
				}
			}
		}
	}

	// All goroutines have returned and pushed their final events.
	// Surface the first error to the caller, unless we're already
	// draining.
	if pendingErr != nil {
		yield(nil, pendingErr)
	}
}

// handleEvent updates the per-node accumulator and is called once
// per event. The event itself has already been read from the queue
// and will be yielded to the caller by the consumer loop.
func (s *scheduler) handleEvent(it eventItem) {
	nr := s.runsByName[it.nodeName]
	if nr == nil {
		// Defensive: completion already processed for this node;
		// shouldn't happen if producer goroutines preserve send order.
		return
	}
	if it.ev == nil {
		return
	}
	if it.ev.Routes != nil {
		nr.setRoutingEvent(it.ev, it.nodeName)
	}
	if it.ev.RequestedInput != nil {
		nr.setInputRequest(it.ev.RequestedInput, it.nodeName)
	}
	if it.ev.Actions.StateDelta != nil {
		if out, ok := it.ev.Actions.StateDelta["output"]; ok {
			nr.setOutput(out, it.nodeName)
		}
	}
}

// handleCompletion finalises a node's run: transitions its lifecycle
// status, removes the live task, and (if scheduleSuccessors is true)
// schedules its successors. When the consumer is draining (caller
// stopped or a node failed), pass scheduleSuccessors=false so the
// workflow does not keep dispatching new nodes after cancellation.
//
// The returned error is the node's own error (NodeFailed); nil on
// clean success or sibling cancellation.
//
// # Human-input waiting branch
//
// When an activation completes cleanly and recorded a non-nil
// inputRequest (via setInputRequest from handleEvent), the node
// transitions to NodeWaiting instead of NodeCompleted, the request
// is persisted on NodeState.PendingRequest, and successors are not
// scheduled. The scheduler's main loop terminates naturally when
// every live node has either completed or moved into NodeWaiting,
// at which point Workflow.Run's iterator exhausts and the caller
// observes the pause by inspecting RunState.
//
// The waiting branch is checked after the error/cancel branches,
// so a node that fails for any reason (returned error, panic,
// context cancel, multiple-output, multiple-routing-event,
// multiple-input-request) does not silently park in NodeWaiting:
// failures take precedence and surface as NodeFailed.
func (s *scheduler) handleCompletion(it completionItem, scheduleSuccessors bool) error {
	ns := s.state.EnsureNode(it.nodeName)
	nr := s.runsByName[it.nodeName]
	delete(s.runsByName, it.nodeName)
	delete(s.runCancels, it.nodeName)

	if errors.Is(it.err, context.Canceled) {
		ns.Status = NodeCancelled
		return nil // sibling cancellation; not the original error
	}
	if it.err != nil {
		ns.Status = NodeFailed
		return it.err
	}
	if nr != nil && nr.err != nil {
		ns.Status = NodeFailed
		return nr.err
	}

	// Happy path: decide between NodeWaiting (a recorded human-
	// input request) or NodeCompleted. The waiting branch fires
	// regardless of the scheduleSuccessors flag — a request that
	// survived the run must be observable in RunState even when
	// the consumer is draining, so the caller can persist it.
	if nr != nil && nr.inputRequest != nil {
		ns.Status = NodeWaiting
		ns.PendingRequest = nr.inputRequest
		return nil
	}

	ns.Status = NodeCompleted
	// Release the accumulated re-entry response history; the node
	// has finished and a future activation (if any, e.g. via
	// loop-back routing) starts a fresh lifecycle.
	ns.ResumedInputs = nil

	if !scheduleSuccessors {
		return nil
	}

	// Schedule successors. Find them via the routing-aware helper,
	// which reads any routing event off this completion's accumulator.
	currentNode := s.nodesByName[it.nodeName]
	if currentNode == nil {
		return nil
	}
	var input any
	var routingEv *session.Event
	if nr != nil {
		input = nr.output
		routingEv = nr.routingEvent
	}
	// START's own output is empty by definition; for START we
	// propagate the workflow's seed input (carried as the START
	// node's NodeState.Input).
	if currentNode == Start {
		input = ns.Input
	}

	for _, succ := range findSuccessors(s.graph, currentNode, input, routingEv) {
		s.scheduleNode(succ.node, succ.input, succ.triggeredBy)
	}
	return nil
}

// successor is the per-target dispatch tuple produced by
// findSuccessors. The triggeredBy field carries the upstream node's
// name for downstream visibility via ctx.TriggeredBy().
type successor struct {
	node        Node
	input       any
	triggeredBy string
}

// findSuccessors evaluates the outgoing edges of currentNode against
// the optional routing event and returns the dispatch list:
//
//   - Edges with no Route always fire (and do not suppress Default).
//   - Edges with a concrete Route fire only if Route.Matches(event) is true.
//   - Duplicate To targets are deduplicated (same target node may not
//     be queued twice for one parent activation).
//   - The Default edge fires when no concrete Route matched. An
//     unconditional edge does not count as a "match" for this
//     purpose, so a graph with one unconditional edge and one
//     Default edge fans out to both targets.
//   - If every outgoing edge has a concrete Route and none matched,
//     and no Default is present, the workflow silently dead-ends at
//     this node — by design, mirroring adk-python's routing semantics.
func findSuccessors(g *graph, currentNode Node, input any, event *session.Event) []successor {
	succs := g.successorsOf(currentNode)
	if len(succs) == 0 {
		return nil
	}
	from := currentNode.Name()
	concreteMatched := false // any concrete Route fired (controls Default)
	out := []successor{}
	added := map[Node]struct{}{}
	var defaultRouteNode Node
	for _, edge := range succs {
		if _, ok := added[edge.To]; ok {
			continue
		}
		if edge.Route == nil {
			out = append(out, successor{node: edge.To, input: input, triggeredBy: from})
			added[edge.To] = struct{}{}
			continue
		}
		if edge.Route == Default {
			defaultRouteNode = edge.To
			continue
		}
		if event != nil && edge.Route.Matches(event) {
			out = append(out, successor{node: edge.To, input: input, triggeredBy: from})
			added[edge.To] = struct{}{}
			concreteMatched = true
		}
	}
	if !concreteMatched && defaultRouteNode != nil {
		out = append(out, successor{node: defaultRouteNode, input: input, triggeredBy: from})
	}
	return out
}
