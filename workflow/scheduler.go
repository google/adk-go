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
)

// scheduler drives a single Workflow.Run invocation. It owns the
// per-activation task table, the event channel, lifecycle counters,
// and the parent invocation context — none of which survive across
// processes. The persistable view of the same run (node statuses,
// inputs, triggers) lives on the embedded *RunState.
//
// Concurrency model: producer-consumer over eventQueue.
//
//   - Producers are the per-activation goroutines started by
//     activate. They only send to eventQueue (events and a final
//     completion); they never read from it and never touch any
//     other scheduler field.
//   - The consumer is the single goroutine running scheduler.run
//     (the caller of Workflow.Run). It is the only reader of
//     eventQueue and the only mutator of runsByActivation,
//     activationCancels, runningByName, triggerBuffer, joinState,
//     and state.Nodes.
//
// Because producers and the consumer share only eventQueue (a
// channel — already safe for concurrent use), the consumer-only
// fields below need no mutex.
//
// # Multiple concurrent activations of the same node
//
// A node identified by its Name may be activated multiple times
// during a single workflow run — once per upstream completion that
// triggers it (Python parity: there is no automatic merge for
// non-Join nodes). To keep each activation's bookkeeping
// independent, the scheduler keys per-task state by a monotonically
// increasing activationID rather than by node name.
//
// To preserve the ordering required by stateful fan-in nodes
// (JoinNode and any custom WaitForOutput=true node that accumulates
// across activations), the scheduler serialises activations of the
// same name: a second trigger arriving while the first is still
// running is buffered in triggerBuffer[name] and dispatched after
// the first activation completes. This guarantees that a JoinNode
// observes upstream activations one at a time and does not race on
// its own accumulator.
type scheduler struct {
	state       *RunState       // persisted lifecycle state
	graph       *graph          // adjacency, terminal lookup, predecessors
	nodesByName map[string]Node // built once at construction; lets handleCompletion resolve a completion's name back to its Node in O(1)

	// Monotonic per-scheduler counter; the next id handed out by
	// activate. Owned by the consumer goroutine (only mutated
	// there).
	nextActivationID uint64

	// runsByActivation holds the consumer-side accumulator for
	// every in-flight activation. Created on activate, deleted on
	// the activation's completion. Owned by the consumer.
	runsByActivation map[uint64]*nodeRun

	// activationCancels holds the cancel func for every in-flight
	// activation. Owned by the consumer.
	activationCancels map[uint64]context.CancelFunc

	// runningByName tracks how many in-flight activations exist
	// per node name. Used to decide whether a new trigger should
	// run immediately or be queued in triggerBuffer. Owned by the
	// consumer.
	runningByName map[string]int

	// triggerBuffer queues pending activations per node name, FIFO.
	// A trigger landing on a node that is already running is
	// pushed here and consumed by the consumer when the running
	// activation (or, for WaitForOutput nodes, the activation that
	// produced the output) completes. Owned by the consumer.
	triggerBuffer map[string][]pendingTrigger

	// joinAccumulators holds the per-node fan-in accumulator used
	// by JoinNode (and any custom node that accumulates inputs
	// across per-predecessor activations). Keyed by node Name. The
	// scheduler creates an entry lazily on first activation of a
	// fan-in node and clears it once the node emits its terminal
	// output (see handleCompletion). Owned by the consumer; node
	// implementations read/write it via the *joinAccumulator
	// reference handed in through the per-node context.
	joinAccumulators map[string]*joinAccumulator

	// eventQueue carries events and completions from producer
	// goroutines to the consumer.
	eventQueue chan queueItem
	wg         sync.WaitGroup

	parentCtx agent.InvocationContext
}

// pendingTrigger is one buffered activation request waiting for the
// node to be free.
type pendingTrigger struct {
	input       any
	triggeredBy string
}

// nodeRun is the per-activation consumer-side accumulator. It
// buffers the activation's routing event (if any) and its single
// output between the time those events arrive and the time the
// activation's completion arrives, at which point successors are
// scheduled.
//
// The single-output and single-routing-event constraints are
// enforced here: a second event carrying either kind sets err
// without overwriting the first value, and the consumer surfaces
// the error at completion.
type nodeRun struct {
	nodeName     string         // the activated node's Name; needed at completion to dispatch successors
	routingEvent *session.Event // at most one; multiple is an error
	output       any            // single StateDelta["output"]; nil if hasOutput is false
	hasOutput    bool           // distinguishes "no output yet" from "output was nil"
	err          error          // set on duplicate output or duplicate routing event
}

// recordErr stores err as the accumulator's first error. Subsequent
// calls are no-ops, preserving the first failure for handleCompletion
// to surface.
func (nr *nodeRun) recordErr(err error) {
	if nr.err == nil {
		nr.err = err
	}
}

// setRoutingEvent stores ev as the activation's single routing
// event. A second call records ErrMultipleRoutingEvents instead of
// overwriting.
func (nr *nodeRun) setRoutingEvent(ev *session.Event) {
	if nr.routingEvent != nil {
		nr.recordErr(fmt.Errorf("%w: node %q", ErrMultipleRoutingEvents, nr.nodeName))
		return
	}
	nr.routingEvent = ev
}

// setOutput stores out as the activation's single output value. A
// second call records ErrMultipleOutputs instead of overwriting.
func (nr *nodeRun) setOutput(out any) {
	if nr.hasOutput {
		nr.recordErr(fmt.Errorf("%w: node %q", ErrMultipleOutputs, nr.nodeName))
		return
	}
	nr.output = out
	nr.hasOutput = true
}

// queueItem is sealed: only types in this package can satisfy it.
// The unexported sentinel method enforces the seal at compile time.
type queueItem interface{ isQueueItem() }

// eventItem carries one event from a node-runner goroutine to the
// consumer. activationID is required so the consumer can correlate
// the event with the right nodeRun without relying on channel-FIFO-
// per-task semantics (which Go channels do not provide).
type eventItem struct {
	activationID uint64
	ev           *session.Event
}

func (eventItem) isQueueItem() {}

// completionItem signals that a node-runner goroutine has finished.
// err is nil on success; non-nil errors are classified by the
// consumer via errors.Is (currently: context.Canceled,
// context.DeadlineExceeded, anything else → NodeFailed).
type completionItem struct {
	activationID uint64
	err          error
}

func (completionItem) isQueueItem() {}

// newScheduler returns an initialised scheduler ready for the
// consumer to drive. The caller is responsible for seeding the
// initial trigger (typically Start with the user input).
func newScheduler(parent agent.InvocationContext, g *graph) *scheduler {
	return &scheduler{
		state:             NewRunState(),
		graph:             g,
		nodesByName:       buildNodesByName(g),
		runsByActivation:  map[uint64]*nodeRun{},
		activationCancels: map[uint64]context.CancelFunc{},
		runningByName:     map[string]int{},
		triggerBuffer:     map[string][]pendingTrigger{},
		joinAccumulators:  map[string]*joinAccumulator{},
		eventQueue:        make(chan queueItem, defaultEventQueueCapacity),
		parentCtx:         parent,
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

// trigger schedules n with the given input. If n already has an
// in-flight activation, the trigger is buffered and dispatched when
// the current activation completes. This per-name FIFO serialisation
// is what stateful fan-in nodes (JoinNode, custom WaitForOutput
// nodes) rely on to safely accumulate across per-predecessor
// activations.
//
// trigger runs only on the consumer goroutine.
func (s *scheduler) trigger(n Node, input any, triggeredBy string) {
	name := n.Name()
	if s.runningByName[name] > 0 {
		s.triggerBuffer[name] = append(s.triggerBuffer[name], pendingTrigger{input: input, triggeredBy: triggeredBy})
		return
	}
	s.activate(n, input, triggeredBy)
}

// activate launches a per-activation goroutine for n with the given
// input. The activation's lifecycle status transitions to
// NodeRunning, and the activation is registered in
// runsByActivation, activationCancels, and runningByName. The
// goroutine wrapper is responsible for pushing exactly one
// completionItem when it returns (success, error, panic, or
// cancellation).
//
// activate runs only on the consumer goroutine.
func (s *scheduler) activate(n Node, input any, triggeredBy string) {
	name := n.Name()

	// Per-activation context: WithTimeout when Config().Timeout > 0,
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
	// Hand the activation a stable per-name fan-in accumulator so
	// JoinNode (and any custom in-package fan-in node) can merge
	// across per-predecessor activations. The same *joinAccumulator
	// is reused on every activation of the same node name until
	// the node emits its terminal output, at which point
	// handleCompletion clears the entry.
	acc, ok := s.joinAccumulators[name]
	if !ok {
		acc = &joinAccumulator{}
		s.joinAccumulators[name] = acc
	}
	perNodeCtx := newNodeContext(s.parentCtx.WithContext(nodeCtx), triggeredBy, s.graph.inNodeNamesOf(n), acc)

	ns := s.state.EnsureNode(name)
	ns.Status = NodeRunning
	ns.Input = input
	ns.TriggeredBy = triggeredBy

	s.nextActivationID++
	id := s.nextActivationID
	s.runsByActivation[id] = &nodeRun{nodeName: name}
	s.activationCancels[id] = cancel
	s.runningByName[name]++
	s.wg.Add(1)

	go runNode(s.eventQueue, &s.wg, id, n, perNodeCtx, input)
}

// runNode is the per-activation goroutine wrapper. It drives the
// node's iter.Seq2, pushes events into the queue, and ends with
// exactly one completionItem. A panic in the node body is recovered
// and reported as a completion error so the consumer never
// deadlocks waiting for a vanished goroutine.
//
// Event sends select on ctx.Done(): if the scheduler has cancelled
// this activation, an in-progress send to a full eventQueue does
// not deadlock — the goroutine drops the pending event and proceeds
// to completion. The completion send is unconditional because the
// consumer's runsByActivation bookkeeping relies on it.
func runNode(
	out chan<- queueItem,
	wg *sync.WaitGroup,
	activationID uint64,
	n Node,
	ctx agent.InvocationContext,
	input any,
) {
	defer wg.Done()

	// completion holds the final completionItem. It is sent in the
	// outer defer so panic recovery, normal exit, and cancellation
	// all funnel through the same send path.
	completion := completionItem{activationID: activationID}
	defer func() {
		if r := recover(); r != nil {
			completion.err = fmt.Errorf("node %q panicked: %v", n.Name(), r)
		}
		out <- completion
	}()

	for ev, err := range n.Run(ctx, input) {
		if err != nil {
			completion.err = err
			return
		}
		select {
		case out <- eventItem{activationID: activationID, ev: ev}:
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

// cancelAll cancels every in-flight activation. Idempotent:
// cancelled goroutines may still push events that already left the
// producer before observing ctx.Done(); the consumer continues
// draining until runsByActivation is empty.
//
// cancelAll runs only on the consumer goroutine.
func (s *scheduler) cancelAll() {
	for _, cancel := range s.activationCancels {
		cancel()
	}
}

// run is the single-consumer loop. It drains the eventQueue, applies
// state-side effects, yields events to the caller, and schedules
// successor nodes when an activation completes. Returns when all
// in-flight activations have signalled completion.
//
// On non-nil yield-return-false (caller broke from the range loop)
// or on a non-retryable node error, run cancels all in-flight
// tasks and continues draining until runsByActivation is empty,
// then surfaces the original error (if any) via yield.
//
// run runs on the caller's goroutine (the goroutine that called
// Workflow.Run); it is the only mutator of state.Nodes,
// runsByActivation, runningByName, triggerBuffer, joinState, and
// the per-activation accumulators.
func (s *scheduler) run(yield func(*session.Event, error) bool) {
	var pendingErr error // first non-nil node error; surfaced after drain
	draining := false    // true once cancelAll has run; remaining queue items are drained without yielding or scheduling new activations

	for len(s.runsByActivation) > 0 {
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

// handleEvent updates the per-activation accumulator and is called
// once per event. The event itself has already been read from the
// queue and will be yielded to the caller by the consumer loop.
func (s *scheduler) handleEvent(it eventItem) {
	nr := s.runsByActivation[it.activationID]
	if nr == nil {
		// Defensive: completion already processed for this
		// activation; shouldn't happen if producer goroutines
		// preserve send order.
		return
	}
	if it.ev == nil {
		return
	}
	if it.ev.Routes != nil {
		nr.setRoutingEvent(it.ev)
	}
	if it.ev.Actions.StateDelta != nil {
		if out, ok := it.ev.Actions.StateDelta["output"]; ok {
			nr.setOutput(out)
		}
	}
}

// handleCompletion finalises an activation: transitions the node's
// lifecycle status, removes the live task, and (if
// scheduleSuccessors is true) schedules its successors. When the
// consumer is draining (caller stopped or a node failed), pass
// scheduleSuccessors=false so the workflow does not keep
// dispatching new activations after cancellation.
//
// The returned error is the activation's own error (NodeFailed); nil
// on clean success or sibling cancellation.
//
// # WaitForOutput / fan-in semantics
//
// When the completed node has Config().WaitForOutput=true and its
// activation produced no output (hasOutput=false), the node moves
// to NodeWaiting instead of NodeCompleted and successors are not
// scheduled. This is what JoinNode (and any custom fan-in node)
// uses to swallow per-predecessor activations until it has
// accumulated enough state to emit its merged output. The next
// buffered trigger for the same node, if any, is still drained
// after this completion — that is what ultimately drives the
// JoinNode forward to its terminal "all predecessors arrived"
// activation.
func (s *scheduler) handleCompletion(it completionItem, scheduleSuccessors bool) error {
	nr := s.runsByActivation[it.activationID]
	delete(s.runsByActivation, it.activationID)
	delete(s.activationCancels, it.activationID)

	if nr == nil {
		// Defensive: shouldn't happen, but if it does, we cannot
		// resolve the node name to dispatch successors.
		return nil
	}
	name := nr.nodeName
	s.runningByName[name]--
	if s.runningByName[name] <= 0 {
		delete(s.runningByName, name)
	}

	ns := s.state.EnsureNode(name)
	currentNode := s.nodesByName[name]

	switch {
	case it.err == nil:
		// fall through — final status is decided below by the
		// WaitForOutput branch.
	case errors.Is(it.err, context.Canceled):
		ns.Status = NodeCancelled
		s.drainBufferedTriggersIfFree(name, scheduleSuccessors)
		return nil // sibling cancellation; not the original error
	default:
		ns.Status = NodeFailed
		// Drop any buffered triggers for this node — the workflow
		// is going to drain.
		delete(s.triggerBuffer, name)
		return it.err
	}

	if nr.err != nil {
		ns.Status = NodeFailed
		delete(s.triggerBuffer, name)
		return nr.err
	}

	// WaitForOutput: a completed activation that did not produce an
	// output keeps the node in NodeWaiting and skips successor
	// scheduling. This is the JoinNode / fan-in path.
	cfg := NodeConfig{}
	if currentNode != nil {
		cfg = currentNode.Config()
	}
	waitForOutput := cfg.WaitForOutput != nil && *cfg.WaitForOutput
	if waitForOutput && !nr.hasOutput {
		ns.Status = NodeWaiting
		s.drainBufferedTriggersIfFree(name, scheduleSuccessors)
		return nil
	}

	ns.Status = NodeCompleted

	if !scheduleSuccessors || currentNode == nil {
		s.drainBufferedTriggersIfFree(name, scheduleSuccessors)
		return nil
	}

	// Once a JoinNode actually emits its terminal output we can
	// drop any leftover accumulator state — the join has fired
	// and any future re-entry should start fresh.
	if waitForOutput && nr.hasOutput {
		delete(s.joinAccumulators, name)
	}

	input := nr.output
	routingEv := nr.routingEvent
	// START's own output is empty by definition; for START we
	// propagate the workflow's seed input (carried as the START
	// node's NodeState.Input).
	if currentNode == Start {
		input = ns.Input
	}

	for _, succ := range findSuccessors(s.graph, currentNode, input, routingEv) {
		s.trigger(succ.node, succ.input, succ.triggeredBy)
	}

	// If a buffered trigger arrived for this same node while it
	// was running, dispatch the next one now that the slot is
	// free. (Successors of the just-finished activation may have
	// re-buffered into the same name if the graph contains a
	// cycle, but the contract is FIFO either way.)
	s.drainBufferedTriggersIfFree(name, scheduleSuccessors)
	return nil
}

// drainBufferedTriggersIfFree pops the next pending trigger for
// name, if the node is now free of in-flight activations and
// scheduling is still enabled. Called from every code path in
// handleCompletion so a buffered trigger never gets stranded.
//
// Only the head of the buffer is dispatched; subsequent buffered
// triggers will be dispatched one-by-one as each activation
// completes. This preserves the per-name serialisation that
// stateful fan-in nodes rely on.
func (s *scheduler) drainBufferedTriggersIfFree(name string, scheduleSuccessors bool) {
	if !scheduleSuccessors {
		// Workflow is draining; drop pending triggers for this
		// name. (They cannot escape into a cancelled engine.)
		delete(s.triggerBuffer, name)
		return
	}
	if s.runningByName[name] > 0 {
		return
	}
	pending, ok := s.triggerBuffer[name]
	if !ok || len(pending) == 0 {
		return
	}
	next := pending[0]
	if len(pending) == 1 {
		delete(s.triggerBuffer, name)
	} else {
		s.triggerBuffer[name] = pending[1:]
	}
	node := s.nodesByName[name]
	if node == nil {
		return
	}
	s.activate(node, next.input, next.triggeredBy)
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
