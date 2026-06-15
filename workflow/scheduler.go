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
	"time"

	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/telemetry"
	"google.golang.org/adk/session"
)

// defaultEventQueueCapacity bounds the buffered channel between
// node-runner goroutines and the consumer. A small fixed capacity
// keeps backpressure tight without serialising producers.
const defaultEventQueueCapacity = 16

var (
	// ErrMultipleOutputs is returned when a node yields more than
	// one event with Event.Output set. A node activation may emit
	// at most one output value.
	ErrMultipleOutputs = errors.New("workflow: node produced multiple events with output values; only one event per execution can carry output")

	// ErrMultipleRoutingEvents is returned when a node yields more
	// than one event whose Routes field is set. A node activation
	// may emit at most one routing decision.
	ErrMultipleRoutingEvents = errors.New("workflow: node produced multiple events with route tags; only one event per execution can specify routes")
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

	// Timers for scheduled retries. Owned by the consumer goroutine.
	retryTimers map[string]*time.Timer

	// eventQueue carries events and completions from producer
	// goroutines to the consumer.
	eventQueue chan queueItem
	wg         sync.WaitGroup

	parentCtx agent.InvocationContext

	// maxConcurrency caps len(runsByName); 0 disables the cap.
	// When at the cap, scheduleResumedNode enqueues into
	// pendingQueue (status NodePending) and the consumer drains
	// pendingQueue via tryDispatchPending on each completion.
	// Set once at construction; immutable thereafter.
	maxConcurrency int

	// pendingQueue holds activations queued while the
	// concurrency cap is saturated. FIFO; nodes are dispatched
	// in arrival order as in-flight nodes complete. Owned by the
	// consumer goroutine.
	pendingQueue []pendingActivation
}

// pendingActivation is a deferred scheduleResumedNode call kept on
// the consumer-owned pendingQueue while the concurrency cap is
// saturated. Drained by tryDispatchPending on each completion.
type pendingActivation struct {
	node         Node
	input        any
	triggeredBy  string
	branch       string
	resumeInputs map[string]any // nil for non-resume schedules
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
	routingEvent *session.Event // at most one; multiple is an error
	output       any            // single Event.Output; nil if hasOutput is false
	hasOutput    bool           // distinguishes "no output yet" from "output was nil"
	err          error          // set on duplicate output or duplicate routing event
	branch       string         // composite branch assigned at scheduling; used to stamp Event.Branch when the node leaves it empty

	// interruptIDs are unresolved long-running tool call IDs raised by
	// the node's events. A non-empty set at completion parks the node
	// in NodeWaiting. Mirrors adk-python's Context._interrupt_ids.
	interruptIDs map[string]struct{}
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

// trackInterrupts accumulates the node's long-running tool call IDs.
//
// It does NOT resolve an interrupt from a FunctionResponse seen in the
// same run: that is the tool's own initial "pending" response, not the
// reply. The real reply arrives on a later turn — a fresh run that does
// not re-raise the call, so its interrupt set is empty and the node
// completes. Mirrors adk-python: the long-running call ends the turn
// and the pause persists until a new invocation answers it.
func (nr *nodeRun) trackInterrupts(ev *session.Event) {
	if ev == nil {
		return
	}
	for _, id := range ev.LongRunningToolIDs {
		if id == "" {
			continue
		}
		if nr.interruptIDs == nil {
			nr.interruptIDs = map[string]struct{}{}
		}
		nr.interruptIDs[id] = struct{}{}
	}
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
//
// processed, when non-nil, is a back-pressure handshake: the producing
// goroutine blocks until the consumer closes it (after the event is
// yielded and persisted), so a non-partial function-response is in the
// session before the node's flow rebuilds the next model request.
// Mirrors adk-python's enqueue_event/processed_signal handshake. Nil
// for partial events, which are fire-and-forget.
type eventItem struct {
	nodeName  string
	ev        *session.Event
	processed chan struct{}
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

// retryItem signals that a node should be retried after a delay.
type retryItem struct {
	node        Node
	input       any
	triggeredBy string
	branch      string
}

func (retryItem) isQueueItem() {}

// newScheduler returns an initialised scheduler ready for the
// consumer to drive. The caller is responsible for seeding the
// initial trigger (typically Start with the user input).
//
// maxConcurrency caps len(runsByName) at any point in time:
// scheduleResumedNode enqueues into pendingQueue when the cap is
// reached, and the consumer drains the queue via
// tryDispatchPending as in-flight nodes complete. 0 disables the
// cap (unlimited).
func newScheduler(parent agent.InvocationContext, g *graph, maxConcurrency int) *scheduler {
	return &scheduler{
		state:          NewRunState(),
		graph:          g,
		nodesByName:    buildNodesByName(g),
		runsByName:     map[string]*nodeRun{},
		runCancels:     map[string]context.CancelFunc{},
		retryTimers:    map[string]*time.Timer{},
		eventQueue:     make(chan queueItem, defaultEventQueueCapacity),
		parentCtx:      parent,
		maxConcurrency: maxConcurrency,
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
// branch is the composite branch string this activation runs under;
// empty means inherit the workflow's root branch. Branch scopes
// LLM history visibility (via the flow processor's branch-prefix
// filter) and gets stamped onto every emitted event when the node
// leaves Event.Branch empty.
//
// When the engine's max-concurrency cap is reached, the activation
// is enqueued instead of started; the node enters NodePending and
// the scheduler dispatches it as in-flight nodes complete.
//
// scheduleNode runs only on the consumer goroutine.
func (s *scheduler) scheduleNode(n Node, input any, triggeredBy, branch string) {
	s.scheduleResumedNode(n, input, triggeredBy, branch, nil)
}

// atConcurrencyLimit reports whether the number of in-flight
// activations has reached the configured cap. Always false when
// maxConcurrency is 0 (unlimited).
func (s *scheduler) atConcurrencyLimit() bool {
	return s.maxConcurrency > 0 && len(s.runsByName) >= s.maxConcurrency
}

// tryDispatchPending starts as many queued activations as the
// current concurrency budget allows. FIFO over pendingQueue;
// stops when the queue is empty or the cap is reached again.
//
// Called after every completion (and after retry-timer expiry)
// to make room-becoming-available immediately observable.
func (s *scheduler) tryDispatchPending() {
	for len(s.pendingQueue) > 0 && !s.atConcurrencyLimit() {
		next := s.pendingQueue[0]
		s.pendingQueue = s.pendingQueue[1:]
		s.startNode(next.node, next.input, next.triggeredBy, next.branch, next.resumeInputs)
	}
}

// scheduleResumedNode is like scheduleNode but additionally
// injects resumeInputs into the per-node context, so re-entry
// nodes can read the user-supplied response payload via
// ctx.ResumedInput(interruptID). resumeInputs is keyed by
// InterruptID; nil disables re-entry semantics and yields the same
// behaviour as scheduleNode.
//
// When the engine's max-concurrency cap is reached, the
// activation is enqueued onto pendingQueue and the node enters
// NodePending instead of starting immediately. tryDispatchPending
// drains the queue as in-flight nodes complete.
//
// scheduleResumedNode runs only on the consumer goroutine.
func (s *scheduler) scheduleResumedNode(n Node, input any, triggeredBy, branch string, resumeInputs map[string]any) {
	if s.atConcurrencyLimit() {
		name := n.Name()
		ns := s.state.EnsureNode(name)
		ns.Status = NodePending
		ns.Input = input
		ns.TriggeredBy = triggeredBy
		ns.Branch = branch
		s.pendingQueue = append(s.pendingQueue, pendingActivation{
			node:         n,
			input:        input,
			triggeredBy:  triggeredBy,
			branch:       branch,
			resumeInputs: resumeInputs,
		})
		return
	}
	s.startNode(n, input, triggeredBy, branch, resumeInputs)
}

// startNode is the unguarded core of scheduleResumedNode: it
// creates the per-node context, registers bookkeeping, and
// launches the runner goroutine. Always honours the call; the
// concurrency-cap check is done by scheduleResumedNode (the
// public entry point) before reaching here.
func (s *scheduler) startNode(n Node, input any, triggeredBy, branch string, resumeInputs map[string]any) {
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
	// Order matters: WithContext sets up per-node cancellation on
	// the underlying InvocationContext; withBranch then wraps the
	// result in branchOverride. Reversing would either lose the
	// cancellation context or strip the branch override.
	wrapped := withBranch(s.parentCtx.WithContext(nodeCtx), branch)
	perNodeCtx := newNodeContext(wrapped, resumeInputs)

	// EXPERIMENTAL: stash perNodeCtx in the embedded context.Context
	// so tools running inside an LlmAgent that is itself running as
	// this node can recover the NodeContext via
	// workflow.NodeContextFromGoContext. The value rides through
	// every downstream NewInvocationContext / WithContext call by
	// virtue of context.Context value-chain propagation.
	//
	// See WithNodeContext / NodeContextFromGoContext in
	// node_context_bridge.go. Top-level static activations have
	// subScheduler == nil, so RunNode will still reject them; the
	// stash is harmless in that case.
	perNodeCtx.InvocationContext = perNodeCtx.InvocationContext.WithContext(
		WithNodeContext(perNodeCtx.InvocationContext, perNodeCtx),
	)

	ns := s.state.EnsureNode(name)
	ns.Status = NodeRunning
	ns.Input = input
	ns.TriggeredBy = triggeredBy
	ns.Branch = branch
	s.runsByName[name] = &nodeRun{branch: branch}

	s.runCancels[name] = cancel
	s.wg.Add(1)

	go runNode(s.eventQueue, &s.wg, name, n, perNodeCtx, input)
}

// scheduleRetry schedules a retry for node n after the given delay.
// branch is preserved across attempts so retries do not silently
// move the node to a different branch.
func (s *scheduler) scheduleRetry(n Node, input any, triggeredBy, branch string, delay time.Duration) {
	timer := time.AfterFunc(delay, func() {
		go func() {
			select {
			case s.eventQueue <- retryItem{node: n, input: input, triggeredBy: triggeredBy, branch: branch}:
			case <-s.parentCtx.Done():
			}
		}()
	})
	s.retryTimers[n.Name()] = timer
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

	span, ctx := startNodeSpan(ctx, n)
	defer span.End()

	// completion holds the final completionItem. It is sent in the
	// outer defer so panic recovery, normal exit, and cancellation
	// all funnel through the same send path.
	completion := completionItem{nodeName: name}
	defer func() {
		if r := recover(); r != nil {
			completion.err = fmt.Errorf("node %q panicked: %v", name, r)
		}
		if completion.err != nil && !errors.Is(completion.err, context.Canceled) {
			span.RecordError(completion.err)
			span.SetStatus(codes.Error, completion.err.Error())
		}
		out <- completion
	}()

	validated, err := n.ValidateInput(input)
	if err != nil {
		completion.err = fmt.Errorf("%w for node %q: %w", ErrInputValidation, name, err)
		return
	}

	for ev, err := range n.Run(ctx, validated) {
		if err != nil {
			completion.err = err
			return
		}
		// Block on non-partial events until the consumer has persisted
		// them (see eventItem.processed). Partial events are
		// fire-and-forget.
		var processed chan struct{}
		if ev != nil && !ev.LLMResponse.Partial {
			processed = make(chan struct{})
		}
		select {
		case out <- eventItem{nodeName: name, ev: ev, processed: processed}:
		case <-ctx.Done():
			completion.err = ctx.Err()
			return
		}
		if processed != nil {
			select {
			case <-processed:
			case <-ctx.Done():
				completion.err = ctx.Err()
				return
			}
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
// Also drops the pendingQueue so queued (NodePending) activations
// do not start after cancellation. Their NodeState is left as
// NodePending — the surrounding RunState snapshot preserves the
// fact that they were ready-but-not-yet-started.
//
// cancelAll runs only on the consumer goroutine.
func (s *scheduler) cancelAll() {
	for _, cancel := range s.runCancels {
		cancel()
	}
	for _, t := range s.retryTimers {
		t.Stop()
	}
	s.retryTimers = nil
	s.pendingQueue = nil
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
	var pendingErr error  // first non-nil node error; surfaced after drain
	draining := false     // true once cancelAll has run; remaining queue items are drained without yielding or scheduling new successors
	consumerGone := false // true once the caller broke the range loop; no further yield is allowed

	doneChan := s.parentCtx.Done()

	for len(s.runsByName) > 0 || len(s.retryTimers) > 0 {
		var item queueItem
		select {
		case item = <-s.eventQueue:
		case <-doneChan:
			if !draining {
				draining = true
				s.cancelAll()
			}
			doneChan = nil // Disable this case so we don't busy loop
		}

		switch it := item.(type) {
		case eventItem:
			s.handleEvent(it)
			if !draining {
				if !yield(it.ev, nil) {
					draining = true
					consumerGone = true
					s.cancelAll()
				}
			}
			// Release the producer's handshake now that the event is
			// yielded and persisted. Always signal — even when
			// draining — so a blocked producer does not leak.
			if it.processed != nil {
				close(it.processed)
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
			// A slot just freed up; promote any queued
			// activations to running.
			if !draining {
				s.tryDispatchPending()
			}
		case retryItem:
			delete(s.retryTimers, it.node.Name())
			if !draining {
				s.scheduleNode(it.node, it.input, it.triggeredBy, it.branch)
			}
		}
	}

	// All goroutines have returned and pushed their final events.
	// Surface the first error to the caller — but not if the caller
	// already broke the range loop, since yielding after a false return
	// panics the iterator.
	if pendingErr != nil && !consumerGone {
		yield(nil, pendingErr)
	}
}

// defaultContentRole picks the role for node Content that left it
// empty. A FunctionResponse part carries the result of a locally
// executed function tool back to the model, so it is app-authored and
// takes genai.RoleUser (matching the LLM flow and adk-python). Note
// this is about the part type, not tools in general: built-in tools
// like google_search run inside the model and surface as ordinary
// model output with no FunctionResponse part. Everything else is
// model-authored.
func defaultContentRole(c *genai.Content) string {
	for _, p := range c.Parts {
		if p != nil && p.FunctionResponse != nil {
			return genai.RoleUser
		}
	}
	return genai.RoleModel
}

// handleEvent updates the per-node accumulator. The event has
// already been read from the queue and will be yielded to the
// caller by the consumer loop.
//
// Descendant events (NodeInfo.Path != it.nodeName) are dynamic-node
// children forwarded by the sub-scheduler. Their Output/Routes
// belong to the child, not the parent — skip the parent accumulator.
//
// RequestedInput is the exception: the child's pause unwinds the
// orchestrator (dynamic_scheduler.go runNode), and Workflow.Resume
// matches InterruptID against the parent's NodeState.PendingRequest,
// so the parent must transition to NodeWaiting on a descendant pause.
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
	// Stamp the activation's branch onto events that left
	// Event.Branch empty; nodes that set a non-empty Event.Branch
	// keep it.
	if it.ev.Branch == "" && nr.branch != "" {
		it.ev.Branch = nr.branch
	}
	// Default Content.Role for nodes that left it empty
	// (FunctionNode/BaseNode set Parts but not Role); clients like
	// the web UI rely on it. Before the descendant short-circuit so
	// dynamic children are covered too.
	if it.ev.Content != nil && it.ev.Content.Role == "" {
		it.ev.Content.Role = defaultContentRole(it.ev.Content)
	}
	var path string
	if it.ev.NodeInfo != nil {
		path = it.ev.NodeInfo.Path
	}
	isDescendant := path != "" && path != it.nodeName
	// Track long-running interrupts before the descendant
	// short-circuit so a dynamic child's pause is promoted to the
	// parent node (a RequestInput pause rides on LongRunningToolIDs).
	nr.trackInterrupts(it.ev)
	if isDescendant {
		return
	}
	// Stamp the node name onto the event so history rehydration can
	// attribute it back to this node. Static nodes leave Path empty
	// (the node name is not otherwise on the event — Author is the
	// workflow agent, not the node). Matches adk-python, which sets
	// node_info.path on every event and attributes by it.
	if path == "" {
		if it.ev.NodeInfo == nil {
			it.ev.NodeInfo = &session.NodeInfo{}
		}
		it.ev.NodeInfo.Path = it.nodeName
	}
	if it.ev.Routes != nil {
		nr.setRoutingEvent(it.ev, it.nodeName)
	}
	if out, ok := childEventOutput(it.ev); ok {
		nr.setOutput(out, it.nodeName)
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
	// For retryable nodes still delete them from run variables. If the node is retried,
	// it will get a fresh accumulator in scheduleNode to avoid ErrMultipleOutputs
	// from partial results of this failed run.
	delete(s.runsByName, it.nodeName)
	delete(s.runCancels, it.nodeName)

	if errors.Is(it.err, context.Canceled) {
		ns.Status = NodeCancelled
		return nil // sibling cancellation; not the original error
	}
	if it.err != nil {
		currentNode := s.nodesByName[it.nodeName]
		if currentNode != nil {
			cfg := currentNode.Config()
			if cfg.RetryConfig != nil {
				ns.Attempt = ns.Attempt + 1

				if ShouldRetry(cfg.RetryConfig, it.err, ns.Attempt) {
					delay := CalculateDelay(cfg.RetryConfig, ns.Attempt)
					ns.Status = NodePending
					s.scheduleRetry(currentNode, ns.Input, ns.TriggeredBy, ns.Branch, delay)
					// Return nil to continue the scheduler loop. Successors will
					// be scheduled only when a retry attempt eventually succeeds
					// and reaches the bottom of this function.
					return nil // Don't fail the workflow
				}
			}
		}
		ns.Status = NodeFailed
		return it.err
	}
	if nr != nil && nr.err != nil {
		ns.Status = NodeFailed
		return nr.err
	}

	// Happy path: decide between NodeWaiting (an open interrupt) or
	// NodeCompleted. The waiting branch fires regardless of the
	// scheduleSuccessors flag — an interrupt that survived the run
	// must be observable in RunState even when the consumer is
	// draining.
	//
	// Long-running-tool pause (incl. RequestInput, which rides on
	// LongRunningToolIDs): park WAITING with the open interrupt IDs
	// so resume can match a human's FunctionResponse back to this
	// node. Mirrors adk-python _handle_completion.
	if nr != nil && len(nr.interruptIDs) > 0 {
		ns.Status = NodeWaiting
		ns.Interrupts = ns.Interrupts[:0]
		for id := range nr.interruptIDs {
			ns.Interrupts = append(ns.Interrupts, id)
		}
		return nil
	}

	ns.Status = NodeCompleted
	ns.Attempt = 0
	if nr != nil && nr.hasOutput {
		ns.Output = nr.output
	}
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

	for _, succ := range findSuccessors(s.graph, s.state, currentNode, input, routingEv, ns.Branch) {
		s.scheduleNode(succ.node, succ.input, succ.triggeredBy, succ.branch)
	}
	return nil
}

// successor is the per-target dispatch tuple produced by
// findSuccessors. The triggeredBy field carries the upstream
// node's name for persistence on NodeState (used by Resume to
// reconstruct the activation chain across pause/resume turns).
// The branch field carries the composite branch string the
// successor should run under — empty for chains that inherit the
// parent's branch, populated when fan-out derives a sub-branch or
// when JoinNode resolves its common predecessor prefix.
type successor struct {
	node        Node
	input       any
	triggeredBy string
	branch      string
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
//     this node. The absence of a matching route is treated as a
//     deliberate decision not to continue, not an error.
//
// JoinNode successors are gated by appendSuccessor; see its docs.
//
// Branch derivation:
//
//   - When this activation fans out to >1 successor, each non-Join
//     successor is given a sub-branch
//     "<parentBranch>.<successorName>@1" (run_id "1" because the
//     static scheduler does not auto-counter activations of the
//     same name within one fan-out; loop-back routing re-enters the
//     same node and the sub-branch stays stable).
//   - When this activation has a single successor, the successor
//     inherits parentBranch unchanged.
//   - JoinNode successors compute their own branch in
//     appendSuccessor as the common dot-prefix of the branches of
//     all completed predecessors — see aggregatePredecessorBranches.
func findSuccessors(g *graph, state *RunState, currentNode Node, input any, event *session.Event, parentBranch string) []successor {
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
			out = appendSuccessor(out, g, state, edge.To, input, from, parentBranch)
			added[edge.To] = struct{}{}
			continue
		}
		if edge.Route == Default {
			defaultRouteNode = edge.To
			continue
		}
		if event != nil && edge.Route.Matches(event) {
			out = appendSuccessor(out, g, state, edge.To, input, from, parentBranch)
			added[edge.To] = struct{}{}
			concreteMatched = true
		}
	}
	if !concreteMatched && defaultRouteNode != nil {
		out = appendSuccessor(out, g, state, defaultRouteNode, input, from, parentBranch)
	}
	// Second pass: if we fanned out to more than one successor,
	// stamp a sub-branch onto each non-Join entry whose branch is
	// still the inherited parentBranch. JoinNode entries already
	// carry their common-prefix branch from appendSuccessor and
	// must not be re-derived.
	if len(out) > 1 {
		for i, s := range out {
			if _, isJoin := s.node.(*JoinNode); isJoin {
				continue
			}
			if s.branch != parentBranch {
				continue
			}
			out[i].branch = deriveSubBranch(parentBranch, s.node.Name()+"@1")
		}
	}
	return out
}

// appendSuccessor records a routing match in the dispatch list.
// Non-JoinNode targets are recorded with parentBranch as their
// initial branch; findSuccessors may upgrade to a sub-branch in a
// second pass if this activation fanned out.
//
// A *JoinNode target is recorded only when every declared
// predecessor has completed; its input is replaced with the
// aggregated predecessor outputs, and its branch is the common
// dot-prefix of all predecessor branches. A barrier-blocked
// JoinNode is silently skipped — a later predecessor completion
// re-evaluates the barrier.
func appendSuccessor(out []successor, g *graph, state *RunState, target Node, input any, triggeredBy, parentBranch string) []successor {
	if _, isJoin := target.(*JoinNode); !isJoin {
		return append(out, successor{
			node:        target,
			input:       input,
			triggeredBy: triggeredBy,
			branch:      parentBranch,
		})
	}
	aggregated, ok := aggregatePredecessorOutputs(g, state, target)
	if !ok {
		return out
	}
	joinBranch := commonBranchPrefix(aggregatePredecessorBranches(g, state, target))
	return append(out, successor{
		node:        target,
		input:       aggregated,
		triggeredBy: triggeredBy,
		branch:      joinBranch,
	})
}

// aggregatePredecessorOutputs returns a map of predecessor name to
// recorded Output for every predecessor of target. Returns
// (nil, false) if any predecessor is not yet NodeCompleted; a
// predecessor that completed without an output contributes a nil
// value (same as the absence the predecessor itself emitted).
func aggregatePredecessorOutputs(g *graph, state *RunState, target Node) (map[string]any, bool) {
	predEdges := g.predecessorsOf(target)
	aggregated := make(map[string]any, len(predEdges))
	for _, edge := range predEdges {
		name := edge.From.Name()
		ns := state.Nodes[name]
		if ns == nil || ns.Status != NodeCompleted {
			return nil, false
		}
		aggregated[name] = ns.Output
	}
	return aggregated, true
}

// aggregatePredecessorBranches returns the branch strings recorded
// for every predecessor of target. Order is the graph's
// predecessor-edge order, which is deterministic per construction.
// Callers feed the result into commonBranchPrefix to compute the
// join's own branch.
//
// Assumes appendSuccessor's caller has already verified all
// predecessors are NodeCompleted (via aggregatePredecessorOutputs
// returning ok); a missing entry contributes "" which conservatively
// short-circuits the common-prefix to root.
func aggregatePredecessorBranches(g *graph, state *RunState, target Node) []string {
	predEdges := g.predecessorsOf(target)
	branches := make([]string, 0, len(predEdges))
	for _, edge := range predEdges {
		name := edge.From.Name()
		var br string
		if ns := state.Nodes[name]; ns != nil {
			br = ns.Branch
		}
		branches = append(branches, br)
	}
	return branches
}

func startNodeSpan(ctx agent.InvocationContext, n Node) (trace.Span, agent.InvocationContext) {
	if n == Start {
		// Don't create span for the Start node.
		return noop.Span{}, ctx
	}
	if n.Config().EmitsOwnSpan {
		// The node body starts its own span (e.g. an LlmAgent node's
		// wrapped agent emits invoke_agent); don't add an invoke_node
		// wrapper around it.
		return noop.Span{}, ctx
	}
	spanCtx, span := telemetry.StartNodeSpan(ctx, ctx, telemetry.OperationNode{Node: n})
	return span, ctx.WithContext(spanCtx)
}
