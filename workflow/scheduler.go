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
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/sessionutils"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// defaultEventQueueCapacity bounds the buffered channel between
// node-runner goroutines and the consumer. A small fixed capacity
// keeps backpressure tight without serializing producers.
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

	// ErrMultipleTerminalOutputs is returned when more than one
	// terminal node produced output in a run, making the workflow's
	// output ambiguous. See scheduler.finalize for the exact rule.
	ErrMultipleTerminalOutputs = errors.New("workflow: multiple terminal nodes produced output; a workflow must have at most one terminal output")
)

// scheduler drives a single Workflow.Run invocation. It owns the
// per-node task table, the event channel, lifecycle counters, and
// the parent invocation context — none of which survive across
// processes. The persistable view of the same run (node statuses,
// inputs, triggers) lives on the embedded *RunState.
//
// Concurrency model: producer-consumer over eventQueue.
//
//   - Producers are the per-node goroutines started by startNode.
//     They only send to eventQueue (events and a final completion);
//     they never read from it and never touch any other scheduler
//     field except counters, which has its own mutex.
//   - The consumer is the single goroutine running scheduler.run
//     (the caller of Workflow.Run). It is the only reader of
//     eventQueue and the only mutator of runsByName, runCancels,
//     triggers, lastOutputs, state.Nodes, and per-node accumulators.
//
// Because producers and the consumer share only eventQueue (a
// channel — already safe for concurrent use) and counters, the
// consumer-only fields below need no mutex.
//
// A node has at most one live activation. Every trigger goes through
// the trigger buffer, and scheduleReady starts a trigger only when its
// node is not busy (see busy); the retry case of run restarts a node
// that stayed busy through its backoff. So runsByName, runCancels and
// retryTimers, keyed by node name, never hold two activations of one
// node. Mirrors adk-python's _schedule_ready_nodes
// (workflow/_workflow.py).
type scheduler struct {
	state       *RunState       // persisted lifecycle state
	graph       *graph          // adjacency, terminal lookup
	nodesByName map[string]Node // built once at construction; lets handleCompletion resolve a completion's name back to its Node in O(1)
	terminals   map[string]bool // terminal graph nodes (designated use_as_output)

	// Per-node accumulators, created when an activation starts and
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

	parentCtx    agent.Context
	workflowName string

	// maxConcurrency caps len(runsByName)+len(retryTimers); 0
	// disables the cap. A trigger that finds the cap reached waits in
	// triggers without taking a slot. Set once at construction;
	// immutable thereafter.
	maxConcurrency int

	// triggers holds every trigger that has not started, whether it
	// waits for a busy node or for a free slot. Owned by the consumer
	// goroutine.
	triggers triggerBuffer

	// counters allocates the run ids of this run's static activations
	// and, through the node contexts, of their dynamic children.
	counters *runCounters

	// lastOutputs is the last non-nil output any activation of a node
	// produced in this run, by node name. A nil map reads as empty.
	// Mirrors adk-python's loop_state.node_outputs. Owned by the
	// consumer goroutine.
	lastOutputs map[string]any
}

// pendingActivation is a trigger. runID == "" means "new activation: allocate
// the next run id"; a non-empty runID is an activation that already has one
// (a retry attempt, a resumed re-entry).
type pendingActivation struct {
	node         Node
	input        any
	triggeredBy  string
	branch       string // the branch to inherit; the final branch when subBranch is false
	subBranch    bool   // start derives "<branch>.<name>@<run id>"
	runID        string
	resumeInputs map[string]any // nil for a plain trigger
}

// triggerBuffer is the set of triggers that have not started: one FIFO
// bucket per node name, the buckets kept in the order their names were
// inserted. A bucket is removed when it empties, so a later trigger of
// that name goes to the end. Mirrors adk-python's trigger_buffer dict
// and _pop_trigger (workflow/_workflow.py). The zero value is empty
// and ready to use. Not safe for concurrent use.
type triggerBuffer struct {
	order   []string
	buckets map[string][]pendingActivation
}

// push appends t to the bucket of its node.
func (b *triggerBuffer) push(t pendingActivation) {
	name := t.node.Name()
	if b.buckets == nil {
		b.buckets = map[string][]pendingActivation{}
	}
	if _, ok := b.buckets[name]; !ok {
		b.order = append(b.order, name)
	}
	b.buckets[name] = append(b.buckets[name], t)
}

// pop removes and returns the oldest trigger of name.
func (b *triggerBuffer) pop(name string) (pendingActivation, bool) {
	bucket := b.buckets[name]
	if len(bucket) == 0 {
		return pendingActivation{}, false
	}
	t := bucket[0]
	if len(bucket) == 1 {
		delete(b.buckets, name)
		b.order = slices.DeleteFunc(b.order, func(n string) bool { return n == name })
	} else {
		b.buckets[name] = bucket[1:]
	}
	return t, true
}

// names returns the names that have triggers, in insertion order.
func (b *triggerBuffer) names() []string {
	return slices.Clone(b.order)
}

// clear drops every trigger.
func (b *triggerBuffer) clear() {
	b.order = nil
	b.buckets = nil
}

// nodeRun accumulates mid-flight state for a running node goroutine.
// Owned exclusively by the consumer goroutine: mutations happen on
// eventItem delivery and reads happen on completionItem arrival.
//
// Duplicate outputs or routing events record an error on the struct
// without overwriting the first value, and the consumer surfaces
// the error at completion.
type nodeRun struct {
	routingEvent *session.Event // at most one; multiple is an error
	output       any            // single Event.Output; nil if hasOutput is false
	hasOutput    bool           // distinguishes "no output yet" from "output was nil"
	err          error          // set on duplicate output or duplicate routing event
	branch       string         // composite branch assigned at scheduling; used to stamp Event.Branch when the node leaves it empty
	nodePath     string         // hierarchical path assigned at scheduling (e.g. "parent/child@1" or "child@1")
	runID        string         // the activation's run id; a retry of the activation reuses it

	// interruptIDs are unresolved long-running tool call IDs raised by
	// the node's events. A non-empty set at completion parks the node
	// in NodeWaiting. Mirrors adk-python's Context._interrupt_ids.
	interruptIDs map[string]struct{}

	// usedResponses is set at completion when the activation held user
	// responses from a resume. completionEvent records that completion
	// so rehydration stops offering the responses to later visits.
	usedResponses bool

	// end is the node's state as this activation left it, saved before
	// its successors are scheduled. A successor can be the node itself:
	// starting or queueing it resets the node's NodeState before
	// completionEvent reads it.
	end *NodeState
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
//
// The consumer also closes processed for an event the caller did not
// take, after it has canceled the producer's context. So the node's
// yield returns true for a non-partial event only if the caller took
// the event and the node's context is not done when the producer
// resumes; otherwise it returns false and the node's completion
// carries ctx.Err().
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
	nodeName  string
	err       error
	inventory failureInventory
}

func (completionItem) isQueueItem() {}

// retryItem signals that a retry backoff ended: act is the activation
// to restart, with its run id.
type retryItem struct{ act pendingActivation }

func (retryItem) isQueueItem() {}

// newScheduler returns an initialized scheduler ready for the
// consumer to drive. The caller is responsible for seeding the
// initial trigger (typically Start with the user input).
//
// maxConcurrency caps the number of activations that run or wait
// for a retry backoff at any point in time; triggers beyond it wait in
// the trigger buffer and start as slots free. 0 disables the cap
// (unlimited).
func newScheduler(parent agent.Context, g *graph, maxConcurrency int) *scheduler {
	return &scheduler{
		state:          NewRunState(),
		graph:          g,
		nodesByName:    buildNodesByName(g),
		terminals:      g.terminalNodeNames(),
		runsByName:     map[string]*nodeRun{},
		runCancels:     map[string]context.CancelFunc{},
		retryTimers:    map[string]*time.Timer{},
		eventQueue:     make(chan queueItem, defaultEventQueueCapacity),
		parentCtx:      parent,
		maxConcurrency: maxConcurrency,
		counters:       &runCounters{},
		lastOutputs:    map[string]any{},
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

// scheduleNode buffers a plain trigger for n with the given input and
// starts every trigger that can start (scheduleReady). Workflow.RunNode
// seeds START through it.
//
// branch is the composite branch string this activation runs under;
// empty means inherit the workflow's root branch. Branch scopes
// LLM history visibility (via the flow processor's branch-prefix
// filter) and gets stamped onto every emitted event when the node
// leaves Event.Branch empty.
//
// scheduleNode runs only on the consumer goroutine.
func (s *scheduler) scheduleNode(n Node, input any, triggeredBy, branch string) {
	s.enqueue(pendingActivation{node: n, input: input, triggeredBy: triggeredBy, branch: branch})
	s.scheduleReady()
}

// enqueue appends t to the trigger buffer. It starts nothing; the
// caller follows with scheduleReady when the trigger may start now.
func (s *scheduler) enqueue(t pendingActivation) {
	s.triggers.push(t)
}

// hasQueued reports whether a trigger for name waits in the buffer.
func (s *scheduler) hasQueued(name string) bool {
	return len(s.triggers.buckets[name]) > 0
}

// dequeue removes and returns the oldest buffered trigger of name.
func (s *scheduler) dequeue(name string) (pendingActivation, bool) {
	return s.triggers.pop(name)
}

// queuedFor returns a copy of the buffered triggers of name, oldest
// first.
func (s *scheduler) queuedFor(name string) []pendingActivation {
	return slices.Clone(s.triggers.buckets[name])
}

// scheduleReady starts buffered triggers. It visits the node names in
// the order their triggers were buffered, skips a busy node, stops when
// the concurrency cap is reached, and otherwise starts the node's
// oldest trigger. A node that starts is busy, so one call starts at
// most one activation per node and never starts a busy node. Port of
// adk-python's _schedule_ready_nodes (workflow/_workflow.py).
//
// scheduleReady runs only on the consumer goroutine.
func (s *scheduler) scheduleReady() {
	for _, name := range s.triggers.names() {
		if s.busy(name) {
			continue
		}
		if s.atConcurrencyLimit() {
			return
		}
		if t, ok := s.dequeue(name); ok {
			s.startNode(t)
		}
	}
}

// busy reports whether name must not start another activation now: it
// has a live attempt, it waits for a retry backoff, or it waits for a
// user response. A node parked by WaitForOutput without output is not
// busy, so its next trigger starts a new activation.
func (s *scheduler) busy(name string) bool {
	if s.runsByName[name] != nil || s.retryTimers[name] != nil {
		return true
	}
	ns := s.state.Nodes[name]
	return ns != nil && ns.Status == NodeWaiting && len(ns.Interrupts) > 0
}

// atConcurrencyLimit reports whether the number of activations that
// run or wait for a retry backoff has reached the configured cap.
// Always false when maxConcurrency is 0 (unlimited).
func (s *scheduler) atConcurrencyLimit() bool {
	return s.maxConcurrency > 0 && len(s.runsByName)+len(s.retryTimers) >= s.maxConcurrency
}

// startNode starts activation t: it allocates the run id when t has
// none, derives the node path and the branch, creates the per-node
// context, registers the activation in runsByName and runCancels and
// launches the runner goroutine. It is called only by scheduleReady,
// for a node that is not busy, and by the retry case of run, for the
// node whose backoff just ended; so it never replaces a live
// activation.
func (s *scheduler) startNode(t pendingActivation) {
	n := t.node
	name := n.Name()

	// Per-node context: WithTimeout when Config().Timeout > 0,
	// WithCancel otherwise. Either way it inherits from parentCtx,
	// so an ambient deadline on the workflow invocation still
	// applies.
	cfg := n.Config()
	var cancel context.CancelFunc

	runID := t.runID
	if runID == "" {
		runID = s.counters.next(s.parentCtx.Path(), name)
	}
	nodePath := name + "@" + runID
	if p := s.parentCtx.Path(); p != "" {
		nodePath = p + "/" + nodePath
	} else if s.graph.isRootWrapper {
		nodePath = ""
	}
	branch := t.branch
	if t.subBranch {
		branch = deriveSubBranch(t.branch, name+"@"+runID)
	}

	ns := s.state.EnsureNode(name)
	if t.runID == "" {
		// A new activation gets the whole retry budget; a retry or a
		// resumed re-entry keeps the attempts it has used.
		ns.Attempt = 0
	}
	// ns.Attempt counts the failed attempts of this activation, so the
	// node reads 1 on its first run and one more on each retry.
	attempt := ns.Attempt + 1

	resumeInputs := t.resumeInputs
	ofa := s.terminalAncestors(name)
	var dss agent.DynamicSubScheduler
	perNodeCtx := s.parentCtx.WithDelta(&agent.CommonContextDelta{
		ResumeInputs: &resumeInputs,
		InvocationContextDelta: &agent.InvocationContextDelta{
			Branch: &branch,
		},
		Path:               &nodePath,
		RunID:              &runID,
		SubScheduler:       &dss,
		OutputForAncestors: &ofa,
		AttemptCount:       &attempt,
	})
	perNodeCtx = withRunCounters(perNodeCtx, s.counters)

	if cfg.Timeout > 0 {
		perNodeCtx, cancel = perNodeCtx.WithAgentTimeout(cfg.Timeout)
	} else {
		perNodeCtx, cancel = perNodeCtx.WithAgentCancel()
	}

	ns.Status = NodeRunning
	ns.Output = nil
	ns.Input = t.input
	ns.TriggeredBy = t.triggeredBy
	ns.Branch = branch
	s.runsByName[name] = &nodeRun{branch: branch, nodePath: nodePath, runID: runID}

	s.runCancels[name] = cancel
	s.wg.Add(1)

	go runNode(s.eventQueue, &s.wg, name, n, perNodeCtx, t.input)
}

// scheduleRetry restarts activation t after delay. The node stays busy,
// and keeps its concurrency slot, until the timer's retryItem is
// handled; t carries the run id and the final branch, so the retried
// attempt keeps the activation's path and branch.
func (s *scheduler) scheduleRetry(t pendingActivation, delay time.Duration) {
	timer := time.AfterFunc(delay, func() {
		select {
		case s.eventQueue <- retryItem{act: t}:
		case <-s.parentCtx.Done():
		}
	})
	s.retryTimers[t.node.Name()] = timer
}

// wouldRetry reports whether the RetryConfig of the node named name runs
// its activation again after failure err, its failedAttempts-th.
func (s *scheduler) wouldRetry(name string, err error, failedAttempts int) bool {
	n := s.nodesByName[name]
	return n != nil && ShouldRetry(n.Config().RetryConfig, err, failedAttempts)
}

// runNode is the per-node goroutine wrapper. It drives the node's
// iter.Seq2, pushes events into the queue, and ends with exactly
// one completionItem. A panic in the node body is recovered and
// reported as a completion error so the consumer never deadlocks
// waiting for a vanished goroutine.
//
// Event sends select on ctx.Done(): if the scheduler has canceled
// this node, an in-progress send to a full eventQueue does not
// deadlock — the goroutine drops the pending event and proceeds to
// completion. The completion send is unconditional because the
// consumer's runsByName bookkeeping relies on it.
//
// The node's yield returns true for a non-partial event only if the
// consumer yielded the event, the caller took it, and ctx is not done
// when the handshake is released (see eventItem.processed). Otherwise
// yield returns false and the completion carries ctx.Err(), so a node
// that returns when its yield returns false does not go on past an
// event the caller did not take. A partial event is fire-and-forget:
// its yield returns true once the event is queued.
func runNode(
	out chan<- queueItem,
	wg *sync.WaitGroup,
	name string,
	n Node,
	ctx agent.Context,
	input any,
) {
	defer wg.Done()
	own, ownErr := nodeStateValidator(ctx, n)
	ctx, writes := beginActivation(ctx, n, own)

	span, ctx := startNodeSpan(ctx, n)
	defer span.end()

	// completion holds the final completionItem. It is sent in the
	// outer defer so panic recovery, normal exit, and cancellation
	// all funnel through the same send path.
	completion := completionItem{nodeName: name}
	defer func() {
		if r := recover(); r != nil {
			completion.err = fmt.Errorf("node %q panicked: %v", name, r)
			completion.inventory = failureInventory{}
		}
		span.recordError(completion.err, nil)
		// State writes the activation has not sent yet go out on one
		// state-only event before the completion, with a plain send like
		// the completion's: the consumer reads until every node completes.
		if flushesPending(completion.err) {
			if ev := writes.pending(ctx); ev != nil {
				writes.forwardUp(ev)
				out <- eventItem{nodeName: name, ev: ev}
			}
		}
		out <- completion
	}()

	// The failure is reported here, once the completion is registered, so
	// the scheduler still receives it; the node does not run.
	if ownErr != nil {
		completion.err = ownErr
		return
	}

	validated, err := n.ValidateInput(input)
	if err != nil {
		completion.err = fmt.Errorf("%w for node %q: %w", ErrInputValidation, name, err)
		return
	}
	ctx, inventoryRecorder := withFailureInventory(ctx, n)

	for ev, err := range n.Run(ctx, validated) {
		if err != nil {
			completion.err = err
			completion.inventory = inventoryRecorder.snapshot()
			return
		}
		// An event whose state delta the schema refuses is not sent; its
		// error fails the node unwrapped, so errors.As reaches it.
		if aerr := writes.admit(ev); aerr != nil {
			completion.err = aerr
			completion.inventory = inventoryRecorder.snapshot()
			return
		}
		writes.forwardUp(ev)
		// Block on non-partial events until the consumer has persisted
		// them (see eventItem.processed). Partial events are
		// fire-and-forget, so send a deep copy that shares no mutable
		// memory with the continuing producer goroutine.
		var processed chan struct{}
		sendEv := ev
		if ev != nil {
			if ev.LLMResponse.Partial {
				sendEv = sessionutils.CloneValue(ev)
			} else {
				processed = make(chan struct{})
			}
		}
		select {
		case out <- eventItem{nodeName: name, ev: sendEv, processed: processed}:
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
			// A released handshake is not proof that the event was taken: the
			// consumer releases it after a stop too, and select picks at random
			// when processed and ctx.Done are both ready.
			if err := ctx.Err(); err != nil {
				completion.err = err
				return
			}
		}
	}
	// If the node's iter returned cleanly but the context was
	// canceled or its deadline elapsed, surface that as the
	// completion error: the node likely returned because it observed
	// ctx.Done(), and the consumer needs to classify it.
	if ctxErr := ctx.Err(); ctxErr != nil {
		completion.err = ctxErr
	}
}

// echoesCancellation reports whether err is the dying invocation
// context reflected back by a node rather than a failure the node
// determined for itself. runNode sets completion.err to ctx.Err() when
// a node returns because its context died, so the echo takes three
// shapes: the parent's own error (context.DeadlineExceeded for a
// request timeout), its cause (set via context.WithCancelCause), or a
// plain context.Canceled from the per-node cancel that cancelAll fires
// in response — a node can observe that before the parent's deadline.
//
// Only meaningful once s.parentCtx has failed; context.Cause returns
// nil for a live context and errors.Is(err, nil) is never true.
func (s *scheduler) echoesCancellation(err error) bool {
	return errors.Is(err, context.Canceled) ||
		errors.Is(err, s.parentCtx.Err()) ||
		errors.Is(err, context.Cause(s.parentCtx))
}

// cancelAll cancels every running task. Idempotent: canceled
// goroutines may still push events that already left the producer
// before observing ctx.Done(); the consumer continues draining
// until runsByName is empty.
//
// Also drops the trigger buffer so buffered triggers do not start
// after cancellation. They leave no NodeState of their own.
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
	s.triggers.clear()
}

// run is the single-consumer loop. It drains the eventQueue, applies
// state-side effects, yields events to the caller, and schedules
// successor nodes when a node completes. Returns when all running
// tasks have signaled completion.
// Start with draining=true only after canceling tasks for a caller that
// has already stopped consuming. In that mode yield is never called.
//
// On non-nil yield-return-false (caller broke from the range loop)
// or on a non-retryable node error, run cancels all in-flight
// tasks and continues draining until runsByName is empty, then
// surfaces the original error (if any) via yield.
//
// run runs on the caller's goroutine (the goroutine that called
// Workflow.Run); it is the only mutator of state.Nodes and the
// node-side accumulators.
func (s *scheduler) run(yield func(*session.Event, error) bool, draining bool) {
	var pendingErr error // first non-nil node error; surfaced after drain
	var pendingInventory failureInventory
	var cancelErr error      // cause of an external cancellation; surfaced only when no node reported an error
	consumerGone := draining // no further yield is allowed after consumer exit, including initial cleanup

	doneChan := s.parentCtx.Done()

	for len(s.runsByName) > 0 || len(s.retryTimers) > 0 {
		var item queueItem
		select {
		case item = <-s.eventQueue:
		case <-doneChan:
			doneChan = nil // Disable this case so we don't busy loop
		}

		// Cancellation of the invocation context is external to the
		// scheduler: cancelAll only cancels the per-node contexts, so
		// parentCtx stays live through sibling cancellation. Record the
		// cause and start draining here rather than in the doneChan case
		// alone, because a ready doneChan does not have to win the select
		// — a queued completion can be picked ahead of it, and the loop
		// can exit before it is ever selected (e.g. cancelAll stops the
		// last pending retry timer). Either way the run must not report
		// success.
		if cancelErr == nil && s.parentCtx.Err() != nil {
			cancelErr = context.Cause(s.parentCtx)
			if !draining {
				draining = true
				s.cancelAll()
			}
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
			// draining — so a blocked producer does not leak. The
			// producer infers "not taken" from its own canceled context,
			// so every path that stops the consumer cancels the producer
			// (cancelAll, or an ancestor context) before this release; a
			// new drain path must keep that order.
			if it.processed != nil {
				close(it.processed)
			}
		case completionItem:
			nr := s.runsByName[it.nodeName]
			err := s.handleCompletion(it, !draining)
			if !consumerGone {
				if ev := s.completionEvent(it.nodeName, nr, it.inventory); ev != nil && !yield(ev, nil) {
					consumerGone = true
					draining = true
					s.cancelAll()
				}
			}
			if err != nil && pendingErr == nil {
				pendingErr = err
				if it.err != nil {
					pendingInventory = it.inventory.clone()
				}
				if !draining {
					draining = true
					s.cancelAll()
				}
			}
			// A node and a slot just freed up, and the completion
			// may have buffered triggers: start what can start.
			if !draining {
				s.scheduleReady()
			}
		case retryItem:
			// The node stayed busy through its backoff, so nothing
			// else started it: restart it directly, keeping its slot.
			delete(s.retryTimers, it.act.node.Name())
			if !draining {
				s.startNode(it.act)
			}
		}
	}

	// All goroutines have returned and pushed their final events.
	// Surface the first error to the caller — but not if the caller
	// already broke the range loop, since yielding after a false return
	// panics the iterator.
	//
	// A node's own error outranks the cancellation cause: when a caller
	// cancels because a node failed, "context canceled" alone would hide
	// the reason the run stopped.
	runErr := pendingErr
	if runErr == nil {
		runErr = cancelErr
	}
	if runErr != nil && !consumerGone {
		// A WorkflowNode forwards this error unchanged. Bridge its recovery
		// evidence to the enclosing activation without wrapping that error.
		publishFailureInventory(s.parentCtx, s.graph, pendingInventory)
		yield(nil, runErr)
		return
	}

	// Skip finalize when draining (canceled, consumer gone, or a node
	// errored): the run did not complete normally.
	if !draining {
		if err := s.finalize(); err != nil {
			yield(nil, err)
		}
	}
}

// Output events alone cannot distinguish success from a partial result
// followed by failure, or record successful completion with no output.
// Keep those outcomes in history without persisting the node's error text.
//
// A completion with output is recorded only for an activation that used
// resume responses. Nothing else reads such a record, and one per node
// would add an event to every step of every run.
func (s *scheduler) completionEvent(name string, nr *nodeRun, inventory failureInventory) *session.Event {
	if name == Start.Name() || s.graph.isRootWrapper || s.graph.isNodeRootWrapper || nr == nil {
		return nil
	}
	ns := s.state.Nodes[name]
	if nr.end != nil {
		ns = nr.end
	}
	if ns == nil {
		return nil
	}
	var outcome string
	completedKey := workflowNodeCompletedKey
	switch ns.Status {
	case NodeFailed, NodePending:
		outcome = workflowNodeFailureOutcome
	case NodeCancelled:
		outcome = workflowNodeCancelledOutcome
	case NodeCompleted:
		if ns.Output != nil {
			if !nr.usedResponses {
				return nil
			}
			completedKey = workflowNodeResumeCompletedKey
		}
	default:
		return nil
	}
	ev := session.NewEvent(s.parentCtx, s.parentCtx.InvocationID())
	ev.Branch = ns.Branch
	path := nr.nodePath
	if path == "" {
		path = name
	}
	ev.NodeInfo = &session.NodeInfo{Path: path}
	ev.CustomMetadata = map[string]any{workflowBasePathPrefix + s.workflowName: s.parentCtx.Path()}
	if outcome != "" {
		// A failed attempt can still recover through retries or a parent
		// fallback. ErrorCode would terminate AgentTool/A2A consumers.
		ev.CustomMetadata[workflowNodeOutcomeKey] = outcome
		recordFailedChildPaths(ev.CustomMetadata, inventory)
	} else {
		ev.CustomMetadata[completedKey] = true
	}
	return ev
}

// finalize errors if more than one terminal node (no outgoing edges,
// excluding START) produced output in this run. It counts actual
// outputs, not graph shape, so fan-out and conditional-routing graphs
// with several terminal branches are fine as long as at most one yields
// output. A terminal node counts once however many activations it had,
// and counts when any of them produced output. No-op while a node waits
// for a user response: the run has not finished. A node parked by
// WaitForOutput does not hold finalize back. Mirrors adk-python's
// Workflow._finalize.
func (s *scheduler) finalize() error {
	for _, ns := range s.state.Nodes {
		if ns.Status == NodeWaiting && len(ns.Interrupts) > 0 {
			return nil
		}
	}

	var producers []string
	for name := range s.graph.terminalNodeNames() {
		if s.lastOutputs[name] != nil {
			producers = append(producers, name)
			continue
		}
		ns, ok := s.state.Nodes[name]
		if !ok || ns.Status != NodeCompleted {
			continue
		}
		if ns.Output != nil {
			producers = append(producers, name)
		}
	}

	if len(producers) > 1 {
		slices.Sort(producers)
		return fmt.Errorf("%w: %s", ErrMultipleTerminalOutputs, strings.Join(producers, ", "))
	}
	// The workflow's output schema (WithOutputSchema) applies to its
	// single terminal output. The conformed value is not emitted, as
	// adk-python's _finalize emits none.
	if len(producers) == 1 && s.graph.outputSchema != nil {
		if _, err := strictValidateOutput(s.state.Nodes[producers[0]].Output, s.graph.outputSchema); err != nil {
			return outputValidationError(s.workflowName, err)
		}
	}
	return nil
}

// defaultContentRole picks the role for node Content that left it
// empty: a FunctionResponse part is app-authored and takes RoleUser;
// everything else is model-authored.
func defaultContentRole(c *genai.Content) string {
	for _, p := range c.Parts {
		if p != nil && p.FunctionResponse != nil {
			return genai.RoleUser
		}
	}
	return genai.RoleModel
}

// carriesFunctionParts reports whether c holds a function call or a
// function response part.
func carriesFunctionParts(c *genai.Content) bool {
	return len(utils.FunctionCalls(c)) > 0 || len(utils.FunctionResponses(c)) > 0
}

// terminalAncestors returns the ancestor paths for a terminal node's output,
// or nil if nodeName is not terminal or the workflow is running at root level.
func (s *scheduler) terminalAncestors(nodeName string) []string {
	if !s.terminals[nodeName] || s.parentCtx.Path() == "" {
		return nil
	}
	return append([]string{s.parentCtx.Path()}, s.parentCtx.OutputForAncestors()...)
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
// matches InterruptID against the parent's NodeState.Interrupts,
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
	if !s.graph.isRootWrapper && !s.graph.isNodeRootWrapper {
		metadata := maps.Clone(it.ev.CustomMetadata)
		if metadata == nil {
			metadata = map[string]any{}
		}
		metadata[workflowBasePathPrefix+s.workflowName] = s.parentCtx.Path()
		it.ev.CustomMetadata = metadata
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
	// dynamic children are covered too. An authored event (an agent's
	// own) keeps the role its producer gave it, even an empty one, as
	// adk-python's node runtime adds no role to an agent's events;
	// except when it carries a function call or response, which the
	// next model request drops without a role.
	if it.ev.Content != nil && it.ev.Content.Role == "" &&
		(it.ev.Author == "" || carriesFunctionParts(it.ev.Content)) {
		it.ev.Content.Role = defaultContentRole(it.ev.Content)
	}
	expectedPath := nr.nodePath
	if expectedPath == "" {
		expectedPath = it.nodeName
	}
	var path string
	if it.ev.NodeInfo != nil {
		path = it.ev.NodeInfo.Path
	}
	isDescendant := path != "" && path != expectedPath && path != it.nodeName
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
		it.ev.NodeInfo.Path = expectedPath
		path = expectedPath
	}
	if value, delegated := completedDelegatedOutput(it.ev); delegated {
		if value == nil {
			return
		}
		// The dynamic body completed with a delegated (possibly cached)
		// result. It is now safe to use it as this activation's output.
		validated, err := validateAndStampOutput(s.nodesByName[it.nodeName], value, it.ev)
		if err != nil {
			nr.recordErr(err)
			return
		}
		it.ev.CustomMetadata[workflowDelegatedOutputKey] = validated
		nr.setOutput(validated, it.nodeName)
		return
	}
	if it.ev.Routes != nil {
		nr.setRoutingEvent(it.ev, it.nodeName)
	}
	if out, ok := childEventOutput(it.ev); ok {
		n := s.nodesByName[it.nodeName]
		if n == nil {
			// handleEvent only runs for registered nodes; a miss means
			// the registry is out of sync. Fail rather than forward
			// unvalidated.
			nr.recordErr(fmt.Errorf("%w: output validation: node %q not found in graph", ErrNodeFailed, it.nodeName))
			return
		}
		validated, err := validateAndStampOutput(n, out, it.ev)
		if err != nil {
			nr.recordErr(err)
			return
		}
		nr.setOutput(validated, it.nodeName)
		outputFor := append([]string{path}, s.terminalAncestors(it.nodeName)...)
		if it.ev.NodeInfo == nil {
			it.ev.NodeInfo = &session.NodeInfo{}
		}
		it.ev.NodeInfo.OutputFor = outputFor
	}
}

// handleCompletion finalizes a node's run: transitions its lifecycle
// status, removes the live task, and (if scheduleNewWork is true)
// schedules retries or successors. When the consumer is draining (caller
// stopped or a node failed), pass scheduleNewWork=false so the workflow
// does not keep dispatching new nodes after cancellation.
//
// The returned error is the node's own error (NodeFailed); nil on
// clean success or sibling cancellation.
//
// # Human-input waiting branch
//
// When an activation completes cleanly having raised at least one
// long-running interrupt (the nodeRun collects these from
// Event.LongRunningToolIDs), the node transitions to NodeWaiting
// instead of NodeCompleted, the interrupt IDs are recorded on
// NodeState.Interrupts, and successors are not scheduled. The
// scheduler's main loop terminates naturally when every live node
// has either completed or moved into NodeWaiting,
// at which point Workflow.Run's iterator exhausts and the caller
// observes the pause by inspecting RunState.
//
// The waiting branch is checked after the error/cancel branches,
// so a node that fails for any reason (returned error, panic,
// context cancel, multiple-output, multiple-routing-event,
// multiple-input-request) does not silently park in NodeWaiting:
// failures take precedence and surface as NodeFailed.
func (s *scheduler) handleCompletion(it completionItem, scheduleNewWork bool) error {
	ns := s.state.EnsureNode(it.nodeName)
	nr := s.runsByName[it.nodeName]
	// For retryable nodes still delete them from run variables. If the node is retried,
	// it will get a fresh accumulator in startNode to avoid ErrMultipleOutputs
	// from partial results of this failed run.
	delete(s.runsByName, it.nodeName)
	delete(s.runCancels, it.nodeName)

	// A dead invocation context means the cancellation came from outside
	// the scheduler, so this completion is not the benign sibling
	// cancellation handled below — cancelAll never touches parentCtx.
	// Classified before the retry block: ShouldRetry's default predicate
	// does not exclude cancellation, and re-running a node under a dead
	// context only re-fails.
	//
	// run surfaces the cancellation cause on its own, so a node that has
	// nothing of its own to report returns nil here. One that failed for
	// its own reason keeps that error, which says more than "canceled",
	// unless RetryConfig would run it again: that failure is not the
	// node's outcome, and the retry it waits for is what the cancellation
	// stops, as adk-python's backoff sleep is canceled
	// (_node_runner.py _attempt_retry).
	if s.parentCtx.Err() != nil {
		if it.err != nil && !s.echoesCancellation(it.err) && !s.wouldRetry(it.nodeName, it.err, ns.Attempt+1) {
			ns.Status = NodeFailed
			return it.err
		}
		// Matches adk-python, which marks every task it reaps during
		// shutdown CANCELLED regardless of who canceled it //nolint:misspell
		// (_workflow.py _cleanup_all_tasks, run from a finally).
		ns.Status = NodeCancelled
		return nil
	}
	if errors.Is(it.err, context.Canceled) {
		ns.Status = NodeCancelled
		return nil // sibling cancellation; not the original error
	}
	// WaitForOutput park: a pause, not a failure, and with no interrupt
	// ID. Mirrors adk-python's wait_for_output WAITING state.
	if errors.Is(it.err, ErrNodeWaitingForOutput) {
		ns.Status = NodeWaiting
		return nil
	}
	if it.err != nil {
		currentNode := s.nodesByName[it.nodeName]
		if scheduleNewWork && currentNode != nil {
			cfg := currentNode.Config()
			if cfg.RetryConfig != nil {
				ns.Attempt = ns.Attempt + 1

				if ShouldRetry(cfg.RetryConfig, it.err, ns.Attempt) {
					delay := CalculateDelay(cfg.RetryConfig, ns.Attempt)
					ns.Status = NodePending
					var runID string
					if nr != nil {
						runID = nr.runID
						// A dynamic child failed the attempt: the retried
						// body starts its children at run id 1 again. Any
						// other error keeps the children's count going, so
						// the retry re-runs them under new run ids. Mirrors
						// adk-python's DynamicNodeFailError branch
						// (workflow/_node_runner.py).
						var nre *NodeRunError
						if errors.As(it.err, &nre) && errors.Is(nre, ErrNodeFailed) {
							s.counters.reset(nr.nodePath)
						}
					}
					// ns.Branch is the activation's final branch, so the
					// retry derives no sub-branch.
					s.scheduleRetry(pendingActivation{
						node: currentNode, input: ns.Input, triggeredBy: ns.TriggeredBy,
						branch: ns.Branch, runID: runID,
					}, delay)
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
	// scheduleNewWork flag — an interrupt that survived the run
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

	// WaitForOutput park: an activation of a WaitForOutput node that
	// produced neither output nor route waits for its next trigger. It
	// has no interrupt ID, triggers no successor, and does not make the
	// node busy. Mirrors adk-python _handle_completion.
	if cn := s.nodesByName[it.nodeName]; cn != nil && waitsForOutput(cn) &&
		(nr == nil || ((!nr.hasOutput || nr.output == nil) && nr.routingEvent == nil)) {
		ns.Status = NodeWaiting
		ns.Interrupts = nil
		return nil
	}

	ns.Status = NodeCompleted
	ns.Attempt = 0
	if nr != nil && nr.hasOutput {
		ns.Output = nr.output
		if nr.output != nil {
			s.lastOutputs[it.nodeName] = nr.output
		}
	}
	// Release the accumulated re-entry response history; the node
	// has finished and a future activation (if any, e.g. via
	// loop-back routing) starts a fresh lifecycle.
	if nr != nil {
		nr.usedResponses = len(ns.ResumedInputs) > 0
	}
	ns.ResumedInputs = nil
	if nr != nil {
		end := *ns
		nr.end = &end
	}

	if !scheduleNewWork {
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

	// The run loop starts the buffered triggers (scheduleReady) once
	// this completion is handled.
	s.bufferSuccessors(it.nodeName, findSuccessors(s.graph, currentNode, input, routingEv, ns.Branch))
	return nil
}

// successor is findSuccessors' result for one target. The triggeredBy
// field carries the upstream node's name for persistence on NodeState
// (used by Resume to reconstruct the activation chain across
// pause/resume turns).
type successor struct {
	node        Node
	input       any // the completing node's output; nil for a join target
	triggeredBy string
	branch      string // the completing activation's branch
	subBranch   bool   // true for a non-join target when the completion has more than one target
	join        bool   // target is a *JoinNode; its barrier is evaluated by bufferSuccessors
}

// findSuccessors evaluates the outgoing edges of currentNode against
// the optional routing event and returns the targets this completion
// triggers:
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
// JoinNode targets are returned with join set and no input; their
// barrier is evaluated by bufferSuccessors.
//
// Branch derivation: every target carries parentBranch, the
// completing activation's branch. When the completion has more than
// one target (JoinNode targets included), each non-Join target gets
// subBranch, and startNode derives "<parentBranch>.<name>@<run id>"
// from the run id it allocates. A single target inherits parentBranch
// unchanged. Mirrors adk-python's use_sub_branch = len(next_nodes) > 1
// (workflow/_workflow.py _buffer_downstream_triggers).
func findSuccessors(g *graph, currentNode Node, input any, event *session.Event, parentBranch string) []successor {
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
			out = appendSuccessor(out, g, edge.To, input, from, parentBranch)
			added[edge.To] = struct{}{}
			continue
		}
		if edge.Route == Default {
			defaultRouteNode = edge.To
			continue
		}
		if event != nil && edge.Route.Matches(event) {
			out = appendSuccessor(out, g, edge.To, input, from, parentBranch)
			added[edge.To] = struct{}{}
			concreteMatched = true
		}
	}
	if !concreteMatched && defaultRouteNode != nil {
		out = appendSuccessor(out, g, defaultRouteNode, input, from, parentBranch)
	}
	if len(out) > 1 {
		for i := range out {
			out[i].subBranch = !out[i].join
		}
	}
	return out
}

// appendSuccessor records a routing match in the dispatch list. A
// non-JoinNode target receives input as resolved for it by
// graph.startSeedFor; a JoinNode target is marked join and receives
// its input when bufferBarrier finds its barrier satisfied.
func appendSuccessor(out []successor, g *graph, target Node, input any, triggeredBy, parentBranch string) []successor {
	if _, isJoin := target.(*JoinNode); isJoin {
		return append(out, successor{node: target, triggeredBy: triggeredBy, branch: parentBranch, join: true})
	}
	return append(out, successor{
		node:        target,
		input:       g.startSeedFor(input, target),
		triggeredBy: triggeredBy,
		branch:      parentBranch,
	})
}

// bufferSuccessors buffers the triggers of one completion of from and
// returns how many it buffered. Every non-Join trigger is buffered
// before any JoinNode barrier is judged, so a predecessor this same
// completion re-triggers counts as unsettled. It starts nothing.
func (s *scheduler) bufferSuccessors(from string, succs []successor) int {
	n := 0
	for _, sc := range succs {
		if sc.join {
			continue
		}
		s.enqueue(pendingActivation{
			node: sc.node, input: sc.input, triggeredBy: from,
			branch: sc.branch, subBranch: sc.subBranch,
		})
		n++
	}
	for _, sc := range succs {
		if sc.join && s.bufferBarrier(from, sc.node) {
			n++
		}
	}
	return n
}

// bufferBarrier buffers a trigger for JoinNode target when every
// predecessor is settled, and reports whether it did. A predecessor is
// settled when it is START, or is NodeCompleted and either is from (the
// node whose completion is being handled) or has no buffered trigger.
// The trigger's input maps each predecessor to its last output, and its
// branch is the common prefix of the predecessors' branches.
func (s *scheduler) bufferBarrier(from string, target Node) bool {
	settled := func(name string) bool { return name == from || !s.hasQueued(name) }
	in, ok := aggregatePredecessorOutputs(s.graph, s.state, s.lastOutputs, target, settled)
	if !ok {
		return false
	}
	branch := commonBranchPrefix(aggregatePredecessorBranches(s.graph, s.state, target))
	s.enqueue(pendingActivation{node: target, input: in, triggeredBy: from, branch: branch})
	return true
}

// aggregatePredecessorOutputs returns a map of predecessor name to
// output for every predecessor of target. Returns (nil, false) unless
// every predecessor other than START is NodeCompleted with settled(name)
// true. A predecessor's value is lastOutputs[name], the last non-nil
// output any of its activations produced in this run; a name absent from
// lastOutputs (its output came from an earlier turn, or it never
// produced one) contributes its NodeState.Output, which may be nil.
// START contributes the workflow's seed as resolved by
// graph.startSeedFor.
func aggregatePredecessorOutputs(g *graph, state *RunState, lastOutputs map[string]any, target Node, settled func(name string) bool) (map[string]any, bool) {
	predEdges := g.predecessorsOf(target)
	aggregated := make(map[string]any, len(predEdges))
	for _, edge := range predEdges {
		name := edge.From.Name()
		ns := state.Nodes[name]
		if ns == nil || ns.Status != NodeCompleted {
			return nil, false
		}
		if edge.From == Start {
			// START carries the workflow's seed in its Input; it has
			// no output of its own.
			aggregated[name] = g.startSeedFor(ns.Input, target)
			continue
		}
		if !settled(name) {
			return nil, false
		}
		if out, ok := lastOutputs[name]; ok {
			aggregated[name] = out
			continue
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
