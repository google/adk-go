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
	"iter"
	"sync"
	"time"

	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow/internal/engine"
)

// runSequential drives the workflow with goroutine-per-node scheduling.
// Events stream live through a multiplexed channel; the iter.Seq2 returned
// to the caller forwards them as they arrive.
//
// Parallelism is bounded by Workflow.maxConcurrency. When zero, scheduling
// is unbounded. JoinNodes (RequiresAllPredecessors) fire only after every
// predecessor has completed.
//
// Phases 4-5 add resume and dynamic-node scheduling on top of this loop.
func (w *Workflow) runSequential(
	ic agent.InvocationContext,
	input any,
	parentPath string,
) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		path := engine.JoinPath(parentPath, w.Name(), 1)
		state := newRunState(w.graph, input, path)

		// Phase 4: rehydrate from session events. If the session already has
		// completed events for nodes in this workflow, skip them and queue
		// only the non-completed successors.
		w.rehydrate(state, ic, path)

		ctx, cancel := context.WithCancel(ic)
		defer cancel()

		// Two channels: events flow live to the consumer; completions tell
		// the main loop when a node finished (so it can queue successors).
		events := make(chan eventOrErr, 16)
		completions := make(chan nodeCompletion, 16)

		var wg sync.WaitGroup
		var sem chan struct{}
		if w.maxConcurrency > 0 {
			sem = make(chan struct{}, w.maxConcurrency)
		}

		// pending tracks in-flight nodes. Incremented when a goroutine is
		// spawned, decremented when its completion is consumed.
		pending := 0

		spawn := func() {
			for {
				ref := state.popReady()
				if ref == nil {
					return
				}
				if sem != nil {
					select {
					case sem <- struct{}{}:
					case <-ctx.Done():
						return
					}
				}
				wg.Add(1)
				pending++
				go w.runNode(ctx, ic, *ref, &wg, sem, events, completions)
			}
		}

		spawn()

		// Main loop: drive completions, queue successors, forward events.
		// Exits when there are no in-flight tasks AND no pending triggers.
		for pending > 0 {
			select {
			case msg := <-events:
				if !yield(msg.e, msg.err) {
					cancel()
					goto drain
				}
			case c := <-completions:
				pending--
				if c.err != nil {
					yield(nil, fmt.Errorf("workflow %q: node %q: %w", w.Name(), c.nodeName, c.err))
					cancel()
					goto drain
				}
				state.complete(c.nodeName, c.outputs, c.route)
				spawn()
			case <-ctx.Done():
				goto drain
			}
		}

		// All goroutines have completed and been observed. Drain any events
		// still in the buffer.
		wg.Wait()
		for {
			select {
			case msg := <-events:
				if !yield(msg.e, msg.err) {
					return
				}
			default:
				return
			}
		}

	drain:
		// Aborted path: cancel, then absorb signals from in-flight goroutines
		// until they all finish so they don't block on send.
		cancel()
		drainDone := make(chan struct{})
		go func() {
			wg.Wait()
			close(drainDone)
		}()
		for {
			select {
			case <-events:
			case <-completions:
			case <-drainDone:
				return
			}
		}
	}
}

// nodeCompletion is sent on the completions channel after a node's
// RunImpl returns.
type nodeCompletion struct {
	nodeName string
	outputs  []any
	route    *Route
	err      error
}

// eventOrErr is the value type of the events channel.
type eventOrErr struct {
	e   *session.Event
	err error
}

// runNode is the goroutine body for a single node attempt. It builds the
// NodeContext and emitter, applies retry/timeout policy, and posts the
// completion when done.
func (w *Workflow) runNode(
	ctx context.Context,
	ic agent.InvocationContext,
	ref readyRef,
	wg *sync.WaitGroup,
	sem chan struct{},
	events chan<- eventOrErr,
	completions chan<- nodeCompletion,
) {
	defer wg.Done()
	if sem != nil {
		defer func() { <-sem }()
	}

	node, ok := w.graph.nodes[ref.name]
	if !ok {
		completions <- nodeCompletion{
			nodeName: ref.name,
			err:      fmt.Errorf("node %q missing from graph", ref.name),
		}
		return
	}

	// Per-attempt loop. RetryConfig defaults to no retry when nil.
	spec := node.Spec()
	maxAttempts := 1
	if spec.RetryConfig != nil {
		c := spec.RetryConfig.withDefaults()
		maxAttempts = c.MaxAttempts
	}

	var (
		err      error
		outputs  []any
		emitted  []*session.Event
		gotRoute *Route
	)
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		em := newCollectingEmitter(node, ic, ref.runID, w.nodePathFor(ref.name, ref.runID))
		ctxAttempt := ctx
		var cancelAttempt context.CancelFunc
		if spec.Timeout > 0 {
			ctxAttempt, cancelAttempt = context.WithTimeout(ctx, spec.Timeout)
		}

		nodeCtx := &NodeContext{
			InvocationContext: ic,
			nodePath:          em.nodePath,
			runID:             fmt.Sprintf("%d", ref.runID),
			actions: &session.EventActions{
				StateDelta:    map[string]any{},
				ArtifactDelta: map[string]int64{},
			},
			parentEmitter: em,
			resumeInputs:  ref.resumeInputs,
		}
		em.ctx = nodeCtx

		// Run on the goroutine; cancellation triggers timeout via ctxAttempt.
		err = runNodeOnce(ctxAttempt, node, nodeCtx, em, ref.input)
		if cancelAttempt != nil {
			cancelAttempt()
		}

		emitted = em.events
		outputs = em.outputs
		gotRoute = em.route

		if err == nil {
			break
		}
		if !shouldRetry(err, spec.RetryConfig) {
			break
		}
		if attempt == maxAttempts {
			break
		}
		// Drop events from this failed attempt; the next attempt re-emits.
		emitted = nil
		outputs = nil
		gotRoute = nil

		delay := spec.RetryConfig.DelayFor(attempt)
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			err = ctx.Err()
			break
		}
	}

	// Forward emitted events.
	for _, ev := range emitted {
		select {
		case events <- eventOrErr{e: ev}:
		case <-ctx.Done():
			completions <- nodeCompletion{nodeName: ref.name, err: ctx.Err()}
			return
		}
	}

	completions <- nodeCompletion{
		nodeName: ref.name,
		outputs:  outputs,
		route:    gotRoute,
		err:      err,
	}
}

// runNodeOnce runs the node a single time with timeout enforcement.
func runNodeOnce(
	ctx context.Context,
	node Node,
	nodeCtx *NodeContext,
	em *collectingEmitter,
	input any,
) error {
	type result struct{ err error }
	done := make(chan result, 1)
	go func() {
		// Recover panics so a misbehaving node can't crash the orchestrator.
		defer func() {
			if r := recover(); r != nil {
				done <- result{err: fmt.Errorf("node panic: %v", r)}
			}
		}()
		done <- result{err: node.RunImpl(nodeCtx, input, em)}
	}()
	select {
	case r := <-done:
		return r.err
	case <-ctx.Done():
		// Note: the node goroutine may keep running; we report the timeout
		// promptly so retry can kick in. The leaked goroutine ends when the
		// node naturally returns.
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return ErrNodeTimeout
		}
		return ctx.Err()
	}
}

// ErrNodeTimeout is returned when a node's per-attempt timeout fires.
var ErrNodeTimeout = errors.New("workflow: node timed out")

// shouldRetry reports whether err is retryable under cfg.
func shouldRetry(err error, cfg *RetryConfig) bool {
	if err == nil || cfg == nil {
		return false
	}
	if len(cfg.Retryable) == 0 {
		return true
	}
	for _, s := range cfg.Retryable {
		if errors.Is(err, s) {
			return true
		}
	}
	return false
}

// nodePathFor builds the hierarchical path for a child node.
func (w *Workflow) nodePathFor(name string, runID int) string {
	return engine.JoinPath(w.Name()+"@1", name, runID)
}

// runState manages the orchestrator's working memory: pending triggers,
// per-name run counter, predecessor counts (for JoinNode), and per-node
// completion tracking.
type runState struct {
	graph  *workflowGraph
	queue  []readyRef
	runIDs map[string]int

	// pendingPredecessors counts how many predecessor edges remain to
	// fire before a JoinNode can run. Decremented as each predecessor
	// completes; reaches 0 when ready.
	pendingPredecessors map[string]int

	// joinedInputs aggregates predecessor outputs for JoinNodes, keyed
	// by predecessor name.
	joinedInputs map[string]map[string]any

	// completed tracks which nodes have produced a final output (not
	// strictly needed in Phase 3 but kept for Phase 4 resume).
	completed map[string]bool

	// resumeInputs maps interrupt IDs (originally emitted as
	// adk_request_input FunctionCall ids) to the user's matching
	// FunctionResponse value. Populated by rehydrate on resume.
	resumeInputs map[string]any

	mu sync.Mutex
}

type readyRef struct {
	name         string
	runID        int
	input        any
	resumeInputs map[string]any
}

func newRunState(g *workflowGraph, input any, _ string) *runState {
	s := &runState{
		graph:               g,
		runIDs:              map[string]int{},
		pendingPredecessors: map[string]int{},
		joinedInputs:        map[string]map[string]any{},
		completed:           map[string]bool{},
	}
	// Compute initial predecessor counts for join-style nodes.
	for name, node := range g.nodes {
		if name == "__START__" {
			continue
		}
		if node.Spec().RequiresAllPredecessors {
			s.pendingPredecessors[name] = len(g.in[name])
		}
	}
	for _, name := range g.startNames {
		s.runIDs[name]++
		s.queue = append(s.queue, readyRef{name: name, runID: s.runIDs[name], input: input})
	}
	return s
}

func (s *runState) popReady() *readyRef {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.queue) == 0 {
		return nil
	}
	ref := s.queue[0]
	s.queue = s.queue[1:]
	return &ref
}

// complete records the completion of a node and queues any successors
// whose firing condition is satisfied. The node's last emitted output is
// forwarded as the input to each successor; for JoinNodes the inputs are
// collected into a map keyed by predecessor name.
func (s *runState) complete(name string, outputs []any, route *Route) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.completed[name] = true

	var lastOut any
	if len(outputs) > 0 {
		lastOut = outputs[len(outputs)-1]
	}

	for _, e := range s.graph.out[name] {
		// Phase 3: route matching. An edge with no Routes is unconditional;
		// otherwise the upstream node's emitted route must match one of the
		// edge's allowed routes.
		if len(e.routes) > 0 && !routeMatches(route, e.routes) {
			continue
		}
		toNode := s.graph.nodes[e.to]
		if toNode != nil && toNode.Spec().RequiresAllPredecessors {
			if _, ok := s.joinedInputs[e.to]; !ok {
				s.joinedInputs[e.to] = map[string]any{}
			}
			s.joinedInputs[e.to][name] = lastOut
			s.pendingPredecessors[e.to]--
			if s.pendingPredecessors[e.to] <= 0 {
				s.runIDs[e.to]++
				agg := s.joinedInputs[e.to]
				s.queue = append(s.queue, readyRef{
					name: e.to, runID: s.runIDs[e.to], input: agg,
					resumeInputs: s.resumeInputs,
				})
				delete(s.joinedInputs, e.to)
			}
			continue
		}
		s.runIDs[e.to]++
		s.queue = append(s.queue, readyRef{
			name: e.to, runID: s.runIDs[e.to], input: lastOut,
			resumeInputs: s.resumeInputs,
		})
	}
}

// routeMatches reports whether got matches any of the route values
// allowed by an edge.
func routeMatches(got *Route, allowed []Route) bool {
	if got == nil {
		// No route emitted: only the default branch fires.
		for _, r := range allowed {
			if r.Match(DefaultRoute) {
				return true
			}
		}
		return false
	}
	for _, r := range allowed {
		if got.Match(r) {
			return true
		}
	}
	// Fallback: if no explicit match, take the default branch when present.
	for _, r := range allowed {
		if r.Match(DefaultRoute) {
			return true
		}
	}
	return false
}

// collectingEmitter buffers events during a node's RunImpl invocation.
// The orchestrator forwards the buffered events to the live events
// channel after the node completes (or on each yield in streaming nodes —
// streaming is implemented in the live forwarding path; the orchestrator
// reads em.events after RunImpl returns).
type collectingEmitter struct {
	node     Node
	ic       agent.InvocationContext
	runID    int
	nodePath string

	ctx          *NodeContext
	events       []*session.Event
	outputs      []any
	route        *Route
	interruptIDs []string
}

func newCollectingEmitter(node Node, ic agent.InvocationContext, runID int, nodePath string) *collectingEmitter {
	return &collectingEmitter{node: node, ic: ic, runID: runID, nodePath: nodePath}
}

func (e *collectingEmitter) Event(ev *session.Event) error {
	if ev == nil {
		return errors.New("emitter: nil event")
	}
	e.attachNodeInfo(ev, false)
	e.events = append(e.events, ev)
	return nil
}

func (e *collectingEmitter) Output(v any) error {
	ev := session.NewEvent(e.ic.InvocationID())
	ev.Author = e.node.Name()
	ev.Branch = e.ic.Branch()
	ev.LLMResponse = model.LLMResponse{}
	if e.ctx != nil {
		ev.Actions.StateDelta = e.ctx.Actions().StateDelta
		ev.Actions.ArtifactDelta = e.ctx.Actions().ArtifactDelta
	}
	e.attachNodeInfo(ev, false)
	ev.Actions.NodeInfo.Output = v
	e.events = append(e.events, ev)
	e.outputs = append(e.outputs, v)
	return nil
}

func (e *collectingEmitter) RequestInput(r RequestInput) error {
	if r.InterruptID == "" {
		r.InterruptID = uuid.NewString()
	}
	args := map[string]any{}
	if r.Prompt != "" {
		args["prompt"] = r.Prompt
	}
	// Encode the interrupt as a long-running FunctionCall so the existing
	// runner / agent client semantics treat it as a pause point.
	ev := session.NewEvent(e.ic.InvocationID())
	ev.Author = e.node.Name()
	ev.Branch = e.ic.Branch()
	ev.LLMResponse = model.LLMResponse{
		Content: &genai.Content{
			Role: genai.RoleModel,
			Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{
					ID:   r.InterruptID,
					Name: "adk_request_input",
					Args: args,
				},
			}},
		},
	}
	ev.LongRunningToolIDs = []string{r.InterruptID}
	e.attachNodeInfo(ev, true)
	ev.Actions.NodeInfo.InterruptID = r.InterruptID
	e.events = append(e.events, ev)
	e.interruptIDs = append(e.interruptIDs, r.InterruptID)
	return nil
}

func (e *collectingEmitter) StateDelta(delta map[string]any) error {
	if e.ctx == nil {
		return nil
	}
	for k, v := range delta {
		e.ctx.Actions().StateDelta[k] = v
	}
	return nil
}

func (e *collectingEmitter) ArtifactDelta(delta map[string]int64) error {
	if e.ctx == nil {
		return nil
	}
	for k, v := range delta {
		e.ctx.Actions().ArtifactDelta[k] = v
	}
	return nil
}

// SetRoute records the route value the node emits. Future Phase 5+
// emitter API may make this an explicit method; for Phase 3 it's
// internal so wrappers (LoopNode, conditional helpers) can drive route
// matching deterministically.
func (e *collectingEmitter) SetRoute(r Route) { e.route = &r }

func (e *collectingEmitter) attachNodeInfo(ev *session.Event, interrupt bool) {
	if ev.Actions.NodeInfo == nil {
		ev.Actions.NodeInfo = &session.NodeInfo{Path: e.nodePath, Interrupt: interrupt}
	} else {
		ev.Actions.NodeInfo.Path = e.nodePath
		if interrupt {
			ev.Actions.NodeInfo.Interrupt = true
		}
	}
}
