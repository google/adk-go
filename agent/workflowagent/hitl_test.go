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

package workflowagent

import (
	"context"
	"errors"
	"iter"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// =============================================================================
// Tests
// =============================================================================

// TestWorkflowAgent_RunThenResume_Handoff exercises the canonical
// round-trip: a fresh Run pauses on a node that requested input,
// and a follow-up Resume turn delivers the response which flows
// to the asker's successor as its input.
func TestWorkflowAgent_RunThenResume_Handoff(t *testing.T) {
	var handlerInput atomic.Value
	asker := newAskerNode("approve_or_reject", "Please decide", nil)
	handler := newStringHandlerNode("handler", &handlerInput)

	a := makeAgent(t, workflow.Chain(workflow.Start, asker, handler))
	sess := newFakeSession()

	// Turn 1: fresh Run; should pause with a RequestedInput.
	turn1 := runFreshTurn(t, sess, a, "draft")
	if got := findRequest(turn1); got != "approve_or_reject" {
		t.Fatalf("turn 1 RequestedInput = %q, want %q", got, "approve_or_reject")
	}
	if v := handlerInput.Load(); v != nil {
		t.Errorf("handler ran during turn 1; got input %v, want it not to run", v)
	}

	// Turn 2: resume with a payload; handler should run and
	// receive the payload as its input.
	turn2 := drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("approve_or_reject", "approve"))), nil)
	if findRequest(turn2) != "" {
		t.Errorf("turn 2 unexpectedly emitted a RequestedInput")
	}
	if got, want := handlerInput.Load(), "approve"; got != want {
		t.Errorf("handler input = %v, want %q", got, want)
	}
}

// TestWorkflowAgent_Resume_RestoresStateFromSession verifies that
// the run state survives between agent instances backed by the
// same session: after Run, a fresh agent built from the same
// edges (simulating a process restart) can still Resume.
func TestWorkflowAgent_Resume_RestoresStateFromSession(t *testing.T) {
	var handlerCalled atomic.Bool

	// makeNodes returns fresh node instances per agent so the test
	// proves resume goes through session.State, not through any
	// shared in-memory references between a1 and a2.
	makeNodes := func() (workflow.Node, workflow.Node) {
		return newAskerNode("human_approval", "approve?", nil),
			newFlagHandlerNode("handler", &handlerCalled)
	}

	sess := newFakeSession()

	// First agent instance: Run → pause.
	asker1, handler1 := makeNodes()
	a1 := makeAgent(t, workflow.Chain(workflow.Start, asker1, handler1))
	turn1 := runFreshTurn(t, sess, a1, "draft")
	if findRequest(turn1) != "human_approval" {
		t.Fatalf("first agent did not pause as expected")
	}

	// Second agent instance, same session: Resume.
	asker2, handler2 := makeNodes()
	a2 := makeAgent(t, workflow.Chain(workflow.Start, asker2, handler2))
	drainAgent(t, sess, a2.Run(newMockCtx(sess, a2, resumeMessage("human_approval", "yes"))), nil)
	if !handlerCalled.Load() {
		t.Error("handler did not run after resume on a fresh agent instance")
	}
}

// TestWorkflowAgent_Resume_Idempotent verifies that two Resume
// calls with the same payload run the handler only once.
func TestWorkflowAgent_Resume_Idempotent(t *testing.T) {
	var handlerRuns atomic.Int32
	asker := newAskerNode("approve", "?", nil)
	handler := newCountingHandlerNode("handler", &handlerRuns)

	a := makeAgent(t, workflow.Chain(workflow.Start, asker, handler))
	sess := newFakeSession()

	runFreshTurn(t, sess, a, "x")
	// First resume: matches the waiting node, runs the handler.
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("approve", "yes"))), nil)
	// Second resume with the same payload: PendingRequest was
	// consumed by the first call, so no waiting node matches and
	// Resume yields ErrNothingToResume rather than re-running.
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("approve", "yes"))), workflow.ErrNothingToResume)

	if got := handlerRuns.Load(); got != 1 {
		t.Errorf("handler runs = %d, want 1 (duplicate Resume must not re-run the handler)", got)
	}
}

// TestWorkflowAgent_Resume_NoMatchingResponse verifies the
// stale-response signal: a Resume turn that carries a
// FunctionResponse for an InterruptID that does not match any
// waiting node yields ErrNothingToResume so the caller can
// distinguish a no-op resume from a real one (e.g. show "your
// reply targets a stale request" in the UI).
func TestWorkflowAgent_Resume_NoMatchingResponse(t *testing.T) {
	asker := newAskerNode("real_id", "?", nil)

	a := makeAgent(t, workflow.Chain(workflow.Start, asker))
	sess := newFakeSession()

	// Pause once.
	runFreshTurn(t, sess, a, "x")

	// Submit a FunctionResponse for an unknown ID. detectResume
	// will see the magic name, load state, build a responses map,
	// but no waiting node will match — Resume yields
	// ErrNothingToResume so the caller can distinguish the
	// successful-but-no-effect case from a real resume.
	turn := drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("unknown_id", "x"))), workflow.ErrNothingToResume)
	if findRequest(turn) != "" {
		t.Errorf("unmatched resume produced a new RequestedInput; got %v", turn)
	}
}

// TestWorkflowAgent_Resume_SchemaValidation_Pass verifies that a
// response payload conforming to ResponseSchema is delivered to
// the handler unchanged (the validator coerces but here the
// shape already matches).
func TestWorkflowAgent_Resume_SchemaValidation_Pass(t *testing.T) {
	var handlerInput atomic.Value
	asker := newAskerNode("approval", "decide", approvalSchema())
	handler := newMapHandlerNode("handler", &handlerInput)

	a := makeAgent(t, workflow.Chain(workflow.Start, asker, handler))
	sess := newFakeSession()

	runFreshTurn(t, sess, a, "x")
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("approval", map[string]any{"approved": true}))), nil)

	got, ok := handlerInput.Load().(map[string]any)
	if !ok || got["approved"] != true {
		t.Errorf("handler input = %v, want map with approved=true", handlerInput.Load())
	}
}

// TestWorkflowAgent_Resume_SchemaValidation_Fail verifies that a
// response payload that violates ResponseSchema surfaces
// ErrInvalidResumeResponse and leaves the node parked, so a
// follow-up turn with a corrected payload still works.
func TestWorkflowAgent_Resume_SchemaValidation_Fail(t *testing.T) {
	var handlerRuns atomic.Int32
	asker := newAskerNode("approval", "decide", approvalSchema())
	handler := newCountingHandlerNode("handler", &handlerRuns)

	a := makeAgent(t, workflow.Chain(workflow.Start, asker, handler))
	sess := newFakeSession()

	// Pause.
	runFreshTurn(t, sess, a, "x")

	// Submit invalid payload (string instead of {approved: bool}).
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("approval", "not an object"))), workflow.ErrInvalidResumeResponse)
	if handlerRuns.Load() != 0 {
		t.Fatal("handler ran despite schema validation failure")
	}

	// Retry with valid payload — should succeed.
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("approval", map[string]any{"approved": true}))), nil)
	if handlerRuns.Load() != 1 {
		t.Errorf("handler runs after retry = %d, want 1", handlerRuns.Load())
	}
}

// TestWorkflowAgent_Resume_FanOut verifies that a handoff resume
// from an asker with multiple successors fans out the response
// to every successor, exactly as a normal output would.
func TestWorkflowAgent_Resume_FanOut(t *testing.T) {
	var hits atomic.Int32
	asker := newAskerNode("fan", "?", nil)
	h1 := newCountingHandlerNode("h1", &hits)
	h2 := newCountingHandlerNode("h2", &hits)
	h3 := newCountingHandlerNode("h3", &hits)

	a := makeAgent(t, []workflow.Edge{
		{From: workflow.Start, To: asker},
		{From: asker, To: h1},
		{From: asker, To: h2},
		{From: asker, To: h3},
	})
	sess := newFakeSession()

	runFreshTurn(t, sess, a, "x")
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("fan", "go"))), nil)

	if got := hits.Load(); got != 3 {
		t.Errorf("successor hits = %d, want 3", got)
	}
}

// TestWorkflowAgent_FreshTurn_NotMistakenForResume verifies the
// detectResume guard: a fresh user message that happens to have
// no FunctionResponse part does NOT trip the resume path even if
// a RunState is persisted (e.g. from a paused or completed prior
// workflow). Important because session.State may carry leftover
// state from previous runs.
func TestWorkflowAgent_FreshTurn_NotMistakenForResume(t *testing.T) {
	var firstRun atomic.Bool
	var secondRun atomic.Bool

	// Custom asker (not newAskerNode) because each instance must
	// flip its own flag before the request is yielded, so the
	// test can detect that the asker truly re-ran on turn 2.
	makeAsker := func(flag *atomic.Bool) workflow.Node {
		return newHitlNode("asker", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			flag.Store(true)
			yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
				InterruptID: "ask",
				Message:     "?",
			}), nil)
		})
	}

	a := makeAgent(t, workflow.Chain(workflow.Start, makeAsker(&firstRun)))
	sess := newFakeSession()

	// Turn 1: fresh; pauses.
	runFreshTurn(t, sess, a, "x")
	if !firstRun.Load() {
		t.Fatal("asker did not run on turn 1")
	}

	// Turn 2: another fresh user message (no FunctionResponse).
	// detectResume should return false; Workflow.Run is invoked.
	a2 := makeAgent(t, workflow.Chain(workflow.Start, makeAsker(&secondRun)))
	runFreshTurn(t, sess, a2, "fresh")
	if !secondRun.Load() {
		t.Error("a fresh user message was misinterpreted as a resume; asker did not run on turn 2")
	}
}

// TestWorkflowAgent_RunThenResume_DynamicNodeOrchestrator verifies
// that a child RequestedInput inside a dynamic-node orchestrator
// (called via workflow.RunNode) transitions the orchestrator to
// NodeWaiting, so Workflow.Resume matches by InterruptID and the
// orchestrator re-enters to produce the final output.
func TestWorkflowAgent_RunThenResume_DynamicNodeOrchestrator(t *testing.T) {
	const interruptID = "ask_name_dyn"

	asker := newHitlNode("ask_name", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		if resp, ok := ctx.ResumedInput(interruptID); ok {
			ev := session.NewEventWithContext(ctx, ctx.InvocationID())
			ev.Output = resp
			yield(ev, nil)
			return
		}
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID: interruptID,
			Message:     "What's your name?",
		}), nil)
	})

	orchestrator := workflow.NewDynamicNode[string, string]("hitl_demo",
		func(nc workflow.NodeContext, _ string, _ func(*session.Event) error) (string, error) {
			out, err := workflow.RunNode[any](nc, asker, nil)
			if err != nil {
				// Pause: err is ErrNodeInterrupted (swallowed by dynamicNode.Run).
				// Resume: err is nil and out is the child's response.
				return "", err
			}
			name, _ := out.(string)
			if name == "" {
				name = "stranger"
			}
			return "Hello, " + name + "!", nil
		},
		workflow.NodeConfig{},
	)

	a := makeAgent(t, workflow.Chain(workflow.Start, orchestrator))
	sess := newFakeSession()

	// Turn 1: fresh Run; orchestrator schedules asker; asker pauses.
	turn1 := runFreshTurn(t, sess, a, "start")
	if got := findRequest(turn1); got != interruptID {
		t.Fatalf("turn 1 RequestedInput = %q, want %q", got, interruptID)
	}

	// Turn 2: resume with the reply.
	turn2 := drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage(interruptID, "Wolo"))), nil)
	if got := findRequest(turn2); got != "" {
		t.Errorf("turn 2 unexpectedly emitted a RequestedInput: %q", got)
	}

	var got any
	for _, ev := range turn2 {
		if ev.Output != nil && ev.NodeInfo != nil && strings.Contains(ev.NodeInfo.Path, "hitl_demo") {
			got = ev.Output
		}
	}
	if want := "Hello, Wolo!"; got != want {
		t.Errorf("orchestrator output = %v, want %q", got, want)
	}
}

// =============================================================================
// Test fixtures and helpers
// =============================================================================

// fakeSession is a minimal session.Session that records appended
// events, as the real session services do. HITL resume reconstructs
// paused state from this event history (Workflow.ReconstructRunState),
// so the test must append every yielded event — and the inbound user
// FunctionResponse on a resume turn — into the session.
//
// drainAgent (this file) appends every event the agent yields, and
// appendUserMessage records the inbound resume message, together
// simulating what the runner does in production.
type fakeSession struct {
	session.Session
	state  *fakeSessionState
	mu     sync.Mutex
	events []*session.Event
}

func newFakeSession() *fakeSession {
	return &fakeSession{state: &fakeSessionState{m: map[string]any{}}}
}

func (s *fakeSession) ID() string           { return "test-session-id" }
func (s *fakeSession) State() session.State { return s.state }

func (s *fakeSession) Events() session.Events {
	s.mu.Lock()
	defer s.mu.Unlock()
	return fakeEvents(append([]*session.Event(nil), s.events...))
}

// appendEvent records an event in history (the test analogue of
// session.Service.AppendEvent) and applies any StateDelta.
func (s *fakeSession) appendEvent(ev *session.Event) {
	if ev == nil {
		return
	}
	s.mu.Lock()
	s.events = append(s.events, ev)
	s.mu.Unlock()
	if len(ev.Actions.StateDelta) > 0 {
		s.state.mu.Lock()
		for k, v := range ev.Actions.StateDelta {
			s.state.m[k] = v
		}
		s.state.mu.Unlock()
	}
}

// appendUserMessage records an inbound user message as a "user"
// event so a resume turn's FunctionResponse is visible to
// ReconstructRunState, mirroring the runner appending the user turn.
func (s *fakeSession) appendUserMessage(msg *genai.Content) {
	if msg == nil {
		return
	}
	ev := session.NewEventWithContext(context.Background(), "test-invocation-id")
	ev.Author = "user"
	ev.LLMResponse = model.LLMResponse{Content: msg}
	s.appendEvent(ev)
}

// fakeEvents is a session.Events over a fixed slice.
type fakeEvents []*session.Event

func (e fakeEvents) Len() int                { return len(e) }
func (e fakeEvents) At(i int) *session.Event { return e[i] }
func (e fakeEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, ev := range e {
			if !yield(ev) {
				return
			}
		}
	}
}

// fakeSessionState exposes session.State semantics with one subtle
// constraint compared to the real services: callers that bypass
// the runner cannot mutate state via Set; they must construct an
// event with Actions.StateDelta and route it through
// fakeSession.applyStateDelta instead. Get reflects the
// AppendEvent-applied view.
type fakeSessionState struct {
	mu sync.Mutex
	m  map[string]any
}

func (s *fakeSessionState) Get(key string) (any, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, ok := s.m[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return v, nil
}

// Set is a no-op writer in this mock to surface accidental direct
// modification. Production code must route state changes through
// Event.Actions.StateDelta. Tests that need to pre-seed state can
// write directly to the underlying map via the fakeSession
// constructor.
func (s *fakeSessionState) Set(key string, value any) error {
	// Intentionally not persisted: real session services do not
	// propagate direct Set from inside an invocation either.
	// Returning nil keeps the call non-fatal so production code
	// that defensively writes through State.Set still compiles
	// and runs.
	return nil
}

func (s *fakeSessionState) All() iter.Seq2[string, any] {
	s.mu.Lock()
	defer s.mu.Unlock()
	snapshot := make(map[string]any, len(s.m))
	for k, v := range s.m {
		snapshot[k] = v
	}
	return func(yield func(string, any) bool) {
		for k, v := range snapshot {
			if !yield(k, v) {
				return
			}
		}
	}
}

// hitlNode is a custom Node used by the HITL resume tests. The
// Run callback is supplied per test so each scenario can shape
// its own emission pattern.
type hitlNode struct {
	workflow.BaseNode
	run func(ctx agent.Context, input any, yield func(*session.Event, error) bool)
}

func newHitlNode(name string, run func(ctx agent.Context, input any, yield func(*session.Event, error) bool)) *hitlNode {
	return &hitlNode{
		BaseNode: workflow.NewBaseNode(name, "", workflow.NodeConfig{}),
		run:      run,
	}
}

func (n *hitlNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.run(ctx, input, yield)
	}
}

// makeAgent builds a workflowagent with the given edges and the
// canonical "test_workflow" name (the name is what
// session.State persistence is keyed by).
func makeAgent(t *testing.T, edges []workflow.Edge) agent.Agent {
	t.Helper()
	a, err := New(Config{Name: "test_workflow", Edges: edges})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	return a
}

// newMockCtx returns an InvocationContext suitable for driving the
// workflow agent. The same session is reused across calls so
// pause/resume round-trips through fakeSessionState as they would
// in production.
func newMockCtx(sess session.Session, agt agent.Agent, msg *genai.Content) *MockInvocationContext {
	// Append the inbound user turn to history first, as the runner
	// does in production, so a resume turn's FunctionResponse is
	// visible to ReconstructRunState.
	if fs, ok := sess.(*fakeSession); ok {
		fs.appendUserMessage(msg)
	}
	return &MockInvocationContext{
		Context:     context.TODO(),
		sess:        sess,
		userContent: msg,
		myAgent:     agt,
	}
}

// drainAgent consumes the agent's iter.Seq2, collecting events and
// appending each to sess. The append step is the test analogue of
// the runner's AppendEvent: it builds the session history that the
// next turn's ReconstructRunState reads. Fails the test if the
// iterator yields a non-nil error the test did not opt into.
func drainAgent(t *testing.T, sess *fakeSession, seq iter.Seq2[*session.Event, error], wantErr error) []*session.Event {
	t.Helper()
	var got []*session.Event
	var sawErr error
	for ev, err := range seq {
		if err != nil {
			if sawErr == nil {
				sawErr = err
			}
			continue
		}
		got = append(got, ev)
		sess.appendEvent(ev)
	}
	switch {
	case wantErr == nil && sawErr != nil:
		t.Fatalf("unexpected error from agent: %v", sawErr)
	case wantErr != nil && sawErr == nil:
		t.Fatalf("expected error %v, got none", wantErr)
	case wantErr != nil && !errors.Is(sawErr, wantErr):
		t.Fatalf("expected error %v, got %v", wantErr, sawErr)
	}
	return got
}

// findRequest scans events for the first one carrying a
// RequestedInput and returns the InterruptID it carried, or "" if
// none was found.
func findRequest(events []*session.Event) string {
	for _, ev := range events {
		if ev != nil && ev.RequestedInput != nil {
			return ev.RequestedInput.InterruptID
		}
	}
	return ""
}

// resumeMessage builds a user message carrying a FunctionResponse
// that targets a previously-emitted RequestInput.
func resumeMessage(interruptID string, payload any) *genai.Content {
	return &genai.Content{
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:   interruptID,
				Name: workflow.WorkflowInputFunctionCallName,
				Response: map[string]any{
					"payload": payload,
				},
			},
		}},
	}
}

// newAskerNode returns a hitlNode whose Run yields a single
// RequestInput event carrying the given InterruptID, message, and
// optional schema, then exits. This is the canonical "asker"
// pattern: a node that pauses the workflow waiting for human input.
func newAskerNode(interruptID, message string, schema *jsonschema.Schema) *hitlNode {
	return newHitlNode("asker", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID:    interruptID,
			Message:        message,
			ResponseSchema: schema,
		}), nil)
	})
}

// runFreshTurn drives the agent through a single turn whose
// inbound user content is plain text (no FunctionResponse). Used
// to seed the canonical "first turn" of pause/resume tests where
// the actual text payload does not matter.
func runFreshTurn(t *testing.T, sess *fakeSession, a agent.Agent, text string) []*session.Event {
	t.Helper()
	return drainAgent(t, sess, a.Run(newMockCtx(sess, a, &genai.Content{
		Parts: []*genai.Part{{Text: text}},
	})), nil)
}

// newStringHandlerNode returns a FunctionNode that stores its
// string input into dst and returns "handled:<input>".
func newStringHandlerNode(name string, dst *atomic.Value) workflow.Node {
	return workflow.NewFunctionNode(
		name,
		func(_ agent.Context, input string) (string, error) {
			dst.Store(input)
			return "handled:" + input, nil
		},
		workflow.NodeConfig{},
	)
}

// newMapHandlerNode returns a FunctionNode that stores its
// map[string]any input into dst and returns nil.
func newMapHandlerNode(name string, dst *atomic.Value) workflow.Node {
	return workflow.NewFunctionNode(
		name,
		func(_ agent.Context, input map[string]any) (any, error) {
			dst.Store(input)
			return nil, nil
		},
		workflow.NodeConfig{},
	)
}

// newCountingHandlerNode returns a FunctionNode that increments counter
// each time it runs. Input is typed any so the helper accepts
// payloads of any shape without coercion.
func newCountingHandlerNode(name string, counter *atomic.Int32) workflow.Node {
	return workflow.NewFunctionNode(
		name,
		func(_ agent.Context, _ any) (any, error) {
			counter.Add(1)
			return nil, nil
		},
		workflow.NodeConfig{},
	)
}

// newFlagHandlerNode returns a FunctionNode that sets flag to true
// each time it runs. Input is typed any so the helper accepts
// payloads of any shape without coercion.
func newFlagHandlerNode(name string, flag *atomic.Bool) workflow.Node {
	return workflow.NewFunctionNode(
		name,
		func(_ agent.Context, _ any) (any, error) {
			flag.Store(true)
			return nil, nil
		},
		workflow.NodeConfig{},
	)
}

// approvalSchema returns the canonical schema for an "approval"
// payload: an object with a single required boolean field named
// "approved". Shared across the SchemaValidation tests.
func approvalSchema() *jsonschema.Schema {
	return &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"approved": {Type: "boolean"},
		},
		Required: []string{"approved"},
	}
}
