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
	"fmt"
	"iter"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
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
			ev := session.NewEvent(ctx, ctx.InvocationID())
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
		func(nc agent.Context, _ string, _ func(*session.Event) error) (string, error) {
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

// TestWorkflowAgent_AgentNode_CredentialResume pins resume for interactive
// consent inside a workflow: an AgentNode whose agent pauses on a long-running
// tool request (adk_request_credential — NOT the workflow's own
// adk_request_input) must, on the follow-up turn, RESUME the paused node rather
// than restart the graph or hand off. Concretely:
//
//   - the consent FunctionResponse (a non-adk_request_input name) is detected
//     as a resume (name-agnostic detectResume);
//   - the paused node re-enters and its agent finishes (AgentNode defaults to
//     RerunOnResume, so Resume re-runs it instead of the handoff path);
//   - the upstream node does NOT re-run (a fresh Workflow.Run would replay it);
//   - the resumed agent is not re-fed its node input (it continues from
//     history, so a single_turn/task agent would not re-call the tool).
func TestWorkflowAgent_AgentNode_CredentialResume(t *testing.T) {
	const fcID = "cred-1"

	var upstreamRuns atomic.Int32
	upstream := workflow.NewFunctionNode("upstream",
		func(_ agent.Context, _ any) (string, error) {
			upstreamRuns.Add(1)
			return "upstream-out", nil
		}, workflow.NodeConfig{})

	var workerRuns atomic.Int32
	var resumeGotInput atomic.Bool
	// The fake isn't an LlmAgent, so opt into RerunOnResume explicitly (a
	// real LlmAgent node gets it by default).
	worker, err := workflow.NewAgentNode(
		newConsentAgent(t, "worker", fcID, &workerRuns, &resumeGotInput),
		workflow.NodeConfig{RerunOnResume: ptrTrue()},
	)
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}

	a := makeAgent(t, workflow.Chain(workflow.Start, upstream, worker))
	sess := newFakeSession()

	// Turn 1: fresh run. upstream runs, then worker's tool needs consent, so
	// the node pauses on the long-running request.
	turn1 := runFreshTurn(t, sess, a, "start")
	if !hasLongRunning(turn1, fcID) {
		t.Fatalf("turn 1 did not pause on the long-running consent request %q", fcID)
	}
	if got := upstreamRuns.Load(); got != 1 {
		t.Fatalf("upstream runs after turn 1 = %d, want 1", got)
	}
	if got := workerRuns.Load(); got != 1 {
		t.Fatalf("worker runs after turn 1 = %d, want 1", got)
	}

	// Turn 2: the user grants consent. The name is adk_request_credential, so
	// this also exercises the name-agnostic resume dispatch.
	turn2 := drainAgent(t, sess, a.Run(newMockCtx(sess, a, credentialResume(fcID))), nil)

	if hasAnyLongRunning(turn2) {
		t.Errorf("turn 2 paused again instead of completing: %v", turn2)
	}
	if !hasOutput(turn2, "done") {
		t.Errorf("turn 2 did not produce the worker's completion output %q", "done")
	}
	if got := upstreamRuns.Load(); got != 1 {
		t.Errorf("upstream runs after resume = %d, want 1 (upstream must NOT re-run)", got)
	}
	if got := workerRuns.Load(); got != 2 {
		t.Errorf("worker runs = %d, want 2 (paused once, resumed once)", got)
	}
	if resumeGotInput.Load() {
		t.Error("worker was re-fed its node input on resume; it must continue from history")
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
	ev := session.NewEvent(context.Background(), "test-invocation-id")
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

// drainAgentErr is drainAgent without the error assertion: it records events in
// the session and returns the first error instead of failing on it, so a test
// can check its side-effect assertions before the diagnostic.
func drainAgentErr(t *testing.T, sess *fakeSession, seq iter.Seq2[*session.Event, error]) ([]*session.Event, error) {
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
	return got, sawErr
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

// newConsentAgent builds a fake agent that, on its first activation, emits a
// long-running adk_request_credential request (a tool needing interactive
// consent) and pauses; once that request is answered in history it completes
// with output "done". It records its activation count and whether it was given
// node input on the resume activation.
func newConsentAgent(t *testing.T, name, fcID string, runs *atomic.Int32, gotInputOnResume *atomic.Bool) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				runs.Add(1)
				if answeredInHistory(ctx.Session(), fcID) {
					if ctx.UserContent() != nil {
						gotInputOnResume.Store(true)
					}
					done := session.NewEvent(ctx, ctx.InvocationID())
					done.Author = name
					done.Output = "done"
					done.LLMResponse = model.LLMResponse{Content: &genai.Content{
						Role:  genai.RoleModel,
						Parts: []*genai.Part{{Text: "done"}},
					}}
					yield(done, nil)
					return
				}
				req := session.NewEvent(ctx, ctx.InvocationID())
				req.Author = name
				req.LongRunningToolIDs = []string{fcID}
				req.LLMResponse = model.LLMResponse{Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
						ID:   fcID,
						Name: "adk_request_credential",
					}}},
				}}
				yield(req, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return a
}

// newTwoInterruptAgent pauses on TWO long-running requests at once, the shape
// that rehydrates through inferNodeState's partial-resume arm. Once the first
// is answered it performs a counted action and pauses on the second alone;
// once both are answered it completes. approved counts the actions, so a
// replayed answer that re-runs the node shows up as a second count.
func newTwoInterruptAgent(t *testing.T, name, idX, idY string, runs, approved *atomic.Int32) agent.Agent {
	t.Helper()
	pause := func(ctx agent.InvocationContext, ids ...string) *session.Event {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Author = name
		ev.LongRunningToolIDs = ids
		parts := make([]*genai.Part, 0, len(ids))
		for _, id := range ids {
			parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{
				ID: id, Name: "adk_request_credential",
			}})
		}
		ev.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}}
		return ev
	}
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				runs.Add(1)
				switch {
				case !answeredInHistory(ctx.Session(), idX):
					yield(pause(ctx, idX, idY), nil)
				case !answeredInHistory(ctx.Session(), idY):
					approved.Add(1)
					yield(pause(ctx, idY), nil)
				default:
					approved.Add(1)
					done := session.NewEvent(ctx, ctx.InvocationID())
					done.Author = name
					done.Output = "done"
					done.LLMResponse = model.LLMResponse{Content: &genai.Content{
						Role:  genai.RoleModel,
						Parts: []*genai.Part{{Text: "done"}},
					}}
					yield(done, nil)
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return a
}

// credentialResume builds a user message carrying a consent FunctionResponse.
// The name is adk_request_credential (not adk_request_input) to exercise the
// name-agnostic resume dispatch.
func credentialResume(fcID string) *genai.Content {
	return &genai.Content{
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       fcID,
				Name:     "adk_request_credential",
				Response: map[string]any{"status": "approved"},
			},
		}},
	}
}

// answeredInHistory reports whether a FunctionResponse with the given id is in
// session history.
func answeredInHistory(sess session.Session, id string) bool {
	events := sess.Events()
	if events == nil {
		return false
	}
	for i := 0; i < events.Len(); i++ {
		for _, fr := range utils.FunctionResponses(utils.Content(events.At(i))) {
			if fr != nil && fr.ID == id {
				return true
			}
		}
	}
	return false
}

// hasLongRunning reports whether any event carries id in LongRunningToolIDs.
func hasLongRunning(events []*session.Event, id string) bool {
	for _, ev := range events {
		for _, x := range ev.LongRunningToolIDs {
			if x == id {
				return true
			}
		}
	}
	return false
}

// hasAnyLongRunning reports whether any event carries a long-running interrupt.
func hasAnyLongRunning(events []*session.Event) bool {
	for _, ev := range events {
		if len(ev.LongRunningToolIDs) > 0 {
			return true
		}
	}
	return false
}

// hasOutput reports whether any event carries the given Output value.
func hasOutput(events []*session.Event, want any) bool {
	for _, ev := range events {
		if ev.Output == want {
			return true
		}
	}
	return false
}

// TestWorkflowAgent_UnmatchedFunctionResponseRunsFresh pins that dropping the
// function-name filter did not turn every function-response turn into a resume.
// A reply that answers no waiting interrupt must fall through to a fresh
// Workflow.Run, as it did before the ID-keyed matching landed — the runner
// deliberately allows a turn to carry both text and a function response
// (see runner.buildResumeResponses, which filters the same way).
func TestWorkflowAgent_UnmatchedFunctionResponseRunsFresh(t *testing.T) {
	var upstream atomic.Int32
	asker := newAskerNode("approval", "decide", nil)
	a := makeAgent(t, workflow.Chain(workflow.Start,
		newCountingHandlerNode("upstream", &upstream), asker))
	sess := newFakeSession()

	runFreshTurn(t, sess, a, "start")
	if got := upstream.Load(); got != 1 {
		t.Fatalf("upstream runs after turn 1 = %d, want 1", got)
	}

	// A function response that answers nothing, alongside real user text.
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "forget it, start over"},
		{FunctionResponse: &genai.FunctionResponse{
			ID: "unrelated-call", Name: "get_weather", Response: map[string]any{"temp": 20},
		}},
	}}
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, msg)), nil)

	if got := upstream.Load(); got != 2 {
		t.Errorf("upstream runs after the unmatched reply = %d, want 2 (a fresh run); "+
			"the turn was misrouted to Resume and the user's text was dropped", got)
	}
}

// TestWorkflowAgent_Resume_ReentryIsIdempotent is the re-entry twin of
// TestWorkflowAgent_Resume_Idempotent, which only covers handoff. Re-entry is
// now the default for LlmAgent nodes, so a replayed approval must not re-run
// the node — otherwise a double-submit re-executes whatever the human approved.
func TestWorkflowAgent_Resume_ReentryIsIdempotent(t *testing.T) {
	const fcID = "cred-1"
	var runs atomic.Int32
	var gotInput atomic.Bool
	worker, err := workflow.NewAgentNode(
		newConsentAgent(t, "worker", fcID, &runs, &gotInput),
		workflow.NodeConfig{RerunOnResume: ptrTrue()},
	)
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	a := makeAgent(t, workflow.Chain(workflow.Start, worker))
	sess := newFakeSession()

	runFreshTurn(t, sess, a, "start")
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, credentialResume(fcID))), nil)
	afterFirst := runs.Load()

	// Byte-identical replay of the same approval.
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, credentialResume(fcID))), workflow.ErrNothingToResume)

	if got := runs.Load(); got != afterFirst {
		t.Errorf("worker runs = %d after a replayed approval, want %d "+
			"(a duplicate resume must reschedule nothing)", got, afterFirst)
	}
}

// TestWorkflowAgent_SettledReplyDoesNotDiscardNewText covers the case the
// ID-keyed filter alone gets wrong. History never un-answers an interrupt, so a
// re-entry node's settled ID stays recognisable for the rest of the session; a
// client that echoes it back alongside the human's next instruction would
// otherwise route the whole turn to Resume, which schedules nothing and fails
// with ErrNothingToResume — losing the instruction. The bare replay above is
// still an error; this one has work to do.
func TestWorkflowAgent_SettledReplyDoesNotDiscardNewText(t *testing.T) {
	const fcID = "cred-2"
	var runs atomic.Int32
	var gotInput atomic.Bool
	worker, err := workflow.NewAgentNode(
		newConsentAgent(t, "worker", fcID, &runs, &gotInput),
		workflow.NodeConfig{RerunOnResume: ptrTrue()},
	)
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	a := makeAgent(t, workflow.Chain(workflow.Start, worker))
	sess := newFakeSession()

	runFreshTurn(t, sess, a, "start")
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, credentialResume(fcID))), nil)
	afterFirst := runs.Load()

	// The human's next instruction, with the settled approval echoed back.
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "now do something else"},
		{FunctionResponse: &genai.FunctionResponse{
			ID: fcID, Name: "adk_request_credential",
			Response: map[string]any{"status": "approved"},
		}},
	}}
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, msg)), nil)

	if got := runs.Load(); got == afterFirst {
		t.Errorf("worker runs = %d, want more than %d; the turn was routed to "+
			"Resume on a settled reply and the user's text was dropped", got, afterFirst)
	}
}

// TestWorkflowAgent_SettledReplyDoesNotDiscardOtherParts is the non-text half
// of the test above. carriesOtherContent must discount only a matched
// FunctionResponse and count every other part, including a part kind it has
// never heard of: an allow-list of known kinds reads an unknown one as "nothing
// else in this message", routes the turn to Resume, and drops the part when
// Resume fails with ErrNothingToResume. ExecutableCode stands in for whatever
// genai.Part grows next.
func TestWorkflowAgent_SettledReplyDoesNotDiscardOtherParts(t *testing.T) {
	tests := []struct {
		name string
		part *genai.Part
	}{
		{"inline data", &genai.Part{InlineData: &genai.Blob{MIMEType: "text/plain", Data: []byte("hi")}}},
		{"file data", &genai.Part{FileData: &genai.FileData{MIMEType: "text/plain", FileURI: "gs://b/o"}}},
		{"executable code", &genai.Part{ExecutableCode: &genai.ExecutableCode{Code: "print(1)", Language: genai.LanguagePython}}},
	}
	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fcID := fmt.Sprintf("cred-other-%d", i)
			var runs atomic.Int32
			var gotInput atomic.Bool
			worker, err := workflow.NewAgentNode(
				newConsentAgent(t, "worker", fcID, &runs, &gotInput),
				workflow.NodeConfig{RerunOnResume: ptrTrue()},
			)
			if err != nil {
				t.Fatalf("NewAgentNode: %v", err)
			}
			a := makeAgent(t, workflow.Chain(workflow.Start, worker))
			sess := newFakeSession()

			runFreshTurn(t, sess, a, "start")
			drainAgent(t, sess, a.Run(newMockCtx(sess, a, credentialResume(fcID))), nil)
			afterFirst := runs.Load()

			// The settled approval echoed back alongside the part under test.
			msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
				tc.part,
				{FunctionResponse: &genai.FunctionResponse{
					ID: fcID, Name: "adk_request_credential",
					Response: map[string]any{"status": "approved"},
				}},
			}}
			drainAgent(t, sess, a.Run(newMockCtx(sess, a, msg)), nil)

			if got := runs.Load(); got == afterFirst {
				t.Errorf("worker runs = %d, want more than %d; the turn was routed to "+
					"Resume on a settled reply and the %s part was dropped", got, afterFirst, tc.name)
			}
		})
	}
}

// TestWorkflowAgent_Resume_ReentryIsIdempotentWithAnOpenInterrupt is the
// partial-resume twin of TestWorkflowAgent_Resume_ReentryIsIdempotent. A node
// pausing on two long-running IDs at once rehydrates through a different arm of
// inferNodeState, and a node still holding an open interrupt is no less exposed
// to a duplicate answer than one that has none.
func TestWorkflowAgent_Resume_ReentryIsIdempotentWithAnOpenInterrupt(t *testing.T) {
	const idX, idY = "cred-3", "confirm-3"
	var runs, approved atomic.Int32
	worker, err := workflow.NewAgentNode(
		newTwoInterruptAgent(t, "worker", idX, idY, &runs, &approved),
		workflow.NodeConfig{RerunOnResume: ptrTrue()},
	)
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	a := makeAgent(t, workflow.Chain(workflow.Start, worker))
	sess := newFakeSession()

	runFreshTurn(t, sess, a, "start")
	// Answer the first of the two; the node re-enters, acts on it, and pauses
	// again on the second.
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, credentialResume(idX))), nil)
	afterFirst := approved.Load()
	if afterFirst != 1 {
		t.Fatalf("approved actions after the first answer = %d, want 1", afterFirst)
	}

	// Replay that same answer. The second interrupt is still open, so the node
	// is not settled — but this answer is, and re-running on it would redo
	// what the human already approved.
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, credentialResume(idX))), workflow.ErrNothingToResume)
	if got := approved.Load(); got != afterFirst {
		t.Errorf("approved actions after a replayed answer = %d, want %d", got, afterFirst)
	}

	// The still-open interrupt must remain answerable.
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, credentialResume(idY))), nil)
	if got := approved.Load(); got <= afterFirst {
		t.Errorf("approved actions after answering the open interrupt = %d, want more than %d "+
			"(the replay guard swallowed a genuine answer)", got, afterFirst)
	}
}

// TestWorkflowAgent_Handoff_PartialAnswerKeepsNodeWaiting covers a handoff node
// that raised several interrupts at once and has been answered on only some of
// them. Completing it there would drop the unanswered ones and hand its
// successors an output standing in for a decision nobody made — for a rejected
// confirmation, one gating the very work it rejected. adk-python's replay
// interceptor keeps such a node waiting and re-bubbles the unresolved IDs.
func TestWorkflowAgent_Handoff_PartialAnswerKeepsNodeWaiting(t *testing.T) {
	const idA, idB = "confirm-a", "confirm-b"
	var successorRuns atomic.Int32
	worker, err := workflow.NewAgentNode(
		newTwoConfirmAgent(t, "approve_payment", idA, idB),
		workflow.NodeConfig{}, // handoff: the engine default for a non-LlmAgent
	)
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	guarded := workflow.NewFunctionNode("execute_payment",
		func(_ agent.Context, _ any) (string, error) {
			successorRuns.Add(1)
			return "paid", nil
		}, workflow.NodeConfig{})
	a := makeAgent(t, workflow.Chain(workflow.Start, worker, guarded))
	sess := newFakeSession()
	runFreshTurn(t, sess, a, "pay 500")

	// Reject the first; leave the second unanswered.
	reject := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
		FunctionResponse: &genai.FunctionResponse{
			ID: idA, Name: "adk_request_confirmation",
			Response: map[string]any{"confirmed": false},
		},
	}}}
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, reject)), nil)

	if got := successorRuns.Load(); got != 0 {
		t.Errorf("successor ran %d time(s) while an interrupt was still unanswered, want 0", got)
	}
}

// TestWorkflowAgent_Handoff_AllAnswersCompleteTheNode is the control for the
// test above: once every interrupt has an answer the node completes and hands
// off as before.
func TestWorkflowAgent_Handoff_AllAnswersCompleteTheNode(t *testing.T) {
	const idA, idB = "confirm-c", "confirm-d"
	var successorRuns atomic.Int32
	worker, err := workflow.NewAgentNode(
		newTwoConfirmAgent(t, "approve_payment", idA, idB),
		workflow.NodeConfig{},
	)
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	guarded := workflow.NewFunctionNode("execute_payment",
		func(_ agent.Context, _ any) (string, error) {
			successorRuns.Add(1)
			return "paid", nil
		}, workflow.NodeConfig{})
	a := makeAgent(t, workflow.Chain(workflow.Start, worker, guarded))
	sess := newFakeSession()
	runFreshTurn(t, sess, a, "pay 500")

	both := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{FunctionResponse: &genai.FunctionResponse{
			ID: idA, Name: "adk_request_confirmation",
			Response: map[string]any{"confirmed": true},
		}},
		{FunctionResponse: &genai.FunctionResponse{
			ID: idB, Name: "adk_request_confirmation",
			Response: map[string]any{"confirmed": true},
		}},
	}}
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, both)), nil)

	if got := successorRuns.Load(); got != 1 {
		t.Errorf("successor ran %d time(s) once every interrupt was answered, want 1", got)
	}
}

// TestWorkflowAgent_Resume_FailedActivationStaysResumable covers an activation
// that persists an event and then fails — an agent whose tool call succeeds and
// whose next model call does not. Treating any emitted event as proof the node
// acted on its answers wedges the run for good, because history never
// un-answers an interrupt: every retry is skipped as a replay and the approved
// work never settles.
func TestWorkflowAgent_Resume_FailedActivationStaysResumable(t *testing.T) {
	const fcID = "cred-4"
	var runs atomic.Int32
	var failResume atomic.Bool
	worker, err := workflow.NewAgentNode(
		newFailingResumeAgent(t, "worker", fcID, &runs, &failResume),
		workflow.NodeConfig{RerunOnResume: ptrTrue()},
	)
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	a := makeAgent(t, workflow.Chain(workflow.Start, worker))
	sess := newFakeSession()
	runFreshTurn(t, sess, a, "start")

	// The approval arrives, the tool result is persisted, then the agent dies.
	failResume.Store(true)
	for ev, err := range a.Run(newMockCtx(sess, a, credentialResume(fcID))) {
		if err != nil {
			continue
		}
		sess.appendEvent(ev)
	}
	afterFailure := runs.Load()

	// Retrying the same approval has to re-run the node.
	failResume.Store(false)
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, credentialResume(fcID))), nil)
	if got := runs.Load(); got == afterFailure {
		t.Errorf("worker runs = %d after retrying a failed resume, want more than %d "+
			"(the run is wedged and the approved work can never settle)", got, afterFailure)
	}
}

// newTwoConfirmAgent raises TWO confirmation interrupts in one event, the shape
// a model produces when two tool calls each need approval.
func newTwoConfirmAgent(t *testing.T, name, idA, idB string) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = name
				ev.LongRunningToolIDs = []string{idA, idB}
				ev.LLMResponse = model.LLMResponse{Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{ID: idA, Name: "adk_request_confirmation"}},
						{FunctionCall: &genai.FunctionCall{ID: idB, Name: "adk_request_confirmation"}},
					},
				}}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return a
}

// newFailingResumeAgent pauses on fcID, and on the resume activation persists a
// tool result before optionally failing — an event that is neither an output
// nor a new interrupt, so it leaves the node mid-flight rather than settled.
func newFailingResumeAgent(t *testing.T, name, fcID string, runs *atomic.Int32, failResume *atomic.Bool) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				runs.Add(1)
				if !answeredInHistory(ctx.Session(), fcID) {
					req := session.NewEvent(ctx, ctx.InvocationID())
					req.Author = name
					req.LongRunningToolIDs = []string{fcID}
					req.LLMResponse = model.LLMResponse{Content: &genai.Content{
						Role: genai.RoleModel,
						Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
							ID: fcID, Name: "adk_request_credential",
						}}},
					}}
					yield(req, nil)
					return
				}
				toolResult := session.NewEvent(ctx, ctx.InvocationID())
				toolResult.Author = name
				toolResult.LLMResponse = model.LLMResponse{Content: &genai.Content{
					Role: genai.RoleUser,
					Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
						ID: "tool-call", Name: "charge_card",
						Response: map[string]any{"ok": true},
					}}},
				}}
				yield(toolResult, nil)
				if failResume.Load() {
					yield(nil, errors.New("model call failed"))
					return
				}
				done := session.NewEvent(ctx, ctx.InvocationID())
				done.Author = name
				done.Output = "charged"
				done.LLMResponse = model.LLMResponse{Content: &genai.Content{
					Role:  genai.RoleModel,
					Parts: []*genai.Part{{Text: "charged"}},
				}}
				yield(done, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return a
}

// TestWorkflowAgent_Handoff_PartialAnswerEchoDoesNotDiscardNewText is the
// partly-answered-handoff twin of
// TestWorkflowAgent_SettledReplyDoesNotDiscardNewText. A handoff node answered
// on one of two interrupts stays waiting, and its delivered answer stays in
// ResumedInputs. A client that echoes that settled answer alongside the
// human's next instruction must still get the instruction run: the answer
// cannot be acted on again, so routing the turn to Resume schedules nothing
// and fails it with ErrNothingToResume, losing the text.
func TestWorkflowAgent_Handoff_PartialAnswerEchoDoesNotDiscardNewText(t *testing.T) {
	const idA, idB = "confirm-echo-a", "confirm-echo-b"
	var upstreamRuns atomic.Int32
	upstream := workflow.NewFunctionNode("prepare", func(_ agent.Context, _ any) (string, error) {
		upstreamRuns.Add(1)
		return "prepared", nil
	}, workflow.NodeConfig{})
	worker, err := workflow.NewAgentNode(
		newTwoConfirmAgent(t, "approve_payment", idA, idB),
		workflow.NodeConfig{}, // handoff: the engine default for a non-LlmAgent
	)
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	a := makeAgent(t, workflow.Chain(workflow.Start, upstream, worker))
	sess := newFakeSession()
	runFreshTurn(t, sess, a, "pay 500")

	confirmA := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
		FunctionResponse: &genai.FunctionResponse{
			ID: idA, Name: "adk_request_confirmation",
			Response: map[string]any{"confirmed": true},
		},
	}}}
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, confirmA)), nil)
	afterFirst := upstreamRuns.Load()

	// The human's next instruction, with the settled first approval echoed.
	echo := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "cancel the rest"},
		{FunctionResponse: &genai.FunctionResponse{
			ID: idA, Name: "adk_request_confirmation",
			Response: map[string]any{"confirmed": true},
		}},
	}}
	turn := drainAgent(t, sess, a.Run(newMockCtx(sess, a, echo)), nil)

	// The turn ran rather than failing, and running means a fresh run: the
	// graph restarts and both requests are put to the human again.
	if got := upstreamRuns.Load(); got != afterFirst+1 {
		t.Errorf("upstream runs = %d, want %d; the echo turn did not run", got, afterFirst+1)
	}
	raised := map[string]bool{}
	for _, ev := range turn {
		for _, id := range ev.LongRunningToolIDs {
			raised[id] = true
		}
	}
	if !raised[idA] || !raised[idB] {
		t.Errorf("interrupts re-raised on the echo turn = %v, want both %q and %q", raised, idA, idB)
	}
}

// TestWorkflowAgent_Resume_FailedDelegatingActivationStaysResumable is the
// delegating twin of TestWorkflowAgent_Resume_FailedActivationStaysResumable.
// A re-entry orchestrator resumes, delegates to a child that completes with an
// output, and only then fails. scanHistory attributes events by static node —
// a dynamic child folds into its ancestor — so the child's output must not be
// read as the ORCHESTRATOR settling on its answer. If it is, the answer is
// marked consumed, every retry is skipped as a replay, and the run is wedged
// for good: history never un-answers an interrupt.
func TestWorkflowAgent_Resume_FailedDelegatingActivationStaysResumable(t *testing.T) {
	const interruptID = "ask_deleg"
	var failNext atomic.Bool
	var completions, approvedWork atomic.Int32

	asker := newHitlNode("ask_child", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		if resp, ok := ctx.ResumedInput(interruptID); ok {
			// Stands in for the side effect the human approved.
			approvedWork.Add(1)
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = resp
			yield(ev, nil)
			return
		}
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID: interruptID, Message: "name?",
		}), nil)
	})

	orchestrator := workflow.NewDynamicNode[string, string]("deleg_orch",
		func(nc agent.Context, _ string, _ func(*session.Event) error) (string, error) {
			out, err := workflow.RunNode[any](nc, asker, nil)
			if err != nil {
				return "", err
			}
			// The child has completed and persisted its output. Now the
			// orchestrator's own remaining work fails.
			if failNext.Load() {
				return "", errors.New("orchestrator step failed after the child completed")
			}
			completions.Add(1)
			name, _ := out.(string)
			return "Hello, " + name + "!", nil
		},
		workflow.NodeConfig{},
	)

	a := makeAgent(t, workflow.Chain(workflow.Start, orchestrator))
	sess := newFakeSession()
	runFreshTurn(t, sess, a, "start")

	// Turn 2: the answer arrives, the child completes, the orchestrator fails.
	failNext.Store(true)
	for ev, err := range a.Run(newMockCtx(sess, a, resumeMessage(interruptID, "Wolo"))) {
		if err != nil {
			continue // the deliberate failure
		}
		if ev != nil {
			sess.appendEvent(ev)
		}
	}
	if got := completions.Load(); got != 0 {
		t.Fatalf("orchestrator completed %d time(s) on the failing turn, want 0", got)
	}

	// Turn 3: the human retries with the same answer. The approved work must
	// still be able to settle.
	failNext.Store(false)
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage(interruptID, "Wolo"))), nil)
	if got := completions.Load(); got != 1 {
		t.Errorf("orchestrator completions after the retry = %d, want 1; the retry was "+
			"skipped as a replay because the child's output marked the answer consumed", got)
	}
	// Retrying the orchestrator must not redo what the human already
	// approved: the child completed on the failing turn, so the retry has to
	// fast-forward it from history rather than run it again.
	if got := approvedWork.Load(); got != 1 {
		t.Errorf("approved child work ran %d time(s), want 1; retrying the orchestrator "+
			"re-executed a child that had already completed", got)
	}
}

// TestWorkflowAgent_Handoff_BareDuplicateSubmitDoesNotRestartTheGraph covers a
// double-clicked approve button: the identical FunctionResponse posted a second
// time, on its own. The node has settled on it, so there is nothing left to
// resume and the right answer is ErrNothingToResume. Dropping a settled node
// from the actionable-ID map instead makes the reply look like it answers
// nothing in this run, and the turn starts a fresh Workflow.Run that replays
// every completed upstream node.
func TestWorkflowAgent_Handoff_BareDuplicateSubmitDoesNotRestartTheGraph(t *testing.T) {
	const fcID = "confirm-dup"
	var upstreamRuns, successorRuns atomic.Int32
	upstream := workflow.NewFunctionNode("prepare",
		func(_ agent.Context, _ any) (string, error) {
			upstreamRuns.Add(1)
			return "prepared", nil
		}, workflow.NodeConfig{})
	worker, err := workflow.NewAgentNode(
		newOneConfirmAgent(t, "approve_payment", fcID),
		workflow.NodeConfig{}, // handoff
	)
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	successor := workflow.NewFunctionNode("execute",
		func(_ agent.Context, _ any) (string, error) {
			successorRuns.Add(1)
			return "done", nil
		}, workflow.NodeConfig{})
	a := makeAgent(t, workflow.Chain(workflow.Start, upstream, worker, successor))
	sess := newFakeSession()
	runFreshTurn(t, sess, a, "pay")

	approve := func() *genai.Content {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID: fcID, Name: "adk_request_confirmation",
				Response: map[string]any{"confirmed": true},
			},
		}}}
	}
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, approve())), nil)
	if got := successorRuns.Load(); got != 1 {
		t.Fatalf("successor runs after the approval = %d, want 1", got)
	}
	afterFirst := upstreamRuns.Load()

	// The same approval again, with nothing else in the message. Assert the
	// side effects before the error, so a regression reports the re-run rather
	// than only the missing diagnostic.
	_, sawErr := drainAgentErr(t, sess, a.Run(newMockCtx(sess, a, approve())))

	if got := upstreamRuns.Load(); got != afterFirst {
		t.Errorf("upstream runs = %d, want %d; the duplicate submit restarted the graph", got, afterFirst)
	}
	if got := successorRuns.Load(); got != 1 {
		t.Errorf("successor runs = %d, want 1; the duplicate submit re-ran approved work", got)
	}
	if !errors.Is(sawErr, workflow.ErrNothingToResume) {
		t.Errorf("error = %v, want %v; a settled reply must be reported as a no-op, "+
			"not silently started as a fresh run", sawErr, workflow.ErrNothingToResume)
	}
}

// TestWorkflowAgent_Handoff_ReplayDoesNotReRunAParkedSuccessor covers the
// second way a duplicate answer escapes the idempotency guard. The asker
// raised two interrupts; one answer is a replay and the other is not, so the
// node is legitimately acted on — but Pass 2 must still not re-trigger a
// successor that already ran and is now parked on an interrupt of its own.
// Rehydration drops a waiting node from state.completed, so the completed-set
// guard cannot see it.
func TestWorkflowAgent_Handoff_ReplayDoesNotReRunAParkedSuccessor(t *testing.T) {
	const idA, idB, idS = "park-a", "park-b", "park-s"
	var successorRuns atomic.Int32
	worker, err := workflow.NewAgentNode(
		newTwoConfirmAgent(t, "approve_payment", idA, idB),
		workflow.NodeConfig{}, // handoff
	)
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	// The successor does its work and then asks for a confirmation of its own.
	guarded, err := workflow.NewAgentNode(
		newSideEffectThenConfirmAgent(t, "execute_payment", idS, &successorRuns),
		workflow.NodeConfig{},
	)
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	a := makeAgent(t, workflow.Chain(workflow.Start, worker, guarded))
	sess := newFakeSession()
	runFreshTurn(t, sess, a, "pay 500")

	confirm := func(id string) *genai.Content {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID: id, Name: "adk_request_confirmation",
				Response: map[string]any{"confirmed": true},
			},
		}}}
	}
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, confirm(idA))), nil)
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, confirm(idB))), nil)
	if got := successorRuns.Load(); got != 1 {
		t.Fatalf("successor runs after both answers = %d, want 1", got)
	}

	// Replay the first answer. The successor is parked on idS, not completed.
	_, sawErr := drainAgentErr(t, sess, a.Run(newMockCtx(sess, a, confirm(idA))))
	if got := successorRuns.Load(); got != 1 {
		t.Errorf("successor runs after a replayed answer = %d, want 1; a parked "+
			"successor was re-triggered and redid its side effect", got)
	}
	if !errors.Is(sawErr, workflow.ErrNothingToResume) {
		t.Errorf("error = %v, want %v", sawErr, workflow.ErrNothingToResume)
	}
}

// newOneConfirmAgent pauses once on a single confirmation request.
func newOneConfirmAgent(t *testing.T, name, fcID string) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = name
				ev.LongRunningToolIDs = []string{fcID}
				ev.LLMResponse = model.LLMResponse{Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
						ID: fcID, Name: "adk_request_confirmation",
					}}},
				}}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return a
}

// newSideEffectThenConfirmAgent records one side effect and then pauses on its
// own confirmation request, so the node ends the turn parked rather than
// completed.
func newSideEffectThenConfirmAgent(t *testing.T, name, fcID string, runs *atomic.Int32) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				runs.Add(1)
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = name
				ev.LongRunningToolIDs = []string{fcID}
				ev.LLMResponse = model.LLMResponse{Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
						ID: fcID, Name: "adk_request_confirmation",
					}}},
				}}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return a
}

// TestWorkflowAgent_PoisonedHistoryDoesNotFailAnUnrelatedTurn covers what
// rehydrating before filtering costs. A payload that fails its schema stays in
// history for good, so every later ReconstructRunState fails — including on a
// turn that answers nothing in this run. Such a turn has work of its own and
// must run rather than inherit an earlier turn's fault.
func TestWorkflowAgent_PoisonedHistoryDoesNotFailAnUnrelatedTurn(t *testing.T) {
	var handlerRuns atomic.Int32
	asker := newAskerNode("approval2", "decide", approvalSchema())
	handler := newCountingHandlerNode("handler", &handlerRuns)
	a := makeAgent(t, workflow.Chain(workflow.Start, asker, handler))
	sess := newFakeSession()

	runFreshTurn(t, sess, a, "x")
	// Poison the run: a payload that can never validate.
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("approval2", "not an object"))),
		workflow.ErrInvalidResumeResponse)

	// A later turn with the human's own instruction and a tool reply aimed
	// somewhere else entirely.
	unrelated := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "forget it, start over"},
		{FunctionResponse: &genai.FunctionResponse{
			ID: "some-other-tool-call", Name: "get_weather",
			Response: map[string]any{"result": "sunny"},
		}},
	}}
	drainAgentErrOnly := func() error {
		_, err := drainAgentErr(t, sess, a.Run(newMockCtx(sess, a, unrelated)))
		return err
	}
	if err := drainAgentErrOnly(); err != nil {
		t.Errorf("unrelated turn failed with %v; one bad answer in history must not "+
			"fail a turn that answers nothing in this run and carries the user's text", err)
	}
	// Running means a FRESH run, so the graph restarts from Start: the asker
	// re-asks and the handler, which never ran, still has not. Recorded here
	// because it is the cost of the fall-through, not a detail — a poisoned
	// answer makes every later content-bearing turn replay the graph.
	if got := handlerRuns.Load(); got != 0 {
		t.Errorf("handler runs = %d, want 0; the fall-through ran work past the "+
			"unanswered interrupt", got)
	}

	// The bare retry must still surface the diagnostic rather than running fresh.
	if _, err := drainAgentErr(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("approval2", "still not an object")))); !errors.Is(err, workflow.ErrInvalidResumeResponse) {
		t.Errorf("bare retry error = %v, want %v", err, workflow.ErrInvalidResumeResponse)
	}
}

// TestWorkflowAgent_ReRaisedInterruptIDStaysAnswerable covers a node that
// rejects a payload and asks again under the SAME interrupt ID —
// session.RequestInput takes a caller-chosen, stable InterruptID, and
// examples/workflow/hitl_rerun builds one per run, so this is a shape the API
// invites. Rehydration dedupes the raise, so without re-opening the ID the
// first answer keeps the interrupt resolved for good and the corrected answer
// can never reach the node.
func TestWorkflowAgent_ReRaisedInterruptIDStaysAnswerable(t *testing.T) {
	const fcID = "revalidate-1"
	var activations atomic.Int32
	inner, err := agent.New(agent.Config{
		Name: "revalidator",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				n := activations.Add(1)
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "revalidator"
				if n < 3 {
					// Reject and re-prompt under the same ID.
					ev.LongRunningToolIDs = []string{fcID}
					ev.LLMResponse = model.LLMResponse{Content: &genai.Content{
						Role: genai.RoleModel,
						Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
							ID: fcID, Name: "adk_request_confirmation",
						}}},
					}}
				} else {
					ev.Output = "settled"
				}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	node, err := workflow.NewAgentNode(inner, workflow.NodeConfig{RerunOnResume: ptrTrue()})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	a := makeAgent(t, workflow.Chain(workflow.Start, node))
	sess := newFakeSession()
	runFreshTurn(t, sess, a, "go")

	reply := func(v string) *genai.Content {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID: fcID, Name: "adk_request_confirmation",
				Response: map[string]any{"payload": v},
			},
		}}}
	}
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, reply("rejected-shape"))), nil)
	if got := activations.Load(); got != 2 {
		t.Fatalf("activations after the first answer = %d, want 2", got)
	}
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, reply("corrected"))), nil)
	if got := activations.Load(); got != 3 {
		t.Errorf("activations after the corrected answer = %d, want 3; the re-raised "+
			"interrupt stayed resolved by the answer the node had already rejected", got)
	}
}

// TestWorkflowAgent_Resume_FailedActivationRetryWithTextStillResumes is the
// non-bare twin of TestWorkflowAgent_Resume_FailedActivationStaysResumable. A
// human whose approved work failed mid-flight retries and says something while
// doing it. The answer is a replay in history but the node never acted on it,
// so the turn must still reach Resume and re-run only that node — routing it
// to a fresh Run restarts the graph and re-executes every completed upstream
// node.
func TestWorkflowAgent_Resume_FailedActivationRetryWithTextStillResumes(t *testing.T) {
	const fcID = "cred-retry-text"
	var runs, upstreamRuns atomic.Int32
	var failResume atomic.Bool
	upstream := workflow.NewFunctionNode("charge", func(_ agent.Context, _ any) (string, error) {
		upstreamRuns.Add(1)
		return "charged", nil
	}, workflow.NodeConfig{})
	worker, err := workflow.NewAgentNode(
		newFailingResumeAgent(t, "worker", fcID, &runs, &failResume),
		workflow.NodeConfig{RerunOnResume: ptrTrue()},
	)
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	a := makeAgent(t, workflow.Chain(workflow.Start, upstream, worker))
	sess := newFakeSession()
	runFreshTurn(t, sess, a, "go")
	afterFirst := upstreamRuns.Load()

	// Approve; the tool result is persisted and then the activation fails.
	failResume.Store(true)
	_, _ = drainAgentErr(t, sess, a.Run(newMockCtx(sess, a, credentialResume(fcID))))

	// Retry, with the human saying something too.
	failResume.Store(false)
	retry := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "please try again"},
		{FunctionResponse: &genai.FunctionResponse{
			ID: fcID, Name: "adk_request_credential",
			Response: map[string]any{"status": "approved"},
		}},
	}}
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, retry)), nil)

	if got := upstreamRuns.Load(); got != afterFirst {
		t.Errorf("upstream runs = %d, want %d; the retry started a fresh run and "+
			"replayed the graph instead of resuming the node that failed", got, afterFirst)
	}
}

// TestWorkflowAgent_UseAsOutputOrchestratorReplayIsANoOp covers a delegating
// orchestrator that never emits an event of its own: a WithUseAsOutput child
// carries the orchestrator's output, and dynamic_node suppresses the parent's
// terminal event in that case. Attribution by path alone therefore sees no
// settling event for the orchestrator at all, so its answer is never marked
// consumed and a duplicate submit re-runs the body — redoing whatever the
// human approved. The output attribution in NodeInfo.OutputFor is what says
// the orchestrator settled.
func TestWorkflowAgent_UseAsOutputOrchestratorReplayIsANoOp(t *testing.T) {
	const interruptID = "uao-1"
	var approvedWork atomic.Int32

	asker := newHitlNode("uao_asker", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		if resp, ok := ctx.ResumedInput(interruptID); ok {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = resp
			yield(ev, nil)
			return
		}
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID: interruptID, Message: "approve?",
		}), nil)
	})
	receipt := workflow.NewFunctionNode("uao_receipt", func(_ agent.Context, in any) (string, error) {
		s, _ := in.(string)
		return "receipt:" + s, nil
	}, workflow.NodeConfig{})

	orch := workflow.NewDynamicNode[string, any]("uao_orch",
		func(nc agent.Context, _ string, _ func(*session.Event) error) (any, error) {
			ans, err := workflow.RunNode[any](nc, asker, nil)
			if err != nil {
				return nil, err
			}
			// Stands in for the side effect the human approved.
			approvedWork.Add(1)
			if _, err := workflow.RunNode[any](nc, receipt, ans, workflow.WithUseAsOutput()); err != nil {
				return nil, err
			}
			return nil, nil
		}, workflow.NodeConfig{})

	a := makeAgent(t, workflow.Chain(workflow.Start, orch))
	sess := newFakeSession()
	runFreshTurn(t, sess, a, "start")

	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage(interruptID, "yes"))), nil)
	if got := approvedWork.Load(); got != 1 {
		t.Fatalf("approved work after the answer = %d, want 1", got)
	}

	// The same answer again, on its own.
	_, sawErr := drainAgentErr(t, sess, a.Run(newMockCtx(sess, a, resumeMessage(interruptID, "yes"))))
	if got := approvedWork.Load(); got != 1 {
		t.Errorf("approved work after a duplicate submit = %d, want 1; the orchestrator "+
			"re-ran because its own settling event was suppressed by the delegated output", got)
	}
	if !errors.Is(sawErr, workflow.ErrNothingToResume) {
		t.Errorf("error = %v, want %v", sawErr, workflow.ErrNothingToResume)
	}
}
