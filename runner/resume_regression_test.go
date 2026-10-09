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

package runner_test

import (
	"context"
	"errors"
	"iter"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// countingAsker pauses on RequestInput and, when re-entered with the
// reply, emits it as output. failFirstResume makes its first re-entry
// return an error.
type countingAsker struct {
	workflow.BaseNode
	id              string
	runs            atomic.Int32
	failFirstResume bool
	failed          atomic.Bool
}

func newCountingAsker(name, id string) *countingAsker {
	yes := true
	return &countingAsker{BaseNode: workflow.NewBaseNode(name, "", workflow.NodeConfig{RerunOnResume: &yes}), id: id}
}

func (n *countingAsker) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.runs.Add(1)
		if resp, ok := ctx.ResumedInput(n.id); ok {
			if n.failFirstResume && !n.failed.Swap(true) {
				yield(nil, errTransient)
				return
			}
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = resp
			yield(ev, nil)
			return
		}
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{InterruptID: n.id, Message: n.id}), nil)
	}
}

type transientErr struct{}

func (transientErr) Error() string { return "transient failure" }

var errTransient = transientErr{}

func runReturningErr(t *testing.T, r *runner.Runner, msg *genai.Content) ([]*session.Event, error) {
	t.Helper()
	var evs []*session.Event
	var first error
	for ev, err := range r.Run(t.Context(), nodeTestUser, nodeTestSession, msg, agent.RunConfig{}) {
		if err != nil {
			if first == nil {
				first = err
			}
			continue
		}
		evs = append(evs, ev)
	}
	return evs, first
}

// Two approval steps in sequence. Answering the second must not re-run the first.
func TestResumeRegression_ChainedReentryAskers(t *testing.T) {
	a, b := newCountingAsker("approveA", "a"), newCountingAsker("approveB", "b")
	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, a, b))

	t1, _ := runReturningErr(t, r, userText("start"))
	idA, nameA := findLongRunningInterrupt(t1)
	t2, _ := runReturningErr(t, r, resumeContent(idA, nameA, "ok-a"))
	idB, nameB := findLongRunningInterrupt(t2)
	if _, err := runReturningErr(t, r, resumeContent(idB, nameB, "ok-b")); err != nil {
		t.Fatalf("answering b: %v", err)
	}
	if a.runs.Load() != 2 || b.runs.Load() != 2 {
		t.Errorf("runs: approveA=%d approveB=%d, want 2 and 2", a.runs.Load(), b.runs.Load())
	}
}

// A resume that fails must stay retryable with the same reply.
func TestResumeRegression_RetryAfterFailedResume(t *testing.T) {
	a := newCountingAsker("asker", "q")
	a.failFirstResume = true
	var sinkRuns atomic.Int32
	sink := workflow.NewFunctionNode("sink", func(_ agent.Context, in any) (any, error) {
		sinkRuns.Add(1)
		return in, nil
	}, workflow.NodeConfig{})
	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, a, sink))

	t1, _ := runReturningErr(t, r, userText("start"))
	id, name := findLongRunningInterrupt(t1)
	if _, err := runReturningErr(t, r, resumeContent(id, name, "ok")); err == nil {
		t.Fatal("first resume: want the injected failure")
	}
	if _, err := runReturningErr(t, r, resumeContent(id, name, "ok")); err != nil {
		t.Fatalf("retry with the same reply: %v", err)
	}
	if sinkRuns.Load() != 1 {
		t.Errorf("sink runs = %d, want 1", sinkRuns.Load())
	}
}

// An AgentNode wrapping a non-LlmAgent (here a sequential agent) whose
// inner LlmAgent asks for tool confirmation. Approving must run the tool.
func TestResumeRegression_ConfirmationInsideSequentialAgentNode(t *testing.T) {
	var toolRuns atomic.Int32
	confirmTool, err := functiontool.New(functiontool.Config{
		Name: "confirm_action", Description: "x", RequireConfirmation: true,
	}, func(agent.Context, struct{}) (map[string]string, error) {
		toolRuns.Add(1)
		return map[string]string{"result": "executed"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	gated, err := llmagent.New(llmagent.Config{Name: "gated", Model: &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromFunctionCall("confirm_action", map[string]any{}, "model"),
		genai.NewContentFromText("finished", "model"),
	}}, Tools: []tool.Tool{confirmTool}})
	if err != nil {
		t.Fatal(err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{gated}}})
	if err != nil {
		t.Fatal(err)
	}
	// Wrapping a composite agent is an explicit opt-in after the default was
	// narrowed to LlmAgent and remote A2A agents; this test pins the nested
	// workflow behavior when the caller asks for re-entry.
	rerun := true
	node, err := workflow.NewAgentNode(seq, workflow.NodeConfig{RerunOnResume: &rerun})
	if err != nil {
		t.Fatal(err)
	}
	wf, err := workflowagent.New(workflowagent.Config{Name: workflowAgentName, Edges: workflow.Chain(workflow.Start, node)})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	r := newNodeTestRunner(t, wf, svc)

	t1, _ := runReturningErr(t, r, userText("start"))
	id, name := findLongRunningInterrupt(t1)
	approve := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: id, Name: name, Response: map[string]any{"confirmed": true},
	}}}}
	if _, err := runReturningErr(t, r, approve); err != nil {
		t.Fatalf("approve: %v", err)
	}
	if toolRuns.Load() != 1 {
		t.Errorf("confirmed tool runs = %d, want 1", toolRuns.Load())
	}
}

// TestResumeRegression_ResumePromptIncludesHistory verifies that the model
// call after a resumed confirmation still sees the node's own input and
// pending function call, rather than only the tool response.
func TestResumeRegression_ResumePromptIncludesHistory(t *testing.T) {
	m := &resumeRecordingModel{}
	f := newConfirmationGraphFixture(t, m)

	t1, _ := runReturningErr(t, f.runner, userText("start"))
	id, _ := findLongRunningInterrupt(t1)
	if id == "" {
		t.Fatal("turn 1 produced no interrupt")
	}
	if _, err := runReturningErr(t, f.runner, confirmationReply(id, true)); err != nil {
		t.Fatalf("approve: %v", err)
	}
	if len(m.contents) < 2 {
		t.Fatalf("model calls = %d, want at least 2", len(m.contents))
	}

	var sawInput, sawCall, sawResponse bool
	for _, content := range m.contents[1] {
		for _, part := range content.Parts {
			if part.Text == "prepared" {
				sawInput = true
			}
			if fc := part.FunctionCall; fc != nil && fc.Name == "confirm_action" {
				sawCall = true
			}
			if part.FunctionResponse != nil {
				sawResponse = true
			}
		}
	}
	if !sawInput || !sawCall || !sawResponse {
		for i, content := range m.contents[1] {
			t.Logf("content[%d] role=%q", i, content.Role)
			for j, part := range content.Parts {
				t.Logf("  part[%d]=%+v fc=%+v fr=%+v", j, part, part.FunctionCall, part.FunctionResponse)
			}
		}
		t.Errorf("resume prompt missing required context: input=%v call=%v response=%v", sawInput, sawCall, sawResponse)
	}
}

// TestResumeRegression_ConsumedHandoffReplyRejected verifies that a different
// payload for an already-consumed handoff interrupt is still reported as a
// stale resume rather than silently accepted.
func TestResumeRegression_ConsumedHandoffReplyRejected(t *testing.T) {
	asker := newHitlAsker("asker", "input-1", false)
	sink := workflow.NewFunctionNode("sink", func(agent.Context, any) (any, error) {
		return "done", nil
	}, workflow.NodeConfig{})
	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, asker, sink))

	t1, _ := runReturningErr(t, r, userText("start"))
	id, name := findLongRunningInterrupt(t1)
	if id == "" {
		t.Fatal("first turn produced no interrupt")
	}
	if _, err := runReturningErr(t, r, resumeContent(id, name, "yes")); err != nil {
		t.Fatalf("first resume: %v", err)
	}
	_, err := runReturningErr(t, r, resumeContent(id, name, "no"))
	if !errors.Is(err, workflow.ErrNothingToResume) {
		t.Fatalf("second resume error = %v, want %v", err, workflow.ErrNothingToResume)
	}
}

// TestResumeRegression_RootLlmAgentDuplicateReplyStartsFreshRun pins the
// pre-existing root-agent behavior: the graph-specific resume bookkeeping must
// not turn a duplicate reply outside a graph into ErrNothingToResume.
func TestResumeRegression_RootLlmAgentDuplicateReplyStartsFreshRun(t *testing.T) {
	m := &resumeRecordingModel{}
	confirmTool, err := functiontool.New(functiontool.Config{
		Name:                "confirm_action",
		Description:         "performs an action after confirmation",
		RequireConfirmation: true,
	}, func(agent.Context, struct{}) (map[string]string, error) {
		return map[string]string{"result": "executed"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:  "root",
		Model: m,
		Tools: []tool.Tool{confirmTool},
	})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	r := newNodeTestRunner(t, root, svc)

	t1, _ := runReturningErr(t, r, userText("start"))
	id, _ := findLongRunningInterrupt(t1)
	if id == "" {
		t.Fatal("first turn produced no interrupt")
	}
	reply := confirmationReply(id, true)
	if _, err := runReturningErr(t, r, reply); err != nil {
		t.Fatalf("first resume: %v", err)
	}
	callsAfterResume := len(m.contents)
	if _, err := runReturningErr(t, r, reply); err != nil {
		t.Fatalf("duplicate reply: %v", err)
	}
	if got := len(m.contents); got <= callsAfterResume {
		t.Errorf("model calls after duplicate = %d, want more than %d", got, callsAfterResume)
	}
}

// progressThenFailNode emits a non-terminal progress event and then fails on
// its first resume. Its next attempt succeeds, proving the reply was not
// consumed just because an event reached history.
type progressThenFailNode struct {
	workflow.BaseNode
	failed atomic.Bool
}

func newProgressThenFailNode() *progressThenFailNode {
	yes := true
	return &progressThenFailNode{
		BaseNode: workflow.NewBaseNode("asker", "", workflow.NodeConfig{RerunOnResume: &yes}),
	}
}

func (n *progressThenFailNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if resp, ok := ctx.ResumedInput("q"); ok {
			if !n.failed.Swap(true) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Content = genai.NewContentFromText("working", genai.RoleModel)
				if yield(ev, nil) {
					yield(nil, errors.New("transient failure"))
				}
				return
			}
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = resp
			yield(ev, nil)
			return
		}
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "q", Message: "q"}), nil)
	}
}

// TestResumeRegression_RetryAfterProgressThenFailure verifies that a resume
// which emitted progress before failing remains retryable with the same reply.
func TestResumeRegression_RetryAfterProgressThenFailure(t *testing.T) {
	asker := newProgressThenFailNode()
	var sinkRuns atomic.Int32
	sink := workflow.NewFunctionNode("sink", func(_ agent.Context, in any) (any, error) {
		sinkRuns.Add(1)
		return in, nil
	}, workflow.NodeConfig{})
	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, asker, sink))

	turn1, _ := runReturningErr(t, r, userText("start"))
	id, name := findLongRunningInterrupt(turn1)
	if _, err := runReturningErr(t, r, resumeContent(id, name, "ok")); err == nil {
		t.Fatal("first resume: want the injected failure")
	}
	if _, err := runReturningErr(t, r, resumeContent(id, name, "ok")); err != nil {
		t.Fatalf("retry with the same reply: %v", err)
	}
	if got := sinkRuns.Load(); got != 1 {
		t.Errorf("sink runs = %d, want 1", got)
	}
}

// parallelSuccessor emits an event on success; failFirst makes the first
// attempt wait for the sibling branch and then fail once.
type parallelSuccessor struct {
	workflow.BaseNode
	failFirst bool
	wait      chan struct{}
	done      chan struct{}
	runs      atomic.Int32
	failed    atomic.Bool
}

func (n *parallelSuccessor) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.runs.Add(1)
		if n.failFirst && !n.failed.Swap(true) {
			select {
			case <-n.wait:
			case <-time.After(2 * time.Second):
			}
			yield(nil, errors.New("transient failure"))
			return
		}
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Content = genai.NewContentFromText(n.Name()+" ran", genai.RoleModel)
		if yield(ev, nil) && n.done != nil {
			close(n.done)
			n.done = nil
		}
	}
}

// allLongRunningInterrupts returns every interrupt ID in the turn keyed to
// the function-call name that raised it.
func allLongRunningInterrupts(events []*session.Event) map[string]string {
	out := map[string]string{}
	for _, ev := range events {
		if ev == nil || len(ev.LongRunningToolIDs) == 0 || ev.Content == nil {
			continue
		}
		for _, id := range ev.LongRunningToolIDs {
			for _, part := range ev.Content.Parts {
				if part != nil && part.FunctionCall != nil && part.FunctionCall.ID == id {
					out[id] = part.FunctionCall.Name
				}
			}
		}
	}
	return out
}

// TestResumeRegression_ParallelHandoffRetry verifies that one branch's
// successor finishing does not consume the other branch's handoff reply.
func TestResumeRegression_ParallelHandoffRetry(t *testing.T) {
	saDone := make(chan struct{})
	a := newHitlAsker("askA", "qa", false)
	b := newHitlAsker("askB", "qb", false)
	sa := &parallelSuccessor{BaseNode: workflow.NewBaseNode("sa", "", workflow.NodeConfig{}), done: saDone}
	sb := &parallelSuccessor{BaseNode: workflow.NewBaseNode("sb", "", workflow.NodeConfig{}), failFirst: true, wait: saDone}
	r := newWorkflowRunner(t, append(workflow.Chain(workflow.Start, a, sa), workflow.Chain(workflow.Start, b, sb)...))

	turn1, _ := runReturningErr(t, r, userText("start"))
	ints := allLongRunningInterrupts(turn1)
	both := &genai.Content{Role: genai.RoleUser}
	for id, name := range ints {
		both.Parts = append(both.Parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{
			ID: id, Name: name, Response: map[string]any{"payload": "ans-" + id},
		}})
	}
	if _, err := runReturningErr(t, r, both); err == nil {
		t.Fatal("first resume: want branch B's injected failure")
	}
	if _, err := runReturningErr(t, r, resumeContent("qb", ints["qb"], "ans-qb")); err != nil {
		t.Fatalf("retry of branch B: %v", err)
	}
	if got := sb.runs.Load(); got != 2 {
		t.Errorf("branch B successor runs = %d, want 2", got)
	}
}

func newGatedAgentNode(t *testing.T, name string, m model.LLM) *workflow.AgentNode {
	t.Helper()
	confirmTool, err := functiontool.New(functiontool.Config{
		Name: "confirm_action", Description: "x", RequireConfirmation: true,
	}, func(agent.Context, struct{}) (map[string]string, error) {
		return map[string]string{"result": "executed"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: name, Model: m, Tools: []tool.Tool{confirmTool}})
	if err != nil {
		t.Fatal(err)
	}
	n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	return n
}

// TestResumeRegression_UnrelatedReplyDoesNotRestartGatedNode verifies that a
// handoff reply is consumed when its successor reaches a terminal event, so a
// later unrelated branch reply cannot restart the gated agent.
func TestResumeRegression_UnrelatedReplyDoesNotRestartGatedNode(t *testing.T) {
	m := &resumeRecordingModel{}
	exec := newGatedAgentNode(t, "executor", m)
	approve := newHitlAsker("approve", "ia", false)
	other := newHitlAsker("other", "io", false)
	osink := workflow.NewFunctionNode("osink", func(_ agent.Context, in any) (any, error) { return in, nil }, workflow.NodeConfig{})
	r := newWorkflowRunner(t, append(workflow.Chain(workflow.Start, approve, exec), workflow.Chain(workflow.Start, other, osink)...))

	turn1, _ := runReturningErr(t, r, userText("start"))
	ints := allLongRunningInterrupts(turn1)
	if _, err := runReturningErr(t, r, resumeContent("ia", ints["ia"], "go")); err != nil {
		t.Fatalf("answering approve: %v", err)
	}
	before := len(m.contents)
	if _, err := runReturningErr(t, r, resumeContent("io", ints["io"], "other")); err != nil {
		t.Fatalf("answering unrelated asker: %v", err)
	}
	if got := len(m.contents); got != before {
		t.Errorf("executor model calls %d -> %d after an unrelated reply", before, got)
	}
}

// TestResumeRegression_SingleTurnNodeKeepsOwnContextOnResume verifies that a
// single-turn node's resume request contains its own node input and pending
// call, not an earlier unredacted upstream turn.
func TestResumeRegression_SingleTurnNodeKeepsOwnContextOnResume(t *testing.T) {
	writer, err := llmagent.New(llmagent.Config{
		Name:  "writer",
		Model: &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("WRITER-PRIVATE", genai.RoleModel)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	wn, err := workflow.NewAgentNode(writer, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	redact := workflow.NewFunctionNode("redact", func(agent.Context, any) (any, error) { return "REDACTED", nil }, workflow.NodeConfig{})
	m := &resumeRecordingModel{}
	gn := newGatedAgentNode(t, "gated", m)
	wf, err := workflowagent.New(workflowagent.Config{Name: workflowAgentName, Edges: workflow.Chain(workflow.Start, wn, redact, gn)})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	r := newNodeTestRunner(t, wf, svc)

	turn1, _ := runReturningErr(t, r, userText("start"))
	id, _ := findLongRunningInterrupt(turn1)
	if _, err := runReturningErr(t, r, confirmationReply(id, true)); err != nil {
		t.Fatalf("approve: %v", err)
	}
	if len(m.contents) < 2 {
		t.Fatalf("gated model calls = %d, want at least 2", len(m.contents))
	}
	var sawPrivate, sawInput bool
	for _, content := range m.contents[1] {
		for _, part := range content.Parts {
			if part == nil {
				continue
			}
			sawPrivate = sawPrivate || part.Text == "WRITER-PRIVATE" || part.Text == "[writer] said: WRITER-PRIVATE"
			sawInput = sawInput || part.Text == "REDACTED"
		}
	}
	if sawPrivate || !sawInput {
		t.Errorf("resume request: contains writer text = %v, contains own input = %v; want false, true", sawPrivate, sawInput)
	}
}

// TestResumeRegression_CustomAgentNodeKeepsHandoffDefault pins the
// compatibility rule for custom agents: without an explicit opt-in their
// successor still receives the resume payload, as on main.
func TestResumeRegression_CustomAgentNodeKeepsHandoffDefault(t *testing.T) {
	var customRuns atomic.Int32
	custom, err := agent.New(agent.Config{
		Name: "custom",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				customRuns.Add(1)
				yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "custom-q", Message: "q"}), nil)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	node, err := workflow.NewAgentNode(custom, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if rr := node.Config().RerunOnResume; rr != nil {
		t.Errorf("custom AgentNode RerunOnResume = %v, want nil (handoff default)", *rr)
	}
	var got any
	sink := workflow.NewFunctionNode("sink", func(_ agent.Context, in any) (any, error) {
		got = in
		return in, nil
	}, workflow.NodeConfig{})
	r := newWorkflowRunner(t, workflow.Chain(workflow.Start, node, sink))

	turn1, _ := runReturningErr(t, r, userText("start"))
	id, name := findLongRunningInterrupt(turn1)
	if _, err := runReturningErr(t, r, resumeContent(id, name, "reply")); err != nil {
		t.Fatalf("handoff resume: %v", err)
	}
	if customRuns.Load() != 1 {
		t.Errorf("custom agent runs = %d, want 1", customRuns.Load())
	}
	if got != "reply" {
		t.Errorf("successor input = %#v, want %q", got, "reply")
	}
}

type resumeRecordingModel struct {
	contents [][]*genai.Content
}

func (m *resumeRecordingModel) Name() string { return "resume-recording" }

func (m *resumeRecordingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	contents := append([]*genai.Content(nil), req.Contents...)
	m.contents = append(m.contents, contents)
	call := len(m.contents) - 1
	return func(yield func(*model.LLMResponse, error) bool) {
		var content *genai.Content
		if call == 0 {
			content = genai.NewContentFromFunctionCall("confirm_action", map[string]any{}, "model")
		} else {
			content = genai.NewContentFromText("resume complete", "model")
		}
		yield(&model.LLMResponse{Content: content}, nil)
	}
}
