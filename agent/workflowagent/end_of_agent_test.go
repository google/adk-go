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

package workflowagent_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// endTimeout bounds every run in this file, so that a run that never returns
// fails the test instead of hanging it.
const endTimeout = 10 * time.Second

// endTurn is what one call of Runner.Run produced.
type endTurn struct {
	events []*session.Event
	errs   []error
}

// describe renders ev as "author:END" for an end-of-agent event, and
// otherwise as "author:" followed by "FC(name)" for each function call,
// "FR(name)" for each function response and the text of each text part.
func describe(ev *session.Event) string {
	if ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent {
		return ev.Author + ":END"
	}
	var b strings.Builder
	if ev.Content != nil {
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				fmt.Fprintf(&b, "FC(%s)", p.FunctionCall.Name)
			case p.FunctionResponse != nil:
				fmt.Fprintf(&b, "FR(%s)", p.FunctionResponse.Name)
			default:
				b.WriteString(p.Text)
			}
		}
	}
	return ev.Author + ":" + b.String()
}

func describeAll(evs []*session.Event) []string {
	var out []string
	for _, ev := range evs {
		out = append(out, describe(ev))
	}
	return out
}

// endEvents returns the end-of-agent events among evs.
func endEvents(evs []*session.Event) []*session.Event {
	var out []*session.Event
	for _, ev := range evs {
		if ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent {
			out = append(out, ev)
		}
	}
	return out
}

// newEndRunner returns a runner over root with an in-memory session service
// that creates sessions on demand.
func newEndRunner(t *testing.T, root agent.Agent, resumable bool) (*runner.Runner, session.Service) {
	t.Helper()
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             root,
		SessionService:    svc,
		AutoCreateSession: true,
		Resumable:         resumable,
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return r, svc
}

// runEndTurn runs one turn in session "s1" under ctx. stop, when non-nil,
// sees each event as it is yielded and returns true to stop reading there.
func runEndTurn(ctx context.Context, r *runner.Runner, msg *genai.Content, stop func(*session.Event) bool) endTurn {
	ctx, cancel := context.WithTimeout(ctx, endTimeout)
	defer cancel()
	var turn endTurn
	for ev, err := range r.Run(ctx, "user", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			turn.errs = append(turn.errs, err)
			continue
		}
		turn.events = append(turn.events, ev)
		if stop != nil && stop(ev) {
			break
		}
	}
	return turn
}

// storedEvents returns the events stored in session "s1".
func storedEvents(t *testing.T, svc session.Service) []*session.Event {
	t.Helper()
	got, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var out []*session.Event
	for ev := range got.Session.Events().All() {
		out = append(out, ev)
	}
	return out
}

func userText(text string) *genai.Content {
	return genai.NewContentFromText(text, genai.RoleUser)
}

func mustWorkflow(t *testing.T, cfg workflowagent.Config) agent.Agent {
	t.Helper()
	a, err := workflowagent.New(cfg)
	if err != nil {
		t.Fatalf("workflowagent.New(%q) error = %v", cfg.Name, err)
	}
	return a
}

// textNode returns an agent node over an agent named name that yields the
// text text, counting its runs in runs when runs is non-nil.
func textNode(t *testing.T, name, text string, runs *atomic.Int32) workflow.Node {
	t.Helper()
	a, err := agent.New(agent.Config{Name: name, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			if runs != nil {
				runs.Add(1)
			}
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = name
			ev.Content = genai.NewContentFromText(text, genai.RoleModel)
			yield(ev, nil)
		}
	}})
	if err != nil {
		t.Fatalf("agent.New(%q) error = %v", name, err)
	}
	n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(%q) error = %v", name, err)
	}
	return n
}

// funcNode is a workflow node whose run is supplied by the test.
type funcNode struct {
	workflow.BaseNode
	run func(ctx agent.Context, yield func(*session.Event, error) bool)
}

func newFuncNode(name string, cfg workflow.NodeConfig, run func(ctx agent.Context, yield func(*session.Event, error) bool)) *funcNode {
	return &funcNode{BaseNode: workflow.NewBaseNode(name, "", cfg), run: run}
}

func (n *funcNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) { n.run(ctx, yield) }
}

// askerNode returns a node named name that asks for input with interrupt ID
// interruptID and response schema schema. Answered, the workflow hands the
// answer to the node's successors.
func askerNode(name, interruptID string, schema *jsonschema.Schema) workflow.Node {
	return newFuncNode(name, workflow.NodeConfig{}, func(ctx agent.Context, yield func(*session.Event, error) bool) {
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID:    interruptID,
			Message:        "answer " + interruptID,
			ResponseSchema: schema,
		}), nil)
	})
}

// requestedIDs returns the interrupt IDs of the input requests among evs.
func requestedIDs(evs []*session.Event) []string {
	var out []string
	for _, ev := range evs {
		if ev.RequestedInput != nil {
			out = append(out, ev.RequestedInput.InterruptID)
		}
	}
	return out
}

// TestWorkflowAgent_EndOfAgentEvent_FreshRun checks that a resumable run of a
// graph workflow agent that finishes ends with one end-of-agent event of the
// agent, and that a run outside a resumable runner yields none.
//
// Fails if the agent writes no end event, writes it with another author,
// invocation, branch or content, or writes it when the run is not resumable
// or has no run config.
func TestWorkflowAgent_EndOfAgentEvent_FreshRun(t *testing.T) {
	newWF := func(t *testing.T) agent.Agent {
		return mustWorkflow(t, workflowagent.Config{
			Name:  "wf",
			Edges: workflow.Chain(workflow.Start, textNode(t, "d", "D", nil)),
		})
	}

	t.Run("resumable_true", func(t *testing.T) {
		r, svc := newEndRunner(t, newWF(t), true)

		turn := runEndTurn(t.Context(), r, userText("go"), nil)

		if len(turn.errs) != 0 {
			t.Fatalf("Run() errors = %v, want none", turn.errs)
		}
		if diff := cmp.Diff([]string{"d:D", "wf:END"}, describeAll(turn.events)); diff != "" {
			t.Fatalf("yielded events (-want +got):\n%s", diff)
		}
		end := turn.events[1]
		if end.InvocationID == "" || end.InvocationID != turn.events[0].InvocationID {
			t.Errorf("end event InvocationID = %q, want the run's invocation %q", end.InvocationID, turn.events[0].InvocationID)
		}
		if end.Branch != "" {
			t.Errorf("end event Branch = %q, want %q", end.Branch, "")
		}
		if end.Content != nil {
			t.Errorf("end event Content = %+v, want nil", end.Content)
		}
		stored := storedEvents(t, svc)
		if last := describe(stored[len(stored)-1]); last != "wf:END" {
			t.Errorf("last stored event = %q, want %q", last, "wf:END")
		}
	})

	t.Run("resumable_false", func(t *testing.T) {
		r, _ := newEndRunner(t, newWF(t), false)

		turn := runEndTurn(t.Context(), r, userText("go"), nil)

		if len(turn.errs) != 0 {
			t.Fatalf("Run() errors = %v, want none", turn.errs)
		}
		if diff := cmp.Diff([]string{"d:D"}, describeAll(turn.events)); diff != "" {
			t.Errorf("yielded events (-want +got):\n%s", diff)
		}
	})

	t.Run("direct_run_without_runconfig", func(t *testing.T) {
		wf := newWF(t)
		svc := session.InMemoryService()
		created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s1"})
		if err != nil {
			t.Fatalf("Create() error = %v", err)
		}
		ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
			Session:     created.Session,
			Agent:       wf,
			UserContent: userText("go"),
		})

		var got []string
		for ev, err := range wf.Run(ctx) {
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			got = append(got, describe(ev))
		}

		if diff := cmp.Diff([]string{"d:D"}, got); diff != "" {
			t.Errorf("yielded events (-want +got):\n%s", diff)
		}
	})
}

// TestWorkflowAgent_EndOfAgentEvent_PauseAndMultiNodeResume checks that a
// resumable graph workflow agent yields no end event while a node waits for
// input, including a node that has waited since an earlier turn, and exactly
// one when the resume that answers the last request finishes the workflow.
//
// Fails if a turn that raised an input request writes the end event, if a
// resume writes it while another node still waits from an earlier turn, or
// if the resume that finishes the workflow writes none.
func TestWorkflowAgent_EndOfAgentEvent_PauseAndMultiNodeResume(t *testing.T) {
	join := workflow.NewJoinNode("join")
	asker1 := askerNode("asker1", "ask-1", nil)
	asker2 := askerNode("asker2", "ask-2", nil)
	wf := mustWorkflow(t, workflowagent.Config{Name: "wf", Edges: []workflow.Edge{
		{From: workflow.Start, To: asker1},
		{From: workflow.Start, To: asker2},
		{From: asker1, To: join},
		{From: asker2, To: join},
	}})
	r, svc := newEndRunner(t, wf, true)

	turn1 := runEndTurn(t.Context(), r, userText("go"), nil)
	if len(turn1.errs) != 0 {
		t.Fatalf("turn 1 Run() errors = %v, want none", turn1.errs)
	}
	if diff := cmp.Diff([]string{"ask-1", "ask-2"}, requestedIDs(turn1.events), cmpopts.SortSlices(func(a, b string) bool { return a < b })); diff != "" {
		t.Errorf("turn 1 input requests (-want +got):\n%s", diff)
	}
	if n := len(endEvents(turn1.events)); n != 0 {
		t.Errorf("turn 1 yielded %d end events, want 0", n)
	}

	turn2 := runEndTurn(t.Context(), r, testutil.InputResponse("ask-1", "one"), nil)
	if len(turn2.errs) != 0 {
		t.Fatalf("turn 2 Run() errors = %v, want none", turn2.errs)
	}
	if diff := cmp.Diff([]string(nil), describeAll(turn2.events)); diff != "" {
		t.Errorf("turn 2 yielded events (-want +got):\n%s", diff)
	}
	if n := len(endEvents(storedEvents(t, svc))); n != 0 {
		t.Errorf("session holds %d end events after turn 2, want 0", n)
	}

	turn3 := runEndTurn(t.Context(), r, testutil.InputResponse("ask-2", "two"), nil)
	if len(turn3.errs) != 0 {
		t.Fatalf("turn 3 Run() errors = %v, want none", turn3.errs)
	}
	if diff := cmp.Diff([]string{"wf:", "wf:END"}, describeAll(turn3.events)); diff != "" {
		t.Errorf("turn 3 yielded events (-want +got):\n%s", diff)
	}
	if n := len(endEvents(storedEvents(t, svc))); n != 1 {
		t.Errorf("session holds %d end events after turn 3, want 1", n)
	}
}

// TestWorkflowAgent_EndOfAgentEvent_WaitForOutputDoesNotBlock checks that a
// node parked because a child it ran with WaitForOutput returned no output
// does not keep the workflow from writing its end event, as only a node that
// waits on an interrupt does.
//
// Fails if every waiting node, with or without an interrupt, withholds the
// end event.
func TestWorkflowAgent_EndOfAgentEvent_WaitForOutputDoesNotBlock(t *testing.T) {
	waitForOutput := true
	child := newFuncNode("child", workflow.NodeConfig{WaitForOutput: &waitForOutput}, func(agent.Context, func(*session.Event, error) bool) {})
	parent := workflow.NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
		return workflow.RunNode[any](ctx, child, nil)
	}, workflow.NodeConfig{})
	wf := mustWorkflow(t, workflowagent.Config{
		Name:  "wf",
		Edges: workflow.Chain(workflow.Start, askerNode("asker", "ask", nil), parent),
	})
	r, _ := newEndRunner(t, wf, true)

	turn1 := runEndTurn(t.Context(), r, userText("go"), nil)
	if len(turn1.errs) != 0 {
		t.Fatalf("turn 1 Run() errors = %v, want none", turn1.errs)
	}
	if n := len(endEvents(turn1.events)); n != 0 {
		t.Errorf("turn 1 yielded %d end events, want 0", n)
	}

	turn2 := runEndTurn(t.Context(), r, testutil.InputResponse("ask", "yes"), nil)

	if len(turn2.errs) != 0 {
		t.Fatalf("turn 2 Run() errors = %v, want none", turn2.errs)
	}
	// The child yields no event, so the parked parent leaves only the end.
	if diff := cmp.Diff([]string{"wf:END"}, describeAll(turn2.events)); diff != "" {
		t.Errorf("turn 2 yielded events (-want +got):\n%s", diff)
	}
}

// approvalSchema is the response schema of an approval: an object with a
// required boolean "approved".
func approvalSchema() *jsonschema.Schema {
	return &jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"approved": {Type: "boolean"}},
		Required:   []string{"approved"},
	}
}

// TestWorkflowAgent_EndOfAgentEvent_WithheldOnErrorCancelAndEarlyStop checks
// that a resumable graph workflow agent writes no end event when its run
// fails, is canceled or is not read to the end.
//
// Fails if an error in the workflow's stream, a failed resume, or a caller
// that stops reading still lets the agent write or record its end.
func TestWorkflowAgent_EndOfAgentEvent_WithheldOnErrorCancelAndEarlyStop(t *testing.T) {
	// failingRun runs the workflow over edges in one turn and returns what the
	// turn yielded and what the session holds.
	failingRun := func(t *testing.T, ctx context.Context, edges []workflow.Edge) (endTurn, []*session.Event) {
		t.Helper()
		r, svc := newEndRunner(t, mustWorkflow(t, workflowagent.Config{Name: "wf", Edges: edges}), true)
		turn := runEndTurn(ctx, r, userText("go"), nil)
		return turn, storedEvents(t, svc)
	}
	checkErr := func(t *testing.T, errs []error, match func(error) bool, want string) {
		t.Helper()
		if len(errs) != 1 || !match(errs[0]) {
			t.Errorf("Run() errors = %v, want one error matching %s", errs, want)
		}
	}
	checkNoEnd := func(t *testing.T, yielded, stored []*session.Event) {
		t.Helper()
		if n := len(endEvents(yielded)); n != 0 {
			t.Errorf("yielded %d end events, want 0", n)
		}
		if n := len(endEvents(stored)); n != 0 {
			t.Errorf("session holds %d end events, want 0", n)
		}
	}

	t.Run("node_error", func(t *testing.T) {
		boom := workflow.NewFunctionNode("boom", func(agent.Context, any) (string, error) {
			return "", errors.New("node boom")
		}, workflow.NodeConfig{})

		turn, stored := failingRun(t, t.Context(), workflow.Chain(workflow.Start, boom))

		checkErr(t, turn.errs, func(err error) bool { return strings.Contains(err.Error(), "node boom") }, `"node boom"`)
		checkNoEnd(t, turn.events, stored)
	})

	t.Run("multiple_terminal_outputs", func(t *testing.T) {
		out := func(name string) workflow.Node {
			return workflow.NewFunctionNode(name, func(agent.Context, any) (string, error) { return name, nil }, workflow.NodeConfig{})
		}
		a, b := out("a"), out("b")

		turn, stored := failingRun(t, t.Context(), []workflow.Edge{{From: workflow.Start, To: a}, {From: workflow.Start, To: b}})

		checkErr(t, turn.errs, func(err error) bool { return errors.Is(err, workflow.ErrMultipleTerminalOutputs) }, "workflow.ErrMultipleTerminalOutputs")
		checkNoEnd(t, turn.events, stored)
	})

	t.Run("context_canceled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		canceler := workflow.NewFunctionNode("canceler", func(nctx agent.Context, _ any) (string, error) {
			cancel()
			<-nctx.Done()
			return "", nctx.Err()
		}, workflow.NodeConfig{})

		turn, stored := failingRun(t, ctx, workflow.Chain(workflow.Start, canceler))

		checkErr(t, turn.errs, func(err error) bool { return errors.Is(err, context.Canceled) }, "context.Canceled")
		checkNoEnd(t, turn.events, stored)
	})

	t.Run("caller_stops_at_node_event", func(t *testing.T) {
		// d waits until pauser's long-running tool has run, so pauser's call is
		// stored before the caller stops at d's event.
		toolRan := make(chan struct{})
		var closeOnce sync.Once
		var dRuns atomic.Int32
		d, err := agent.New(agent.Config{Name: "d", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				dRuns.Add(1)
				select {
				case <-toolRan:
				case <-ctx.Done():
					return
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "d"
				ev.Content = genai.NewContentFromText("D", genai.RoleModel)
				yield(ev, nil)
			}
		}})
		if err != nil {
			t.Fatalf("agent.New(d) error = %v", err)
		}
		dn, err := workflow.NewAgentNode(d, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(d) error = %v", err)
		}
		wf := mustWorkflow(t, workflowagent.Config{Name: "wf", Edges: workflow.Chain(workflow.Start, dn)})
		type args struct{}
		approve, err := functiontool.New(functiontool.Config{Name: "approve", Description: "asks for approval", IsLongRunning: true},
			func(agent.Context, args) (map[string]any, error) {
				closeOnce.Do(func() { close(toolRan) })
				return nil, nil
			})
		if err != nil {
			t.Fatalf("functiontool.New() error = %v", err)
		}
		pauser, err := llmagent.New(llmagent.Config{
			Name: "pauser",
			Model: &testutil.MockModel{Responses: []*genai.Content{
				{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "lr-1", Name: "approve", Args: map[string]any{}}}}},
				genai.NewContentFromText("approved", genai.RoleModel),
			}},
			Tools: []tool.Tool{approve},
		})
		if err != nil {
			t.Fatalf("llmagent.New() error = %v", err)
		}
		par, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{wf, pauser}}})
		if err != nil {
			t.Fatalf("parallelagent.New() error = %v", err)
		}
		r, svc := newEndRunner(t, par, true)

		turn1 := runEndTurn(t.Context(), r, userText("go"), func(ev *session.Event) bool { return describe(ev) == "d:D" })

		if len(turn1.errs) != 0 {
			t.Fatalf("turn 1 Run() errors = %v, want none", turn1.errs)
		}
		checkNoEnd(t, turn1.events, storedEvents(t, svc))

		answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID: "lr-1", Name: "approve", Response: map[string]any{"status": "approved"},
		}}}}
		turn2 := runEndTurn(t.Context(), r, answer, nil)

		if len(turn2.errs) != 0 {
			t.Fatalf("turn 2 Run() errors = %v, want none", turn2.errs)
		}
		if got := dRuns.Load(); got != 2 {
			t.Errorf("d ran %d times over both turns, want 2 (wf did not end in turn 1)", got)
		}
	})

	t.Run("caller_stops_then_runs_parallel_agent_again", func(t *testing.T) {
		// root stops reading par at d's event, then runs par again in the same
		// run. par skips a sub-agent whose end the run recorded, so wf runs a
		// second time only if the stop recorded no end.
		var dRuns atomic.Int32
		wf := mustWorkflow(t, workflowagent.Config{Name: "wf", Edges: workflow.Chain(workflow.Start, textNode(t, "d", "D", &dRuns))})
		par, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{wf}}})
		if err != nil {
			t.Fatalf("parallelagent.New() error = %v", err)
		}
		root, err := agent.New(agent.Config{Name: "root", SubAgents: []agent.Agent{par}, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for ev, err := range par.Run(ctx) {
					if err == nil && describe(ev) == "d:D" {
						break
					}
					if !yield(ev, err) {
						return
					}
				}
				for ev, err := range par.Run(ctx) {
					if !yield(ev, err) {
						return
					}
				}
			}
		}})
		if err != nil {
			t.Fatalf("agent.New(root) error = %v", err)
		}
		r, _ := newEndRunner(t, root, true)

		turn := runEndTurn(t.Context(), r, userText("go"), nil)

		if len(turn.errs) != 0 {
			t.Fatalf("Run() errors = %v, want none", turn.errs)
		}
		if diff := cmp.Diff([]string{"par:", "par:", "d:D", "wf:END", "par:END"}, describeAll(turn.events)); diff != "" {
			t.Errorf("yielded events (-want +got):\n%s", diff)
		}
		if got := dRuns.Load(); got != 2 {
			t.Errorf("d ran %d times, want 2 (wf did not end when root stopped reading)", got)
		}
	})

	t.Run("resume_invalid_schema", func(t *testing.T) {
		wf := mustWorkflow(t, workflowagent.Config{Name: "wf", Edges: workflow.Chain(workflow.Start, askerNode("asker", "approval", approvalSchema()))})
		r, svc := newEndRunner(t, wf, true)
		if turn1 := runEndTurn(t.Context(), r, userText("go"), nil); len(turn1.errs) != 0 {
			t.Fatalf("turn 1 Run() errors = %v, want none", turn1.errs)
		}

		turn2 := runEndTurn(t.Context(), r, testutil.InputResponse("approval", "not an object"), nil)

		checkErr(t, turn2.errs, func(err error) bool { return errors.Is(err, workflow.ErrInvalidResumeResponse) }, "workflow.ErrInvalidResumeResponse")
		checkNoEnd(t, turn2.events, storedEvents(t, svc))
	})
}
