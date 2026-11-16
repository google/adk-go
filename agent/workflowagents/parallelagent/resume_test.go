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

package parallelagent_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// resumeTimeout bounds every run in this file, so that a parallel agent that
// fails to cancel a branch blocked on its context fails the test instead of
// hanging it.
const resumeTimeout = 10 * time.Second

// resumeAgent returns an agent built with agent.New, named name, whose run is
// body.
func resumeAgent(t *testing.T, name string, body func(ctx agent.InvocationContext, yield func(*session.Event, error) bool)) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) { body(ctx, yield) }
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%q) error = %v", name, err)
	}
	return a
}

// resumeText returns an event of ctx's agent with the text content text.
func resumeText(ctx agent.InvocationContext, text string) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = ctx.Agent().Name()
	ev.Branch = ctx.Branch()
	ev.Content = genai.NewContentFromText(text, genai.RoleModel)
	return ev
}

// resumeBlocker returns an agent named name that yields nothing and waits for
// its context to be canceled. After the wait, onCancel runs before the agent's
// Run returns, so a test can observe whether the parallel agent awaited it.
func resumeBlocker(t *testing.T, name string, onCancel func()) agent.Agent {
	return resumeAgent(t, name, func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
		<-ctx.Done()
		if onCancel != nil {
			onCancel()
		}
	})
}

// resumeParallel returns a parallel agent named "par" over subAgents.
func resumeParallel(t *testing.T, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	par, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "par", SubAgents: subAgents}})
	if err != nil {
		t.Fatalf("parallelagent.New() error = %v", err)
	}
	return par
}

// resumeRunner returns a runner over root with an in-memory session service
// that creates sessions on demand.
func resumeRunner(t *testing.T, root agent.Agent, resumable bool) (*runner.Runner, session.Service) {
	t.Helper()
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             root,
		SessionService:    sessions,
		AutoCreateSession: true,
		Resumable:         resumable,
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return r, sessions
}

// resumeTurn is what one call of Runner.Run produced.
type resumeTurn struct {
	events       []string // rendered by resumeDescribe
	errs         []string
	invocationID string
}

// resumeRun runs one turn in session "s1" and collects its events and errors.
// A nil msg resumes the invocation named by opts. onEvent, when non-nil, sees
// each event as it is yielded and returns false to stop the run there.
func resumeRun(t *testing.T, r *runner.Runner, msg *genai.Content, onEvent func(*session.Event) bool, opts ...runner.RunOption) resumeTurn {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), resumeTimeout)
	defer cancel()
	var turn resumeTurn
	for ev, err := range r.Run(ctx, "user", "s1", msg, agent.RunConfig{}, opts...) {
		if err != nil {
			turn.errs = append(turn.errs, err.Error())
			continue
		}
		turn.invocationID = ev.InvocationID
		turn.events = append(turn.events, resumeDescribe(ev))
		if onEvent != nil && !onEvent(ev) {
			break
		}
	}
	return turn
}

// resumeSessionEvents returns the events stored in session "s1", other than
// the user's messages, rendered by resumeDescribe.
func resumeSessionEvents(t *testing.T, sessions session.Service) []string {
	t.Helper()
	got, err := sessions.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var events []string
	for ev := range got.Session.Events().All() {
		if ev.Author == "user" {
			continue
		}
		events = append(events, resumeDescribe(ev))
	}
	return events
}

// resumeDescribe renders ev as "author:END" for an end-of-agent event,
// "author:STATE({})" for an agent-state event with an empty state (and
// "author:STATE(k=v,...)" otherwise), or "author:" followed by "FC(name)" for
// each function call and the text of each text part, with "[ESC]" appended
// when the event escalates.
func resumeDescribe(ev *session.Event) string {
	if ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent {
		return ev.Author + ":END"
	}
	if ev.Actions.AgentState != nil {
		if len(ev.Actions.AgentState) == 0 {
			return ev.Author + ":STATE({})"
		}
		var kv []string
		for k, v := range ev.Actions.AgentState {
			kv = append(kv, fmt.Sprintf("%s=%v", k, v))
		}
		slices.Sort(kv)
		return ev.Author + ":STATE(" + strings.Join(kv, ",") + ")"
	}
	var b strings.Builder
	if ev.Content != nil {
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil {
				fmt.Fprintf(&b, "FC(%s)", p.FunctionCall.Name)
				continue
			}
			b.WriteString(p.Text)
		}
	}
	if ev.Actions.Escalate {
		b.WriteString("[ESC]")
	}
	return ev.Author + ":" + b.String()
}

// TestParallelAgent_CheckpointOrderAndLazyStart checks that a resumable
// parallel agent stores its {} checkpoint before any branch starts, and that
// an empty one yields nothing.
//
// Fails if the branches are started before the checkpoint is yielded (as when
// the errgroup is started outside the iterator), or if the empty-agent return
// comes after the checkpoint.
func TestParallelAgent_CheckpointOrderAndLazyStart(t *testing.T) {
	t.Run("checkpoint is handled before any branch starts", func(t *testing.T) {
		var checkpointSeen, branchStarted atomic.Bool
		b1 := resumeAgent(t, "b1", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			branchStarted.Store(true)
			if !yield(resumeText(ctx, fmt.Sprintf("saw=%t", checkpointSeen.Load())), nil) {
				return
			}
			agent.RecordEndOfAgent(ctx)
		})
		r, _ := resumeRunner(t, resumeParallel(t, b1), true)

		var startedAtCheckpoint []bool
		turn := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), func(ev *session.Event) bool {
			if ev.Author == "par" && ev.Actions.AgentState != nil {
				// A branch started before the checkpoint would have run by now.
				time.Sleep(50 * time.Millisecond)
				startedAtCheckpoint = append(startedAtCheckpoint, branchStarted.Load())
				checkpointSeen.Store(true)
			}
			return true
		})

		if len(turn.errs) != 0 {
			t.Fatalf("Run() errors = %q, want none", turn.errs)
		}
		if diff := cmp.Diff([]bool{false}, startedAtCheckpoint); diff != "" {
			t.Errorf("branch started when the caller handled the checkpoint (-want +got):\n%s", diff)
		}
		want := []string{"par:STATE({})", "b1:saw=true", "par:END"}
		if diff := cmp.Diff(want, turn.events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
	})

	t.Run("empty parallel agent yields nothing", func(t *testing.T) {
		r, _ := resumeRunner(t, resumeParallel(t), true)

		turn := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), nil)

		if len(turn.errs) != 0 {
			t.Fatalf("Run() errors = %q, want none", turn.errs)
		}
		if len(turn.events) != 0 {
			t.Errorf("events = %q, want none", turn.events)
		}
	})
}

// TestParallelAgent_ConsumerStopsAtCheckpoint checks that a caller that stops
// at the {} checkpoint of a resumable parallel agent ends the run before any
// branch starts, and that no event follows the stop.
//
// Fails if the parallel agent ignores the false that yield returns for its
// checkpoint: it then starts the branches and yields their events to a loop
// body that already returned false, which panics.
func TestParallelAgent_ConsumerStopsAtCheckpoint(t *testing.T) {
	var branchRuns atomic.Int32
	b1 := resumeAgent(t, "b1", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
		branchRuns.Add(1)
		if !yield(resumeText(ctx, "b1 text"), nil) {
			return
		}
		agent.RecordEndOfAgent(ctx)
	})
	r, sessions := resumeRunner(t, resumeParallel(t, b1), true)

	turn := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), func(*session.Event) bool { return false })

	if len(turn.errs) != 0 {
		t.Fatalf("Run() errors = %q, want none", turn.errs)
	}
	want := []string{"par:STATE({})"}
	if diff := cmp.Diff(want, turn.events); diff != "" {
		t.Errorf("events (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(want, resumeSessionEvents(t, sessions)); diff != "" {
		t.Errorf("session events (-want +got):\n%s", diff)
	}
	if got := branchRuns.Load(); got != 0 {
		t.Errorf("branch ran %d times, want 0", got)
	}
}

// TestParallelAgent_GraphResumeEndsOnlyWhenEverySubAgentEnded checks that a
// resumable parallel agent resumed by the answer to a call in a graph
// workflow runs only the branch holding that workflow, and writes no end
// event while a sub-agent outside that branch has not ended, even though the
// branch it ran did end.
//
// Fails if the end decision looks only at the branches selected by graph
// narrowing instead of at every sub-agent.
func TestParallelAgent_GraphResumeEndsOnlyWhenEverySubAgentEnded(t *testing.T) {
	// a pauses on a call of its own that the user never answers, so it has
	// no end mark in either turn.
	var aRuns atomic.Int32
	a := resumeAgent(t, "a", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
		aRuns.Add(1)
		ev := resumeText(ctx, "")
		ev.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "a-call", Name: "ask", Args: map[string]any{}}},
		}}
		ev.LongRunningToolIDs = []string{"a-call"}
		yield(ev, nil)
	})
	seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{testutil.NewInputWorkflow(t, "wf")}}})
	if err != nil {
		t.Fatalf("sequentialagent.New() error = %v", err)
	}
	r, _ := resumeRunner(t, resumeParallel(t, a, seq), true)
	turn1 := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), nil)
	if len(turn1.errs) != 0 {
		t.Fatalf("turn 1 Run() errors = %q, want none", turn1.errs)
	}

	turn2 := resumeRun(t, r, testutil.InputResponse("b", "approved"), nil)

	if len(turn2.errs) != 0 {
		t.Fatalf("turn 2 Run() errors = %q, want none", turn2.errs)
	}
	var parEvents []string
	for _, ev := range turn2.events {
		if strings.HasPrefix(ev, "par:") {
			parEvents = append(parEvents, ev)
		}
	}
	if len(parEvents) != 0 {
		t.Errorf("par's turn 2 events = %q, want none", parEvents)
	}
	if !slices.Contains(turn2.events, "d:D") || !slices.Contains(turn2.events, "seq:END") {
		t.Errorf("turn 2 events = %q, want d's output and seq's end", turn2.events)
	}
	if got := aRuns.Load(); got != 1 {
		t.Errorf("a ran %d times over both turns, want 1", got)
	}
}

// resumeEscalator returns an agent named "esc" that yields ev(ctx) and, if
// asked for another event, records that in followUpPulled and yields a
// follow-up.
func resumeEscalator(t *testing.T, followUpPulled *atomic.Bool, ev func(ctx agent.InvocationContext) *session.Event) agent.Agent {
	return resumeAgent(t, "esc", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
		if !yield(ev(ctx), nil) {
			return
		}
		followUpPulled.Store(true)
		yield(resumeText(ctx, "follow-up"), nil)
	})
}

// resumeEscalating returns a text event "escalating" of ctx's agent that
// escalates.
func resumeEscalating(ctx agent.InvocationContext) *session.Event {
	ev := resumeText(ctx, "escalating")
	ev.Actions.Escalate = true
	return ev
}

// TestParallelAgent_EscalationShortCircuitAndTeardown checks that an
// escalation by a direct sub-agent ends the parallel agent: the escalating
// branch is not asked for another event, a sibling blocked on its context is
// canceled and awaited, and a resumable run then writes its end event, unless
// the escalating event or an earlier event of the run pauses the invocation.
//
// Fails if the escalating branch's ack is closed before the merge loop stops,
// if the end event is built before the branches are shut down, if the
// escalation check ignores the author, if paused is left out of the
// condition that withholds the end event, or if a later event that does not
// pause clears paused.
func TestParallelAgent_EscalationShortCircuitAndTeardown(t *testing.T) {
	t.Run("resumable escalation awaits teardown before END and never pulls follow-up", func(t *testing.T) {
		var followUpPulled, siblingDone atomic.Bool
		sibling := resumeBlocker(t, "sibling", func() {
			time.Sleep(20 * time.Millisecond)
			siblingDone.Store(true)
		})
		r, _ := resumeRunner(t, resumeParallel(t, resumeEscalator(t, &followUpPulled, resumeEscalating), sibling), true)

		var siblingDoneAtEnd []bool
		turn := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), func(ev *session.Event) bool {
			if ev.Actions.EndOfAgent != nil {
				siblingDoneAtEnd = append(siblingDoneAtEnd, siblingDone.Load())
			}
			return true
		})

		if len(turn.errs) != 0 {
			t.Fatalf("Run() errors = %q, want none", turn.errs)
		}
		want := []string{"par:STATE({})", "esc:escalating[ESC]", "par:END"}
		if diff := cmp.Diff(want, turn.events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if followUpPulled.Load() {
			t.Error("the escalating agent was asked for its follow-up event")
		}
		if diff := cmp.Diff([]bool{true}, siblingDoneAtEnd); diff != "" {
			t.Errorf("sibling teardown finished before the end event (-want +got):\n%s", diff)
		}
	})

	t.Run("non-resumable escalation short-circuits without state or END", func(t *testing.T) {
		var followUpPulled atomic.Bool
		r, _ := resumeRunner(t, resumeParallel(t, resumeEscalator(t, &followUpPulled, resumeEscalating), resumeBlocker(t, "sibling", nil)), false)

		turn := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), nil)

		if len(turn.errs) != 0 {
			t.Fatalf("Run() errors = %q, want none", turn.errs)
		}
		if diff := cmp.Diff([]string{"esc:escalating[ESC]"}, turn.events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if followUpPulled.Load() {
			t.Error("the escalating agent was asked for its follow-up event")
		}
	})

	t.Run("escalation by a nested agent keeps siblings running", func(t *testing.T) {
		escSeen := make(chan struct{})
		inner := resumeAgent(t, "inner", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			if yield(resumeEscalating(ctx), nil) {
				close(escSeen)
			}
		})
		loop, err := loopagent.New(loopagent.Config{MaxIterations: 1, AgentConfig: agent.Config{Name: "loop", SubAgents: []agent.Agent{inner}}})
		if err != nil {
			t.Fatalf("loopagent.New() error = %v", err)
		}
		sibling := resumeAgent(t, "sibling", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			select {
			case <-escSeen:
				yield(resumeText(ctx, "sibling text"), nil)
			case <-ctx.Done():
			}
		})
		r, _ := resumeRunner(t, resumeParallel(t, loop, sibling), false)

		turn := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), nil)

		if len(turn.errs) != 0 {
			t.Fatalf("Run() errors = %q, want none", turn.errs)
		}
		if diff := cmp.Diff([]string{"inner:escalating[ESC]", "sibling:sibling text"}, turn.events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
	})

	t.Run("escalation with pause withholds END", func(t *testing.T) {
		var followUpPulled atomic.Bool
		pausingEscalation := func(ctx agent.InvocationContext) *session.Event {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = ctx.Agent().Name()
			ev.Branch = ctx.Branch()
			ev.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "lr-1", Name: "ask", Args: map[string]any{}}},
			}}
			ev.LongRunningToolIDs = []string{"lr-1"}
			ev.Actions.Escalate = true
			return ev
		}
		r, _ := resumeRunner(t, resumeParallel(t, resumeEscalator(t, &followUpPulled, pausingEscalation), resumeBlocker(t, "sibling", nil)), true)

		turn := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), nil)

		if len(turn.errs) != 0 {
			t.Fatalf("Run() errors = %q, want none", turn.errs)
		}
		if diff := cmp.Diff([]string{"par:STATE({})", "esc:FC(ask)[ESC]"}, turn.events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
	})

	t.Run("pause followed by a later escalation withholds END", func(t *testing.T) {
		fastDone := make(chan struct{})
		fast := resumeAgent(t, "fast", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			defer close(fastDone)
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = ctx.Agent().Name()
			ev.Branch = ctx.Branch()
			ev.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "lr-1", Name: "ask", Args: map[string]any{}}},
			}}
			ev.LongRunningToolIDs = []string{"lr-1"}
			yield(ev, nil)
		})
		// esc escalates only after the parallel agent has handled fast's
		// pausing event, so the escalation is the last event of the run.
		esc := resumeAgent(t, "esc", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			select {
			case <-fastDone:
				yield(resumeEscalating(ctx), nil)
			case <-ctx.Done():
			}
		})
		r, _ := resumeRunner(t, resumeParallel(t, fast, esc), true)

		turn := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), nil)

		if len(turn.errs) != 0 {
			t.Fatalf("Run() errors = %q, want none", turn.errs)
		}
		want := []string{"par:STATE({})", "fast:FC(ask)", "esc:escalating[ESC]"}
		if diff := cmp.Diff(want, turn.events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
	})
}

// TestParallelAgent_ErrorOrEarlyStopWithholdsEndEvent checks that a resumable
// parallel agent writes no end event when a branch fails, even after every
// sub-agent recorded its end, or when its consumer stops on an escalating
// event; and that a stored state of the wrong shape fails the run before any
// branch starts.
//
// Fails if failed or stopped is left out of the condition that withholds the
// end event, if stopped is recorded only after the escalation check (the end
// event is then yielded after yield returned false, which panics), or if the
// stored state is not decoded.
func TestParallelAgent_ErrorOrEarlyStopWithholdsEndEvent(t *testing.T) {
	t.Run("branch error withholds END", func(t *testing.T) {
		okDone := make(chan struct{})
		okBranch := resumeAgent(t, "ok_branch", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			defer close(okDone)
			if !yield(resumeText(ctx, "done"), nil) {
				return
			}
			agent.RecordEndOfAgent(ctx)
		})
		badBranch := resumeAgent(t, "bad_branch", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			<-okDone
			agent.RecordEndOfAgent(ctx)
			yield(nil, errors.New("boom"))
		})
		r, _ := resumeRunner(t, resumeParallel(t, okBranch, badBranch), true)

		turn := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), nil)

		if diff := cmp.Diff([]string{"par:STATE({})", "ok_branch:done"}, turn.events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if len(turn.errs) != 1 || !strings.Contains(turn.errs[0], `failed to run sub-agent "bad_branch": boom`) {
			t.Errorf("Run() errors = %q, want one containing %q", turn.errs, `failed to run sub-agent "bad_branch": boom`)
		}
	})

	t.Run("consumer breaks on the escalating event", func(t *testing.T) {
		var followUpPulled atomic.Bool
		r, sessions := resumeRunner(t, resumeParallel(t, resumeEscalator(t, &followUpPulled, resumeEscalating), resumeBlocker(t, "slow", nil)), true)

		turn := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), func(ev *session.Event) bool {
			return !ev.Actions.Escalate
		})

		if len(turn.errs) != 0 {
			t.Fatalf("Run() errors = %q, want none", turn.errs)
		}
		want := []string{"par:STATE({})", "esc:escalating[ESC]"}
		if diff := cmp.Diff(want, turn.events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(want, resumeSessionEvents(t, sessions)); diff != "" {
			t.Errorf("session events (-want +got):\n%s", diff)
		}
	})

	t.Run("malformed stored state", func(t *testing.T) {
		var branchRuns atomic.Int32
		b1 := resumeAgent(t, "b1", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			branchRuns.Add(1)
			yield(resumeText(ctx, "b1 text"), nil)
		})
		par := resumeParallel(t, b1)
		rc := &runconfig.RunConfig{Resumable: true}
		rc.SeedAgentRuns(map[string]map[string]any{"par": {"extra": 1}}, nil)
		ctx := icontext.NewInvocationContext(runconfig.ToContext(t.Context(), rc), icontext.InvocationContextParams{
			Agent:        par,
			InvocationID: "inv-1",
		})

		var events, errs []string
		for ev, err := range par.Run(ctx) {
			if err != nil {
				errs = append(errs, err.Error())
				continue
			}
			events = append(events, resumeDescribe(ev))
		}

		if len(events) != 0 {
			t.Errorf("events = %q, want none", events)
		}
		if len(errs) != 1 || !strings.Contains(errs[0], `parallel agent "par": stored agent state`) {
			t.Errorf("errors = %q, want one containing %q", errs, `parallel agent "par": stored agent state`)
		}
		if got := branchRuns.Load(); got != 0 {
			t.Errorf("branch runs = %d, want 0", got)
		}
	})
}

// TestParallelAgent_CustomSubAgentRecordsEnd checks that a resumable parallel
// agent over an agent.New sub-agent ends only when the sub-agent calls
// agent.RecordEndOfAgent, and that the record is not stored, so resuming an
// invocation whose parallel agent did not end runs the sub-agent again.
//
// Fails if the parallel agent ignores the run's end records (allEnded), if it
// writes its end event without them, or if it runs an ended sub-agent again.
func TestParallelAgent_CustomSubAgentRecordsEnd(t *testing.T) {
	t.Run("resumable parallel ends when custom sub-agent records end", func(t *testing.T) {
		c1 := resumeAgent(t, "c1", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			if !yield(resumeText(ctx, "hi"), nil) {
				return
			}
			agent.RecordEndOfAgent(ctx)
		})
		r, _ := resumeRunner(t, resumeParallel(t, c1), true)

		turn := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), nil)

		if len(turn.errs) != 0 {
			t.Fatalf("Run() errors = %q, want none", turn.errs)
		}
		if diff := cmp.Diff([]string{"par:STATE({})", "c1:hi", "par:END"}, turn.events); diff != "" {
			t.Errorf("events (-want +got):\n%s", diff)
		}
	})

	t.Run("resumable parallel does not end when custom sub-agent omits RecordEndOfAgent, and resume re-runs it", func(t *testing.T) {
		c1 := resumeAgent(t, "c1", func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			yield(resumeText(ctx, "hi"), nil)
		})
		r, _ := resumeRunner(t, resumeParallel(t, c1), true)

		turn1 := resumeRun(t, r, genai.NewContentFromText("go", genai.RoleUser), nil)
		if len(turn1.errs) != 0 {
			t.Fatalf("turn 1 Run() errors = %q, want none", turn1.errs)
		}
		if diff := cmp.Diff([]string{"par:STATE({})", "c1:hi"}, turn1.events); diff != "" {
			t.Errorf("turn 1 events (-want +got):\n%s", diff)
		}

		turn2 := resumeRun(t, r, nil, nil, runner.WithInvocationID(turn1.invocationID))
		if len(turn2.errs) != 0 {
			t.Fatalf("turn 2 Run() errors = %q, want none", turn2.errs)
		}
		if diff := cmp.Diff([]string{"c1:hi"}, turn2.events); diff != "" {
			t.Errorf("turn 2 events (-want +got):\n%s", diff)
		}
	})
}
