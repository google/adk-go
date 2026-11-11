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

package agentanalytics

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"weak"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

// signal is fired once and waited for with a timeout, so that a test whose
// interleaving breaks fails on its assertions rather than hanging.
type signal struct {
	once sync.Once
	ch   chan struct{}
}

func newSignal() *signal { return &signal{ch: make(chan struct{})} }

func (s *signal) fire() { s.once.Do(func() { close(s.ch) }) }

func (s *signal) wait() {
	select {
	case <-s.ch:
	case <-time.After(10 * time.Second):
	}
}

// afterAgentRunKey holds, in the context of an activation that a test's
// custom agent runs, the number of the run.
type afterAgentRunKey struct{}

func runNumber(ctx context.Context) int {
	n, _ := ctx.Value(afterAgentRunKey{}).(int)
	return n
}

// textEvent returns an event that session.NewEvent creates for author on
// branch with text as its content.
func textEvent(ctx agent.InvocationContext, author, branch, text string) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = author
	ev.Branch = branch
	ev.Content = textContent("model", text)
	return ev
}

// runItem is an event or error of run n, which forwardRun sends to the custom
// agent that started the run and which that agent acknowledges once it
// delivered or held the item.
type runItem struct {
	n   int
	ev  *session.Event
	err error
	ack chan struct{}
}

// forwardRun runs a with run number n on a goroutine and sends its items to
// items, one at a time. wg.Done is called once the run ended.
func forwardRun(ctx agent.InvocationContext, a agent.Agent, n int, items chan<- runItem, done <-chan struct{}, wg *sync.WaitGroup) {
	go func() {
		defer wg.Done()
		runCtx := ctx.WithContext(context.WithValue(ctx, afterAgentRunKey{}, n))
		for ev, err := range a.Run(runCtx) {
			ack := make(chan struct{})
			select {
			case items <- runItem{n: n, ev: ev, err: err, ack: ack}:
			case <-done:
				return
			}
			select {
			case <-ack:
			case <-done:
				return
			}
		}
	}()
}

// failingChild returns an agent named child that yields a partial answer and
// then fails with msg.
func failingChild(t *testing.T, msg string) agent.Agent {
	t.Helper()
	child, err := agent.New(agent.Config{
		Name: "child",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				if !yield(textEvent(ctx, "child", ctx.Branch(), "child partial"), nil) {
					return
				}
				yield(nil, errors.New(msg))
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child): %v", err)
	}
	return child
}

func wantInvocationFailure(t *testing.T, rows []map[string]any, msg string) {
	t.Helper()
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
	}
	invErr := onlyRowOfType(t, rows, EventInvocationError)
	if got, _ := invErr["error_message"].(string); !strings.Contains(got, msg) {
		t.Errorf("INVOCATION_ERROR error_message = %q, want %q", got, msg)
	}
}

func wantInvocationCompleted(t *testing.T, rows []map[string]any) {
	t.Helper()
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
	}
	_ = onlyRowOfType(t, rows, EventInvocationCompleted)
}

// TestCleanAfterAgentEventKeepsPeerFailure_Runner runs activations of the
// custom agent inner under a custom root. One activation of inner finishes
// cleanly and its AfterAgentCallback returns a summary only once another
// activation of inner drained the failure of its child under par and did not
// recover from it. The summary is the clean activation's after-agent event,
// so it never clears the other activation's failure, whether it arrives while
// that activation is open or once it retired, and whatever shows that the
// clean activation's run ended earlier:
//
//   - pop_then_push: the first activation yielded progress and returned, and
//     root starts the second one while ADK runs the first one's
//     AfterAgentCallback, so no other activation was open when the first one
//     retired;
//   - mid_retired: an intermediate custom agent mid runs both activations
//     concurrently and forwards their events, and root holds the summary
//     until mid retired;
//   - fifth_activation: four activations retired one after another without
//     a response before the failing one started, and the second one's
//     AfterAgentCallback returns the summary.
//
// The failing activation's own recovery still completes the run. It recovers
// in its run in pop_then_push, and in its AfterAgentCallback otherwise: the
// after-agent event of an activation that retired while the failing one was
// open, or of one that retired without a response, may still come then, so
// an event that the failing activation creates in its run could be that
// event with both action maps replaced (see Config).
func TestCleanAfterAgentEventKeepsPeerFailure_Runner(t *testing.T) {
	const failure = "failing inner's child failure"
	type config struct {
		shape            string
		deliverWhileOpen bool
		failingRecovers  bool
	}
	run := func(t *testing.T, c config) []map[string]any {
		t.Helper()
		tm := newTraceManager()
		p, takeRows := newQueuedRowsPlugin(t, tm)
		tm.selfPlugin = weak.Make(p)
		// clean is the run whose AfterAgentCallback returns the summary, and
		// failing the run that drains its child's failure.
		clean, failing, runs := 1, 2, 2
		if c.shape == "fifth_activation" {
			clean, failing, runs = 2, 5, 5
		}
		summary := fmt.Sprintf("inner-%d summary", clean)
		started := make([]*signal, runs+1)
		popped := make([]*signal, runs+1)
		for n := range started {
			started[n], popped[n] = newSignal(), newSignal()
		}
		swallowed, summarySeen := newSignal(), newSignal()
		defer swallowed.fire()
		defer summarySeen.fire()
		for n := 1; n <= runs; n++ {
			defer started[n].fire()
			defer popped[n].fire()
		}
		par := newParallelAgent(t, "par", failingChild(t, failure))
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					n := runNumber(ctx)
					started[n].fire()
					if n != failing {
						// The earlier activations of fifth_activation retire
						// one after another without a response, so that they
						// look alike.
						if c.shape != "fifth_activation" && !yield(textEvent(ctx, "inner", ctx.Branch(), fmt.Sprintf("inner-%d progress", n)), nil) {
							return
						}
						if c.shape == "mid_retired" {
							// The other activation is open when this one
							// retires.
							started[failing].wait()
						}
						return
					}
					drained := false
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							drained = true
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					if drained {
						// par retired, so the failure is recorded.
						swallowed.fire()
					}
					if c.failingRecovers && c.shape == "pop_then_push" && !yield(textEvent(ctx, "inner", ctx.Branch(), "failing inner recovery"), nil) {
						return
					}
					// The summary is created while this activation runs.
					summarySeen.wait()
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(cctx agent.Context) (*genai.Content, error) {
					n := runNumber(cctx)
					popped[n].fire()
					if n == failing && c.failingRecovers && c.shape != "pop_then_push" {
						return textContent("model", "failing inner recovery"), nil
					}
					if n != clean {
						return nil, nil
					}
					swallowed.wait()
					return textContent("model", summary), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		// runInner runs the activations of inner on goroutines of the agent
		// whose context is ctx, starting each when the shape says so, and
		// returns their items.
		runInner := func(ctx agent.InvocationContext, done <-chan struct{}) <-chan runItem {
			items := make(chan runItem)
			var wg sync.WaitGroup
			wg.Add(runs)
			go func() {
				wg.Wait()
				close(items)
			}()
			go func() {
				for n := 1; n <= runs; n++ {
					if n > 1 {
						prev := popped[n-1]
						if c.shape == "mid_retired" {
							prev = started[n-1]
						}
						select {
						case <-prev.ch:
						case <-done:
							for ; n <= runs; n++ {
								wg.Done()
							}
							return
						}
					}
					forwardRun(ctx, inner, n, items, done, &wg)
				}
			}()
			return items
		}
		// deliver delivers or holds the items, holding the summary until the
		// items end unless the case delivers it at once, and reports whether
		// the caller still reads.
		deliver := func(items <-chan runItem, yield func(*session.Event, error) bool) bool {
			var held []*session.Event
			for it := range items {
				isSummary := it.err == nil && it.ev != nil && isTextContent(it.ev.Content, summary)
				if isSummary && !c.deliverWhileOpen {
					held = append(held, it.ev)
					summarySeen.fire()
					close(it.ack)
					continue
				}
				ok := yield(it.ev, it.err)
				if isSummary {
					summarySeen.fire()
				}
				close(it.ack)
				if !ok {
					return false
				}
			}
			for _, ev := range held {
				if !yield(ev, nil) {
					return false
				}
			}
			return true
		}
		var root agent.Agent
		if c.shape == "mid_retired" {
			mid, err := agent.New(agent.Config{
				Name:      "mid",
				SubAgents: []agent.Agent{inner},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						done := make(chan struct{})
						defer close(done)
						for it := range runInner(ctx, done) {
							ok := yield(it.ev, it.err)
							close(it.ack)
							if !ok {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(mid): %v", err)
			}
			root, err = agent.New(agent.Config{
				Name:      "root",
				SubAgents: []agent.Agent{mid},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						var held []*session.Event
						for ev, runErr := range mid.Run(ctx) {
							if runErr == nil && ev != nil && isTextContent(ev.Content, summary) {
								held = append(held, ev)
								summarySeen.fire()
								continue
							}
							if !yield(ev, runErr) {
								return
							}
						}
						for _, ev := range held {
							if !yield(ev, nil) {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(root): %v", err)
			}
		} else {
			root, err = agent.New(agent.Config{
				Name:      "root",
				SubAgents: []agent.Agent{inner},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						done := make(chan struct{})
						defer close(done)
						deliver(runInner(ctx, done), yield)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(root): %v", err)
			}
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-clean-after-agent", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return takeRows()
	}
	for _, tc := range []struct {
		name string
		c    config
	}{
		{"pop_then_push/summary_delivered_while_the_failing_activation_is_open", config{shape: "pop_then_push", deliverWhileOpen: true}},
		{"pop_then_push/summary_delivered_once_the_failing_activation_retired", config{shape: "pop_then_push"}},
		{"mid_retired/summary_delivered_once_mid_retired", config{shape: "mid_retired"}},
		{"fifth_activation/summary_delivered_while_the_failing_activation_is_open", config{shape: "fifth_activation", deliverWhileOpen: true}},
		{"fifth_activation/summary_delivered_once_the_failing_activation_retired", config{shape: "fifth_activation"}},
	} {
		t.Run(tc.name+"/keeps_the_failure", func(t *testing.T) {
			rows := run(t, tc.c)
			wantInvocationFailure(t, rows, failure)
			// The summary is no open activation's, so its row is placed
			// under the invocation, not under the failing activation.
			summary := "inner-1 summary"
			if tc.c.shape == "fifth_activation" {
				summary = "inner-2 summary"
			}
			invSpan := placeOf(onlyRowOfType(t, rows, EventInvocationStarting)).span
			var placed []string
			for _, r := range rowsOfType(rows, EventAgentResponse) {
				if strings.Contains(fmt.Sprint(r["content"]), summary) {
					placed = append(placed, placeOf(r).span)
				}
			}
			if len(placed) != 1 || placed[0] != invSpan {
				t.Errorf("summary rows placed at %v, want one under the invocation span %q", placed, invSpan)
			}
		})
		t.Run(tc.name+"/own_recovery_completes", func(t *testing.T) {
			c := tc.c
			c.failingRecovers = true
			wantInvocationCompleted(t, run(t, c))
		})
	}
}

// TestRecoveryWithoutOwnBranchSettlesRetiredAuthor_Runner runs root -> par ->
// inner -> ipar -> child, where inner, the only agent named inner and on
// the branch par.inner, drains the child's failure and recovers with an
// event that session.NewEvent creates without a branch, or with par's
// branch, as the workflow scheduler stamps an enclosing node run's branch.
// root holds the recovery until inner retired: until par finished, or, in a
// second iteration of par, until the next activation of inner on par.inner
// is open. No activation of inner on another branch can have created the
// event, so it is the retired activation's recovery and the run completes.
// Without the recovery, the failure stays.
func TestRecoveryWithoutOwnBranchSettlesRetiredAuthor_Runner(t *testing.T) {
	const failure = "inner's child failure"
	type iterKey struct{}
	run := func(t *testing.T, evBranch string, nextIteration, recovers bool) []map[string]any {
		t.Helper()
		tm := newTraceManager()
		p, takeRows := newQueuedRowsPlugin(t, tm)
		tm.selfPlugin = weak.Make(p)
		iteration := func(ctx context.Context) int {
			n, _ := ctx.Value(iterKey{}).(int)
			return n
		}
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if !yield(textEvent(ctx, "child", ctx.Branch(), "child partial"), nil) {
						return
					}
					if iteration(ctx) == 1 {
						yield(nil, errors.New(failure))
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		ipar := newParallelAgent(t, "ipar", child)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{ipar},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if iteration(ctx) == 2 {
						yield(textEvent(ctx, "inner", ctx.Branch(), "inner-2 progress"), nil)
						return
					}
					for ev, runErr := range ipar.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					if recovers {
						yield(textEvent(ctx, "inner", evBranch, "inner recovery"), nil)
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		par := newParallelAgent(t, "par", inner)
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var held []*session.Event
					iterations := 1
					if nextIteration {
						iterations = 2
					}
					for n := 1; n <= iterations; n++ {
						iterCtx := ctx.WithContext(context.WithValue(ctx, iterKey{}, n))
						for ev, runErr := range par.Run(iterCtx) {
							if runErr == nil && ev != nil && isTextContent(ev.Content, "inner recovery") {
								held = append(held, ev)
								continue
							}
							if n == 2 && runErr == nil && ev != nil && ev.Author == "inner" {
								// The next activation of inner is open.
								for _, h := range held {
									if !yield(h, nil) {
										return
									}
								}
								held = nil
							}
							if !yield(ev, runErr) {
								return
							}
						}
					}
					for _, ev := range held {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-recovery-without-branch", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return takeRows()
	}
	for _, branch := range []struct{ name, value string }{{"without_branch", ""}, {"with_enclosing_branch", "par"}} {
		for _, delivery := range []struct {
			name          string
			nextIteration bool
		}{{"after_par_finished", false}, {"while_the_next_activation_is_open", true}} {
			name := branch.name + "/" + delivery.name
			t.Run(name+"/recovery_completes", func(t *testing.T) {
				wantInvocationCompleted(t, run(t, branch.value, delivery.nextIteration, true))
			})
			t.Run(name+"/no_recovery_keeps_the_failure", func(t *testing.T) {
				wantInvocationFailure(t, run(t, branch.value, delivery.nextIteration, false), failure)
			})
		}
	}
}

// TestRetiredPeerRunAndAfterAgentEventsSettleOneCredit_Runner runs, under a
// custom root, an llmagent worker whose model event AfterModelCallback holds,
// and meanwhile two custom agents named worker. Each of them retires while the
// llmagent is open, without a response, since root holds its events, so each
// leaves one credit for a response that it may still deliver. The first one
// creates an event in its run and one in its AfterAgentCallback, and the
// second one creates an event in its run after the first one retired. root
// then delivers the held events. Each retired activation's events settle its
// own credit once, so the llmagent still awaits its model event afterwards,
// and its own answer is placed under it.
func TestRetiredPeerRunAndAfterAgentEventsSettleOneCredit_Runner(t *testing.T) {
	run := func(t *testing.T, firstRunEvent bool) int {
		t.Helper()
		tm := newTraceManager()
		p, takeRows := newQueuedRowsPlugin(t, tm)
		tm.selfPlugin = weak.Make(p)
		paused, release := newSignal(), newSignal()
		defer release.fire()
		llm, err := llmagent.New(llmagent.Config{
			Name: "worker",
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				yield(&model.LLMResponse{Content: textContent("model", "llm worker answer"), TurnComplete: true}, nil)
			}},
			AfterModelCallbacks: []llmagent.AfterModelCallback{func(_ agent.Context, res *model.LLMResponse, _ error) (*model.LLMResponse, error) {
				if res != nil && !res.Partial {
					paused.fire()
					release.wait()
				}
				return nil, nil
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(worker): %v", err)
		}
		first, err := agent.New(agent.Config{
			Name: "worker",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if firstRunEvent {
						yield(textEvent(ctx, "worker", ctx.Branch(), "first worker answer"), nil)
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{func(agent.Context) (*genai.Content, error) {
				return textContent("model", "first worker summary"), nil
			}},
		})
		if err != nil {
			t.Fatalf("agent.New(first worker): %v", err)
		}
		second, err := agent.New(agent.Config{
			Name: "worker",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(textEvent(ctx, "worker", ctx.Branch(), "second worker answer"), nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(second worker): %v", err)
		}
		awaiting := -1
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{llm},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					items := make(chan runItem)
					done := make(chan struct{})
					defer close(done)
					var wg sync.WaitGroup
					wg.Add(1)
					go func() {
						wg.Wait()
						close(items)
					}()
					forwardRun(ctx, llm, 3, items, done, &wg)
					paused.wait()
					var held []*session.Event
					for _, a := range []agent.Agent{first, second} {
						for ev, runErr := range a.Run(ctx) {
							if runErr != nil {
								yield(nil, runErr)
								return
							}
							held = append(held, ev)
						}
					}
					for _, ev := range held {
						if !yield(ev, nil) {
							return
						}
					}
					tm.mu.Lock()
					inv := tm.invocations[ctx.InvocationID()]
					tm.mu.Unlock()
					if inv != nil {
						inv.mu.Lock()
						for _, rec := range inv.agentStack[1:] {
							if rec.name == "worker" {
								awaiting = rec.awaitingModelEvent
							}
						}
						inv.mu.Unlock()
					}
					release.fire()
					for it := range items {
						ok := yield(it.ev, it.err)
						close(it.ack)
						if !ok {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-retired-peer-credit", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := takeRows()
		var llmSpan string
		for _, r := range rowsOfType(rows, EventLLMRequest) {
			llmSpan, _ = r["parent_span_id"].(string)
		}
		for _, r := range rowsOfType(rows, EventAgentResponse) {
			if strings.Contains(fmt.Sprint(r["content"]), "llm worker answer") {
				if got, _ := r["span_id"].(string); llmSpan == "" || got != llmSpan {
					t.Errorf("llmagent's answer placed at %q, want its span %q", got, llmSpan)
				}
			}
		}
		wantInvocationCompleted(t, rows)
		return awaiting
	}
	t.Run("after-agent_event_only", func(t *testing.T) {
		if got := run(t, false); got != 1 {
			t.Fatalf("awaitingModelEvent of the llmagent once the held events were delivered = %d, want 1", got)
		}
	})
	t.Run("run_event_and_after-agent_event", func(t *testing.T) {
		if got := run(t, true); got != 1 {
			t.Fatalf("awaitingModelEvent of the llmagent once the held events were delivered = %d, want 1", got)
		}
	})
}

// actionsContext is a context whose struct holds the EventActions of an
// agent's after-agent callback context, as ADK's callback context does (see
// callbackAfterAgentActions).
type actionsContext struct {
	context.Context
	actions *session.EventActions
}

// TestGuardlessRetirementKeepsAfterAgentEntry retires, through popAgentSpan,
// the later of two concurrent activations of step under wf, an agent that
// forwards errors and holds no events, as a workflow agent does for the agent
// of a node: no custom ancestor guards the retired activation's entry. It
// retires without a response while the other one runs, so it leaves a
// credit, and its entry keeps the credit and the action maps of its
// after-agent callbacks from the start, before anything prunes the entries
// of step:
//
//   - its after-agent event without content settles its credit;
//   - its after-agent event with content, created before the other activation
//     retired holding a failure that it swallowed, is its own and does not
//     clear that failure.
//
// An activation that responded and retired with no other activation of step
// open keeps its entry and maps too while its callbacks may still run.
func TestGuardlessRetirementKeepsAfterAgentEntry(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	var ticks atomic.Int64
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time {
		return t0.Add(time.Duration(ticks.Add(1)) * time.Millisecond)
	})
	wfID := identity{invocationID: "inv-guardless", agentName: "wf"}
	stepID := identity{invocationID: "inv-guardless", agentName: "step"}
	k := llmKey(stepID)
	errChild := errors.New("step-1's child failure")
	setup := func(t *testing.T) (*traceManager, *invocationTrace, *spanRecord, *session.EventActions) {
		t.Helper()
		tm := newTraceManager()
		tm.pushAgentSpan(ctx, wfID)
		inv := tm.invocation(ctx, wfID)
		inv.mu.Lock()
		inv.agentStack[1].propagatesChildError = true
		inv.mu.Unlock()
		tm.pushAgentSpan(ctx, stepID)
		tm.pushAgentSpan(ctx, stepID)
		inv.mu.Lock()
		wf, step1, step2 := inv.agentStack[1], inv.agentStack[2], inv.agentStack[3]
		step1.hasAfterAgentCallbacks, step2.hasAfterAgentCallbacks = true, true
		inv.mu.Unlock()
		// popAgentSpan retires the latest activation, step-2.
		actions := &session.EventActions{StateDelta: map[string]any{}, ArtifactDelta: map[string]int64{}}
		tm.popAgentSpan(&actionsContext{Context: ctx, actions: actions}, stepID)
		inv.mu.Lock()
		defer inv.mu.Unlock()
		if inv.unrespondedRetired[k] != 1 {
			t.Fatalf("unrespondedRetired = %d once step-2 retired, want 1", inv.unrespondedRetired[k])
		}
		// The consequences below show what a missing entry does, so it is
		// reported without stopping.
		if idx := inv.findRetiredActivationIndexLocked(step2.spanID); idx < 0 {
			t.Error("step-2 retired without an entry")
		} else if ra := inv.retiredActivations[idx]; !ra.afterAgentCredit || !ra.peerRetirement || !ra.afterAgentActions.shares(eventActionMaps(&session.Event{Actions: *actions})) {
			t.Errorf("step-2's entry: credit %v, peer retirement %v, maps %+v; want true, true and the callback's maps", ra.afterAgentCredit, ra.peerRetirement, ra.afterAgentActions)
		}
		return tm, inv, wf, actions
	}
	afterAgentEvent := func(actions *session.EventActions, content bool) *session.Event {
		ev := session.NewEvent(ctx, wfID.invocationID)
		ev.Author = "step"
		ev.Actions = *actions
		if content {
			ev.Content = textContent("model", "step-2 summary")
		}
		return ev
	}
	t.Run("after-agent event without content settles the credit", func(t *testing.T) {
		tm, inv, _, actions := setup(t)
		actions.StateDelta["k"] = "v"
		tm.clearLastPartialForEvent(ctx, stepID, "step", "", afterAgentEvent(actions, false))
		inv.mu.Lock()
		defer inv.mu.Unlock()
		if inv.unrespondedRetired[k] != 0 {
			t.Fatalf("unrespondedRetired = %d after step-2's after-agent event, want 0", inv.unrespondedRetired[k])
		}
	})
	t.Run("responded activation keeps its maps while its callbacks may run", func(t *testing.T) {
		tm := newTraceManager()
		tm.pushAgentSpan(ctx, wfID)
		inv := tm.invocation(ctx, wfID)
		inv.mu.Lock()
		inv.agentStack[1].propagatesChildError = true
		inv.mu.Unlock()
		tm.pushAgentSpan(ctx, stepID)
		inv.mu.Lock()
		step := inv.agentStack[2]
		step.hasAfterAgentCallbacks, step.hasResponse, step.ownResponse = true, true, true
		inv.mu.Unlock()
		actions := &session.EventActions{StateDelta: map[string]any{}}
		tm.popAgentSpan(&actionsContext{Context: ctx, actions: actions}, stepID)
		inv.mu.Lock()
		defer inv.mu.Unlock()
		inv.pruneSettledRetiredActivationsLocked("step", "", "")
		if idx := inv.findRetiredActivationIndexLocked(step.spanID); idx < 0 {
			t.Fatal("step retired without an entry")
		} else if ra := inv.retiredActivations[idx]; !ra.afterAgentActions.shares(eventActionMaps(&session.Event{Actions: *actions})) || ra.afterAgentActionsRef == nil {
			t.Fatalf("step's entry keeps maps %+v (ref %v), want the callback's", ra.afterAgentActions, ra.afterAgentActionsRef != nil)
		}
	})
	t.Run("after-agent event with content keeps the other activation's failure", func(t *testing.T) {
		tm, inv, wf, actions := setup(t)
		ev := afterAgentEvent(actions, true)
		inv.mu.Lock()
		step1 := inv.agentStack[2]
		step1.hasResponse, step1.ownResponse = true, true
		step1.retiredCauses = []retiredCause{{err: errChild, name: "child", platformTime: t0}}
		inv.mu.Unlock()
		tm.popAgentSpan(ctx, stepID)
		tm.completeResponseForEvent(ctx, stepID, "step", "", nil, false, ev)
		inv.mu.Lock()
		defer inv.mu.Unlock()
		if !slices.ContainsFunc(wf.retiredCauses, func(c retiredCause) bool { return errors.Is(c.err, errChild) }) {
			t.Fatalf("step-2's after-agent event cleared step-1's swallowed failure: wf causes %v", wf.retiredCauses)
		}
	})
}

// TestRetiredAfterAgentEntriesStayBounded_Runner runs, under a custom root
// that stays open, many activations of a custom agent child with an
// AfterAgentCallback that returns nothing, and checks how many entries of
// child stay recorded and keep after-agent action maps once they all retired:
//
//   - own_goroutines: root runs each activation on a goroutine of its own and
//     does not wait for it, so nothing shows that its after-agent callbacks
//     returned while root runs. All but the latest few fold into one hull
//     that stands for their after-agent events (see
//     foldPendingAfterAgentLocked);
//   - silent_loop_iterations: a LoopAgent runs child, which yields nothing,
//     on its goroutine, so each next iteration shows that the earlier one's
//     callbacks returned. That an activation retired without a response
//     keeps nothing by itself then, so the entries compact as usual.
//
// The run completes either way.
func TestRetiredAfterAgentEntriesStayBounded_Runner(t *testing.T) {
	const runs = 30
	type counts struct {
		entries, withMaps, pendingHulls int
	}
	run := func(t *testing.T, loop bool) counts {
		t.Helper()
		tm := newTraceManager()
		p, takeRows := newQueuedRowsPlugin(t, tm)
		tm.selfPlugin = weak.Make(p)
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if !loop {
						yield(textEvent(ctx, "child", ctx.Branch(), fmt.Sprintf("child-%d progress", runNumber(ctx))), nil)
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{func(agent.Context) (*genai.Content, error) {
				return nil, nil
			}},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		sub := child
		if loop {
			sub, err = loopagent.New(loopagent.Config{AgentConfig: agent.Config{Name: "loop", SubAgents: []agent.Agent{child}}, MaxIterations: runs})
			if err != nil {
				t.Fatalf("loopagent.New(loop): %v", err)
			}
		}
		var got counts
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{sub},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if loop {
						for ev, runErr := range sub.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
					} else {
						done := make(chan struct{})
						defer close(done)
						for n := 1; n <= runs; n++ {
							items := make(chan runItem)
							var wg sync.WaitGroup
							wg.Add(1)
							go func() {
								wg.Wait()
								close(items)
							}()
							forwardRun(ctx, child, n, items, done, &wg)
							for it := range items {
								ok := yield(it.ev, it.err)
								close(it.ack)
								if !ok {
									return
								}
							}
						}
					}
					tm.mu.Lock()
					inv := tm.invocations[ctx.InvocationID()]
					tm.mu.Unlock()
					if inv == nil {
						return
					}
					inv.mu.Lock()
					defer inv.mu.Unlock()
					for _, ra := range inv.retiredActivations {
						if ra.swallower.name != "child" {
							continue
						}
						got.entries++
						if ra.afterAgentActions != (actionMaps{}) || ra.afterAgentActionsRef != nil {
							got.withMaps++
						}
						if ra.afterAgentPending {
							got.pendingHulls++
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-bounded-after-agent", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		wantInvocationCompleted(t, takeRows())
		return got
	}
	t.Run("own_goroutines", func(t *testing.T) {
		got := run(t, false)
		if got.entries != maxPendingAfterAgent || got.withMaps != maxPendingAfterAgent-1 || got.pendingHulls != 1 {
			t.Fatalf("after %d activations on goroutines of their own: %+v, want %d entries, %d with maps and one hull standing for the others", runs, got, maxPendingAfterAgent, maxPendingAfterAgent-1)
		}
	})
	t.Run("silent_loop_iterations", func(t *testing.T) {
		got := run(t, true)
		if got.entries == 0 || got.entries > 4 || got.withMaps > 1 || got.pendingHulls != 0 {
			t.Fatalf("after %d silent loop iterations: %+v, want 1 to 4 entries, at most the latest with maps, and no hull standing for pending events", runs, got)
		}
	})
}

// TestRetiredAuthorBranchResolution delivers the recovery of a retired
// activation of inner that ran on the branch par.inner, drained a failure and
// created the event in its run, with an empty Branch, as session.NewEvent
// leaves it, or with the enclosing branch par, as the workflow scheduler
// stamps it. No other activation of inner on another branch can have created
// it, so it is the retired activation's and clears the failure that it
// swallowed, whether no activation of inner is open or a later one on
// par.inner, which started after the event was created, is.
func TestRetiredAuthorBranchResolution(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	var clock atomic.Int64
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return at(int(clock.Load())) })
	rootID := identity{invocationID: "inv-branch-resolution", agentName: "root"}
	innerID := identity{invocationID: "inv-branch-resolution", agentName: "inner", branch: "par.inner"}
	errChild := errors.New("inner's child failure")
	for _, evBranch := range []string{"", "par"} {
		for _, laterOpen := range []bool{false, true} {
			t.Run(fmt.Sprintf("event branch %q, later activation open %v", evBranch, laterOpen), func(t *testing.T) {
				clock.Store(0)
				tm := newTraceManager()
				tm.pushAgentSpan(ctx, rootID)
				inv := tm.invocation(ctx, rootID)
				inv.mu.Lock()
				root := inv.agentStack[1]
				sw := retiredSwallower{spanID: "inner-1", seq: inv.nextSeqLocked(), name: "inner", branch: "par.inner"}
				root.retiredCauses = []retiredCause{{err: errChild, name: "child", branch: "par.inner.ipar.child", platformTime: at(1), causeSeq: inv.nextSeqLocked(), swallowers: []retiredSwallower{sw}}}
				inv.retiredActivations = []retiredActivation{{
					swallower:             sw,
					parentSpanID:          "par-1",
					guardSpanIDs:          []string{root.spanID},
					ancestorSpanIDs:       []string{"par-1", root.spanID},
					retiredAtSeq:          inv.nextSeqLocked(),
					completedOrTask:       true,
					joinedAtRetire:        true,
					startPlatformTime:     at(0),
					retiredAtPlatformTime: at(3),
				}}
				inv.mu.Unlock()
				if laterOpen {
					clock.Store(5)
					tm.pushAgentSpan(ctx, innerID)
				}
				tm.completeResponseWithEvent(ctx, identity{invocationID: rootID.invocationID, agentName: "root"}, "inner", evBranch, nil, false, at(2), true, true, actionMaps{}, true)
				inv.mu.Lock()
				defer inv.mu.Unlock()
				if len(root.retiredCauses) != 0 {
					t.Fatalf("root still holds %d causes, want inner's recovery to clear its swallowed failure", len(root.retiredCauses))
				}
				if laterOpen {
					if later := inv.agentStack[len(inv.agentStack)-1]; later.name != "inner" || later.ownResponse {
						t.Fatalf("later activation %s took the retired activation's recovery as its own response", later.name)
					}
				}
			})
		}
	}
}

// TestRetiredEventBranch resolves the branch of the activations of inner that
// can have created an event that carries the enclosing branch par, or no
// branch, when no activation of inner is open (see resolveEventBranchLocked):
// that of the only branch whose activations can have created it, none when
// those on two branches can have, and the event's own when none can have. An
// activation on par itself is one of those branches: the workflow scheduler
// stamps par on an event that an activation under par left without a branch,
// so an event that carries par can be either's, and it is no one's when an
// activation on par and one under it can both have created it. One under par
// that cannot have created it, as one that retired before the event was
// created and has no after-agent callbacks, or one that started after it,
// leaves the event to the activation on par.
func TestRetiredEventBranch(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	retired := func(b string, start, retiredAt time.Time) retiredActivation {
		return retiredActivation{
			swallower:              retiredSwallower{spanID: "inner-" + b, name: "inner", branch: b},
			hasAfterAgentCallbacks: true,
			startPlatformTime:      start,
			retiredAtPlatformTime:  retiredAt,
		}
	}
	for _, tc := range []struct {
		name     string
		evBranch string
		retired  []retiredActivation
		want     string
		wantOK   bool
	}{
		{"two branches under par", "par", []retiredActivation{retired("par.inner", at(0), at(5)), retired("par.other", at(0), at(5))}, "", false},
		{"one branch under par", "par", []retiredActivation{retired("par.inner", at(0), at(5))}, "par.inner", true},
		{"par and a branch under it", "par", []retiredActivation{retired("par.inner", at(0), at(5)), retired("par", at(0), at(5))}, "", false},
		{"par alone", "par", []retiredActivation{retired("par", at(0), at(5))}, "par", true},
		{"par and one under it that retired before the event without after-agent callbacks", "par", []retiredActivation{func() retiredActivation {
			ra := retired("par.inner", at(0), at(1))
			ra.hasAfterAgentCallbacks = false
			return ra
		}(), retired("par", at(0), at(5))}, "par", true},
		{"par and one under it that started after the event", "par", []retiredActivation{retired("par.inner", at(3), at(5)), retired("par", at(0), at(5))}, "par", true},
		{"no candidates", "par", nil, "par", true},
		{"no branch, root branch and a branch under it", "", []retiredActivation{retired("", at(0), at(5)), retired("par.inner", at(0), at(5))}, "", false},
		{"no branch, root branch alone", "", []retiredActivation{retired("", at(0), at(5))}, "", true},
		{"no branch, one branch", "", []retiredActivation{retired("par.inner", at(0), at(5))}, "par.inner", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}}, retiredActivations: tc.retired}
			i, got, ok := inv.resolveEventBranchLocked("", "inner", tc.evBranch, at(2), true, true, actionMaps{})
			if i > 0 || got != tc.want || ok != tc.wantOK {
				t.Fatalf("resolveEventBranchLocked = %d, %q, %v; want no open activation, %q, %v", i, got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

// TestSettleRetiredResponseCredit settles the credits that retired
// activations of worker left, for events that a retired activation created in
// its run: such an event settles its own activation's credit, and none once
// that credit was settled, while an event that more than one activation, or
// none, can have created settles one credit of the agent.
func TestSettleRetiredResponseCredit(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	k := llmKey(identity{agentName: "worker"})
	setup := func() *invocationTrace {
		inv := &invocationTrace{unrespondedRetired: map[string]int{k: 2}}
		inv.retiredActivations = []retiredActivation{
			{swallower: retiredSwallower{spanID: "w1", name: "worker"}, afterAgentCredit: true, joinedAtRetire: true, startPlatformTime: at(0), retiredAtPlatformTime: at(2)},
			{swallower: retiredSwallower{spanID: "w2", name: "worker"}, afterAgentCredit: true, joinedAtRetire: true, startPlatformTime: at(3), retiredAtPlatformTime: at(5)},
		}
		return inv
	}
	t.Run("own credit, then none", func(t *testing.T) {
		inv := setup()
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(1), true, true) {
			t.Fatal("w1's run event settled no credit")
		}
		if ra := inv.retiredActivations[0]; ra.afterAgentCredit || !ra.creditSettled || inv.unrespondedRetired[k] != 1 || !inv.retiredActivations[1].afterAgentCredit {
			t.Fatalf("after w1's run event: w1 credit %v settled %v, w2 credit %v, unrespondedRetired %d; want false, true, true, 1", ra.afterAgentCredit, ra.creditSettled, inv.retiredActivations[1].afterAgentCredit, inv.unrespondedRetired[k])
		}
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(2), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("w1's second event settled a credit: unrespondedRetired %d, want 1", inv.unrespondedRetired[k])
		}
		// w1's after-agent event finds its credit settled too.
		inv.consumeAfterAgentCreditLocked(0)
		if inv.unrespondedRetired[k] != 1 {
			t.Fatalf("w1's after-agent event settled w2's credit: unrespondedRetired %d, want 1", inv.unrespondedRetired[k])
		}
	})
	for _, tc := range []struct {
		name       string
		evAt       time.Time
		hasEventID bool
	}{
		{"created while both could have", at(4), false},
		{"created before both started", t0.Add(-time.Millisecond), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inv := setup()
			inv.retiredActivations[1].startPlatformTime = at(1)
			if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", tc.evAt, tc.hasEventID, false) || inv.unrespondedRetired[k] != 1 {
				t.Fatalf("unrespondedRetired = %d, want 1", inv.unrespondedRetired[k])
			}
			if !inv.retiredActivations[0].afterAgentCredit || !inv.retiredActivations[1].afterAgentCredit {
				t.Fatal("an event that is no one activation's settled an activation's own credit")
			}
		})
	}
	t.Run("no credit outstanding", func(t *testing.T) {
		inv := setup()
		inv.unrespondedRetired[k] = 0
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(1), true, true) || inv.unrespondedRetired[k] != 0 || !inv.retiredActivations[0].afterAgentCredit {
			t.Fatal("an event settled a credit while none was outstanding")
		}
	})
	t.Run("createdInRun", func(t *testing.T) {
		ra := retiredActivation{completedOrTask: true, joinedAtRetire: true, startPlatformTime: at(1), retiredAtPlatformTime: at(3)}
		for _, c := range []struct {
			evAt       time.Time
			hasEventID bool
			want       bool
		}{
			{at(0), true, false},
			{at(1), true, true},
			{at(3), true, true},
			{at(4), true, false},
			{at(4), false, true},
			{time.Time{}, true, true},
		} {
			if got := ra.createdInRun(c.evAt, c.hasEventID); got != c.want {
				t.Errorf("createdInRun(%v, %v) = %v, want %v", c.evAt.Sub(t0), c.hasEventID, got, c.want)
			}
		}
		ra.joinedAtRetire = false
		if !ra.createdInRun(at(4), true) {
			t.Error("an unjoined finish_task retirement's run may outlast it, but createdInRun(after retirement) = false")
		}
	})
}

// TestPendingAfterAgentHull folds the entries of child under an open custom
// root that keep their after-agent action maps only because their callbacks
// may still run into a hull once there are more than maxPendingAfterAgent of
// them. The hull stands for their after-agent events: an event created after
// they retired may be one, so a later activation's failure is not cleared by
// it, and the open activation that root may hold events for clears none with
// its own events either. Once root's run returned, the hull no longer stands
// for them.
func TestPendingAfterAgentHull(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	setup := func(t *testing.T) (*invocationTrace, *spanRecord) {
		t.Helper()
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		for n := range maxPendingAfterAgent + 2 {
			m := map[string]any{}
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("child-%d", n), seq: uint64(10 + n), name: "child", path: fmt.Sprintf("p%d", n%2)},
				parentSpanID:           root.spanID,
				guardSpanIDs:           []string{root.spanID},
				retiredAtSeq:           uint64(100 + n),
				completedOrTask:        true,
				joinedAtRetire:         true,
				hasAfterAgentCallbacks: true,
				afterAgentActions:      actionMaps{stateDelta: reflect.ValueOf(m).Pointer()},
				afterAgentActionsRef:   m,
				startPlatformTime:      at(n),
				retiredAtPlatformTime:  at(n + 1),
			})
		}
		return inv, root
	}
	inv, root := setup(t)
	inv.pruneSettledRetiredActivationsLocked("child", "", "")
	var hull *retiredActivation
	for idx := range inv.retiredActivations {
		if ra := &inv.retiredActivations[idx]; ra.afterAgentPending {
			if hull != nil {
				t.Fatal("more than one hull")
			}
			hull = ra
		}
	}
	if len(inv.retiredActivations) != maxPendingAfterAgent || hull == nil {
		t.Fatalf("%d entries, hull %v; want %d with one hull", len(inv.retiredActivations), hull != nil, maxPendingAfterAgent)
	}
	if !hull.compactedHull || !hull.holdsAfterAgentEvent() || hull.afterAgentActionsRef != nil || hull.swallower.path != "" || hull.swallower.spanID != "child-2" || hull.retiredAtSeq != 102 || !hull.startPlatformTime.Equal(at(0)) || !hull.retiredAtPlatformTime.Equal(at(3)) {
		t.Fatalf("hull = %+v, want the three oldest entries on any path, without maps", *hull)
	}
	// A later activation of child holds a failure and created an event
	// after the hull's members retired.
	later := &spanRecord{spanID: "child-later", parentSpanID: root.spanID, parent: root, kind: spanKindAgent, name: "child", path: "p0", seq: 200, startPlatformTime: at(20)}
	if hull.excludedByEventTimestamp(later.spanID, later.seq, 0, false, later.startPlatformTime, at(21), true) {
		t.Fatal("a later activation excluded the hull by timestamp")
	}
	inv.agentStack = append(inv.agentStack, later)
	if !inv.hasGuardedRetiredPeerForEventLocked(later, "p0", at(21), true) {
		t.Fatal("the open activation's own event clears its failure while the hull may stand for a held after-agent event")
	}
	if author, ok := inv.eventAuthorLocked("child", "", "p0", at(21), true, true, actionMaps{}, true); ok {
		t.Fatalf("event author = %+v, want none: the hull or the open activation may have created it", author)
	}
	// Once root's run returned, the hull's members' after-agent events, if
	// any can still be delivered, were created, and nothing else needs it.
	root.completed = true
	inv.agentStack = inv.agentStack[:1]
	inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(30), root)
	for _, ra := range inv.retiredActivations {
		if ra.holdsAfterAgentEvent() {
			t.Fatalf("entry %s still holds an after-agent event once root retired: %+v", ra.swallower.spanID, ra)
		}
	}
}

// TestAfterAgentCallbacksReturnedOn records that the after-agent callbacks of
// the activations that retired on a goroutine returned once an agent starts
// or ends on it, and drops their action maps unless something still needs
// them, such as the activation's outstanding credit.
func TestAfterAgentCallbacksReturnedOn(t *testing.T) {
	m1, m2, m3, m4 := map[string]any{}, map[string]any{}, map[string]any{}, map[string]any{}
	mapsOf := func(m map[string]any) actionMaps { return actionMaps{stateDelta: reflect.ValueOf(m).Pointer()} }
	root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
	inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}, unrespondedRetired: map[string]int{llmKey(identity{agentName: "worker"}): 1}}
	inv.retiredActivations = []retiredActivation{
		{swallower: retiredSwallower{spanID: "a", name: "child"}, guardSpanIDs: []string{root.spanID}, popGoroutineID: 7, afterAgentActions: mapsOf(m1), afterAgentActionsRef: m1},
		{swallower: retiredSwallower{spanID: "b", name: "worker"}, guardSpanIDs: []string{root.spanID}, popGoroutineID: 7, afterAgentCredit: true, afterAgentActions: mapsOf(m2), afterAgentActionsRef: m2},
		{swallower: retiredSwallower{spanID: "c", name: "other"}, guardSpanIDs: []string{root.spanID}, popGoroutineID: 8, afterAgentActions: mapsOf(m3), afterAgentActionsRef: m3},
		{swallower: retiredSwallower{spanID: "d", name: "worker"}, guardSpanIDs: []string{root.spanID}, popGoroutineID: 7, afterAgentActions: mapsOf(m4), afterAgentActionsRef: m4},
	}
	inv.afterAgentCallbacksReturnedOnLocked(0, time.Time{}, 0)
	for _, ra := range inv.retiredActivations {
		if ra.afterAgentDone {
			t.Fatal("an unknown goroutine showed that callbacks returned")
		}
	}
	inv.afterAgentCallbacksReturnedOnLocked(7, time.Unix(1_700_000_000, 0), 0)
	a, b, c := inv.retiredActivations[0], inv.retiredActivations[1], inv.retiredActivations[2]
	if !a.afterAgentDone || !b.afterAgentDone || c.afterAgentDone {
		t.Fatalf("afterAgentDone = %v, %v, %v; want true, true, false", a.afterAgentDone, b.afterAgentDone, c.afterAgentDone)
	}
	d := inv.retiredActivations[3]
	if a.holdsAfterAgentEvent() || !b.holdsAfterAgentEvent() || !c.holdsAfterAgentEvent() || !d.holdsAfterAgentEvent() {
		t.Fatalf("maps kept = %v, %v, %v, %v; want false (settled), true (own credit outstanding), true (callbacks may run), true (another activation's credit outstanding)", a.holdsAfterAgentEvent(), b.holdsAfterAgentEvent(), c.holdsAfterAgentEvent(), d.holdsAfterAgentEvent())
	}
}

// TestSettledAfterAgentEventPlacement places the rows of an event that
// completeResponse took for a retired activation's after-agent event under
// the invocation, as settledAfterAgent reports, even though an open
// activation of the same agent that started earlier would own its timestamp,
// and leaves other rows, and rows that a callback places meanwhile, where
// they were.
func TestSettledAfterAgentEventPlacement(t *testing.T) {
	ctx := t.Context()
	rootID := identity{invocationID: "inv-settled-placement", agentName: "root"}
	innerID := identity{invocationID: "inv-settled-placement", agentName: "inner"}
	tm := newTraceManager()
	tm.pushAgentSpan(ctx, rootID)
	tm.pushAgentSpan(ctx, innerID)
	inv := tm.invocation(ctx, rootID)
	inv.mu.Lock()
	invSpan, innerSpan := inv.agentStack[0].spanID, inv.agentStack[2].spanID
	inv.mu.Unlock()
	if got := tm.eventSpan(ctx, innerID, "inner", "").spanID; got != innerSpan {
		t.Fatalf("a callback's row placed at %q, want inner's span %q", got, innerSpan)
	}
	if got := tm.eventSpanAt(ctx, innerID, "inner", "", time.Time{}, false, false, false).spanID; got != innerSpan {
		t.Fatalf("another event's row placed at %q, want inner's span %q", got, innerSpan)
	}
	if got := tm.eventSpanAt(ctx, innerID, "inner", "", time.Time{}, false, false, true).spanID; got != invSpan {
		t.Fatalf("the settled after-agent event's row placed at %q, want the invocation span %q", got, invSpan)
	}
	if got := tm.eventSpan(ctx, innerID, "inner", "").spanID; got != innerSpan {
		t.Fatalf("a callback's row placed after the settled event at %q, want inner's span %q", got, innerSpan)
	}
}

// TestAfterAgentDoneBoundsRetiredEvents pins that a retired activation whose
// after-agent callbacks were seen to have returned at afterAgentDoneAt can
// own no event created after that time (see canOwnEventCreatedAt), so that
// neither hasGuardedRetiredPeerForEventLocked nor soleOpenAuthorLocked takes
// it for a possible author of an open activation's later event, whether its
// after-agent event was delivered unrecognized or never came, while an event
// that carries its after-agent action maps stays its own whenever it was
// created, as when the callbacks ran an agent themselves. It also pins how
// that time is recorded and merged.
func TestAfterAgentDoneBoundsRetiredEvents(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	retired := func(spanID string) retiredActivation {
		return retiredActivation{
			swallower:              retiredSwallower{spanID: spanID, seq: 2, name: "worker"},
			parentSpanID:           "root-1",
			guardSpanIDs:           []string{"root-1"},
			retiredAtSeq:           3,
			completedOrTask:        true,
			joinedAtRetire:         true,
			unrespondedAtPop:       true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(0),
			retiredAtPlatformTime:  at(10),
		}
	}
	t.Run("canOwnEventCreatedAt", func(t *testing.T) {
		ra := retired("w-1")
		if !ra.canOwnEventCreatedAt(at(30), true) {
			t.Fatal("an activation whose after-agent event may still come cannot own a later event")
		}
		ra.markAfterAgentDone(at(20))
		for _, c := range []struct {
			ev   time.Time
			want bool
		}{{at(15), true}, {at(20), true}, {at(30), false}} {
			if got := ra.canOwnEventCreatedAt(c.ev, true); got != c.want {
				t.Errorf("canOwnEventCreatedAt(%v) with callbacks seen to return at 20ms = %v, want %v", c.ev.Sub(t0), got, c.want)
			}
		}
		if !ra.canOwnEventAt(at(30), true, true) {
			t.Error("an event that carries the activation's after-agent maps is not its own once created after the callbacks were seen to return")
		}
		ra.markAfterAgentDone(at(25))
		if !ra.afterAgentDoneAt.Equal(at(20)) {
			t.Errorf("afterAgentDoneAt = %v after a later sighting, want the earliest, 20ms", ra.afterAgentDoneAt.Sub(t0))
		}
		ra.markAfterAgentDone(at(15))
		ra.markAfterAgentDone(time.Time{})
		if !ra.afterAgentDone || !ra.afterAgentDoneAt.Equal(at(15)) {
			t.Errorf("afterAgentDone %v at %v, want true at the earliest known time, 15ms", ra.afterAgentDone, ra.afterAgentDoneAt.Sub(t0))
		}
	})
	t.Run("recorded where the callbacks are seen to return", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top"}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top}}
		on7, on8 := retired("w-1"), retired("w-2")
		on7.popGoroutineID, on8.popGoroutineID = 7, 8
		// root-1 runs both under the open custom agent top.
		on7.guardSpanIDs, on8.guardSpanIDs = []string{"root-1", top.spanID}, []string{"root-1", top.spanID}
		inv.retiredActivations = []retiredActivation{on7, on8}
		inv.afterAgentCallbacksReturnedOnLocked(7, at(20), 0)
		if got := inv.retiredActivations[0].afterAgentDoneAt; !got.Equal(at(20)) {
			t.Errorf("afterAgentDoneAt of the activation retired on the goroutine = %v, want 20ms", got.Sub(t0))
		}
		if got := inv.retiredActivations[1]; got.afterAgentDone || !got.afterAgentDoneAt.IsZero() {
			t.Errorf("activation retired on another goroutine: done %v at %v, want neither", got.afterAgentDone, got.afterAgentDoneAt)
		}
		// The parent's run returned once it retired.
		parent := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", completed: true}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(40), parent)
		if idx := inv.findRetiredActivationIndexLocked("w-2"); idx < 0 || !inv.retiredActivations[idx].afterAgentDoneAt.Equal(at(40)) {
			t.Fatalf("afterAgentDoneAt once the parent's run returned at 40ms: entry %d %+v", idx, inv.retiredActivations)
		}
	})
	t.Run("hulls keep the latest time of their members, and none when one has none", func(t *testing.T) {
		for _, oneUnknown := range []bool{false, true} {
			inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, {spanID: "root-1", kind: spanKindAgent, name: "root"}}}
			for n := range maxPendingAfterAgent + 2 {
				m := map[string]any{}
				ra := retired(fmt.Sprintf("w-%d", n))
				ra.retiredAtSeq = uint64(10 + n)
				ra.afterAgentActions, ra.afterAgentActionsRef = actionMaps{stateDelta: reflect.ValueOf(m).Pointer()}, m
				ra.markAfterAgentDone(at(20 + n))
				if oneUnknown && n == 1 {
					ra.afterAgentDoneAt = time.Time{}
				}
				inv.retiredActivations = append(inv.retiredActivations, ra)
			}
			inv.foldPendingAfterAgentLocked("worker", "", afterAgentState{swallowers: map[string]bool{}, underOpen: map[string]bool{}})
			hull := inv.retiredActivations[slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.afterAgentPending })]
			want := at(22)
			if oneUnknown {
				want = time.Time{}
			}
			if !hull.afterAgentDone || !hull.afterAgentDoneAt.Equal(want) {
				t.Errorf("pending hull of the three oldest (one without a time: %v): done %v at %v, want true at %v", oneUnknown, hull.afterAgentDone, hull.afterAgentDoneAt, want)
			}
		}
		for _, oneUnknown := range []bool{false, true} {
			inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, {spanID: "root-1", kind: spanKindAgent, name: "root"}}}
			for n := range 5 {
				ra := retired(fmt.Sprintf("w-%d", n))
				ra.hasAfterAgentCallbacks, ra.unrespondedAtPop = false, false
				ra.retiredAtSeq = uint64(10 + n)
				ra.markAfterAgentDone(at(20 + n))
				if oneUnknown && n == 2 {
					ra.afterAgentDoneAt = time.Time{}
				}
				inv.retiredActivations = append(inv.retiredActivations, ra)
			}
			inv.pruneSettledRetiredActivationsLocked("worker", "", "")
			idx := slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.compactedHull })
			if len(inv.retiredActivations) != 3 || idx < 0 {
				t.Fatalf("%d entries after compaction, hull at %d; want 3 with one hull", len(inv.retiredActivations), idx)
			}
			want := at(23)
			if oneUnknown {
				want = time.Time{}
			}
			if hull := inv.retiredActivations[idx]; !hull.afterAgentDoneAt.Equal(want) {
				t.Errorf("compacted hull of the three middle entries (one without a time: %v): done at %v, want %v", oneUnknown, hull.afterAgentDoneAt, want)
			}
		}
	})
	t.Run("an open activation's later event", func(t *testing.T) {
		for _, done := range []bool{false, true} {
			root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root"}
			peer := &spanRecord{spanID: "w-2", kind: spanKindAgent, name: "worker", parent: root, parentSpanID: root.spanID, seq: 4, startPlatformTime: at(25)}
			inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root, peer}}
			ra := retired("w-1")
			if done {
				ra.markAfterAgentDone(at(20))
			}
			inv.retiredActivations = []retiredActivation{ra}
			if got := inv.hasGuardedRetiredPeerForEventLocked(peer, "", at(30), true); got == done {
				t.Errorf("callbacks seen to return before the event %v: hasGuardedRetiredPeerForEventLocked = %v, want %v", done, got, !done)
			}
			want := -1
			if done {
				want = 2
			}
			if got := inv.soleOpenAuthorLocked([]int{2}, "", at(30), func(*spanRecord) bool { return true }); got != want {
				t.Errorf("callbacks seen to return before the event %v: soleOpenAuthorLocked = %d, want %d", done, got, want)
			}
		}
	})
	t.Run("an after-agent event created after the callbacks were seen to return", func(t *testing.T) {
		m := map[string]any{}
		ra := retired("w-1")
		ra.afterAgentActions, ra.afterAgentActionsRef = actionMaps{stateDelta: reflect.ValueOf(m).Pointer()}, m
		ra.markAfterAgentDone(at(20))
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}}, retiredActivations: []retiredActivation{ra}}
		if author, ok := inv.eventAuthorLocked("worker", "", "", at(30), true, true, ra.afterAgentActions, true); !ok || author.idx != 0 {
			t.Errorf("event with the activation's maps: author %+v, %v; want the activation", author, ok)
		}
		if author, ok := inv.eventAuthorLocked("worker", "", "", at(30), true, true, actionMaps{stateDelta: 1}, true); ok {
			t.Errorf("event without the activation's maps: author %+v, want none", author)
		}
	})
}

// TestAfterAgentEventOwnershipSettlesCredit pins that an event that carries a
// map of the context in which a retired activation's after-agent callbacks
// ran settles that activation's credit, and is placed as its after-agent
// event (settledAfterAgent), only when that activation can own it
// (see ownsAfterAgentEvent): with creation provenance and a timestamp at
// which the activation can have created it. An event that carries the map
// but fails either test, or that carries maps of two retired activations, is
// absorbed and settles nothing, whether an open peer runs
// (selectResponseOwnerWithIDLocked) or none does (completeResponseWithEvent).
func TestAfterAgentEventOwnershipSettlesCredit(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	k := llmKey(identity{agentName: "worker"})
	m1, m2 := map[string]any{}, map[string]any{}
	maps1 := actionMaps{stateDelta: reflect.ValueOf(m1).Pointer()}
	entry := func(spanID string, m map[string]any) retiredActivation {
		return retiredActivation{
			swallower:              retiredSwallower{spanID: spanID, name: "worker"},
			guardSpanIDs:           []string{"root-1"},
			hasAfterAgentCallbacks: true,
			afterAgentCredit:       true,
			peerRetirement:         true,
			completedOrTask:        true,
			joinedAtRetire:         true,
			startPlatformTime:      at(2),
			retiredAtPlatformTime:  at(5),
			afterAgentActions:      actionMaps{stateDelta: reflect.ValueOf(m).Pointer()},
			afterAgentActionsRef:   m,
		}
	}
	for _, tc := range []struct {
		name              string
		ev                time.Time
		hasID, nonEmptyID bool
		customUUID        bool
		twoSharers        bool
		want              bool
	}{
		{"owned", at(6), true, true, false, false, true},
		{"non-UUID ID of an activation with a custom UUID provider", at(6), false, true, true, false, true},
		{"without creation provenance", at(6), false, false, false, false, false},
		{"non-UUID ID of an activation without a custom UUID provider", at(6), false, true, false, false, false},
		{"created before the activation started", at(1), true, true, false, false, false},
		{"maps of two retired activations", at(6), true, true, false, true, false},
	} {
		setup := func(inv *invocationTrace) {
			inv.unrespondedRetired = map[string]int{k: 2}
			ra := entry("w-1", m1)
			ra.hasCustomUUIDProvider = tc.customUUID
			inv.retiredActivations = []retiredActivation{ra}
			if tc.twoSharers {
				other := entry("w-3", m2)
				other.afterAgentActions.stateDelta = maps1.stateDelta
				inv.retiredActivations = append(inv.retiredActivations, other)
			}
		}
		check := func(t *testing.T, inv *invocationTrace, settled bool) {
			t.Helper()
			wantCredits := 2
			if tc.want {
				wantCredits = 1
			}
			idx := inv.findRetiredActivationIndexLocked("w-1")
			if idx < 0 {
				t.Fatal("w-1 lost its entry")
			}
			if ra := inv.retiredActivations[idx]; inv.unrespondedRetired[k] != wantCredits || ra.afterAgentCredit == tc.want || ra.creditSettled != tc.want {
				t.Errorf("unrespondedRetired %d, w-1 credit %v settled %v; want %d, %v, %v", inv.unrespondedRetired[k], ra.afterAgentCredit, ra.creditSettled, wantCredits, !tc.want, tc.want)
			}
			if settled != tc.want {
				t.Errorf("event placed as the after-agent event: %v, want %v", settled, tc.want)
			}
		}
		t.Run("open peer/"+tc.name, func(t *testing.T) {
			root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root"}
			peer := &spanRecord{spanID: "w-2", kind: spanKindAgent, name: "worker", parent: root, parentSpanID: root.spanID, concurrentPeer: true, startPlatformTime: at(0)}
			inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root, peer}}
			setup(inv)
			_, _, absorbed, retiredAuthor, settled := inv.selectResponseOwnerWithIDLocked(2, "", nil, false, false, "", at(7), tc.ev, tc.hasID, tc.nonEmptyID, maps1)
			if !absorbed || !retiredAuthor {
				t.Fatalf("absorbed %v, retired author %v; want both", absorbed, retiredAuthor)
			}
			check(t, inv, settled)
		})
		t.Run("no open activation/"+tc.name, func(t *testing.T) {
			ctx := t.Context()
			tm := newTraceManager()
			rootID := identity{invocationID: "inv-after-agent-ownership", agentName: "root"}
			tm.pushAgentSpan(ctx, rootID)
			inv := tm.invocation(ctx, rootID)
			inv.mu.Lock()
			inv.agentStack[1].spanID = "root-1"
			setup(inv)
			inv.mu.Unlock()
			settled := tm.completeResponseWithEvent(ctx, rootID, "worker", "", nil, false, tc.ev, tc.hasID, tc.nonEmptyID, maps1, true)
			inv.mu.Lock()
			defer inv.mu.Unlock()
			check(t, inv, settled)
		})
	}
}

// TestPeerRetirementsStayUncompactedWhilePeerOpen retires concurrent peers
// of an open activation of worker under an open custom root. Their entries
// are neither deleted nor compacted while the peer is open, so that
// soleOpenAuthorLocked still finds an entry for every retired peer that
// retiredPeerCount counts and gives the open activation its event created
// after they all retired. Once no peer is open, they compact as usual.
//
// That holds when more than maxPendingAfterAgent of them keep the maps of
// their after-agent callbacks, whether those callbacks may still run or were
// seen to return: foldPendingAfterAgentLocked folds none of them into a hull
// while the peer is open. Nor does it fold, while the peer is open, entries
// that retired before it started and keep their maps because their callbacks
// returned and the open peer could be taken for the author of their
// after-agent events. Once no peer is open, the entries whose callbacks may
// still run fold as usual, and the others lose their maps and compact.
func TestPeerRetirementsStayUncompactedWhilePeerOpen(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	k := llmKey(identity{agentName: "worker"})
	type kind int
	const (
		plain kind = iota
		pendingMaps
		returnedMaps
		returnedBeforePeer
	)
	setup := func(n int, kd kind) *invocationTrace {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root"}
		peer := &spanRecord{spanID: "w-open", kind: spanKindAgent, name: "worker", parent: root, parentSpanID: root.spanID, concurrentPeer: true, seq: 1, startPlatformTime: at(0)}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root, peer}, retiredPeerCount: map[string]int{}}
		for i := range n {
			ra := retiredActivation{
				swallower:             retiredSwallower{spanID: fmt.Sprintf("w-%d", i), seq: uint64(2 + i), name: "worker"},
				parentSpanID:          root.spanID,
				guardSpanIDs:          []string{root.spanID},
				retiredAtSeq:          uint64(100 + i),
				completedOrTask:       true,
				joinedAtRetire:        true,
				concurrentPeer:        kd != returnedBeforePeer,
				peerRetirement:        kd != returnedBeforePeer,
				startPlatformTime:     at(1 + i),
				retiredAtPlatformTime: at(2 + i),
			}
			if kd != plain {
				m := map[string]any{}
				ra.hasAfterAgentCallbacks = true
				ra.afterAgentActions, ra.afterAgentActionsRef = actionMaps{stateDelta: reflect.ValueOf(m).Pointer()}, m
			}
			if kd == returnedMaps || kd == returnedBeforePeer {
				ra.markAfterAgentDone(at(2 + i))
			}
			if ra.peerRetirement {
				inv.retiredPeerCount[k]++
			}
			inv.retiredActivations = append(inv.retiredActivations, ra)
		}
		return inv
	}
	uncompacted := func(t *testing.T, inv *invocationTrace, n int, wantPeers bool) {
		t.Helper()
		if len(inv.retiredActivations) != n || slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			return ra.compactedHull || ra.afterAgentPending || ra.peerRetirement != wantPeers
		}) {
			t.Fatalf("entries while the peer is open: %+v, want the %d entries as they were", inv.retiredActivations, n)
		}
		if !inv.hasOpenPeerLocked(inv.retiredActivations[0]) {
			t.Fatal("hasOpenPeerLocked = false while the peer is open")
		}
	}
	t.Run("without after-agent maps", func(t *testing.T) {
		inv := setup(4, plain)
		inv.pruneSettledRetiredActivationsLocked("worker", "", "")
		uncompacted(t, inv, 4, true)
		if got := inv.soleOpenAuthorLocked([]int{2}, "", at(20), func(*spanRecord) bool { return true }); got != 2 {
			t.Fatalf("soleOpenAuthorLocked = %d, want the open peer, 2", got)
		}
		inv.agentStack = inv.agentStack[:2]
		if inv.hasOpenPeerLocked(inv.retiredActivations[0]) {
			t.Fatal("hasOpenPeerLocked = true once no peer is open")
		}
		inv.pruneSettledRetiredActivationsLocked("worker", "", "")
		if len(inv.retiredActivations) != 3 {
			t.Fatalf("%d entries once no peer is open, want 3 after compaction", len(inv.retiredActivations))
		}
	})
	for _, n := range []int{maxPendingAfterAgent + 1, 15} {
		t.Run(fmt.Sprintf("%d peer retirements whose callbacks may still run", n), func(t *testing.T) {
			inv := setup(n, pendingMaps)
			inv.pruneSettledRetiredActivationsLocked("worker", "", "")
			uncompacted(t, inv, n, true)
			inv.agentStack = inv.agentStack[:2]
			inv.pruneSettledRetiredActivationsLocked("worker", "", "")
			if len(inv.retiredActivations) != maxPendingAfterAgent || !slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.afterAgentPending }) {
				t.Fatalf("%d entries once no peer is open, want %d with one hull standing for the oldest", len(inv.retiredActivations), maxPendingAfterAgent)
			}
		})
		t.Run(fmt.Sprintf("%d peer retirements whose callbacks returned", n), func(t *testing.T) {
			inv := setup(n, returnedMaps)
			inv.pruneSettledRetiredActivationsLocked("worker", "", "")
			uncompacted(t, inv, n, true)
			if got := inv.soleOpenAuthorLocked([]int{2}, "", at(40), func(*spanRecord) bool { return true }); got != 2 {
				t.Fatalf("soleOpenAuthorLocked = %d, want the open peer, 2", got)
			}
			inv.agentStack = inv.agentStack[:2]
			inv.pruneSettledRetiredActivationsLocked("worker", "", "")
			if len(inv.retiredActivations) != 3 || slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.holdsAfterAgentEvent() }) {
				t.Fatalf("entries once no peer is open: %+v, want 3 after compaction, none with maps", inv.retiredActivations)
			}
		})
	}
	t.Run("entries that retired before the peer started and whose callbacks returned", func(t *testing.T) {
		n := maxPendingAfterAgent + 1
		inv := setup(n, returnedBeforePeer)
		inv.pruneSettledRetiredActivationsLocked("worker", "", "")
		uncompacted(t, inv, n, false)
		inv.agentStack = inv.agentStack[:2]
		inv.pruneSettledRetiredActivationsLocked("worker", "", "")
		if len(inv.retiredActivations) != 3 || slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.holdsAfterAgentEvent() }) {
			t.Fatalf("entries once no peer is open: %+v, want 3 after compaction, none with maps", inv.retiredActivations)
		}
	})
}

// TestResolveEventBranch pins how resolveEventBranchLocked resolves the
// branch of a session event of worker among open and retired activations:
// an event without a branch, or with an enclosing branch, is no one's when
// activations on that branch and on one under it can both have created it,
// and an activation that cannot have created it, such as one that started
// after it, does not count. When only a retired activation on one branch can
// have created it while a later activation runs on another branch,
// completeResponse settles the retired activation on its branch: the
// recovery clears the failure that it swallowed, and the later activation
// does not take the event.
func TestResolveEventBranch(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	open := func(spanID, branch string, start time.Time) *spanRecord {
		return &spanRecord{spanID: spanID, kind: spanKindAgent, name: "worker", branch: branch, startPlatformTime: start}
	}
	retired := func(branch string, callbacks bool, start, retiredAt time.Time) retiredActivation {
		return retiredActivation{
			swallower:              retiredSwallower{spanID: "w-" + branch, name: "worker", branch: branch},
			hasAfterAgentCallbacks: callbacks,
			joinedAtRetire:         true,
			startPlatformTime:      start,
			retiredAtPlatformTime:  retiredAt,
		}
	}
	for _, tc := range []struct {
		name         string
		evBranch     string
		open         []*spanRecord
		retired      []retiredActivation
		wantIdx      int
		wantBranch   string
		wantResolved bool
	}{
		{"open on the root branch and on a branch under it", "", []*spanRecord{open("w-root", "", at(5)), open("w-b1", "par.b1", at(5))}, nil, -1, "", false},
		{"open on the root branch, and on a branch under it since after the event", "", []*spanRecord{open("w-root", "", at(5)), open("w-b1", "par.b1", at(20))}, nil, 1, "", true},
		{"open on the root branch and retired under it", "", []*spanRecord{open("w-root", "", at(5))}, []retiredActivation{retired("par.b1", true, at(5), at(8))}, -1, "", false},
		{"open on the enclosing branch and retired under it", "par", []*spanRecord{open("w-par", "par", at(5))}, []retiredActivation{retired("par.inner", true, at(5), at(8))}, -1, "", false},
		{"open on the enclosing branch and retired under it before the event", "par", []*spanRecord{open("w-par", "par", at(5))}, []retiredActivation{retired("par.inner", false, at(5), at(8))}, 1, "par", true},
		{"retired on one branch while a later one runs on another", "", []*spanRecord{open("w-b2", "top.b2", at(20))}, []retiredActivation{retired("top.b1", false, at(5), at(12))}, -1, "top.b1", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inv := &invocationTrace{agentStack: append([]*spanRecord{{kind: spanKindInvocation}}, tc.open...), retiredActivations: tc.retired}
			i, got, ok := inv.resolveEventBranchLocked("", "worker", tc.evBranch, at(10), true, true, actionMaps{})
			if i != tc.wantIdx || got != tc.wantBranch || ok != tc.wantResolved {
				t.Fatalf("resolveEventBranchLocked = %d, %q, %v; want %d, %q, %v", i, got, ok, tc.wantIdx, tc.wantBranch, tc.wantResolved)
			}
		})
	}
	t.Run("completeResponse settles the retired activation on its branch", func(t *testing.T) {
		var clock atomic.Int64
		ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return at(int(clock.Load())) })
		tm := newTraceManager()
		rootID := identity{invocationID: "inv-resolve-branch", agentName: "root"}
		tm.pushAgentSpan(ctx, rootID)
		inv := tm.invocation(ctx, rootID)
		inv.mu.Lock()
		root := inv.agentStack[1]
		sw := retiredSwallower{spanID: "w-b1", seq: inv.nextSeqLocked(), name: "worker", branch: "top.b1"}
		root.retiredCauses = []retiredCause{{err: errors.New("b1's child failure"), name: "child", branch: "top.b1.b1par.child", platformTime: at(3), causeSeq: inv.nextSeqLocked(), swallowers: []retiredSwallower{sw}}}
		inv.retiredActivations = []retiredActivation{{
			swallower:             sw,
			parentSpanID:          root.spanID,
			guardSpanIDs:          []string{root.spanID},
			ancestorSpanIDs:       []string{root.spanID},
			retiredAtSeq:          inv.nextSeqLocked(),
			completedOrTask:       true,
			joinedAtRetire:        true,
			startPlatformTime:     at(1),
			retiredAtPlatformTime: at(12),
		}}
		inv.mu.Unlock()
		clock.Store(20)
		tm.pushAgentSpan(ctx, identity{invocationID: rootID.invocationID, agentName: "worker", branch: "top.b2"})
		tm.completeResponseWithEvent(ctx, rootID, "worker", "", nil, false, at(10), true, true, actionMaps{}, true)
		inv.mu.Lock()
		defer inv.mu.Unlock()
		if len(root.retiredCauses) != 0 {
			t.Fatalf("root still holds %d causes, want the recovery of worker on top.b1 to clear its swallowed failure", len(root.retiredCauses))
		}
		if b2 := inv.agentStack[len(inv.agentStack)-1]; b2.name != "worker" || b2.ownResponse || b2.hasResponse {
			t.Fatalf("worker on top.b2 took the retired activation's recovery: own %v, responded %v", b2.ownResponse, b2.hasResponse)
		}
	})
}

// TestAfterAgentSharerResolvesEventBranch pins that an event that carries an
// action map of exactly one retired activation's after-agent callbacks on
// the branches that the event's branch covers resolves to that activation's
// branch (see resolveEventBranchLocked), even when activations on another of
// those branches could have created an event at its timestamp, while the same
// event without the map is no one's, and that completeResponse then settles
// that activation's credit and clearLastPartialForEvent recognizes its
// after-agent event without content on an enclosing branch too.
func TestAfterAgentSharerResolvesEventBranch(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	m1, m2 := map[string]any{}, map[string]any{}
	maps1 := actionMaps{stateDelta: reflect.ValueOf(m1).Pointer()}
	open := func(spanID, branch string) *spanRecord {
		return &spanRecord{spanID: spanID, kind: spanKindAgent, name: "worker", branch: branch, startPlatformTime: at(1)}
	}
	retired := func(spanID, branch string, m map[string]any) retiredActivation {
		ra := retiredActivation{
			swallower:              retiredSwallower{spanID: spanID, name: "worker", branch: branch},
			hasAfterAgentCallbacks: true,
			completedOrTask:        true,
			joinedAtRetire:         true,
			startPlatformTime:      at(2),
			retiredAtPlatformTime:  at(5),
		}
		if m != nil {
			ra.afterAgentActions, ra.afterAgentActionsRef = actionMaps{stateDelta: reflect.ValueOf(m).Pointer()}, m
		}
		return ra
	}
	for _, tc := range []struct {
		name         string
		evBranch     string
		open         []*spanRecord
		retired      []retiredActivation
		withMaps     bool
		wantIdx      int
		wantBranch   string
		wantResolved bool
	}{
		{"root branch, peer open and a sub-branch open", "", []*spanRecord{open("a2", ""), open("b", "par.wb")}, []retiredActivation{retired("a1", "", m1)}, true, 1, "", true},
		{"root branch, peer open and a sub-branch open, without the map", "", []*spanRecord{open("a2", ""), open("b", "par.wb")}, []retiredActivation{retired("a1", "", m1)}, false, -1, "", false},
		{"enclosing branch and a sub-branch open", "outer.wa", []*spanRecord{open("b", "outer.wa.par.wb")}, []retiredActivation{retired("a", "outer.wa", m1)}, true, -1, "outer.wa", true},
		{"enclosing branch and a sub-branch open, without the map", "outer.wa", []*spanRecord{open("b", "outer.wa.par.wb")}, []retiredActivation{retired("a", "outer.wa", m1)}, false, -1, "", false},
		{"sharer on a sub-branch, event without a branch", "", []*spanRecord{open("w", "")}, []retiredActivation{retired("b", "par.wb", m1)}, true, -1, "par.wb", true},
		{"sharer on a sibling branch, event on another branch", "top.b2", []*spanRecord{open("w", "top.b2")}, []retiredActivation{retired("b1", "top.b1", m1)}, true, 1, "top.b2", true},
		{"two sharers on different branches", "", []*spanRecord{open("a2", ""), open("b", "par.wb")}, []retiredActivation{retired("a1", "", m1), retired("b1", "par.wb", m1)}, true, -1, "", false},
		{"a compacted hull's map", "", []*spanRecord{open("a2", ""), open("b", "par.wb")}, []retiredActivation{func() retiredActivation {
			ra := retired("a1", "", m1)
			ra.compactedHull = true
			return ra
		}()}, true, -1, "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inv := &invocationTrace{agentStack: append([]*spanRecord{{kind: spanKindInvocation}}, tc.open...), retiredActivations: tc.retired}
			evActions := actionMaps{stateDelta: reflect.ValueOf(m2).Pointer()}
			if tc.withMaps {
				evActions = maps1
			}
			i, got, ok := inv.resolveEventBranchLocked("", "worker", tc.evBranch, at(10), true, true, evActions)
			if i != tc.wantIdx || got != tc.wantBranch || ok != tc.wantResolved {
				t.Fatalf("resolveEventBranchLocked = %d, %q, %v; want %d, %q, %v", i, got, ok, tc.wantIdx, tc.wantBranch, tc.wantResolved)
			}
		})
	}
	k := llmKey(identity{agentName: "worker"})
	setup := func(t *testing.T) (*traceManager, *invocationTrace, identity) {
		t.Helper()
		ctx := t.Context()
		tm := newTraceManager()
		rootID := identity{invocationID: "inv-sharer-branch", agentName: "root"}
		tm.pushAgentSpan(ctx, rootID)
		tm.pushAgentSpan(ctx, identity{invocationID: rootID.invocationID, agentName: "worker"})
		tm.pushAgentSpan(ctx, identity{invocationID: rootID.invocationID, agentName: "worker", branch: "par.wb"})
		inv := tm.invocation(ctx, rootID)
		inv.mu.Lock()
		defer inv.mu.Unlock()
		for _, rec := range inv.agentStack[1:] {
			rec.startPlatformTime = at(1)
		}
		a1 := retired("a1", "", m1)
		a1.afterAgentCredit, a1.peerRetirement = true, true
		inv.retiredActivations = []retiredActivation{a1}
		inv.unrespondedRetired = map[string]int{k: 1}
		return tm, inv, rootID
	}
	t.Run("completeResponse settles the sharer's credit while a sub-branch runs", func(t *testing.T) {
		tm, inv, rootID := setup(t)
		if !tm.completeResponseWithEvent(t.Context(), rootID, "worker", "", nil, false, at(10), true, true, maps1, true) {
			t.Error("the after-agent event was not taken for a1's")
		}
		inv.mu.Lock()
		defer inv.mu.Unlock()
		ra := inv.retiredActivations[inv.findRetiredActivationIndexLocked("a1")]
		if inv.unrespondedRetired[k] != 0 || ra.afterAgentCredit || !ra.afterAgentEventSeen {
			t.Fatalf("unrespondedRetired %d, a1 credit %v, after-agent event seen %v; want 0, false, true", inv.unrespondedRetired[k], ra.afterAgentCredit, ra.afterAgentEventSeen)
		}
	})
	t.Run("clearLastPartialForEvent recognizes the sharer on a branch under the event's", func(t *testing.T) {
		for _, tc := range []struct {
			name     string
			sharerOn string
			evMaps   map[string]any
			want     bool
		}{
			{"event without a branch", "par.wb", m1, true},
			{"event on the sharer's branch", "", m1, true},
			{"other maps", "par.wb", m2, false},
		} {
			t.Run(tc.name, func(t *testing.T) {
				tm, inv, rootID := setup(t)
				inv.mu.Lock()
				inv.retiredActivations[0].swallower.branch = tc.sharerOn
				inv.unrespondedRetired = map[string]int{llmKey(identity{agentName: "worker", branch: tc.sharerOn}): 1}
				inv.mu.Unlock()
				ev := session.NewEvent(t.Context(), rootID.invocationID)
				ev.Author = "worker"
				ev.Timestamp = at(10)
				ev.Actions = session.EventActions{StateDelta: tc.evMaps}
				if got := tm.clearLastPartialForEvent(t.Context(), rootID, "worker", "", ev); got != tc.want {
					t.Errorf("clearLastPartialForEvent = %v, want %v", got, tc.want)
				}
				inv.mu.Lock()
				defer inv.mu.Unlock()
				if ra := inv.retiredActivations[0]; ra.afterAgentEventSeen != tc.want || ra.afterAgentCredit == tc.want {
					t.Errorf("a1 after-agent event seen %v, credit %v; want %v, %v", ra.afterAgentEventSeen, ra.afterAgentCredit, tc.want, !tc.want)
				}
			})
		}
	})
}

// TestAfterAgentCallbacksStillRunning pins how the runs of after-agent
// callbacks that enclose a start or end on a goroutine are counted in its
// stack trace (see afterAgentCallbackRunsOfTrace), and that the activations
// retired on that goroutine whose callbacks still run, the ones that retired
// first among those not recorded yet, are not recorded as returned (see
// markAfterAgentCallbacksReturnedOnLocked).
func TestAfterAgentCallbacksStillRunning(t *testing.T) {
	frame := func(fn string) string {
		return fn + "(0x1, 0x2)\n\t/src/file.go:10 +0x20\n"
	}
	run := agentPkgPath + ".runAfterAgentCallbacks"
	for _, tc := range []struct {
		name  string
		trace string
		want  int
	}{
		{"none", "goroutine 7 [running]:\n" + frame(agentPkgPath+".runBeforeAgentCallbacks") + frame(agentPkgPath+".(*agent).Run.func1"), 0},
		{"one", "goroutine 7 [running]:\n" + frame(agentPkgPath+".runBeforeAgentCallbacks") + frame(run) + frame(agentPkgPath+".(*agent).Run.func1"), 1},
		{"nested", "goroutine 7 [running]:\n" + frame(run) + frame("main.callback") + frame(run), 2},
		{"another package's function of that name", "goroutine 7 [running]:\n" + frame("example.com/other.runAfterAgentCallbacks"), 0},
		{"a function inside it", "goroutine 7 [running]:\n" + frame(run+".func1"), 0},
		{"after the created by line", "goroutine 7 [running]:\n" + frame("main.f") + "created by " + run + " in goroutine 1\n\t/src/file.go:3 +0x1\n", 0},
		{"no frames", "goroutine 7 [running]:", 0},
	} {
		if got := afterAgentCallbackRunsOfTrace([]byte(tc.trace)); got != tc.want {
			t.Errorf("afterAgentCallbackRunsOfTrace(%s) = %d, want %d", tc.name, got, tc.want)
		}
	}
	if got := afterAgentCallbackRunsOfTrace(func() []byte { b, _ := currentGoroutineTrace(); return b }()); got != 0 {
		t.Errorf("afterAgentCallbackRunsOfTrace(this goroutine) = %d, want 0", got)
	}
	setup := func() *invocationTrace {
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, {spanID: "root-1", kind: spanKindAgent, name: "root"}}}
		for _, seq := range []uint64{30, 10, 20} {
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:      retiredSwallower{spanID: fmt.Sprintf("r%d", seq), name: "inner"},
				retiredAtSeq:   seq,
				popGoroutineID: 7,
			})
		}
		inv.retiredActivations = append(inv.retiredActivations, retiredActivation{swallower: retiredSwallower{spanID: "other", name: "inner"}, retiredAtSeq: 5, popGoroutineID: 8})
		return inv
	}
	done := func(inv *invocationTrace) map[string]bool {
		got := map[string]bool{}
		for _, ra := range inv.retiredActivations {
			got[ra.swallower.spanID] = ra.afterAgentDone
		}
		return got
	}
	at := time.Unix(1_700_000_000, 0)
	for _, tc := range []struct {
		running    int
		want       map[string]bool
		wantMarked bool
	}{
		{0, map[string]bool{"r10": true, "r20": true, "r30": true, "other": false}, true},
		{1, map[string]bool{"r10": false, "r20": true, "r30": true, "other": false}, true},
		{2, map[string]bool{"r10": false, "r20": false, "r30": true, "other": false}, true},
		{3, map[string]bool{"r10": false, "r20": false, "r30": false, "other": false}, false},
	} {
		inv := setup()
		if marked := inv.markAfterAgentCallbacksReturnedOnLocked(7, at, tc.running); marked != tc.wantMarked {
			t.Errorf("running %d: marked = %v, want %v", tc.running, marked, tc.wantMarked)
		}
		if got := done(inv); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("running %d: afterAgentDone = %v, want %v", tc.running, got, tc.want)
		}
	}
	// Recorded ones count no more: a later start with one run of callbacks
	// enclosing it keeps the earliest of the rest.
	inv := setup()
	inv.retiredActivations[1].markAfterAgentDone(at)
	inv.markAfterAgentCallbacksReturnedOnLocked(7, at, 1)
	if got, want := done(inv), (map[string]bool{"r10": true, "r20": false, "r30": true, "other": false}); !reflect.DeepEqual(got, want) {
		t.Errorf("with r10 recorded earlier: afterAgentDone = %v, want %v", got, want)
	}
}

// TestSettledAfterAgentPlacementSurvivesInterleavedEvent handles two session
// events of inner on two goroutines: completeResponse takes the first for a
// retired activation's after-agent event, and before its rows are placed,
// the second, an event of the open activation of inner, is handled in full.
// The first event's rows still go under the invocation, as its own call
// reports, and the second's under the open activation.
func TestSettledAfterAgentPlacementSurvivesInterleavedEvent(t *testing.T) {
	ctx := t.Context()
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	rootID := identity{invocationID: "inv-interleaved-placement", agentName: "root"}
	innerID := identity{invocationID: rootID.invocationID, agentName: "inner"}
	tm := newTraceManager()
	tm.pushAgentSpan(ctx, rootID)
	tm.pushAgentSpan(ctx, innerID)
	inv := tm.invocation(ctx, rootID)
	inv.mu.Lock()
	invSpan, innerSpan := inv.agentStack[0].spanID, inv.agentStack[2].spanID
	inv.agentStack[2].startPlatformTime = at(0)
	actions := &session.EventActions{StateDelta: map[string]any{}, ArtifactDelta: map[string]int64{}}
	inv.retiredActivations = []retiredActivation{{
		swallower:              retiredSwallower{spanID: "inner-retired", name: "inner"},
		guardSpanIDs:           []string{inv.agentStack[1].spanID},
		hasAfterAgentCallbacks: true,
		completedOrTask:        true,
		joinedAtRetire:         true,
		startPlatformTime:      at(1),
		retiredAtPlatformTime:  at(2),
		afterAgentActions:      eventActionMaps(&session.Event{Actions: *actions}),
		afterAgentActionsRef:   actions,
	}}
	inv.mu.Unlock()
	newEvent := func(text string, ms int, acts *session.EventActions) *session.Event {
		ev := session.NewEvent(ctx, rootID.invocationID)
		ev.Author = "inner"
		ev.Timestamp = at(ms)
		ev.Content = textContent("model", text)
		if acts != nil {
			ev.Actions = *acts
		}
		return ev
	}
	afterAgent := newEvent("inner summary", 3, actions)
	other := newEvent("inner answer", 4, nil)
	settledOne, otherDone := make(chan bool), make(chan struct{})
	var firstSpan, otherSpan string
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		settled := tm.completeResponseForEvent(ctx, innerID, "inner", "", nil, false, afterAgent)
		settledOne <- settled
		<-otherDone
		firstSpan = tm.eventSpanAt(ctx, innerID, "inner", "", afterAgent.Timestamp, true, true, settled).spanID
	}()
	go func() {
		defer wg.Done()
		defer close(otherDone)
		if !<-settledOne {
			t.Error("the after-agent event was not taken for the retired activation's")
		}
		settled := tm.completeResponseForEvent(ctx, innerID, "inner", "", nil, false, other)
		otherSpan = tm.eventSpanAt(ctx, innerID, "inner", "", other.Timestamp, true, true, settled).spanID
	}()
	wg.Wait()
	if firstSpan != invSpan {
		t.Errorf("the after-agent event's rows placed at %q, want the invocation span %q", firstSpan, invSpan)
	}
	if otherSpan != innerSpan {
		t.Errorf("the open activation's event placed at %q, want its span %q", otherSpan, innerSpan)
	}
}

// TestStartingActivationKeepsSameAgentAfterAgentMaps pins that pushAgentSpan
// drops the after-agent maps of a retired activation whose callbacks
// returned only once the starting activation is open: while an activation of
// the same agent runs, an after-agent event created in the tick in which it
// started could be taken for its own, so the retired activation keeps the
// maps that recognize that event, and loses them once the activation ended.
func TestStartingActivationKeepsSameAgentAfterAgentMaps(t *testing.T) {
	ctx := t.Context()
	rootID := identity{invocationID: "inv-same-agent-start", agentName: "root"}
	workerID := identity{invocationID: rootID.invocationID, agentName: "worker"}
	tm := newTraceManager()
	tm.pushAgentSpan(ctx, rootID)
	inv := tm.invocation(ctx, rootID)
	inv.mu.Lock()
	m := map[string]any{}
	ra := retiredActivation{
		swallower:              retiredSwallower{spanID: "worker-1", name: "worker"},
		parentSpanID:           inv.agentStack[1].spanID,
		guardSpanIDs:           []string{inv.agentStack[1].spanID},
		hasAfterAgentCallbacks: true,
		completedOrTask:        true,
		joinedAtRetire:         true,
		afterAgentActions:      actionMaps{stateDelta: reflect.ValueOf(m).Pointer()},
		afterAgentActionsRef:   m,
	}
	ra.markAfterAgentDone(time.Time{})
	inv.retiredActivations = []retiredActivation{ra}
	inv.mu.Unlock()
	tm.pushAgentSpan(ctx, workerID)
	inv.mu.Lock()
	if !inv.retiredActivations[0].holdsAfterAgentEvent() {
		t.Error("worker-1 dropped its maps when the next activation of worker started")
	}
	inv.mu.Unlock()
	tm.popAgentSpan(ctx, workerID)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	if idx := inv.findRetiredActivationIndexLocked("worker-1"); idx >= 0 && inv.retiredActivations[idx].holdsAfterAgentEvent() {
		t.Error("worker-1 keeps its maps once the next activation of worker ended")
	}
}
