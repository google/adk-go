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
	"fmt"
	"iter"
	"maps"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"weak"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

// TestAfterAgentEventHeldPastNextActivationKeepsRecovery_Runner runs two
// activations of the custom agent inner one after the other on the goroutine
// of a custom root. The first yields nothing in its run, so it retires
// without a response, and its AfterAgentCallback returns content, only
// changes state, or returns nothing. root holds the event that ADK builds
// from it until the second activation is open: the second one started on the
// goroutine that ran the first one's callbacks, which shows that they had
// returned and that their event was created by then. The second activation
// drains its child's failure under par and recovers with an event that it
// creates afterwards, so that event is not the first activation's after-agent
// event, whether that event was delivered first, was not recognized, or never
// came, and the run completes. Without the recovery the failure stays.
func TestAfterAgentEventHeldPastNextActivationKeepsRecovery_Runner(t *testing.T) {
	const failure = "inner-2's child failure"
	run := func(t *testing.T, shape string, recovers bool) []map[string]any {
		t.Helper()
		tm := newTraceManager()
		p, takeRows := newQueuedRowsPlugin(t, tm)
		tm.selfPlugin = weak.Make(p)
		par := newParallelAgent(t, "par", failingChild(t, failure))
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if runNumber(ctx) == 1 {
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
					if drained && recovers {
						yield(textEvent(ctx, "inner", ctx.Branch(), "inner-2 recovery"), nil)
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{func(cctx agent.Context) (*genai.Content, error) {
				if runNumber(cctx) != 1 {
					return nil, nil
				}
				switch shape {
				case "content":
					return textContent("model", "inner-1 summary"), nil
				case "state_only":
					return nil, cctx.State().Set("inner_1_state", true)
				}
				return nil, nil
			}},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var held []*session.Event
					for n := 1; n <= 2; n++ {
						runCtx := ctx.WithContext(context.WithValue(ctx, afterAgentRunKey{}, n))
						for ev, runErr := range inner.Run(runCtx) {
							if n == 1 && runErr == nil && ev != nil {
								held = append(held, ev)
								continue
							}
							if n == 2 && len(held) > 0 {
								// The second activation of inner is open.
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
					for _, h := range held {
						if !yield(h, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-held-after-agent-event", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return takeRows()
	}
	for _, shape := range []string{"content", "state_only", "no_event"} {
		t.Run(shape+"/recovery_completes", func(t *testing.T) {
			wantInvocationCompleted(t, run(t, shape, true))
		})
		t.Run(shape+"/no_recovery_keeps_the_failure", func(t *testing.T) {
			wantInvocationFailure(t, run(t, shape, false), failure)
		})
	}
}

// TestSubBranchEventKeepsEnclosingBranchFailure_Runner runs one custom agent,
// worker, as activation A on a branch and as activation B on a branch under
// it, both under a custom root. A drains the failure of its child under apar.
// B creates an event once that failure was recorded, without a branch, as
// session.NewEvent leaves it, when A runs on the root branch, or with A's
// branch, as the workflow scheduler stamps an enclosing node run's branch on
// an event that left Branch empty. Either activation can have created such an
// event, so it is no one's: it does not clear A's failure, whether it arrives
// while both are open or once both retired. A recovery that A creates once B
// retired, which B cannot have created, still completes the run.
func TestSubBranchEventKeepsEnclosingBranchFailure_Runner(t *testing.T) {
	const failure = "A's child failure"
	type config struct {
		// enclosing runs A on outer.wa and B on outer.wa.par.wb, and
		// stamps outer.wa on B's event; otherwise A runs on the root
		// branch and B on par.wb, and B's event has no branch.
		enclosing bool
		// retired holds B's event until A retired too.
		retired bool
		// bEvent makes B create its event.
		bEvent bool
		// aRecovers makes A recover once B retired.
		aRecovers bool
	}
	run := func(t *testing.T, c config) []map[string]any {
		t.Helper()
		tm := newTraceManager()
		p, takeRows := newQueuedRowsPlugin(t, tm)
		tm.selfPlugin = weak.Make(p)
		aFailed, bCreated, bRetired, bDelivered := newSignal(), newSignal(), newSignal(), newSignal()
		defer aFailed.fire()
		defer bCreated.fire()
		defer bRetired.fire()
		defer bDelivered.fire()
		aBranch, bBranch := "", "par.wb"
		if c.enclosing {
			aBranch, bBranch = "outer.wa", "outer.wa.par.wb"
		}
		apar := newParallelAgent(t, "apar", failingChild(t, failure))
		worker, err := agent.New(agent.Config{
			Name:      "worker",
			SubAgents: []agent.Agent{apar},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					switch ctx.Branch() {
					case aBranch:
						for ev, runErr := range apar.Run(ctx) {
							if runErr != nil {
								// apar retires once it ends, which records
								// the failure.
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
						aFailed.fire()
						bRetired.wait()
						if c.bEvent && !c.retired {
							bDelivered.wait()
						}
						if c.aRecovers {
							yield(textEvent(ctx, "worker", aBranch, "A recovery"), nil)
						}
					case bBranch:
						aFailed.wait()
						if c.bEvent {
							evBranch := ""
							if c.enclosing {
								evBranch = aBranch
							}
							ev := textEvent(ctx, "worker", evBranch, "B answer")
							bCreated.fire()
							yield(ev, nil)
						}
					default:
						yield(nil, fmt.Errorf("worker on unexpected branch %q", ctx.Branch()))
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(worker): %v", err)
		}
		// wb runs worker as B and fires bRetired once B retired.
		wb, err := agent.New(agent.Config{
			Name: "wb",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					defer bRetired.fire()
					for ev, runErr := range worker.Run(ctx) {
						if !yield(ev, runErr) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(wb): %v", err)
		}
		par := newParallelAgent(t, "par", wb)
		// runBoth runs worker as A and par, which runs B, on goroutines of
		// the agent whose context is ctx and forwards their items. B's event
		// is held until A's run ended if the case says so.
		runBoth := func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			done := make(chan struct{})
			defer close(done)
			items := make(chan runItem)
			var wg sync.WaitGroup
			wg.Add(2)
			go func() {
				wg.Wait()
				close(items)
			}()
			aEnded := newSignal()
			defer aEnded.fire()
			go func() {
				defer aEnded.fire()
				var inner sync.WaitGroup
				inner.Add(1)
				forwardRun(ctx, worker, 1, items, done, &inner)
				inner.Wait()
				wg.Done()
			}()
			forwardRun(ctx, par, 2, items, done, &wg)
			var held []*session.Event
			for it := range items {
				isB := it.err == nil && it.ev != nil && isTextContent(it.ev.Content, "B answer")
				if isB && c.retired {
					held = append(held, it.ev)
					close(it.ack)
					continue
				}
				ok := yield(it.ev, it.err)
				if isB {
					bDelivered.fire()
				}
				close(it.ack)
				if !ok {
					return
				}
			}
			aEnded.wait()
			for _, ev := range held {
				if !yield(ev, nil) {
					return
				}
			}
		}
		// The agent that runs A declares worker as a sub-agent, so that A
		// runs under it and not under par, which it runs beside on the same
		// branch.
		top := par
		subAgents := []agent.Agent{par, worker}
		if c.enclosing {
			wa, err := agent.New(agent.Config{
				Name:      "wa",
				SubAgents: []agent.Agent{par, worker},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						runBoth(ctx, yield)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(wa): %v", err)
			}
			top = newParallelAgent(t, "outer", wa)
			subAgents = []agent.Agent{top}
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: subAgents,
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if !c.enclosing {
						runBoth(ctx, yield)
						return
					}
					for ev, runErr := range top.Run(ctx) {
						if !yield(ev, runErr) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-sub-branch-event", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return takeRows()
	}
	for _, shape := range []struct {
		name      string
		enclosing bool
	}{{"root_branch_and_event_without_branch", false}, {"enclosing_branch_stamped_on_the_event", true}} {
		for _, delivery := range []struct {
			name    string
			retired bool
		}{{"while_both_are_open", false}, {"once_both_retired", true}} {
			name := shape.name + "/" + delivery.name
			t.Run(name+"/B's_event_keeps_A's_failure", func(t *testing.T) {
				wantInvocationFailure(t, run(t, config{enclosing: shape.enclosing, retired: delivery.retired, bEvent: true}), failure)
			})
			t.Run(name+"/without_B's_event_A's_failure_stays", func(t *testing.T) {
				wantInvocationFailure(t, run(t, config{enclosing: shape.enclosing, retired: delivery.retired}), failure)
			})
			t.Run(name+"/A's_recovery_once_B_retired_completes", func(t *testing.T) {
				wantInvocationCompleted(t, run(t, config{enclosing: shape.enclosing, retired: delivery.retired, bEvent: true, aRecovers: true}))
			})
		}
	}
}

// TestRetiredBranchRecoveryWhileLaterSiblingOpen_Runner runs one custom
// agent, worker, under the branches b1 and b2 of the parallel agent top. On
// b1, worker drains the failure of its child under b1par and recovers with
// an event without a branch, as session.NewEvent leaves it, or with top's
// branch, as the workflow scheduler stamps an enclosing node run's branch.
// b1 holds the recovery until worker on b2, which starts only after the
// recovery was created, is open. Only the retired activation on b1 can have
// created the recovery, so it clears the failure that that activation
// swallowed, although the only open activation of worker runs on b2, and the
// run completes. Without the recovery the failure stays.
func TestRetiredBranchRecoveryWhileLaterSiblingOpen_Runner(t *testing.T) {
	const failure = "b1's child failure"
	run := func(t *testing.T, evBranch string, recovers bool) []map[string]any {
		t.Helper()
		tm := newTraceManager()
		p, takeRows := newQueuedRowsPlugin(t, tm)
		tm.selfPlugin = weak.Make(p)
		created, b2Open, delivered := newSignal(), newSignal(), newSignal()
		defer created.fire()
		defer b2Open.fire()
		defer delivered.fire()
		b1par := newParallelAgent(t, "b1par", failingChild(t, failure))
		worker, err := agent.New(agent.Config{
			Name:      "worker",
			SubAgents: []agent.Agent{b1par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if ctx.Branch() == "top.b2" {
						b2Open.fire()
						delivered.wait()
						yield(textEvent(ctx, "worker", ctx.Branch(), "b2 answer"), nil)
						return
					}
					for ev, runErr := range b1par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					if !recovers {
						created.fire()
						return
					}
					ev := textEvent(ctx, "worker", evBranch, "b1 recovery")
					created.fire()
					yield(ev, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(worker): %v", err)
		}
		b1, err := agent.New(agent.Config{
			Name: "b1",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					defer delivered.fire()
					var held []*session.Event
					for ev, runErr := range worker.Run(ctx) {
						if runErr == nil && ev != nil && isTextContent(ev.Content, "b1 recovery") {
							held = append(held, ev)
							continue
						}
						if !yield(ev, runErr) {
							return
						}
					}
					// worker on b1 retired; deliver its recovery once worker
					// on b2 is open.
					b2Open.wait()
					for _, ev := range held {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(b1): %v", err)
		}
		b2, err := agent.New(agent.Config{
			Name: "b2",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					created.wait()
					for ev, runErr := range worker.Run(ctx) {
						if !yield(ev, runErr) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(b2): %v", err)
		}
		top := newParallelAgent(t, "top", b1, b2)
		for _, runErr := range newTestRunner(t, top, p).Run(t.Context(), "u1", "s-retired-branch-recovery", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return takeRows()
	}
	for _, branch := range []struct{ name, value string }{{"without_branch", ""}, {"with_enclosing_branch", "top"}} {
		t.Run(branch.name+"/recovery_completes", func(t *testing.T) {
			wantInvocationCompleted(t, run(t, branch.value, true))
		})
		t.Run(branch.name+"/no_recovery_keeps_the_failure", func(t *testing.T) {
			wantInvocationFailure(t, run(t, branch.value, false), failure)
		})
	}
}

// TestOpenPeerRecoveryAfterManyPeersRetired_Runner runs, under a custom root,
// a long activation of the custom agent worker on a goroutine of root's and,
// while it is open, a number of short activations of worker one after
// another on root's goroutine, each of which responds and retires as a
// concurrent peer of the long one. The long activation then drains its
// child's failure under par and recovers. No short activation can have
// created the recovery, which the long one created after all of them
// retired, so it clears the failure, also when more than three short ones
// retired and their entries would otherwise be compacted into a hull. Without
// the recovery the failure stays.
//
// With an AfterAgentCallback on worker, each short activation keeps the maps
// of its after-agent callbacks while the long one is open, and root runs the
// clean agent marker on its goroutine after the short ones, which shows that
// the last one's callbacks returned too. More than maxPendingAfterAgent short
// ones then retired under root, and their entries are still not folded into
// a hull that stands for after-agent events that may still come: they are
// peer retirements of the open long activation, and their callbacks were seen
// to return.
func TestOpenPeerRecoveryAfterManyPeersRetired_Runner(t *testing.T) {
	const failure = "long worker's child failure"
	run := func(t *testing.T, shorts int, callbacks, recovers bool) []map[string]any {
		t.Helper()
		tm := newTraceManager()
		p, takeRows := newQueuedRowsPlugin(t, tm)
		tm.selfPlugin = weak.Make(p)
		longStarted, shortsDone := newSignal(), newSignal()
		defer longStarted.fire()
		defer shortsDone.fire()
		par := newParallelAgent(t, "par", failingChild(t, failure))
		var afterAgent []agent.AfterAgentCallback
		if callbacks {
			afterAgent = []agent.AfterAgentCallback{func(agent.Context) (*genai.Content, error) {
				return nil, nil
			}}
		}
		marker, err := agent.New(agent.Config{
			Name: "marker",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(textEvent(ctx, "marker", ctx.Branch(), "marker answer"), nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(marker): %v", err)
		}
		worker, err := agent.New(agent.Config{
			Name:                "worker",
			SubAgents:           []agent.Agent{par},
			AfterAgentCallbacks: afterAgent,
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					n := runNumber(ctx)
					if n > 0 {
						yield(textEvent(ctx, "worker", ctx.Branch(), fmt.Sprintf("short worker %d answer", n)), nil)
						return
					}
					longStarted.fire()
					shortsDone.wait()
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
					if drained && recovers {
						yield(textEvent(ctx, "worker", ctx.Branch(), "long worker recovery"), nil)
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(worker): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{worker, marker},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					done := make(chan struct{})
					defer close(done)
					items := make(chan runItem)
					var wg sync.WaitGroup
					wg.Add(1)
					go func() {
						wg.Wait()
						close(items)
					}()
					forwardRun(ctx, worker, 0, items, done, &wg)
					longStarted.wait()
					for n := 1; n <= shorts; n++ {
						runCtx := ctx.WithContext(context.WithValue(ctx, afterAgentRunKey{}, n))
						for ev, runErr := range worker.Run(runCtx) {
							if !yield(ev, runErr) {
								return
							}
						}
					}
					if callbacks {
						for ev, runErr := range marker.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
					}
					shortsDone.fire()
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
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-many-peers-retired", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return takeRows()
	}
	for _, shorts := range []int{3, 4, 6} {
		t.Run(fmt.Sprintf("%d_short_peers/recovery_completes", shorts), func(t *testing.T) {
			wantInvocationCompleted(t, run(t, shorts, false, true))
		})
		t.Run(fmt.Sprintf("%d_short_peers/no_recovery_keeps_the_failure", shorts), func(t *testing.T) {
			wantInvocationFailure(t, run(t, shorts, false, false), failure)
		})
	}
	for _, shorts := range []int{3, maxPendingAfterAgent, maxPendingAfterAgent + 1, 15} {
		t.Run(fmt.Sprintf("with_after-agent_callbacks/%d_short_peers/recovery_completes", shorts), func(t *testing.T) {
			wantInvocationCompleted(t, run(t, shorts, true, true))
		})
		t.Run(fmt.Sprintf("with_after-agent_callbacks/%d_short_peers/no_recovery_keeps_the_failure", shorts), func(t *testing.T) {
			wantInvocationFailure(t, run(t, shorts, true, false), failure)
		})
	}
}

// TestEnclosingBranchAfterAgentRecoveryWhileSubBranchOpen_Runner runs one
// custom agent, worker, as activation A on a branch and as activation B on a
// branch under it, as TestSubBranchEventKeepsEnclosingBranchFailure_Runner
// does. A drains the failure of its child under apar and retires, and its
// AfterAgentCallback recovers while B is still open. ADK gives that event A's
// branch, which covers B's, and the action maps of A's callback context, which
// no other event carries, so it is A's after-agent event and clears A's
// failure, although B could have created an event of that branch then. When
// an earlier plugin replaced both maps, nothing tells the event from one that
// B created, so it clears nothing, and without the recovery the failure stays.
func TestEnclosingBranchAfterAgentRecoveryWhileSubBranchOpen_Runner(t *testing.T) {
	const failure = "A's child failure"
	type config struct {
		// enclosing runs A on outer.wa and B on outer.wa.par.wb; otherwise A
		// runs on the root branch and B on par.wb.
		enclosing bool
		// recovers makes A's AfterAgentCallback return a recovery.
		recovers bool
		// copyMaps makes an earlier plugin replace both action maps of A's
		// recovery.
		copyMaps bool
	}
	run := func(t *testing.T, c config) []map[string]any {
		t.Helper()
		tm := newTraceManager()
		p, takeRows := newQueuedRowsPlugin(t, tm)
		tm.selfPlugin = weak.Make(p)
		bOpen, aEnded := newSignal(), newSignal()
		defer bOpen.fire()
		defer aEnded.fire()
		aBranch, bBranch := "", "par.wb"
		if c.enclosing {
			aBranch, bBranch = "outer.wa", "outer.wa.par.wb"
		}
		apar := newParallelAgent(t, "apar", failingChild(t, failure))
		worker, err := agent.New(agent.Config{
			Name:      "worker",
			SubAgents: []agent.Agent{apar},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					switch ctx.Branch() {
					case aBranch:
						for ev, runErr := range apar.Run(ctx) {
							if runErr != nil {
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
						bOpen.wait()
					case bBranch:
						bOpen.fire()
						// B stays open until A's events, its after-agent
						// event included, were delivered.
						aEnded.wait()
						yield(textEvent(ctx, "worker", bBranch, "B answer"), nil)
					default:
						yield(nil, fmt.Errorf("worker on unexpected branch %q", ctx.Branch()))
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{func(cctx agent.Context) (*genai.Content, error) {
				if runNumber(cctx) != 1 || !c.recovers {
					return nil, nil
				}
				return textContent("model", "A recovery"), nil
			}},
		})
		if err != nil {
			t.Fatalf("agent.New(worker): %v", err)
		}
		wb, err := agent.New(agent.Config{
			Name: "wb",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range worker.Run(ctx) {
						if !yield(ev, runErr) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(wb): %v", err)
		}
		par := newParallelAgent(t, "par", wb)
		// runBoth runs par, which runs B, and worker as A on goroutines of the
		// agent whose context is ctx and forwards their items. aEnded fires
		// once A's run, after-agent callbacks included, ended and root
		// delivered its items.
		runBoth := func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			done := make(chan struct{})
			defer close(done)
			items := make(chan runItem)
			var wg sync.WaitGroup
			wg.Add(2)
			go func() {
				wg.Wait()
				close(items)
			}()
			forwardRun(ctx, par, 2, items, done, &wg)
			bOpen.wait()
			go func() {
				var a sync.WaitGroup
				a.Add(1)
				forwardRun(ctx, worker, 1, items, done, &a)
				a.Wait()
				aEnded.fire()
				wg.Done()
			}()
			for it := range items {
				ok := yield(it.ev, it.err)
				close(it.ack)
				if !ok {
					return
				}
			}
		}
		top := par
		subAgents := []agent.Agent{par, worker}
		if c.enclosing {
			wa, err := agent.New(agent.Config{
				Name:      "wa",
				SubAgents: []agent.Agent{par, worker},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						runBoth(ctx, yield)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(wa): %v", err)
			}
			top = newParallelAgent(t, "outer", wa)
			subAgents = []agent.Agent{top}
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: subAgents,
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if !c.enclosing {
						runBoth(ctx, yield)
						return
					}
					for ev, runErr := range top.Run(ctx) {
						if !yield(ev, runErr) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		plugins := []*baseplugin.Plugin{p}
		if c.copyMaps {
			copier, err := baseplugin.New(baseplugin.Config{
				Name: "action_map_copier",
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if ev != nil && isTextContent(ev.Content, "A recovery") {
						ev.Actions.StateDelta = maps.Clone(ev.Actions.StateDelta)
						ev.Actions.ArtifactDelta = maps.Clone(ev.Actions.ArtifactDelta)
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(action_map_copier): %v", err)
			}
			plugins = []*baseplugin.Plugin{copier, p}
		}
		for _, runErr := range newTestRunner(t, root, plugins...).Run(t.Context(), "u1", "s-enclosing-after-agent", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return takeRows()
	}
	for _, shape := range []struct {
		name      string
		enclosing bool
	}{{"root_branch_and_sub-branch", false}, {"enclosing_branch_and_sub-branch", true}} {
		t.Run(shape.name+"/after-agent_recovery_completes", func(t *testing.T) {
			wantInvocationCompleted(t, run(t, config{enclosing: shape.enclosing, recovers: true}))
		})
		t.Run(shape.name+"/both_maps_copied_keeps_the_failure", func(t *testing.T) {
			wantInvocationFailure(t, run(t, config{enclosing: shape.enclosing, recovers: true, copyMaps: true}), failure)
		})
		t.Run(shape.name+"/no_recovery_keeps_the_failure", func(t *testing.T) {
			wantInvocationFailure(t, run(t, config{enclosing: shape.enclosing}), failure)
		})
	}
}

// TestEnclosingBranchAfterAgentCreditWhileSubBranchOpen_Runner runs, under a
// custom root on the root branch, an activation A1 of the custom agent worker,
// a peer A2 of the same name, and a third activation B of worker on the branch
// par.wb under the root branch. A1 retires without a response while A2 is
// open, so it leaves a credit for the response that it may still deliver, and
// its AfterAgentCallback returns a summary while B is open. The summary
// carries A1's after-agent maps, so it is A1's after-agent event: it settles
// A1's credit and shows that A1 creates no later event. Once B finished, A2's
// only response is its own:
//
//   - llm_peer: A2 is an llmagent named worker, whose model answers then. Its
//     event is A2's awaited model event, which it consumes, and it is placed
//     under A2; the clean run completes without an AGENT_ERROR row.
//   - recovering_peer: A2 is the custom agent, which drained its child's
//     failure under par, and its response is its recovery, which clears that
//     failure, so the run completes.
//
// Without B both complete as well.
func TestEnclosingBranchAfterAgentCreditWhileSubBranchOpen_Runner(t *testing.T) {
	const failure = "A2's child failure"
	type result struct {
		rows []map[string]any
		// a2Span is A2's span, and awaiting its awaitingModelEvent once its
		// response was delivered, -1 when it was not seen.
		a2Span   string
		awaiting int
	}
	run := func(t *testing.T, withB, llmPeer bool) result {
		t.Helper()
		tm := newTraceManager()
		p, takeRows := newQueuedRowsPlugin(t, tm)
		tm.selfPlugin = weak.Make(p)
		bOpen, a2Open, a1Ended, parEnded := newSignal(), newSignal(), newSignal(), newSignal()
		defer bOpen.fire()
		defer a2Open.fire()
		defer a1Ended.fire()
		defer parEnded.fire()
		par2 := newParallelAgent(t, "par2", failingChild(t, failure))
		worker, err := agent.New(agent.Config{
			Name:      "worker",
			SubAgents: []agent.Agent{par2},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					switch runNumber(ctx) {
					case 2:
						a2Open.fire()
						drained := false
						for ev, runErr := range par2.Run(ctx) {
							if runErr != nil {
								drained = true
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
						a1Ended.wait()
						parEnded.wait()
						if drained {
							yield(textEvent(ctx, "worker", ctx.Branch(), "A2 answer"), nil)
						}
					case 3:
						bOpen.fire()
						a1Ended.wait()
						yield(textEvent(ctx, "worker", ctx.Branch(), "B answer"), nil)
					}
					// A1 yields nothing and retires while A2 is open.
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{func(cctx agent.Context) (*genai.Content, error) {
				if runNumber(cctx) != 1 {
					return nil, nil
				}
				return textContent("model", "A1 summary"), nil
			}},
		})
		if err != nil {
			t.Fatalf("agent.New(worker): %v", err)
		}
		a2 := worker
		if llmPeer {
			// This A2 is another agent named worker that root runs itself,
			// so it is not a sub-agent in the tree, whose names are unique.
			// Its model answers once A1's events were delivered and B
			// finished.
			a2, err = llmagent.New(llmagent.Config{
				Name: "worker",
				Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
					a2Open.fire()
					a1Ended.wait()
					parEnded.wait()
					yield(&model.LLMResponse{Content: textContent("model", "A2 answer"), TurnComplete: true}, nil)
				}},
			})
			if err != nil {
				t.Fatalf("llmagent.New(worker): %v", err)
			}
		}
		wb, err := agent.New(agent.Config{
			Name: "wb",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range worker.Run(ctx) {
						if !yield(ev, runErr) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(wb): %v", err)
		}
		par := newParallelAgent(t, "par", wb)
		res := result{awaiting: -1}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{par, worker},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					done := make(chan struct{})
					defer close(done)
					items := make(chan runItem)
					var wg sync.WaitGroup
					wg.Add(3)
					go func() {
						wg.Wait()
						close(items)
					}()
					// fwd runs a with run number n and fires ended once its
					// run ended and root delivered its items.
					fwd := func(a agent.Agent, n int, ended *signal) {
						go func() {
							var one sync.WaitGroup
							one.Add(1)
							forwardRun(ctx, a, n, items, done, &one)
							one.Wait()
							ended.fire()
							wg.Done()
						}()
					}
					if withB {
						fwd(par, 3, parEnded)
						bOpen.wait()
					} else {
						parEnded.fire()
						wg.Done()
					}
					fwd(a2, 2, newSignal())
					a2Open.wait()
					fwd(worker, 1, a1Ended)
					for it := range items {
						ok := yield(it.ev, it.err)
						if it.err == nil && it.ev != nil && isTextContent(it.ev.Content, "A2 answer") {
							// A2 waits for the acknowledgement, so it is
							// still open.
							tm.mu.Lock()
							inv := tm.invocations[ctx.InvocationID()]
							tm.mu.Unlock()
							if inv != nil {
								inv.mu.Lock()
								for _, rec := range inv.agentStack[1:] {
									if rec.name == "worker" && rec.branch == "" {
										res.a2Span, res.awaiting = rec.spanID, rec.awaitingModelEvent
									}
								}
								inv.mu.Unlock()
							}
						}
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
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-enclosing-after-agent-credit", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		res.rows = takeRows()
		return res
	}
	for _, tc := range []struct {
		name  string
		withB bool
	}{{"while_an_activation_runs_on_a_sub-branch", true}, {"without_an_activation_on_a_sub-branch", false}} {
		t.Run(tc.name+"/llm_peer_takes_its_model_event", func(t *testing.T) {
			res := run(t, tc.withB, true)
			if res.awaiting != 0 {
				t.Errorf("A2's awaitingModelEvent once its model event was delivered = %d, want 0", res.awaiting)
			}
			for _, r := range res.rows {
				if r["event_type"] == EventAgentResponse && strings.Contains(fmt.Sprint(r["content"]), "A2 answer") {
					if got, _ := r["span_id"].(string); res.a2Span == "" || got != res.a2Span {
						t.Errorf("A2's answer placed at %q, want A2's span %q", got, res.a2Span)
					}
				}
			}
			if n := len(rowsOfType(res.rows, EventAgentError)); n != 0 {
				t.Errorf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(res.rows))
			}
			wantInvocationCompleted(t, res.rows)
		})
		t.Run(tc.name+"/recovering_peer_completes", func(t *testing.T) {
			wantInvocationCompleted(t, run(t, tc.withB, false).rows)
		})
	}
}

// TestAfterAgentCallbackRunningAgentKeepsItsMaps_Runner runs, under a custom
// root, the custom agent inner on a goroutine of root's, and its
// AfterAgentCallback runs the custom agent helper on that goroutine, with the
// context of inner's run, before it returns. helper starts and ends inside
// inner's after-agent callbacks, which have not returned then, so inner keeps
// the maps that recognize its after-agent event. Meanwhile root runs a second
// activation of inner on another goroutine, which drains its child's failure
// under par and stays open, and only then does the first activation's
// callback return:
//
//   - clean_summary: the first activation responded and its callback returns
//     a summary, which is its after-agent event and does not clear the second
//     activation's failure;
//   - second_recovers: the second activation also recovers after that, which
//     completes the run;
//   - own_recovery: the first activation drained a failure itself and its
//     callback's recovery clears it, while the second runs cleanly.
func TestAfterAgentCallbackRunningAgentKeepsItsMaps_Runner(t *testing.T) {
	const failure = "child failure"
	type config struct {
		ownRecovery, secondRecovers bool
	}
	run := func(t *testing.T, c config) []map[string]any {
		t.Helper()
		tm := newTraceManager()
		p, takeRows := newQueuedRowsPlugin(t, tm)
		tm.selfPlugin = weak.Make(p)
		helperDone, secondDrained, firstDelivered := newSignal(), newSignal(), newSignal()
		defer helperDone.fire()
		defer secondDrained.fire()
		defer firstDelivered.fire()
		par := newParallelAgent(t, "par", failingChild(t, failure))
		helper, err := agent.New(agent.Config{
			Name: "helper",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(textEvent(ctx, "helper", ctx.Branch(), "helper answer"), nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(helper): %v", err)
		}
		var firstCtx agent.InvocationContext
		drain := func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) (drained, ok bool) {
			for ev, runErr := range par.Run(ctx) {
				if runErr != nil {
					drained = true
					continue
				}
				if !yield(ev, nil) {
					return drained, false
				}
			}
			return drained, true
		}
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par, helper},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					switch runNumber(ctx) {
					case 1:
						firstCtx = ctx
						if c.ownRecovery {
							drain(ctx, yield)
							return
						}
						yield(textEvent(ctx, "inner", ctx.Branch(), "inner-1 progress"), nil)
					case 2:
						if c.ownRecovery {
							yield(textEvent(ctx, "inner", ctx.Branch(), "inner-2 answer"), nil)
							return
						}
						drained, ok := drain(ctx, yield)
						// par retired, so the failure is recorded.
						secondDrained.fire()
						firstDelivered.wait()
						if ok && drained && c.secondRecovers {
							yield(textEvent(ctx, "inner", ctx.Branch(), "inner-2 recovery"), nil)
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{func(cctx agent.Context) (*genai.Content, error) {
				if runNumber(cctx) != 1 {
					return nil, nil
				}
				// The callback runs helper on this goroutine with the
				// context of inner's run; its events go nowhere.
				for _, runErr := range helper.Run(firstCtx) {
					if runErr != nil {
						return nil, runErr
					}
				}
				helperDone.fire()
				if c.ownRecovery {
					return textContent("model", "inner-1 recovery"), nil
				}
				secondDrained.wait()
				return textContent("model", "inner-1 summary"), nil
			}},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					done := make(chan struct{})
					defer close(done)
					items := make(chan runItem)
					var wg sync.WaitGroup
					wg.Add(2)
					go func() {
						wg.Wait()
						close(items)
					}()
					forwardRun(ctx, inner, 1, items, done, &wg)
					// The second activation starts once helper ran.
					go func() {
						helperDone.wait()
						forwardRun(ctx, inner, 2, items, done, &wg)
					}()
					for it := range items {
						ok := yield(it.ev, it.err)
						if it.n == 1 && it.err == nil && it.ev != nil && (isTextContent(it.ev.Content, "inner-1 summary") || isTextContent(it.ev.Content, "inner-1 recovery")) {
							firstDelivered.fire()
						}
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
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-callback-runs-agent", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return takeRows()
	}
	t.Run("clean_summary_keeps_the_other_activation's_failure", func(t *testing.T) {
		wantInvocationFailure(t, run(t, config{}), failure)
	})
	t.Run("second_recovers_completes", func(t *testing.T) {
		wantInvocationCompleted(t, run(t, config{secondRecovers: true}))
	})
	t.Run("own_recovery_completes", func(t *testing.T) {
		wantInvocationCompleted(t, run(t, config{ownRecovery: true}))
	})
}

// TestSameTickStartKeepsHeldAfterAgentEvent_Runner runs, under a custom root
// on a clock that advances only when the test says so, a custom activation of
// worker that responds and whose AfterAgentCallback returns a summary in the
// next tick, and then, on root's goroutine and in that same tick, an llmagent
// named worker. root holds the summary until the llmagent awaits its model
// event. The summary was created in the tick in which the llmagent started,
// so its timestamp does not show that the llmagent did not create it, but it
// carries the first activation's after-agent maps, which that activation
// keeps while the llmagent is open: it is the first activation's after-agent
// event, takes nothing from the llmagent, which still awaits its model event
// afterwards, and is not placed under it, while the llmagent's own answer is.
func TestSameTickStartKeepsHeldAfterAgentEvent_Runner(t *testing.T) {
	tm := newTraceManager()
	p, takeRows := newQueuedRowsPlugin(t, tm)
	tm.selfPlugin = weak.Make(p)
	t0 := time.Unix(1_700_000_000, 0)
	var tick atomic.Int64
	runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
		return t0.Add(time.Duration(tick.Load()) * time.Second)
	})
	first, err := agent.New(agent.Config{
		Name: "worker",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(textEvent(ctx, "worker", ctx.Branch(), "worker-1 progress"), nil)
			}
		},
		AfterAgentCallbacks: []agent.AfterAgentCallback{func(agent.Context) (*genai.Content, error) {
			// ADK builds the summary's event in the next tick, in which root
			// starts the llmagent.
			tick.Store(1)
			return textContent("model", "worker-1 summary"), nil
		}},
	})
	if err != nil {
		t.Fatalf("agent.New(worker): %v", err)
	}
	// The second activation is another agent named worker that root runs
	// itself, so it is not a sub-agent in the tree, whose names are unique.
	second, err := llmagent.New(llmagent.Config{
		Name: "worker",
		Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
			yield(&model.LLMResponse{Content: textContent("model", "worker-2 answer"), TurnComplete: true}, nil)
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}
	awaiting, secondSpan := -1, ""
	root, err := agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{first},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				var held []*session.Event
				for ev, runErr := range first.Run(ctx) {
					if runErr == nil && ev != nil && isTextContent(ev.Content, "worker-1 summary") {
						held = append(held, ev)
						continue
					}
					if !yield(ev, runErr) {
						return
					}
				}
				for ev, runErr := range second.Run(ctx) {
					if len(held) > 0 {
						// The llmagent awaits its model event.
						for _, h := range held {
							if !yield(h, nil) {
								return
							}
						}
						held = nil
						tm.mu.Lock()
						inv := tm.invocations[ctx.InvocationID()]
						tm.mu.Unlock()
						if inv != nil {
							inv.mu.Lock()
							for _, rec := range inv.agentStack[1:] {
								if rec.name == "worker" {
									awaiting, secondSpan = rec.awaitingModelEvent, rec.spanID
								}
							}
							inv.mu.Unlock()
						}
					}
					if !yield(ev, runErr) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}
	for _, runErr := range newTestRunner(t, root, p).Run(runCtx, "u1", "s-same-tick-start", textContent("user", "go"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("runner.Run error = %v, want nil", runErr)
		}
	}
	rows := takeRows()
	if awaiting != 1 {
		t.Errorf("awaitingModelEvent of the llmagent once the held summary was delivered = %d, want 1", awaiting)
	}
	for _, r := range rows {
		if r["event_type"] != EventAgentResponse {
			continue
		}
		got, _ := r["span_id"].(string)
		switch content := fmt.Sprint(r["content"]); {
		case strings.Contains(content, "worker-1 summary") && got == secondSpan:
			t.Errorf("the first activation's summary placed under the llmagent's span %q", secondSpan)
		case strings.Contains(content, "worker-2 answer") && (secondSpan == "" || got != secondSpan):
			t.Errorf("the llmagent's answer placed at %q, want its span %q", got, secondSpan)
		}
	}
	wantInvocationCompleted(t, rows)
}
