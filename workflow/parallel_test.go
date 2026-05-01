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

package workflow_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

func runWorkflow(t *testing.T, wf *workflow.Workflow) []*session.Event {
	t.Helper()
	wfAgent, err := wf.AsAgent()
	if err != nil {
		t.Fatalf("AsAgent: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "test",
		Agent:             wfAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}}
	var events []*session.Event
	for ev, err := range r.Run(context.Background(), "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run err: %v", err)
		}
		events = append(events, ev)
	}
	return events
}

func TestWorkflow_FanOutFanIn_WithJoinNode(t *testing.T) {
	type val struct {
		N int `json:"n"`
	}
	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) (val, error) { return val{N: 10}, nil })
	a := workflow.Func("a",
		func(_ *workflow.NodeContext, in val) (val, error) { return val{N: in.N + 1}, nil })
	b := workflow.Func("b",
		func(_ *workflow.NodeContext, in val) (val, error) { return val{N: in.N + 2}, nil })
	c := workflow.Func("c",
		func(_ *workflow.NodeContext, in val) (val, error) { return val{N: in.N + 3}, nil })
	join := workflow.Join("join")
	final := workflow.Func("final",
		func(_ *workflow.NodeContext, in any) (int, error) {
			m := in.(map[string]any)
			total := 0
			for _, v := range m {
				total += v.(val).N
			}
			return total, nil
		})

	wf, err := workflow.New(workflow.Config{
		Name: "fanout",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, a),
			workflow.Connect(seed, b),
			workflow.Connect(seed, c),
			workflow.Connect(a, join),
			workflow.Connect(b, join),
			workflow.Connect(c, join),
			workflow.Connect(join, final),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events := runWorkflow(t, wf)
	var got int
	for _, ev := range events {
		if ev.Author == "final" && ev.Actions.NodeInfo != nil && ev.Actions.NodeInfo.Output != nil {
			got = ev.Actions.NodeInfo.Output.(int)
		}
	}
	// 10 -> a:11, b:12, c:13 -> join: {a:11,b:12,c:13} -> final: 36
	if got != 36 {
		t.Errorf("final = %d, want 36", got)
	}
}

func TestWorkflow_RetryOnTransientError(t *testing.T) {
	var attempts atomic.Int32
	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) (int, error) { return 1, nil })
	flaky := workflow.Func("flaky",
		func(_ *workflow.NodeContext, in int) (int, error) {
			n := attempts.Add(1)
			if n < 3 {
				return 0, errors.New("transient")
			}
			return in * 10, nil
		},
		workflow.WithRetry(&workflow.RetryConfig{
			MaxAttempts:  5,
			InitialDelay: time.Microsecond,
			MaxDelay:     time.Microsecond,
			Jitter:       -1,
		}),
	)

	wf, err := workflow.New(workflow.Config{
		Name: "retrywf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, flaky),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events := runWorkflow(t, wf)
	var got int
	for _, ev := range events {
		if ev.Author == "flaky" && ev.Actions.NodeInfo != nil && ev.Actions.NodeInfo.Output != nil {
			got = ev.Actions.NodeInfo.Output.(int)
		}
	}
	if got != 10 {
		t.Errorf("output = %d, want 10", got)
	}
	if attempts.Load() != 3 {
		t.Errorf("attempts = %d, want 3", attempts.Load())
	}
}

func TestWorkflow_NodeTimeout(t *testing.T) {
	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) (int, error) { return 1, nil })
	slow := workflow.Func("slow",
		func(ctx *workflow.NodeContext, _ int) (int, error) {
			select {
			case <-time.After(2 * time.Second):
				return 0, nil
			case <-ctx.InvocationContext.Done():
				return 0, ctx.InvocationContext.Err()
			}
		},
		workflow.WithTimeout(50*time.Millisecond),
	)

	wf, err := workflow.New(workflow.Config{
		Name: "timeoutwf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, slow),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	wfAgent, _ := wf.AsAgent()
	r, _ := runner.New(runner.Config{
		AppName:           "t",
		Agent:             wfAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})

	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}}
	sawErr := false
	start := time.Now()
	for _, err := range r.Run(context.Background(), "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			sawErr = true
		}
	}
	if !sawErr {
		t.Error("expected timeout error from slow node")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("timeout did not abort quickly: elapsed = %v", elapsed)
	}
}

// TestWorkflow_NodeTimeoutWrapsBothSentinels verifies that a per-attempt
// timeout produces an error chain matching BOTH ErrNodeTimeout (workflow
// sentinel) and context.DeadlineExceeded (root cause). Mirrors
// adk-python NodeTimeoutError(asyncio.TimeoutError).
func TestWorkflow_NodeTimeoutWrapsBothSentinels(t *testing.T) {
	slow := workflow.Func("slow",
		func(ctx *workflow.NodeContext, _ any) (int, error) {
			select {
			case <-time.After(2 * time.Second):
				return 0, nil
			case <-ctx.InvocationContext.Done():
				return 0, ctx.InvocationContext.Err()
			}
		},
		workflow.WithTimeout(20*time.Millisecond),
	)
	wf, err := workflow.New(workflow.Config{
		Name:  "twrap",
		Edges: []workflow.Edge{workflow.Connect(workflow.START, slow)},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	wfAgent, _ := wf.AsAgent()
	r, _ := runner.New(runner.Config{
		AppName:           "t",
		Agent:             wfAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}}
	var firstErr error
	for _, e := range r.Run(context.Background(), "u", "s", msg, agent.RunConfig{}) {
		if e != nil {
			firstErr = e
			break
		}
	}
	if firstErr == nil {
		t.Fatal("expected timeout error")
	}
	if !errors.Is(firstErr, workflow.ErrNodeTimeout) {
		t.Errorf("err = %v, want errors.Is(_, ErrNodeTimeout)", firstErr)
	}
	if !errors.Is(firstErr, context.DeadlineExceeded) {
		t.Errorf("err = %v, want errors.Is(_, context.DeadlineExceeded)", firstErr)
	}
}

func TestWorkflow_ParallelWorker_FansOutAndCollects(t *testing.T) {
	var seen sync.Map
	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) ([]int, error) {
			return []int{2, 3, 5, 7}, nil
		})
	square := workflow.Func("square",
		func(_ *workflow.NodeContext, n int) (int, error) {
			seen.Store(n, true)
			return n * n, nil
		})
	worker := workflow.Parallel[int](square)

	wf, err := workflow.New(workflow.Config{
		Name: "pwf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, worker),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events := runWorkflow(t, wf)
	var got []any
	for _, ev := range events {
		if ev.Author == worker.Name() && ev.Actions.NodeInfo != nil && ev.Actions.NodeInfo.Output != nil {
			got = ev.Actions.NodeInfo.Output.([]any)
		}
	}
	if len(got) != 4 {
		t.Fatalf("results = %v, want length 4", got)
	}
	want := map[int]bool{4: true, 9: true, 25: true, 49: true}
	for _, r := range got {
		if !want[r.(int)] {
			t.Errorf("unexpected result %v", r)
		}
	}
}

func TestWorkflow_MaxConcurrency_BoundsParallelism(t *testing.T) {
	// Three nodes can fire after seed; with MaxConcurrency=2, peak in-flight
	// must be <= 2.
	type val struct{ N int }
	var inFlight atomic.Int32
	var peak atomic.Int32

	work := func(name string) workflow.Node {
		return workflow.Func(name,
			func(_ *workflow.NodeContext, _ val) (val, error) {
				cur := inFlight.Add(1)
				for {
					p := peak.Load()
					if cur <= p || peak.CompareAndSwap(p, cur) {
						break
					}
				}
				time.Sleep(20 * time.Millisecond)
				inFlight.Add(-1)
				return val{}, nil
			})
	}
	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) (val, error) { return val{}, nil })
	a, b, c := work("a"), work("b"), work("c")
	wf, err := workflow.New(workflow.Config{
		Name:           "boundedwf",
		MaxConcurrency: 2,
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, a),
			workflow.Connect(seed, b),
			workflow.Connect(seed, c),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	runWorkflow(t, wf)
	if got := peak.Load(); got > 2 {
		t.Errorf("peak in-flight = %d, want <= 2", got)
	}
}

func TestWorkflow_ConditionalRouting_DefaultBranch(t *testing.T) {
	// classifier emits no route; only the default-branch edge should fire.
	classifier := workflow.Func("classify",
		func(_ *workflow.NodeContext, _ any) (int, error) { return 1, nil })
	branchA := workflow.Func("a",
		func(_ *workflow.NodeContext, _ int) (string, error) { return "a", nil })
	branchDefault := workflow.Func("d",
		func(_ *workflow.NodeContext, _ int) (string, error) { return "d", nil })

	wf, err := workflow.New(workflow.Config{
		Name: "rwf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, classifier),
			workflow.Connect(classifier, branchA, workflow.RouteString("alpha")),
			workflow.Connect(classifier, branchDefault, workflow.DefaultRoute),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events := runWorkflow(t, wf)

	var sawA, sawD bool
	for _, ev := range events {
		if ev.Author == "a" {
			sawA = true
		}
		if ev.Author == "d" {
			sawD = true
		}
	}
	if sawA {
		t.Error("branch A should not have fired (no matching route)")
	}
	if !sawD {
		t.Error("default branch should have fired")
	}
}
