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
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// faninSession is a session whose history grows as the run yields events,
// as a runner's session does, so a dynamic node's cache sees what earlier
// activations and attempts produced.
type faninSession struct {
	mu     sync.Mutex
	events sliceEvents
}

func (s *faninSession) ID() string                { return "test-session" }
func (s *faninSession) AppName() string           { return "test-app" }
func (s *faninSession) UserID() string            { return "test-user" }
func (s *faninSession) State() session.State      { return nil }
func (s *faninSession) LastUpdateTime() time.Time { return time.Time{} }

func (s *faninSession) Events() session.Events {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.events)
}

func (s *faninSession) append(ev *session.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, ev)
}

// faninRunWithSession runs w under a session that records every yielded
// event, with the checks of faninRun.
func faninRunWithSession(t *testing.T, w *Workflow) faninResult {
	t.Helper()
	ctx, _ := faninCtx(t, faninBudget)
	sess := &faninSession{}
	ctx.sess = sess
	return faninRunObserved(t, w.Run(ctx), sess.append)
}

// faninValues returns the Value of each observation.
func faninValues(obs []faninObs) []any {
	out := make([]any, len(obs))
	for i, o := range obs {
		out[i] = o.Value
	}
	return out
}

// faninPaths returns the Path of each observation.
func faninPaths(obs []faninObs) []string {
	out := make([]string, len(obs))
	for i, o := range obs {
		out[i] = o.Path
	}
	return out
}

// retryAlways is a retry policy with the given attempt budget, a 1 ms
// delay and a predicate that retries every error.
func retryAlways(maxAttempts int) *RetryConfig {
	return &RetryConfig{
		MaxAttempts:   maxAttempts,
		InitialDelay:  time.Millisecond,
		BackoffFactor: 1,
		ShouldRetry:   func(error) bool { return true },
	}
}

// TestDynamicNode_RepeatedStaticActivation_DoesNotReplayPriorActivationChildren
// is the regression test for brief Appendix B and ruling 4: every
// activation of a static node had the path ".../loop@1", so a later
// activation's RunNode child was served from the first activation's
// cached result instead of running.
//
// Fails if startNode gives every activation run id "1".
func TestDynamicNode_RepeatedStaticActivation_DoesNotReplayPriorActivationChildren(t *testing.T) {
	var childCalls, loopCalls atomic.Int32
	child := NewFunctionNode("child", func(agent.Context, string) (string, error) {
		return fmt.Sprintf("child-run-%d", childCalls.Add(1)), nil
	}, NodeConfig{})
	loop := NewDynamicNode("loop", func(ctx agent.Context, _ any, emit func(*session.Event) error) (string, error) {
		i := loopCalls.Add(1)
		out, err := RunNode[string](ctx, child, "x")
		if err != nil {
			return "", err
		}
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Routes = []string{"again"}
		if i == 3 {
			ev.Routes = []string{"done"}
		}
		if err := emit(ev); err != nil {
			return "", err
		}
		return fmt.Sprintf("iter-%d:%s", i, out), nil
	}, NodeConfig{})
	w, err := New("wf", []Edge{
		{From: Start, To: loop},
		{From: loop, To: loop, Route: StringRoute("again")},
	})
	if err != nil {
		t.Fatal(err)
	}

	res := faninRunWithSession(t, w)

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	if got := childCalls.Load(); got != 3 {
		t.Errorf("child executions = %d, want 3", got)
	}
	wantChildPaths := []string{"wf@1/loop@1/child@1", "wf@1/loop@2/child@1", "wf@1/loop@3/child@1"}
	if diff := cmp.Diff(wantChildPaths, faninPaths(faninOutputs(res.events, "child"))); diff != "" {
		t.Errorf("child output paths mismatch (-want +got):\n%s", diff)
	}
	wantLoop := []any{"iter-1:child-run-1", "iter-2:child-run-2", "iter-3:child-run-3"}
	if diff := cmp.Diff(wantLoop, faninValues(faninOutputs(res.events, "loop"))); diff != "" {
		t.Errorf("loop outputs mismatch (-want +got):\n%s", diff)
	}
}

// TestDynamicNode_ParentRetry_RerunsDynamicChildren covers B12 and u4965
// (ruling 4 as corrected): when a static dynamic node retries after its
// own error, each attempt runs its RunNode child again, under the next
// child run id of the same activation, as Python's per-run counter does.
//
// Fails if the child run ids restart at 1 on every attempt, so the cache
// serves attempt 1's child result to attempts 2 and 3.
func TestDynamicNode_ParentRetry_RerunsDynamicChildren(t *testing.T) {
	var childCalls, parentCalls atomic.Int32
	var parentPaths faninLog
	child := NewFunctionNode("child", func(agent.Context, any) (string, error) {
		return fmt.Sprintf("child_run_%d", childCalls.Add(1)), nil
	}, NodeConfig{})
	parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		parentPaths.add(faninObs{Path: ctx.Path()})
		out, err := RunNode[string](ctx, child, nil)
		if err != nil {
			return "", err
		}
		if parentCalls.Add(1) < 3 {
			return "", errors.New("parent failed")
		}
		return out, nil
	}, NodeConfig{RetryConfig: retryAlways(3)})
	w, err := New("wf", []Edge{{From: Start, To: parent}})
	if err != nil {
		t.Fatal(err)
	}

	res := faninRunWithSession(t, w)

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	wantParent := []string{"wf@1/parent@1", "wf@1/parent@1", "wf@1/parent@1"}
	if diff := cmp.Diff(wantParent, faninPaths(parentPaths.get())); diff != "" {
		t.Errorf("parent attempt paths mismatch (-want +got):\n%s", diff)
	}
	wantChild := []faninObs{
		{Path: "wf@1/parent@1/child@1", Value: "child_run_1"},
		{Path: "wf@1/parent@1/child@2", Value: "child_run_2"},
		{Path: "wf@1/parent@1/child@3", Value: "child_run_3"},
	}
	gotChild := faninOutputs(res.events, "child")
	for i := range gotChild {
		gotChild[i].Branch = ""
	}
	if diff := cmp.Diff(wantChild, gotChild); diff != "" {
		t.Errorf("child outputs mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]any{"child_run_3"}, faninValues(faninOutputs(res.events, "parent"))); diff != "" {
		t.Errorf("parent outputs mismatch (-want +got):\n%s", diff)
	}
}

// TestDynamicNode_ParentRetryAfterChildFailure_RestartsChildRunIDs covers
// B12: when an attempt fails because its RunNode child failed, the retry
// starts its children at run id 1 again, as Python's reset_for_retry does
// for a DynamicNodeFailError (_node_runner.py:152-159).
//
// Fails if handleCompletion does not reset the run counters under the
// activation's path after a child failure, so attempt 2 runs child@2.
func TestDynamicNode_ParentRetryAfterChildFailure_RestartsChildRunIDs(t *testing.T) {
	var childCalls atomic.Int32
	var childPaths faninLog
	child := NewFunctionNode("child", func(ctx agent.Context, _ any) (string, error) {
		childPaths.add(faninObs{Path: ctx.Path()})
		if childCalls.Add(1) == 1 {
			return "", errors.New("child failed")
		}
		return "child-ok", nil
	}, NodeConfig{})
	parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		return RunNode[string](ctx, child, nil)
	}, NodeConfig{RetryConfig: retryAlways(2)})
	w, err := New("wf", []Edge{{From: Start, To: parent}})
	if err != nil {
		t.Fatal(err)
	}

	res := faninRunWithSession(t, w)

	if res.err != nil {
		t.Errorf("Run error = %v, want nil", res.err)
	}
	want := []string{"wf@1/parent@1/child@1", "wf@1/parent@1/child@1"}
	if diff := cmp.Diff(want, faninPaths(childPaths.get())); diff != "" {
		t.Errorf("child attempt paths mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]any{"child-ok"}, faninValues(faninOutputs(res.events, "parent"))); diff != "" {
		t.Errorf("parent outputs mismatch (-want +got):\n%s", diff)
	}
}
