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

package tool_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// TestSequentialRunner_SurvivesWrappers checks that WithConfirmation and
// WithToolNamePrefix, alone and stacked, answer RunsSequentially with the
// value of the function tool they wrap.
func TestSequentialRunner_SurvivesWrappers(t *testing.T) {
	for _, tc := range []struct {
		name       string
		sequential bool
	}{
		{name: "sequential", sequential: true},
		{name: "plain", sequential: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ft, err := functiontool.New(
				functiontool.Config{Name: "a", Description: "a", Sequential: tc.sequential},
				func(agent.Context, struct{}) (map[string]any, error) { return nil, nil })
			if err != nil {
				t.Fatalf("functiontool.New() error = %v", err)
			}
			base := tool.AsToolset(ft)

			if got := runsSequentially(ft); got != tc.sequential {
				t.Errorf("function tool RunsSequentially = %v, want %v", got, tc.sequential)
			}
			if got := runsSequentially(onlyTool(t, tool.WithConfirmation(base, false, nil))); got != tc.sequential {
				t.Errorf("WithConfirmation RunsSequentially = %v, want %v", got, tc.sequential)
			}
			if got := runsSequentially(onlyTool(t, tool.WithToolNamePrefix(base, "p"))); got != tc.sequential {
				t.Errorf("WithToolNamePrefix RunsSequentially = %v, want %v", got, tc.sequential)
			}
			prefixedConfirmed := tool.WithToolNamePrefix(tool.WithConfirmation(base, false, nil), "p")
			if got := runsSequentially(onlyTool(t, prefixedConfirmed)); got != tc.sequential {
				t.Errorf("prefix over confirmation RunsSequentially = %v, want %v", got, tc.sequential)
			}
		})
	}
}

// runsSequentially reports whether tl is a tool.SequentialRunner that
// answers true.
func runsSequentially(tl tool.Tool) bool {
	s, ok := tl.(tool.SequentialRunner)
	return ok && s.RunsSequentially()
}

// TestSequentialRunner_LockDoesNotOrderCalls pins that a tool's own lock keeps
// each read-modify-write atomic but does not order the calls of a turn: an
// unmarked counter tool whose call 1 takes the lock first answers 2 to call 0
// and 1 to call 1, and the merged event stores call 1's 1. It fails if
// unmarked calls start in call order, or if the merge keeps the session's
// last write of a scalar.
func TestSequentialRunner_LockDoesNotOrderCalls(t *testing.T) {
	var mu sync.Mutex
	call1Wrote := make(chan struct{})
	incr := seqTestTool(t, functiontool.Config{Name: "incr", Description: "increments count"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			if ctx.FunctionCallID() == "c0" {
				<-call1Wrote
			}
			mu.Lock()
			n := 0
			if v, err := ctx.State().Get("count"); err == nil {
				n = v.(int)
			}
			err := ctx.State().Set("count", n+1)
			mu.Unlock()
			if ctx.FunctionCallID() == "c1" {
				close(call1Wrote)
			}
			return map[string]any{"count": n + 1}, err
		})
	r, svc, _ := seqTestRunner(t, []tool.Tool{incr},
		&genai.FunctionCall{ID: "c0", Name: "incr", Args: map[string]any{}},
		&genai.FunctionCall{ID: "c1", Name: "incr", Args: map[string]any{}})
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	if err := seqTestRun(ctx, r); err != nil {
		t.Fatalf("run error = %v", err)
	}

	sess := seqTestSession(t, svc)
	type result struct {
		Responses map[string]map[string]any
		Stored    any
	}
	stored, _ := sess.State().Get("count")
	got := result{Responses: seqTestResponses(sess), Stored: stored}
	// A function tool's result reaches the response through JSON.
	want := result{
		Responses: map[string]map[string]any{"c0": {"count": float64(2)}, "c1": {"count": float64(1)}},
		Stored:    1,
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("turn mismatch (-want +got):\n%s", diff)
	}
}

// TestSequentialRunner_WaitForLaterSequentialCallEndsWithRunContext pins what
// a user sees when a plain call 0 waits for the write of a sequential call 1,
// which waits for call 0: the run does not end until its context does, then
// it ends with the context's error, the model is not asked again, and the
// session holds the model's two calls with no function response and none of
// the calls' writes. It fails if the sequential call stops waiting for
// earlier plain calls, if the wait ends on its own, or if the error answer of
// the wait reaches the session.
func TestSequentialRunner_WaitForLaterSequentialCallEndsWithRunContext(t *testing.T) {
	waiting := make(chan struct{})
	waiter := seqTestTool(t, functiontool.Config{Name: "waiter", Description: "waits for b"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			close(waiting)
			for {
				if _, err := ctx.State().Get("b"); err == nil {
					return map[string]any{}, ctx.State().Set("a", true)
				}
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-time.After(5 * time.Millisecond):
				}
			}
		})
	writer := seqTestTool(t, functiontool.Config{Name: "writer", Description: "writes b", Sequential: true},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			return map[string]any{}, ctx.State().Set("b", true)
		})
	r, svc, llm := seqTestRunner(t, []tool.Tool{waiter, writer},
		&genai.FunctionCall{ID: "c0", Name: "waiter", Args: map[string]any{}},
		&genai.FunctionCall{ID: "c1", Name: "writer", Args: map[string]any{}})
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	done := make(chan error, 1)
	go func() { done <- seqTestRun(ctx, r) }()
	select {
	case <-waiting:
	case <-time.After(5 * time.Second):
		t.Fatal("call 0 did not start within 5s")
	}
	select {
	case err := <-done:
		t.Fatalf("run ended before its context, with error %v", err)
	case <-time.After(200 * time.Millisecond):
	}
	cancel()
	var runErr error
	select {
	case runErr = <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("run did not end within 5s of its context")
	}

	if !errors.Is(runErr, context.Canceled) {
		t.Errorf("run error = %v, want context.Canceled", runErr)
	}
	if n := len(llm.Requests); n != 1 {
		t.Errorf("model requests = %d, want 1", n)
	}
	sess := seqTestSession(t, svc)
	if diff := cmp.Diff([]string{"c0", "c1"}, seqTestCallIDs(sess)); diff != "" {
		t.Errorf("stored function calls mismatch (-want +got):\n%s", diff)
	}
	if got := seqTestResponses(sess); len(got) != 0 {
		t.Errorf("stored function responses = %v, want none", got)
	}
	for k, v := range sess.State().All() {
		t.Errorf("stored state %s = %v, want no key", k, v)
	}
}

// TestSequentialRunner_WaitErrorAnswersCallWhoseOwnContextEnds pins that the
// error answer of a failed wait reaches the session and the model when only
// the call's own context ends: a task runner gives the sequential call 1 a
// context that has ended, while call 0 is still running. Call 1 is answered
// with the wait's error, its tool does not run, and the run goes on to the
// model's next response. It fails if a failed wait runs the tool anyway or
// answers with other text.
func TestSequentialRunner_WaitErrorAnswersCallWhoseOwnContextEnds(t *testing.T) {
	release := make(chan struct{})
	slow := seqTestTool(t, functiontool.Config{Name: "slow", Description: "waits for release"},
		func(agent.Context, struct{}) (map[string]any, error) {
			<-release
			return map[string]any{"ok": true}, nil
		})
	writer := seqTestTool(t, functiontool.Config{Name: "writer", Description: "writes b", Sequential: true},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			return map[string]any{"ok": true}, ctx.State().Set("b", true)
		})
	r, svc, llm := seqTestRunner(t, []tool.Tool{slow, writer},
		&genai.FunctionCall{ID: "c0", Name: "slow", Args: map[string]any{}},
		&genai.FunctionCall{ID: "c1", Name: "writer", Args: map[string]any{}})
	// Starts the tasks in order, gives the second an ended context, and
	// releases the first once the second has returned.
	runTasks := func(ctx context.Context, tasks []func(context.Context)) {
		ended, end := context.WithCancel(ctx)
		end()
		var wg sync.WaitGroup
		wg.Go(func() {
			tasks[0](ctx)
		})
		tasks[1](ended)
		close(release)
		wg.Wait()
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	if err := seqTestRun(platform.WithTaskRunner(ctx, runTasks), r); err != nil {
		t.Fatalf("run error = %v", err)
	}

	sess := seqTestSession(t, svc)
	want := map[string]map[string]any{
		"c0": {"ok": true},
		"c1": {"error": "wait for earlier function calls: context canceled"},
	}
	if diff := cmp.Diff(want, seqTestResponses(sess)); diff != "" {
		t.Errorf("stored function responses mismatch (-want +got):\n%s", diff)
	}
	if v, err := sess.State().Get("b"); !errors.Is(err, session.ErrStateKeyNotExist) {
		t.Errorf("stored b = %v, %v; want ErrStateKeyNotExist", v, err)
	}
	if n := len(llm.Requests); n != 2 {
		t.Errorf("model requests = %d, want 2", n)
	}
}

func seqTestTool[A any](t *testing.T, cfg functiontool.Config, fn functiontool.Func[A, map[string]any]) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(cfg, fn)
	if err != nil {
		t.Fatalf("functiontool.New(%q) error = %v", cfg.Name, err)
	}
	return tl
}

// seqTestRunner returns a runner of an LLM agent with tools whose model
// answers the first request with calls and the next with text.
func seqTestRunner(t *testing.T, tools []tool.Tool, calls ...*genai.FunctionCall) (*runner.Runner, session.Service, *testutil.MockModel) {
	t.Helper()
	parts := make([]*genai.Part, len(calls))
	for i, c := range calls {
		parts[i] = &genai.Part{FunctionCall: c}
	}
	llm := &testutil.MockModel{Responses: []*genai.Content{
		{Role: genai.RoleModel, Parts: parts},
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "agent", Model: llm, Tools: tools})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc, AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	return r, svc, llm
}

// seqTestRun runs one user message and returns the first error the run yields.
func seqTestRun(ctx context.Context, r *runner.Runner) error {
	for _, err := range r.Run(ctx, "user", "s", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			return err
		}
	}
	return nil
}

func seqTestSession(t *testing.T, svc session.Service) session.Session {
	t.Helper()
	got, err := svc.Get(context.Background(), &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}
	return got.Session
}

// seqTestResponses returns the stored function responses of sess by call ID.
func seqTestResponses(sess session.Session) map[string]map[string]any {
	got := map[string]map[string]any{}
	for ev := range sess.Events().All() {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil {
				got[p.FunctionResponse.ID] = p.FunctionResponse.Response
			}
		}
	}
	return got
}

// seqTestCallIDs returns the IDs of the function calls stored in sess, in
// order.
func seqTestCallIDs(sess session.Session) []string {
	var got []string
	for ev := range sess.Events().All() {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil {
				got = append(got, p.FunctionCall.ID)
			}
		}
	}
	return got
}
