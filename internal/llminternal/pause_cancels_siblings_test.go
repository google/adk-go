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

package llminternal_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/generatortool"
	"google.golang.org/adk/v2/workflow"
)

// pcsSiblingWait bounds how long a sibling call waits to be canceled. A
// sibling that is not canceled runs on after it, so the tests fail instead of
// hanging.
const pcsSiblingWait = 10 * time.Second

// pcsArgs is the argument type of the test tools.
type pcsArgs struct{}

// pcsCounts counts the callbacks the agent ran for the tool named watched.
type pcsCounts struct {
	watched                string
	mu                     sync.Mutex
	before, onError, after int
}

func (c *pcsCounts) count(t tool.Tool, n *int) {
	if t.Name() != c.watched {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	*n++
}

func (c *pcsCounts) get() (before, onError, after int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.before, c.onError, c.after
}

// pcsHarness is a runner over an in-memory session whose agent root_agent
// answers from a scripted model.
type pcsHarness struct {
	t        *testing.T
	runner   *runner.Runner
	sessions session.Service
}

// pcsConfig is what newPCSHarness builds: the tools, the model's responses,
// the callbacks to count (nil for none) and a before-tool callback of the test.
type pcsConfig struct {
	tools     []tool.Tool
	responses []*genai.Content
	counts    *pcsCounts
	before    llmagent.BeforeToolCallback
}

func newPCSHarness(t *testing.T, cfg pcsConfig) *pcsHarness {
	t.Helper()
	acfg := llmagent.Config{Name: "root_agent", Model: &testutil.MockModel{Responses: cfg.responses}, Tools: cfg.tools}
	if cfg.before != nil {
		acfg.BeforeToolCallbacks = append(acfg.BeforeToolCallbacks, cfg.before)
	}
	if c := cfg.counts; c != nil {
		acfg.BeforeToolCallbacks = append(acfg.BeforeToolCallbacks, func(_ agent.Context, tl tool.Tool, _ map[string]any) (map[string]any, error) {
			c.count(tl, &c.before)
			return nil, nil
		})
		acfg.OnToolErrorCallbacks = []llmagent.OnToolErrorCallback{func(_ agent.Context, tl tool.Tool, _ map[string]any, _ error) (map[string]any, error) {
			c.count(tl, &c.onError)
			return nil, nil
		}}
		acfg.AfterToolCallbacks = []llmagent.AfterToolCallback{func(_ agent.Context, tl tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
			c.count(tl, &c.after)
			return nil, nil
		}}
	}
	a, err := llmagent.New(acfg)
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: sessions, AutoCreateSession: true})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return &pcsHarness{t: t, runner: r, sessions: sessions}
}

// run sends msg under ctx and returns the run's first error.
func (h *pcsHarness) run(ctx context.Context, msg *genai.Content) error {
	h.t.Helper()
	for _, err := range h.runner.Run(ctx, "user", "s", msg, agent.RunConfig{}) {
		if err != nil {
			return err
		}
	}
	return nil
}

func (h *pcsHarness) session() session.Session {
	h.t.Helper()
	resp, err := h.sessions.Get(h.t.Context(), &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s"})
	if err != nil {
		h.t.Fatalf("session Get() error = %v", err)
	}
	return resp.Session
}

// responses returns the function responses of the stored events, in order.
func (h *pcsHarness) responses() []*genai.FunctionResponse {
	h.t.Helper()
	var out []*genai.FunctionResponse
	for ev := range h.session().Events().All() {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil && p.FunctionResponse.Name != workflow.WorkflowInputFunctionCallName {
				out = append(out, p.FunctionResponse)
			}
		}
	}
	return out
}

func pcsCalls(fcs ...*genai.FunctionCall) *genai.Content {
	parts := make([]*genai.Part, len(fcs))
	for i, fc := range fcs {
		parts[i] = &genai.Part{FunctionCall: fc}
	}
	return &genai.Content{Role: genai.RoleModel, Parts: parts}
}

func pcsCall(id, name string) *genai.FunctionCall {
	return &genai.FunctionCall{ID: id, Name: name, Args: map[string]any{}}
}

func pcsUserText(text string) *genai.Content { return genai.NewContentFromText(text, genai.RoleUser) }

// pcsAskTool is a generator tool named ask that requests input under the
// interrupt id region and returns the answer. It first waits until started is
// closed, so that the sibling under test is running when ask pauses.
func pcsAskTool(t *testing.T, started <-chan struct{}) tool.Tool {
	t.Helper()
	tl, err := generatortool.New(functiontool.Config{Name: "ask"}, func(ctx agent.Context, _ pcsArgs, emit func(*session.Event) error) (any, error) {
		pcsAwait(started)
		return workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "region"})
	})
	if err != nil {
		t.Fatalf("generatortool.New() error = %v", err)
	}
	return tl
}

// pcsAwait waits until ch is closed, or pcsSiblingWait has passed. A nil ch
// is closed already.
func pcsAwait(ch <-chan struct{}) {
	if ch == nil {
		return
	}
	select {
	case <-ch:
	case <-time.After(pcsSiblingWait):
	}
}

// pcsFunctionTool is a function tool named name with handler h.
func pcsFunctionTool(t *testing.T, cfg functiontool.Config, h functiontool.Func[pcsArgs, map[string]any]) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(cfg, h)
	if err != nil {
		t.Fatalf("functiontool.New(%q) error = %v", cfg.Name, err)
	}
	return tl
}

// pcsWaitTool is a function tool named wait that closes started, waits until
// its context is done, stores ctx.Err() in *saw and returns it. If the context
// is still open after pcsSiblingWait, it sets state key k to "v" and returns
// {"ok": true}.
func pcsWaitTool(t *testing.T, saw *atomic.Value, started chan<- struct{}) tool.Tool {
	return pcsFunctionTool(t, functiontool.Config{Name: "wait"}, func(ctx agent.Context, _ pcsArgs) (map[string]any, error) {
		close(started)
		select {
		case <-ctx.Done():
			saw.Store(ctx.Err())
			return nil, ctx.Err()
		case <-time.After(pcsSiblingWait):
		}
		if err := ctx.State().Set("k", "v"); err != nil {
			return nil, err
		}
		return map[string]any{"ok": true}, nil
	})
}

func pcsStateK(t *testing.T, h *pcsHarness) {
	t.Helper()
	if v, err := h.session().State().Get("k"); !errors.Is(err, session.ErrStateKeyNotExist) {
		t.Errorf("stored state k = %v, %v; want session.ErrStateKeyNotExist", v, err)
	}
}

// Fails if a pause leaves a running sibling call to finish on its own: the
// sibling then writes k after its wait, and its error goes to the on-tool-error
// and after-tool callbacks, which adk-python's CancelledError skips.
func TestPause_CancelsRunningSibling(t *testing.T) {
	var saw atomic.Value
	started := make(chan struct{})
	counts := &pcsCounts{watched: "wait"}
	h := newPCSHarness(t, pcsConfig{
		tools:     []tool.Tool{pcsAskTool(t, started), pcsWaitTool(t, &saw, started)},
		responses: []*genai.Content{pcsCalls(pcsCall("call_1", "ask"), pcsCall("call_2", "wait"))},
		counts:    counts,
	})

	if err := h.run(t.Context(), pcsUserText("go")); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if got, _ := saw.Load().(error); got != context.Canceled {
		t.Errorf("wait saw ctx.Err() = %v, want context.Canceled", got)
	}
	pcsStateK(t, h)
	if got := h.responses(); len(got) != 0 {
		t.Errorf("stored function responses = %v, want none", got)
	}
	before, onError, after := counts.get()
	if diff := cmp.Diff([3]int{1, 0, 0}, [3]int{before, onError, after}); diff != "" {
		t.Errorf("wait callbacks (before, on-error, after) mismatch (-want +got):\n%s", diff)
	}
}

// Fails if a call that has not started when a sibling pauses still runs its
// before-tool callbacks or its tool. The task runner starts the calls one
// after the other, so wait starts after ask has paused.
func TestPause_SiblingNotStartedRunsNothing(t *testing.T) {
	var runs atomic.Int32
	note := pcsFunctionTool(t, functiontool.Config{Name: "note"}, func(ctx agent.Context, _ pcsArgs) (map[string]any, error) {
		runs.Add(1)
		return map[string]any{"ok": true}, ctx.State().Set("k", "v")
	})
	counts := &pcsCounts{watched: "note"}
	h := newPCSHarness(t, pcsConfig{
		tools:     []tool.Tool{pcsAskTool(t, nil), note},
		responses: []*genai.Content{pcsCalls(pcsCall("call_1", "ask"), pcsCall("call_2", "note"))},
		counts:    counts,
	})
	inOrder := func(ctx context.Context, tasks []func(context.Context)) {
		for _, task := range tasks {
			task(ctx)
		}
	}

	if err := h.run(platform.WithTaskRunner(t.Context(), inOrder), pcsUserText("go")); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if got := runs.Load(); got != 0 {
		t.Errorf("note runs = %d, want 0", got)
	}
	before, onError, after := counts.get()
	if diff := cmp.Diff([3]int{0, 0, 0}, [3]int{before, onError, after}); diff != "" {
		t.Errorf("note callbacks (before, on-error, after) mismatch (-want +got):\n%s", diff)
	}
	pcsStateK(t, h)
}

// Fails if a call whose before-tool callback was running when a sibling
// paused goes on to run its tool.
func TestPause_SiblingInBeforeToolCallbackDoesNotRunTool(t *testing.T) {
	var runs atomic.Int32
	var saw atomic.Value
	started := make(chan struct{})
	note := pcsFunctionTool(t, functiontool.Config{Name: "note"}, func(agent.Context, pcsArgs) (map[string]any, error) {
		runs.Add(1)
		return map[string]any{"ok": true}, nil
	})
	waitInBefore := func(ctx agent.Context, tl tool.Tool, _ map[string]any) (map[string]any, error) {
		if tl.Name() != "note" {
			return nil, nil
		}
		close(started)
		select {
		case <-ctx.Done():
			saw.Store(ctx.Err())
		case <-time.After(pcsSiblingWait):
		}
		return nil, nil
	}
	h := newPCSHarness(t, pcsConfig{
		tools:     []tool.Tool{pcsAskTool(t, started), note},
		responses: []*genai.Content{pcsCalls(pcsCall("call_1", "ask"), pcsCall("call_2", "note"))},
		before:    waitInBefore,
	})

	if err := h.run(t.Context(), pcsUserText("go")); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if got, _ := saw.Load().(error); got != context.Canceled {
		t.Errorf("before-tool callback saw ctx.Err() = %v, want context.Canceled", got)
	}
	if got := runs.Load(); got != 0 {
		t.Errorf("note runs = %d, want 0", got)
	}
}

// Fails if a sequential call that waits for a paused earlier call runs its
// callbacks or its tool once that call returns. Its turn comes only after the
// paused call has stopped the step, so the outcome does not depend on whether
// it was already waiting.
func TestPause_WaitingSequentialSiblingRunsNothing(t *testing.T) {
	var runs atomic.Int32
	note := pcsFunctionTool(t, functiontool.Config{Name: "note", Sequential: true}, func(ctx agent.Context, _ pcsArgs) (map[string]any, error) {
		runs.Add(1)
		return map[string]any{"ok": true}, ctx.State().Set("k", "v")
	})
	counts := &pcsCounts{watched: "note"}
	h := newPCSHarness(t, pcsConfig{
		tools:     []tool.Tool{pcsAskTool(t, nil), note},
		responses: []*genai.Content{pcsCalls(pcsCall("call_1", "ask"), pcsCall("call_2", "note"))},
		counts:    counts,
	})

	if err := h.run(t.Context(), pcsUserText("go")); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if got := runs.Load(); got != 0 {
		t.Errorf("note runs = %d, want 0", got)
	}
	before, onError, after := counts.get()
	if diff := cmp.Diff([3]int{0, 0, 0}, [3]int{before, onError, after}); diff != "" {
		t.Errorf("note callbacks (before, on-error, after) mismatch (-want +got):\n%s", diff)
	}
	pcsStateK(t, h)
}

// Fails if an error that ends the run, here a generator tool's second output,
// leaves a running sibling call to finish on its own.
func TestFailRun_CancelsRunningSibling(t *testing.T) {
	var saw atomic.Value
	started := make(chan struct{})
	gen, err := generatortool.New(functiontool.Config{Name: "gen"}, func(_ agent.Context, _ pcsArgs, emit func(*session.Event) error) (any, error) {
		pcsAwait(started)
		if err := emit(&session.Event{Output: 1}); err != nil {
			return nil, err
		}
		return nil, emit(&session.Event{Output: 2})
	})
	if err != nil {
		t.Fatalf("generatortool.New() error = %v", err)
	}
	h := newPCSHarness(t, pcsConfig{
		tools:     []tool.Tool{gen, pcsWaitTool(t, &saw, started)},
		responses: []*genai.Content{pcsCalls(pcsCall("call_1", "gen"), pcsCall("call_2", "wait"))},
	})

	err = h.run(t.Context(), pcsUserText("go"))

	if !errors.Is(err, workflow.ErrMultipleOutputs) {
		t.Errorf("Run() error = %v, want one matching workflow.ErrMultipleOutputs", err)
	}
	if got, _ := saw.Load().(error); got != context.Canceled {
		t.Errorf("wait saw ctx.Err() = %v, want context.Canceled", got)
	}
	pcsStateK(t, h)
}

// texts returns "<node path> <text>" for every stored event of author with
// a text part.
func (h *pcsHarness) texts(author string) []string {
	h.t.Helper()
	var out []string
	for ev := range h.session().Events().All() {
		if ev.Author != author || ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.Text == "" {
				continue
			}
			path := ""
			if ev.NodeInfo != nil {
				path = ev.NodeInfo.Path
			}
			out = append(out, path+" "+p.Text)
		}
	}
	return out
}

// pcsAnswer is the user's answer "APAC" to the request for input region.
func pcsAnswer() *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: "region", Name: workflow.WorkflowInputFunctionCallName, Response: map[string]any{"result": "APAC"},
	}}}}
}

// Fails if a generator tool call running when its sibling pauses is not
// canceled: it then runs to its end and stores the state key k it wrote, or,
// after the answer, is answered from what its first run left instead of
// running again on the call's own path.
func TestPause_CanceledSiblingRunsAgainOnResume(t *testing.T) {
	var runs atomic.Int32
	started := make(chan struct{})
	slow, err := generatortool.New(functiontool.Config{Name: "slow"}, func(ctx agent.Context, _ pcsArgs, emit func(*session.Event) error) (any, error) {
		n := runs.Add(1)
		if err := emit(&session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(fmt.Sprintf("p%d", n), genai.RoleModel)}}); err != nil {
			return nil, err
		}
		if n > 1 {
			return "r2", nil
		}
		if err := ctx.State().Set("k", "v"); err != nil {
			return nil, err
		}
		close(started)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(pcsSiblingWait):
		}
		return "r1", nil
	})
	if err != nil {
		t.Fatalf("generatortool.New() error = %v", err)
	}
	h := newPCSHarness(t, pcsConfig{
		tools:     []tool.Tool{pcsAskTool(t, started), slow},
		responses: []*genai.Content{pcsCalls(pcsCall("call_1", "ask"), pcsCall("call_2", "slow")), genai.NewContentFromText("done", genai.RoleModel)},
	})

	if err := h.run(t.Context(), pcsUserText("go")); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	pcsStateK(t, h)
	if err := h.run(t.Context(), pcsAnswer()); err != nil {
		t.Fatalf("Run() after the answer error = %v", err)
	}

	if got := runs.Load(); got != 2 {
		t.Errorf("slow runs = %d, want 2", got)
	}
	want := []*genai.FunctionResponse{
		{ID: "call_1", Name: "ask", Response: map[string]any{"result": "APAC"}},
		{ID: "call_2", Name: "slow", Response: map[string]any{"result": "r2"}},
	}
	if diff := cmp.Diff(want, h.responses()); diff != "" {
		t.Errorf("stored function responses mismatch (-want +got):\n%s", diff)
	}
	// The canceled run left no output, so the second run takes the call's own
	// path, as every Python call does (run id = call id).
	wantTexts := []string{"root_agent/slow@call_2 p1", "root_agent/slow@call_2 p2"}
	if diff := cmp.Diff(wantTexts, h.texts("slow")); diff != "" {
		t.Errorf("slow text events mismatch (-want +got):\n%s", diff)
	}
}
