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
	"fmt"
	"iter"
	"slices"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// executionIDRecorder records what agent.ExecutionIDFromContext and the
// invocation ID read in each plugin callback of a run.
type executionIDRecorder struct {
	mu          sync.Mutex
	ids         map[string][]string
	invocations []string
}

func (r *executionIDRecorder) record(callback string, ctx context.Context) {
	id, ok := agent.ExecutionIDFromContext(ctx)
	if !ok {
		id = "<none>"
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ids == nil {
		r.ids = map[string][]string{}
	}
	r.ids[callback] = append(r.ids[callback], id)
}

// take returns what was recorded since the last call, and resets it.
func (r *executionIDRecorder) take() (ids map[string][]string, invocations []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	ids, invocations = r.ids, r.invocations
	r.ids, r.invocations = nil, nil
	return ids, invocations
}

func (r *executionIDRecorder) plugin(t *testing.T) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: "execution_ids",
		OnUserMessageCallback: func(ctx agent.InvocationContext, _ *genai.Content) (*genai.Content, error) {
			r.record("OnUserMessage", ctx)
			return nil, nil
		},
		BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) {
			r.record("BeforeRun", ctx)
			r.mu.Lock()
			r.invocations = append(r.invocations, ctx.InvocationID())
			r.mu.Unlock()
			return nil, nil
		},
		AfterRunCallback: func(ctx agent.InvocationContext) { r.record("AfterRun", ctx) },
		OnEventCallback: func(ctx agent.InvocationContext, _ *session.Event) (*session.Event, error) {
			r.record("OnEvent", ctx)
			return nil, nil
		},
		BeforeAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
			r.record("BeforeAgent", ctx)
			return nil, nil
		},
		AfterAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
			r.record("AfterAgent", ctx)
			return nil, nil
		},
		BeforeModelCallback: func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
			r.record("BeforeModel", ctx)
			return nil, nil
		},
		AfterModelCallback: func(ctx agent.Context, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			r.record("AfterModel", ctx)
			return nil, nil
		},
		BeforeToolCallback: func(ctx agent.Context, _ tool.Tool, _ map[string]any) (map[string]any, error) {
			r.record("BeforeTool", ctx)
			return nil, nil
		},
		AfterToolCallback: func(ctx agent.Context, _ tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
			r.record("AfterTool", ctx)
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p
}

// soleExecutionID checks that every callback in want recorded an execution
// ID and that all recorded IDs are the same, and returns that ID.
func soleExecutionID(t *testing.T, ids map[string][]string, want ...string) string {
	t.Helper()
	for _, callback := range want {
		if len(ids[callback]) == 0 {
			t.Errorf("%s callback did not run", callback)
		}
	}
	distinct := map[string]bool{}
	for callback, got := range ids {
		for _, id := range got {
			if id == "<none>" {
				t.Errorf("ExecutionIDFromContext() in %s = (\"\", false), want an ID", callback)
				continue
			}
			distinct[id] = true
		}
	}
	if len(distinct) != 1 {
		t.Fatalf("callbacks of one call read %d execution IDs %v, want 1; by callback: %v", len(distinct), distinct, ids)
	}
	for id := range distinct {
		return id
	}
	return ""
}

func drainRun(t *testing.T, events iter.Seq2[*session.Event, error]) {
	t.Helper()
	for _, err := range events {
		if err != nil {
			t.Fatalf("run error = %v", err)
		}
	}
}

// executionIDLiveAgent is an agent that runs live, as the runner's
// unexported liveAgent interface asks.
type executionIDLiveAgent struct {
	agent.Agent
	runLive func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
}

func (a *executionIDLiveAgent) RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	return a.runLive(ctx)
}

type executionIDLiveSession struct{}

func (executionIDLiveSession) Send(agent.LiveRequest) error { return nil }
func (executionIDLiveSession) Close() error                 { return nil }

func TestExecutionIDFromContext_OneIDPerRunnerCall(t *testing.T) {
	answer := func(ctx agent.InvocationContext, author string) *session.Event {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Author = author
		ev.Content = genai.NewContentFromText("answer", genai.RoleModel)
		return ev
	}
	type run struct {
		agent agent.Agent
		live  bool
		want  []string
	}
	cases := map[string]func(t *testing.T, rec *executionIDRecorder) run{
		"agent path": func(t *testing.T, rec *executionIDRecorder) run {
			child, err := agent.New(agent.Config{Name: "child", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					rec.record("child Run", ctx)
					yield(answer(ctx, "child"), nil)
				}
			}})
			if err != nil {
				t.Fatal(err)
			}
			root, err := agent.New(agent.Config{Name: "root", SubAgents: []agent.Agent{child}, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					rec.record("root Run", ctx)
					for ev, err := range child.Run(ctx) {
						if !yield(ev, err) {
							return
						}
					}
				}
			}})
			if err != nil {
				t.Fatal(err)
			}
			return run{agent: root, want: []string{"OnUserMessage", "BeforeRun", "BeforeAgent", "AfterAgent", "OnEvent", "AfterRun", "root Run", "child Run"}}
		},
		"node path": func(t *testing.T, rec *executionIDRecorder) run {
			echo, err := functiontool.New(functiontool.Config{Name: "echo", Description: "echoes"}, func(_ agent.Context, _ struct{}) (map[string]string, error) {
				return map[string]string{"echo": "ok"}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			m := &scriptedModel{responses: []*genai.Content{
				genai.NewContentFromFunctionCall("echo", map[string]any{}, genai.RoleModel),
				genai.NewContentFromText("done", genai.RoleModel),
				genai.NewContentFromFunctionCall("echo", map[string]any{}, genai.RoleModel),
				genai.NewContentFromText("done", genai.RoleModel),
			}}
			a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: m, Tools: []tool.Tool{echo}})
			if err != nil {
				t.Fatal(err)
			}
			return run{agent: a, want: []string{"OnUserMessage", "BeforeRun", "BeforeAgent", "AfterAgent", "BeforeModel", "AfterModel", "BeforeTool", "AfterTool", "OnEvent", "AfterRun"}}
		},
		"live": func(t *testing.T, rec *executionIDRecorder) run {
			base, err := agent.New(agent.Config{Name: "live"})
			if err != nil {
				t.Fatal(err)
			}
			a := &executionIDLiveAgent{Agent: base, runLive: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
				rec.record("RunLive", ctx)
				return executionIDLiveSession{}, func(yield func(*session.Event, error) bool) {
					yield(answer(ctx, "live"), nil)
				}, nil
			}}
			return run{agent: a, live: true, want: []string{"BeforeRun", "RunLive", "OnEvent", "AfterRun"}}
		},
	}
	for name, setup := range cases {
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			rec := &executionIDRecorder{}
			c := setup(t, rec)
			r, err := runner.New(runner.Config{
				AppName:           "app",
				Agent:             c.agent,
				SessionService:    session.InMemoryService(),
				AutoCreateSession: true,
				PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{rec.plugin(t)}},
			})
			if err != nil {
				t.Fatalf("runner.New() error = %v", err)
			}
			var seen []string
			for i := range 2 {
				if c.live {
					live, events, err := r.RunLive(ctx, "u", "s", agent.LiveRunConfig{})
					if err != nil {
						t.Fatalf("RunLive() error = %v", err)
					}
					drainRun(t, events)
					if err := live.Close(); err != nil {
						t.Fatalf("Close() error = %v", err)
					}
				} else {
					drainRun(t, r.Run(ctx, "u", "s", genai.NewContentFromText(fmt.Sprintf("question %d", i), genai.RoleUser), agent.RunConfig{}))
				}
				ids, _ := rec.take()
				seen = append(seen, soleExecutionID(t, ids, c.want...))
			}
			if seen[0] == seen[1] {
				t.Errorf("two runner calls share execution ID %q", seen[0])
			}
		})
	}
}

// TestExecutionIDFromContext_ResumeSharingInvocationID checks that a HITL
// resume, which reuses the invocation ID of the run that it answers, gets an
// execution ID of its own.
func TestExecutionIDFromContext_ResumeSharingInvocationID(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	ask, err := functiontool.New(functiontool.Config{Name: "ask_human", Description: "asks a human and waits", IsLongRunning: true},
		func(_ agent.Context, _ struct{}) (map[string]string, error) {
			return map[string]string{"status": "pending"}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	m := &scriptedModel{responses: []*genai.Content{
		genai.NewContentFromFunctionCall("ask_human", map[string]any{}, genai.RoleModel),
		genai.NewContentFromText("waiting for approval", genai.RoleModel),
		genai.NewContentFromText("all done", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "approver", Model: m, Tools: []tool.Tool{ask}})
	if err != nil {
		t.Fatal(err)
	}
	rec := &executionIDRecorder{}
	r, err := runner.New(runner.Config{
		AppName:        nodeTestApp,
		Agent:          a,
		SessionService: svc,
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{rec.plugin(t)}},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	var callID string
	for ev, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("please approve"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("paused run error = %v", err)
		}
		if ev != nil && len(ev.LongRunningToolIDs) > 0 {
			callID = ev.LongRunningToolIDs[0]
		}
	}
	if callID == "" {
		t.Fatal("the run did not pause on the long-running call")
	}
	pausedIDs, pausedInvocations := rec.take()
	paused := soleExecutionID(t, pausedIDs, "BeforeRun", "BeforeTool", "AfterRun")

	reply := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: callID, Name: "ask_human", Response: map[string]any{"output": "approved"},
	}}}}
	drainRun(t, r.Run(ctx, nodeTestUser, nodeTestSession, reply, agent.RunConfig{}))
	resumedIDs, resumedInvocations := rec.take()
	resumed := soleExecutionID(t, resumedIDs, "BeforeRun", "AfterRun")

	if len(pausedInvocations) != 1 || !slices.Equal(pausedInvocations, resumedInvocations) {
		t.Fatalf("invocation IDs = %v then %v, want one shared ID", pausedInvocations, resumedInvocations)
	}
	if paused == resumed {
		t.Errorf("the paused run and its resume share execution ID %q", paused)
	}
}

func TestExecutionIDFromContext_AbsentOutsideARunnerCall(t *testing.T) {
	var unset context.Context
	for name, ctx := range map[string]context.Context{"background": context.Background(), "nil": unset} {
		if id, ok := agent.ExecutionIDFromContext(ctx); ok || id != "" {
			t.Errorf("ExecutionIDFromContext(%s) = (%q, %t), want (\"\", false)", name, id, ok)
		}
	}
}

// TestExecutionIDFromContext_KeepsUUIDSequence checks that the execution ID
// draws no UUID: the invocation ID of a fresh turn is the first UUID that the
// run takes from the context's provider, as without it.
func TestExecutionIDFromContext_KeepsUUIDSequence(t *testing.T) {
	var mu sync.Mutex
	next := 0
	ctx := platform.WithUUIDProvider(t.Context(), func() string {
		mu.Lock()
		defer mu.Unlock()
		next++
		return fmt.Sprintf("uuid-%d", next)
	})
	rec := &executionIDRecorder{}
	root, err := agent.New(agent.Config{Name: "root", Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(func(*session.Event, error) bool) {}
	}})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:        "app",
		Agent:          root,
		SessionService: svc,
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{rec.plugin(t)}},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	drainRun(t, r.Run(ctx, "u", "s", nil, agent.RunConfig{}))
	ids, invocations := rec.take()
	soleExecutionID(t, ids, "BeforeRun", "AfterRun")
	if want := []string{"e-uuid-1"}; !slices.Equal(invocations, want) {
		t.Errorf("invocation IDs = %v, want %v", invocations, want)
	}
}

// executionIDSummarizer records the execution ID that each SummarizeEvents
// call reads and returns a fixed summary.
type executionIDSummarizer struct {
	rec *executionIDRecorder
}

func (s executionIDSummarizer) SummarizeEvents(ctx context.Context, _ []*session.Event) (compaction.SummarizeResult, error) {
	s.rec.record("Summarizer", ctx)
	return compaction.SummarizeResult{Content: genai.NewContentFromText("summary", genai.RoleModel)}, nil
}

// TestExecutionIDFromContext_PostInvocationCompaction checks that the
// post-invocation compaction of a call, which runs after the agent returns
// and outside every plugin callback, reads the call's execution ID on the
// agent path and on the node path alike.
func TestExecutionIDFromContext_PostInvocationCompaction(t *testing.T) {
	cases := map[string]func(t *testing.T) agent.Agent{
		"agent path": func(t *testing.T) agent.Agent {
			root, err := agent.New(agent.Config{Name: "root", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "root"
					ev.Content = genai.NewContentFromText("answer", genai.RoleModel)
					yield(ev, nil)
				}
			}})
			if err != nil {
				t.Fatal(err)
			}
			return root
		},
		"node path": func(t *testing.T) agent.Agent {
			a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: &scriptedModel{responses: []*genai.Content{
				genai.NewContentFromText("answer", genai.RoleModel),
			}}})
			if err != nil {
				t.Fatal(err)
			}
			return a
		},
	}
	for name, setup := range cases {
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			rec := &executionIDRecorder{}
			r, err := runner.New(runner.Config{
				AppName:           "app",
				Agent:             setup(t),
				SessionService:    session.InMemoryService(),
				AutoCreateSession: true,
				PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{rec.plugin(t)}},
				Compaction:        &compaction.Config{CompactionInterval: 1, Summarizer: executionIDSummarizer{rec: rec}},
			})
			if err != nil {
				t.Fatalf("runner.New() error = %v", err)
			}
			var seen []string
			for i := range 2 {
				drainRun(t, r.Run(ctx, "u", "s", genai.NewContentFromText(fmt.Sprintf("question %d", i), genai.RoleUser), agent.RunConfig{}))
				ids, _ := rec.take()
				seen = append(seen, soleExecutionID(t, ids, "BeforeRun", "AfterRun", "Summarizer"))
			}
			if seen[0] == seen[1] {
				t.Errorf("two runner calls share execution ID %q", seen[0])
			}
		})
	}
}

// TestExecutionIDFromContext_NestedAgentToolCall checks that the runner call
// an agent tool starts for its agent is its own call: the child's callbacks
// read an execution ID of their own, while the callbacks of the outer call
// that run after the tool returns still read the outer call's ID.
func TestExecutionIDFromContext_NestedAgentToolCall(t *testing.T) {
	// byAgent records the execution ID of each callback under the name of the
	// agent that the callback ran for.
	byAgent := &executionIDRecorder{}
	agentCallback := func(callback string) func(agent.Context) (*genai.Content, error) {
		return func(ctx agent.Context) (*genai.Content, error) {
			byAgent.record(callback+"@"+ctx.AgentName(), ctx)
			return nil, nil
		}
	}
	p, err := plugin.New(plugin.Config{
		Name: "nested_execution_ids",
		BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) {
			byAgent.record("BeforeRun@"+ctx.Agent().Name(), ctx)
			return nil, nil
		},
		AfterRunCallback:    func(ctx agent.InvocationContext) { byAgent.record("AfterRun@"+ctx.Agent().Name(), ctx) },
		BeforeAgentCallback: agentCallback("BeforeAgent"),
		AfterAgentCallback:  agentCallback("AfterAgent"),
		BeforeModelCallback: func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
			byAgent.record("BeforeModel@"+ctx.AgentName(), ctx)
			return nil, nil
		},
		AfterToolCallback: func(ctx agent.Context, _ tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
			byAgent.record("AfterTool@"+ctx.AgentName(), ctx)
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	child, err := llmagent.New(llmagent.Config{
		Name:        "child",
		Description: "answers the delegated request",
		Model:       &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("child answer", genai.RoleModel)}},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name: "root",
		Model: &scriptedModel{responses: []*genai.Content{
			callContent("child", map[string]any{"request": "delegated"}),
			genai.NewContentFromText("done", genai.RoleModel),
		}},
		Tools: []tool.Tool{agenttool.New(child, nil)},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             root,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	drainRun(t, r.Run(t.Context(), "u", "s", genai.NewContentFromText("question", genai.RoleUser), agent.RunConfig{}))
	ids, _ := byAgent.take()

	one := func(callback string) string {
		t.Helper()
		got := ids[callback]
		if len(got) == 0 {
			t.Fatalf("%s callback did not run; recorded: %v", callback, ids)
		}
		for _, id := range got[1:] {
			if id != got[0] {
				t.Fatalf("%s callback read execution IDs %v, want one", callback, got)
			}
		}
		if got[0] == "<none>" {
			t.Fatalf("ExecutionIDFromContext() in %s = (\"\", false), want an ID", callback)
		}
		return got[0]
	}
	outer := one("BeforeRun@root")
	for _, callback := range []string{"BeforeAgent@root", "BeforeModel@root", "AfterTool@root", "AfterAgent@root", "AfterRun@root"} {
		if got := one(callback); got != outer {
			t.Errorf("%s read execution ID %q, want the outer call's %q", callback, got, outer)
		}
	}
	inner := one("BeforeAgent@child")
	if got := one("BeforeModel@child"); got != inner {
		t.Errorf("BeforeModel@child read execution ID %q, want the nested call's %q", got, inner)
	}
	if inner == outer {
		t.Errorf("the nested agent tool call shares the outer call's execution ID %q", outer)
	}
}
