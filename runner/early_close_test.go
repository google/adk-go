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

package runner

import (
	"iter"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// TestRunner_EarlyBreakRunsAfterRunCallback checks that a consumer that stops
// ranging after the first event still gets the after-run callbacks, once, on
// every run path. adk-python runs after_run in the finally of the run, which
// also covers a caller that closes the generator early (runners.py,
// workflow/_node_runner_utils.py).
func TestRunner_EarlyBreakRunsAfterRunCallback(t *testing.T) {
	userMsg := genai.NewContentFromText("hi", genai.RoleUser)
	// run ranges the runner's events and returns the first one after a break.
	type run func(t *testing.T, r *Runner) *session.Event
	runAndBreak := func(opts ...RunOption) run {
		return func(t *testing.T, r *Runner) *session.Event {
			for ev, err := range r.Run(t.Context(), "u", "s", userMsg, agent.RunConfig{}, opts...) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
				return ev
			}
			t.Fatal("Run() yielded no event")
			return nil
		}
	}
	twoEvents := func(ctx agent.InvocationContext, author string) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			for _, text := range []string{"one", "two"} {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = author
				ev.LLMResponse.Content = genai.NewContentFromText(text, genai.RoleModel)
				if !yield(ev, nil) {
					return
				}
			}
		}
	}
	functionNode := func(t *testing.T) Config {
		return Config{Node: workflow.NewFunctionNode("fn", func(agent.Context, any) (string, error) { return "ok", nil }, workflow.NodeConfig{})}
	}

	tests := []struct {
		name       string
		config     func(t *testing.T) Config
		run        run
		wantAuthor string
	}{
		{
			name: "custom agent root",
			config: func(t *testing.T) Config {
				root, err := agent.New(agent.Config{
					Name: "custom",
					Run:  func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] { return twoEvents(ctx, "custom") },
				})
				if err != nil {
					t.Fatalf("agent.New() error = %v", err)
				}
				return Config{Agent: root}
			},
			run:        runAndBreak(),
			wantAuthor: "custom",
		},
		{
			name: "llm agent root calling a tool",
			config: func(t *testing.T) Config {
				noop, err := functiontool.New(functiontool.Config{Name: "noop", Description: "does nothing"},
					func(agent.Context, struct{}) (string, error) { return "ok", nil })
				if err != nil {
					t.Fatalf("functiontool.New() error = %v", err)
				}
				root, err := llmagent.New(llmagent.Config{
					Name:  "root",
					Model: &toolCallingModel{toolName: "noop"},
					Tools: []tool.Tool{noop},
				})
				if err != nil {
					t.Fatalf("llmagent.New() error = %v", err)
				}
				return Config{Agent: root}
			},
			run:        runAndBreak(),
			wantAuthor: "root",
		},
		{
			name:   "function node root",
			config: functionNode,
			run:    runAndBreak(),
		},
		{
			name: "live agent root",
			config: func(t *testing.T) Config {
				return Config{Agent: &mockLiveAgent{
					Agent: must(agent.New(agent.Config{Name: "live"})),
					runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
						return &dummyLiveSession{}, twoEvents(ctx, "live"), nil
					},
				}}
			},
			run: func(t *testing.T, r *Runner) *session.Event {
				_, events, err := r.RunLive(t.Context(), "u", "s", agent.LiveRunConfig{})
				if err != nil {
					t.Fatalf("RunLive() error = %v", err)
				}
				for ev, err := range events {
					if err != nil {
						t.Fatalf("RunLive() event error = %v", err)
					}
					return ev
				}
				t.Fatal("RunLive() yielded no event")
				return nil
			},
			wantAuthor: "live",
		},
		{
			name:       "function node root, break on the user message",
			config:     functionNode,
			run:        runAndBreak(WithYieldUserMessage()),
			wantAuthor: "user",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var afterRunCalls atomic.Int32
			p, err := plugin.New(plugin.Config{
				Name:             "count_after_run",
				AfterRunCallback: func(agent.InvocationContext) { afterRunCalls.Add(1) },
			})
			if err != nil {
				t.Fatalf("plugin.New() error = %v", err)
			}
			cfg := tc.config(t)
			cfg.AppName = "early_close_test"
			cfg.SessionService = session.InMemoryService()
			cfg.AutoCreateSession = true
			cfg.PluginConfig = PluginConfig{Plugins: []*plugin.Plugin{p}}
			r, err := New(cfg)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			first := tc.run(t, r)

			if tc.wantAuthor != "" && first.Author != tc.wantAuthor {
				t.Errorf("first event author = %q, want %q", first.Author, tc.wantAuthor)
			}
			if got := afterRunCalls.Load(); got != 1 {
				t.Errorf("after-run calls = %d, want 1", got)
			}
		})
	}
}
