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

package instructionutil_test

import (
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/util/instructionutil"
)

const (
	appName   = "app"
	userID    = "user"
	sessionID = "session"
)

// setup is the agent configuration and plugins of one run.
type setup struct {
	cfg     llmagent.Config
	plugins []*plugin.Plugin
}

// hook installs inject in one of the places where ADK hands a context to
// application code.
type hook func(t *testing.T, s *setup, inject func(agent.ReadonlyContext))

func echoTool(t *testing.T, handler func(agent.Context)) tool.Tool {
	t.Helper()
	echo, err := functiontool.New(functiontool.Config{Name: "echo", Description: "echo"},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			handler(ctx)
			return map[string]any{}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	return echo
}

// runAgent runs one turn of an agent whose model calls the echo tool once and
// then answers. The session starts with state; artifacts, if not nil, is the
// runner's artifact service.
func runAgent(t *testing.T, state map[string]any, artifacts artifact.Service, install hook, inject func(agent.ReadonlyContext)) {
	t.Helper()
	s := &setup{cfg: llmagent.Config{
		Name: "agent",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			genai.NewContentFromFunctionCall("echo", map[string]any{}, genai.RoleModel),
			genai.NewContentFromText("done", genai.RoleModel),
		}},
		Tools: []tool.Tool{echoTool(t, func(agent.Context) {})},
	}}
	install(t, s, inject)
	a, err := llmagent.New(s.cfg)
	if err != nil {
		t.Fatal(err)
	}

	sessions := session.InMemoryService()
	if _, err := sessions.Create(t.Context(), &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID, State: state}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:         appName,
		Agent:           a,
		SessionService:  sessions,
		ArtifactService: artifacts,
		PluginConfig:    runner.PluginConfig{Plugins: s.plugins},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range r.Run(t.Context(), userID, sessionID, genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() failed: %v", err)
		}
	}
}

func inBeforeModelCallback(_ *testing.T, s *setup, inject func(agent.ReadonlyContext)) {
	s.cfg.BeforeModelCallbacks = []llmagent.BeforeModelCallback{func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
		inject(ctx)
		return nil, nil
	}}
}

func TestInjectSessionState(t *testing.T) {
	for _, tc := range []struct {
		name    string
		install hook
	}{
		{
			name: "instruction provider",
			install: func(_ *testing.T, s *setup, inject func(agent.ReadonlyContext)) {
				s.cfg.InstructionProvider = func(ctx agent.ReadonlyContext) (string, error) {
					inject(ctx)
					return "", nil
				}
			},
		},
		{
			name: "before agent callback",
			install: func(_ *testing.T, s *setup, inject func(agent.ReadonlyContext)) {
				s.cfg.BeforeAgentCallbacks = []agent.BeforeAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
					inject(ctx)
					return nil, nil
				}}
			},
		},
		{
			name:    "before model callback",
			install: inBeforeModelCallback,
		},
		{
			name: "plugin before model callback",
			install: func(t *testing.T, s *setup, inject func(agent.ReadonlyContext)) {
				p, err := plugin.New(plugin.Config{Name: "inject", BeforeModelCallback: func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
					inject(ctx)
					return nil, nil
				}})
				if err != nil {
					t.Fatal(err)
				}
				s.plugins = append(s.plugins, p)
			},
		},
		{
			name: "before tool callback",
			install: func(_ *testing.T, s *setup, inject func(agent.ReadonlyContext)) {
				s.cfg.BeforeToolCallbacks = []llmagent.BeforeToolCallback{func(ctx agent.Context, _ tool.Tool, _ map[string]any) (map[string]any, error) {
					inject(ctx)
					return nil, nil
				}}
			},
		},
		{
			name: "tool",
			install: func(t *testing.T, s *setup, inject func(agent.ReadonlyContext)) {
				s.cfg.Tools = []tool.Tool{echoTool(t, func(ctx agent.Context) { inject(ctx) })}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			artifacts := artifact.InMemoryService()
			if _, err := artifacts.Save(t.Context(), &artifact.SaveRequest{
				AppName: appName, UserID: userID, SessionID: sessionID,
				FileName: "note.txt", Part: genai.NewPartFromText("the notes"),
			}); err != nil {
				t.Fatal(err)
			}

			called := false
			runAgent(t, map[string]any{"name": "Ada"}, artifacts, tc.install, func(ctx agent.ReadonlyContext) {
				called = true
				got, err := instructionutil.InjectSessionState(ctx, "{name} read {artifact.note.txt}{missing?}.")
				if err != nil {
					t.Fatalf("InjectSessionState() failed: %v", err)
				}
				if want := "Ada read the notes."; got != want {
					t.Errorf("InjectSessionState() = %q, want %q", got, want)
				}
			})
			if !called {
				t.Fatal("the hook was never called")
			}
		})
	}
}

// A value the callback set earlier is visible, as in adk-python, where setting
// a callback's state also sets the session's.
func TestInjectSessionState_StateSetInCallback(t *testing.T) {
	var got string
	runAgent(t, map[string]any{"name": "Ada"}, nil, inBeforeModelCallback, func(ctx agent.ReadonlyContext) {
		if err := ctx.(agent.Context).State().Set("name", "Grace"); err != nil {
			t.Fatal(err)
		}
		var err error
		if got, err = instructionutil.InjectSessionState(ctx, "{name}"); err != nil {
			t.Fatalf("InjectSessionState() failed: %v", err)
		}
	})
	if want := "Grace"; got != want {
		t.Errorf("InjectSessionState() = %q, want %q", got, want)
	}
}

func TestInjectSessionState_NoArtifactService(t *testing.T) {
	var err error
	runAgent(t, nil, nil, inBeforeModelCallback, func(ctx agent.ReadonlyContext) {
		_, err = instructionutil.InjectSessionState(ctx, "{artifact.note.txt}")
	})
	if err == nil || !strings.Contains(err.Error(), "artifact service is not initialized") {
		t.Errorf("InjectSessionState() error = %v, want an artifact service is not initialized error", err)
	}
}

// An agent.Context with no state, such as a fake built on agent.ContextMock,
// gets an error for a state placeholder rather than a panic.
func TestInjectSessionState_ContextWithoutState(t *testing.T) {
	type fakeContext struct {
		agent.ContextMock
	}
	ctx := &fakeContext{}
	if _, err := instructionutil.InjectSessionState(ctx, "{name}"); err == nil || !strings.Contains(err.Error(), "state is not available") {
		t.Errorf("InjectSessionState() error = %v, want a state is not available error", err)
	}
	const template = "No placeholders."
	if got, err := instructionutil.InjectSessionState(ctx, template); err != nil || got != template {
		t.Errorf("InjectSessionState() = (%q, %v), want (%q, nil)", got, err, template)
	}
}

type otherContext struct {
	agent.ReadonlyContext
}

func TestInjectSessionState_UnsupportedContext(t *testing.T) {
	if _, err := instructionutil.InjectSessionState(otherContext{}, "{name}"); err == nil {
		t.Error("InjectSessionState() succeeded for a context ADK did not create, want an error")
	}
}
