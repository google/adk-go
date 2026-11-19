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

package llmagent_test

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

type plannerRun struct {
	events []*session.Event
	err    error
	// sess is the stored session, read back after the run.
	sess session.Session
}

// runPlannerAgent runs a once on a new in-memory session that holds history,
// and collects the events until the first error.
func runPlannerAgent(t *testing.T, a agent.Agent, history ...*session.Event) plannerRun {
	t.Helper()
	return runPlannerAgentWithPlugins(t, a, nil, history...)
}

// runPlannerAgentWithPlugins is runPlannerAgent with plugins installed on the
// runner.
func runPlannerAgentWithPlugins(t *testing.T, a agent.Agent, plugins []*plugin.Plugin, history ...*session.Event) plannerRun {
	t.Helper()
	ctx := t.Context()
	svc := session.InMemoryService()
	key := session.GetRequest{AppName: "test_app", UserID: "test_user", SessionID: "s1"}
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: key.AppName, UserID: key.UserID, SessionID: key.SessionID})
	if err != nil {
		t.Fatalf("session Create: %v", err)
	}
	for _, ev := range history {
		if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
			t.Fatalf("session AppendEvent: %v", err)
		}
	}
	r, err := runner.New(runner.Config{
		AppName:        key.AppName,
		Agent:          a,
		SessionService: svc,
		PluginConfig:   runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	var run plannerRun
	for ev, err := range r.Run(ctx, key.UserID, key.SessionID, genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			run.err = err
			break
		}
		run.events = append(run.events, ev)
	}
	got, err := svc.Get(ctx, &key)
	if err != nil {
		t.Fatalf("session Get: %v", err)
	}
	run.sess = got.Session
	return run
}

func newPlannerAgent(t *testing.T, cfg llmagent.Config) agent.Agent {
	t.Helper()
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	return a
}
