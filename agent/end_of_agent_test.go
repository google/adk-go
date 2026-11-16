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

package agent_test

import (
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// endRecordingAgent returns an agent named "c1" that yields one text event
// "hi", calls agent.RecordEndOfAgent, and stores its run's run config in *rc.
func endRecordingAgent(t *testing.T, rc **runconfig.RunConfig) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: "c1",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				*rc = runconfig.FromContext(ctx)
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = ctx.Agent().Name()
				ev.Content = genai.NewContentFromText("hi", genai.RoleModel)
				if !yield(ev, nil) {
					return
				}
				agent.RecordEndOfAgent(ctx)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	return a
}

// runEndRecording runs a in a new session of a runner that is resumable or
// not, and returns the session's events other than the user's message, each
// rendered as "author:text".
func runEndRecording(t *testing.T, a agent.Agent, resumable bool) []string {
	t.Helper()
	ctx := t.Context()
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: sessions, Resumable: resumable})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	for _, err := range r.Run(ctx, "user", created.Session.ID(), genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}
	got, err := sessions.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: created.Session.ID()})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var events []string
	for ev := range got.Session.Events().All() {
		if ev.Author == "user" {
			continue
		}
		text := ""
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				text += p.Text
			}
		}
		events = append(events, ev.Author+":"+text)
	}
	return events
}

func TestRecordEndOfAgent(t *testing.T) {
	t.Run("resumable run marks the agent ended and writes nothing", func(t *testing.T) {
		var rc *runconfig.RunConfig
		events := runEndRecording(t, endRecordingAgent(t, &rc), true)

		if diff := cmp.Diff([]string{"c1:hi"}, events); diff != "" {
			t.Errorf("session events mismatch (-want +got):\n%s", diff)
		}
		if rc == nil {
			t.Fatal("the agent's run had no run config")
		}
		if !rc.AgentEnded("c1") {
			t.Error("AgentEnded(c1) = false; want true")
		}
		if st, ok := rc.AgentState("c1"); st != nil || ok {
			t.Errorf("AgentState(c1) = %v, %v; want nil, false", st, ok)
		}
	})

	t.Run("non-resumable run records nothing", func(t *testing.T) {
		var rc *runconfig.RunConfig
		events := runEndRecording(t, endRecordingAgent(t, &rc), false)

		if diff := cmp.Diff([]string{"c1:hi"}, events); diff != "" {
			t.Errorf("session events mismatch (-want +got):\n%s", diff)
		}
		if rc == nil {
			t.Fatal("the agent's run had no run config")
		}
		if rc.AgentEnded("c1") {
			t.Error("AgentEnded(c1) = true; want false")
		}
	})

	t.Run("nil agent is a no-op", func(t *testing.T) {
		rc := &runconfig.RunConfig{Resumable: true}
		ctx := icontext.NewInvocationContext(runconfig.ToContext(t.Context(), rc), icontext.InvocationContextParams{})
		agent.RecordEndOfAgent(ctx)
		if rc.AgentEnded("") {
			t.Error(`AgentEnded("") = true; want false`)
		}
	})

	t.Run("context without a run config is a no-op", func(t *testing.T) {
		var unused *runconfig.RunConfig
		ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: endRecordingAgent(t, &unused)})
		agent.RecordEndOfAgent(ctx)
	})

	t.Run("second call leaves the agent ended", func(t *testing.T) {
		var unused *runconfig.RunConfig
		rc := &runconfig.RunConfig{Resumable: true}
		ctx := icontext.NewInvocationContext(runconfig.ToContext(t.Context(), rc), icontext.InvocationContextParams{Agent: endRecordingAgent(t, &unused)})
		agent.RecordEndOfAgent(ctx)
		agent.RecordEndOfAgent(ctx)
		if !rc.AgentEnded("c1") {
			t.Error("AgentEnded(c1) after two calls = false; want true")
		}
	})
}
