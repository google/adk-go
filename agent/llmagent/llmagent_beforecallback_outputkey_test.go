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
	"context"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// runWithBeforeAgentCallback runs a single turn of an llmagent configured with
// the given output key and before-agent callback. It returns the events the
// runner yielded and the session state read back from the session service.
func runWithBeforeAgentCallback(t *testing.T, outputKey string, cb agent.BeforeAgentCallback) ([]*session.Event, session.State) {
	t.Helper()
	ctx := context.Background()

	a, err := llmagent.New(llmagent.Config{
		Name:                 "a",
		Model:                &FakeLLM{},
		OutputKey:            outputKey,
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{cb},
	})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc})
	if err != nil {
		t.Fatal(err)
	}
	cr, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatal(err)
	}

	var events []*session.Event
	for ev, err := range r.Run(ctx, "u", cr.Session.ID(), genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}
	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: cr.Session.ID()})
	if err != nil {
		t.Fatal(err)
	}
	return events, got.Session.State()
}

func TestLLMAgent_OutputKeyWithBeforeAgentCallback(t *testing.T) {
	const cached = "cached answer"
	shortCircuit := func(agent.Context) (*genai.Content, error) {
		return genai.NewContentFromText(cached, genai.RoleModel), nil
	}
	passThrough := func(agent.Context) (*genai.Content, error) { return nil, nil }

	tests := []struct {
		name      string
		outputKey string
		cb        agent.BeforeAgentCallback
		wantState map[string]any
		wantText  string // text of the single event expected from the run
	}{
		{
			name:      "short-circuit content is saved under output_key",
			outputKey: "result",
			cb:        shortCircuit,
			wantState: map[string]any{"result": cached},
			wantText:  cached,
		},
		{
			name:      "short-circuit without output_key leaves state unchanged",
			outputKey: "",
			cb:        shortCircuit,
			wantState: map[string]any{},
			wantText:  cached,
		},
		{
			name:      "nil callback result runs the model and saves its response",
			outputKey: "result",
			cb:        passThrough,
			wantState: map[string]any{"result": "fake model response"},
			wantText:  "fake model response",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			events, state := runWithBeforeAgentCallback(t, tc.outputKey, tc.cb)

			if len(events) != 1 {
				t.Fatalf("got %d events, want 1 (the state delta must ride on the response event)", len(events))
			}
			if got := events[0].Content.Parts[0].Text; got != tc.wantText {
				t.Errorf("event text = %q, want %q", got, tc.wantText)
			}
			for k, want := range tc.wantState {
				if got, err := state.Get(k); err != nil || got != want {
					t.Errorf("state[%q] = %v (err %v), want %v", k, got, err, want)
				}
				if got := events[0].Actions.StateDelta[k]; got != want {
					t.Errorf("event StateDelta[%q] = %v, want %v", k, got, want)
				}
			}
			if len(tc.wantState) == 0 {
				if _, err := state.Get("result"); err == nil {
					t.Error("state[result] set, want it unset")
				}
				if len(events[0].Actions.StateDelta) != 0 {
					t.Errorf("event StateDelta has %d keys, want 0", len(events[0].Actions.StateDelta))
				}
			}
		})
	}
}
