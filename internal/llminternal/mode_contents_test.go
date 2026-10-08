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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// An agent that declares no mode gets one from its placement, and the
// contents processor must honour the mode bound to THAT agent rather than
// the agent's blank declaration: a single_turn placement sees the current
// turn only, a chat placement sees the whole conversation. A binding made
// for a different agent must be ignored.
func TestContentsRequestProcessor_HonoursResolvedModeOverDeclaration(t *testing.T) {
	const agentName = "worker"

	history := []*session.Event{
		{
			Author:      "user",
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("older turn", "user")},
		},
		{
			Author:      agentName,
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("older answer", "model")},
		},
		{
			Author:      "user",
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("current turn", "user")},
		},
	}

	tests := []struct {
		name        string
		resolved    llminternal.Mode
		wantHistory bool
	}{
		{name: "single_turn placement drops history", resolved: llminternal.ModeSingleTurn, wantHistory: false},
		{name: "chat placement keeps history", resolved: llminternal.ModeChat, wantHistory: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			testAgent := utils.Must(llmagent.New(llmagent.Config{
				Name:  agentName,
				Model: &testModel{},
			}))

			ctx := icontext.NewInvocationContext(
				llminternal.WithBoundMode(t.Context(), agentName, stateOf(t, testAgent), tc.resolved),
				icontext.InvocationContextParams{
					Agent:   testAgent,
					Session: &fakeSession{events: history},
				},
			)

			req := &model.LLMRequest{}
			for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
				if err != nil {
					t.Fatalf("ContentsRequestProcessor: %v", err)
				}
			}

			var gotHistory bool
			for _, c := range req.Contents {
				for _, p := range c.Parts {
					if p != nil && p.Text == "older turn" {
						gotHistory = true
					}
				}
			}
			if gotHistory != tc.wantHistory {
				t.Errorf("history present = %v, want %v (contents: %v)", gotHistory, tc.wantHistory, req.Contents)
			}
		})
	}
}

// A mode bound for one agent must not reach another: the second agent falls
// back to its own declaration.
func TestContentsRequestProcessor_BoundModeIsScopedToItsAgent(t *testing.T) {
	// The agent's own reply has to sit between the two user turns: the
	// backward scan in buildContentsCurrentTurnContextOnly skips this agent's
	// events, so a history that ends on one pivots at index 0 and returns
	// everything whether or not history is being hidden.
	history := []*session.Event{
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("older turn", "user")}},
		{Author: "other", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("older answer", "model")}},
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("current turn", "user")}},
	}
	testAgent := utils.Must(llmagent.New(llmagent.Config{Name: "other", Model: &testModel{}}))

	// single_turn was resolved under the name "worker", not the name of the
	// agent running here. The IDENTITY is this agent's, so the name is the only
	// thing that differs and the only thing this test can be passing on.
	ctx := icontext.NewInvocationContext(
		llminternal.WithBoundMode(t.Context(), "worker", stateOf(t, testAgent), llminternal.ModeSingleTurn),
		icontext.InvocationContextParams{Agent: testAgent, Session: &fakeSession{events: history}},
	)

	req := &model.LLMRequest{}
	for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if err != nil {
			t.Fatalf("ContentsRequestProcessor: %v", err)
		}
	}
	var sawHistory bool
	for _, c := range req.Contents {
		for _, p := range c.Parts {
			if p != nil && p.Text == "older turn" {
				sawHistory = true
			}
		}
	}
	if !sawHistory {
		t.Errorf("a mode bound for another agent suppressed this agent's history; contents = %v", req.Contents)
	}
}

// Declaring single_turn shapes the turn, but it does not by itself hide the
// conversation: only a placement that put this agent in that mode does. A
// declared single_turn agent reached some other way keeps its history — a
// child run directly by a sequential, loop or parallel agent, say, which calls
// subAgent.Run and never goes through the node wrapper. Transfer is not that
// way: isUntransferableMode keeps a declared single_turn agent off the sub-agent
// and peer target lists, and the one route that does reach it, a child
// transferring back up to it, goes through the wrapper, which re-binds.
func TestContentsRequestProcessor_DeclaredSingleTurnWithoutPlacementKeepsHistory(t *testing.T) {
	history := []*session.Event{
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("older turn", "user")}},
		{Author: "worker", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("older answer", "model")}},
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("current turn", "user")}},
	}
	testAgent := utils.Must(llmagent.New(llmagent.Config{
		Name:  "worker",
		Model: &testModel{},
		Mode:  llmagent.ModeSingleTurn,
	}))

	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:   testAgent,
		Session: &fakeSession{events: history},
	})
	req := &model.LLMRequest{}
	for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if err != nil {
			t.Fatalf("ContentsRequestProcessor: %v", err)
		}
	}
	var sawHistory bool
	for _, c := range req.Contents {
		for _, p := range c.Parts {
			if p != nil && p.Text == "older turn" {
				sawHistory = true
			}
		}
	}
	if !sawHistory {
		t.Errorf("declared single_turn outside a placement lost its history; contents = %v", req.Contents)
	}
}

// Shaping the turn is the other half of the resolution, and it does consult
// the declaration: an agent that declares nothing and is placed single_turn is
// told it will get no follow-up questions, the same as one that declares the
// mode. The nudge only reaches a scoped run, which is where a delegated
// single_turn agent lives.
func TestContentsRequestProcessor_ResolvedSingleTurnGetsTheNudge(t *testing.T) {
	const agentName = "worker"

	tests := []struct {
		name     string
		declared llmagent.Mode
		bind     bool
		want     bool
	}{
		{name: "placement resolves it", bind: true, want: true},
		{name: "declaration alone", declared: llmagent.ModeSingleTurn, want: true},
		{name: "neither", want: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			testAgent := utils.Must(llmagent.New(llmagent.Config{
				Name:  agentName,
				Model: &testModel{},
				Mode:  tc.declared,
			}))

			stdCtx := t.Context()
			if tc.bind {
				stdCtx = llminternal.WithBoundMode(stdCtx, agentName, stateOf(t, testAgent), llminternal.ModeSingleTurn)
			}
			ctx := icontext.NewInvocationContext(stdCtx, icontext.InvocationContextParams{
				Agent:          testAgent,
				Session:        &fakeSession{},
				IsolationScope: "scope-1",
				UserContent:    genai.NewContentFromText("do the thing", "user"),
			})

			req := &model.LLMRequest{}
			for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
				if err != nil {
					t.Fatalf("ContentsRequestProcessor: %v", err)
				}
			}

			var gotNudge bool
			for _, c := range req.Contents {
				for _, p := range c.Parts {
					if p != nil && p.Text == llminternal.SingleTurnNudge {
						gotNudge = true
					}
				}
			}
			if gotNudge != tc.want {
				t.Errorf("single-turn nudge present = %v, want %v (contents: %v)", gotNudge, tc.want, req.Contents)
			}
		})
	}
}

// A binding found under this agent's name is not necessarily this agent's.
// Names are unique only across SubAgents(), and a graph node's agent is not in
// SubAgents(), so a chat agent can share a name with one a placement resolved
// single_turn for. Its own declaration says it is conversational, so the
// foreign placement must not take its history away.
//
// This was the direction that separated a name-only lookup from one that also
// checks which agent the binding was resolved for: with the name alone, every
// other test in the suite stayed green while this agent lost its history.
func TestContentsRequestProcessor_AForeignSingleTurnBindingDoesNotHideAChatAgentsHistory(t *testing.T) {
	// The agent's own reply sits between the two user turns, as above: a
	// history ending on one pivots at index 0 and returns everything whether
	// or not history is being hidden, which would make this vacuous.
	history := []*session.Event{
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("older turn", "user")}},
		{Author: "worker", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("older answer", "model")}},
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("current turn", "user")}},
	}
	testAgent := utils.Must(llmagent.New(llmagent.Config{
		Name:  "worker",
		Model: &testModel{},
		Mode:  llmagent.ModeChat,
	}))

	// single_turn resolved for a DIFFERENT agent that happens to be called
	// "worker" too — the collision runner.New does not reject. A distinct
	// *State is exactly what makes it a different agent.
	ctx := icontext.NewInvocationContext(
		llminternal.WithBoundMode(t.Context(), "worker", &llminternal.State{}, llminternal.ModeSingleTurn),
		icontext.InvocationContextParams{Agent: testAgent, Session: &fakeSession{events: history}},
	)

	req := &model.LLMRequest{}
	for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if err != nil {
			t.Fatalf("ContentsRequestProcessor: %v", err)
		}
	}
	if len(req.Contents) == 0 {
		t.Fatal("no contents were built, so this test pins nothing")
	}
	var sawHistory bool
	for _, c := range req.Contents {
		for _, p := range c.Parts {
			if p != nil && p.Text == "older turn" {
				sawHistory = true
			}
		}
	}
	if !sawHistory {
		t.Errorf("a single_turn placement resolved for another agent hid this chat agent's history; contents = %v", req.Contents)
	}
}

// stateOf returns the llminternal state behind an LlmAgent — the same pointer a
// binder and a reader both see, and therefore the agent's identity as far as a
// mode binding is concerned.
func stateOf(t *testing.T, a agent.Agent) *llminternal.State {
	t.Helper()
	internal, ok := a.(llminternal.Agent)
	if !ok {
		t.Fatalf("agent %q is not an llminternal.Agent", a.Name())
	}
	return llminternal.Reveal(internal)
}
