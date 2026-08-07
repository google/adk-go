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
				llminternal.WithBoundMode(t.Context(), agentName, tc.resolved),
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
	history := []*session.Event{
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("older turn", "user")}},
		{Author: "other", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("current turn", "user")}},
	}
	testAgent := utils.Must(llmagent.New(llmagent.Config{Name: "other", Model: &testModel{}}))

	// single_turn was resolved for "worker", not for the agent running here.
	ctx := icontext.NewInvocationContext(
		llminternal.WithBoundMode(t.Context(), "worker", llminternal.ModeSingleTurn),
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
// conversation: only a placement that seeded the agent with one synthetic turn
// does that. A single_turn agent run outside such a placement keeps its history.
func TestContentsRequestProcessor_DeclaredSingleTurnWithoutPlacementKeepsHistory(t *testing.T) {
	history := []*session.Event{
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("older turn", "user")}},
		{Author: "worker", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("current turn", "user")}},
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
