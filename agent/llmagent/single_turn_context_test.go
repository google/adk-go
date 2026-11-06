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

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
)

// A root chat agent that calls a single_turn sub-agent gets the sub-agent's
// output once, in the function response. The sub-agent's own events live on
// the tool's sub-branch and must not come back in later turns.
func TestSingleTurnSubAgent_TranscriptStaysOutOfCoordinatorContext(t *testing.T) {
	tests := []struct {
		name       string
		callID     string
		wantBranch string
	}{
		{name: "branch keyed by call id", callID: "fc1", wantBranch: "translator@fc1"},
		// RunNode rejects these as run ids, but the branch can still carry them.
		{name: "numeric call id", callID: "7", wantBranch: "translator@7"},
		{name: "call id with slash", callID: "a/b", wantBranch: "translator@a/b"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			innerModel := &testutil.MockModel{Responses: []*genai.Content{
				genai.NewContentFromText("hola", genai.RoleModel),
			}}
			translator, err := llmagent.New(llmagent.Config{
				Name:        "translator",
				Description: "Translates a phrase.",
				Model:       innerModel,
				Mode:        llmagent.ModeSingleTurn,
			})
			if err != nil {
				t.Fatal(err)
			}
			call := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID: tc.callID, Name: "translator", Args: map[string]any{"request": "hello"},
			}}}}
			coordModel := &testutil.MockModel{Responses: []*genai.Content{
				call,
				genai.NewContentFromText("done1", genai.RoleModel),
				genai.NewContentFromText("done2", genai.RoleModel),
			}}
			coordinator, err := llmagent.New(llmagent.Config{
				Name:      "coordinator",
				Model:     coordModel,
				Mode:      llmagent.ModeChat,
				SubAgents: []agent.Agent{translator},
			})
			if err != nil {
				t.Fatal(err)
			}

			r := testutil.NewTestAgentRunner(t, coordinator)
			events, err := testutil.CollectEvents(r.Run(t, "s1", "turn1"))
			if err != nil {
				t.Fatalf("turn1: %v", err)
			}
			if _, err := testutil.CollectEvents(r.Run(t, "s1", "turn2")); err != nil {
				t.Fatalf("turn2: %v", err)
			}

			var branches []string
			for _, ev := range events {
				if ev.Author == "translator" {
					branches = append(branches, ev.Branch)
				}
			}
			if diff := cmp.Diff([]string{tc.wantBranch}, branches); diff != "" {
				t.Errorf("translator event branches mismatch (-want +got):\n%s", diff)
			}

			if len(coordModel.Requests) != 3 {
				t.Fatalf("coordinator requests = %d, want 3", len(coordModel.Requests))
			}
			// The ids are client side only when the model did not assign them,
			// so the ones the model sent come back unchanged.
			resp := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID: tc.callID, Name: "translator", Response: map[string]any{"result": "hola"},
			}}}}
			want := []*genai.Content{
				genai.NewContentFromText("turn1", genai.RoleUser),
				call,
				resp,
				genai.NewContentFromText("done1", genai.RoleModel),
				genai.NewContentFromText("turn2", genai.RoleUser),
			}
			if diff := cmp.Diff(want, coordModel.Requests[2].Contents); diff != "" {
				t.Errorf("turn2 coordinator contents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Call ids come from the model, so one can repeat within an invocation. Each
// call must still run the sub-agent with its own args.
func TestSingleTurnSubAgent_ReusedCallIDRunsTwice(t *testing.T) {
	inner := &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromText("hola", genai.RoleModel),
		genai.NewContentFromText("bonjour", genai.RoleModel),
	}}
	translator, err := llmagent.New(llmagent.Config{
		Name: "translator", Description: "Translates.", Model: inner, Mode: llmagent.ModeSingleTurn,
	})
	if err != nil {
		t.Fatal(err)
	}
	call := func(req string) *genai.Content {
		return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: "call_0", Name: "translator", Args: map[string]any{"request": req},
		}}}}
	}
	coordModel := &testutil.MockModel{Responses: []*genai.Content{
		call("hello to Spanish"),
		call("hello to French"),
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	coordinator, err := llmagent.New(llmagent.Config{
		Name: "coordinator", Model: coordModel, Mode: llmagent.ModeChat, SubAgents: []agent.Agent{translator},
	})
	if err != nil {
		t.Fatal(err)
	}
	r := testutil.NewTestAgentRunner(t, coordinator)
	if _, err := testutil.CollectEvents(r.Run(t, "s1", "turn1")); err != nil {
		t.Fatal(err)
	}
	if len(inner.Requests) != 2 {
		t.Errorf("translator model called %d times, want 2", len(inner.Requests))
	}
}
