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

package agenttool_test

import (
	"iter"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// TestAgentTool_Run_Memory checks that a sub-agent run through an agent tool
// searches and writes the caller's memory service, as in adk-python
// (src/google/adk/tools/agent_tool.py, 4e0ac64).
func TestAgentTool_Run_Memory(t *testing.T) {
	store := memory.InMemoryService()
	sessions := session.InMemoryService()
	past, err := sessions.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "past"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := sessions.AppendEvent(t.Context(), past.Session, &session.Event{
		ID:          "e1",
		Author:      "user",
		Timestamp:   time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("my favourite city is Paris", genai.RoleUser)},
	}); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}
	if err := store.AddSessionToMemory(t.Context(), past.Session); err != nil {
		t.Fatalf("AddSessionToMemory: %v", err)
	}

	var childFound int
	var childSearchErr, childAddErr error
	var childApp string
	child, err := agent.New(agent.Config{
		Name: "child",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				resp, err := ctx.Memory().SearchMemory(ctx, "Paris")
				childSearchErr = err
				if resp != nil {
					childFound = len(resp.Memories)
				}
				event := session.NewEvent(ctx, ctx.InvocationID())
				event.Author = "child"
				event.Content = genai.NewContentFromText("noted London", genai.RoleModel)
				if !yield(event, nil) {
					return
				}
				childApp = ctx.Session().AppName()
				childAddErr = ctx.Memory().AddSessionToMemory(ctx, ctx.Session())
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName: "app", Agent: artifactParent(t, "parent", child), SessionService: sessions,
		MemoryService: store, AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	for _, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText("run", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	}

	if childSearchErr != nil || childFound != 1 {
		t.Errorf("child SearchMemory(%q) found %d memories (err %v), want the 1 stored by the caller's memory service", "Paris", childFound, childSearchErr)
	}
	if childAddErr != nil {
		t.Fatalf("child AddSessionToMemory: %v", childAddErr)
	}
	added, err := store.SearchMemory(t.Context(), &memory.SearchRequest{AppName: childApp, UserID: "user", Query: "London"})
	if err != nil {
		t.Fatalf("SearchMemory: %v", err)
	}
	if len(added.Memories) != 1 {
		t.Errorf("caller's memory service holds %d memories of the child's session, want 1", len(added.Memories))
	}
}

// TestAgentTool_Run_MemoryWithoutCallerService checks that a sub-agent still
// gets a working, empty memory when the caller has no memory service.
func TestAgentTool_Run_MemoryWithoutCallerService(t *testing.T) {
	var childSearchErr error
	var childSearched bool
	child, err := agent.New(agent.Config{
		Name: "child",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				childSearched = ctx.Memory() != nil
				if childSearched {
					_, childSearchErr = ctx.Memory().SearchMemory(ctx, "Paris")
				}
				event := session.NewEvent(ctx, ctx.InvocationID())
				event.Content = genai.NewContentFromText("done", genai.RoleModel)
				yield(event, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName: "app", Agent: artifactParent(t, "parent", child), SessionService: session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	for _, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText("run", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	}
	if !childSearched || childSearchErr != nil {
		t.Errorf("child memory: present %v, SearchMemory error %v; want a scratch store that searches without error", childSearched, childSearchErr)
	}
}
