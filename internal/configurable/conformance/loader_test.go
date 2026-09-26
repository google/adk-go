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

package conformance_test

import (
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/configurable/conformance"
)

func createTestAgent(t *testing.T, name string) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name:        name,
		Description: "test agent " + name,
	})
	if err != nil {
		t.Fatalf("failed to create test agent %q: %v", name, err)
	}
	return a
}

func TestNewConformanceAgentLoader(t *testing.T) {
	t.Parallel()

	t.Run("empty agent map", func(t *testing.T) {
		t.Parallel()
		loader, err := conformance.NewConformanceAgentLoader(map[string]agent.Agent{})
		if err != nil {
			t.Fatalf("unexpected error creating loader: %v", err)
		}
		if loader == nil {
			t.Fatal("expected non-nil loader")
		}

		names := loader.ListAgents()
		if len(names) != 0 {
			t.Errorf("expected 0 agents, got %d: %v", len(names), names)
		}

		if root := loader.RootAgent(); root != nil {
			t.Errorf("expected nil RootAgent for empty loader, got %v", root.Name())
		}
	})

	t.Run("nil agent map", func(t *testing.T) {
		t.Parallel()
		loader, err := conformance.NewConformanceAgentLoader(nil)
		if err != nil {
			t.Fatalf("unexpected error creating loader with nil map: %v", err)
		}
		if loader == nil {
			t.Fatal("expected non-nil loader")
		}

		names := loader.ListAgents()
		if len(names) != 0 {
			t.Errorf("expected 0 agents, got %d: %v", len(names), names)
		}
	})

	t.Run("multiple agents loaded", func(t *testing.T) {
		t.Parallel()
		agentC := createTestAgent(t, "charlie")
		agentA := createTestAgent(t, "alpha")
		agentB := createTestAgent(t, "bravo")

		agentMap := map[string]agent.Agent{
			"charlie": agentC,
			"alpha":   agentA,
			"bravo":   agentB,
		}

		loader, err := conformance.NewConformanceAgentLoader(agentMap)
		if err != nil {
			t.Fatalf("unexpected error creating loader: %v", err)
		}

		expectedNames := []string{"alpha", "bravo", "charlie"}
		gotNames := loader.ListAgents()
		if diff := cmp.Diff(expectedNames, gotNames); diff != "" {
			t.Errorf("ListAgents() mismatch (-want +got):\n%s", diff)
		}

		// RootAgent should return the first agent in alphabetical order ("alpha")
		root := loader.RootAgent()
		if root == nil {
			t.Fatal("expected non-nil RootAgent")
		}
		if root.Name() != "alpha" {
			t.Errorf("expected RootAgent to be 'alpha', got %q", root.Name())
		}
	})
}

func TestListAgents(t *testing.T) {
	t.Parallel()

	t.Run("alphabetical ordering", func(t *testing.T) {
		t.Parallel()
		agents := map[string]agent.Agent{
			"zebra":    createTestAgent(t, "zebra"),
			"apple":    createTestAgent(t, "apple"),
			"Mango":    createTestAgent(t, "Mango"),
			"banana":   createTestAgent(t, "banana"),
			"123agent": createTestAgent(t, "123agent"),
		}

		loader, err := conformance.NewConformanceAgentLoader(agents)
		if err != nil {
			t.Fatalf("failed to create loader: %v", err)
		}

		want := []string{"123agent", "Mango", "apple", "banana", "zebra"}
		got := loader.ListAgents()
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ListAgents() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("slice isolation and immutability", func(t *testing.T) {
		t.Parallel()
		agent1 := createTestAgent(t, "agent1")
		agent2 := createTestAgent(t, "agent2")

		loader, err := conformance.NewConformanceAgentLoader(map[string]agent.Agent{
			"agent1": agent1,
			"agent2": agent2,
		})
		if err != nil {
			t.Fatalf("failed to create loader: %v", err)
		}

		names1 := loader.ListAgents()
		if len(names1) != 2 {
			t.Fatalf("expected 2 agent names, got %d", len(names1))
		}

		// Mutate returned slice
		names1[0] = "corrupted_name"

		names2 := loader.ListAgents()
		want := []string{"agent1", "agent2"}
		if diff := cmp.Diff(want, names2); diff != "" {
			t.Errorf("loader state was mutated via returned slice (-want +got):\n%s", diff)
		}
	})
}

func TestLoadAgent(t *testing.T) {
	t.Parallel()

	agentX := createTestAgent(t, "agentX")
	agentY := createTestAgent(t, "agentY")

	loader, err := conformance.NewConformanceAgentLoader(map[string]agent.Agent{
		"agentX": agentX,
		"agentY": agentY,
	})
	if err != nil {
		t.Fatalf("failed to create loader: %v", err)
	}

	t.Run("happy path - load existing agent", func(t *testing.T) {
		t.Parallel()
		a, err := loader.LoadAgent("agentX")
		if err != nil {
			t.Fatalf("unexpected error loading agentX: %v", err)
		}
		if a == nil {
			t.Fatal("expected non-nil agent")
		}
		if a.Name() != "agentX" {
			t.Errorf("expected agent name 'agentX', got %q", a.Name())
		}
	})

	t.Run("error path - unknown agent", func(t *testing.T) {
		t.Parallel()
		a, err := loader.LoadAgent("unknownAgent")
		if err == nil {
			t.Fatalf("expected error for unknown agent, got nil agent %v", a)
		}

		// Verify error message describes missing agent and enumerates available agents
		errMsg := err.Error()
		if !strings.Contains(errMsg, "agent unknownAgent not found") {
			t.Errorf("error message missing expected prefix, got: %q", errMsg)
		}
		if !strings.Contains(errMsg, "agentX") || !strings.Contains(errMsg, "agentY") {
			t.Errorf("error message should enumerate available agents, got: %q", errMsg)
		}
	})

	t.Run("empty string lookup", func(t *testing.T) {
		t.Parallel()
		a, err := loader.LoadAgent("")
		if err == nil {
			t.Fatalf("expected error for empty string lookup, got agent %v", a)
		}
		if !strings.Contains(err.Error(), "agent  not found") {
			t.Errorf("unexpected error format: %v", err)
		}
	})

	t.Run("lookup on empty loader", func(t *testing.T) {
		t.Parallel()
		emptyLoader, err := conformance.NewConformanceAgentLoader(map[string]agent.Agent{})
		if err != nil {
			t.Fatalf("failed to create empty loader: %v", err)
		}
		a, err := emptyLoader.LoadAgent("agentX")
		if err == nil {
			t.Fatalf("expected error for lookup on empty loader, got agent %v", a)
		}
		if !strings.Contains(err.Error(), "Please specify one of those: []") {
			t.Errorf("expected empty available agent list in error, got: %v", err)
		}
	})

	t.Run("nil agent map entry", func(t *testing.T) {
		t.Parallel()
		nilMapLoader, err := conformance.NewConformanceAgentLoader(map[string]agent.Agent{
			"nilAgent": nil,
		})
		if err != nil {
			t.Fatalf("failed to create loader with nil map entry: %v", err)
		}
		a, err := nilMapLoader.LoadAgent("nilAgent")
		if err != nil {
			t.Fatalf("unexpected error loading registered nil agent: %v", err)
		}
		if a != nil {
			t.Errorf("expected nil agent for registered nil entry, got %v", a)
		}
	})

	t.Run("case sensitivity", func(t *testing.T) {
		t.Parallel()
		tests := []struct {
			name    string
			lookup  string
			wantErr bool
		}{
			{
				name:    "exact casing match",
				lookup:  "agentX",
				wantErr: false,
			},
			{
				name:    "different casing - lowercase",
				lookup:  "agentx",
				wantErr: true,
			},
			{
				name:    "different casing - uppercase",
				lookup:  "AGENTX",
				wantErr: true,
			},
		}

		for _, tc := range tests {
			tc := tc
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				a, err := loader.LoadAgent(tc.lookup)
				if (err != nil) != tc.wantErr {
					t.Errorf("LoadAgent(%q) error = %v, wantErr %v", tc.lookup, err, tc.wantErr)
				}
				if !tc.wantErr && a.Name() != tc.lookup {
					t.Errorf("LoadAgent(%q) name = %q, want %q", tc.lookup, a.Name(), tc.lookup)
				}
			})
		}
	})
}

func TestRootAgent(t *testing.T) {
	t.Parallel()

	t.Run("empty loader returns nil", func(t *testing.T) {
		t.Parallel()
		loader, err := conformance.NewConformanceAgentLoader(map[string]agent.Agent{})
		if err != nil {
			t.Fatalf("failed to create loader: %v", err)
		}
		if root := loader.RootAgent(); root != nil {
			t.Errorf("expected nil RootAgent for empty loader, got %v", root)
		}
	})

	t.Run("nil agent map returns nil", func(t *testing.T) {
		t.Parallel()
		loader, err := conformance.NewConformanceAgentLoader(nil)
		if err != nil {
			t.Fatalf("unexpected error creating loader: %v", err)
		}
		if root := loader.RootAgent(); root != nil {
			t.Errorf("expected nil RootAgent for nil map, got %v", root.Name())
		}
	})

	t.Run("first agent in sorted order is root", func(t *testing.T) {
		t.Parallel()
		agentB := createTestAgent(t, "beta")
		agentA := createTestAgent(t, "alpha")
		loader, err := conformance.NewConformanceAgentLoader(map[string]agent.Agent{
			"beta":  agentB,
			"alpha": agentA,
		})
		if err != nil {
			t.Fatalf("failed to create loader: %v", err)
		}
		root := loader.RootAgent()
		if root == nil || root.Name() != "alpha" {
			t.Errorf("expected RootAgent 'alpha', got %v", root)
		}
	})

	t.Run("first agent in sorted order is nil entry", func(t *testing.T) {
		t.Parallel()
		loader, err := conformance.NewConformanceAgentLoader(map[string]agent.Agent{
			"a_nil":   nil,
			"b_agent": createTestAgent(t, "b_agent"),
		})
		if err != nil {
			t.Fatalf("failed to create loader: %v", err)
		}
		root := loader.RootAgent()
		if root != nil {
			t.Errorf("expected nil RootAgent for nil entry, got %v", root)
		}
	})

	t.Run("key mismatch in agentMap returns nil", func(t *testing.T) {
		t.Parallel()
		agentX := createTestAgent(t, "agentX")
		m := map[string]agent.Agent{
			"agentX": agentX,
		}
		loader2, err := conformance.NewConformanceAgentLoader(m)
		if err != nil {
			t.Fatalf("unexpected error creating loader: %v", err)
		}
		delete(m, "agentX")
		if root := loader2.RootAgent(); root != nil {
			t.Errorf("expected nil RootAgent when key is missing from agentMap, got %v", root.Name())
		}
	})
}

func TestConcurrency(t *testing.T) {
	t.Parallel()

	agent1 := createTestAgent(t, "agent1")
	agent2 := createTestAgent(t, "agent2")
	agent3 := createTestAgent(t, "agent3")

	loader, err := conformance.NewConformanceAgentLoader(map[string]agent.Agent{
		"agent1": agent1,
		"agent2": agent2,
		"agent3": agent3,
	})
	if err != nil {
		t.Fatalf("failed to create loader: %v", err)
	}

	const goroutines = 20
	const iterations = 100

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				// Concurrent ListAgents
				names := loader.ListAgents()
				if len(names) != 3 {
					t.Errorf("goroutine %d: expected 3 names, got %d", id, len(names))
				}

				// Concurrent LoadAgent happy path
				a, err := loader.LoadAgent("agent2")
				if err != nil || a == nil || a.Name() != "agent2" {
					t.Errorf("goroutine %d: failed to load agent2: %v", id, err)
				}

				// Concurrent LoadAgent error path
				_, err = loader.LoadAgent("nonexistent")
				if err == nil {
					t.Errorf("goroutine %d: expected error for nonexistent agent", id)
				}

				// Concurrent RootAgent
				root := loader.RootAgent()
				if root == nil || root.Name() != "agent1" {
					t.Errorf("goroutine %d: unexpected RootAgent: %v", id, root)
				}
			}
		}(i)
	}

	wg.Wait()
}
