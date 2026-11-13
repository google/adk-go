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
	"fmt"
	"slices"
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
)

// embeddingWrapper is a user type that embeds agent.Agent, as the Agent
// interface documents for wrappers.
type embeddingWrapper struct {
	agent.Agent
}

func (w embeddingWrapper) FindAgent(name string) agent.Agent {
	if w.Name() == name {
		return w
	}
	return w.FindSubAgent(name)
}

func mustNewAgent(t *testing.T, cfg agent.Config) agent.Agent {
	t.Helper()
	a, err := agent.New(cfg)
	if err != nil {
		t.Fatalf("agent.New(%q) error = %v", cfg.Name, err)
	}
	return a
}

// parentOf returns the parent link recorded on the inner agent a, which must
// be a value agent.New returned.
func parentOf(t *testing.T, a agent.Agent) any {
	t.Helper()
	inner, ok := a.(agentinternal.Agent)
	if !ok {
		t.Fatalf("%T is not an inner agent", a)
	}
	parent := agentinternal.Reveal(inner).Parent
	if parent == nil {
		return nil
	}
	return parent
}

func TestNew_SetsParentOnEverySubAgent(t *testing.T) {
	plain := mustNewAgent(t, agent.Config{Name: "plain"})
	wrappedInner := mustNewAgent(t, agent.Config{Name: "wrapped"})

	parent := mustNewAgent(t, agent.Config{
		Name:      "parent",
		SubAgents: []agent.Agent{plain, embeddingWrapper{wrappedInner}},
	})

	if got := parentOf(t, plain); got != any(parent) {
		t.Errorf("Parent of plain sub-agent = %v, want the parent agent", got)
	}
	if got := parentOf(t, wrappedInner); got != any(parent) {
		t.Errorf("Parent of wrapped sub-agent = %v, want the parent agent", got)
	}
	if got := parentOf(t, parent); got != nil {
		t.Errorf("Parent of root agent = %v, want nil", got)
	}
}

func TestNew_SecondParentWins_NoError(t *testing.T) {
	child := mustNewAgent(t, agent.Config{Name: "child"})

	if _, err := agent.New(agent.Config{Name: "parent1", SubAgents: []agent.Agent{child}}); err != nil {
		t.Fatalf("agent.New(parent1) error = %v, want nil", err)
	}
	parent2, err := agent.New(agent.Config{Name: "parent2", SubAgents: []agent.Agent{child}})
	if err != nil {
		t.Fatalf("agent.New(parent2) error = %v, want nil", err)
	}

	if got := parentOf(t, child); got != any(parent2) {
		t.Errorf("Parent of child = %v, want parent2", got)
	}
}

// Constructing several parents of one sub-agent at once is safe; run with
// -race. The sub-agent ends with one of them as its parent.
func TestNew_ConcurrentParentsOverOneSubAgent(t *testing.T) {
	child := mustNewAgent(t, agent.Config{Name: "child"})

	parents := make([]any, 8)
	var wg sync.WaitGroup
	for i := range parents {
		wg.Go(func() {
			p, err := agent.New(agent.Config{Name: fmt.Sprintf("parent-%d", i), SubAgents: []agent.Agent{child}})
			if err != nil {
				t.Errorf("agent.New(parent-%d) error = %v", i, err)
				return
			}
			parents[i] = p
		})
	}
	wg.Wait()

	got := parentOf(t, child)
	if slices.Contains(parents, got) {
		return
	}
	t.Errorf("Parent of child = %v, want one of the constructed parents", got)
}

func TestNew_FailedNewWritesNoParent(t *testing.T) {
	tests := []struct {
		name string
		cfg  func(child agent.Agent) agent.Config
	}{
		{
			name: "repeated sub-agent",
			cfg: func(child agent.Agent) agent.Config {
				return agent.Config{Name: "parent", SubAgents: []agent.Agent{child, child}}
			},
		},
		{
			name: "reserved name user",
			cfg: func(child agent.Agent) agent.Config {
				return agent.Config{Name: "user", SubAgents: []agent.Agent{child}}
			},
		},
		{
			name: "empty name",
			cfg: func(child agent.Agent) agent.Config {
				return agent.Config{SubAgents: []agent.Agent{child}}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			child := mustNewAgent(t, agent.Config{Name: "child"})

			if _, err := agent.New(tt.cfg(child)); err == nil {
				t.Fatal("agent.New() error = nil, want an error")
			}

			if got := parentOf(t, child); got != nil {
				t.Errorf("Parent of child = %v, want nil", got)
			}
		})
	}
}

func TestNew_NilSubAgentIsAccepted(t *testing.T) {
	if _, err := agent.New(agent.Config{Name: "parent", SubAgents: []agent.Agent{nil}}); err != nil {
		t.Errorf("agent.New() with a nil sub-agent error = %v, want nil", err)
	}
}

// A constructor that wraps an agent (llmagent.New) builds it unlinked and
// publishes the link last, so a concurrent walk never sees the parent
// without its owner.
func TestNewUnlinked_PublishesTheLinkOnlyWhenLinkIsCalled(t *testing.T) {
	child := mustNewAgent(t, agent.Config{Name: "child"})
	childState := agentinternal.InnerState(embeddingWrapper{child})

	parent, link, err := agentinternal.NewUnlinked(agent.Config{Name: "parent", SubAgents: []agent.Agent{child}})
	if err != nil {
		t.Fatalf("NewUnlinked() error = %v", err)
	}
	if got, _ := agentinternal.ParentOf(childState); got != nil {
		t.Errorf("Parent of child before link = %v, want nil", got)
	}

	owner := embeddingWrapper{parent.(agent.Agent)}
	link(owner)

	gotParent, gotOwner := agentinternal.ParentOf(childState)
	if gotParent != parent {
		t.Errorf("Parent of child after link = %v, want the unlinked parent", gotParent)
	}
	if gotOwner != any(owner) {
		t.Errorf("Owner of the parent after link = %v, want the owner passed to link", gotOwner)
	}
}
