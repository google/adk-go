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

package llminternal

import (
	"context"
	"iter"
	"testing"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	"google.golang.org/adk/v2/model"
)

// namedTestLLM is a model that only has a name.
type namedTestLLM struct{ name string }

func (m *namedTestLLM) Name() string { return m.name }

func (m *namedTestLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(func(*model.LLMResponse, error) bool) {}
}

// newLinkedLLMAgent builds an LLM agent linked the way llmagent.New links
// one: its base agent lists subAgents, State.Base is that base agent, and
// the base agent's Owner is the LLM agent.
func newLinkedLLMAgent(t *testing.T, name string, m model.LLM, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	base, err := agent.New(agent.Config{Name: name, SubAgents: subAgents})
	if err != nil {
		t.Fatal(err)
	}
	inner := base.(agentinternal.Agent)
	a := &mockLLMAgent{Agent: base, s: &State{Model: m, Base: inner}}
	agentinternal.Reveal(inner).Owner = a
	return a
}

func newCustomAgent(t *testing.T, name string, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{Name: name, SubAgents: subAgents})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func resolvedName(t *testing.T, ctx context.Context, a agent.Agent) string {
	t.Helper()
	m, err := ResolveModel(ctx, a)
	if err != nil {
		t.Fatalf("ResolveModel(%q): %v", a.Name(), err)
	}
	return m.Name()
}

func TestResolveModel_InRunListedAgentUsesTheRunTree(t *testing.T) {
	leaf := newLinkedLLMAgent(t, "leaf", nil)
	newLinkedLLMAgent(t, "static_parent", &namedTestLLM{name: "static-model"}, leaf)
	runParent := newLinkedLLMAgent(t, "run_parent", &namedTestLLM{name: "run-model"})
	ctx := parentmap.ToContext(t.Context(), parentmap.Map{"leaf": runParent})

	if got := resolvedName(t, ctx, leaf); got != "run-model" {
		t.Errorf("ResolveModel(leaf) in a run = %s, want run-model (the parent in the run's tree)", got)
	}
	if got := resolvedName(t, context.Background(), leaf); got != "static-model" {
		t.Errorf("ResolveModel(leaf) outside a run = %s, want static-model", got)
	}
}

func TestResolveModel_RunRootContinuesWithStaticLink(t *testing.T) {
	leaf := newLinkedLLMAgent(t, "leaf", nil)
	newLinkedLLMAgent(t, "static_parent", &namedTestLLM{name: "static-model"}, leaf)
	// leaf is the run's root: the map does not list it.
	ctx := parentmap.ToContext(t.Context(), parentmap.Map{})

	if got := resolvedName(t, ctx, leaf); got != "static-model" {
		t.Errorf("ResolveModel(run root) = %s, want static-model", got)
	}
}

func TestResolveModel_StaticWalkNeverConsultsTheRunTree(t *testing.T) {
	// The run tree is leaf -> root (a custom agent, the run's root). From
	// root the walk follows the static links: root -> middle -> top.
	leaf := newLinkedLLMAgent(t, "leaf", nil)
	root := newCustomAgent(t, "root")
	middle := newCustomAgent(t, "middle", root)
	newLinkedLLMAgent(t, "top", &namedTestLLM{name: "top-model"}, middle)
	other := newLinkedLLMAgent(t, "other", &namedTestLLM{name: "other-model"})
	// A map entry for middle that a walk consulting the map after root
	// would follow.
	ctx := parentmap.ToContext(t.Context(), parentmap.Map{"leaf": root, "middle": other})

	if got := resolvedName(t, ctx, leaf); got != "top-model" {
		t.Errorf("ResolveModel(leaf) = %s, want top-model (static links from the run's root)", got)
	}
}
