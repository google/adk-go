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
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// withParents returns ctx carrying the parent map of the tree rooted at root.
func withParents(t *testing.T, ctx context.Context, root agent.Agent) context.Context {
	t.Helper()
	parents, err := parentmap.New(root)
	if err != nil {
		t.Fatal(err)
	}
	return parentmap.ToContext(ctx, parents)
}

// transferRequest runs AgentTransferRequestProcessor for cur on ctx and
// returns the request it built. Any yielded event or error fails the test.
func transferRequest(t *testing.T, ctx context.Context, cur agent.Agent) *model.LLMRequest {
	t.Helper()
	ictx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Agent: cur})
	req := &model.LLMRequest{}
	for ev, err := range llminternal.AgentTransferRequestProcessor(ictx, req, &llminternal.Flow{}) {
		if ev != nil {
			t.Fatalf("AgentTransferRequestProcessor yielded an event: %v", ev)
		}
		if err != nil {
			t.Fatalf("AgentTransferRequestProcessor(%q) failed: %v", cur.Name(), err)
		}
	}
	return req
}

func newChatAgent(t *testing.T, name string, m model.LLM, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: name, Model: m, SubAgents: subAgents})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// An agent run without a parent map, as the runner's agent below the root of
// the tree it was built in, is offered its sub-agents, then its parent, then
// its peers.
// Fails when the parent is read only from the run's parent map (enum
// ["grand"]), when the link's inner agent is returned instead of its owner
// (the parent is not seen as an LLM agent: enum ["grand"]), or when the
// order of the three groups changes.
func TestAgentTransferRequestProcessor_TargetOrderWithLinkParent(t *testing.T) {
	m := searchGuardModel{name: "unused"}
	grand := newChatAgent(t, "grand", m)
	child := newChatAgent(t, "child", m, grand)
	peerA := newChatAgent(t, "peerA", m)
	peerB := utils.Must(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "peerB"}}))
	newChatAgent(t, "root", m, child, peerA, peerB)

	req := transferRequest(t, t.Context(), child)

	want := []string{"grand", "root", "peerA", "peerB"}
	if diff := cmp.Diff(want, transferEnum(t, req)); diff != "" {
		t.Errorf("transfer_to_agent enum mismatch (-want +got):\n%s", diff)
	}
}

// An agent listed in two trees takes its parent from the tree being run when
// that tree knows it, and from the parent linked last when the run starts at
// the agent itself.
func TestAgentTransferRequestProcessor_AgentInTwoTrees(t *testing.T) {
	build := func(t *testing.T) (shared, root1 agent.Agent) {
		m := searchGuardModel{name: "unused"}
		shared = newChatAgent(t, "shared", m)
		root1 = newChatAgent(t, "root1", m, shared, newChatAgent(t, "p1", m))
		newChatAgent(t, "root2", m, shared, newChatAgent(t, "p2", m))
		return shared, root1
	}

	// Fails when the static link is read before the run's parent map.
	t.Run("RunTreeKnowsAgent", func(t *testing.T) {
		shared, root1 := build(t)

		req := transferRequest(t, withParents(t, t.Context(), root1), shared)

		if diff := cmp.Diff([]string{"root1", "p1"}, transferEnum(t, req)); diff != "" {
			t.Errorf("transfer_to_agent enum mismatch (-want +got):\n%s", diff)
		}
	})

	// Fails when the static link is not read for an agent the run's map does
	// not know.
	t.Run("RunStartsAtAgent", func(t *testing.T) {
		shared, _ := build(t)

		req := transferRequest(t, withParents(t, t.Context(), shared), shared)

		if diff := cmp.Diff([]string{"root2", "p2"}, transferEnum(t, req)); diff != "" {
			t.Errorf("transfer_to_agent enum mismatch (-want +got):\n%s", diff)
		}
	})
}

// The resume lookup finds a transfer target through the parent link when the
// context carries no parent map. A name no agent of the tree has is an error.
// Fails when TransferTarget reads only the run's parent map (no result for
// "peer" and "root"), or returns no error for a name it does not find.
func TestTransferTarget_ParentFromLinkWhenMapHasNone(t *testing.T) {
	m := searchGuardModel{name: "unused"}
	for _, tc := range []struct {
		name    string
		target  string
		want    string // "" for no target
		wantErr bool
	}{
		{name: "peer", target: "peer", want: "peer"},
		{name: "parent", target: "root", want: "root"},
		{name: "unknown name", target: "nobody", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			child := newChatAgent(t, "child", m)
			newChatAgent(t, "root", m, child, newChatAgent(t, "peer", m))
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: child})

			a, err := llminternal.TransferTarget(ctx, tc.target)

			got := ""
			if a != nil {
				got = a.Name()
			}
			if got != tc.want || (err != nil) != tc.wantErr {
				t.Errorf("TransferTarget(%q) = %q, %v; want %q, error %t", tc.target, got, err, tc.want, tc.wantErr)
			}
		})
	}
}

// On resume, a pending transfer to an agent that is not among the caller's
// transfer targets runs the agent of that name in the caller's tree, as
// adk-python's LlmAgent.__get_agent_to_run does (agents/llm_agent.py:1017-1043).
// Fails when TransferTarget looks only at the caller's transfer targets, or
// searches from an ancestor below the tree's root (rows "two levels up").
func TestTransferTarget_FallsBackToTheAgentTree(t *testing.T) {
	m := searchGuardModel{name: "unused"}
	for _, tc := range []struct {
		name          string
		cur           string // the agent that transferred
		disallowPeers bool   // set on child
		withMap       bool   // the run's parent map, built from root, is in the context
		target        string
	}{
		{name: "agent below a sub-agent", cur: "root", withMap: true, target: "grandchild"},
		{name: "agent in a peer's subtree, from the link", cur: "child", target: "cousin"},
		{name: "agent in a peer's subtree, from the run's map", cur: "child", withMap: true, target: "cousin"},
		{name: "peer with DisallowTransferToPeers", cur: "child", disallowPeers: true, target: "peer"},
		{name: "agent two levels up, from the link", cur: "grandchild", target: "cousin"},
		{name: "agent two levels up, from the run's map", cur: "grandchild", withMap: true, target: "cousin"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			grandchild := newChatAgent(t, "grandchild", m)
			child := utils.Must(llmagent.New(llmagent.Config{Name: "child", Model: m, DisallowTransferToPeers: tc.disallowPeers, SubAgents: []agent.Agent{grandchild}}))
			peer := newChatAgent(t, "peer", m, newChatAgent(t, "cousin", m))
			root := newChatAgent(t, "root", m, child, peer)
			cur := map[string]agent.Agent{"root": root, "child": child, "grandchild": grandchild}[tc.cur]
			ctx := t.Context()
			if tc.withMap {
				ctx = withParents(t, ctx, root)
			}

			a, err := llminternal.TransferTarget(icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Agent: cur}), tc.target)
			if err != nil {
				t.Fatalf("TransferTarget(%q) from %s: %v", tc.target, tc.cur, err)
			}

			if got := a.Name(); got != tc.target {
				t.Errorf("TransferTarget(%q) from %s = %q, want %q", tc.target, tc.cur, got, tc.target)
			}
		})
	}
}

// On resume, a transfer target wins over a same-named agent elsewhere in the
// tree, which a depth-first search of the tree would meet first.
// Fails when TransferTarget searches the tree before the caller's targets.
func TestTransferTarget_TargetWinsOverSameNamedAgentInTree(t *testing.T) {
	m := searchGuardModel{name: "unused"}
	undeclared := newChatAgent(t, "shared_name", m)
	sub1 := utils.Must(llmagent.New(llmagent.Config{Name: "sub1", Model: m, DisallowTransferToParent: true, DisallowTransferToPeers: true, SubAgents: []agent.Agent{undeclared}}))
	declared := newChatAgent(t, "shared_name", m)
	caller := newChatAgent(t, "caller", m, sub1, declared)
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: caller})

	got, err := llminternal.TransferTarget(ctx, "shared_name")
	if err != nil {
		t.Fatalf("TransferTarget(shared_name): %v", err)
	}

	if got != declared {
		t.Errorf("TransferTarget(shared_name) = %p, want the declared sub-agent %p (undeclared: %p)", got, declared, undeclared)
	}
}

// On resume, a peer that is a transfer target wins over a same-named agent in
// the caller's own subtree, which a depth-first search from the tree's root
// meets first. The caller's parent comes from the parent link here, as no
// parent map is in the context.
// Fails when TransferTarget checks the targets without the caller's parent:
// the tree search then returns the agent below the caller.
func TestTransferTarget_PeerTargetWinsOverSameNamedAgentInOwnSubtree(t *testing.T) {
	m := searchGuardModel{name: "unused"}
	below := newChatAgent(t, "dup", m)
	sub1 := newChatAgent(t, "sub1", m, below)
	caller := newChatAgent(t, "caller", m, sub1)
	peer := newChatAgent(t, "dup", m)
	newChatAgent(t, "root", m, caller, peer)
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: caller})

	got, err := llminternal.TransferTarget(ctx, "dup")
	if err != nil {
		t.Fatalf("TransferTarget(dup): %v", err)
	}

	if got != peer {
		t.Errorf("TransferTarget(dup) = %p, want the peer %p (the agent below the caller: %p)", got, peer, below)
	}
}

// A transfer the model asks for runs the target when the caller takes its
// parent, and so the target, from the parent link: the flow accepts what the
// request offered.
// Fails when Flow.agentToRun reads the parent only from the run's parent map
// (the run fails with "failed to find agent: peer").
func TestFlow_TransferToPeerFromLinkParentRunsPeer(t *testing.T) {
	transfer := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{Name: "transfer_to_agent", Args: map[string]any{"agent_name": "peer"}},
	}}}
	child := newChatAgent(t, "child", &testutil.MockModel{Responses: []*genai.Content{transfer}})
	peer := newChatAgent(t, "peer", &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("from peer", genai.RoleModel)}})
	newChatAgent(t, "root", searchGuardModel{name: "unused"}, child, peer)
	sess, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:       child,
		Session:     sess.Session,
		UserContent: genai.NewContentFromText("go", genai.RoleUser),
		RunConfig:   &agent.RunConfig{},
	})

	var got []string
	for ev, err := range child.Run(ctx) {
		if err != nil {
			t.Fatalf("child.Run failed after %q: %v", got, err)
		}
		if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text != "" {
			got = append(got, ev.Author+": "+ev.Content.Parts[0].Text)
		}
	}

	if diff := cmp.Diff([]string{"peer: from peer"}, got); diff != "" {
		t.Errorf("text events mismatch (-want +got):\n%s", diff)
	}
}

// Reading an agent's parent link while another goroutine constructs a new
// parent for the same agent is safe, as agent.New allows concurrent
// construction.
// Fails, under -race, when the link is read without agentinternal.ParentOf's
// lock; fails on the enum when the link is not read at all.
func TestAgentTransferRequestProcessor_LinkReadIsRaceFree(t *testing.T) {
	m := searchGuardModel{name: "unused"}
	child := newChatAgent(t, "child", m)
	newChatAgent(t, "root0", m, child)

	var wg sync.WaitGroup
	wg.Go(func() {
		for i := 1; i <= 50; i++ {
			if _, err := llmagent.New(llmagent.Config{Name: fmt.Sprintf("root%d", i), Model: m, SubAgents: []agent.Agent{child}}); err != nil {
				t.Errorf("llmagent.New(root%d): %v", i, err)
				return
			}
		}
	})
	defer wg.Wait()

	for range 50 {
		enum := transferEnum(t, transferRequest(t, t.Context(), child))
		if len(enum) != 1 || !strings.HasPrefix(enum[0], "root") {
			t.Fatalf("transfer_to_agent enum = %q, want one parent named root<N>", enum)
		}
	}
}

// Transfer and model inheritance take an agent's parent from the same place,
// so they cannot disagree on it.
func TestParentOf_TransferAndModelInheritanceAgree(t *testing.T) {
	// Fails when the transfer path does not read the static link that model
	// resolution reads.
	t.Run("OutsideItsTree", func(t *testing.T) {
		child := newChatAgent(t, "child", nil)
		newChatAgent(t, "root", searchGuardModel{name: "root-model"}, child)
		ctx := t.Context()

		enum := transferEnum(t, transferRequest(t, ctx, child))
		got, err := llminternal.ResolveModel(ctx, child)
		if err != nil {
			t.Fatalf("ResolveModel(child): %v", err)
		}

		if diff := cmp.Diff([]string{"root"}, enum); diff != "" {
			t.Errorf("transfer_to_agent enum mismatch (-want +got):\n%s", diff)
		}
		if got.Name() != "root-model" {
			t.Errorf("ResolveModel(child) = %s, want root-model", got.Name())
		}
	})

	// Fails when either reader reads the static link (root2) before the
	// run's parent map (root1).
	t.Run("RunTreeAndLinkDisagree", func(t *testing.T) {
		child := newChatAgent(t, "child", nil)
		root1 := newChatAgent(t, "root1", searchGuardModel{name: "m1"}, child)
		newChatAgent(t, "root2", searchGuardModel{name: "m2"}, child)
		ctx := withParents(t, t.Context(), root1)

		enum := transferEnum(t, transferRequest(t, ctx, child))
		got, err := llminternal.ResolveModel(ctx, child)
		if err != nil {
			t.Fatalf("ResolveModel(child): %v", err)
		}

		if diff := cmp.Diff([]string{"root1"}, enum); diff != "" {
			t.Errorf("transfer_to_agent enum mismatch (-want +got):\n%s", diff)
		}
		if got.Name() != "m1" {
			t.Errorf("ResolveModel(child) = %s, want m1", got.Name())
		}
	})
}
