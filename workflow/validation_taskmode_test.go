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

package workflow_test

import (
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/workflow"
)

func TestValidateNoTaskModeGraphNodes(t *testing.T) {
	t.Parallel()

	newAgent := func(t *testing.T, name string, mode llmagent.Mode) agent.Agent {
		t.Helper()
		a, err := llmagent.New(llmagent.Config{Name: name, Mode: mode})
		if err != nil {
			t.Fatalf("llmagent.New(%q, %q): %v", name, mode, err)
		}
		return a
	}

	newNode := func(t *testing.T, a agent.Agent) workflow.Node {
		t.Helper()
		n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(%q): %v", a.Name(), err)
		}
		return n
	}

	t.Run("rejects task-mode agent as the only graph node", func(t *testing.T) {
		t.Parallel()
		taskAgent := newAgent(t, "doer", llmagent.ModeTask)
		if _, err := workflow.New("wf-task-root", []workflow.Edge{
			{From: workflow.Start, To: newNode(t, taskAgent)},
		}); err == nil {
			t.Fatal("expected error rejecting task-mode static node, got nil")
		}
	})

	t.Run("rejects task-mode agent deeper in the graph", func(t *testing.T) {
		t.Parallel()
		chatAgent := newAgent(t, "chatter", llmagent.ModeChat)
		taskAgent := newAgent(t, "doer", llmagent.ModeTask)
		chatNode := newNode(t, chatAgent)
		taskNode := newNode(t, taskAgent)
		if _, err := workflow.New("wf-task-downstream", []workflow.Edge{
			{From: workflow.Start, To: chatNode},
			{From: chatNode, To: taskNode},
		}); err == nil {
			t.Fatal("expected error rejecting task-mode static node, got nil")
		}
	})

	t.Run("accepts chat-mode agent as static node", func(t *testing.T) {
		t.Parallel()
		chatAgent := newAgent(t, "chatter", llmagent.ModeChat)
		if _, err := workflow.New("wf-chat", []workflow.Edge{
			{From: workflow.Start, To: newNode(t, chatAgent)},
		}); err != nil {
			t.Errorf("chat-mode agent should be accepted as static node; got %v", err)
		}
	})

	t.Run("accepts single_turn-mode agent as static node", func(t *testing.T) {
		t.Parallel()
		stAgent := newAgent(t, "responder", llmagent.ModeSingleTurn)
		if _, err := workflow.New("wf-single-turn", []workflow.Edge{
			{From: workflow.Start, To: newNode(t, stAgent)},
		}); err != nil {
			t.Errorf("single_turn-mode agent should be accepted as static node; got %v", err)
		}
	})

	t.Run("accepts mode-unset (chat) agent as static node", func(t *testing.T) {
		t.Parallel()
		anyAgent := newAgent(t, "default", llmagent.ModeUnset)
		if _, err := workflow.New("wf-default", []workflow.Edge{
			{From: workflow.Start, To: newNode(t, anyAgent)},
		}); err != nil {
			t.Errorf("mode-unset (chat) agent should be accepted as static node; got %v", err)
		}
	})
}

// A chat-mode agent builds its prompt from conversation history and ignores
// node input, so adk-python only allows it directly after Start. These tests
// pin that same wiring rule for Go.
func TestValidateChatModeWiring(t *testing.T) {
	t.Parallel()

	newNode := func(t *testing.T, name string, mode llmagent.Mode) workflow.Node {
		t.Helper()
		a, err := llmagent.New(llmagent.Config{Name: name, Mode: mode})
		if err != nil {
			t.Fatalf("llmagent.New(%q, %q): %v", name, mode, err)
		}
		n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("workflow.NewAgentNode(%q): %v", name, err)
		}
		return n
	}

	t.Run("rejects chat-mode agent following a non-Start node", func(t *testing.T) {
		t.Parallel()
		responder := newNode(t, "responder", llmagent.ModeSingleTurn)
		coordinator := newNode(t, "coordinator", llmagent.ModeChat)
		_, err := workflow.New("wf-chat-midgraph", []workflow.Edge{
			{From: workflow.Start, To: responder},
			{From: responder, To: coordinator},
		})
		if err == nil {
			t.Fatal("expected error rejecting chat-mode node with a non-Start predecessor, got nil")
		}
	})

	t.Run("rejects chat-mode agent with a Start edge plus a non-Start edge", func(t *testing.T) {
		t.Parallel()
		responder := newNode(t, "responder", llmagent.ModeSingleTurn)
		coordinator := newNode(t, "coordinator", llmagent.ModeChat)
		_, err := workflow.New("wf-chat-mixed-edges", []workflow.Edge{
			{From: workflow.Start, To: coordinator},
			{From: workflow.Start, To: responder},
			{From: responder, To: coordinator},
		})
		if err == nil {
			t.Fatal("expected error: a non-Start edge into a chat node is rejected even when a Start edge also exists, got nil")
		}
	})

	t.Run("accepts chat-mode agent directly after Start", func(t *testing.T) {
		t.Parallel()
		coordinator := newNode(t, "coordinator", llmagent.ModeChat)
		if _, err := workflow.New("wf-chat-root", []workflow.Edge{
			{From: workflow.Start, To: coordinator},
		}); err != nil {
			t.Errorf("chat-mode agent after Start should be accepted; got %v", err)
		}
	})

	t.Run("allows single_turn agent following a non-Start node", func(t *testing.T) {
		t.Parallel()
		first := newNode(t, "first", llmagent.ModeSingleTurn)
		second := newNode(t, "second", llmagent.ModeSingleTurn)
		if _, err := workflow.New("wf-single-turn-chain", []workflow.Edge{
			{From: workflow.Start, To: first},
			{From: first, To: second},
		}); err != nil {
			t.Errorf("single_turn agent mid-graph should be accepted; got %v", err)
		}
	})
}
