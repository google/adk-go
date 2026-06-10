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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/workflow"
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
