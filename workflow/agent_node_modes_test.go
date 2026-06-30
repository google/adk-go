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
	"context"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

func TestNewAgentNode_NameInheritedFromAgent(t *testing.T) {
	t.Parallel()
	const agentName = "my_inner_agent"
	a, err := llmagent.New(llmagent.Config{Name: agentName, Mode: llmagent.ModeChat})
	if err != nil {
		t.Fatal(err)
	}
	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := node.Name(), agentName; got != want {
		t.Errorf("node.Name() = %q, want %q (must inherit from wrapped agent)", got, want)
	}
}

// Task-mode agents' natural-language text must NOT be promoted to
// the node Output: their authoritative output is set via finish_task
// (in llmagent.runTask). Promoting plain text would make the chat
// coordinator close the delegation prematurely and route the next
// user reply away from the still-open task scope.
func TestAgentNode_TaskMode_DoesNotPromoteModelText(t *testing.T) {
	t.Parallel()

	const userFacingText = "Please tell me what you'd like to order."
	llm := &scriptedLLM{turns: []*model.LLMResponse{{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: userFacingText}},
		},
	}}}

	taskAgent, err := llmagent.New(llmagent.Config{
		Name:  "order_collector",
		Mode:  llmagent.ModeTask,
		Model: llm,
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	node, err := workflow.NewAgentNode(taskAgent, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}

	exCtx := newRunnableNodeContext(t, taskAgent)

	var lastEv *session.Event
	for ev, runErr := range node.Run(exCtx, nil) {
		if runErr != nil {
			t.Fatalf("node.Run yielded err: %v", runErr)
		}
		if ev == nil || ev.LLMResponse.Partial {
			continue
		}
		lastEv = ev
	}
	if lastEv == nil {
		t.Fatal("no non-partial events yielded")
	}
	if lastEv.Output != nil {
		t.Errorf("Output = %v, want nil (task-mode text must not be promoted)", lastEv.Output)
	}
	if lastEv.NodeInfo != nil && lastEv.NodeInfo.MessageAsOutput {
		t.Errorf("NodeInfo.MessageAsOutput = true, want false/unset for task-mode text")
	}
}

func TestAgentNode_SequentialAgentAsNode_NoMultipleOutputs(t *testing.T) {
	t.Parallel()

	seqNode, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "seq_node",
			SubAgents: []agent.Agent{fixedTextAgent(t, "seq_a", "SEQ_A"), fixedTextAgent(t, "seq_b", "SEQ_B")},
		},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New: %v", err)
	}

	node, err := workflow.NewAgentNode(seqNode, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}

	ctx := newRunnableNodeContext(t, seqNode)
	assertSubAgentOutputsNotSynthesized(t, node, "seq_node", ctx, "seq_a", "seq_b")
}

func TestAgentNode_ParallelAgentAsNode_NoMultipleOutputs(t *testing.T) {
	t.Parallel()

	parNode, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:      "par_node",
			SubAgents: []agent.Agent{fixedTextAgent(t, "par_a", "PAR_A"), fixedTextAgent(t, "par_b", "PAR_B")},
		},
	})
	if err != nil {
		t.Fatalf("parallelagent.New: %v", err)
	}

	node, err := workflow.NewAgentNode(parNode, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}

	ctx := newRunnableNodeContext(t, parNode)
	assertSubAgentOutputsNotSynthesized(t, node, "par_node", ctx, "par_a", "par_b")
}

// newRunnableNodeContext builds the minimal agent.Context an LlmAgent
// flow needs: in-memory Session, Agent, and a RunConfig on the
// embedded context (the flow reads it via runconfig.FromContext and
// nil-derefs otherwise).
func newRunnableNodeContext(t *testing.T, a agent.Agent) agent.Context {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName: "app",
		UserID:  "u",
	})
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	stdCtx := runconfig.ToContext(t.Context(), &runconfig.RunConfig{
		StreamingMode: runconfig.StreamingModeNone,
	})
	ic := icontext.NewInvocationContext(stdCtx, icontext.InvocationContextParams{
		Agent:        a,
		Session:      resp.Session,
		InvocationID: "inv-test",
	})
	return agent.NewContext(ic)
}

// scriptedLLM yields one LLMResponse per GenerateContent call,
// falling back to a terminal "done" event so the flow loop exits.
type scriptedLLM struct {
	turns   []*model.LLMResponse
	callIdx int
}

func (*scriptedLLM) Name() string { return "scripted-mock" }

func (s *scriptedLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	idx := s.callIdx
	s.callIdx++
	return func(yield func(*model.LLMResponse, error) bool) {
		if idx < len(s.turns) {
			yield(s.turns[idx], nil)
			return
		}
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: "done"}},
			},
		}, nil)
	}
}

var _ model.LLM = (*scriptedLLM)(nil)

func fixedTextAgent(t *testing.T, name, text string) agent.Agent {
	t.Helper()
	llm := &scriptedLLM{turns: []*model.LLMResponse{{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: text}},
		},
	}}}
	a, err := llmagent.New(llmagent.Config{Name: name, Model: llm})
	if err != nil {
		t.Fatalf("llmagent.New(%q): %v", name, err)
	}
	return a
}

// assertSubAgentOutputsNotSynthesized fails if a composite's sub-agent events
// were promoted to node outputs (each would trip the one-output-per-node rule).
func assertSubAgentOutputsNotSynthesized(t *testing.T, node workflow.Node, nodeName string, ctx agent.Context, wantAuthors ...string) {
	t.Helper()

	seen := map[string]bool{}
	for ev, err := range node.Run(ctx, nil) {
		if err != nil {
			t.Fatalf("node.Run yielded err: %v", err)
		}
		if ev == nil || ev.LLMResponse.Partial {
			continue
		}
		if ev.Author != "" && ev.Author != nodeName {
			seen[ev.Author] = true
			if ev.Output != nil {
				t.Errorf("sub-agent %q event Output = %v, want nil", ev.Author, ev.Output)
			}
			if ev.NodeInfo != nil && ev.NodeInfo.MessageAsOutput {
				t.Errorf("sub-agent %q event MessageAsOutput = true, want false", ev.Author)
			}
		}
	}

	for _, a := range wantAuthors {
		if !seen[a] {
			t.Errorf("missing final event from sub-agent %q", a)
		}
	}
}
