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
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// capturingLLM records the first request it is asked to serve.
type capturingLLM struct{ got *model.LLMRequest }

func (*capturingLLM) Name() string { return "capturing" }

func (c *capturingLLM) GenerateContent(_ context.Context, r *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	if c.got == nil {
		c.got = r
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: "answer"}},
		}}, nil)
	}
}

func (c *capturingLLM) systemInstruction() string {
	if c.got == nil || c.got.Config == nil || c.got.Config.SystemInstruction == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range c.got.Config.SystemInstruction.Parts {
		if p != nil {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

func newModeTestCtx(t *testing.T, a agent.Agent, prior ...*session.Event) agent.Context {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	for _, ev := range prior {
		if err := svc.AppendEvent(t.Context(), resp.Session, ev); err != nil {
			t.Fatalf("AppendEvent: %v", err)
		}
	}
	stdCtx := runconfig.ToContext(t.Context(), &runconfig.RunConfig{
		StreamingMode: runconfig.StreamingModeNone,
	})
	ic := icontext.NewInvocationContext(stdCtx, icontext.InvocationContextParams{
		Agent:        a,
		Session:      resp.Session,
		UserContent:  genai.NewContentFromText("current question", "user"),
		InvocationID: "inv-mode-test",
	})
	return agent.NewContext(ic)
}

// An LlmAgent that declares no mode runs single_turn at a graph node, and every
// request processor must agree on that: no identity preamble, no transfer
// tooling, no conversation history. Without the node's mode binding the agent
// would look undeclared to those processors and get all three.
func TestAgentNode_UnsetMode_RunsAsSingleTurnEverywhere(t *testing.T) {
	t.Parallel()

	llm := &capturingLLM{}
	peer, err := llmagent.New(llmagent.Config{Name: "peer", Model: &capturingLLM{}})
	if err != nil {
		t.Fatalf("llmagent.New(peer): %v", err)
	}
	// A sub-agent makes transfer wiring reachable, so the transfer
	// suppression is actually exercised rather than vacuously absent.
	a, err := llmagent.New(llmagent.Config{
		Name:        "worker",
		Description: "does the work",
		Model:       llm,
		Instruction: "WORKER_INSTRUCTION",
		SubAgents:   []agent.Agent{peer},
	})
	if err != nil {
		t.Fatalf("llmagent.New(with sub): %v", err)
	}

	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node))
	if err != nil {
		t.Fatalf("workflow.New: %v", err)
	}

	prior := &session.Event{
		Author:      "user",
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("EARLIER_TURN", "user")},
	}
	ic := newModeTestCtx(t, a, prior)
	for _, err := range wf.Run(ic) {
		if err != nil {
			t.Fatalf("workflow.Run: %v", err)
		}
	}

	if llm.got == nil {
		t.Fatal("model was never called")
	}
	si := llm.systemInstruction()
	if strings.Contains(si, "You are an agent. Your internal name is") {
		t.Errorf("single_turn node got the identity preamble; system instruction:\n%s", si)
	}
	if strings.Contains(si, "transfer_to_agent") {
		t.Errorf("single_turn node got transfer instructions; system instruction:\n%s", si)
	}
	// The transfer_to_agent DECLARATION still ships here, as it does before
	// this change: only the instruction was ever gated on the mode. Asserting
	// its absence would pin behavior this change does not claim to alter.
	for _, c := range llm.got.Contents {
		for _, p := range c.Parts {
			if p != nil && strings.Contains(p.Text, "EARLIER_TURN") {
				t.Errorf("single_turn node saw conversation history; contents = %v", llm.got.Contents)
			}
		}
	}
}

// A declared mode beats the node's default: a chat coordinator placed at a
// graph node stays a chat coordinator, keeping its history and its identity
// rather than being demoted to the single_turn a bare node implies.
func TestAgentNode_DeclaredChat_BeatsNodeDefault(t *testing.T) {
	t.Parallel()

	coordLLM := &capturingLLM{}
	coord, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Model:       coordLLM,
		Mode:        llmagent.ModeChat,
		Instruction: "COORD_INSTRUCTION",
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	node, err := workflow.NewAgentNode(coord, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node))
	if err != nil {
		t.Fatalf("workflow.New: %v", err)
	}

	// A chat agent is not seeded with a synthetic turn the way a single_turn
	// one is, so the history has to supply the current turn itself. It also has
	// to end on a user turn: the backward scan that isolates the current turn
	// skips this agent's own events, so a history ending on one pivots at index
	// 0 and returns everything whether or not history is being hidden.
	prior := []*session.Event{
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("EARLIER_TURN", "user")}},
		{Author: "coordinator", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("earlier answer", "model")}},
		{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("current question", "user")}},
	}
	ic := newModeTestCtx(t, coord, prior...)
	for _, err := range wf.Run(ic) {
		if err != nil {
			t.Fatalf("workflow.Run: %v", err)
		}
	}

	if coordLLM.got == nil {
		t.Fatal("model was never called")
	}
	// The declaration wins over the node's single_turn default, so the
	// coordinator keeps its history and its identity.
	var sawHistory bool
	for _, c := range coordLLM.got.Contents {
		for _, p := range c.Parts {
			if p != nil && strings.Contains(p.Text, "EARLIER_TURN") {
				sawHistory = true
			}
		}
	}
	if !sawHistory {
		t.Errorf("chat-declared agent at a node lost its history; contents = %v", coordLLM.got.Contents)
	}
	if si := coordLLM.systemInstruction(); !strings.Contains(si, "You are an agent. Your internal name is") {
		t.Errorf("chat-declared agent at a node lost its identity preamble; system instruction:\n%s", si)
	}
}

// A declared single_turn agent is seeded with one synthetic turn at a graph
// node, so it must not also see the conversation history.
func TestAgentNode_DeclaredSingleTurn_DropsHistory(t *testing.T) {
	t.Parallel()

	llm := &capturingLLM{}
	a, err := llmagent.New(llmagent.Config{
		Name:  "worker",
		Model: llm,
		Mode:  llmagent.ModeSingleTurn,
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node))
	if err != nil {
		t.Fatalf("workflow.New: %v", err)
	}

	prior := &session.Event{
		Author:      "user",
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("EARLIER_TURN", "user")},
	}
	for _, err := range wf.Run(newModeTestCtx(t, a, prior)) {
		if err != nil {
			t.Fatalf("workflow.Run: %v", err)
		}
	}
	if llm.got == nil {
		t.Fatal("model was never called")
	}
	for _, c := range llm.got.Contents {
		for _, p := range c.Parts {
			if p != nil && strings.Contains(p.Text, "EARLIER_TURN") {
				t.Errorf("single_turn node saw history; contents = %v", llm.got.Contents)
			}
		}
	}
}

// Asking for history explicitly beats the node's single_turn placement, the
// way adk-python only forces include_contents="none" when the caller left the
// field unset.
func TestAgentNode_ExplicitIncludeContents_BeatsPlacement(t *testing.T) {
	t.Parallel()

	llm := &capturingLLM{}
	a, err := llmagent.New(llmagent.Config{
		Name:            "worker",
		Model:           llm,
		IncludeContents: llmagent.IncludeContentsDefault,
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node))
	if err != nil {
		t.Fatalf("workflow.New: %v", err)
	}

	prior := &session.Event{
		Author:      "user",
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("EARLIER_TURN", "user")},
	}
	for _, err := range wf.Run(newModeTestCtx(t, a, prior)) {
		if err != nil {
			t.Fatalf("workflow.Run: %v", err)
		}
	}
	if llm.got == nil {
		t.Fatal("model was never called")
	}
	var sawHistory bool
	for _, c := range llm.got.Contents {
		for _, p := range c.Parts {
			if p != nil && strings.Contains(p.Text, "EARLIER_TURN") {
				sawHistory = true
			}
		}
	}
	if !sawHistory {
		t.Errorf("explicit IncludeContentsDefault was overridden by the placement; contents = %v", llm.got.Contents)
	}
}

// The same override, for an agent that DECLARES single_turn rather than
// leaving the mode to its placement. Before this change the node forced
// IncludeContents="none" onto it, so the request went out without the
// conversation whatever the caller had asked for.
func TestAgentNode_DeclaredSingleTurnWithExplicitIncludeContents_KeepsHistory(t *testing.T) {
	t.Parallel()

	llm := &capturingLLM{}
	a, err := llmagent.New(llmagent.Config{
		Name:            "worker",
		Model:           llm,
		Mode:            llmagent.ModeSingleTurn,
		IncludeContents: llmagent.IncludeContentsDefault,
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node))
	if err != nil {
		t.Fatalf("workflow.New: %v", err)
	}

	prior := &session.Event{
		Author:      "user",
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("EARLIER_TURN", "user")},
	}
	for _, err := range wf.Run(newModeTestCtx(t, a, prior)) {
		if err != nil {
			t.Fatalf("workflow.Run: %v", err)
		}
	}
	if llm.got == nil {
		t.Fatal("model was never called")
	}
	var sawHistory bool
	for _, c := range llm.got.Contents {
		for _, p := range c.Parts {
			if p != nil && strings.Contains(p.Text, "EARLIER_TURN") {
				sawHistory = true
			}
		}
	}
	if !sawHistory {
		t.Errorf("declared single_turn with explicit IncludeContentsDefault lost its history; contents = %v", llm.got.Contents)
	}
}

// Config.Name is documented as required but nothing rejects an empty one, and
// the binding is keyed by name. A nameless agent must still be placed by its
// node rather than quietly falling back to chat and carrying the whole
// transcript into a one-shot node.
func TestAgentNode_UnnamedAgent_IsStillPlacedSingleTurn(t *testing.T) {
	t.Parallel()

	llm := &capturingLLM{}
	a, err := llmagent.New(llmagent.Config{Model: llm})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	if a.Name() != "" {
		t.Fatalf("precondition: agent name = %q, want empty", a.Name())
	}
	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node))
	if err != nil {
		t.Fatalf("workflow.New: %v", err)
	}

	prior := &session.Event{
		Author:      "user",
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("EARLIER_TURN", "user")},
	}
	for _, err := range wf.Run(newModeTestCtx(t, a, prior)) {
		if err != nil {
			t.Fatalf("workflow.Run: %v", err)
		}
	}
	if llm.got == nil {
		t.Fatal("model was never called")
	}
	for _, c := range llm.got.Contents {
		for _, p := range c.Parts {
			if p != nil && strings.Contains(p.Text, "EARLIER_TURN") {
				t.Errorf("nameless agent at a single_turn node saw conversation history; contents = %v", llm.got.Contents)
			}
		}
	}
}
