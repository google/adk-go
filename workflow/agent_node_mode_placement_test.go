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
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
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

// IncludeContents is an unvalidated string, so a typo must not be read as an
// explicit request for history. Only the real IncludeContentsDefault opts out of
// the placement; anything unrecognised falls back to it, as the merge base did by
// forcing "none". Getting this wrong hands a one-shot node the whole transcript.
func TestAgentNode_UnrecognisedIncludeContents_DoesNotDefeatThePlacement(t *testing.T) {
	t.Parallel()

	for _, bad := range []llmagent.IncludeContents{"None", "defualt", "DEFAULT"} {
		t.Run(string(bad), func(t *testing.T) {
			t.Parallel()

			llm := &capturingLLM{}
			a, err := llmagent.New(llmagent.Config{
				Name:            "worker",
				Model:           llm,
				IncludeContents: bad,
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
						t.Errorf("IncludeContents=%q defeated the single_turn placement and leaked history; contents = %v",
							bad, llm.got.Contents)
					}
				}
			}
		})
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

// roundTripLLM replies from a per-agent script, identifying the caller by the
// marker its instruction plants in the system instruction, and records every
// request it served.
type roundTripLLM struct {
	script map[string][]*genai.Content
	seen   []string
	si     []string
	conts  [][]*genai.Content
}

func (*roundTripLLM) Name() string { return "scripted" }

func (m *roundTripLLM) GenerateContent(_ context.Context, r *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	var si strings.Builder
	if r.Config != nil && r.Config.SystemInstruction != nil {
		for _, p := range r.Config.SystemInstruction.Parts {
			if p != nil {
				si.WriteString(p.Text)
			}
		}
	}
	who := "worker"
	if strings.Contains(si.String(), "MARKER_HELPER") {
		who = "helper"
	}
	n := 0
	for _, s := range m.seen {
		if s == who {
			n++
		}
	}
	m.seen = append(m.seen, who)
	m.si = append(m.si, si.String())
	m.conts = append(m.conts, r.Contents)

	out := &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "done"}}}
	if replies := m.script[who]; n < len(replies) {
		out = replies[n]
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: out}, nil)
	}
}

func roundTripTransferTo(name string) *genai.Content {
	return &genai.Content{Role: "model", Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{Name: "transfer_to_agent", Args: map[string]any{"agent_name": name}},
	}}}
}

// A single_turn graph node must stay single_turn for the whole activation,
// including after an agent it transferred to transfers back. The return hop is
// a fresh activation on the callee's context rather than a return to the outer
// frame, so the node's binding has to still be findable there.
//
// Reported by karolpiotrowicz in review of #1267, with this reproduction.
func TestAgentNode_PlacementSurvivesATransferRoundTrip(t *testing.T) {
	m := &roundTripLLM{script: map[string][]*genai.Content{
		"worker": {roundTripTransferTo("helper")},
		"helper": {roundTripTransferTo("worker")},
	}}

	helper, err := llmagent.New(llmagent.Config{
		Name: "helper", Description: "helps", Model: m, Instruction: "MARKER_HELPER",
	})
	if err != nil {
		t.Fatalf("llmagent.New(helper): %v", err)
	}
	// worker declares no mode, so the graph node decides it: single_turn.
	worker, err := llmagent.New(llmagent.Config{
		Name: "worker", Description: "works", Model: m, Instruction: "MARKER_WORKER",
		SubAgents: []agent.Agent{helper},
	})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}

	node, err := workflow.NewAgentNode(worker, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node))
	if err != nil {
		t.Fatalf("workflow.New: %v", err)
	}
	parents, err := parentmap.New(worker)
	if err != nil {
		t.Fatalf("parentmap.New: %v", err)
	}

	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("session.Create: %v", err)
	}
	prior := &session.Event{
		Author:      "user",
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("EARLIER_TURN", "user")},
	}
	if err := svc.AppendEvent(t.Context(), resp.Session, prior); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}

	std := runconfig.ToContext(t.Context(), &runconfig.RunConfig{StreamingMode: runconfig.StreamingModeNone})
	std = parentmap.ToContext(std, parents)
	ic := icontext.NewInvocationContext(std, icontext.InvocationContextParams{
		Agent: worker, Session: resp.Session,
		UserContent:  genai.NewContentFromText("current question", "user"),
		InvocationID: "inv-1",
	})
	for _, err := range wf.Run(agent.NewContext(ic)) {
		if err != nil {
			t.Fatalf("wf.Run: %v", err)
		}
	}

	// Find worker's second request.
	var idx []int
	for i, who := range m.seen {
		if who == "worker" {
			idx = append(idx, i)
		}
	}
	if len(idx) < 2 {
		t.Fatalf("worker was called %d time(s), want 2; call order = %v", len(idx), m.seen)
	}
	i := idx[1]

	// Two of the three checks below are absences, so guard against a request
	// whose system instruction was never assembled at all — every sibling test
	// in this file carries the same guard.
	if !strings.Contains(m.si[i], "MARKER_WORKER") {
		t.Fatalf("worker's own instruction is missing, so the absences below prove nothing; si = %q", m.si[i])
	}

	identity := strings.Contains(m.si[i], "You are an agent. Your internal name is")
	transfer := strings.Contains(m.si[i], "You have a list of other agents to transfer to")
	history := false
	for _, c := range m.conts[i] {
		for _, p := range c.Parts {
			if p != nil && strings.Contains(p.Text, "EARLIER_TURN") {
				history = true
			}
		}
	}
	if identity || transfer || history {
		t.Errorf("after a transfer round trip the node's agent is no longer single_turn: "+
			"identityPreamble=%v transferInstructions=%v sawHistory=%v, want all false",
			identity, transfer, history)
	}
}

// AgentNode.Run must tolerate the two context wrappers whose WithAgentContext
// returns nil rather than erroring, and must still place the agent while doing
// so. Routing the placement back through ctx crashes on a path where every
// symbol is exported. The merge base drained events for such a caller and this
// must keep doing so.
//
// The agent deliberately declares NO mode. A chat declaration would make this
// test blind to half of what it is for: chat and "no binding" are the same
// answer at every reader, so the guarded-nil-check shape — which silences the
// panic by dropping the placement — would pass. Undeclared, the placement is
// the only thing that makes this agent single_turn, and its loss is visible in
// the system instruction.
//
// The two wrappers are one table rather than two tests because for everything
// this function touches they behave identically — Session, Memory and RunConfig
// nil, IsolationScope empty — so the callback row pins that they stay
// equivalent rather than reaching a path of its own.
func TestAgentNode_Run_AcceptsAWrapperContext(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name string
		wrap func(agent.InvocationContext) agent.Context
	}{
		{"tool", func(ic agent.InvocationContext) agent.Context {
			return agent.NewToolContext(ic, "fc-1", &session.EventActions{}, nil)
		}},
		{"callback", func(ic agent.InvocationContext) agent.Context {
			return agent.NewCallbackContext(ic, &session.EventActions{})
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			llm := &capturingLLM{}
			peer, err := llmagent.New(llmagent.Config{Name: "peer", Model: &capturingLLM{}, Description: "a peer"})
			if err != nil {
				t.Fatalf("llmagent.New(peer): %v", err)
			}
			a, err := llmagent.New(llmagent.Config{
				Name: "c", Description: "c", Model: llm,
				Instruction: "OWN_INSTRUCTION",
				SubAgents:   []agent.Agent{peer},
			})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
			if err != nil {
				t.Fatalf("NewAgentNode: %v", err)
			}

			svc := session.InMemoryService()
			resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
			if err != nil {
				t.Fatalf("session.Create: %v", err)
			}
			std := runconfig.ToContext(t.Context(), &runconfig.RunConfig{StreamingMode: runconfig.StreamingModeNone})
			ic := icontext.NewInvocationContext(std, icontext.InvocationContextParams{
				Agent: a, Session: resp.Session,
				UserContent:  genai.NewContentFromText("hi", "user"),
				InvocationID: "inv-" + tc.name + "-ctx",
			})

			// nodeInput must be nil. A non-nil one takes the seeded path, which
			// needs a session neither wrapper reports; that case is the test
			// below.
			got := 0
			for _, err := range node.Run(tc.wrap(ic), nil) {
				if err != nil {
					t.Fatalf("node.Run: %v", err)
				}
				got++
			}
			if got == 0 {
				t.Fatalf("node.Run over a %s context produced no events", tc.name)
			}

			// Draining is only half of it. The placement has to have survived,
			// which is what a nil guard that keeps the old ctx would quietly lose.
			si := llm.systemInstruction()
			if !strings.Contains(si, "OWN_INSTRUCTION") {
				t.Fatalf("the agent's own instruction is missing, so the absences below prove nothing; si = %q", si)
			}
			if strings.Contains(si, "You are an agent") {
				t.Error("the node's agent got the identity preamble, so the single_turn placement was dropped")
			}
			if strings.Contains(si, "transfer_to_agent") {
				t.Error("the node's agent got transfer instructions, so the single_turn placement was dropped")
			}
		})
	}
}

// The second cell of the round-15 class, found the same way and measured the
// same way.
//
// An undeclared sub-agent ADOPTED by a chat coordinator used to be stamped chat
// by installTaskTools, and chat ignores nodeInput, so a node driving it from a
// tool context with a non-nil input never seeded and completed. Nothing stamps
// it now, so the node places it single_turn, it takes the seeded path, and that
// path wraps the session — which a tool context reports as nil.
//
// The panic underneath predates this change and is out of scope; the set of
// callers reaching it did not, so the seeded path reports the missing session
// instead. Measured: the same call completes on the merge base.
//
// The unadopted agent is the control. It reaches the same branch on both sides
// and panicked on both before this guard, so it shows the guard is what changed
// the outcome rather than the adoption.
func TestAgentNode_Run_SeededOverAToolContext_ErrorsRatherThanPanics(t *testing.T) {
	t.Parallel()

	for _, adopted := range []bool{false, true} {
		t.Run(map[bool]string{false: "unadopted", true: "adoptedByAChatCoordinator"}[adopted], func(t *testing.T) {
			t.Parallel()

			sub, err := llmagent.New(llmagent.Config{Name: "worker", Description: "w", Model: &capturingLLM{}})
			if err != nil {
				t.Fatalf("llmagent.New(sub): %v", err)
			}
			if adopted {
				if _, err := llmagent.New(llmagent.Config{
					Name: "coord", Description: "c", Mode: llmagent.ModeChat,
					Model: &capturingLLM{}, SubAgents: []agent.Agent{sub},
				}); err != nil {
					t.Fatalf("llmagent.New(coord): %v", err)
				}
			}
			node, err := workflow.NewAgentNode(sub, workflow.NodeConfig{})
			if err != nil {
				t.Fatalf("NewAgentNode: %v", err)
			}

			svc := session.InMemoryService()
			resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
			if err != nil {
				t.Fatalf("session.Create: %v", err)
			}
			std := runconfig.ToContext(t.Context(), &runconfig.RunConfig{StreamingMode: runconfig.StreamingModeNone})
			ic := icontext.NewInvocationContext(std, icontext.InvocationContextParams{
				Agent: sub, Session: resp.Session,
				UserContent:  genai.NewContentFromText("hi", "user"),
				InvocationID: "inv-seeded-tool-ctx",
			})
			toolCtx := agent.NewToolContext(ic, "fc-1", &session.EventActions{}, nil)

			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("panicked instead of reporting the missing session: %v", r)
				}
			}()

			var gotErr error
			for _, err := range node.Run(toolCtx, "some input") {
				if err != nil {
					gotErr = err
				}
			}
			if gotErr == nil {
				t.Fatal("expected an error for a seeded run over a tool context; got nil")
			}
			if !strings.Contains(gotErr.Error(), "session") {
				t.Errorf("err = %q, want it to name the missing session", gotErr.Error())
			}
		})
	}
}

// AgentNode.Run is exported and now resolves a mode, which reads ctx. It must
// reject a nil context rather than dereference it, the way RunLLMAgentAsNode
// does. The merge base panicked on a nil ctx too — a few lines lower, building
// params — so this turns a crash into an error rather than changing behaviour.
func TestAgentNode_Run_NilContext_Errors(t *testing.T) {
	t.Parallel()

	a, err := llmagent.New(llmagent.Config{Name: "c", Description: "c", Model: &capturingLLM{}})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}

	var gotErr error
	for _, err := range node.Run(nil, nil) {
		if err != nil {
			gotErr = err
		}
	}
	if gotErr == nil {
		t.Fatal("expected an error for a nil context; got nil")
	}
	if !strings.Contains(gotErr.Error(), "nil context") {
		t.Errorf("err = %q, want it to mention a nil context", gotErr.Error())
	}
}

// The mode binding is keyed by agent NAME, and names are only unique across
// SubAgents(). A graph node's agent is not in SubAgents(), so a nested agent
// can share a name with one that already holds a binding, and runner.New does
// not reject the tree.
//
// Here a chat root named "coordinator" transfers to a workflow whose node wraps
// a SequentialAgent — a composite, so the node binds nothing — whose child is a
// DIFFERENT agent, also named "coordinator", declaring single_turn. That child
// must still run single_turn. Before the binding carried the identity of the
// agent it was resolved for, this child read the root's chat binding and was
// handed the identity preamble and transfer instructions the merge base
// correctly withheld.
//
// The binding a name-only key would wrongly hand this child comes from the
// WRAPPER, which re-binds the outer coordinator under that same name — not from
// the runner's root bind. Measured: with the root bind removed and identity
// dropped from the key together, this test still fails. So the root bind is
// inert to this test as well as to every reader, and an earlier version of this
// comment that claimed otherwise was wrong.
func TestAgentNode_ASameNamedNestedAgentKeepsItsOwnDeclaration(t *testing.T) {
	t.Parallel()

	innerLLM := &capturingLLM{}
	peer, err := llmagent.New(llmagent.Config{Name: "peer", Model: &capturingLLM{}, Description: "a peer"})
	if err != nil {
		t.Fatalf("llmagent.New(peer): %v", err)
	}
	// Same name as the root, declaring the mode the root's binding would override.
	inner, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Model:       innerLLM,
		Mode:        llmagent.ModeSingleTurn,
		Instruction: "INNER_INSTRUCTION",
		SubAgents:   []agent.Agent{peer},
	})
	if err != nil {
		t.Fatalf("llmagent.New(inner): %v", err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{inner}},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New: %v", err)
	}
	node, err := workflow.NewAgentNode(seq, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	wf, err := workflowagent.New(workflowagent.Config{
		Name:        "wf",
		Description: "the workflow",
		Edges:       []workflow.Edge{{From: workflow.Start, To: node}},
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name: "coordinator",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			genai.NewContentFromFunctionCall("transfer_to_agent",
				map[string]any{"agent_name": "wf"}, "model"),
			genai.NewContentFromText("done", "model"),
		}},
		SubAgents: []agent.Agent{wf},
	})
	if err != nil {
		t.Fatalf("llmagent.New(root): %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             root,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New rejected two same-named agents; the collision this pins is gone: %v", err)
	}
	msg := genai.NewContentFromText("please do the thing", genai.RoleUser)
	for _, err := range r.Run(t.Context(), "u", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	}

	if innerLLM.got == nil {
		t.Fatal("the nested agent was never reached, so this test pins nothing")
	}
	si := innerLLM.systemInstruction()
	// Both assertions below are absences, and systemInstruction returns "" when
	// the request carries none — so without this the test would still pass if
	// instruction assembly stopped running altogether.
	if !strings.Contains(si, "INNER_INSTRUCTION") {
		t.Fatalf("the agent's own instruction is missing, so the two absences below prove nothing; si = %q", si)
	}
	if strings.Contains(si, "You are an agent") {
		t.Error("the nested single_turn agent got the identity preamble; it read the root's chat binding")
	}
	if strings.Contains(si, "transfer_to_agent") {
		t.Error("the nested single_turn agent got transfer instructions; it read the root's chat binding")
	}
}

// The other direction of the same-name collision, and the one the earlier
// declaration-based compensation could not catch: the nested agent declares
// NOTHING, so there is no declaration to contradict.
//
// A graph node binds single_turn for an outer agent named "worker". That agent
// transfers into a workflow whose node wraps a composite — so nothing rebinds —
// and the composite's child is a DIFFERENT agent, also named "worker", also
// undeclared. Being a composite's child with no placement of its own, it is a
// chat agent and must be given the identity preamble and transfer tooling. The
// merge base gave it both, because it read that agent's own State.
func TestAgentNode_ASingleTurnPlacementDoesNotReachASameNamedUndeclaredDescendant(t *testing.T) {
	t.Parallel()

	innerLLM := &capturingLLM{}
	innerPeer, err := llmagent.New(llmagent.Config{Name: "innerpeer", Model: &capturingLLM{}, Description: "inner peer"})
	if err != nil {
		t.Fatalf("llmagent.New(innerpeer): %v", err)
	}
	inner, err := llmagent.New(llmagent.Config{
		Name:        "worker", // same name as the placed agent below, deliberately
		Model:       innerLLM,
		Instruction: "INNER_INSTRUCTION",
		SubAgents:   []agent.Agent{innerPeer},
	})
	if err != nil {
		t.Fatalf("llmagent.New(inner): %v", err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{inner}},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New: %v", err)
	}
	innerNode, err := workflow.NewAgentNode(seq, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode(inner): %v", err)
	}
	innerWf, err := workflowagent.New(workflowagent.Config{
		Name: "innerwf", Description: "inner workflow",
		Edges: []workflow.Edge{{From: workflow.Start, To: innerNode}},
	})
	if err != nil {
		t.Fatalf("workflowagent.New(inner): %v", err)
	}
	// Placed at a graph node while declaring nothing, so it binds single_turn.
	outer, err := llmagent.New(llmagent.Config{
		Name: "worker",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			genai.NewContentFromFunctionCall("transfer_to_agent",
				map[string]any{"agent_name": "innerwf"}, "model"),
			genai.NewContentFromText("outer done", "model"),
		}},
		SubAgents: []agent.Agent{innerWf},
	})
	if err != nil {
		t.Fatalf("llmagent.New(outer): %v", err)
	}
	outerNode, err := workflow.NewAgentNode(outer, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode(outer): %v", err)
	}
	wf, err := workflowagent.New(workflowagent.Config{
		Name: "wf", Description: "outer workflow",
		Edges: []workflow.Edge{{From: workflow.Start, To: outerNode}},
	})
	if err != nil {
		t.Fatalf("workflowagent.New(outer): %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name: "coordinator",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			genai.NewContentFromFunctionCall("transfer_to_agent",
				map[string]any{"agent_name": "wf"}, "model"),
			genai.NewContentFromText("done", "model"),
		}},
		SubAgents: []agent.Agent{wf},
	})
	if err != nil {
		t.Fatalf("llmagent.New(root): %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             root,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New rejected two same-named agents; the collision this pins is gone: %v", err)
	}
	msg := genai.NewContentFromText("go", genai.RoleUser)
	for _, err := range r.Run(t.Context(), "u", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	}

	if innerLLM.got == nil {
		t.Fatal("the nested agent was never reached, so this test pins nothing")
	}
	si := innerLLM.systemInstruction()
	// The two assertions below are presences, but an empty instruction would
	// fail them for the wrong reason, so pin the agent's own text first.
	if !strings.Contains(si, "INNER_INSTRUCTION") {
		t.Fatalf("the agent's own instruction is missing, so the checks below prove nothing; si = %q", si)
	}
	if !strings.Contains(si, "You are an agent") {
		t.Error("the nested undeclared agent lost its identity preamble; it read the outer agent's single_turn binding")
	}
	if !strings.Contains(si, "transfer_to_agent") {
		t.Error("the nested undeclared agent lost its transfer instructions; it read the outer agent's single_turn binding")
	}
}

// The enumerated behaviour change that had no test: a composite's UNDECLARED
// child, re-entered as a transfer target, runs chat where the merge base ran
// single_turn.
//
// installTaskTools never sees this agent — its parent is a SequentialAgent, not
// an LlmAgent — and no node wraps it, so on the base nothing had stamped it and
// the wrapper stamped single_turn at the re-entry. Head has no binding for it
// and resolves the chat fallback instead.
//
// Measured both ways on this tree. Base: no identity preamble, no transfer
// block, no history. Head: all three. Those three are the user-visible cost of
// the flip and are what the PR description enumerates; asserting them here is
// what makes the enumeration checkable.
func TestAgentNode_ACompositeChildReenteredByTransferRunsChat(t *testing.T) {
	m := &roundTripLLM{script: map[string][]*genai.Content{
		"worker": {roundTripTransferTo("helper")},
		"helper": {roundTripTransferTo("worker")},
	}}
	helper, err := llmagent.New(llmagent.Config{
		Name: "helper", Description: "helps", Model: m, Instruction: "MARKER_HELPER",
	})
	if err != nil {
		t.Fatalf("llmagent.New(helper): %v", err)
	}
	worker, err := llmagent.New(llmagent.Config{
		Name: "worker", Description: "works", Model: m, Instruction: "MARKER_WORKER",
		SubAgents: []agent.Agent{helper},
	})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "seq", Description: "seq", SubAgents: []agent.Agent{worker}},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName: "app", Agent: seq,
		SessionService: session.InMemoryService(), AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	for _, err := range r.Run(t.Context(), "u", "s1", genai.NewContentFromText("EARLIER_TURN", "user"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run: %v", err)
		}
	}

	var idx []int
	for i, who := range m.seen {
		if who == "worker" {
			idx = append(idx, i)
		}
	}
	if len(idx) < 2 {
		t.Fatalf("worker was called %d time(s), want 2; call order = %v", len(idx), m.seen)
	}
	si := m.si[idx[1]]
	if !strings.Contains(si, "MARKER_WORKER") {
		t.Fatalf("worker's own instruction is missing, so the presences below prove nothing; si = %q", si)
	}
	if !strings.Contains(si, "You are an agent") {
		t.Error("no identity preamble on re-entry; an unbound undeclared agent must run chat here")
	}
	if !strings.Contains(si, "transfer_to_agent") {
		t.Error("no transfer instructions on re-entry; an unbound undeclared agent must run chat here")
	}
	history := false
	for _, c := range m.conts[idx[1]] {
		for _, p := range c.Parts {
			if p != nil && strings.Contains(p.Text, "EARLIER_TURN") {
				history = true
			}
		}
	}
	if !history {
		t.Error("no conversation history on re-entry; an unbound undeclared agent must run chat here")
	}
}
