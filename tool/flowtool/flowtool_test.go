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

package flowtool_test

import (
	"context"
	"encoding/json"
	"iter"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/flowtool"
)

func TestSpec_UnmarshalJSON_Agent(t *testing.T) {
	var s flowtool.Spec
	if err := json.Unmarshal([]byte(`{"type":"agent","agent":"foo","input":"hi"}`), &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if s.Type != flowtool.KindAgent || s.Agent != "foo" || s.Input != "hi" {
		t.Errorf("unexpected: %+v", s)
	}
}

func TestSpec_UnmarshalJSON_Seq(t *testing.T) {
	var s flowtool.Spec
	body := `{"type":"seq","nodes":[
		{"type":"agent","agent":"a"},
		{"type":"agent","agent":"b"}]}`
	if err := json.Unmarshal([]byte(body), &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if s.Type != flowtool.KindSeq || len(s.Nodes) != 2 {
		t.Errorf("unexpected: %+v", s)
	}
}

func TestSpec_UnmarshalJSON_Errors(t *testing.T) {
	cases := map[string]string{
		"empty seq":         `{"type":"seq","nodes":[]}`,
		"agent without name": `{"type":"agent"}`,
		"unknown type":      `{"type":"loop"}`,
		"agent with nodes":  `{"type":"agent","agent":"x","nodes":[{"type":"agent","agent":"y"}]}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			var s flowtool.Spec
			if err := json.Unmarshal([]byte(body), &s); err == nil {
				t.Errorf("expected error for %q", body)
			}
		})
	}
}

func TestAssignPaths(t *testing.T) {
	s := flowtool.Spec{
		Type: flowtool.KindSeq,
		Nodes: []flowtool.Spec{
			{Type: flowtool.KindAgent, Agent: "researcher"},
			{Type: flowtool.KindParallel, Nodes: []flowtool.Spec{
				{Type: flowtool.KindAgent, Agent: "drafter"},
				{Type: flowtool.KindAgent, Agent: "fact_checker"},
			}},
			{Type: flowtool.KindAgent, Agent: "editor"},
		},
	}
	flowtool.AssignPaths(&s, "")

	want := map[int]string{0: "seq[0].researcher", 2: "seq[2].editor"}
	for i, expectedPath := range want {
		if s.Nodes[i].Path != expectedPath {
			t.Errorf("node %d path = %q, want %q", i, s.Nodes[i].Path, expectedPath)
		}
	}
	parallel := s.Nodes[1]
	if parallel.Path != "seq[1].parallel" {
		t.Errorf("parallel path = %q, want %q", parallel.Path, "seq[1].parallel")
	}
	if parallel.Nodes[0].Path != "seq[1].parallel[0].drafter" {
		t.Errorf("drafter path = %q", parallel.Nodes[0].Path)
	}
	if parallel.Nodes[1].Path != "seq[1].parallel[1].fact_checker" {
		t.Errorf("fact_checker path = %q", parallel.Nodes[1].Path)
	}
}

func TestDepth_Count_Width(t *testing.T) {
	leaf := flowtool.Spec{Type: flowtool.KindAgent, Agent: "x"}
	parallel := flowtool.Spec{Type: flowtool.KindParallel, Nodes: []flowtool.Spec{leaf, leaf, leaf}}
	seq := flowtool.Spec{Type: flowtool.KindSeq, Nodes: []flowtool.Spec{leaf, parallel, leaf}}

	if got, want := flowtool.Depth(&seq), 3; got != want {
		t.Errorf("Depth = %d, want %d", got, want)
	}
	if got, want := flowtool.CountNodes(&seq), 1+1+1+1+3; got != want {
		t.Errorf("CountNodes = %d, want %d", got, want)
	}
	if got, want := flowtool.MaxParallelWidth(&seq), 3; got != want {
		t.Errorf("MaxParallelWidth = %d, want %d", got, want)
	}
}

func TestRun_UnknownAgent(t *testing.T) {
	a := mockAgent(t, "knownAgent", "hello")
	tl := flowtool.New(map[string]agent.Agent{"knownAgent": a})
	tc := newToolContext(t, a)
	ft := tl.(toolinternal.FunctionTool)

	args := map[string]any{
		"spec": map[string]any{"type": "agent", "agent": "missingAgent"},
	}
	res, err := ft.Run(tc, args)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	errMsg, ok := res["error"].(string)
	if !ok || !strings.Contains(errMsg, "missingAgent") {
		t.Errorf("expected error mentioning missingAgent, got %v", res)
	}
}

func TestRun_MaxDepth(t *testing.T) {
	a := mockAgent(t, "a", "out")
	tl := flowtool.New(map[string]agent.Agent{"a": a}, flowtool.WithMaxDepth(2))
	ft := tl.(toolinternal.FunctionTool)
	tc := newToolContext(t, a)

	args := map[string]any{
		"spec": map[string]any{
			"type": "seq",
			"nodes": []any{
				map[string]any{"type": "seq", "nodes": []any{
					map[string]any{"type": "agent", "agent": "a"},
				}},
			},
		},
	}
	res, err := ft.Run(tc, args)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res["error"].(string), "max_depth") {
		t.Errorf("expected max_depth error, got %v", res["error"])
	}
}

func TestRun_MaxParallelWidth(t *testing.T) {
	a := mockAgent(t, "a", "x")
	tl := flowtool.New(map[string]agent.Agent{"a": a}, flowtool.WithMaxParallelWidth(2))
	ft := tl.(toolinternal.FunctionTool)
	tc := newToolContext(t, a)

	args := map[string]any{
		"spec": map[string]any{
			"type": "parallel",
			"nodes": []any{
				map[string]any{"type": "agent", "agent": "a"},
				map[string]any{"type": "agent", "agent": "a"},
				map[string]any{"type": "agent", "agent": "a"},
			},
		},
	}
	res, err := ft.Run(tc, args)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res["error"].(string), "max_parallel_width") {
		t.Errorf("expected max_parallel_width error, got %v", res["error"])
	}
}

func TestRun_SeqAgent_PassesOutput(t *testing.T) {
	first := mockAgent(t, "first", "FIRST_OUT")
	second := mockAgent(t, "second", "SECOND_OUT")
	tl := flowtool.New(map[string]agent.Agent{
		"first":  first,
		"second": second,
	})
	ft := tl.(toolinternal.FunctionTool)
	tc := newToolContext(t, first)

	args := map[string]any{
		"initial_input": "go",
		"spec": map[string]any{
			"type": "seq",
			"nodes": []any{
				map[string]any{"type": "agent", "agent": "first"},
				map[string]any{"type": "agent", "agent": "second"},
			},
		},
	}
	res, err := ft.Run(tc, args)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := res["final_output"]; got != "SECOND_OUT" {
		t.Errorf("final_output = %v, want SECOND_OUT", got)
	}
	outputs := res["outputs"].(map[string]any)
	if len(outputs) != 2 {
		t.Errorf("expected 2 outputs, got %d: %v", len(outputs), outputs)
	}
	if _, ok := outputs["seq[0].first"]; !ok {
		t.Errorf("missing seq[0].first in outputs: %v", outputs)
	}
}

// recordingModel captures every LLMRequest it sees (across stream + non-stream
// paths) so tests can assert on the user input the agent received.
type recordingModel struct {
	Response *genai.Content
	Calls    []string
}

func (m *recordingModel) Name() string { return "recording" }

func (m *recordingModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	var b strings.Builder
	for _, c := range req.Contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			if p != nil && p.Text != "" {
				b.WriteString(p.Text)
				b.WriteString("\n")
			}
		}
	}
	m.Calls = append(m.Calls, b.String())
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: m.Response}, nil)
	}
}

func TestRun_TemplateRendering(t *testing.T) {
	first := mockAgent(t, "first", "ALPHA")
	secondLLM := &recordingModel{
		Response: genai.NewContentFromText("DONE", genai.RoleModel),
	}
	second, err := llmagent.New(llmagent.Config{
		Name: "second", Description: "second", Model: secondLLM, Instruction: "Respond.",
	})
	if err != nil {
		t.Fatalf("llmagent.New(second): %v", err)
	}

	tl := flowtool.New(map[string]agent.Agent{
		"first":  first,
		"second": second,
	})
	ft := tl.(toolinternal.FunctionTool)
	tc := newToolContext(t, first)

	args := map[string]any{
		"spec": map[string]any{
			"type": "seq",
			"nodes": []any{
				map[string]any{"type": "agent", "agent": "first"},
				map[string]any{
					"type":  "agent",
					"agent": "second",
					"input": "got: {{nodes.seq[0].first.output}}",
				},
			},
		},
	}
	if _, err := ft.Run(tc, args); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(secondLLM.Calls) == 0 {
		t.Fatalf("second agent never called")
	}
	if !strings.Contains(secondLLM.Calls[0], "got: ALPHA") {
		t.Errorf("second agent input did not include rendered template; got %q", secondLLM.Calls[0])
	}
}

func TestRun_RecursionGuard(t *testing.T) {
	a := mockAgent(t, "a", "x")
	tl := flowtool.New(map[string]agent.Agent{"a": a}, flowtool.WithMaxRecursion(1))
	ft := tl.(toolinternal.FunctionTool)
	tc := newToolContextAtDepth(t, a, 1)

	args := map[string]any{
		"spec": map[string]any{"type": "agent", "agent": "a"},
	}
	res, err := ft.Run(tc, args)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	errMsg, _ := res["error"].(string)
	if !strings.Contains(errMsg, "recursion") {
		t.Errorf("expected recursion error, got %v", res)
	}
}

// helpers ---------------------------------------------------------------------

func mockAgent(t *testing.T, name, response string) agent.Agent {
	t.Helper()
	llm := &testutil.MockModel{
		Responses: []*genai.Content{
			genai.NewContentFromText(response, genai.RoleModel),
		},
	}
	a, err := llmagent.New(llmagent.Config{
		Name:        name,
		Description: name,
		Model:       llm,
		Instruction: "Respond.",
	})
	if err != nil {
		t.Fatalf("llmagent.New(%q): %v", name, err)
	}
	return a
}

func newToolContext(t *testing.T, a agent.Agent) tool.Context {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName: "testApp", UserID: "testUser", SessionID: "testSession",
	})
	if err != nil {
		t.Fatalf("session create: %v", err)
	}
	ic := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
		Session: resp.Session,
		Agent:   a,
	})
	return toolinternal.NewToolContext(ic, "", &session.EventActions{}, nil)
}

func newToolContextAtDepth(t *testing.T, a agent.Agent, depth int) tool.Context {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName: "testApp", UserID: "testUser", SessionID: "testSession",
	})
	if err != nil {
		t.Fatalf("session create: %v", err)
	}
	ctxWithDepth := flowtool.WithRecursionDepth(context.Background(), depth)
	ic := icontext.NewInvocationContext(ctxWithDepth, icontext.InvocationContextParams{
		Session: resp.Session,
		Agent:   a,
	})
	return toolinternal.NewToolContext(ic, "", &session.EventActions{}, nil)
}
