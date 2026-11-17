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

package llmagent_test

import (
	"slices"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// requestTexts returns the text of the first part of each content of req, or
// "<call NAME>" and "<response NAME>" for a function call or response part.
func requestTexts(req *model.LLMRequest) []string {
	var texts []string
	for _, c := range req.Contents {
		p := c.Parts[0]
		switch {
		case p.FunctionCall != nil:
			texts = append(texts, c.Role+" <call "+p.FunctionCall.Name+">")
		case p.FunctionResponse != nil:
			texts = append(texts, c.Role+" <response "+p.FunctionResponse.Name+">")
		default:
			texts = append(texts, c.Role+" "+p.Text)
		}
	}
	return texts
}

func TestLLMAgent_ModelInputContext_RequestsGetOwnCopy(t *testing.T) {
	type echoArgs struct {
		Message string `json:"message"`
	}
	echo, err := functiontool.New(functiontool.Config{Name: "echo", Description: "echoes the message"},
		func(_ agent.Context, args echoArgs) (echoArgs, error) { return args, nil })
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}
	mockModel := &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromFunctionCall("echo", map[string]any{"message": "payload"}, genai.RoleModel),
		genai.NewContentFromText("Answer", genai.RoleModel),
	}}
	calls := 0
	a, err := llmagent.New(llmagent.Config{
		Name:  "mic_agent",
		Model: mockModel,
		Tools: []tool.Tool{echo},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			calls++
			if calls == 1 {
				for _, c := range req.Contents {
					c.Parts[0].Text = "EDITED"
				}
			}
			return nil, nil
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r := testutil.NewTestAgentRunner(t, a)
	cfg := agent.RunConfig{ModelInputContext: &[]*genai.Content{genai.NewContentFromText("ctx-text", genai.RoleUser)}}

	if _, err := testutil.CollectEvents(r.RunContentWithConfig(t, "s", genai.NewContentFromText("Question", genai.RoleUser), cfg)); err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if len(mockModel.Requests) != 2 {
		t.Fatalf("model requests = %d, want 2", len(mockModel.Requests))
	}
	want := []string{"user ctx-text", "user Question", "model <call echo>", "user <response echo>"}
	if got := requestTexts(mockModel.Requests[1]); !slices.Equal(got, want) {
		t.Errorf("request 1 contents = %q, want %q", got, want)
	}
	if got := (*cfg.ModelInputContext)[0].Parts[0].Text; got != "ctx-text" {
		t.Errorf("caller's ModelInputContext[0] text = %q, want %q", got, "ctx-text")
	}
	resp, err := r.SessionService().Get(t.Context(), &session.GetRequest{AppName: "test_app", UserID: "test_user", SessionID: "s"})
	if err != nil {
		t.Fatalf("session Get: %v", err)
	}
	if got := resp.Session.Events().Len(); got != 4 {
		t.Fatalf("session events = %d, want 4", got)
	}
	if got := resp.Session.Events().At(0).Content.Parts[0].Text; got != "Question" {
		t.Errorf("session event 0 text = %q, want %q", got, "Question")
	}
}

func TestLLMAgent_ModelInputContext_ReachesSequentialSubAgents(t *testing.T) {
	model1 := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("one", genai.RoleModel)}}
	model2 := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("two", genai.RoleModel)}}
	agent1, err := llmagent.New(llmagent.Config{Name: "agent1", Model: model1})
	if err != nil {
		t.Fatalf("llmagent.New(agent1): %v", err)
	}
	agent2, err := llmagent.New(llmagent.Config{Name: "agent2", Model: model2})
	if err != nil {
		t.Fatalf("llmagent.New(agent2): %v", err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{
		Name: "seq", SubAgents: []agent.Agent{agent1, agent2},
	}})
	if err != nil {
		t.Fatalf("sequentialagent.New: %v", err)
	}
	r := testutil.NewTestAgentRunner(t, seq)
	cfg := agent.RunConfig{ModelInputContext: &[]*genai.Content{genai.NewContentFromText("ctx-text", genai.RoleUser)}}

	if _, err := testutil.CollectEvents(r.RunContentWithConfig(t, "s", genai.NewContentFromText("Original", genai.RoleUser), cfg)); err != nil {
		t.Fatalf("run failed: %v", err)
	}

	want := []string{"user ctx-text", "user Original"}
	for name, m := range map[string]*testutil.MockModel{"agent1": model1, "agent2": model2} {
		if len(m.Requests) != 1 {
			t.Errorf("%s model requests = %d, want 1", name, len(m.Requests))
			continue
		}
		got := requestTexts(m.Requests[0])
		if len(got) < 2 || !slices.Equal(got[:2], want) {
			t.Errorf("%s request contents = %q, want them to start with %q", name, got, want)
		}
	}
}

func TestLLMAgent_ModelInputContext_NilEntryRefused(t *testing.T) {
	mockModel := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("Answer", genai.RoleModel)}}
	a, err := llmagent.New(llmagent.Config{Name: "mic_agent", Model: mockModel})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r := testutil.NewTestAgentRunner(t, a)
	cfg := agent.RunConfig{ModelInputContext: &[]*genai.Content{nil}}

	_, err = testutil.CollectEvents(r.RunContentWithConfig(t, "s", genai.NewContentFromText("Question", genai.RoleUser), cfg))

	const wantErr = "agent.RunConfig.ModelInputContext[0] is nil"
	if err == nil || !strings.Contains(err.Error(), wantErr) {
		t.Errorf("run error = %v, want one containing %q", err, wantErr)
	}
	if got := len(mockModel.Requests); got != 0 {
		t.Errorf("model requests = %d, want 0", got)
	}
}
