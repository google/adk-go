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

package runner_test

import (
	"context"
	"iter"
	"slices"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// approvalModel answers its first request with "hello", its second with a
// call of the tool "approve", and every later one with "done". It records
// every request.
type approvalModel struct {
	requests []*model.LLMRequest
}

func (*approvalModel) Name() string { return "approval-model" }

func (m *approvalModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.requests = append(m.requests, req)
	var content *genai.Content
	switch len(m.requests) {
	case 1:
		content = genai.NewContentFromText("hello", genai.RoleModel)
	case 2:
		content = genai.NewContentFromFunctionCall("approve", map[string]any{}, genai.RoleModel)
	default:
		content = genai.NewContentFromText("done", genai.RoleModel)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil)
	}
}

// idPairingModel is an approvalModel whose provider pairs function calls with
// their responses by ID, so ADK keeps its "adk-" IDs in the requests.
type idPairingModel struct{ *approvalModel }

func (idPairingModel) Capabilities() model.Capabilities {
	return model.Capabilities{ClientFunctionCallIDs: true}
}

// contentLabels returns the role and text of the first part of each content,
// or "<call NAME>" and "<response NAME>" for a function call or response part.
func contentLabels(contents []*genai.Content) []string {
	var labels []string
	for _, c := range contents {
		p := c.Parts[0]
		switch {
		case p.FunctionCall != nil:
			labels = append(labels, c.Role+" <call "+p.FunctionCall.Name+">")
		case p.FunctionResponse != nil:
			labels = append(labels, c.Role+" <response "+p.FunctionResponse.Name+">")
		default:
			labels = append(labels, c.Role+" "+p.Text)
		}
	}
	return labels
}

// TestRun_ModelInputContext_PostedBackFunctionResponse posts a long-running
// tool's result back with its "adk-" call ID and checks where the context
// lands in the last request. On an LLM agent root the run resumes the
// invocation that made the call, whose user content is the message that
// started it ("start"). On a sequential agent root the user content is the
// posted response, which the history matches only when the model keeps the
// IDs.
func TestRun_ModelInputContext_PostedBackFunctionResponse(t *testing.T) {
	tests := []struct {
		name      string
		root      string
		keepsIDs  bool
		wantLabel []string
	}{
		{
			name:     "LLM agent root, model strips IDs",
			root:     "llm",
			keepsIDs: false,
			wantLabel: []string{
				"user earlier", "model hello", "user CTX", "user start",
				"model <call approve>", "user <response approve>",
			},
		},
		{
			name:     "LLM agent root, model keeps IDs",
			root:     "llm",
			keepsIDs: true,
			wantLabel: []string{
				"user earlier", "model hello", "user CTX", "user start",
				"model <call approve>", "user <response approve>",
			},
		},
		{
			name:     "sequential agent root, model strips IDs",
			root:     "sequential",
			keepsIDs: false,
			wantLabel: []string{
				"user CTX", "user earlier", "model hello", "user start",
				"model <call approve>", "user <response approve>",
			},
		},
		{
			name:     "sequential agent root, model keeps IDs",
			root:     "sequential",
			keepsIDs: true,
			wantLabel: []string{
				"user earlier", "model hello", "user start",
				"model <call approve>", "user CTX", "user <response approve>",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := &approvalModel{}
			var llm model.LLM = recorder
			if tt.keepsIDs {
				llm = idPairingModel{recorder}
			}
			approve, err := functiontool.New(functiontool.Config{Name: "approve", Description: "asks a human", IsLongRunning: true},
				func(agent.Context, struct{}) (map[string]any, error) { return map[string]any{"status": "pending"}, nil })
			if err != nil {
				t.Fatalf("functiontool.New: %v", err)
			}
			var root agent.Agent
			root, err = llmagent.New(llmagent.Config{Name: "assistant", Model: llm, Tools: []tool.Tool{approve}})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			if tt.root == "sequential" {
				root, err = sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{root}}})
				if err != nil {
					t.Fatalf("sequentialagent.New: %v", err)
				}
			}
			r, err := runner.New(runner.Config{AppName: "app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true})
			if err != nil {
				t.Fatalf("runner.New: %v", err)
			}
			run := func(msg *genai.Content, cfg agent.RunConfig) (callID string) {
				t.Helper()
				for ev, err := range r.Run(t.Context(), "user", "session", msg, cfg) {
					if err != nil {
						t.Fatalf("Run: %v", err)
					}
					if ev.Content == nil {
						continue
					}
					for _, p := range ev.Content.Parts {
						if p.FunctionCall != nil {
							callID = p.FunctionCall.ID
						}
					}
				}
				return callID
			}

			run(genai.NewContentFromText("earlier", genai.RoleUser), agent.RunConfig{})
			callID := run(genai.NewContentFromText("start", genai.RoleUser), agent.RunConfig{})
			if !strings.HasPrefix(callID, "adk-") {
				t.Fatalf("call ID = %q, want an \"adk-\" ID", callID)
			}
			response := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID: callID, Name: "approve", Response: map[string]any{"status": "approved"},
			}}}}
			run(response, agent.RunConfig{ModelInputContext: &[]*genai.Content{genai.NewContentFromText("CTX", genai.RoleUser)}})

			// Run 2 calls the model again after the tool's "pending" result.
			if len(recorder.requests) != 4 {
				t.Fatalf("model got %d requests, want 4", len(recorder.requests))
			}
			got := contentLabels(recorder.requests[3].Contents)
			if !slices.Equal(got, tt.wantLabel) {
				t.Errorf("last request contents = %q, want %q", got, tt.wantLabel)
			}
		})
	}
}
