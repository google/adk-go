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

package adkrest_test

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http/httptest"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

func TestRESTHITLConfirmationReplayDoesNotReexecuteTool(t *testing.T) {
	recorder := &confirmationReplayRecorder{}
	srv := httptest.NewServer(newConfirmationReplayServer(t, recorder))
	defer srv.Close()

	sid := createConfirmationReplaySession(t, srv.URL)
	paused := runConfirmationReplayTurn(t, srv.URL, sid, genai.NewContentFromText("approve transfer", genai.RoleUser))
	confirmationID := confirmationReplayID(paused)
	if confirmationID == "" {
		t.Fatalf("fresh turn did not request tool confirmation; events = %+v", paused)
	}

	runConfirmationReplayTurn(t, srv.URL, sid, confirmationReplayResponse(t, confirmationID, map[string]any{
		"approved_scope": "low-risk",
	}))
	recorder.want(t, 1, `{"approved_scope":"low-risk"}`)

	runConfirmationReplayTurn(t, srv.URL, sid, confirmationReplayResponse(t, confirmationID, map[string]any{
		"approved_scope": "high-risk",
	}))
	recorder.want(t, 1, `{"approved_scope":"low-risk"}`)
}

type confirmationReplayArgs struct {
	Target string `json:"target"`
}

type confirmationReplayResult struct {
	OK bool `json:"ok"`
}

type confirmationReplayRecorder struct {
	mu              sync.Mutex
	executions      int
	lastPayloadJSON string
}

func (r *confirmationReplayRecorder) record(payload any) {
	raw, _ := json.Marshal(payload)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.executions++
	r.lastPayloadJSON = string(raw)
}

func (r *confirmationReplayRecorder) want(t *testing.T, executions int, payloadJSON string) {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.executions != executions || r.lastPayloadJSON != payloadJSON {
		t.Fatalf("tool executions = %d and last payload = %s, want %d and %s",
			r.executions, r.lastPayloadJSON, executions, payloadJSON)
	}
}

type confirmationReplayModel struct {
	model.LLM
	calls int
}

func (m *confirmationReplayModel) Name() string {
	return "confirmation-replay-model"
}

func (m *confirmationReplayModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		if m.calls > 1 {
			yield(&model.LLMResponse{
				Content: genai.NewContentFromText("done", genai.RoleModel),
			}, nil)
			return
		}
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   "transfer-call-1",
						Name: "transfer_funds",
						Args: map[string]any{"target": "prod"},
					},
				}},
			},
		}, nil)
	}
}

type confirmationReplayEvent struct {
	Content *genai.Content `json:"content"`
}

func newConfirmationReplayServer(t *testing.T, recorder *confirmationReplayRecorder) *adkrest.Server {
	t.Helper()
	transferTool, err := functiontool.New(functiontool.Config{
		Name:                "transfer_funds",
		Description:         "transfers funds",
		RequireConfirmation: true,
	}, func(ctx agent.Context, args confirmationReplayArgs) (confirmationReplayResult, error) {
		recorder.record(ctx.ToolConfirmation().Payload)
		return confirmationReplayResult{OK: true}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	a, err := llmagent.New(llmagent.Config{
		Name:  confirmationReplayApp,
		Model: &confirmationReplayModel{},
		Tools: []tool.Tool{transferTool},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	srv, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService: session.InMemoryService(),
		AgentLoader:    agent.NewSingleLoader(a),
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() error = %v", err)
	}
	return srv
}

func createConfirmationReplaySession(t *testing.T, baseURL string) string {
	t.Helper()
	var resp struct {
		ID string `json:"id"`
	}
	postJSON(t, fmt.Sprintf("%s/apps/%s/users/%s/sessions", baseURL, confirmationReplayApp, confirmationReplayUser),
		map[string]any{}, &resp)
	if resp.ID == "" {
		t.Fatal("create session returned an empty ID")
	}
	return resp.ID
}

func runConfirmationReplayTurn(t *testing.T, baseURL, sid string, msg *genai.Content) []confirmationReplayEvent {
	t.Helper()
	body := map[string]any{
		"appName":    confirmationReplayApp,
		"userId":     confirmationReplayUser,
		"sessionId":  sid,
		"newMessage": msg,
	}
	var events []confirmationReplayEvent
	postJSON(t, baseURL+"/run", body, &events)
	return events
}

func confirmationReplayID(events []confirmationReplayEvent) string {
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if part.FunctionCall != nil && part.FunctionCall.Name == toolconfirmation.FunctionCallName {
				return part.FunctionCall.ID
			}
		}
	}
	return ""
}

func confirmationReplayResponse(t *testing.T, confirmationID string, payload map[string]any) *genai.Content {
	t.Helper()
	confirmation := toolconfirmation.ToolConfirmation{Confirmed: true, Payload: payload}
	raw, err := json.Marshal(confirmation)
	if err != nil {
		t.Fatalf("marshal confirmation: %v", err)
	}
	return &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       confirmationID,
				Name:     toolconfirmation.FunctionCallName,
				Response: map[string]any{"response": string(raw)},
			},
		}},
	}
}

const (
	confirmationReplayApp  = "confirmation_replay"
	confirmationReplayUser = "u"
)
