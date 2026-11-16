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

package bigquerytoolset

import (
	"context"
	"iter"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"google.golang.org/api/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// callsThenDoneModel answers with calls until the request ends with function
// responses, and then with the text "done".
type callsThenDoneModel []*genai.FunctionCall

func (callsThenDoneModel) Name() string { return "calls-then-done" }

func (m callsThenDoneModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		content := &genai.Content{Role: genai.RoleModel}
		last := req.Contents[len(req.Contents)-1]
		answered := false
		for _, p := range last.Parts {
			answered = answered || p.FunctionResponse != nil
		}
		if answered {
			content.Parts = []*genai.Part{genai.NewPartFromText("done")}
		} else {
			for _, c := range m {
				content.Parts = append(content.Parts, &genai.Part{FunctionCall: c})
			}
		}
		yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil)
	}
}

// TestToolset_TokenReadDoesNotWaitForEarlierCall pins that a BigQuery tool
// reads its access token without waiting for the earlier calls of its batch:
// call 0 uses State and then waits for call 1's request.
func TestToolset_TokenReadDoesNotWaitForEarlierCall(t *testing.T) {
	requestSeen := make(chan struct{})
	var seenOnce sync.Once
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "Bearer tok" {
			seenOnce.Do(func() { close(requestSeen) })
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"datasets":[]}`))
	}))
	t.Cleanup(server.Close)
	ts := newTestToolset(t, Config{
		ClientOptions:          []option.ClientOption{option.WithEndpoint(server.URL + "/bigquery/v2/")},
		ExternalAccessTokenKey: "bq_token",
	})
	holder, err := functiontool.New(functiontool.Config{Name: "holder", Description: "holder"}, func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		if err := ctx.State().Set("a", 1); err != nil {
			return nil, err
		}
		select {
		case <-requestSeen:
			return map[string]any{"released": true}, nil
		case <-time.After(5 * time.Second):
			return map[string]any{"released": false}, nil
		}
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name: "root",
		Model: callsThenDoneModel{
			{ID: "call-0", Name: "holder", Args: map[string]any{}},
			{ID: "call-1", Name: "list_dataset_ids", Args: map[string]any{"project_id": "p"}},
		},
		Tools:    []tool.Tool{holder},
		Toolsets: []tool.Toolset{ts},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: root, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", State: map[string]any{"bq_token": "tok"}})
	if err != nil {
		t.Fatalf("session.Create() error = %v", err)
	}

	got := map[string]map[string]any{}
	for ev, err := range r.Run(t.Context(), "user", created.Session.ID(), genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if fr := p.FunctionResponse; fr != nil {
				got[fr.Name] = fr.Response
			}
		}
	}

	if released := got["holder"]["released"]; released != true {
		t.Errorf("holder response = %v, want released true", got["holder"])
	}
	if status := got["list_dataset_ids"]["status"]; status == "ERROR" {
		t.Errorf("list_dataset_ids response = %v, want success", got["list_dataset_ids"])
	}
}

// TestToolset_TokenWrittenIntoTheCallsDelta pins that a token written straight
// into the call's state delta (as a before-tool callback can do) is found,
// although the session state does not hold it, and that it wins over the
// session's value, as State().Get reads the delta first.
func TestToolset_TokenWrittenIntoTheCallsDelta(t *testing.T) {
	tests := []struct {
		name         string
		sessionState map[string]any
	}{
		{name: "session does not hold the token"},
		{name: "delta wins over the session value", sessionState: map[string]any{"bq_token": "old"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeBigQuery(t, map[string]fakeResponse{
				"GET /bigquery/v2/projects/p/datasets": {body: `{"datasets":[]}`},
			})
			ts := newTestToolset(t, Config{
				ClientOptions:          []option.ClientOption{option.WithEndpoint(f.server.URL + "/bigquery/v2/")},
				ExternalAccessTokenKey: "bq_token",
			})
			created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", State: tt.sessionState})
			if err != nil {
				t.Fatalf("session.Create() error = %v", err)
			}
			invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: created.Session})
			toolCtx := agent.NewToolContext(invCtx, "call-1", &session.EventActions{StateDelta: map[string]any{"bq_token": "tok"}}, nil)

			got, err := toolByName(t, ts, "list_dataset_ids").Run(toolCtx, map[string]any{"project_id": "p"})
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}

			if got["status"] == "ERROR" {
				t.Fatalf("Run() = %v, want success", got)
			}
			if auth := f.lastRequest().Header.Get("Authorization"); auth != "Bearer tok" {
				t.Errorf("Authorization = %q, want %q", auth, "Bearer tok")
			}
		})
	}
}
