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
	"context"
	"errors"
	"iter"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

var errScriptExhausted = errors.New("scripted model: no response left")

// scriptedModel is a model.LLM double that returns its responses in order,
// records every request it receives, and fails once the script runs out.
type scriptedModel struct {
	mu        sync.Mutex
	responses []*model.LLMResponse
	requests  []*model.LLMRequest
}

func (m *scriptedModel) Name() string { return "scripted" }

func (m *scriptedModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		m.requests = append(m.requests, req)
		if len(m.responses) == 0 {
			m.mu.Unlock()
			yield(nil, errScriptExhausted)
			return
		}
		resp := m.responses[0]
		m.responses = m.responses[1:]
		m.mu.Unlock()
		yield(resp, nil)
	}
}

func (m *scriptedModel) recorded() []*model.LLMRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.requests
}

func textResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{
		Content:      genai.NewContentFromText(text, genai.RoleModel),
		FinishReason: genai.FinishReasonStop,
	}
}

// agentRun holds what runAgentOnce saw: the events up to the first error, and
// that error.
type agentRun struct {
	events []*session.Event
	err    error
}

// runAgentOnce runs a once on a new in-memory session and collects the events
// until the first error.
func runAgentOnce(t *testing.T, a agent.Agent) agentRun {
	t.Helper()
	ctx := t.Context()
	svc := session.InMemoryService()
	key := session.GetRequest{AppName: "test_app", UserID: "test_user", SessionID: "s1"}
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: key.AppName, UserID: key.UserID, SessionID: key.SessionID}); err != nil {
		t.Fatalf("session Create: %v", err)
	}
	r, err := runner.New(runner.Config{AppName: key.AppName, Agent: a, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	var run agentRun
	for ev, err := range r.Run(ctx, key.UserID, key.SessionID, genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			run.err = err
			break
		}
		run.events = append(run.events, ev)
	}
	return run
}

func newTestAgent(t *testing.T, cfg llmagent.Config) agent.Agent {
	t.Helper()
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	return a
}

func hasText(events []*session.Event, text string) bool {
	for _, ev := range events {
		if ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.Text == text {
				return true
			}
		}
	}
	return false
}
