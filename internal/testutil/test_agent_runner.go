// Copyright 2025 Google LLC
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

package testutil

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
	"google.golang.org/genai"
)

type TestAgentRunner struct {
	agent          agent.Agent
	sessionService sessionservice.Service
	lastSession    sessionservice.StoredSession
	appName        string
	// TODO: move runner definition to the adk package and it's a part of public api, but the logic to the internal runner
	runner *runner.Runner
}

func (r *TestAgentRunner) session(t *testing.T, appName, userID, sessionID string) (sessionservice.StoredSession, error) {
	ctx := t.Context()
	if last := r.lastSession; last != nil && last.ID().SessionID == sessionID {
		resp, err := r.sessionService.Get(ctx, &sessionservice.GetRequest{
			ID: session.ID{
				AppName:   "test_app",
				UserID:    "test_user",
				SessionID: sessionID,
			},
		})
		r.lastSession = resp.Session
		return resp.Session, err
	}
	resp, err := r.sessionService.Create(ctx, &sessionservice.CreateRequest{
		AppName:   "test_app",
		UserID:    "test_user",
		SessionID: sessionID,
	})
	r.lastSession = resp.Session
	return resp.Session, err
}

func (r *TestAgentRunner) Run(t *testing.T, sessionID, newMessage string) iter.Seq2[*session.Event, error] {
	t.Helper()
	ctx := t.Context()

	userID := "test_user"

	session, err := r.session(t, r.appName, userID, sessionID)
	if err != nil {
		t.Fatalf("failed to get/create session: %v", err)
	}

	var content *genai.Content
	if newMessage != "" {
		content = genai.NewContentFromText(newMessage, genai.RoleUser)
	}

	return r.runner.Run(ctx, userID, session.ID().SessionID, content, &runner.RunConfig{})
}

func NewTestAgentRunner(t *testing.T, agent agent.Agent) *TestAgentRunner {
	appName := "test_app"
	sessionService := sessionservice.Mem()

	runner, err := runner.New(&runner.Config{
		AppName:        appName,
		Agent:          agent,
		SessionService: sessionService,
	})
	if err != nil {
		t.Fatal(err)
	}

	return &TestAgentRunner{
		agent:          agent,
		sessionService: sessionService,
		appName:        appName,
		runner:         runner,
	}
}

type MockModel struct {
	Requests  []*llm.Request
	Responses []*genai.Content
}

var errNoModelData = errors.New("no data")

// GenerateContent implements llm.Model.
func (m *MockModel) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	m.Requests = append(m.Requests, req)
	if len(m.Responses) == 0 {
		return nil, errNoModelData
	}

	resp := &llm.Response{
		Content: m.Responses[0],
	}

	m.Responses = m.Responses[1:]

	return resp, nil
}

func (m *MockModel) GenerateStream(ctx context.Context, req *llm.Request) iter.Seq2[*llm.Response, error] {
	return func(yield func(*llm.Response, error) bool) {
		if len(m.Responses) > 0 {
			resp := &llm.Response{Content: m.Responses[0]}
			m.Responses = m.Responses[1:]
			yield(resp, nil)
			return
		}
		yield(nil, fmt.Errorf("no more data"))
	}
}

// Name implements llm.Model.
func (m *MockModel) Name() string {
	return "mock"
}

var _ llm.Model = (*MockModel)(nil)

// CollectTextParts collects all text parts from the llm response until encountering an error.
// It returns all collected text parts and the last error.
func CollectTextParts(stream iter.Seq2[*session.Event, error]) ([]string, error) {
	var texts []string
	for ev, err := range stream {
		if err != nil {
			return texts, err
		}
		if ev == nil || ev.LLMResponse == nil || ev.LLMResponse.Content == nil {
			return texts, fmt.Errorf("unexpected empty event: %v", ev)
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.Text != "" {
				texts = append(texts, p.Text)
			}
		}
	}
	return texts, nil
}
