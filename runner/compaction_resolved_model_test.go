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
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// defaultSummaryLLM answers every request with the text "ok" and counts the
// requests.
type defaultSummaryLLM struct {
	mu       sync.Mutex
	requests int
}

func (*defaultSummaryLLM) Name() string { return "default-summary-model" }

func (m *defaultSummaryLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	m.requests++
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("ok", genai.RoleModel), FinishReason: genai.FinishReasonStop}, nil)
	}
}

func (m *defaultSummaryLLM) requestCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.requests
}

func TestCompaction_DefaultSummarizerUsesResolvedModel(t *testing.T) {
	defaultLLM := &defaultSummaryLLM{}
	if err := llmagent.SetDefaultModel(defaultLLM); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := llmagent.SetDefaultModelByName(llmagent.DefaultModel); err != nil {
			t.Errorf("restore default model: %v", err)
		}
	})
	root, err := llmagent.New(llmagent.Config{Name: "root"})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             root,
		SessionService:    svc,
		AutoCreateSession: true,
		Compaction:        &compaction.Config{CompactionInterval: 2},
	})
	if err != nil {
		t.Fatalf("runner.New with a model-less root and no Summarizer = %v, want nil (the default summarizer uses the resolved model)", err)
	}

	for _, text := range []string{"q1", "q2"} {
		for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText(text, genai.RoleUser), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("Run(%q): %v", text, err)
			}
		}
	}

	// Two turns, then the summary of both once the interval is reached.
	if got := defaultLLM.requestCount(); got != 3 {
		t.Errorf("the resolved default model got %d requests, want 3 (two turns and one summary)", got)
	}
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}
	compactions := 0
	for ev := range resp.Session.Events().All() {
		if ev.Actions.Compaction != nil {
			compactions++
		}
	}
	if compactions != 1 {
		t.Errorf("session holds %d compaction events, want 1", compactions)
	}
}
