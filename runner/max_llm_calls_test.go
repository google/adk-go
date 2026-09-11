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

package runner

import (
	"context"
	"errors"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// loopingModel never produces a final answer: every turn asks for the same tool
// again. Without a budget the flow calls it forever, appending an event per turn
// and re-sending a growing history.
type loopingModel struct {
	model.LLM
	calls int
}

func (m *loopingModel) Name() string { return "looping-mock" }

func (m *loopingModel) GenerateContent(ctx context.Context, req *model.LLMRequest, useStream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					ID:   "fc",
					Name: "noop",
					Args: map[string]any{},
				}}},
			},
		}, nil)
	}
}

func noopToolForTest(t *testing.T) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: "noop"}, func(ctx agent.Context, in struct{}) (struct{}, error) {
		return struct{}{}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}
	return tl
}

func runLoopingAgent(t *testing.T, cfg agent.RunConfig) (calls int, err error) {
	t.Helper()

	m := &loopingModel{}
	a, aerr := llmagent.New(llmagent.Config{
		Name:  "looper",
		Model: m,
		Tools: []tool.Tool{noopToolForTest(t)},
	})
	if aerr != nil {
		t.Fatalf("llmagent.New() failed: %v", aerr)
	}
	r, rerr := New(Config{
		AppName:           "testApp",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if rerr != nil {
		t.Fatalf("runner.New() failed: %v", rerr)
	}

	msg := genai.NewContentFromText("go", genai.RoleUser)
	for _, e := range r.Run(t.Context(), "u", "s", msg, cfg) {
		if e != nil {
			err = e
			break
		}
	}
	return m.calls, err
}

func TestRun_MaxLLMCallsBoundsTheRun(t *testing.T) {
	calls, err := runLoopingAgent(t, agent.RunConfig{MaxLLMCalls: 3})

	if !errors.Is(err, agent.ErrLLMCallsLimitExceeded) {
		t.Fatalf("Run() error = %v, want it to wrap ErrLLMCallsLimitExceeded", err)
	}
	// The budget is spent, then the next call is refused, so the model is
	// reached exactly MaxLLMCalls times.
	if calls != 3 {
		t.Errorf("model calls = %d, want 3", calls)
	}
}

func TestRun_MaxLLMCallsDefaultsToABudget(t *testing.T) {
	// An empty RunConfig must not mean "unlimited": that is the state this
	// change exists to remove. Keep the budget small via the environment so the
	// test does not make 500 calls.
	t.Setenv("ADK_MAX_LLM_CALLS", "2")

	calls, err := runLoopingAgent(t, agent.RunConfig{})

	if !errors.Is(err, agent.ErrLLMCallsLimitExceeded) {
		t.Fatalf("Run() error = %v, want it to wrap ErrLLMCallsLimitExceeded", err)
	}
	if calls != 2 {
		t.Errorf("model calls = %d, want 2", calls)
	}
}

func TestRun_MaxLLMCallsNegativeMeansUnlimited(t *testing.T) {
	// Cannot assert "runs forever", so assert the limit is not what stops it:
	// cancel the context and check the error is the cancellation, not the budget.
	m := &loopingModel{}
	a, err := llmagent.New(llmagent.Config{
		Name:  "looper",
		Model: m,
		Tools: []tool.Tool{noopToolForTest(t)},
	})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	r, err := New(Config{
		AppName:           "testApp",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New() failed: %v", err)
	}

	msg := genai.NewContentFromText("go", genai.RoleUser)
	const stopAfter = 12
	var gotErr error
	for _, e := range r.Run(t.Context(), "u", "s", msg, agent.RunConfig{MaxLLMCalls: -1}) {
		if e != nil {
			gotErr = e
			break
		}
		if m.calls >= stopAfter {
			break
		}
	}

	if errors.Is(gotErr, agent.ErrLLMCallsLimitExceeded) {
		t.Errorf("Run() was stopped by the budget with MaxLLMCalls = -1")
	}
	if m.calls < stopAfter {
		t.Errorf("model calls = %d, want at least %d before the consumer stopped", m.calls, stopAfter)
	}
}
