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

package agenttool_test

import (
	"context"
	"errors"
	"iter"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
)

// parallelGuard bounds every wait in these tests that must end.
const parallelGuard = 5 * time.Second

// funcModel answers every request with its function.
type funcModel func(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error)

func (funcModel) Name() string { return "func-model" }

func (m funcModel) GenerateContent(ctx context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(m(ctx, req))
	}
}

func textResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleModel), FinishReason: genai.FinishReasonStop}
}

// callsThenDone answers with calls until the request ends with function
// responses, and then with the text "done".
func callsThenDone(calls ...*genai.FunctionCall) funcModel {
	return func(_ context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
		last := req.Contents[len(req.Contents)-1]
		for _, p := range last.Parts {
			if p.FunctionResponse != nil {
				return textResponse("done"), nil
			}
		}
		content := &genai.Content{Role: genai.RoleModel}
		for _, c := range calls {
			content.Parts = append(content.Parts, &genai.Part{FunctionCall: c})
		}
		return &model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil
	}
}

func newLLMAgent(t *testing.T, cfg llmagent.Config) agent.Agent {
	t.Helper()
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New(%s) error = %v", cfg.Name, err)
	}
	return a
}

// runOnce runs a over a new in-memory session and returns the function
// responses of the run by tool name. It returns errors rather than failing the
// test, because tools call it on the flow's goroutines.
func runOnce(ctx context.Context, a agent.Agent) (map[string]map[string]any, error) {
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc})
	if err != nil {
		return nil, err
	}
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		return nil, err
	}
	got := map[string]map[string]any{}
	for ev, err := range r.Run(ctx, "user", created.Session.ID(), genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			return nil, err
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
	return got, nil
}

// readK is a before-model callback that uses State.
func readK(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
	if _, err := ctx.State().Get("k"); err != nil && !errors.Is(err, session.ErrStateKeyNotExist) {
		return nil, err
	}
	return nil, nil
}

// TestAgentTool_ParallelAgentToolsOverlapWithStateCallbacks pins that two
// agent tools called in one turn run their sub-agents at the same time, even
// when each sub-agent's before-model callback uses State: each sub-agent's
// model waits until the other's has started.
func TestAgentTool_ParallelAgentToolsOverlapWithStateCallbacks(t *testing.T) {
	aStarted, bStarted := make(chan struct{}), make(chan struct{})
	waitingModel := func(started chan<- struct{}, sibling <-chan struct{}) funcModel {
		var once sync.Once
		return func(context.Context, *model.LLMRequest) (*model.LLMResponse, error) {
			once.Do(func() { close(started) })
			select {
			case <-sibling:
				return textResponse("ok"), nil
			case <-time.After(parallelGuard):
				return nil, errors.New("sibling sub-agent never started")
			}
		}
	}
	subA := newLLMAgent(t, llmagent.Config{
		Name:                 "sub_a",
		Description:          "sub_a",
		Model:                waitingModel(aStarted, bStarted),
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{readK},
	})
	subB := newLLMAgent(t, llmagent.Config{
		Name:                 "sub_b",
		Description:          "sub_b",
		Model:                waitingModel(bStarted, aStarted),
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{readK},
	})
	root := newLLMAgent(t, llmagent.Config{
		Name: "root",
		Model: callsThenDone(
			&genai.FunctionCall{ID: "call-0", Name: "sub_a", Args: map[string]any{"request": "a"}},
			&genai.FunctionCall{ID: "call-1", Name: "sub_b", Args: map[string]any{"request": "b"}},
		),
		Tools: []tool.Tool{agenttool.New(subA, nil), agenttool.New(subB, nil)},
	})

	got, err := runOnce(t.Context(), root)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	want := map[string]map[string]any{
		"sub_a": {"result": "ok"},
		"sub_b": {"result": "ok"},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
}
