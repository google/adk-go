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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// slowModel asks for a tool call on every turn until it has been called
// finishAfter times, then answers. It stands in for a child agent that needs
// more model calls than some budget allows.
type slowModel struct {
	model.LLM
	finishAfter int
	calls       int
}

func (m *slowModel) Name() string { return "slow-mock" }

func (m *slowModel) GenerateContent(ctx context.Context, req *model.LLMRequest, useStream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		part := &genai.Part{FunctionCall: &genai.FunctionCall{ID: "fc", Name: "noop", Args: map[string]any{}}}
		if m.calls >= m.finishAfter {
			part = genai.NewPartFromText("done")
		}
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

// runChildUnderBudget runs a child agent through agenttool from a tool context
// whose invocation has the given resolved model-call budget, the way the
// runner sets it up for the calling agent.
func runChildUnderBudget(t *testing.T, m *slowModel, maxLLMCalls int) (map[string]any, error) {
	t.Helper()

	noop, err := functiontool.New(functiontool.Config{Name: "noop"}, func(ctx agent.Context, in struct{}) (struct{}, error) {
		return struct{}{}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}
	child, err := llmagent.New(llmagent.Config{
		Name:  "child",
		Model: m,
		Tools: []tool.Tool{noop},
	})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}

	sessionService := session.InMemoryService()
	created, err := sessionService.Create(t.Context(), &session.CreateRequest{AppName: "testApp", UserID: "testUser"})
	if err != nil {
		t.Fatalf("session Create() failed: %v", err)
	}
	ctx := runconfig.ToContext(t.Context(), &runconfig.RunConfig{MaxLLMCalls: maxLLMCalls})
	ictx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Session: created.Session})
	toolCtx := agent.NewToolContext(ictx, "", &session.EventActions{}, nil)

	toolImpl, ok := agenttool.New(child, nil).(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}
	return toolImpl.Run(toolCtx, map[string]any{"request": "go"})
}

func TestAgentTool_Run_ForwardsMaxLLMCalls(t *testing.T) {
	t.Run("no limit lets a child past the default finish", func(t *testing.T) {
		// Shrink the default so the child does not have to make 500 calls to
		// go past it. Without forwarding, the nested run would get this default.
		t.Setenv(runconfig.MaxLLMCallsEnvVar, "5")
		need := runconfig.ResolveMaxLLMCalls(0) + 1
		m := &slowModel{finishAfter: need}

		result, err := runChildUnderBudget(t, m, -1)
		if err != nil {
			t.Fatalf("Run() failed: %v", err)
		}
		if got := result["result"]; got != "done" {
			t.Errorf("Run() result = %v, want the child's final answer", result)
		}
		if m.calls != need {
			t.Errorf("child model calls = %d, want %d", m.calls, need)
		}
	})

	t.Run("a small limit bounds the child too", func(t *testing.T) {
		m := &slowModel{finishAfter: 10}

		_, err := runChildUnderBudget(t, m, 2)
		if !errors.Is(err, agent.ErrLLMCallsLimitExceeded) {
			t.Fatalf("Run() error = %v, want it to wrap ErrLLMCallsLimitExceeded", err)
		}
		if m.calls != 2 {
			t.Errorf("child model calls = %d, want 2", m.calls)
		}
	})
}
