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
	"errors"
	"iter"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// TestRunner_NodeRoot_RetryReadsAttemptCount checks that a node given as
// Config.Node reads its own retry numbers through the runner's root wrapper.
//
// Fails if the scheduler leaves the number out of the context of a node run
// by a root wrapper: the node reads [1, 1, 1].
func TestRunner_NodeRoot_RetryReadsAttemptCount(t *testing.T) {
	ctx := t.Context()
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	var got []int
	flaky := workflow.NewFunctionNode("flaky", func(ctx agent.Context, _ any) (string, error) {
		got = append(got, agent.AttemptCount(ctx))
		if len(got) < 3 {
			return "", errors.New("attempt failed")
		}
		return "ok", nil
	}, workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{MaxAttempts: 3}})
	r := newNodeRootRunner(t, flaky, svc)

	for _, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}

	if diff := cmp.Diff([]int{1, 2, 3}, got); diff != "" {
		t.Errorf("attempt numbers mismatch (-want +got):\n%s", diff)
	}
}

// attemptToolModel calls the tool "record" on every odd call, fails the
// second and fourth calls, and answers with text on the others.
type attemptToolModel struct {
	mu    sync.Mutex
	calls int
}

func (*attemptToolModel) Name() string { return "scripted" }

func (m *attemptToolModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	m.calls++
	call := m.calls
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		switch {
		case call%2 == 1:
			content := genai.NewContentFromFunctionCall("record", map[string]any{}, genai.RoleModel)
			yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil)
		case call < 6:
			yield(nil, errors.New("model unavailable"))
		default:
			yield(&model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel), FinishReason: genai.FinishReasonStop}, nil)
		}
	}
}

// TestAgentNode_AttemptCount_LlmAgentToolReadsOne checks that a tool the
// model calls inside an LlmAgent reads 1 on every attempt of the retried
// AgentNode that runs the agent, as in adk-python, where every agent run
// gets a fresh context. The agent runs in chat mode, which runs it on a copy
// of the context the AgentNode hands it rather than on a new invocation
// context.
//
// Fails if the AgentNode hands the agent a context that carries the node's
// number and agent.Run does not reset it to 1: the tool reads [1, 2, 3].
func TestAgentNode_AttemptCount_LlmAgentToolReadsOne(t *testing.T) {
	ctx := t.Context()
	var mu sync.Mutex
	var got []int
	record, err := functiontool.New(functiontool.Config{Name: "record", Description: "Records the attempt."},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			mu.Lock()
			defer mu.Unlock()
			got = append(got, agent.AttemptCount(ctx))
			return map[string]any{}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	assistant, err := llmagent.New(llmagent.Config{Name: "assistant", Mode: llmagent.ModeChat, Model: &attemptToolModel{}, Tools: []tool.Tool{record}})
	if err != nil {
		t.Fatal(err)
	}
	node, err := workflow.NewAgentNode(assistant, workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{MaxAttempts: 3}})
	if err != nil {
		t.Fatal(err)
	}
	w, err := workflow.New("wf", workflow.Chain(workflow.Start, node))
	if err != nil {
		t.Fatal(err)
	}
	root, err := agent.New(agent.Config{Name: w.Name(), Run: w.Run})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	newNodeTestSession(t, ctx, svc)
	r, err := runner.New(runner.Config{AppName: nodeTestApp, Agent: root, SessionService: svc})
	if err != nil {
		t.Fatal(err)
	}

	for _, err := range r.Run(ctx, nodeTestUser, nodeTestSession, userText("go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if diff := cmp.Diff([]int{1, 1, 1}, got); diff != "" {
		t.Errorf("tool attempt numbers mismatch (-want +got):\n%s", diff)
	}
}
