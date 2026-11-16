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

package llminternal_test

import (
	"context"
	"iter"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// bfcModel answers its requests with responses in order, and the text "done"
// once they run out. It records every request.
type bfcModel struct {
	mu        sync.Mutex
	responses []*model.LLMResponse
	requests  []*model.LLMRequest
}

func (m *bfcModel) Name() string { return "bfc-model" }

func (m *bfcModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		m.requests = append(m.requests, req)
		resp := &model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel)}
		if len(m.responses) > 0 {
			resp, m.responses = m.responses[0], m.responses[1:]
		}
		m.mu.Unlock()
		yield(resp, nil)
	}
}

// bfcRun runs a to its end through a runner with plugins, on the user text
// "go", and returns the events it yielded.
func bfcRun(t *testing.T, a agent.Agent, plugins ...*plugin.Plugin) []*session.Event {
	t.Helper()
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:        "app",
		Agent:          a,
		SessionService: svc,
		PluginConfig:   runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatal(err)
	}
	var events []*session.Event
	for ev, err := range r.Run(t.Context(), "user", "s", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		events = append(events, ev)
	}
	return events
}

func bfcAgent(t *testing.T, cfg llmagent.Config) agent.Agent {
	t.Helper()
	cfg.Name = "agent_1"
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// A non-live run builds the function response event without a live session
// id even when the model response it answers has one, as adk-python's
// non-live path passes none (flows/llm_flows/tools/_batch_executor.py:287-299).
func TestRun_NonLiveFunctionResponseDoesNotCopyLiveSessionID(t *testing.T) {
	myTool, err := functiontool.New(functiontool.Config{Name: "my_tool", Description: "my_tool"}, func(agent.Context, struct{}) (map[string]any, error) {
		return map[string]any{"ok": true}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	m := &bfcModel{responses: []*model.LLMResponse{{
		Content:       &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "my_tool", Args: map[string]any{}}}}},
		LiveSessionID: "spurious-non-live-id",
	}}}

	events := bfcRun(t, bfcAgent(t, llmagent.Config{Model: m, Tools: []tool.Tool{myTool}}))

	var got []string
	for _, ev := range events {
		got = append(got, ev.LiveSessionID)
	}
	// The model call event, the function response event, the final text.
	if diff := cmp.Diff([]string{"spurious-non-live-id", "", ""}, got); diff != "" {
		t.Errorf("LiveSessionID of the events mismatch (-want +got):\n%s", diff)
	}
}

// For a runnable function tool, the before-tool callbacks of plugins and of
// the agent, and the tool, share one deep copy of the call's arguments: a
// callback's change reaches the tool, and leaves the model event and the next
// model request as the model sent them.
func TestHandleFunctionCalls_RunnableToolCallbackMutationDoesNotMutateModelEventOrNextRequest(t *testing.T) {
	var toolArgs map[string]any
	mutateTool, err := functiontool.New(functiontool.Config{Name: "mutate_tool", Description: "mutate_tool"}, func(_ agent.Context, args map[string]any) (map[string]any, error) {
		toolArgs = args
		nested, _ := args["nested"].(map[string]any)
		return map[string]any{"seen_k": nested["k"], "seen_added": args["added"]}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	p, err := plugin.New(plugin.Config{
		Name: "mutating_plugin",
		BeforeToolCallback: func(_ agent.Context, _ tool.Tool, args map[string]any) (map[string]any, error) {
			args["nested"].(map[string]any)["k"] = "from_plugin"
			return nil, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	m := &bfcModel{responses: []*model.LLMResponse{{
		Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: "call-1", Name: "mutate_tool", Args: map[string]any{"top": "orig", "nested": map[string]any{"k": "v"}},
		}}}},
	}}}
	a := bfcAgent(t, llmagent.Config{
		Model: m,
		Tools: []tool.Tool{mutateTool},
		BeforeToolCallbacks: []llmagent.BeforeToolCallback{func(_ agent.Context, _ tool.Tool, args map[string]any) (map[string]any, error) {
			args["added"] = "from_agent"
			return nil, nil
		}},
	})

	events := bfcRun(t, a, p)

	if len(events) != 3 || len(m.requests) != 2 {
		t.Fatalf("got %d events and %d model requests, want 3 and 2", len(events), len(m.requests))
	}
	modelSent := map[string]any{"top": "orig", "nested": map[string]any{"k": "v"}}
	if diff := cmp.Diff(map[string]any{"top": "orig", "nested": map[string]any{"k": "from_plugin"}, "added": "from_agent"}, toolArgs); diff != "" {
		t.Errorf("arguments the tool received mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(modelSent, events[0].Content.Parts[0].FunctionCall.Args); diff != "" {
		t.Errorf("model event's call arguments mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(modelSent, m.requests[1].Contents[1].Parts[0].FunctionCall.Args); diff != "" {
		t.Errorf("next request's call arguments mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"seen_k": "from_plugin", "seen_added": "from_agent"}, events[1].Content.Parts[0].FunctionResponse.Response); diff != "" {
		t.Errorf("function response mismatch (-want +got):\n%s", diff)
	}
}
