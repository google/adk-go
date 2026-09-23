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

package toolresultlimit

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

type resultTool struct {
	result map[string]any
	err    error
}

func (*resultTool) Name() string        { return "fetch" }
func (*resultTool) Description() string { return "Return test data." }
func (*resultTool) IsLongRunning() bool { return false }
func (*resultTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: "fetch", Description: "Return test data."}
}

func (t *resultTool) Run(agent.Context, any) (map[string]any, error) {
	return t.result, t.err
}

func (t *resultTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

func TestRunnerIntegration(t *testing.T) {
	const limit = 384
	large := map[string]any{"data": map[string]any{"items": []string{strings.Repeat("中文🙂", 1000)}}}
	exact := map[string]any{"data": strings.Repeat("x", limit-len(`{"data":""}`))}
	tests := []struct {
		name              string
		result            map[string]any
		toolErr           error
		earlierResult     map[string]any
		calls             int
		wantTruncated     bool
		wantLaterCallback bool
	}{
		{name: "large nested result", result: large, calls: 1, wantTruncated: true},
		{name: "parallel calls sharing a result", result: large, calls: 3, wantTruncated: true},
		{name: "result at limit", result: exact, calls: 1, wantLaterCallback: true},
		{name: "partial result and error", result: large, toolErr: errors.New("fetch failed"), calls: 1, wantLaterCallback: true},
		{name: "unencodable result and error", result: map[string]any{"value": make(chan int)}, toolErr: errors.New("fetch failed"), calls: 1, wantLaterCallback: true},
		{name: "long error preserved", toolErr: errors.New(strings.Repeat("failed ", 100)), calls: 1, wantLaterCallback: true},
		{name: "failure in result", result: map[string]any{"error": strings.Repeat("failed ", 1000)}, calls: 1, wantTruncated: true},
		{name: "earlier replacement bypasses limiter", result: exact, earlierResult: large, calls: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var beforeCalls, afterCalls, agentCalls atomic.Int32
			before, err := plugin.New(plugin.Config{
				Name: "before_limiter",
				AfterToolCallback: func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
					beforeCalls.Add(1)
					return tt.earlierResult, nil
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			limiter, err := New(Config{MaxResultBytes: limit})
			if err != nil {
				t.Fatal(err)
			}
			after, err := plugin.New(plugin.Config{
				Name: "after_limiter",
				AfterToolCallback: func(_ agent.Context, _ tool.Tool, _, _ map[string]any, toolErr error) (map[string]any, error) {
					afterCalls.Add(1)
					if !errors.Is(toolErr, tt.toolErr) {
						t.Error("tool error changed")
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatal(err)
			}

			parts := make([]*genai.Part, tt.calls)
			callIDs := make(map[string]bool, tt.calls)
			for i := range tt.calls {
				id := fmt.Sprint(i)
				callIDs[id] = true
				parts[i] = &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: "fetch", Args: map[string]any{}}}
			}
			model := &testutil.MockModel{Responses: []*genai.Content{
				{Role: genai.RoleModel, Parts: parts},
				genai.NewContentFromText("done", genai.RoleModel),
			}}
			a, err := llmagent.New(llmagent.Config{
				Name:  "test_agent",
				Model: model,
				Tools: []tool.Tool{&resultTool{result: tt.result, err: tt.toolErr}},
				AfterToolCallbacks: []llmagent.AfterToolCallback{
					func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
						agentCalls.Add(1)
						return nil, nil
					},
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			sessions := session.InMemoryService()
			r, err := runner.New(runner.Config{
				AppName:           "test_app",
				Agent:             a,
				SessionService:    sessions,
				AutoCreateSession: true,
				PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{before, limiter, after}},
			})
			if err != nil {
				t.Fatal(err)
			}

			checkContent := func(content *genai.Content, seen map[string]bool) {
				t.Helper()
				if content == nil {
					return
				}
				for _, part := range content.Parts {
					fr := part.FunctionResponse
					if fr == nil {
						continue
					}
					if fr.Name != "fetch" || !callIDs[fr.ID] {
						t.Error("function response name or ID does not match a call")
					}
					if seen[fr.ID] {
						t.Errorf("duplicate response for call %q", fr.ID)
					}
					seen[fr.ID] = true
					if tt.wantTruncated {
						checkTruncated(t, fr.Response, encode(t, tt.result), limit, tt.result["error"] != nil)
						continue
					}
					want := tt.result
					if tt.earlierResult != nil {
						want = tt.earlierResult
					} else if tt.toolErr != nil {
						want = map[string]any{"error": tt.toolErr.Error()}
					}
					if !bytes.Equal(encode(t, fr.Response), encode(t, want)) {
						t.Error("function response differs from the expected result")
					}
				}
			}
			yielded := make(map[string]bool)
			for event, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText("fetch", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Fatal(err)
				}
				checkContent(event.Content, yielded)
			}
			if len(yielded) != tt.calls {
				t.Fatalf("yielded %d tool responses, want %d", len(yielded), tt.calls)
			}
			if len(model.Requests) != 2 {
				t.Fatalf("model received %d requests, want 2", len(model.Requests))
			}
			modelResults := make(map[string]bool)
			for _, content := range model.Requests[1].Contents {
				checkContent(content, modelResults)
			}
			stored, err := sessions.Get(t.Context(), &session.GetRequest{AppName: "test_app", UserID: "user", SessionID: "session"})
			if err != nil {
				t.Fatal(err)
			}
			storedResults := make(map[string]bool)
			for event := range stored.Session.Events().All() {
				checkContent(event.Content, storedResults)
			}
			if len(modelResults) != tt.calls || len(storedResults) != tt.calls {
				t.Errorf("model/session tool responses = %d/%d, want %d each", len(modelResults), len(storedResults), tt.calls)
			}
			wantLater := int32(0)
			if tt.wantLaterCallback {
				wantLater = int32(tt.calls)
			}
			if beforeCalls.Load() != int32(tt.calls) || afterCalls.Load() != wantLater || agentCalls.Load() != wantLater {
				t.Errorf("callback counts before/after/agent = %d/%d/%d, want %d/%d/%d", beforeCalls.Load(), afterCalls.Load(), agentCalls.Load(), tt.calls, wantLater, wantLater)
			}
		})
	}
}
