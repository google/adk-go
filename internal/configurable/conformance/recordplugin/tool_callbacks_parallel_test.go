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

package recordplugin_test

import (
	"context"
	"iter"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/configurable/conformance/recordplugin"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// twoCalls is a model turn of the calls call-0 to wait_for_sibling and call-1
// to quick, and then the text "done".
type twoCalls struct{}

func (twoCalls) Name() string { return "two-calls" }

func (twoCalls) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
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
			content.Parts = []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call-0", Name: "wait_for_sibling", Args: map[string]any{}}},
				{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "quick", Args: map[string]any{}}},
			}
		}
		yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil)
	}
}

// TestRecordPlugin_ToolCallbacksDoNotWaitForEarlierCalls pins that the
// plugin's before-tool and after-tool callbacks do not make a call wait for the
// earlier calls of its batch: call 0 waits until the agent's after-tool
// callback, which runs after the plugin's, has seen call 1.
func TestRecordPlugin_ToolCallbacksDoNotWaitForEarlierCalls(t *testing.T) {
	siblingDone := make(chan struct{})
	waitForSibling, err := functiontool.New(functiontool.Config{Name: "wait_for_sibling", Description: "wait_for_sibling"}, func(agent.Context, struct{}) (map[string]any, error) {
		select {
		case <-siblingDone:
			return map[string]any{"released": true}, nil
		case <-time.After(5 * time.Second):
			return map[string]any{"released": false}, nil
		}
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	quick, err := functiontool.New(functiontool.Config{Name: "quick", Description: "quick"}, func(agent.Context, struct{}) (map[string]any, error) {
		return map[string]any{"ok": true}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:  "root",
		Model: twoCalls{},
		Tools: []tool.Tool{waitForSibling, quick},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			func(ctx agent.Context, _ tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
				if ctx.FunctionCallID() == "call-1" {
					close(siblingDone)
				}
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	record, err := recordplugin.New(t.TempDir())
	if err != nil {
		t.Fatalf("recordplugin.New() error = %v", err)
	}
	r := testutil.NewTestAgentRunnerWithPluginManager(t, root, runner.PluginConfig{Plugins: []*plugin.Plugin{record}})

	got := map[string]map[string]any{}
	for ev, err := range r.Run(t, "session", "go") {
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

	want := map[string]map[string]any{
		"wait_for_sibling": {"released": true},
		"quick":            {"ok": true},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
}
