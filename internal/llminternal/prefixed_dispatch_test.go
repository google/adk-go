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
	"iter"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// listToolset lists tools on every call.
type listToolset struct{ tools []tool.Tool }

func (l *listToolset) Name() string                                     { return "list" }
func (l *listToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return l.tools, nil }

// displayTool asks for its result to be shown when it skips summarization,
// and does skip it.
type displayTool struct{ name string }

func (d *displayTool) Name() string                           { return d.name }
func (d *displayTool) Description() string                    { return "shows its result" }
func (d *displayTool) IsLongRunning() bool                    { return false }
func (d *displayTool) DisplayResultOnSkipSummarization() bool { return true }
func (d *displayTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: d.name, Description: "shows its result"}
}

func (d *displayTool) Run(ctx agent.Context, _ any) (map[string]any, error) {
	ctx.Actions().SkipSummarization = true
	return map[string]any{"result": "final answer"}, nil
}

func (d *displayTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, d)
}

// letterStreamTool streams "a" and "b".
type letterStreamTool struct{ name string }

func (s *letterStreamTool) Name() string        { return s.name }
func (s *letterStreamTool) Description() string { return "streams letters" }
func (s *letterStreamTool) IsLongRunning() bool { return false }
func (s *letterStreamTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: s.name, Description: "streams letters"}
}

func (s *letterStreamTool) RunStream(agent.Context, any) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		if yield("a", nil) {
			yield("b", nil)
		}
	}
}

func (s *letterStreamTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, s)
}

// runWithPrefixedTool runs an agent whose own tools are own and whose only
// toolset lists listed under the prefix "p", against a model that calls the
// tool named call once. It returns the events and the sorted names in the
// first model request's req.Tools.
func runWithPrefixedTool(t *testing.T, own []tool.Tool, listed tool.Tool, call string) ([]*session.Event, []string) {
	t.Helper()
	var requestTools []string
	capture := func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
		if requestTools == nil {
			requestTools = slices.Sorted(maps.Keys(req.Tools))
		}
		return nil, nil
	}
	a, err := llmagent.New(llmagent.Config{
		Name:                 "tester",
		Model:                &singleCallMockModel{toolName: call, fcID: "fc-1"},
		Tools:                own,
		Toolsets:             []tool.Toolset{tool.WithToolNamePrefix(&listToolset{tools: []tool.Tool{listed}}, "p")},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{capture},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	return runSkipSummarizationScenario(t, a), requestTools
}

// functionResponseEvents returns the events that carry a function response.
func functionResponseEvents(events []*session.Event) []*session.Event {
	var out []*session.Event
	for _, ev := range events {
		if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].FunctionResponse != nil {
			out = append(out, ev)
		}
	}
	return out
}

func TestRun_PrefixedToolDispatch(t *testing.T) {
	t.Run("deferring tool skips its nil response", func(t *testing.T) {
		d := &deferringTool{name: "deferring", defers: true}

		events, _ := runWithPrefixedTool(t, nil, d, "p_deferring")

		if d.runCount != 1 {
			t.Errorf("runCount = %d, want 1", d.runCount)
		}
		if len(events) != 2 {
			t.Errorf("got %d events, want 2 (call, final text): %v", len(events), events)
		}
		if got := functionResponseEvents(events); len(got) != 0 {
			t.Errorf("function response events = %v, want none", got)
		}
	})

	t.Run("displaying tool shows its result", func(t *testing.T) {
		events, _ := runWithPrefixedTool(t, nil, &displayTool{name: "show"}, "p_show")

		frs := functionResponseEvents(events)
		if len(frs) != 1 {
			t.Fatalf("got %d function response events, want 1: %v", len(frs), events)
		}
		parts := frs[0].Content.Parts
		if len(parts) != 2 {
			t.Fatalf("got %d parts in the function response event, want 2: %v", len(parts), parts)
		}
		if got := parts[0].FunctionResponse.Name; got != "p_show" {
			t.Errorf("FunctionResponse.Name = %q, want %q", got, "p_show")
		}
		if got := parts[1].Text; got != "final answer" {
			t.Errorf("parts[1].Text = %q, want %q", got, "final answer")
		}
	})

	t.Run("streaming tool is run as a stream", func(t *testing.T) {
		events, _ := runWithPrefixedTool(t, nil, &letterStreamTool{name: "stream"}, "p_stream")

		frs := functionResponseEvents(events)
		if len(frs) != 1 {
			t.Fatalf("got %d function response events, want 1: %v", len(frs), events)
		}
		fr := frs[0].Content.Parts[0].FunctionResponse
		if fr.Name != "p_stream" {
			t.Errorf("FunctionResponse.Name = %q, want %q", fr.Name, "p_stream")
		}
		if diff := cmp.Diff(map[string]any{"result": "ab"}, fr.Response); diff != "" {
			t.Errorf("FunctionResponse.Response mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("unprefixed name is not callable", func(t *testing.T) {
		d := &deferringTool{name: "deferring", defers: true}

		events, _ := runWithPrefixedTool(t, nil, d, "deferring")

		if d.runCount != 0 {
			t.Errorf("runCount = %d, want 0", d.runCount)
		}
		frs := functionResponseEvents(events)
		if len(frs) != 1 {
			t.Fatalf("got %d function response events, want 1: %v", len(frs), events)
		}
		msg, _ := frs[0].Content.Parts[0].FunctionResponse.Response["error"].(string)
		if !strings.HasPrefix(msg, "tool 'deferring' not found.") {
			t.Errorf("FunctionResponse error = %q, want one starting with %q", msg, "tool 'deferring' not found.")
		}
	})

	t.Run("same tool as own and prefixed is offered under both names", func(t *testing.T) {
		x := &deferringTool{name: "x", result: map[string]any{"r": "ok"}}

		events, requestTools := runWithPrefixedTool(t, []tool.Tool{x}, x, "p_x")

		if diff := cmp.Diff([]string{"p_x", "x"}, requestTools); diff != "" {
			t.Errorf("req.Tools names mismatch (-want +got):\n%s", diff)
		}
		if x.runCount != 1 {
			t.Errorf("runCount = %d, want 1", x.runCount)
		}
		if got := functionResponseEvents(events); len(got) != 1 {
			t.Errorf("got %d function response events, want 1: %v", len(got), events)
		}
	})
}
