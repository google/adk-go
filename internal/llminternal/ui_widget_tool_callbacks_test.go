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

package llminternal

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// Widgets rendered by a before-tool callback, the tool and an after-tool
// callback of one call all reach that call's function response event, in the
// order they were rendered, because the three share one tool context.
func TestHandleFunctionCalls_ToolCallbackWidgetsReachFunctionResponse(t *testing.T) {
	type noArgs struct{}
	widget := func(id string) session.UIWidget { return session.UIWidget{ID: id, Provider: "mcp"} }
	renderTool, err := functiontool.New(functiontool.Config{Name: "render_tool", Description: "renders"},
		func(ctx agent.Context, _ noArgs) (map[string]any, error) {
			return map[string]any{"ok": true}, agent.RenderUIWidget(ctx, widget("from_tool"))
		})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	flow := &Flow{
		BeforeToolCallbacks: []BeforeToolCallback{func(ctx agent.Context, _ tool.Tool, _ map[string]any) (map[string]any, error) {
			return nil, agent.RenderUIWidget(ctx, widget("from_before"))
		}},
		AfterToolCallbacks: []AfterToolCallback{func(ctx agent.Context, _ tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
			return nil, agent.RenderUIWidget(ctx, widget("from_after"))
		}},
	}
	resp := &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: "fc1", Name: "render_tool", Args: map[string]any{}}},
	}}}
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		InvocationID: "inv_1", Agent: &mockAgent{name: "agent_1"},
	})

	ev, err := flow.handleFunctionCalls(ic, map[string]tool.Tool{"render_tool": renderTool}, resp, nil, nil)
	if err != nil {
		t.Fatalf("handleFunctionCalls() error = %v", err)
	}

	want := []session.UIWidget{{ID: "from_before", Provider: "mcp"}, {ID: "from_tool", Provider: "mcp"}, {ID: "from_after", Provider: "mcp"}}
	if diff := cmp.Diff(want, ev.Actions.RenderUIWidgets); diff != "" {
		t.Errorf("function response RenderUIWidgets mismatch (-want +got):\n%s", diff)
	}
}
