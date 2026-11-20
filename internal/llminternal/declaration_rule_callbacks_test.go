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
	"context"
	"fmt"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// callsAllModel answers its first request with one function call per name and
// every later request with the text "done".
type callsAllModel struct {
	names []string
	calls int
}

func (m *callsAllModel) Name() string { return "calls-all" }

func (m *callsAllModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		if m.calls > 1 {
			yield(&model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel)}, nil)
			return
		}
		var parts []*genai.Part
		for i, n := range m.names {
			parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{ID: fmt.Sprintf("fc-%d", i), Name: n, Args: map[string]any{}}})
		}
		yield(&model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}}, nil)
	}
}

// runDeclaredBesideCallbackTool runs a flow whose model calls eight declared
// tools while a before-model callback registers an undeclared one, and
// returns the declared tools.
func runDeclaredBesideCallbackTool(t *testing.T) []*packingTool {
	t.Helper()
	declared := make([]*packingTool, 8)
	tools := make([]tool.Tool, len(declared))
	names := make([]string, len(declared))
	for i := range declared {
		names[i] = fmt.Sprintf("decl_%d", i)
		declared[i] = &packingTool{name: names[i], decl: &genai.FunctionDeclaration{Name: names[i]}}
		tools[i] = declared[i]
	}
	cbTool := &packingTool{name: "cb_tool"}
	f := &Flow{
		Model: &callsAllModel{names: names},
		Tools: tools,
		BeforeModelCallbacks: []BeforeModelCallback{
			func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
				return nil, toolutils.PackTool(req, cbTool)
			},
		},
	}
	runToFirstFunctionResponse(t, f)
	return declared
}

// An undeclared tool that a before-model callback registers does not hide the
// declared tools from the callable map, whatever order the map yields them in.
func TestRun_UndeclaredToolFromCallbackDoesNotHideDeclaredTools(t *testing.T) {
	// Go's map order decides whether the undeclared entry comes before a
	// declared one; it comes last about once in nine runs. Thirty runs cover
	// the other orders almost surely.
	for round := range 30 {
		for _, d := range runDeclaredBesideCallbackTool(t) {
			if got := d.runs.Load(); got != 1 {
				t.Fatalf("round %d: %s Run called %d times, want 1", round, d.name, got)
			}
		}
	}
}
