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
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// valueTool is a tool type with a value receiver, so req.Tools holds a struct
// value and not a pointer.
type valueTool struct {
	decl *genai.FunctionDeclaration
}

func (v valueTool) Declaration() *genai.FunctionDeclaration { return v.decl }

func TestPreprocess_ValueTypedEntriesFollowTheirDeclaration(t *testing.T) {
	seed := func(_ agent.InvocationContext, req *model.LLMRequest, _ *Flow) iter.Seq2[*session.Event, error] {
		req.Tools = map[string]any{
			"value_declared":   valueTool{decl: &genai.FunctionDeclaration{Name: "value_declared"}},
			"value_undeclared": valueTool{},
		}
		return func(func(*session.Event, error) bool) {}
	}
	f := &Flow{RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{seed}}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockLLMAgent{s: &State{}}})
	req := &model.LLMRequest{}

	for _, err := range f.preprocess(ctx, req) {
		if err != nil {
			t.Fatalf("preprocess yielded error: %v", err)
		}
	}

	if diff := cmp.Diff([]string{"value_declared"}, record(req).toolKeys); diff != "" {
		t.Errorf("req.Tools keys mismatch (-want +got):\n%s", diff)
	}
}
