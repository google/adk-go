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

// derefTool's Declaration reads a field of its receiver, so calling it on a
// nil *derefTool panics.
type derefTool struct {
	decl *genai.FunctionDeclaration
}

func (d *derefTool) Declaration() *genai.FunctionDeclaration { return d.decl }

func TestPreprocess_KeepsNilPointerEntry(t *testing.T) {
	seed := func(_ agent.InvocationContext, req *model.LLMRequest, _ *Flow) iter.Seq2[*session.Event, error] {
		req.Tools = map[string]any{"nil_ptr": (*derefTool)(nil)}
		return func(func(*session.Event, error) bool) {}
	}
	f := &Flow{RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{seed}}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockLLMAgent{s: &State{}}})
	req := &model.LLMRequest{}

	// A panic would end the whole test binary; report it as this test's failure.
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("preprocess panicked: %v", r)
			}
		}()
		for _, err := range f.preprocess(ctx, req) {
			if err != nil {
				t.Fatalf("preprocess yielded error: %v", err)
			}
		}
	}()

	if diff := cmp.Diff([]string{"nil_ptr"}, recordDeclRequest(req).toolKeys); diff != "" {
		t.Errorf("req.Tools keys mismatch (-want +got):\n%s", diff)
	}
}
