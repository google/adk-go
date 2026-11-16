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
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// parallelGuard bounds every wait in these tests that must end.
const parallelGuard = 5 * time.Second

// funcModel answers every request with gen.
type funcModel func(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error)

func (funcModel) Name() string { return "func-model" }

func (m funcModel) GenerateContent(ctx context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(m(ctx, req))
	}
}

func textResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleModel), FinishReason: genai.FinishReasonStop}
}

// callsThenDone answers with calls until the request ends with function
// responses, and then with the text "done".
func callsThenDone(calls ...*genai.FunctionCall) funcModel {
	return func(_ context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
		last := req.Contents[len(req.Contents)-1]
		for _, p := range last.Parts {
			if p.FunctionResponse != nil {
				return textResponse("done"), nil
			}
		}
		content := &genai.Content{Role: genai.RoleModel}
		for _, c := range calls {
			content.Parts = append(content.Parts, &genai.Part{FunctionCall: c})
		}
		return &model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil
	}
}

func newFunctionTool[Args any](t *testing.T, name string, handler functiontool.Func[Args, map[string]any]) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: name, Description: name}, handler)
	if err != nil {
		t.Fatalf("functiontool.New(%s) error = %v", name, err)
	}
	return tl
}

// runOnce runs a over a new in-memory session and returns the function
// responses of the run by tool name. It returns errors rather than failing the
// test, because tools call it on the flow's goroutines.
func runOnce(ctx context.Context, a agent.Agent) (map[string]map[string]any, error) {
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc})
	if err != nil {
		return nil, err
	}
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		return nil, err
	}
	got := map[string]map[string]any{}
	for ev, err := range r.Run(ctx, "user", created.Session.ID(), genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			return nil, err
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
	return got, nil
}

// TestRun_NestedRunIsNotOrderedByCallingBatch pins that a run started from a
// tool call's context takes no part in the calling batch's order: the nested
// agent's before-model callback uses State while an earlier call of the outer
// batch still runs, and does not wait for it.
func TestRun_NestedRunIsNotOrderedByCallingBatch(t *testing.T) {
	ctx := t.Context()
	nestedModelCalled := make(chan struct{})
	var closeOnce sync.Once

	holder := newFunctionTool(t, "holder", func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		if err := ctx.State().Set("x", 1); err != nil {
			return nil, err
		}
		select {
		case <-nestedModelCalled:
			return map[string]any{"released": true}, nil
		case <-time.After(parallelGuard):
			return map[string]any{"released": false}, nil
		}
	})
	nested := newFunctionTool(t, "nested", func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		inner, err := llmagent.New(llmagent.Config{
			Name: "inner",
			Model: funcModel(func(context.Context, *model.LLMRequest) (*model.LLMResponse, error) {
				closeOnce.Do(func() { close(nestedModelCalled) })
				return textResponse("inner done"), nil
			}),
			BeforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
					if _, err := ctx.State().Get("k"); err != nil && !errors.Is(err, session.ErrStateKeyNotExist) {
						return nil, err
					}
					return nil, nil
				},
			},
		})
		if err != nil {
			return nil, err
		}
		if _, err := runOnce(ctx, inner); err != nil {
			return nil, err
		}
		return map[string]any{"ran": true}, nil
	})
	root, err := llmagent.New(llmagent.Config{
		Name: "root",
		Model: callsThenDone(
			&genai.FunctionCall{ID: "call-0", Name: "holder", Args: map[string]any{}},
			&genai.FunctionCall{ID: "call-1", Name: "nested", Args: map[string]any{}},
		),
		Tools: []tool.Tool{holder, nested},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	got, err := runOnce(ctx, root)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	want := map[string]map[string]any{
		"holder": {"released": true},
		"nested": {"ran": true},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
}
