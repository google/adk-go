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

package generatortool_test

import (
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/generatortool"
	"google.golang.org/adk/v2/workflow"
)

// A21. Fails if the response schema is inferred from the handler's type (so
// event types leak into it), is taken from an inner tool with an any result,
// or ignores the caller's OutputSchema.
func TestNew_ResponseSchema(t *testing.T) {
	type rows struct {
		Rows int `json:"rows"`
	}
	tests := []struct {
		name string
		tool tool.Tool
		want string
	}{
		{
			name: "map result",
			tool: mustNew(t, functiontool.Config{Name: "t"}, func(agent.Context, queryArgs, func(*session.Event) error) (map[string]any, error) {
				return nil, nil
			}),
			want: `{"type":"object","additionalProperties":true}`,
		},
		{
			name: "struct result",
			tool: mustNew(t, functiontool.Config{Name: "t"}, func(agent.Context, queryArgs, func(*session.Event) error) (rows, error) {
				return rows{}, nil
			}),
			want: `{"type":"object","properties":{"rows":{"type":"integer"}},"required":["rows"],"additionalProperties":false}`,
		},
		{
			name: "any result",
			tool: mustNew(t, functiontool.Config{Name: "t"}, func(agent.Context, queryArgs, func(*session.Event) error) (any, error) {
				return nil, nil
			}),
			want: `true`,
		},
		{
			name: "caller's output schema",
			tool: mustNew(t, functiontool.Config{Name: "t", OutputSchema: &jsonschema.Schema{Type: "string"}}, func(agent.Context, queryArgs, func(*session.Event) error) (any, error) {
				return nil, nil
			}),
			want: `{"type":"string"}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			decl, ok := tc.tool.(interface {
				Declaration() *genai.FunctionDeclaration
			})
			if !ok {
				t.Fatalf("tool %T has no Declaration method", tc.tool)
			}
			got, err := json.Marshal(decl.Declaration().ResponseJsonSchema)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if string(got) != tc.want {
				t.Errorf("ResponseJsonSchema = %s, want %s", got, tc.want)
			}
		})
	}
}

// A22. Fails if New accepts an invalid input: functiontool.New reads neither
// the name nor the handler, so New must check both itself.
func TestNew_RejectsInvalid(t *testing.T) {
	handler := func(agent.Context, queryArgs, func(*session.Event) error) (any, error) { return nil, nil }
	tests := []struct {
		name  string
		build func() (tool.Tool, error)
	}{
		{name: "args not a struct or map", build: func() (tool.Tool, error) {
			return generatortool.New(functiontool.Config{Name: "t"}, func(agent.Context, int, func(*session.Event) error) (any, error) { return nil, nil })
		}},
		{name: "nil handler", build: func() (tool.Tool, error) {
			return generatortool.New[queryArgs, any](functiontool.Config{Name: "t"}, nil)
		}},
		{name: "empty name", build: func() (tool.Tool, error) {
			return generatortool.New(functiontool.Config{Name: ""}, handler)
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.build()
			if got != nil || !errors.Is(err, functiontool.ErrInvalidArgument) {
				t.Errorf("New() = (%v, %v), want (nil, an error wrapping functiontool.ErrInvalidArgument)", got, err)
			}
		})
	}
}

// A37. Fails if New drops a Config field that its doc says applies to the tool
// unchanged: InputSchema, IsLongRunning or RequireConfirmationProvider.
func TestNew_PassesConfigThrough(t *testing.T) {
	handler := func(agent.Context, queryArgs, func(*session.Event) error) (any, error) { return nil, nil }

	t.Run("IsLongRunning", func(t *testing.T) {
		tl := mustNew(t, functiontool.Config{Name: "t", IsLongRunning: true}, handler)
		if !tl.IsLongRunning() {
			t.Error("IsLongRunning() = false, want true")
		}
	})

	t.Run("InputSchema", func(t *testing.T) {
		schema := &jsonschema.Schema{
			Type:       "object",
			Properties: map[string]*jsonschema.Schema{"query": {Type: "string", Description: "the sales query"}},
			Required:   []string{"query"},
		}
		tl := mustNew(t, functiontool.Config{Name: "t", InputSchema: schema}, handler)
		decl, ok := tl.(interface {
			Declaration() *genai.FunctionDeclaration
		})
		if !ok {
			t.Fatalf("tool %T has no Declaration method", tl)
		}
		got, err := json.Marshal(decl.Declaration().ParametersJsonSchema)
		if err != nil {
			t.Fatalf("json.Marshal() error = %v", err)
		}
		want := `{"type":"object","properties":{"query":{"type":"string","description":"the sales query"}},"required":["query"]}`
		if string(got) != want {
			t.Errorf("ParametersJsonSchema = %s, want %s", got, want)
		}
	})

	t.Run("RequireConfirmationProvider", func(t *testing.T) {
		tests := []struct {
			query    string
			wantRuns int32
		}{
			{query: "sales", wantRuns: 0},
			{query: "other", wantRuns: 1},
		}
		for _, tc := range tests {
			t.Run(tc.query, func(t *testing.T) {
				var runs atomic.Int32
				tl := mustNew(t, functiontool.Config{
					Name:                        "analyze",
					RequireConfirmationProvider: func(a queryArgs) bool { return a.Query == "sales" },
				}, func(agent.Context, queryArgs, func(*session.Event) error) (any, error) {
					runs.Add(1)
					return "ok", nil
				})
				h := newHarness(t, harnessConfig{tools: []tool.Tool{tl}, responses: []*genai.Content{
					calls(call("call_1", "analyze", map[string]any{"query": tc.query})), modelText("done"),
				}})

				h.mustRun(userText("go"))

				if got := runs.Load(); got != tc.wantRuns {
					t.Errorf("handler runs = %d, want %d", got, tc.wantRuns)
				}
			})
		}
	})
}

// A23. Fails if the handler runs without a workflow node, or if the error does
// not say why.
func TestRun_NoNodeContextFails(t *testing.T) {
	var runs atomic.Int32
	emitted := false
	tl := mustNew(t, functiontool.Config{Name: "analyze"}, func(_ agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		runs.Add(1)
		emitted = emit(message("m")) == nil
		return "ok", nil
	})
	runner, ok := tl.(interface {
		Run(ctx agent.Context, args any) (map[string]any, error)
	})
	if !ok {
		t.Fatalf("tool %T has no Run method", tl)
	}
	ctx := agent.NewToolContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}), "call_1", nil, nil)

	_, err := runner.Run(ctx, map[string]any{"query": "sales"})

	if !errors.Is(err, workflow.ErrInvalidRunNodeContext) {
		t.Errorf("Run() error = %v, want an error wrapping workflow.ErrInvalidRunNodeContext", err)
	}
	if got := runs.Load(); got != 0 || emitted {
		t.Errorf("handler runs = %d, emitted = %v, want 0 runs and nothing emitted", got, emitted)
	}
}
