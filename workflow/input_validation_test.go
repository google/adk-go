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

package workflow

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
)

// echoMapNode returns a function node that outputs the input it receives.
func echoMapNode(t *testing.T, name string, inputSchema *jsonschema.Schema, cfg NodeConfig) *FunctionNode {
	t.Helper()
	n, err := NewFunctionNodeWithSchema(name,
		func(_ agent.Context, in map[string]any) (map[string]any, error) { return in, nil },
		inputSchema, nil, cfg)
	if err != nil {
		t.Fatalf("NewFunctionNodeWithSchema: %v", err)
	}
	return n
}

func TestValidateInput_FillsDefaultsOnStartSuccessor(t *testing.T) {
	schema := &jsonschema.Schema{
		Type:     "object",
		Required: []string{"query"},
		Properties: map[string]*jsonschema.Schema{
			"query": {Type: "string"},
			"limit": {Type: "integer", Default: []byte(`10`)},
		},
	}
	want := map[string]any{"query": "books", "limit": float64(10)}

	t.Run("user_message", func(t *testing.T) {
		node := echoMapNode(t, "search", schema, NodeConfig{})
		w := mustNew(t, []Edge{{From: Start, To: node}})
		ctx := newMockCtx(t)
		ctx.userContent = &genai.Content{Parts: []*genai.Part{{Text: `{"query":"books"}`}}}

		outputs := outputsByNode(drain(t, w.Run(ctx)))

		if diff := cmp.Diff(want, outputs["search"]); diff != "" {
			t.Errorf("search output mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("start_seed", func(t *testing.T) {
		// A nested workflow that follows Start receives the seed itself.
		resolved, err := schema.Resolve(nil)
		if err != nil {
			t.Fatalf("Resolve: %v", err)
		}
		seed := newStartSeed(&genai.Content{Parts: []*genai.Part{{Text: `{"query":"books"}`}}})

		got, err := defaultValidateInput(seed, resolved)
		if err != nil {
			t.Fatalf("defaultValidateInput: %v", err)
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("defaultValidateInput mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestValidateInput_FillsDefaultsOnPredecessorMap(t *testing.T) {
	schema := &jsonschema.Schema{
		Type:     "object",
		Required: []string{"query"},
		Properties: map[string]*jsonschema.Schema{
			"query": {Type: "string"},
			"limit": {Type: "integer", Default: []byte(`10`)},
		},
	}
	producer := newStubNode("producer", map[string]any{"query": "books"})
	consumer := echoMapNode(t, "consumer", schema, NodeConfig{})
	w := mustNew(t, Chain(Start, producer, consumer))

	outputs := outputsByNode(drain(t, w.Run(newMockCtx(t))))

	want := map[string]any{"query": "books", "limit": float64(10)}
	if diff := cmp.Diff(want, outputs["consumer"]); diff != "" {
		t.Errorf("consumer output mismatch (-want +got):\n%s", diff)
	}
}

// A nil map from the predecessor is an absent object, so the input
// schema's defaults fill it.
func TestValidateInput_NilMapGetsObjectDefaults(t *testing.T) {
	schema := &jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"a": {Type: "integer", Default: []byte(`2`)}},
	}
	producer := newStubNode("producer", map[string]any(nil))
	consumer := echoMapNode(t, "consumer", schema, NodeConfig{})
	w := mustNew(t, Chain(Start, producer, consumer))

	outputs := outputsByNode(drain(t, w.Run(newMockCtx(t))))

	if diff := cmp.Diff(map[string]any{"a": float64(2)}, outputs["consumer"]); diff != "" {
		t.Errorf("consumer output mismatch (-want +got):\n%s", diff)
	}
}

func TestValidateInput_UnwrapsContent(t *testing.T) {
	schema := &jsonschema.Schema{
		Type:       "object",
		Required:   []string{"x"},
		Properties: map[string]*jsonschema.Schema{"x": {Type: "integer"}},
	}
	want := map[string]any{"x": float64(1)}

	t.Run("content_input_start_successor", func(t *testing.T) {
		node := echoMapNode(t, "consumer", schema, NodeConfig{ContentInput: true})
		w := mustNew(t, []Edge{{From: Start, To: node}})
		ctx := newMockCtx(t)
		ctx.userContent = &genai.Content{Role: "user", Parts: []*genai.Part{{Text: `{"x": 1}`}}}

		outputs := outputsByNode(drain(t, w.Run(ctx)))

		if diff := cmp.Diff(want, outputs["consumer"]); diff != "" {
			t.Errorf("consumer output mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("content_predecessor_output", func(t *testing.T) {
		producer := newStubNode("producer", &genai.Content{Role: "model", Parts: []*genai.Part{{Text: `{"x": 1}`}}})
		consumer := echoMapNode(t, "consumer", schema, NodeConfig{})
		w := mustNew(t, Chain(Start, producer, consumer))

		outputs := outputsByNode(drain(t, w.Run(newMockCtx(t))))

		if diff := cmp.Diff(want, outputs["consumer"]); diff != "" {
			t.Errorf("consumer output mismatch (-want +got):\n%s", diff)
		}
	})

	resolved, err := schema.Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	direct := []struct {
		name string
		in   any
	}{
		{name: "content_pointer", in: &genai.Content{Parts: []*genai.Part{{Text: `{"x": 1}`}}}},
		{name: "content_value", in: genai.Content{Parts: []*genai.Part{{Text: `{"x": 1}`}}}},
	}
	for _, tc := range direct {
		t.Run(tc.name, func(t *testing.T) {
			got, err := defaultValidateInput(tc.in, resolved)
			if err != nil {
				t.Fatalf("defaultValidateInput: %v", err)
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("defaultValidateInput mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
