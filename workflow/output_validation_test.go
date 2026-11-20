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
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
)

func TestFunctionNode_TypedStructOutputAppliesDefaults(t *testing.T) {
	schema := mustSchema[ioschemaReview](t)
	schema.Properties["reviewer"].Default = []byte(`"auto"`)
	producer, err := NewFunctionNodeWithSchema("producer",
		func(_ agent.Context, _ string) (ioschemaReview, error) {
			return ioschemaReview{Result: "ok", Score: 90}, nil
		}, nil, schema, NodeConfig{})
	if err != nil {
		t.Fatalf("producer: %v", err)
	}
	consumer := NewFunctionNode("consumer",
		func(_ agent.Context, in ioschemaReview) (string, error) {
			return fmt.Sprintf("%s:%d:%s", in.Result, in.Score, in.Reviewer), nil
		}, NodeConfig{})
	w := mustNew(t, Chain(Start, producer, consumer))

	outputs := outputsByNode(drain(t, w.Run(newMockCtx(t))))

	wantProducer := map[string]any{"result": "ok", "score": float64(90), "reviewer": "auto"}
	if diff := cmp.Diff(wantProducer, outputs["producer"]); diff != "" {
		t.Errorf("producer output mismatch (-want +got):\n%s", diff)
	}
	if got := outputs["consumer"]; got != "ok:90:auto" {
		t.Errorf("consumer output = %v, want %q", got, "ok:90:auto")
	}
}

func TestValidateOutput_StringProjectionByNodeKind(t *testing.T) {
	schema := resolveSchemaJSON(t, `{
		"type": "object",
		"required": ["name"],
		"properties": {
			"name": {"type": "string"},
			"role": {"type": "string", "default": "member"}
		}
	}`)
	agentNode := &AgentNode{BaseNode: NewBaseNodeWithSchemas("agent", "", NodeConfig{}, nil, schema)}
	baseNode := NewBaseNodeWithSchemas("base", "", NodeConfig{}, nil, schema)
	toolNode := &ToolNode{BaseNode: NewBaseNodeWithSchemas("tool", "", NodeConfig{}, nil, schema)}
	const jsonText = `{"name":"Alice"}`
	wantMap := map[string]any{"name": "Alice", "role": "member"}

	t.Run("agent_node_parses_model_text", func(t *testing.T) {
		got, err := agentNode.ValidateOutput(jsonText)
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		if diff := cmp.Diff(wantMap, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("base_node_unwraps_content", func(t *testing.T) {
		got, err := baseNode.ValidateOutput(&genai.Content{Parts: []*genai.Part{{Text: jsonText}}})
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		if diff := cmp.Diff(wantMap, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})

	rejected := []struct {
		name string
		node interface{ ValidateOutput(any) (any, error) }
		out  any
	}{
		{name: "base_node_bare_json_string", node: baseNode, out: jsonText},
		{name: "tool_node_bare_json_string", node: toolNode, out: jsonText},
		{name: "tool_node_result_wrapped_json_string", node: toolNode, out: map[string]any{"result": jsonText}},
	}
	for _, tc := range rejected {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.node.ValidateOutput(tc.out)
			if err == nil {
				t.Fatalf("ValidateOutput = %v, want a schema error", got)
			}
			if got != nil {
				t.Errorf("ValidateOutput value = %v, want nil on error", got)
			}
		})
	}

	intSchema := &jsonschema.Schema{Type: "integer"}
	boolSchema := &jsonschema.Schema{Type: "boolean"}
	coercions := []struct {
		name   string
		schema *jsonschema.Schema
		out    string
		want   any
	}{
		{name: "numeric_string_to_integer", schema: intSchema, out: "42", want: float64(42)},
		{name: "boolean_string_to_boolean", schema: boolSchema, out: "true", want: true},
	}
	for _, tc := range coercions {
		t.Run(tc.name, func(t *testing.T) {
			resolved, err := tc.schema.Resolve(nil)
			if err != nil {
				t.Fatalf("Resolve: %v", err)
			}
			got, err := NewBaseNodeWithSchemas("base", "", NodeConfig{}, nil, resolved).ValidateOutput(tc.out)
			if err != nil {
				t.Fatalf("ValidateOutput: %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
			}
		})
	}

	resolvedInt, err := intSchema.Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	intNode := NewBaseNodeWithSchemas("base", "", NodeConfig{}, nil, resolvedInt)
	for _, out := range []string{"[1]", "null"} {
		t.Run("non_scalar_string_against_integer_"+out, func(t *testing.T) {
			if got, err := intNode.ValidateOutput(out); err == nil {
				t.Fatalf("ValidateOutput(%q) = %v, want a schema error", out, got)
			}
		})
	}
}

func TestValidateOutput_ContentText(t *testing.T) {
	text := func(s string) *genai.Content { return &genai.Content{Parts: []*genai.Part{{Text: s}}} }
	intOrString := resolveSchemaJSON(t, `{"anyOf": [{"type": "integer"}, {"type": "string"}]}`)

	accepted := []struct {
		name   string
		schema *jsonschema.Resolved
		out    any
		want   any
	}{
		{name: "non_json_text_validated_as_string", schema: intOrString, out: text("hello"), want: "hello"},
		{name: "content_value_unwrapped", schema: resolveSchemaJSON(t, `{"type": "string"}`), out: genai.Content{Parts: []*genai.Part{{Text: "hi"}}}, want: "hi"},
		{name: "json_text_validated_as_parsed_value", schema: intOrString, out: text("42"), want: float64(42)},
		{name: "string_or_null_schema_returns_text", schema: resolveSchemaJSON(t, `{"type": ["string", "null"]}`), out: text("plain"), want: "plain"},
	}
	for _, tc := range accepted {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, tc.schema).ValidateOutput(tc.out)
			if err != nil {
				t.Fatalf("ValidateOutput: %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("empty_text_fails_object_schema", func(t *testing.T) {
		schema := resolveSchemaJSON(t, `{"type": "object", "required": ["name"]}`)
		got, err := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema).ValidateOutput(text(""))
		if err == nil {
			t.Fatalf("ValidateOutput = %v, want a schema error", got)
		}
	})

	t.Run("empty_schema_keeps_content", func(t *testing.T) {
		empty, err := (&jsonschema.Schema{}).Resolve(nil)
		if err != nil {
			t.Fatalf("Resolve: %v", err)
		}
		content := text("hello")
		got, err := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, empty).ValidateOutput(content)
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		if got != any(content) {
			t.Errorf("ValidateOutput = %#v, want the same *genai.Content", got)
		}
	})
}

// A Content whose text fails an object schema is rejected even when the
// Content's own JSON form ({"role": ..., "parts": [...]}) satisfies it, as
// it does an open object schema with optional properties; adk-python keeps
// a Content only for an annotation that accepts Content.
func TestValidateContent_OpenObjectSchemaRejectsProse(t *testing.T) {
	schema := resolveSchemaJSON(t, `{"type": "object", "properties": {"a": {"type": "string"}}}`)
	node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, schema, schema)
	prose := &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "hello"}}}

	t.Run("output", func(t *testing.T) {
		if got, err := node.ValidateOutput(prose); err == nil {
			t.Fatalf("ValidateOutput = %#v, want a schema error", got)
		}
	})
	t.Run("input", func(t *testing.T) {
		if got, err := node.ValidateInput(prose); err == nil {
			t.Fatalf("ValidateInput = %#v, want a schema error", got)
		}
	})
}
