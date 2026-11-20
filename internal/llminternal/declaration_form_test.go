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
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

// nullableNestedSchema is u4443's shape cut to one field: a nullable, titled
// nested record.
func nullableNestedSchema() *genai.Schema {
	return &genai.Schema{
		Type:  genai.TypeObject,
		Title: "Consolidated",
		Properties: map[string]*genai.Schema{
			"nested": {
				Type:        genai.TypeObject,
				Title:       "SubSchema",
				Nullable:    new(true),
				Description: "Nested model",
				Properties: map[string]*genai.Schema{
					"field1": {Type: genai.TypeString, Description: "Field 1"},
				},
			},
		},
	}
}

// nullableUnionsJSONSchema has one optional union with two non-null arms, one
// union without a null arm, and one optional string.
func nullableUnionsJSONSchema() *jsonschema.Schema {
	return &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"v": {AnyOf: []*jsonschema.Schema{{Type: "string"}, {Type: "integer"}, {Type: "null"}}},
			"u": {AnyOf: []*jsonschema.Schema{{Type: "string"}, {Type: "integer"}}},
			"w": {AnyOf: []*jsonschema.Schema{{Type: "string"}, {Type: "null"}}, Description: "W"},
		},
	}
}

// Python flattens anyOf [X, null] to X plus nullable: true only for Vertex AI
// (tools/_function_tool_declarations.py:351-420, applied at :573-574); the
// Gemini API gets the anyOf as it is.
// Fails when the rewrite is skipped on Vertex AI, applied on the Gemini API,
// drops the use-site description, or flattens a union it should leave alone.
func TestOutputSchemaRequestProcessor_NullableFollowsBackend(t *testing.T) {
	const geminiTyped = `{
		"type": "object",
		"title": "Consolidated",
		"properties": {
			"nested": {"anyOf": [{"$ref": "#/$defs/SubSchema"}, {"type": "null"}], "description": "Nested model"}
		},
		"$defs": {
			"SubSchema": {"type": "object", "title": "SubSchema", "properties": {"field1": {"type": "string", "description": "Field 1"}}}
		}
	}`
	const vertexTyped = `{
		"type": "object",
		"title": "Consolidated",
		"properties": {
			"nested": {"$ref": "#/$defs/SubSchema", "description": "Nested model", "nullable": true}
		},
		"$defs": {
			"SubSchema": {"type": "object", "title": "SubSchema", "properties": {"field1": {"type": "string", "description": "Field 1"}}}
		}
	}`
	const geminiRaw = `{
		"type": "object",
		"required": ["response"],
		"properties": {"response": {
			"type": "object",
			"properties": {
				"v": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]},
				"u": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
				"w": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "W"}
			}
		}}
	}`
	const vertexRaw = `{
		"type": "object",
		"required": ["response"],
		"properties": {"response": {
			"type": "object",
			"properties": {
				"v": {"anyOf": [{"type": "string"}, {"type": "integer"}], "nullable": true},
				"u": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
				"w": {"type": "string", "description": "W", "nullable": true}
			}
		}}
	}`
	for _, tc := range []struct {
		name    string
		backend genai.Backend
		state   func(*State)
		want    string
	}{
		{"typed schema on Gemini API", genai.BackendGeminiAPI, func(s *State) { s.OutputSchema = nullableNestedSchema() }, geminiTyped},
		{"typed schema on Vertex AI", genai.BackendVertexAI, func(s *State) { s.OutputSchema = nullableNestedSchema() }, vertexTyped},
		{"JSON schema on Gemini API", genai.BackendGeminiAPI, func(s *State) { s.OutputJsonSchema = nullableUnionsJSONSchema() }, geminiRaw},
		{"JSON schema on Vertex AI", genai.BackendVertexAI, func(s *State) { s.OutputJsonSchema = nullableUnionsJSONSchema() }, vertexRaw},
		// The arm's own keyword wins over the union's, as Python's setdefault
		// does (tools/_function_tool_declarations.py:387-390).
		{"arm keyword wins on Vertex AI", genai.BackendVertexAI, func(s *State) {
			s.OutputJsonSchema = &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"w": {AnyOf: []*jsonschema.Schema{{Type: "string", Description: "inner"}, {Type: "null"}}, Description: "outer"},
				},
			}
		}, `{
			"type": "object",
			"required": ["response"],
			"properties": {"response": {
				"type": "object",
				"properties": {"w": {"type": "string", "description": "inner", "nullable": true}}
			}}
		}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := tc.backend
			// Not a Gemini model name, so Vertex AI does not grant the
			// native output-schema-and-tools path and the tool is offered.
			state := &State{
				Model: &mockLLM{name: "test-model", variant: &backend},
				Tools: []tool.Tool{&mockTool{name: "other_tool"}},
			}
			tc.state(state)
			a := &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: "a"})), s: state}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a})
			req := &model.LLMRequest{}

			for _, err := range outputSchemaRequestProcessor(ctx, req, &Flow{}) {
				t.Fatalf("outputSchemaRequestProcessor() error = %v", err)
			}

			decl := findDeclaration(req, "set_model_response")
			if decl == nil {
				t.Fatalf("request has no set_model_response declaration; tools = %v", req.Tools)
			}
			if diff := cmp.Diff(jsonValue(t, tc.want), marshaledValue(t, decl.ParametersJsonSchema)); diff != "" {
				t.Errorf("ParametersJsonSchema mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// An agent's own JSON schema wins over a schema a workflow node binds for it,
// as its OutputSchema does, so the request never carries both.
// Fails when outputSchemaFor returns the bound schema for an agent with
// OutputJsonSchema.
func TestOutputSchemaFor_OutputJSONSchemaWinsOverBoundSchema(t *testing.T) {
	bound := &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"a": {Type: genai.TypeString}}}
	state := &State{
		Model:            &mockLLM{name: "test-model"},
		OutputJsonSchema: &jsonschema.Schema{Type: "object", AdditionalProperties: &jsonschema.Schema{Type: "integer"}},
	}
	a := &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: "self"})), s: state}
	ctx := icontext.NewInvocationContext(
		WithBoundOutputSchema(WithBoundMode(t.Context(), "self", state, ModeSingleTurn), "self", state, bound),
		icontext.InvocationContextParams{Agent: a},
	)
	req := &model.LLMRequest{}

	for _, err := range basicRequestProcessor(ctx, req, &Flow{}) {
		t.Fatalf("basicRequestProcessor() error = %v", err)
	}

	if req.Config.ResponseSchema != nil {
		t.Errorf("ResponseSchema = %v, want nil", req.Config.ResponseSchema)
	}
	want := jsonValue(t, `{"type": "object", "additionalProperties": {"type": "integer"}}`)
	if diff := cmp.Diff(want, marshaledValue(t, req.Config.ResponseJsonSchema)); diff != "" {
		t.Errorf("ResponseJsonSchema mismatch (-want +got):\n%s", diff)
	}
}

// Each request gets its own copy of the agent's JSON schema, $defs and extra
// keywords included, so a request callback that edits it does not change the
// next request.
// Fails when the request carries the agent's own pointer, or a copy that loses
// $defs or extra keywords.
func TestBasicRequestProcessor_OutputJSONSchemaIsCopiedPerRequest(t *testing.T) {
	state := &State{
		Model: &mockLLM{name: "test-model"},
		OutputJsonSchema: &jsonschema.Schema{
			Type:       "object",
			Properties: map[string]*jsonschema.Schema{"x": {Ref: "#/$defs/X"}},
			Defs:       map[string]*jsonschema.Schema{"X": {Type: "string"}},
			Extra:      map[string]any{"x-note": "kept"},
		},
	}
	a := &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: "a"})), s: state}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a})
	request := func() *model.LLMRequest {
		req := &model.LLMRequest{}
		for _, err := range basicRequestProcessor(ctx, req, &Flow{}) {
			t.Fatalf("basicRequestProcessor() error = %v", err)
		}
		return req
	}
	const want = `{
		"type": "object",
		"properties": {"x": {"$ref": "#/$defs/X"}},
		"$defs": {"X": {"type": "string"}},
		"x-note": "kept"
	}`

	first := request()
	if diff := cmp.Diff(jsonValue(t, want), marshaledValue(t, first.Config.ResponseJsonSchema)); diff != "" {
		t.Fatalf("first ResponseJsonSchema mismatch (-want +got):\n%s", diff)
	}
	edited, ok := first.Config.ResponseJsonSchema.(*jsonschema.Schema)
	if !ok {
		t.Fatalf("ResponseJsonSchema is %T, want *jsonschema.Schema", first.Config.ResponseJsonSchema)
	}
	edited.Defs["X"].Type = "integer"
	edited.Extra["x-note"] = "edited"
	second := request()

	if diff := cmp.Diff(jsonValue(t, want), marshaledValue(t, second.Config.ResponseJsonSchema)); diff != "" {
		t.Errorf("second ResponseJsonSchema changed after editing the first (-want +got):\n%s", diff)
	}
}

func findDeclaration(req *model.LLMRequest, name string) *genai.FunctionDeclaration {
	if req.Config == nil {
		return nil
	}
	for _, gt := range req.Config.Tools {
		if gt == nil {
			continue
		}
		for _, d := range gt.FunctionDeclarations {
			if d != nil && d.Name == name {
				return d
			}
		}
	}
	return nil
}

func jsonValue(t *testing.T, s string) any {
	t.Helper()
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		t.Fatalf("json.Unmarshal(%s): %v", s, err)
	}
	return v
}

func marshaledValue(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal(%T): %v", v, err)
	}
	return jsonValue(t, string(b))
}
