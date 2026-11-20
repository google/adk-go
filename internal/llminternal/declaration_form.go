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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"reflect"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/exp/feature"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/setmodelresponsetool"
)

// declarationFor returns the set_model_response tool t in the form this
// request declares it. ctx is the request's own, read on every call, so runs
// with different values of the switch can share one agent:
//   - with feature.LegacyFunctionDeclarationSchema, the typed Parameters form;
//   - otherwise, on Vertex AI (llm's backend), the JSON form with each
//     anyOf [X, {"type": "null"}] flattened to X plus "nullable": true, as
//     adk-python does because Vertex AI rejects a sub-schema without a type
//     (tools/_function_tool_declarations.py:351-420, applied at :573-574);
//   - otherwise t as it is.
func declarationFor(ctx context.Context, llm model.LLM, t tool.Tool) (toolinternal.FunctionTool, error) {
	if feature.LegacyFunctionDeclarationSchema(ctx) {
		return asFunctionTool(setmodelresponsetool.AsLegacy(t))
	}
	ft, err := asFunctionTool(t)
	if err != nil || googlellm.GetGoogleLLMVariant(llm) != genai.BackendVertexAI {
		return ft, err
	}
	decl := ft.Declaration()
	params, ok := decl.ParametersJsonSchema.(*jsonschema.Schema)
	if !ok {
		// A typed *genai.Schema already says nullable: true.
		return ft, nil
	}
	flat, err := flattenNullableForVertex(params)
	if err != nil {
		return nil, fmt.Errorf("set_model_response declaration for Vertex AI: %w", err)
	}
	d := *decl
	d.ParametersJsonSchema = flat
	return &declaredTool{FunctionTool: ft, decl: &d}, nil
}

// asFunctionTool returns t as the tool the flow packs and runs.
func asFunctionTool(t tool.Tool) (toolinternal.FunctionTool, error) {
	ft, ok := t.(toolinternal.FunctionTool)
	if !ok {
		return nil, fmt.Errorf("set_model_response tool %T is not a function tool", t)
	}
	return ft, nil
}

// declaredTool is a function tool whose declaration was built for one request.
// It is immutable, so concurrent calls are safe.
type declaredTool struct {
	toolinternal.FunctionTool
	decl *genai.FunctionDeclaration
}

func (t *declaredTool) Declaration() *genai.FunctionDeclaration { return t.decl }

// flattenNullableForVertex returns a copy of s in which every sub-schema of
// the form anyOf [X, {"type": "null"}] becomes X plus "nullable": true. The
// sub-schema's other keywords (description, default) are kept unless X sets
// the same keyword. A union with more than one non-null arm keeps its anyOf
// without the null arm and gets "nullable": true; a union without exactly one
// null arm is left alone. s is not modified.
//
// The rewrite works on the JSON value, as adk-python's does, so every keyword
// of X survives without listing the fields of jsonschema.Schema here.
func flattenNullableForVertex(s *jsonschema.Schema) (*jsonschema.Schema, error) {
	b, err := json.Marshal(s)
	if err != nil {
		return nil, fmt.Errorf("marshal parameters: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, fmt.Errorf("decode parameters: %w", err)
	}
	b, err = json.Marshal(sanitizeForVertex(v))
	if err != nil {
		return nil, fmt.Errorf("marshal flattened parameters: %w", err)
	}
	out := &jsonschema.Schema{}
	if err := json.Unmarshal(b, out); err != nil {
		return nil, fmt.Errorf("unmarshal flattened parameters: %w", err)
	}
	return out, nil
}

// sanitizeForVertex ports _sanitize_json_schema_for_vertex: children first,
// then the value itself.
func sanitizeForVertex(v any) any {
	switch v := v.(type) {
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = sanitizeForVertex(item)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(v))
		for k, item := range v {
			out[k] = sanitizeForVertex(item)
		}
		return flattenOptionalAnyOf(out)
	default:
		return v
	}
}

// flattenOptionalAnyOf ports _flatten_optional_any_of for one schema whose
// children are already rewritten.
func flattenOptionalAnyOf(s map[string]any) map[string]any {
	anyOf, ok := s["anyOf"].([]any)
	if !ok || len(anyOf) < 2 {
		return s
	}
	var nulls []any
	for _, arm := range anyOf {
		if m, ok := arm.(map[string]any); ok && m["type"] == "null" {
			nulls = append(nulls, arm)
		}
	}
	if len(nulls) != 1 {
		return s
	}
	var rest []any
	for _, arm := range anyOf {
		if !reflect.DeepEqual(arm, nulls[0]) {
			rest = append(rest, arm)
		}
	}
	if len(rest) == 1 {
		if x, ok := rest[0].(map[string]any); ok {
			flat := maps.Clone(x)
			for k, val := range s {
				if _, set := flat[k]; !set && k != "anyOf" {
					flat[k] = val
				}
			}
			flat["nullable"] = true
			return flat
		}
	}
	flat := maps.Clone(s)
	flat["anyOf"] = rest
	flat["nullable"] = true
	return flat
}
