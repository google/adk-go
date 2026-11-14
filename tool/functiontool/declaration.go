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

package functiontool

import (
	"context"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/exp/feature"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/internal/typeutil"
)

// declaredFunctionTool is a function tool whose declaration was built for one
// request. Everything but Declaration is the wrapped tool's.
type declaredFunctionTool[TArgs, TResults any] struct {
	*functionTool[TArgs, TResults]
	decl *genai.FunctionDeclaration
}

func (t *declaredFunctionTool[TArgs, TResults]) Declaration() *genai.FunctionDeclaration {
	return t.decl
}

// declaredStreamingTool is the streaming counterpart of declaredFunctionTool.
type declaredStreamingTool[TArgs any] struct {
	*streamingFunctionTool[TArgs]
	decl *genai.FunctionDeclaration
}

func (t *declaredStreamingTool[TArgs]) Declaration() *genai.FunctionDeclaration {
	return t.decl
}

// requestDeclaration returns the declaration of a function tool for the
// request whose context is ctx, given its base declaration and its input and
// output schemas (out is nil for a streaming tool). It returns base itself
// when the request needs no change, and otherwise a new declaration copied
// from base; base, in and out are never modified. The rules follow adk-python
// for the typed-declaration switch (feature.LegacyFunctionDeclarationSchema)
// and the request model's backend (googlellm.BackendFromContext):
//   - switch on: the typed form (Parameters, and Response on Vertex AI only),
//     without JSON-form fields, as
//     tools/_automatic_function_calling_util.py:421-448 builds it. A function
//     without parameters has no parameter field. A parameter schema the typed
//     form cannot hold keeps its JSON form, so no parameter is dropped.
//   - switch off, Vertex AI: the JSON form with every anyOf [X, null]
//     flattened (tools/_function_tool_declarations.py:572-581), and no
//     parameter field for a function without parameters (:436-438).
//   - otherwise, base. adk-python also removes the response schema for the
//     Gemini API with the switch off
//     (_automatic_function_calling_util.py:232-236); adk-go keeps the request
//     its recorded tests replay.
//
// The error is that of typeutil.FlattenNullableForVertex.
func requestDeclaration(ctx context.Context, base *genai.FunctionDeclaration, in, out *jsonschema.Schema) (*genai.FunctionDeclaration, error) {
	if ctx == nil {
		return base, nil
	}
	vertex := googlellm.BackendFromContext(ctx) == genai.BackendVertexAI
	if feature.LegacyFunctionDeclarationSchema(ctx) {
		return typedDeclaration(base, in, out, vertex)
	}
	if !vertex {
		return base, nil
	}
	d := *base
	if base.ParametersJsonSchema != nil {
		if noParameters(in) {
			d.ParametersJsonSchema = nil
		} else {
			flat, err := typeutil.FlattenNullableForVertex(in)
			if err != nil {
				return nil, fmt.Errorf("flatten parameters schema for Vertex AI: %w", err)
			}
			d.ParametersJsonSchema = flat
		}
	}
	if out != nil {
		flat, err := typeutil.FlattenNullableForVertex(out)
		if err != nil {
			return nil, fmt.Errorf("flatten response schema for Vertex AI: %w", err)
		}
		d.ResponseJsonSchema = flat
	}
	return &d, nil
}

// typedDeclaration is requestDeclaration with the switch on.
func typedDeclaration(base *genai.FunctionDeclaration, in, out *jsonschema.Schema, vertex bool) (*genai.FunctionDeclaration, error) {
	d := *base
	d.ParametersJsonSchema, d.ResponseJsonSchema = nil, nil
	if in != nil && !noParameters(in) {
		if p, ok := typeutil.ToGenaiSchema(in); ok {
			p.Required = typedRequired(p, in)
			typeutil.ClearTitles(p)
			d.Parameters = p
		} else if vertex {
			flat, err := typeutil.FlattenNullableForVertex(in)
			if err != nil {
				return nil, fmt.Errorf("flatten parameters schema for Vertex AI: %w", err)
			}
			d.ParametersJsonSchema = flat
		} else {
			d.ParametersJsonSchema = in
		}
	}
	if vertex && out != nil {
		if r, ok := typeutil.ToGenaiSchema(out); ok {
			typeutil.ClearTitles(r)
			d.Response = r
		}
	}
	return &d, nil
}

// noParameters reports whether s is the schema of a function without
// parameters, the closed empty object jsonschema-go infers for struct{}. An
// open object accepts keys, so it is not.
func noParameters(s *jsonschema.Schema) bool {
	return s != nil && s.Type == "object" && len(s.Properties) == 0 && typeutil.IsFalseSchema(s.AdditionalProperties)
}

// typedRequired is adk-python's _get_required_fields
// (tools/_function_parameter_parse_util.py:588-595) for the typed form p of
// in: the names of in's required list, in that order, whose typed schema is
// not nullable and whose JSON schema has no default.
func typedRequired(p *genai.Schema, in *jsonschema.Schema) []string {
	var required []string
	for _, name := range in.Required {
		if ps := p.Properties[name]; ps != nil && ps.Nullable != nil && *ps.Nullable {
			continue
		}
		if js := in.Properties[name]; js != nil && len(js.Default) > 0 {
			continue
		}
		required = append(required, name)
	}
	return required
}

// schemaOf returns the schema r resolved, or nil for a nil r.
func schemaOf(r *jsonschema.Resolved) *jsonschema.Schema {
	if r == nil {
		return nil
	}
	return r.Schema()
}
