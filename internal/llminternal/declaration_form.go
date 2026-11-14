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

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/exp/feature"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/typeutil"
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
	flat, err := typeutil.FlattenNullableForVertex(params)
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
