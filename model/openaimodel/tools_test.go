// Copyright 2025 Google LLC
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

package openaimodel

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"
)

func TestConvertTools(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *genai.GenerateContentConfig
		wantLen int
		wantErr bool
	}{
		{
			name: "valid config",
			cfg: &genai.GenerateContentConfig{
				Tools: []*genai.Tool{
					{
						FunctionDeclarations: []*genai.FunctionDeclaration{
							{Name: "fn1"},
						},
					},
				},
			},
			wantLen: 1,
		},
		{
			name:    "empty config",
			cfg:     nil,
			wantLen: 0,
		},
		{
			name: "invalid tool",
			cfg: &genai.GenerateContentConfig{
				Tools: []*genai.Tool{
					{GoogleSearch: &genai.GoogleSearch{}},
				},
			},
			wantErr: true,
		},
		{
			name: "invalid function declaration",
			cfg: &genai.GenerateContentConfig{
				Tools: []*genai.Tool{
					{
						FunctionDeclarations: []*genai.FunctionDeclaration{
							{}, // missing name
						},
					},
				},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tools, err := convertTools(tc.cfg)
			if (err != nil) != tc.wantErr {
				t.Fatalf("convertTools() error = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr {
				return
			}
			if len(tools) != tc.wantLen {
				t.Fatalf("expected %d tools, got %d", tc.wantLen, len(tools))
			}
			if len(tools) > 0 && (tools[0].OfFunction == nil || tools[0].OfFunction.Name != "fn1") {
				t.Fatalf("unexpected tool: %+v", tools[0])
			}
			for i, tool := range tools {
				if tool.OfFunction == nil {
					t.Errorf("tool %d is not a function tool: %+v", i, tool)
					continue
				}
				if !tool.OfFunction.Strict.Valid() || tool.OfFunction.Strict.Value {
					t.Errorf("tool %d Strict = %+v, want an explicit false", i, tool.OfFunction.Strict)
				}
			}
		})
	}
}

func TestEnsureFunctionToolOnly(t *testing.T) {
	tests := []struct {
		name    string
		tool    *genai.Tool
		wantErr string
	}{
		{
			name:    "nil tool",
			tool:    nil,
			wantErr: "tool 0 is nil",
		},
		{
			name:    "non-function tool",
			tool:    &genai.Tool{GoogleSearch: &genai.GoogleSearch{}},
			wantErr: "non-function tools",
		},
		{
			name:    "no functions",
			tool:    &genai.Tool{},
			wantErr: "does not declare any functions",
		},
		{
			name: "valid",
			tool: &genai.Tool{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "fn1"}}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ensureFunctionToolOnly(0, tc.tool)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected no error, got %v", err)
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
				}
			}
		})
	}
}

func TestConvertFunctionDeclaration(t *testing.T) {
	tests := []struct {
		name    string
		decl    *genai.FunctionDeclaration
		wantErr string
	}{
		{
			name:    "nil declaration",
			decl:    nil,
			wantErr: "nil function declaration",
		},
		{
			name:    "missing name",
			decl:    &genai.FunctionDeclaration{},
			wantErr: "missing name",
		},
		{
			name: "valid",
			decl: &genai.FunctionDeclaration{
				Name:        "test_func",
				Description: "A test function",
			},
		},
		{
			name: "with ParametersJsonSchema",
			decl: &genai.FunctionDeclaration{
				Name:                 "test_func",
				ParametersJsonSchema: map[string]any{"type": "object"},
			},
		},
		{
			name: "invalid ParametersJsonSchema",
			decl: &genai.FunctionDeclaration{
				Name:                 "test_func",
				ParametersJsonSchema: func() {}, // unmarshalable
			},
			wantErr: "json: unsupported type",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fn, err := convertFunctionDeclaration(tc.decl)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected no error, got %v", err)
				}
				if fn.Name != tc.decl.Name {
					t.Fatalf("unexpected fn: %+v", fn)
				}
				if fn.Parameters["type"] != "object" {
					t.Fatalf("expected default object schema, got: %+v", fn.Parameters)
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
				}
			}
		})
	}
}

// TestConvertFunctionDeclarationPinsStrictOff covers every parameter path,
// because the Responses API decides the validation mode from the schema shape
// when strict is absent. Pinning it makes the mode independent of the schema.
func TestConvertFunctionDeclarationPinsStrictOff(t *testing.T) {
	tests := []struct {
		name string
		decl *genai.FunctionDeclaration
	}{
		{
			name: "no parameters",
			decl: &genai.FunctionDeclaration{Name: "fn"},
		},
		{
			name: "genai schema parameters",
			decl: &genai.FunctionDeclaration{
				Name: "fn",
				Parameters: &genai.Schema{
					Type:       genai.TypeObject,
					Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
					Required:   []string{"city"},
				},
			},
		},
		{
			// A hand-written schema can be strict-compatible, which is exactly
			// the case where an absent strict flag would flip the API into
			// strict mode without the caller asking for it.
			name: "strict-compatible ParametersJsonSchema",
			decl: &genai.FunctionDeclaration{
				Name: "fn",
				ParametersJsonSchema: map[string]any{
					"type":                 "object",
					"properties":           map[string]any{"city": map[string]any{"type": "string"}},
					"required":             []any{"city"},
					"additionalProperties": false,
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fn, err := convertFunctionDeclaration(tc.decl)
			if err != nil {
				t.Fatalf("convertFunctionDeclaration() error = %v", err)
			}
			if !fn.Strict.Valid() {
				t.Fatal("Strict is unset; the Responses API would choose the validation mode from the schema shape")
			}
			if fn.Strict.Value {
				t.Errorf("Strict = true, want false")
			}
		})
	}
}

// TestConvertFunctionDeclarationMarshalsStrict guards the wire format. Strict is
// tagged omitzero, so leaving it at its zero value drops the key from the
// payload entirely rather than sending false.
func TestConvertFunctionDeclarationMarshalsStrict(t *testing.T) {
	fn, err := convertFunctionDeclaration(&genai.FunctionDeclaration{Name: "fn"})
	if err != nil {
		t.Fatalf("convertFunctionDeclaration() error = %v", err)
	}
	data, err := json.Marshal(fn)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	strict, ok := payload["strict"]
	if !ok {
		t.Fatalf("strict missing from payload %s", data)
	}
	if strict != false {
		t.Errorf("strict = %v, want false", strict)
	}
}

// TestConvertFunctionDeclarationKeepsOptionalParameters pins the constraint that
// rules out simply reusing enforceStrictOpenAISchema here: it rewrites required
// to list every property, which would make optional tool arguments mandatory.
//
// Both paths are covered: ParametersJsonSchema is the one functiontool.New
// takes, and its schema is already strict-shaped apart from required, so a
// rewrite there is invisible unless the fixture omits a property from required.
func TestConvertFunctionDeclarationKeepsOptionalParameters(t *testing.T) {
	tests := []struct {
		name string
		decl *genai.FunctionDeclaration
		// Value the key must still hold after conversion; nil means absent.
		wantAdditionalProperties any
	}{
		{
			// genai.Schema carries no additionalProperties; it must not be injected.
			name: "genai schema parameters",
			decl: &genai.FunctionDeclaration{
				Name: "fn",
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"city":  {Type: genai.TypeString},
						"units": {Type: genai.TypeString},
					},
					Required: []string{"city"},
				},
			},
			wantAdditionalProperties: nil,
		},
		{
			// What functiontool.New emits for an omitempty/omitzero field.
			name: "ParametersJsonSchema with an optional property",
			decl: &genai.FunctionDeclaration{
				Name: "fn",
				ParametersJsonSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city":  map[string]any{"type": "string"},
						"units": map[string]any{"type": "string"},
					},
					"required":             []any{"city"},
					"additionalProperties": false,
				},
			},
			wantAdditionalProperties: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fn, err := convertFunctionDeclaration(tc.decl)
			if err != nil {
				t.Fatalf("convertFunctionDeclaration() error = %v", err)
			}
			want := []any{"city"}
			if got := fn.Parameters["required"]; !reflect.DeepEqual(got, want) {
				t.Errorf("required = %v, want %v", got, want)
			}
			got, ok := fn.Parameters["additionalProperties"]
			if !ok {
				got = nil
			}
			if !reflect.DeepEqual(got, tc.wantAdditionalProperties) {
				t.Errorf("additionalProperties = %v, want %v, in %v", got, tc.wantAdditionalProperties, fn.Parameters)
			}
		})
	}
}

func TestSchemaToMap(t *testing.T) {
	tests := []struct {
		name    string
		schema  *genai.Schema
		wantErr bool
		want    map[string]any
	}{
		{
			name:   "nil schema",
			schema: nil,
			want:   nil,
		},
		{
			name:   "string type",
			schema: &genai.Schema{Type: genai.TypeString},
			want:   map[string]any{"type": "string"}, // Marshals as "STRING" if using standard json, but we lower it
		},
		{
			name:    "invalid type",
			schema:  &genai.Schema{Example: make(chan int)},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := schemaToMap(tc.schema)
			if (err != nil) != tc.wantErr {
				t.Fatalf("schemaToMap() error = %v, wantErr %v", err, tc.wantErr)
			} else {
				if got["type"] != tc.want["type"] {
					t.Fatalf("unexpected map: %+v, want %+v", got, tc.want)
				}
			}
		})
	}
}

func TestConvertToolChoice(t *testing.T) {
	tests := []struct {
		name    string
		toolCfg *genai.ToolConfig
		wantErr bool
		want    *responses.ResponseNewParamsToolChoiceUnion
	}{
		{
			name:    "nil cfg",
			toolCfg: nil,
			want:    nil,
		},
		{
			name: "mode none",
			toolCfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeNone,
				},
			},
			want: &responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone),
			},
		},
		{
			name: "mode unspecified empty",
			toolCfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeUnspecified,
				},
			},
			want: nil,
		},
		{
			name: "mode unspecified with names",
			toolCfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeUnspecified,
					AllowedFunctionNames: []string{"fn1"},
				},
			},
			want: &responses.ResponseNewParamsToolChoiceUnion{
				OfAllowedTools: &responses.ToolChoiceAllowedParam{
					Mode:  responses.ToolChoiceAllowedModeAuto,
					Type:  constant.AllowedTools("allowed_tools"),
					Tools: []map[string]any{{"type": "function", "name": "fn1"}},
				},
			},
		},
		{
			name: "mode auto empty",
			toolCfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAuto,
				},
			},
			want: nil,
		},
		{
			name: "mode auto with names",
			toolCfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAuto,
					AllowedFunctionNames: []string{"fn1"},
				},
			},
			want: &responses.ResponseNewParamsToolChoiceUnion{
				OfAllowedTools: &responses.ToolChoiceAllowedParam{
					Mode:  responses.ToolChoiceAllowedModeAuto,
					Type:  constant.AllowedTools("allowed_tools"),
					Tools: []map[string]any{{"type": "function", "name": "fn1"}},
				},
			},
		},
		{
			name: "mode any empty",
			toolCfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
				},
			},
			want: &responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired),
			},
		},
		{
			name: "mode any with names",
			toolCfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"fn1", ""},
				},
			},
			want: &responses.ResponseNewParamsToolChoiceUnion{
				OfAllowedTools: &responses.ToolChoiceAllowedParam{
					Mode:  responses.ToolChoiceAllowedModeRequired,
					Type:  constant.AllowedTools("allowed_tools"),
					Tools: []map[string]any{{"type": "function", "name": "fn1"}},
				},
			},
		},
		{
			name: "invalid mode",
			toolCfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: "invalid",
				},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := convertToolChoice(tc.toolCfg)
			if (err != nil) != tc.wantErr {
				t.Fatalf("convertToolChoice() error = %v, wantErr %v", err, tc.wantErr)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("convertToolChoice() = %+v, want %+v", got, tc.want)
			}
		})
	}
}
