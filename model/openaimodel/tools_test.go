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
