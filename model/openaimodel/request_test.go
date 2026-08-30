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
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func TestBuildOpenAIParams_Text(t *testing.T) {
	req := &model.LLMRequest{
		Model: "gpt-4o-mini",
		Contents: []*genai.Content{
			genai.NewContentFromText("ping", genai.RoleUser),
		},
	}
	params, err := buildOpenAIParams("fallback", req)
	if err != nil {
		t.Fatalf("buildOpenAIParams() err = %v", err)
	}
	if got, want := string(params.Model), "gpt-4o-mini"; got != want {
		t.Fatalf("Model mismatch got=%q want=%q", got, want)
	}
	items := params.Input.OfInputItemList
	if len(items) != 1 || items[0].OfMessage == nil {
		t.Fatalf("unexpected input items: %+v", items)
	}
	textParts := items[0].OfMessage.Content.OfInputItemContentList
	if len(textParts) != 1 {
		t.Fatalf("unexpected message parts: %+v", textParts)
	}
	if got, want := textParts[0].OfInputText.Text, "ping"; got != want {
		t.Fatalf("text mismatch got=%q want=%q", got, want)
	}
}

func TestBuildOpenAIParams_FunctionCall(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: string(genai.RoleModel),
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: "lookup", Args: map[string]any{"city": "Paris"}}},
					{FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: map[string]any{"temp": 72}}},
				},
			},
		},
	}
	params, err := buildOpenAIParams("fallback", req)
	if err != nil {
		t.Fatalf("buildOpenAIParams() err = %v", err)
	}
	var call *responses.ResponseFunctionToolCallParam
	var response *responses.ResponseInputItemFunctionCallOutputParam
	for _, item := range params.Input.OfInputItemList {
		switch {
		case item.OfFunctionCall != nil:
			call = item.OfFunctionCall
		case item.OfFunctionCallOutput != nil:
			response = item.OfFunctionCallOutput
		}
	}
	if call == nil || response == nil {
		t.Fatalf("missing function call/response in %+v", params.Input.OfInputItemList)
		return
	}
	if call.CallID == "" || response.CallID == "" {
		t.Fatalf("call IDs must be populated: call=%+v response=%+v", call, response)
		return
	}
	if call.CallID != response.CallID {
		t.Fatalf("call IDs mismatch: %q vs %q", call.CallID, response.CallID)
	}
}

func TestBuildOpenAIParams_JSONSchema(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("respond JSON", genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			ResponseMIMEType: "application/json",
			ResponseSchema: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"answer": {Type: genai.TypeString},
				},
			},
		},
	}
	params, err := buildOpenAIParams("fallback", req)
	if err != nil {
		t.Fatalf("buildOpenAIParams() err = %v", err)
	}
	if params.Text.Format.OfJSONSchema == nil {
		t.Fatalf("expected json schema format, got: %+v", params.Text.Format)
	}
	if got := params.Text.Format.OfJSONSchema.Schema["type"]; got != "object" {
		t.Fatalf("schema mismatch got=%v", got)
	}
}

func TestBuildOpenAIParams_UnsupportedPart(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: string(genai.RoleUser),
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{Data: []byte{0x1}}},
				},
			},
		},
	}
	if _, err := buildOpenAIParams("fallback", req); err == nil {
		t.Fatalf("expected error for inline data part")
	}
}

func TestCallTrackerNewFunctionResponse_UnknownCallID(t *testing.T) {
	tracker := callTracker{pending: []string{"call-1"}}
	fr := &genai.FunctionResponse{
		Name:     "lookup",
		ID:       "call-missing",
		Response: map[string]any{"ok": true},
	}
	if _, err := tracker.newFunctionResponse(fr); err == nil || !strings.Contains(err.Error(), "unknown or already completed") {
		t.Fatalf("expected error for unknown call id, got %v", err)
	}
	if len(tracker.pending) != 1 || tracker.pending[0] != "call-1" {
		t.Fatalf("pending calls should remain untouched, got %+v", tracker.pending)
	}
}

func TestApplyGenerationConfig(t *testing.T) {
	topK := float32(5)
	p := float32(0.5)
	temp := float32(0.8)
	topP := float32(0.9)
	logprobs := int32(2)

	tests := []struct {
		name       string
		cfg        *genai.GenerateContentConfig
		wantErr    error
		wantParams *responses.ResponseNewParams
	}{
		{
			name: "nil config",
			cfg:  nil,
		},
		{
			name:    "TopK not supported",
			cfg:     &genai.GenerateContentConfig{TopK: &topK},
			wantErr: ErrTopKNotSupported,
		},
		{
			name:    "StopSequences not supported",
			cfg:     &genai.GenerateContentConfig{StopSequences: []string{"stop"}},
			wantErr: ErrStopSequencesNotSupported,
		},
		{
			name:    "Multiple candidates not supported",
			cfg:     &genai.GenerateContentConfig{CandidateCount: 2},
			wantErr: ErrMultipleCandidatesNotSupported,
		},
		{
			name:    "Penalties not supported",
			cfg:     &genai.GenerateContentConfig{FrequencyPenalty: &p},
			wantErr: ErrPenaltiesNotSupported,
		},
		{
			name:    "Labels not supported",
			cfg:     &genai.GenerateContentConfig{Labels: map[string]string{"a": "b"}},
			wantErr: ErrLabelsNotSupported,
		},
		{
			name:    "Safety settings not supported",
			cfg:     &genai.GenerateContentConfig{SafetySettings: []*genai.SafetySetting{{}}},
			wantErr: ErrSafetySettingsNotSupported,
		},
		{
			name:    "Unsupported MIME type",
			cfg:     &genai.GenerateContentConfig{ResponseMIMEType: "image/png"},
			wantErr: ErrUnsupportedMIMEType,
		},
		{
			name: "success fully configured",
			cfg: &genai.GenerateContentConfig{
				Temperature:       &temp,
				TopP:              &topP,
				MaxOutputTokens:   100,
				ResponseLogprobs:  true,
				Logprobs:          &logprobs,
				SystemInstruction: genai.NewContentFromText("sys", "system"),
				ResponseMIMEType:  "application/json",
				ResponseSchema:    &genai.Schema{Type: genai.TypeObject},
			},
			wantParams: &responses.ResponseNewParams{
				Temperature:     param.NewOpt(float64(float32(temp))),
				TopP:            param.NewOpt(float64(float32(topP))),
				MaxOutputTokens: param.NewOpt(int64(100)),
				TopLogprobs:     param.NewOpt(int64(int32(logprobs))),
				Include:         []responses.ResponseIncludable{responses.ResponseIncludableMessageOutputTextLogprobs},
				Instructions:    param.NewOpt("sys"),
				Text: responses.ResponseTextConfigParam{
					Format: responses.ResponseFormatTextConfigUnionParam{
						OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
							Name:   "adk_response",
							Strict: param.NewOpt(true),
							Type:   constant.JSONSchema("json_schema"),
							Schema: map[string]any{
								"type": "object",
							},
						},
					},
				},
			},
		},
		{
			name: "success application/json without schema falls back to json_object",
			cfg: &genai.GenerateContentConfig{
				ResponseMIMEType: "application/json",
			},
			wantParams: &responses.ResponseNewParams{
				Text: responses.ResponseTextConfigParam{
					Format: responses.ResponseFormatTextConfigUnionParam{
						OfJSONObject: &shared.ResponseFormatJSONObjectParam{
							Type: constant.JSONObject("json_object"),
						},
					},
				},
			},
		},
		{
			name: "success logprobs only",
			cfg: &genai.GenerateContentConfig{
				ResponseLogprobs: true,
			},
			wantParams: &responses.ResponseNewParams{
				TopLogprobs: param.NewOpt(int64(1)),
				Include:     []responses.ResponseIncludable{responses.ResponseIncludableMessageOutputTextLogprobs},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			params := &responses.ResponseNewParams{}
			err := applyGenerationConfig(params, tc.cfg)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("applyGenerationConfig() error = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantParams != nil && !reflect.DeepEqual(params, tc.wantParams) {
				t.Errorf("applyGenerationConfig() params = %+v, want %+v", params, tc.wantParams)
			}
		})
	}
}

func TestFlattenContentText(t *testing.T) {
	tests := []struct {
		name    string
		content *genai.Content
		want    string
		wantErr bool
	}{
		{
			name:    "nil content",
			content: nil,
			want:    "",
		},
		{
			name: "valid text parts",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "part1"},
					nil,
					{Text: "part2"},
				},
			},
			want: "part1\npart2",
		},
		{
			name: "non-text part",
			content: &genai.Content{
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: "fn"}},
				},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			txt, err := flattenContentText(tc.content)
			if (err != nil) != tc.wantErr {
				t.Fatalf("flattenContentText() error = %v, wantErr %v", err, tc.wantErr)
			}
			if txt != tc.want {
				t.Fatalf("flattenContentText() = %q, want %q", txt, tc.want)
			}
		})
	}
}

func TestNormalizeSchema(t *testing.T) {
	tests := []struct {
		name    string
		schema  any
		want    map[string]any
		wantErr bool
	}{
		{
			name:    "nil schema",
			schema:  nil,
			wantErr: true,
		},
		{
			name:   "map schema",
			schema: map[string]any{"type": "object"},
			want:   map[string]any{"type": "object"},
		},
		{
			name: "struct schema",
			schema: struct {
				Type string `json:"type"`
			}{Type: "array"},
			want: map[string]any{"type": "array"},
		},
		{
			name:    "invalid schema",
			schema:  func() {}, // unmarshalable
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := normalizeSchema(tc.schema)
			if (err != nil) != tc.wantErr {
				t.Fatalf("normalizeSchema() error = %v, wantErr %v", err, tc.wantErr)
			}
			if !tc.wantErr && got["type"] != tc.want["type"] {
				t.Fatalf("normalizeSchema() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestNormalizeRole(t *testing.T) {
	tests := []struct {
		role    genai.Role
		want    responses.EasyInputMessageRole
		wantErr bool
	}{
		{"", responses.EasyInputMessageRoleUser, false},
		{genai.RoleUser, responses.EasyInputMessageRoleUser, false},
		{genai.RoleModel, responses.EasyInputMessageRoleAssistant, false},
		{"system", responses.EasyInputMessageRoleSystem, false},
		{"developer", responses.EasyInputMessageRoleDeveloper, false},
		{"invalid", "", true},
	}
	for _, tc := range tests {
		t.Run(string(tc.role), func(t *testing.T) {
			got, err := normalizeRole(tc.role)
			if (err != nil) != tc.wantErr {
				t.Fatalf("normalizeRole() error = %v, wantErr %v", err, tc.wantErr)
			}
			if got != tc.want {
				t.Fatalf("normalizeRole() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestNewJSONSchemaFormat(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *genai.GenerateContentConfig
		want    *responses.ResponseFormatTextJSONSchemaConfigParam
		wantErr bool
	}{
		{
			name:    "no schema",
			cfg:     &genai.GenerateContentConfig{},
			wantErr: true,
		},
		{
			name: "with response schema",
			cfg: &genai.GenerateContentConfig{
				ResponseSchema: &genai.Schema{Title: "CustomTitle", Type: genai.TypeObject},
			},
			want: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "CustomTitle",
				Strict: param.NewOpt(true),
				Type:   constant.JSONSchema("json_schema"),
				Schema: map[string]any{
					"title": "CustomTitle",
					"type":  "object",
				},
			},
		},

		{
			name: "with json schema",
			cfg: &genai.GenerateContentConfig{
				ResponseJsonSchema: map[string]any{"type": "object"},
			},
			want: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "adk_response",
				Strict: param.NewOpt(true),
				Type:   constant.JSONSchema("json_schema"),
				Schema: map[string]any{
					"type": "object",
				},
			},
		},
		{
			name: "with nested response schema",
			cfg: &genai.GenerateContentConfig{
				ResponseSchema: &genai.Schema{
					Title: "NestedTitle",
					Type:  genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"b_string": {Type: genai.TypeString},
						"a_object": {
							Type: genai.TypeObject,
							Properties: map[string]*genai.Schema{
								"d_int":  {Type: genai.TypeInteger},
								"c_bool": {Type: genai.TypeBoolean},
							},
						},
					},
				},
			},
			want: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "NestedTitle",
				Strict: param.NewOpt(true),
				Type:   constant.JSONSchema("json_schema"),
				Schema: map[string]any{
					"title":                "NestedTitle",
					"type":                 "object",
					"additionalProperties": false,
					"required":             []string{"a_object", "b_string"},
					"properties": map[string]any{
						"b_string": map[string]any{
							"type": "string",
						},
						"a_object": map[string]any{
							"type":                 "object",
							"additionalProperties": false,
							"required":             []string{"c_bool", "d_int"},
							"properties": map[string]any{
								"d_int": map[string]any{
									"type": "integer",
								},
								"c_bool": map[string]any{
									"type": "boolean",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "with complex json schema for strict output",
			cfg: &genai.GenerateContentConfig{
				ResponseJsonSchema: map[string]any{
					"title": "NestedTitle",
					"type":  "object",
					"properties": map[string]any{
						"b_string": map[string]any{"type": "string"},
						"a_object": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"d_int":  map[string]any{"type": "integer"},
								"c_bool": map[string]any{"type": "boolean"},
							},
						},
						"c_array": map[string]any{
							"type": "array",
							"items": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"e_float": map[string]any{"type": "number"},
								},
							},
						},
						"d_ref": map[string]any{
							"$ref":        "#/$defs/my_def",
							"description": "this should be deleted",
						},
					},
					"$defs": map[string]any{
						"my_def": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"f_string": map[string]any{"type": "string"},
							},
						},
					},
					"anyOf": []any{
						map[string]any{
							"type": "object",
							"properties": map[string]any{
								"g_string": map[string]any{"type": "string"},
							},
						},
					},
				},
			},
			want: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "adk_response",
				Strict: param.NewOpt(true),
				Type:   constant.JSONSchema("json_schema"),
				Schema: map[string]any{
					"title":                "NestedTitle",
					"type":                 "object",
					"additionalProperties": false,
					"required":             []string{"a_object", "b_string", "c_array", "d_ref"},
					"properties": map[string]any{
						"b_string": map[string]any{
							"type": "string",
						},
						"a_object": map[string]any{
							"type":                 "object",
							"additionalProperties": false,
							"required":             []string{"c_bool", "d_int"},
							"properties": map[string]any{
								"d_int": map[string]any{
									"type": "integer",
								},
								"c_bool": map[string]any{
									"type": "boolean",
								},
							},
						},
						"c_array": map[string]any{
							"type": "array",
							"items": map[string]any{
								"type":                 "object",
								"additionalProperties": false,
								"required":             []string{"e_float"},
								"properties": map[string]any{
									"e_float": map[string]any{"type": "number"},
								},
							},
						},
						"d_ref": map[string]any{
							"$ref": "#/$defs/my_def",
						},
					},
					"$defs": map[string]any{
						"my_def": map[string]any{
							"type":                 "object",
							"additionalProperties": false,
							"required":             []string{"f_string"},
							"properties": map[string]any{
								"f_string": map[string]any{"type": "string"},
							},
						},
					},
					"anyOf": []any{
						map[string]any{
							"type":                 "object",
							"additionalProperties": false,
							"required":             []string{"g_string"},
							"properties": map[string]any{
								"g_string": map[string]any{"type": "string"},
							},
						},
					},
				},
			},
		},
		{
			name: "with invalid json schema",
			cfg: &genai.GenerateContentConfig{
				ResponseJsonSchema: func() {}, // unmarshalable
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := newJSONSchemaFormat(tc.cfg)
			if (err != nil) != tc.wantErr {
				t.Fatalf("newJSONSchemaFormat() error = %v, wantErr %v", err, tc.wantErr)
			}
			if !tc.wantErr && !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("newJSONSchemaFormat() got = %+v, want %+v", got, tc.want)
			}
		})
	}
}
