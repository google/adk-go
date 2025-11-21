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

package anthropic

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestRequestBuilder_FromLLMRequest(t *testing.T) {
	temperature := float32(0.25)
	topP := float32(0.8)
	topK := float32(12)

	request := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello Claude", genai.RoleUser),
			genai.NewContentFromText("Please finish the task.", genai.RoleModel),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("Be concise", ""),
			Temperature:       &temperature,
			TopP:              &topP,
			TopK:              &topK,
			MaxOutputTokens:   64,
			StopSequences:     []string{"END"},
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "lookup",
					Description: "Fetch from memory",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"query": {Type: genai.TypeString},
						},
						Required: []string{"query"},
					},
				}},
			}},
		},
	}

	builder := RequestBuilder{modelName: "claude-3", maxTokens: 1024}
	got, err := builder.FromLLMRequest(request)
	if err != nil {
		t.Fatalf("RequestBuilder.FromLLMRequest() error = %v", err)
	}

	if got.Model != "claude-3" {
		t.Fatalf("Model mismatch: got %v", got.Model)
	}
	if got.MaxTokens != 64 {
		t.Fatalf("MaxTokens not overridden, got %d", got.MaxTokens)
	}

	if len(got.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(got.Messages))
	}
	if got.Messages[0].Role != anthropic.MessageParamRoleUser {
		t.Fatalf("first role = %v, want user", got.Messages[0].Role)
	}
	if text := got.Messages[0].Content[0].GetText(); text == nil || *text != "Hello Claude" {
		t.Fatalf("first message text = %v", text)
	}
	if got.Messages[1].Role != anthropic.MessageParamRoleAssistant {
		t.Fatalf("second role = %v, want assistant", got.Messages[1].Role)
	}
	if text := got.Messages[1].Content[0].GetText(); text == nil || *text != "Please finish the task." {
		t.Fatalf("second message text = %v", text)
	}

	if len(got.System) != 1 || got.System[0].Text != "Be concise" {
		t.Fatalf("system instruction mismatch: %+v", got.System)
	}
	if seq := got.StopSequences; len(seq) != 1 || seq[0] != "END" {
		t.Fatalf("stop sequences mismatch: %v", seq)
	}

	if !got.Temperature.Valid() || got.Temperature.Value != float64(temperature) {
		t.Fatalf("temperature not propagated: %+v", got.Temperature)
	}
	if !got.TopP.Valid() || got.TopP.Value != float64(topP) {
		t.Fatalf("topP not propagated: %+v", got.TopP)
	}
	if !got.TopK.Valid() || got.TopK.Value != int64(topK) {
		t.Fatalf("topK not propagated: %+v", got.TopK)
	}

	if len(got.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(got.Tools))
	}
	tool := got.Tools[0].OfTool
	if tool == nil || tool.Name != "lookup" {
		t.Fatalf("tool mismatch: %+v", got.Tools[0])
	}
	if tool.Description.Or("") != "Fetch from memory" {
		t.Fatalf("tool description not set: %+v", tool.Description)
	}
	if tool.InputSchema.Type != constant.ValueOf[constant.Object]() {
		t.Fatalf("input schema type mismatch: %v", tool.InputSchema.Type)
	}
	props, ok := tool.InputSchema.Properties.(map[string]any)
	if !ok {
		t.Fatalf("input schema properties type = %T", tool.InputSchema.Properties)
	}
	propSchema, ok := props["query"].(map[string]any)
	if !ok {
		t.Fatalf("query property type = %T", props["query"])
	}
	if propSchema["type"] != "string" {
		t.Fatalf("normalized schema type mismatch: %v", propSchema["type"])
	}
	if tool.InputSchema.Required == nil || tool.InputSchema.Required[0] != "query" {
		t.Fatalf("required fields missing: %v", tool.InputSchema.Required)
	}
	if got.ToolChoice.OfTool == nil {
		t.Fatal("tool choice not set")
	}
}

func TestStringifyFunctionResponse(t *testing.T) {
	tests := []struct {
		name string
		resp map[string]any
		want string
	}{
		{
			name: "prefers_result",
			resp: map[string]any{"result": "ok", "output": "nope"},
			want: "ok",
		},
		{
			name: "falls_back_to_output",
			resp: map[string]any{"output": 42},
			want: "42",
		},
		{
			name: "returns_json",
			resp: map[string]any{"other": "value"},
			want: `{"other":"value"}`,
		},
		{
			name: "nil_map",
			resp: nil,
			want: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := stringifyFunctionResponse(tt.resp); got != tt.want {
				t.Fatalf("stringifyFunctionResponse() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestIsImagePart(t *testing.T) {
	tests := []struct {
		name string
		part *genai.Part
		want bool
	}{
		{
			name: "image_data",
			part: &genai.Part{
				InlineData: &genai.Blob{MIMEType: "image/png"},
			},
			want: true,
		},
		{
			name: "uppercase_mime",
			part: &genai.Part{
				InlineData: &genai.Blob{MIMEType: "IMAGE/JPEG"},
			},
			want: true,
		},
		{
			name: "non_image",
			part: &genai.Part{
				InlineData: &genai.Blob{MIMEType: "text/plain"},
			},
			want: false,
		},
		{
			name: "nil_data",
			part: nil,
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isImagePart(tt.part); got != tt.want {
				t.Fatalf("isImagePart() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNormalizeSchemaType(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"child": {Type: genai.TypeString},
		},
		AnyOf: []*genai.Schema{
			{Type: genai.TypeInteger},
		},
		Items: &genai.Schema{Type: genai.TypeBoolean},
	}

	result, err := schemaToMap(schema)
	if err != nil {
		t.Fatalf("schemaToMap() error = %v", err)
	}

	if result["type"] != "object" {
		t.Fatalf("root type not normalized: %v", result["type"])
	}
	props := result["properties"].(map[string]any)
	childType, _ := props["child"].(map[string]any)["type"].(string)
	if childType != "string" {
		t.Fatalf("property type not normalized: %v", childType)
	}
	anyOfSlice := result["anyOf"].([]any)
	firstAnyOf, _ := anyOfSlice[0].(map[string]any)
	if firstAnyOf["type"] != "integer" {
		t.Fatalf("anyOf type not normalized: %v", firstAnyOf["type"])
	}
	items := result["items"].(map[string]any)
	if items["type"] != "boolean" {
		t.Fatalf("items type not normalized: %v", items["type"])
	}
}
