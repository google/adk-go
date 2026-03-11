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

package converters_test

import (
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/model/anthropic/internal/converters"
)

func TestContentsToMessages_SimpleText(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("Hello", "user"),
	}

	messages, err := converters.ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	if messages[0].Role != "user" {
		t.Errorf("expected role 'user', got %q", messages[0].Role)
	}
}

func TestContentsToMessages_MultiTurn(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("Hello", "user"),
		genai.NewContentFromText("Hi there!", "model"),
		genai.NewContentFromText("How are you?", "user"),
	}

	messages, err := converters.ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(messages))
	}

	expectedRoles := []string{"user", "assistant", "user"}
	for i, msg := range messages {
		if string(msg.Role) != expectedRoles[i] {
			t.Errorf("message %d: expected role %q, got %q", i, expectedRoles[i], msg.Role)
		}
	}
}

func TestContentsToMessages_MergesConsecutiveRoles(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("Hello", "user"),
		genai.NewContentFromText("How are you?", "user"),
	}

	messages, err := converters.ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	// Should merge consecutive user messages
	if len(messages) != 1 {
		t.Fatalf("expected 1 merged message, got %d", len(messages))
	}

	// Should have 2 content blocks
	if len(messages[0].Content) != 2 {
		t.Errorf("expected 2 content blocks, got %d", len(messages[0].Content))
	}
}

func TestContentsToMessages_Empty(t *testing.T) {
	messages, err := converters.ContentsToMessages(nil)
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if messages != nil {
		t.Errorf("expected nil messages, got %v", messages)
	}
}

func TestSystemInstructionToSystem(t *testing.T) {
	tests := []struct {
		name        string
		instruction *genai.Content
		wantLen     int
	}{
		{
			name:        "nil instruction",
			instruction: nil,
			wantLen:     0,
		},
		{
			name:        "single text part",
			instruction: genai.NewContentFromText("You are a helpful assistant.", "system"),
			wantLen:     1,
		},
		{
			name: "multiple text parts",
			instruction: &genai.Content{
				Role: "system",
				Parts: []*genai.Part{
					{Text: "You are a helpful assistant."},
					{Text: "Be concise."},
				},
			},
			wantLen: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			blocks := converters.SystemInstructionToSystem(tt.instruction)
			if len(blocks) != tt.wantLen {
				t.Errorf("SystemInstructionToSystem() returned %d blocks, want %d", len(blocks), tt.wantLen)
			}
		})
	}
}

func TestStopReasonToFinishReason(t *testing.T) {
	tests := []struct {
		name string
		stop anthropic.StopReason
		want genai.FinishReason
	}{
		{"end_turn", anthropic.StopReasonEndTurn, genai.FinishReasonStop},
		{"max_tokens", anthropic.StopReasonMaxTokens, genai.FinishReasonMaxTokens},
		{"stop_sequence", anthropic.StopReasonStopSequence, genai.FinishReasonStop},
		{"tool_use", anthropic.StopReasonToolUse, genai.FinishReasonStop},
		{"unknown", anthropic.StopReason("unknown"), genai.FinishReasonUnspecified},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := converters.StopReasonToFinishReason(tt.stop)
			if got != tt.want {
				t.Errorf("StopReasonToFinishReason(%q) = %v, want %v", tt.stop, got, tt.want)
			}
		})
	}
}

func TestUsageToMetadata(t *testing.T) {
	usage := anthropic.Usage{InputTokens: 10, OutputTokens: 20}
	want := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     10,
		CandidatesTokenCount: 20,
		TotalTokenCount:      30,
	}
	got := converters.UsageToMetadata(usage)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("UsageToMetadata() mismatch (-want +got):\n%s", diff)
	}
}

func TestToolsToAnthropicTools(t *testing.T) {
	tests := []struct {
		name    string
		tools   []*genai.Tool
		wantLen int
	}{
		{
			name:    "nil tools",
			tools:   nil,
			wantLen: 0,
		},
		{
			name: "single tool with one function",
			tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:        "get_weather",
							Description: "Get the weather for a location",
							Parameters: &genai.Schema{
								Type: "object",
								Properties: map[string]*genai.Schema{
									"location": {Type: "string", Description: "The city name"},
								},
								Required: []string{"location"},
							},
						},
					},
				},
			},
			wantLen: 1,
		},
		{
			name: "tool with multiple functions",
			tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{Name: "func1", Description: "Function 1"},
						{Name: "func2", Description: "Function 2"},
					},
				},
			},
			wantLen: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := converters.ToolsToAnthropicTools(tt.tools)
			if len(result) != tt.wantLen {
				t.Errorf("ToolsToAnthropicTools() returned %d tools, want %d", len(result), tt.wantLen)
			}
		})
	}
}

func TestSchemaConversion(t *testing.T) {
	tool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        "complex_func",
				Description: "A function with complex schema",
				Parameters: &genai.Schema{
					Type: "object",
					Properties: map[string]*genai.Schema{
						"name":  {Type: "string"},
						"count": {Type: "integer"},
						"items": {
							Type: "array",
							Items: &genai.Schema{
								Type: "string",
							},
						},
						"nested": {
							Type: "object",
							Properties: map[string]*genai.Schema{
								"inner": {Type: "boolean"},
							},
						},
					},
					Required: []string{"name"},
				},
			},
		},
	}

	result := converters.ToolsToAnthropicTools([]*genai.Tool{tool})
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
}

func TestPartToContentBlock_UnsupportedTypes(t *testing.T) {
	part := &genai.Part{
		ExecutableCode: &genai.ExecutableCode{
			Code:     "print('hello')",
			Language: "python",
		},
	}

	_, err := converters.PartToContentBlock(part)
	if err == nil {
		t.Error("expected error for ExecutableCode, got nil")
	}
}

func TestFunctionResponseToBlock(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{
				FunctionResponse: &genai.FunctionResponse{
					ID:       "call_123",
					Name:     "get_weather",
					Response: map[string]any{"temperature": 72, "condition": "sunny"},
				},
			},
		},
	}

	messages, err := converters.ContentsToMessages([]*genai.Content{content})
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	if len(messages[0].Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(messages[0].Content))
	}
}

func TestFunctionResponseToBlock_RequiresID(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{
				FunctionResponse: &genai.FunctionResponse{
					Name:     "get_weather",
					Response: map[string]any{"temperature": 72},
				},
			},
		},
	}

	_, err := converters.ContentsToMessages([]*genai.Content{content})
	if err == nil {
		t.Fatal("expected error for FunctionResponse without ID, got nil")
	}

	if !strings.Contains(err.Error(), "FunctionResponse.ID is required") {
		t.Errorf("expected error about missing ID, got: %v", err)
	}
}

func TestFunctionResponse_ForcesUserRole(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				FunctionResponse: &genai.FunctionResponse{
					ID:       "toolu_abc123",
					Name:     "get_weather",
					Response: map[string]any{"temperature": 22},
				},
			},
		},
	}

	messages, err := converters.ContentsToMessages([]*genai.Content{content})
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	if messages[0].Role != "user" {
		t.Errorf("expected role 'user' for tool result, got %q", messages[0].Role)
	}
}

func TestFunctionCall_ForcesAssistantRole(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{
				FunctionCall: &genai.FunctionCall{
					ID:   "toolu_abc123",
					Name: "get_weather",
					Args: map[string]any{"location": "London"},
				},
			},
		},
	}

	messages, err := converters.ContentsToMessages([]*genai.Content{content})
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	if messages[0].Role != "assistant" {
		t.Errorf("expected role 'assistant' for tool call, got %q", messages[0].Role)
	}
}

func TestToolCallAndResult_Correlation(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("What's the weather in London?", "user"),
		{
			Role: "model",
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "toolu_weather_123",
						Name: "get_weather",
						Args: map[string]any{"location": "London"},
					},
				},
			},
		},
		{
			Role: "user",
			Parts: []*genai.Part{
				{
					FunctionResponse: &genai.FunctionResponse{
						ID:       "toolu_weather_123",
						Name:     "get_weather",
						Response: map[string]any{"temperature": 15, "condition": "cloudy"},
					},
				},
			},
		},
	}

	messages, err := converters.ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(messages))
	}

	expectedRoles := []string{"user", "assistant", "user"}
	for i, msg := range messages {
		if string(msg.Role) != expectedRoles[i] {
			t.Errorf("message %d: expected role %q, got %q", i, expectedRoles[i], msg.Role)
		}
	}
}

var cmpOpts = []cmp.Option{
	cmp.AllowUnexported(),
}

var _ = cmpOpts

func TestFunctionDeclarationToTool_JsonSchemaNoRequired(t *testing.T) {
	fd := &genai.FunctionDeclaration{
		Name:        "test_func",
		Description: "A test function",
		ParametersJsonSchema: &jsonschema.Schema{
			Properties: map[string]*jsonschema.Schema{
				"name": {Type: "string"},
			},
			// Required intentionally omitted (nil)
		},
	}

	result := converters.FunctionDeclarationToTool(fd)
	if result.OfTool == nil {
		t.Fatal("expected OfTool to be non-nil")
	}

	if result.OfTool.InputSchema.Required != nil {
		t.Errorf("expected Required to be nil when jsonschema.Schema has no required fields, got %v",
			result.OfTool.InputSchema.Required)
	}

	props, ok := result.OfTool.InputSchema.Properties.(map[string]any)
	if !ok {
		t.Fatalf("expected Properties to be map[string]any, got %T", result.OfTool.InputSchema.Properties)
	}
	if _, ok := props["name"]; !ok {
		t.Error("expected 'name' property to be present")
	}
}

func TestFunctionDeclarationToTool_ParametersJsonSchemaMap(t *testing.T) {
	tests := []struct {
		name         string
		fd           *genai.FunctionDeclaration
		wantProps    map[string]any
		wantRequired []string
	}{
		{
			name: "map with []any required (JSON unmarshalled)",
			fd: &genai.FunctionDeclaration{
				Name:        "test_func",
				Description: "A test function",
				ParametersJsonSchema: map[string]any{
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The city name",
						},
					},
					"required": []any{"location"},
				},
			},
			wantProps: map[string]any{
				"location": map[string]any{
					"type":        "string",
					"description": "The city name",
				},
			},
			wantRequired: []string{"location"},
		},
		{
			name: "map with []string required (manually constructed)",
			fd: &genai.FunctionDeclaration{
				Name:        "test_func",
				Description: "A test function",
				ParametersJsonSchema: map[string]any{
					"properties": map[string]any{
						"name": map[string]any{"type": "string"},
						"age":  map[string]any{"type": "integer"},
					},
					"required": []string{"name", "age"},
				},
			},
			wantProps: map[string]any{
				"name": map[string]any{"type": "string"},
				"age":  map[string]any{"type": "integer"},
			},
			wantRequired: []string{"name", "age"},
		},
		{
			name: "map with no required field",
			fd: &genai.FunctionDeclaration{
				Name: "optional_func",
				ParametersJsonSchema: map[string]any{
					"properties": map[string]any{
						"optional": map[string]any{"type": "string"},
					},
				},
			},
			wantProps: map[string]any{
				"optional": map[string]any{"type": "string"},
			},
			wantRequired: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := converters.FunctionDeclarationToTool(tt.fd)

			if result.OfTool == nil {
				t.Fatal("expected OfTool to be non-nil")
			}

			is := result.OfTool.InputSchema

			if diff := cmp.Diff(tt.wantProps, is.Properties); diff != "" {
				t.Errorf("Properties mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.wantRequired, is.Required); diff != "" {
				t.Errorf("Required mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFunctionDeclarationToTool_ParametersPrecedence(t *testing.T) {
	fd := &genai.FunctionDeclaration{
		Name: "test_func",
		Parameters: &genai.Schema{
			Type: "object",
			Properties: map[string]*genai.Schema{
				"from_parameters": {Type: "string"},
			},
			Required: []string{"from_parameters"},
		},
		ParametersJsonSchema: map[string]any{
			"properties": map[string]any{
				"from_json_schema": map[string]any{"type": "string"},
			},
			"required": []any{"from_json_schema"},
		},
	}

	result := converters.FunctionDeclarationToTool(fd)

	if result.OfTool == nil {
		t.Fatal("expected OfTool to be non-nil")
	}

	is := result.OfTool.InputSchema
	props, ok := is.Properties.(map[string]any)
	if !ok {
		t.Fatalf("expected Properties to be map[string]any, got %T", is.Properties)
	}

	if _, ok := props["from_parameters"]; !ok {
		t.Error("expected 'from_parameters' property from Parameters (should take precedence)")
	}
	if _, ok := props["from_json_schema"]; ok {
		t.Error("unexpected 'from_json_schema' property - Parameters should take precedence")
	}
}

func TestSchemaToMap_AllFields(t *testing.T) {
	min, max := 1.0, 100.0
	minLen, maxLen := int64(1), int64(50)
	minItems, maxItems := int64(1), int64(10)
	nullable := true

	tool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        "full_schema_func",
				Description: "Function with all schema fields",
				Parameters: &genai.Schema{
					Type:        "object",
					Description: "Root object",
					Properties: map[string]*genai.Schema{
						"name": {
							Type:        "STRING",
							Description: "User name",
							MinLength:   &minLen,
							MaxLength:   &maxLen,
							Pattern:     "^[a-zA-Z]+$",
						},
						"age": {
							Type:     "INTEGER",
							Minimum:  &min,
							Maximum:  &max,
							Nullable: &nullable,
						},
						"tags": {
							Type:     "ARRAY",
							MinItems: &minItems,
							MaxItems: &maxItems,
							Items:    &genai.Schema{Type: "STRING"},
						},
						"status": {
							Type: "STRING",
							Enum: []string{"active", "inactive"},
						},
						"metadata": {
							Type: "OBJECT",
							Properties: map[string]*genai.Schema{
								"created": {Type: "STRING", Format: "date-time"},
							},
						},
					},
					Required: []string{"name"},
				},
			},
		},
	}

	result := converters.ToolsToAnthropicTools([]*genai.Tool{tool})
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}

	is := result[0].OfTool.InputSchema
	props, ok := is.Properties.(map[string]any)
	if !ok {
		t.Fatalf("expected Properties to be map[string]any, got %T", is.Properties)
	}

	nameSchema, ok := props["name"].(map[string]any)
	if !ok {
		t.Fatal("expected 'name' property")
	}
	if nameSchema["type"] != "string" {
		t.Errorf("name.type = %v, want 'string'", nameSchema["type"])
	}
	if nameSchema["minLength"] != minLen {
		t.Errorf("name.minLength = %v, want %v", nameSchema["minLength"], minLen)
	}
	if nameSchema["pattern"] != "^[a-zA-Z]+$" {
		t.Errorf("name.pattern = %v, want '^[a-zA-Z]+$'", nameSchema["pattern"])
	}

	ageSchema, ok := props["age"].(map[string]any)
	if !ok {
		t.Fatal("expected 'age' property")
	}
	if ageSchema["nullable"] != true {
		t.Errorf("age.nullable = %v, want true", ageSchema["nullable"])
	}
	if ageSchema["minimum"] != min {
		t.Errorf("age.minimum = %v, want %v", ageSchema["minimum"], min)
	}

	tagsSchema, ok := props["tags"].(map[string]any)
	if !ok {
		t.Fatal("expected 'tags' property")
	}
	if tagsSchema["minItems"] != minItems {
		t.Errorf("tags.minItems = %v, want %v", tagsSchema["minItems"], minItems)
	}
	items, ok := tagsSchema["items"].(map[string]any)
	if !ok {
		t.Fatal("expected 'tags.items'")
	}
	if items["type"] != "string" {
		t.Errorf("tags.items.type = %v, want 'string'", items["type"])
	}

	statusSchema, ok := props["status"].(map[string]any)
	if !ok {
		t.Fatal("expected 'status' property")
	}
	if statusSchema["enum"] == nil {
		t.Error("expected 'status.enum' to be set")
	}

	metaSchema, ok := props["metadata"].(map[string]any)
	if !ok {
		t.Fatal("expected 'metadata' property")
	}
	metaProps, ok := metaSchema["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected 'metadata.properties'")
	}
	createdSchema, ok := metaProps["created"].(map[string]any)
	if !ok {
		t.Fatal("expected 'metadata.properties.created'")
	}
	if createdSchema["format"] != "date-time" {
		t.Errorf("created.format = %v, want 'date-time'", createdSchema["format"])
	}
}

func TestSchemaToMap_AnyOf(t *testing.T) {
	tool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name: "anyof_func",
				Parameters: &genai.Schema{
					Type: "object",
					Properties: map[string]*genai.Schema{
						"value": {
							AnyOf: []*genai.Schema{
								{Type: "STRING"},
								{Type: "INTEGER"},
							},
						},
					},
				},
			},
		},
	}

	result := converters.ToolsToAnthropicTools([]*genai.Tool{tool})
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}

	props, ok := result[0].OfTool.InputSchema.Properties.(map[string]any)
	if !ok {
		t.Fatalf("expected Properties to be map[string]any, got %T", result[0].OfTool.InputSchema.Properties)
	}
	valueSchema, ok := props["value"].(map[string]any)
	if !ok {
		t.Fatal("expected 'value' property")
	}

	anyOf, ok := valueSchema["anyOf"].([]map[string]any)
	if !ok {
		t.Fatalf("expected 'anyOf' to be []map[string]any, got %T", valueSchema["anyOf"])
	}
	if len(anyOf) != 2 {
		t.Errorf("expected 2 anyOf entries, got %d", len(anyOf))
	}
}

func TestMessageToLLMResponse_WithCitations(t *testing.T) {
	msgJSON := `{
		"content": [{
			"type": "text",
			"text": "According to the documentation...",
			"citations": [
				{
					"type": "char_location",
					"document_title": "API Reference",
					"start_char_index": 0,
					"end_char_index": 50,
					"cited_text": "some text"
				},
				{
					"type": "web_search_result_location",
					"title": "Official Docs",
					"url": "https://example.com/docs",
					"cited_text": "other text"
				}
			]
		}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 20}
	}`

	var msg anthropic.Message
	if err := msg.UnmarshalJSON([]byte(msgJSON)); err != nil {
		t.Fatalf("failed to unmarshal message: %v", err)
	}

	resp, err := converters.MessageToLLMResponse(&msg)
	if err != nil {
		t.Fatalf("MessageToLLMResponse() error = %v", err)
	}

	if resp.CitationMetadata == nil {
		t.Fatal("expected CitationMetadata to be set")
	}
	if len(resp.CitationMetadata.Citations) != 2 {
		t.Fatalf("expected 2 citations, got %d", len(resp.CitationMetadata.Citations))
	}

	c0 := resp.CitationMetadata.Citations[0]
	if c0.Title != "API Reference" {
		t.Errorf("citation[0].Title = %q, want 'API Reference'", c0.Title)
	}
	if c0.StartIndex != 0 || c0.EndIndex != 50 {
		t.Errorf("citation[0] indices = (%d, %d), want (0, 50)", c0.StartIndex, c0.EndIndex)
	}

	c1 := resp.CitationMetadata.Citations[1]
	if c1.Title != "Official Docs" {
		t.Errorf("citation[1].Title = %q, want 'Official Docs'", c1.Title)
	}
	if c1.URI != "https://example.com/docs" {
		t.Errorf("citation[1].URI = %q, want 'https://example.com/docs'", c1.URI)
	}
}

func TestContentBlockToGenaiPart_WebSearchToolResult(t *testing.T) {
	blockJSON := `{
		"type": "web_search_tool_result",
		"tool_use_id": "toolu_search_123",
		"content": [
			{
				"type": "web_search_result",
				"title": "Example Page",
				"url": "https://example.com",
				"page_age": "2 days ago",
				"encrypted_content": "abc123"
			},
			{
				"type": "web_search_result",
				"title": "Another Page",
				"url": "https://another.com",
				"page_age": "1 week ago",
				"encrypted_content": "def456"
			}
		]
	}`

	var block anthropic.ContentBlockUnion
	if err := block.UnmarshalJSON([]byte(blockJSON)); err != nil {
		t.Fatalf("failed to unmarshal block: %v", err)
	}

	part, err := converters.ContentBlockToGenaiPart(block)
	if err != nil {
		t.Fatalf("ContentBlockToGenaiPart() error = %v", err)
	}

	if part == nil {
		t.Fatal("expected part to be non-nil")
	}
	if part.FunctionResponse == nil {
		t.Fatal("expected FunctionResponse to be set")
	}
	if part.FunctionResponse.ID != "toolu_search_123" {
		t.Errorf("FunctionResponse.ID = %q, want 'toolu_search_123'", part.FunctionResponse.ID)
	}
	if part.FunctionResponse.Name != "web_search" {
		t.Errorf("FunctionResponse.Name = %q, want 'web_search'", part.FunctionResponse.Name)
	}

	results, ok := part.FunctionResponse.Response["results"].([]map[string]any)
	if !ok {
		t.Fatalf("expected results to be []map[string]any, got %T", part.FunctionResponse.Response["results"])
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0]["title"] != "Example Page" {
		t.Errorf("results[0].title = %q, want 'Example Page'", results[0]["title"])
	}
	if results[0]["url"] != "https://example.com" {
		t.Errorf("results[0].url = %q, want 'https://example.com'", results[0]["url"])
	}
}

func int32Ptr(v int32) *int32 { return &v }

func TestThinkingConfigToAnthropicThinking(t *testing.T) {
	tests := []struct {
		name       string
		cfg        *genai.ThinkingConfig
		wantNil    bool // expect zero value (no thinking)
		wantBudget int64
	}{
		{
			name:    "nil config",
			cfg:     nil,
			wantNil: true,
		},
		{
			name:       "HIGH level without budget",
			cfg:        &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
			wantBudget: 10000,
		},
		{
			name:       "LOW level without budget",
			cfg:        &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow},
			wantBudget: 1024,
		},
		{
			name:       "explicit budget",
			cfg:        &genai.ThinkingConfig{ThinkingBudget: int32Ptr(5000)},
			wantBudget: 5000,
		},
		{
			name:       "level with explicit budget - budget wins",
			cfg:        &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow, ThinkingBudget: int32Ptr(8000)},
			wantBudget: 8000,
		},
		{
			name:       "IncludeThoughts alone",
			cfg:        &genai.ThinkingConfig{IncludeThoughts: true},
			wantBudget: 10000,
		},
		{
			name:    "unspecified level no budget no include",
			cfg:     &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelUnspecified},
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := converters.ThinkingConfigToAnthropicThinking(tt.cfg)
			if tt.wantNil {
				if got.OfEnabled != nil {
					t.Errorf("expected zero value, got OfEnabled with budget %d", got.OfEnabled.BudgetTokens)
				}
				return
			}
			if got.OfEnabled == nil {
				t.Fatal("expected OfEnabled to be non-nil")
			}
			if got.OfEnabled.BudgetTokens != tt.wantBudget {
				t.Errorf("BudgetTokens = %d, want %d", got.OfEnabled.BudgetTokens, tt.wantBudget)
			}
		})
	}
}

func TestToolConfigToToolChoice(t *testing.T) {
	tests := []struct {
		name      string
		config    *genai.ToolConfig
		wantAuto  bool
		wantAny   bool
		wantTool  string
		wantZero  bool
		wantError bool
	}{
		{
			name:     "nil config",
			config:   nil,
			wantZero: true,
		},
		{
			name:     "nil FunctionCallingConfig",
			config:   &genai.ToolConfig{},
			wantZero: true,
		},
		{
			name: "ModeNone",
			config: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeNone,
				},
			},
			wantZero: true,
		},
		{
			name: "ModeAuto",
			config: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAuto,
				},
			},
			wantAuto: true,
		},
		{
			name: "ModeAny",
			config: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
				},
			},
			wantAny: true,
		},
		{
			name: "ModeAny with single AllowedFunctionNames",
			config: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"get_weather"},
				},
			},
			wantTool: "get_weather",
		},
		{
			name: "multiple AllowedFunctionNames returns error",
			config: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"func1", "func2"},
				},
			},
			wantError: true,
		},
		{
			name: "unknown mode returns error",
			config: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigMode("UNKNOWN"),
				},
			},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := converters.ToolConfigToToolChoice(tt.config)

			if tt.wantError {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.wantZero {
				if result.OfAuto != nil || result.OfAny != nil || result.OfTool != nil {
					t.Error("expected zero-value result")
				}
				return
			}
			if tt.wantAuto && result.OfAuto == nil {
				t.Error("expected OfAuto to be set")
			}
			if tt.wantAny && result.OfAny == nil {
				t.Error("expected OfAny to be set")
			}
			if tt.wantTool != "" {
				if result.OfTool == nil {
					t.Fatal("expected OfTool to be set")
				}
				if result.OfTool.Name != tt.wantTool {
					t.Errorf("OfTool.Name = %q, want %q", result.OfTool.Name, tt.wantTool)
				}
			}
		})
	}
}

func TestSchemaToJSONString(t *testing.T) {
	tests := []struct {
		name     string
		schema   *genai.Schema
		contains []string
	}{
		{
			name:     "nil_schema",
			schema:   nil,
			contains: []string{"null"},
		},
		{
			name: "simple_object",
			schema: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"name": {Type: genai.TypeString},
				},
				Required: []string{"name"},
			},
			contains: []string{`"type": "object"`, `"properties"`, `"name"`, `"required"`},
		},
		{
			name: "with_description",
			schema: &genai.Schema{
				Type:        genai.TypeString,
				Description: "A user name",
			},
			contains: []string{`"type": "string"`, `"description": "A user name"`},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := converters.SchemaToJSONString(tt.schema)

			for _, want := range tt.contains {
				if !strings.Contains(result, want) {
					t.Errorf("SchemaToJSONString() = %q, want to contain %q", result, want)
				}
			}
		})
	}
}
