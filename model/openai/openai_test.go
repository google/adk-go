package openai

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/sjzsdu/adk-go/model"
	"google.golang.org/genai"
)

type testTool struct {
	name string
	decl *genai.FunctionDeclaration
}

func (t testTool) Name() string {
	return t.name
}

func (t testTool) Declaration() *genai.FunctionDeclaration {
	return t.decl
}

func TestConvertToOpenAIMessages_ToolCallRoundtrip(t *testing.T) {
	m := &openaiModel{name: "test"}

	req := []*genai.Content{
		{
			Role: "assistant",
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "call-123",
						Name: "search",
						Args: map[string]any{"query": "golang"},
					},
				},
			},
		},
		{
			Role: "user",
			Parts: []*genai.Part{
				{
					FunctionResponse: &genai.FunctionResponse{
						ID:   "call-123",
						Name: "search",
						Response: map[string]any{
							"result": "Go is a programming language.",
						},
					},
				},
			},
		},
	}

	messages, err := m.convertToOpenAIMessages(req)
	if err != nil {
		t.Fatalf("convertToOpenAIMessages() error = %v", err)
	}

	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}

	if got := messages[0].Role; got != "assistant" {
		t.Fatalf("expected assistant role, got %s", got)
	}

	if len(messages[0].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(messages[0].ToolCalls))
	}

	if messages[0].ToolCalls[0].ID != "call-123" {
		t.Fatalf("tool call id mismatch: got %s", messages[0].ToolCalls[0].ID)
	}

	if got := messages[1].Role; got != "tool" {
		t.Fatalf("expected tool role, got %s", got)
	}

	if messages[1].ToolCallID != "call-123" {
		t.Fatalf("tool message id mismatch: got %s", messages[1].ToolCallID)
	}

	var respContent map[string]any
	if err := json.Unmarshal([]byte(messages[1].Content), &respContent); err != nil {
		t.Fatalf("tool message content is not valid json: %v", err)
	}

	if respContent["result"] != "Go is a programming language." {
		t.Fatalf("unexpected tool response: %v", respContent)
	}
}

func TestConvertToOpenAIMessages_TextOrdering(t *testing.T) {
	m := &openaiModel{}

	req := []*genai.Content{
		{
			Role: "user",
			Parts: []*genai.Part{
				{Text: "before"},
				{
					FunctionResponse: &genai.FunctionResponse{
						ID:   "call-1",
						Name: "noop",
						Response: map[string]any{
							"status": "ok",
						},
					},
				},
				{Text: "after"},
			},
		},
	}

	messages, err := m.convertToOpenAIMessages(req)
	if err != nil {
		t.Fatalf("convertToOpenAIMessages() error = %v", err)
	}

	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d: %+v", len(messages), messages)
	}

	if messages[0].Role != "user" || strings.TrimSpace(messages[0].Content) != "before" {
		t.Fatalf("first message mismatch: %+v", messages[0])
	}

	if messages[1].Role != "tool" || messages[1].ToolCallID != "call-1" {
		t.Fatalf("tool message mismatch: %+v", messages[1])
	}

	if messages[2].Role != "user" || strings.TrimSpace(messages[2].Content) != "after" {
		t.Fatalf("third message mismatch: %+v", messages[2])
	}
}

func TestConvertTools(t *testing.T) {
	m := &openaiModel{}
	decl := &genai.FunctionDeclaration{
		Name:        "search",
		Description: "searches the web",
		ParametersJsonSchema: &jsonschema.Schema{
			Type: "object",
			Properties: map[string]*jsonschema.Schema{
				"query": {Type: "string"},
			},
			Required: []string{"query"},
		},
	}

	tools := map[string]any{
		"search": testTool{name: "search", decl: decl},
	}

	result, err := m.convertTools(tools)
	if err != nil {
		t.Fatalf("convertTools() error = %v", err)
	}

	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}

	if result[0].Function.Name != "search" {
		t.Fatalf("unexpected tool name: %s", result[0].Function.Name)
	}

	if result[0].Function.Description != "searches the web" {
		t.Fatalf("unexpected tool description: %s", result[0].Function.Description)
	}

	if result[0].Function.Parameters == nil {
		t.Fatal("expected parameters schema to be populated")
	}
}

func TestOpenAIModel_NewModel(t *testing.T) {
	// Skip test if no API key is provided
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	ctx := context.Background()
	config := Config{
		APIKey: os.Getenv("OPENAI_API_KEY"),
	}

	llmModel, err := NewModel(ctx, "gpt-3.5-turbo", config)
	if err != nil {
		t.Fatalf("Failed to create OpenAI model: %v", err)
	}

	if llmModel.Name() != "gpt-3.5-turbo" {
		t.Errorf("Expected model name 'gpt-3.5-turbo', got '%s'", llmModel.Name())
	}
}

func TestOpenAIModel_GenerateContent(t *testing.T) {
	// Skip test if no API key is provided
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	ctx := context.Background()
	config := Config{
		APIKey: os.Getenv("OPENAI_API_KEY"),
	}

	llmModel, err := NewModel(ctx, "gpt-3.5-turbo", config)
	if err != nil {
		t.Fatalf("Failed to create OpenAI model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What is the capital of France? One word.", "user"),
		},
	}

	// Test non-streaming
	for response, err := range llmModel.GenerateContent(ctx, req, false) {
		if err != nil {
			// Skip test if we get region restrictions or other API errors
			if strings.Contains(err.Error(), "unsupported_country_region_territory") ||
				strings.Contains(err.Error(), "403") {
				t.Skipf("OpenAI API access restricted in this region: %v", err)
			}
			t.Fatalf("Generate content failed: %v", err)
		}

		if response == nil {
			t.Fatal("Response is nil")
		}

		if response.Content == nil {
			t.Fatal("Response content is nil")
		}

		// Check that we got some text
		hasText := false
		for _, part := range response.Content.Parts {
			if part.Text != "" {
				hasText = true
				t.Logf("Response text: %s", part.Text)
				break
			}
		}

		if !hasText {
			t.Error("Expected text in response")
		}

		break // Only check first response
	}
}

func TestOpenAIModel_GenerateContentStream(t *testing.T) {
	// Skip test if no API key is provided
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	ctx := context.Background()
	config := Config{
		APIKey: os.Getenv("OPENAI_API_KEY"),
	}

	llmModel, err := NewModel(ctx, "gpt-3.5-turbo", config)
	if err != nil {
		t.Fatalf("Failed to create OpenAI model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Count from 1 to 5.", "user"),
		},
	}

	// Test streaming
	responseCount := 0
	for response, err := range llmModel.GenerateContent(ctx, req, true) {
		if err != nil {
			// Skip test if we get region restrictions or other API errors
			if strings.Contains(err.Error(), "unsupported_country_region_territory") ||
				strings.Contains(err.Error(), "403") {
				t.Skipf("OpenAI API access restricted in this region: %v", err)
			}
			t.Fatalf("Streaming generate content failed: %v", err)
		}

		if response != nil {
			responseCount++
			t.Logf("Streaming response %d received", responseCount)
		}

		// Don't test for too many responses to avoid long test times
		if responseCount >= 10 {
			break
		}
	}

	if responseCount == 0 {
		t.Error("Expected at least one streaming response")
	}
}
