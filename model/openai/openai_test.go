package openai

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/sjzsdu/adk-go/model"
	"google.golang.org/genai"
)

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
