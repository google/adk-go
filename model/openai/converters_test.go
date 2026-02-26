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

package openai

import (
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// Test message conversion roundtrip
func TestMessageConversionRoundtrip(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	tests := []struct {
		name    string
		content *genai.Content
	}{
		{
			name:    "simple text",
			content: genai.NewContentFromText("Hello world", "user"),
		},
		{
			name: "function call",
			content: genai.NewContentFromFunctionCall("get_weather", map[string]any{
				"location": "London",
			}, "model"),
		},
		{
			name: "function response",
			content: &genai.Content{
				Role: "function",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:   "call_test123", // Required ID
							Name: "get_weather",
							Response: map[string]any{
								"temperature": "20C",
								"condition":   "sunny",
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Convert to OpenAI format
			msgs, err := om.convertContent(tt.content)
			if err != nil {
				t.Fatalf("Failed to convert content: %v", err)
			}

			if len(msgs) == 0 {
				t.Fatal("No messages produced")
			}

			// For simple verification, just ensure we got messages
			for _, msg := range msgs {
				if msg.Role == "" {
					t.Error("Message role should not be empty")
				}
			}
		})
	}
}

// TestConvertToOpenAIMessages_Stateless tests the stateless conversion with system instruction,
// JSON mode, and multi-turn contents.
func TestConvertToOpenAIMessages_Stateless(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	t.Run("system instruction", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Hello", "user"),
			},
			Config: &genai.GenerateContentConfig{
				SystemInstruction: &genai.Content{
					Parts: []*genai.Part{
						genai.NewPartFromText("You are a helpful assistant."),
					},
				},
			},
		}

		msgs, err := om.convertToOpenAIMessages(req)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(msgs) != 2 {
			t.Fatalf("Expected 2 messages (system + user), got %d", len(msgs))
		}

		if msgs[0].Role != "system" {
			t.Errorf("Expected first message role 'system', got %q", msgs[0].Role)
		}
		if msgs[0].Content != "You are a helpful assistant." {
			t.Errorf("Unexpected system content: %v", msgs[0].Content)
		}
		if msgs[1].Role != "user" {
			t.Errorf("Expected second message role 'user', got %q", msgs[1].Role)
		}
	})

	t.Run("JSON mode adds instruction", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Give me data", "user"),
			},
			Config: &genai.GenerateContentConfig{
				ResponseMIMEType: "application/json",
			},
		}

		msgs, err := om.convertToOpenAIMessages(req)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(msgs) != 2 {
			t.Fatalf("Expected 2 messages (system JSON + user), got %d", len(msgs))
		}

		if msgs[0].Role != "system" {
			t.Errorf("Expected system message for JSON mode, got %q", msgs[0].Role)
		}
		sysContent, ok := msgs[0].Content.(string)
		if !ok || sysContent != "You must respond with valid JSON." {
			t.Errorf("Unexpected JSON mode system content: %v", msgs[0].Content)
		}
	})

	t.Run("multi-turn contents", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Hello", "user"),
				genai.NewContentFromText("Hi there!", "model"),
				genai.NewContentFromText("How are you?", "user"),
			},
			Config: &genai.GenerateContentConfig{},
		}

		msgs, err := om.convertToOpenAIMessages(req)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(msgs) != 3 {
			t.Fatalf("Expected 3 messages, got %d", len(msgs))
		}

		expectedRoles := []string{"user", "assistant", "user"}
		for i, role := range expectedRoles {
			if msgs[i].Role != role {
				t.Errorf("Message %d: expected role %q, got %q", i, role, msgs[i].Role)
			}
		}
	})

	t.Run("no contents", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: nil,
			Config:   &genai.GenerateContentConfig{},
		}

		msgs, err := om.convertToOpenAIMessages(req)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(msgs) != 0 {
			t.Errorf("Expected 0 messages for nil contents, got %d", len(msgs))
		}
	})
}
