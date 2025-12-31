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
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestNewModel(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		cfg       *Config
		wantErr   bool
	}{
		{
			name:      "valid config",
			modelName: "gpt-4",
			cfg: &Config{
				BaseURL: "http://localhost:1234/v1",
			},
			wantErr: false,
		},
		{
			name:      "nil config",
			modelName: "gpt-4",
			cfg:       nil,
			wantErr:   true,
		},
		{
			name:      "empty base url",
			modelName: "gpt-4",
			cfg: &Config{
				BaseURL: "",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewModel(tt.modelName, tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewModel() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && m == nil {
				t.Error("NewModel() returned nil model")
			}
			if !tt.wantErr && m.Name() != tt.modelName {
				t.Errorf("NewModel().Name() = %v, want %v", m.Name(), tt.modelName)
			}
		})
	}
}

func TestConversationHistoryManagement(t *testing.T) {
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: 5,
		SessionTTL:       1 * time.Second,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	// Test adding to history
	sessionID := "test-session"
	msg1 := &OpenAIMessage{Role: "user", Content: "Hello"}
	msg2 := &OpenAIMessage{Role: "assistant", Content: "Hi there"}

	om.addToHistory(sessionID, msg1, msg2)

	history := om.getConversationHistory(sessionID)
	if len(history) != 2 {
		t.Errorf("Expected 2 messages in history, got %d", len(history))
	}

	// Test max history length
	for i := 0; i < 10; i++ {
		om.addToHistory(sessionID, &OpenAIMessage{Role: "user", Content: "test"})
	}

	history = om.getConversationHistory(sessionID)
	if len(history) > cfg.MaxHistoryLength {
		t.Errorf("History length %d exceeds max %d", len(history), cfg.MaxHistoryLength)
	}

	// Test TTL expiration
	time.Sleep(2 * time.Second)
	history = om.getConversationHistory(sessionID)
	if history != nil {
		t.Error("Expected history to be nil after TTL expiration")
	}

	// Test clear history
	om.addToHistory(sessionID, msg1)
	om.clearHistory(sessionID)
	history = om.getConversationHistory(sessionID)
	if history != nil {
		t.Error("Expected history to be nil after clearing")
	}
}

func TestGenerateContent(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("Expected path /chat/completions, got %s", r.URL.Path)
		}

		var req ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("Failed to decode request: %v", err)
			http.Error(w, "Bad request", http.StatusBadRequest)
			return
		}

		// Send mock response
		resp := ChatCompletionResponse{
			ID:      "test-id",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   req.Model,
			Choices: []Choice{
				{
					Index: 0,
					Message: OpenAIMessage{
						Role:    "assistant",
						Content: "This is a test response",
					},
					FinishReason: "stop",
				},
			},
			Usage: Usage{
				PromptTokens:     10,
				CompletionTokens: 20,
				TotalTokens:      30,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := &Config{
		BaseURL: server.URL,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	// Create test request
	req := &model.LLMRequest{
		Model: "test-model",
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello, how are you?", "user"),
		},
		Config: &genai.GenerateContentConfig{},
		Tools:  make(map[string]any),
	}

	ctx := context.Background()

	// Test non-streaming
	var responses []*model.LLMResponse
	for resp, err := range m.GenerateContent(ctx, req, false) {
		if err != nil {
			t.Fatalf("GenerateContent error: %v", err)
		}
		responses = append(responses, resp)
	}

	if len(responses) != 1 {
		t.Errorf("Expected 1 response, got %d", len(responses))
	}

	if len(responses) > 0 {
		resp := responses[0]
		if resp.Content == nil {
			t.Error("Response content is nil")
		}
		if !resp.TurnComplete {
			t.Error("Expected TurnComplete to be true")
		}
		if resp.UsageMetadata == nil {
			t.Error("Expected usage metadata")
		}
	}
}

func TestToolCalling(t *testing.T) {
	// Create mock server that returns tool calls
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)

		resp := ChatCompletionResponse{
			ID:      "test-id",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   req.Model,
			Choices: []Choice{
				{
					Index: 0,
					Message: OpenAIMessage{
						Role: "assistant",
						ToolCalls: []ToolCall{
							{
								ID:   "call_test",
								Type: "function",
								Function: FunctionCall{
									Name:      "get_weather",
									Arguments: `{"location":"London"}`,
								},
							},
						},
					},
					FinishReason: "tool_calls",
				},
			},
			Usage: Usage{
				PromptTokens:     10,
				CompletionTokens: 20,
				TotalTokens:      30,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := &Config{
		BaseURL: server.URL,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	// Create test request with tools
	req := &model.LLMRequest{
		Model: "test-model",
		Contents: []*genai.Content{
			genai.NewContentFromText("What's the weather in London?", "user"),
		},
		Config: &genai.GenerateContentConfig{},
		Tools: map[string]any{
			"get_weather": map[string]any{
				"description": "Get current weather",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "City name",
						},
					},
				},
			},
		},
	}

	ctx := context.Background()

	var responses []*model.LLMResponse
	for resp, err := range m.GenerateContent(ctx, req, false) {
		if err != nil {
			t.Fatalf("GenerateContent error: %v", err)
		}
		responses = append(responses, resp)
	}

	if len(responses) != 1 {
		t.Errorf("Expected 1 response, got %d", len(responses))
	}

	if len(responses) > 0 {
		resp := responses[0]
		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			t.Error("Expected response with parts")
		}

		// Check for function call part
		foundFunctionCall := false
		for _, part := range resp.Content.Parts {
			if part.FunctionCall != nil {
				foundFunctionCall = true
				if part.FunctionCall.Name != "get_weather" {
					t.Errorf("Expected function name 'get_weather', got '%s'", part.FunctionCall.Name)
				}
			}
		}

		if !foundFunctionCall {
			t.Error("Expected function call in response")
		}
	}
}

func TestConvertContent(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	tests := []struct {
		name     string
		content  *genai.Content
		wantMsgs int
		wantErr  bool
	}{
		{
			name:     "nil content",
			content:  nil,
			wantMsgs: 0,
			wantErr:  false,
		},
		{
			name:     "text content",
			content:  genai.NewContentFromText("Hello", "user"),
			wantMsgs: 1,
			wantErr:  false,
		},
		{
			name:     "function call",
			content:  genai.NewContentFromFunctionCall("test_func", map[string]any{"arg": "value"}, "model"),
			wantMsgs: 1,
			wantErr:  false,
		},
		{
			name: "function response",
			content: &genai.Content{
				Role: "function",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       "call_test456", // Required ID
							Name:     "test_func",
							Response: map[string]any{"result": "ok"},
						},
					},
				},
			},
			wantMsgs: 1,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msgs, err := om.convertContent(tt.content)
			if (err != nil) != tt.wantErr {
				t.Errorf("convertContent() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if len(msgs) != tt.wantMsgs {
				t.Errorf("convertContent() returned %d messages, want %d", len(msgs), tt.wantMsgs)
			}
		})
	}
}
