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

func TestBuildChatRequest(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	temp := float32(0.7)
	topP := float32(0.9)
	presP := float32(0.1)
	freqP := float32(0.2)
	seed := int32(42)
	logprobs := int32(5)

	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			Temperature:      &temp,
			TopP:             &topP,
			MaxOutputTokens:  1024,
			StopSequences:    []string{"STOP"},
			PresencePenalty:  &presP,
			FrequencyPenalty: &freqP,
			Seed:             &seed,
			CandidateCount:   2,
			ResponseLogprobs: true,
			Logprobs:         &logprobs,
		},
		Tools: map[string]any{
			"my_tool": map[string]any{
				"description": "A test tool",
			},
		},
	}

	messages := []OpenAIMessage{{Role: "user", Content: "hello"}}

	chatReq := om.buildChatRequest(messages, false, req)

	if chatReq.Model != "test-model" {
		t.Errorf("Expected model 'test-model', got %q", chatReq.Model)
	}
	if chatReq.Stream {
		t.Error("Expected stream=false")
	}
	if *chatReq.Temperature != 0.7 {
		t.Errorf("Expected temperature 0.7, got %v", *chatReq.Temperature)
	}
	if *chatReq.TopP != 0.9 {
		t.Errorf("Expected topP 0.9, got %v", *chatReq.TopP)
	}
	if *chatReq.MaxTokens != 1024 {
		t.Errorf("Expected maxTokens 1024, got %v", *chatReq.MaxTokens)
	}
	if len(chatReq.Stop) != 1 || chatReq.Stop[0] != "STOP" {
		t.Errorf("Expected stop=['STOP'], got %v", chatReq.Stop)
	}
	if *chatReq.PresencePenalty != 0.1 {
		t.Errorf("Expected presencePenalty 0.1, got %v", *chatReq.PresencePenalty)
	}
	if *chatReq.FrequencyPenalty != 0.2 {
		t.Errorf("Expected frequencyPenalty 0.2, got %v", *chatReq.FrequencyPenalty)
	}
	if *chatReq.Seed != 42 {
		t.Errorf("Expected seed 42, got %v", *chatReq.Seed)
	}
	if chatReq.N != 2 {
		t.Errorf("Expected N=2, got %d", chatReq.N)
	}
	if !chatReq.Logprobs {
		t.Error("Expected logprobs=true")
	}
	if *chatReq.TopLogprobs != 5 {
		t.Errorf("Expected topLogprobs=5, got %v", *chatReq.TopLogprobs)
	}
	if len(chatReq.Tools) != 1 {
		t.Errorf("Expected 1 tool, got %d", len(chatReq.Tools))
	}
	if chatReq.ToolChoice != "auto" {
		t.Errorf("Expected toolChoice='auto', got %v", chatReq.ToolChoice)
	}

	// Test stream=true
	chatReqStream := om.buildChatRequest(messages, true, req)
	if !chatReqStream.Stream {
		t.Error("Expected stream=true for streaming request")
	}

	// Test nil config
	reqNoConfig := &model.LLMRequest{Config: nil}
	chatReqNoConfig := om.buildChatRequest(messages, false, reqNoConfig)
	if chatReqNoConfig.Temperature != nil {
		t.Error("Expected nil temperature with nil config")
	}
}
