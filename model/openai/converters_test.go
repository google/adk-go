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
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// Test that system prompt is always preserved at the beginning
func TestSystemPromptPreservation(t *testing.T) {
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: 5,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "test-session"

	// Add system message
	systemMsg := &OpenAIMessage{Role: "system", Content: "You are a helpful assistant"}
	om.addToHistory(sessionID, systemMsg)

	// Add many messages to trigger trim
	for i := 0; i < 20; i++ {
		om.addToHistory(sessionID, &OpenAIMessage{Role: "user", Content: "test"})
	}

	history := om.getConversationHistory(sessionID)
	if len(history) == 0 {
		t.Fatal("History is empty")
	}

	// Verify system message is still first
	if history[0].Role != "system" {
		t.Errorf("First message should be system, got %s", history[0].Role)
	}

	if content, ok := history[0].Content.(string); ok {
		if content != "You are a helpful assistant" {
			t.Errorf("System message content changed: %s", content)
		}
	}
}

// Test history trim on overflow
func TestHistoryTrimOnOverflow(t *testing.T) {
	maxLen := 10
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: maxLen,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "test-session"

	// Add more messages than max
	for i := 0; i < maxLen*2; i++ {
		om.addToHistory(sessionID, &OpenAIMessage{
			Role:    "user",
			Content: "message",
		})
	}

	history := om.getConversationHistory(sessionID)
	if len(history) > maxLen {
		t.Errorf("History length %d exceeds max %d", len(history), maxLen)
	}
}

// Test history trim with system prompt
func TestHistoryTrimWithSystemPrompt(t *testing.T) {
	maxLen := 5
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: maxLen,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "test-session"

	// Add system message first
	systemMsg := &OpenAIMessage{Role: "system", Content: "System prompt"}
	om.addToHistory(sessionID, systemMsg)

	// Add many more messages
	for i := 0; i < 20; i++ {
		om.addToHistory(sessionID, &OpenAIMessage{
			Role:    "user",
			Content: "user message",
		})
	}

	history := om.getConversationHistory(sessionID)

	// Should not exceed max length
	if len(history) > maxLen {
		t.Errorf("History length %d exceeds max %d", len(history), maxLen)
	}

	// System prompt should still be first
	if history[0].Role != "system" {
		t.Error("System prompt should be preserved at index 0")
	}

	// Verify we kept the most recent messages after system prompt
	// We should have 1 system + (maxLen-1) recent user messages
	expectedLen := maxLen
	if len(history) != expectedLen {
		t.Errorf("Expected %d messages, got %d", expectedLen, len(history))
	}
}

// Test concurrent access to conversation history (race conditions)
func TestConcurrentHistoryAccess(t *testing.T) {
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: 100,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "concurrent-test"

	var wg sync.WaitGroup
	numGoroutines := 10
	numOperations := 100

	// Concurrent writes
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				om.addToHistory(sessionID, &OpenAIMessage{
					Role:    "user",
					Content: "concurrent message",
				})
			}
		}(i)
	}

	// Concurrent reads
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				_ = om.getConversationHistory(sessionID)
			}
		}()
	}

	wg.Wait()

	// Verify history is not corrupted
	history := om.getConversationHistory(sessionID)
	if history == nil {
		t.Fatal("History should not be nil after concurrent operations")
	}

	// Should be trimmed to max length
	if len(history) > cfg.MaxHistoryLength {
		t.Errorf("History length %d exceeds max %d", len(history), cfg.MaxHistoryLength)
	}
}

// Test TTL expiration
func TestTTLExpiration(t *testing.T) {
	shortTTL := 100 * time.Millisecond
	cfg := &Config{
		BaseURL:    "http://localhost:1234/v1",
		SessionTTL: shortTTL,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "ttl-test"

	// Add message
	om.addToHistory(sessionID, &OpenAIMessage{
		Role:    "user",
		Content: "test message",
	})

	// Verify it exists
	history := om.getConversationHistory(sessionID)
	if history == nil || len(history) == 0 {
		t.Fatal("History should exist immediately after adding")
	}

	// Wait for TTL to expire
	time.Sleep(shortTTL * 2)

	// Should be nil now
	history = om.getConversationHistory(sessionID)
	if history != nil {
		t.Error("History should be nil after TTL expiration")
	}
}

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

// Test empty session ID handling
func TestEmptySessionID(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	// Try to add with empty session ID
	om.addToHistory("", &OpenAIMessage{Role: "user", Content: "test"})

	// Should return nil
	history := om.getConversationHistory("")
	if history != nil {
		t.Error("Empty session ID should return nil history")
	}
}

// Test multiple sessions isolation
func TestMultipleSessionsIsolation(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	// Create messages for different sessions
	session1 := "session-1"
	session2 := "session-2"

	msg1 := &OpenAIMessage{Role: "user", Content: "message for session 1"}
	msg2 := &OpenAIMessage{Role: "user", Content: "message for session 2"}

	om.addToHistory(session1, msg1)
	om.addToHistory(session2, msg2)

	// Verify isolation
	history1 := om.getConversationHistory(session1)
	history2 := om.getConversationHistory(session2)

	if len(history1) != 1 || len(history2) != 1 {
		t.Error("Each session should have exactly 1 message")
	}

	// Verify content is different
	content1, ok1 := history1[0].Content.(string)
	content2, ok2 := history2[0].Content.(string)

	if !ok1 || !ok2 {
		t.Fatal("Content should be strings")
	}

	if content1 == content2 {
		t.Error("Sessions should have different content")
	}

	if content1 != "message for session 1" {
		t.Errorf("Session 1 content incorrect: %s", content1)
	}

	if content2 != "message for session 2" {
		t.Errorf("Session 2 content incorrect: %s", content2)
	}
}

// Test clear history functionality
func TestClearHistory(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "clear-test"

	// Add messages
	for i := 0; i < 5; i++ {
		om.addToHistory(sessionID, &OpenAIMessage{
			Role:    "user",
			Content: "test",
		})
	}

	// Verify they exist
	history := om.getConversationHistory(sessionID)
	if len(history) != 5 {
		t.Fatalf("Expected 5 messages, got %d", len(history))
	}

	// Clear
	om.clearHistory(sessionID)

	// Verify cleared
	history = om.getConversationHistory(sessionID)
	if history != nil {
		t.Error("History should be nil after clearing")
	}
}

// Test request to OpenAI format conversion with session context
func TestConvertToOpenAIMessagesWithSession(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	// Create a context with session ID using proper helper
	ctx := WithSessionID(context.Background(), "test-session-123")

	req := &model.LLMRequest{
		Model: "test-model",
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello", "user"),
		},
		Config: &genai.GenerateContentConfig{},
		Tools:  make(map[string]any),
	}

	msgs, err := om.convertToOpenAIMessages(ctx, req)
	if err != nil {
		t.Fatalf("Failed to convert messages: %v", err)
	}

	if len(msgs) == 0 {
		t.Fatal("Should have at least one message")
	}

	// First call should have 1 message
	if len(msgs) != 1 {
		t.Errorf("Expected 1 message, got %d", len(msgs))
	}

	// Second call should accumulate with same session
	req2 := &model.LLMRequest{
		Model: "test-model",
		Contents: []*genai.Content{
			genai.NewContentFromText("World", "user"),
		},
		Config: &genai.GenerateContentConfig{},
		Tools:  make(map[string]any),
	}

	msgs2, err := om.convertToOpenAIMessages(ctx, req2)
	if err != nil {
		t.Fatalf("Failed to convert messages: %v", err)
	}

	// Should have accumulated messages (previous + new)
	if len(msgs2) < 2 {
		t.Errorf("Expected at least 2 accumulated messages, got %d", len(msgs2))
	}
}
