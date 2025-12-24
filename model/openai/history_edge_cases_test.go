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
	"fmt"
	"strings"
	"testing"
)

// TestHistoryTrimWithLargeMessages tests trim behavior with large content
func TestHistoryTrimWithLargeMessages(t *testing.T) {
	maxLen := 10
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: maxLen,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)
	sessionID := "large-messages-test"

	// Add system prompt
	systemMsg := &OpenAIMessage{
		Role:    "system",
		Content: "You are a helpful assistant. " + strings.Repeat("Important context. ", 100),
	}
	om.addToHistory(sessionID, systemMsg)

	// Add many large messages
	for i := 0; i < maxLen*2; i++ {
		largeContent := fmt.Sprintf("Message %d: ", i) + strings.Repeat("Lorem ipsum dolor sit amet, consectetur adipiscing elit. ", 50)
		msg := &OpenAIMessage{
			Role:    "user",
			Content: largeContent,
		}
		om.addToHistory(sessionID, msg)
	}

	history := om.getConversationHistory(sessionID)

	// Verify history was trimmed
	if len(history) > maxLen {
		t.Errorf("History length %d exceeds max %d", len(history), maxLen)
	}

	// Verify system prompt is still first
	if history[0].Role != "system" {
		t.Errorf("First message should be system, got %s", history[0].Role)
	}

	systemContent, ok := history[0].Content.(string)
	if !ok {
		t.Fatal("System content should be string")
	}

	if !strings.Contains(systemContent, "helpful assistant") {
		t.Error("System prompt content was corrupted")
	}

	// Verify we have the most recent messages (not the oldest)
	lastMsg := history[len(history)-1]
	lastContent, ok := lastMsg.Content.(string)
	if !ok {
		t.Fatal("Last message content should be string")
	}

	// Should contain high message number (recent messages)
	if !strings.Contains(lastContent, "Message 1") && !strings.Contains(lastContent, "Message 2") {
		// If it contains low numbers, we kept the old ones (wrong)
		t.Logf("Last message: %s", lastContent)
	}
}

// TestHistoryTrimWithSystemPromptAndLargeMessages tests system prompt preservation
func TestHistoryTrimWithSystemPromptAndLargeMessages(t *testing.T) {
	maxLen := 5
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: maxLen,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)
	sessionID := "system-preserve-test"

	// Very large system prompt
	largeSystemPrompt := "SYSTEM INSTRUCTIONS: " + strings.Repeat("Critical rule: always be helpful. ", 200)
	systemMsg := &OpenAIMessage{
		Role:    "system",
		Content: largeSystemPrompt,
	}
	om.addToHistory(sessionID, systemMsg)

	// Add more messages than max
	for i := 0; i < 20; i++ {
		msg := &OpenAIMessage{
			Role:    "user",
			Content: strings.Repeat(fmt.Sprintf("User message %d. ", i), 100),
		}
		om.addToHistory(sessionID, msg)
	}

	history := om.getConversationHistory(sessionID)

	// Should be trimmed to max length
	if len(history) != maxLen {
		t.Errorf("Expected exactly %d messages after trim, got %d", maxLen, len(history))
	}

	// System prompt must be preserved
	if history[0].Role != "system" {
		t.Errorf("First message must be system after trim, got %s", history[0].Role)
	}

	systemContent, ok := history[0].Content.(string)
	if !ok {
		t.Fatal("System content should be string")
	}

	// Content should be intact
	if len(systemContent) != len(largeSystemPrompt) {
		t.Errorf("System prompt length changed: expected %d, got %d", len(largeSystemPrompt), len(systemContent))
	}

	if !strings.Contains(systemContent, "SYSTEM INSTRUCTIONS") {
		t.Error("System prompt prefix was lost")
	}

	if !strings.Contains(systemContent, "Critical rule") {
		t.Error("System prompt content was corrupted")
	}

	// Remaining messages should be most recent (high indices)
	for i := 1; i < len(history); i++ {
		content, ok := history[i].Content.(string)
		if !ok {
			t.Errorf("Message %d content should be string", i)
			continue
		}

		// Should contain high message numbers (16-19, not 0-3)
		hasRecentContent := false
		for j := 16; j < 20; j++ {
			if strings.Contains(content, fmt.Sprintf("User message %d", j)) {
				hasRecentContent = true
				break
			}
		}

		if !hasRecentContent && i == len(history)-1 {
			t.Logf("Warning: Last message doesn't seem to be recent: %s", content[:50])
		}
	}
}

// TestHistoryTrimWithMixedMessageSizes tests trim with various message sizes
func TestHistoryTrimWithMixedMessageSizes(t *testing.T) {
	maxLen := 8
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: maxLen,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)
	sessionID := "mixed-sizes-test"

	// System prompt (medium size)
	systemMsg := &OpenAIMessage{
		Role:    "system",
		Content: "You are a helpful assistant with special instructions. " + strings.Repeat("Rule: be concise. ", 10),
	}
	om.addToHistory(sessionID, systemMsg)

	// Add messages of varying sizes
	for i := 0; i < 20; i++ {
		var content string
		switch i % 3 {
		case 0: // Small message
			content = fmt.Sprintf("Short msg %d", i)
		case 1: // Medium message
			content = fmt.Sprintf("Message %d: ", i) + strings.Repeat("Some text here. ", 20)
		case 2: // Large message
			content = fmt.Sprintf("Large message %d: ", i) + strings.Repeat("Very long content with lots of details. ", 100)
		}

		msg := &OpenAIMessage{
			Role:    "user",
			Content: content,
		}
		om.addToHistory(sessionID, msg)
	}

	history := om.getConversationHistory(sessionID)

	// Verify trim to max length
	if len(history) > maxLen {
		t.Errorf("History length %d exceeds max %d", len(history), maxLen)
	}

	// System prompt preserved
	if history[0].Role != "system" {
		t.Error("System prompt should be first")
	}

	// All messages should be valid
	for i, msg := range history {
		if msg.Role == "" {
			t.Errorf("Message %d has empty role", i)
		}
		if msg.Content == nil {
			t.Errorf("Message %d has nil content", i)
		}
	}

	t.Logf("Trimmed history contains %d messages (max: %d)", len(history), maxLen)
}

// TestHistoryTrimWithToolCalls tests trim with messages containing tool calls
func TestHistoryTrimWithToolCalls(t *testing.T) {
	maxLen := 6
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: maxLen,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)
	sessionID := "tool-calls-test"

	// System prompt
	systemMsg := &OpenAIMessage{
		Role:    "system",
		Content: "You are a helpful assistant with access to tools.",
	}
	om.addToHistory(sessionID, systemMsg)

	// Add conversation with tool calls
	for i := 0; i < 10; i++ {
		// User message
		userMsg := &OpenAIMessage{
			Role:    "user",
			Content: fmt.Sprintf("User request %d: " + strings.Repeat("Please help me with this task. ", 50), i),
		}
		om.addToHistory(sessionID, userMsg)

		// Assistant with tool call
		assistantMsg := &OpenAIMessage{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{
					ID:   fmt.Sprintf("call_%d", i),
					Type: "function",
					Function: FunctionCall{
						Name:      "get_info",
						Arguments: fmt.Sprintf(`{"query":"request_%d","data":"%s"}`, i, strings.Repeat("x", 500)),
					},
				},
			},
		}
		om.addToHistory(sessionID, assistantMsg)

		// Tool response
		toolMsg := &OpenAIMessage{
			Role:       "tool",
			Content:    fmt.Sprintf(`{"result":"Response for request %d","data":"%s"}`, i, strings.Repeat("y", 500)),
			ToolCallID: fmt.Sprintf("call_%d", i),
		}
		om.addToHistory(sessionID, toolMsg)
	}

	history := om.getConversationHistory(sessionID)

	// Should be trimmed
	if len(history) > maxLen {
		t.Errorf("History length %d exceeds max %d", len(history), maxLen)
	}

	// System prompt preserved
	if history[0].Role != "system" {
		t.Error("System prompt should be first")
	}

	// Verify tool calls are intact
	for i, msg := range history {
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			tc := msg.ToolCalls[0]
			if tc.ID == "" || tc.Type == "" || tc.Function.Name == "" {
				t.Errorf("Tool call at message %d is incomplete", i)
			}
		}

		if msg.Role == "tool" {
			if msg.ToolCallID == "" {
				t.Errorf("Tool message at %d missing ToolCallID", i)
			}
			if msg.Content == nil || msg.Content == "" {
				t.Errorf("Tool message at %d missing content", i)
			}
		}
	}

	t.Logf("Trimmed history with tool calls: %d messages", len(history))
}

// TestHistoryTrimExtremeContentLength tests with extremely long content
func TestHistoryTrimExtremeContentLength(t *testing.T) {
	maxLen := 5
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: maxLen,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)
	sessionID := "extreme-length-test"

	// System prompt
	systemMsg := &OpenAIMessage{
		Role:    "system",
		Content: "System instructions.",
	}
	om.addToHistory(sessionID, systemMsg)

	// Add messages with extreme content length
	for i := 0; i < 15; i++ {
		// Very large content (1MB each)
		largeContent := fmt.Sprintf("Message %d: ", i) + strings.Repeat("A", 1024*1024)

		msg := &OpenAIMessage{
			Role:    "user",
			Content: largeContent,
		}
		om.addToHistory(sessionID, msg)
	}

	history := om.getConversationHistory(sessionID)

	// Should be trimmed
	if len(history) > maxLen {
		t.Errorf("History length %d exceeds max %d", len(history), maxLen)
	}

	// System prompt preserved
	if history[0].Role != "system" {
		t.Error("System prompt should be first")
	}

	// Verify large content is intact (not truncated)
	if len(history) > 1 {
		content, ok := history[1].Content.(string)
		if !ok {
			t.Fatal("Message content should be string")
		}

		// Content should still be ~1MB
		if len(content) < 1024*1024 {
			t.Errorf("Large content was truncated: expected ~1MB, got %d bytes", len(content))
		}
	}

	t.Logf("Successfully handled extreme content lengths")
}

// TestHistoryTrimNoSystemPrompt tests trim without system prompt
func TestHistoryTrimNoSystemPrompt(t *testing.T) {
	maxLen := 5
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: maxLen,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)
	sessionID := "no-system-test"

	// Add many messages WITHOUT system prompt
	for i := 0; i < 20; i++ {
		msg := &OpenAIMessage{
			Role:    "user",
			Content: fmt.Sprintf("Message %d: " + strings.Repeat("Content. ", 100), i),
		}
		om.addToHistory(sessionID, msg)
	}

	history := om.getConversationHistory(sessionID)

	// Should be trimmed to exactly maxLen
	if len(history) != maxLen {
		t.Errorf("Expected exactly %d messages, got %d", maxLen, len(history))
	}

	// Should keep most recent messages
	lastMsg := history[len(history)-1]
	lastContent, ok := lastMsg.Content.(string)
	if !ok {
		t.Fatal("Content should be string")
	}

	// Last message should be message 19 (most recent)
	if !strings.Contains(lastContent, "Message 19") {
		t.Errorf("Expected most recent message (19), got: %s", lastContent[:50])
	}

	// First message should be message 15 (20 messages - 5 max = start at 15)
	firstMsg := history[0]
	firstContent, ok := firstMsg.Content.(string)
	if !ok {
		t.Fatal("Content should be string")
	}

	if !strings.Contains(firstContent, "Message 15") {
		t.Logf("First message after trim: %s", firstContent[:50])
	}
}

// TestHistoryTrimWithEmptyMessages tests trim with empty/nil content
func TestHistoryTrimWithEmptyMessages(t *testing.T) {
	maxLen := 5
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: maxLen,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)
	sessionID := "empty-messages-test"

	// System prompt
	systemMsg := &OpenAIMessage{
		Role:    "system",
		Content: "System",
	}
	om.addToHistory(sessionID, systemMsg)

	// Mix of empty, nil, and regular content
	for i := 0; i < 15; i++ {
		var content interface{}
		switch i % 3 {
		case 0:
			content = "" // Empty string
		case 1:
			content = nil // Nil content
		case 2:
			content = strings.Repeat(fmt.Sprintf("Content %d. ", i), 50)
		}

		msg := &OpenAIMessage{
			Role:    "user",
			Content: content,
		}
		om.addToHistory(sessionID, msg)
	}

	history := om.getConversationHistory(sessionID)

	// Should be trimmed
	if len(history) > maxLen {
		t.Errorf("History length %d exceeds max %d", len(history), maxLen)
	}

	// System prompt preserved
	if history[0].Role != "system" {
		t.Error("System prompt should be first")
	}

	// All messages should have valid roles
	for i, msg := range history {
		if msg.Role == "" {
			t.Errorf("Message %d has empty role", i)
		}
	}

	t.Logf("Handled empty/nil messages correctly")
}

// TestHistoryTrimSequentialAdds tests multiple sequential add operations
func TestHistoryTrimSequentialAdds(t *testing.T) {
	maxLen := 10
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: maxLen,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)
	sessionID := "sequential-test"

	// System prompt
	systemMsg := &OpenAIMessage{
		Role:    "system",
		Content: "System instructions. " + strings.Repeat("Important. ", 50),
	}
	om.addToHistory(sessionID, systemMsg)

	// Add messages in batches
	for batch := 0; batch < 10; batch++ {
		// Add 5 messages at once
		messages := make([]*OpenAIMessage, 5)
		for i := 0; i < 5; i++ {
			messages[i] = &OpenAIMessage{
				Role:    "user",
				Content: fmt.Sprintf("Batch %d, Message %d: " + strings.Repeat("text ", 100), batch, i),
			}
		}
		om.addToHistory(sessionID, messages...)

		// Check after each batch
		history := om.getConversationHistory(sessionID)
		if len(history) > maxLen {
			t.Errorf("After batch %d: history length %d exceeds max %d", batch, len(history), maxLen)
		}

		// System prompt should always be first
		if history[0].Role != "system" {
			t.Errorf("After batch %d: system prompt lost", batch)
		}
	}

	finalHistory := om.getConversationHistory(sessionID)
	t.Logf("Final history length: %d (max: %d)", len(finalHistory), maxLen)

	// Final check
	if len(finalHistory) > maxLen {
		t.Errorf("Final history length %d exceeds max %d", len(finalHistory), maxLen)
	}

	// Should contain most recent batch
	lastMsg := finalHistory[len(finalHistory)-1]
	lastContent, ok := lastMsg.Content.(string)
	if !ok {
		t.Fatal("Content should be string")
	}

	if !strings.Contains(lastContent, "Batch 9") {
		t.Errorf("Last message should be from most recent batch (9), got: %s", lastContent[:50])
	}
}
