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
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"testing"
)

// Test validateMessage with valid messages
func TestValidateMessage_Valid(t *testing.T) {
	tests := []struct {
		name string
		msg  *OpenAIMessage
	}{
		{
			name: "valid user message",
			msg: &OpenAIMessage{
				Role:    "user",
				Content: "Hello, how are you?",
			},
		},
		{
			name: "valid assistant message",
			msg: &OpenAIMessage{
				Role:    "assistant",
				Content: "I'm doing well, thank you!",
			},
		},
		{
			name: "valid system message",
			msg: &OpenAIMessage{
				Role:    "system",
				Content: "You are a helpful assistant",
			},
		},
		{
			name: "valid tool message",
			msg: &OpenAIMessage{
				Role:       "tool",
				Content:    `{"temperature": "20C"}`,
				ToolCallID: "call_get_weather",
			},
		},
		{
			name: "valid assistant with tool calls",
			msg: &OpenAIMessage{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location":"London"}`,
						},
					},
				},
			},
		},
		{
			name: "assistant with tool calls and content",
			msg: &OpenAIMessage{
				Role:    "assistant",
				Content: "Let me check the weather for you.",
				ToolCalls: []ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location":"London"}`,
						},
					},
				},
			},
		},
		{
			name: "assistant with multiple tool calls",
			msg: &OpenAIMessage{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location":"London"}`,
						},
					},
					{
						ID:   "call_2",
						Type: "function",
						Function: FunctionCall{
							Name:      "get_time",
							Arguments: `{}`,
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateMessage(tt.msg)
			if err != nil {
				t.Errorf("Expected valid message, got error: %v", err)
			}
		})
	}
}

// Test validateMessage with invalid messages
func TestValidateMessage_Invalid(t *testing.T) {
	tests := []struct {
		name        string
		msg         *OpenAIMessage
		expectedErr string
	}{
		{
			name:        "nil message",
			msg:         nil,
			expectedErr: "message cannot be nil",
		},
		{
			name:        "empty role",
			msg:         &OpenAIMessage{Content: "Hello"},
			expectedErr: "message role cannot be empty",
		},
		{
			name: "tool message without ToolCallID",
			msg: &OpenAIMessage{
				Role:    "tool",
				Content: `{"result": "success"}`,
			},
			expectedErr: "tool role message must have ToolCallID",
		},
		{
			name: "tool message without content",
			msg: &OpenAIMessage{
				Role:       "tool",
				ToolCallID: "call_123",
			},
			expectedErr: "tool role message must have content",
		},
		{
			name: "tool call without ID",
			msg: &OpenAIMessage{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{
						Type: "function",
						Function: FunctionCall{
							Name:      "get_weather",
							Arguments: `{}`,
						},
					},
				},
			},
			expectedErr: "tool call at index 0 must have an ID",
		},
		{
			name: "tool call without type",
			msg: &OpenAIMessage{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{
						ID: "call_123",
						Function: FunctionCall{
							Name:      "get_weather",
							Arguments: `{}`,
						},
					},
				},
			},
			expectedErr: "tool call at index 0 must have a type",
		},
		{
			name: "tool call without function name",
			msg: &OpenAIMessage{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: FunctionCall{
							Arguments: `{}`,
						},
					},
				},
			},
			expectedErr: "tool call at index 0 must have a function name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateMessage(tt.msg)
			if err == nil {
				t.Errorf("Expected error containing '%s', got nil", tt.expectedErr)
				return
			}
			if !strings.Contains(err.Error(), tt.expectedErr) {
				t.Errorf("Expected error containing '%s', got '%s'", tt.expectedErr, err.Error())
			}
		})
	}
}

// Test addToHistory with validation
func TestAddToHistory_WithValidation(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
		Logger:  log.New(os.Stdout, "[TEST] ", log.LstdFlags),
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "validation-test"

	// Add valid message
	validMsg := &OpenAIMessage{
		Role:    "user",
		Content: "Hello",
	}
	om.addToHistory(sessionID, validMsg)

	history := om.getConversationHistory(sessionID)
	if len(history) != 1 {
		t.Errorf("Expected 1 message in history, got %d", len(history))
	}

	// Try to add invalid message (should be skipped)
	invalidMsg := &OpenAIMessage{
		Role: "tool",
		// Missing ToolCallID and Content
	}
	om.addToHistory(sessionID, invalidMsg)

	history = om.getConversationHistory(sessionID)
	if len(history) != 1 {
		t.Errorf("Expected 1 message (invalid should be skipped), got %d", len(history))
	}
}

// Test addToHistory with mixed valid and invalid messages
func TestAddToHistory_MixedMessages(t *testing.T) {
	var logBuf strings.Builder
	logger := log.New(&logBuf, "", 0)

	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
		Logger:  logger,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "mixed-test"

	messages := []*OpenAIMessage{
		// Valid
		{Role: "user", Content: "Hello"},
		// Invalid (no role)
		{Content: "Invalid"},
		// Valid
		{Role: "assistant", Content: "Hi there"},
		// Invalid (tool without ToolCallID)
		{Role: "tool", Content: "result"},
		// Valid
		{Role: "user", Content: "How are you?"},
	}

	om.addToHistory(sessionID, messages...)

	history := om.getConversationHistory(sessionID)
	// Should have 3 valid messages
	if len(history) != 3 {
		t.Errorf("Expected 3 valid messages, got %d", len(history))
	}

	// Check that warnings were logged
	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "WARNING") {
		t.Error("Expected warnings in log for invalid messages")
	}
	if !strings.Contains(logOutput, "Invalid message skipped") {
		t.Error("Expected 'Invalid message skipped' in log")
	}
}

// Test tool call validation
func TestValidateMessage_ToolCallValidation(t *testing.T) {
	// Valid tool call
	validTC := &OpenAIMessage{
		Role: "assistant",
		ToolCalls: []ToolCall{
			{
				ID:   "call_abc123",
				Type: "function",
				Function: FunctionCall{
					Name:      "get_weather",
					Arguments: `{"location": "London"}`,
				},
			},
		},
	}

	if err := validateMessage(validTC); err != nil {
		t.Errorf("Valid tool call should pass validation: %v", err)
	}

	// Tool call with empty arguments (valid - some functions have no params)
	emptyArgs := &OpenAIMessage{
		Role: "assistant",
		ToolCalls: []ToolCall{
			{
				ID:   "call_xyz",
				Type: "function",
				Function: FunctionCall{
					Name:      "get_current_time",
					Arguments: "",
				},
			},
		},
	}

	if err := validateMessage(emptyArgs); err != nil {
		t.Errorf("Tool call with empty arguments should be valid: %v", err)
	}
}

// Test that existing message creations in converters are valid
func TestConverters_MessageCreationValid(t *testing.T) {
	// Test tool message creation (as in converters.go:117-121)
	toolMsg := &OpenAIMessage{
		Role:       "tool",
		Content:    `{"temperature": "20C", "condition": "sunny"}`,
		ToolCallID: "call_get_weather",
	}

	if err := validateMessage(toolMsg); err != nil {
		t.Errorf("Tool message from converters should be valid: %v", err)
	}

	// Test assistant with tool calls (as in converters.go:151-158)
	assistantMsg := &OpenAIMessage{
		Role: "assistant",
		ToolCalls: []ToolCall{
			{
				ID:   "call_123",
				Type: "function",
				Function: FunctionCall{
					Name:      "get_weather",
					Arguments: `{"location":"London"}`,
				},
			},
		},
	}

	if err := validateMessage(assistantMsg); err != nil {
		t.Errorf("Assistant message with tool calls should be valid: %v", err)
	}

	// Test regular text message (as in converters.go:161-164)
	textMsg := &OpenAIMessage{
		Role:    "user",
		Content: "What's the weather like?",
	}

	if err := validateMessage(textMsg); err != nil {
		t.Errorf("Regular text message should be valid: %v", err)
	}
}

// Test addToHistory with invalid messages - comprehensive integration test
func TestAddToHistoryWithInvalidMsg(t *testing.T) {
	var logBuf strings.Builder
	logger := log.New(&logBuf, "", 0)

	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
		Logger:  logger,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "invalid-test"

	// Add mix of valid and invalid messages
	messages := []*OpenAIMessage{
		// Valid message 1
		{Role: "user", Content: "Valid message 1"},

		// Invalid - no role
		{Role: "", Content: "Invalid - no role"},

		// Invalid - tool without ToolCallID
		{Role: "tool", Content: "Invalid - no ToolCallID"},

		// Valid message 2
		{Role: "user", Content: "Valid message 2"},

		// Invalid - tool without content
		{Role: "tool", ToolCallID: "call_123"},

		// Invalid - nil message
		nil,

		// Valid message 3
		{Role: "assistant", Content: "Valid message 3"},

		// Invalid - tool call without ID
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{
					Type: "function",
					Function: FunctionCall{
						Name:      "test",
						Arguments: `{}`,
					},
				},
			},
		},

		// Invalid - tool call without Type
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{
					ID: "call_456",
					Function: FunctionCall{
						Name:      "test",
						Arguments: `{}`,
					},
				},
			},
		},

		// Invalid - tool call without function name
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{
					ID:   "call_789",
					Type: "function",
					Function: FunctionCall{
						Arguments: `{}`,
					},
				},
			},
		},

		// Valid message 4 - tool call with all required fields
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{
					ID:   "call_valid",
					Type: "function",
					Function: FunctionCall{
						Name:      "get_weather",
						Arguments: `{"location":"London"}`,
					},
				},
			},
		},
	}

	om.addToHistory(sessionID, messages...)

	history := om.getConversationHistory(sessionID)

	// Should only have 4 valid messages
	if len(history) != 4 {
		t.Errorf("Expected 4 valid messages, got %d", len(history))
		for i, msg := range history {
			t.Logf("  History[%d]: Role=%s, Content=%v, ToolCalls=%d",
				i, msg.Role, msg.Content, len(msg.ToolCalls))
		}
	}

	// Check warnings were logged
	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "WARNING") {
		t.Error("Expected warnings for invalid messages")
	}
	if !strings.Contains(logOutput, "Invalid message skipped") {
		t.Error("Expected 'Invalid message skipped' message in log")
	}

	// Count warnings - should have 7 invalid messages (including nil)
	warningCount := strings.Count(logOutput, "WARNING")
	if warningCount != 7 {
		t.Errorf("Expected 7 warnings, got %d. Log output:\n%s", warningCount, logOutput)
	}

	// Verify only valid messages in history
	for i, msg := range history {
		if err := validateMessage(msg); err != nil {
			t.Errorf("Message %d in history is invalid: %v", i, err)
		}
	}

	// Verify the specific valid messages
	expectedRoles := []string{"user", "user", "assistant", "assistant"}
	for i, expectedRole := range expectedRoles {
		if i >= len(history) {
			t.Errorf("Missing message at index %d", i)
			continue
		}
		if history[i].Role != expectedRole {
			t.Errorf("Message %d: expected role %s, got %s", i, expectedRole, history[i].Role)
		}
	}

	// Verify specific content for text messages
	if history[0].Content != "Valid message 1" {
		t.Errorf("First message content mismatch: %v", history[0].Content)
	}
	if history[1].Content != "Valid message 2" {
		t.Errorf("Second message content mismatch: %v", history[1].Content)
	}
	if history[2].Content != "Valid message 3" {
		t.Errorf("Third message content mismatch: %v", history[2].Content)
	}

	// Verify the last message has valid tool call
	if len(history[3].ToolCalls) != 1 {
		t.Errorf("Last message should have 1 tool call, got %d", len(history[3].ToolCalls))
	}
	if len(history[3].ToolCalls) > 0 {
		tc := history[3].ToolCalls[0]
		if tc.ID != "call_valid" {
			t.Errorf("Tool call ID mismatch: %s", tc.ID)
		}
		if tc.Function.Name != "get_weather" {
			t.Errorf("Tool call function name mismatch: %s", tc.Function.Name)
		}
	}
}

// TestValidateMultimodalPrep tests future-proof support for multimodal content.
// OpenAI API supports Content as []interface{} for vision models with text + image_url.
// This test verifies that our validation and message handling can support this format.
func TestValidateMultimodalPrep(t *testing.T) {
	// Define multimodal content part types (as per OpenAI API spec)
	// https://platform.openai.com/docs/guides/vision
	type ContentPart struct {
		Type     string                 `json:"type"`      // "text" or "image_url"
		Text     string                 `json:"text,omitempty"`
		ImageURL map[string]interface{} `json:"image_url,omitempty"`
	}

	tests := []struct {
		name        string
		msg         *OpenAIMessage
		expectValid bool
		description string
	}{
		{
			name: "text only - string content (current standard)",
			msg: &OpenAIMessage{
				Role:    "user",
				Content: "What's in this image?",
			},
			expectValid: true,
			description: "Current format: simple string content",
		},
		{
			name: "multimodal - text + image_url (future format)",
			msg: &OpenAIMessage{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "What's in this image?",
					},
					map[string]interface{}{
						"type": "image_url",
						"image_url": map[string]interface{}{
							"url": "https://example.com/image.jpg",
						},
					},
				},
			},
			expectValid: true,
			description: "Future format: array of content parts (text + image)",
		},
		{
			name: "multimodal - structured with ContentPart",
			msg: &OpenAIMessage{
				Role: "user",
				Content: []ContentPart{
					{
						Type: "text",
						Text: "Describe this image in detail",
					},
					{
						Type: "image_url",
						ImageURL: map[string]interface{}{
							"url":    "https://example.com/photo.png",
							"detail": "high",
						},
					},
				},
			},
			expectValid: true,
			description: "Future format: typed ContentPart structs",
		},
		{
			name: "multimodal - multiple images",
			msg: &OpenAIMessage{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "Compare these two images",
					},
					map[string]interface{}{
						"type": "image_url",
						"image_url": map[string]interface{}{
							"url": "https://example.com/image1.jpg",
						},
					},
					map[string]interface{}{
						"type": "image_url",
						"image_url": map[string]interface{}{
							"url": "https://example.com/image2.jpg",
						},
					},
				},
			},
			expectValid: true,
			description: "Future format: multiple images in one message",
		},
		{
			name: "multimodal - base64 encoded image",
			msg: &OpenAIMessage{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "What do you see?",
					},
					map[string]interface{}{
						"type": "image_url",
						"image_url": map[string]interface{}{
							"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
						},
					},
				},
			},
			expectValid: true,
			description: "Future format: base64 encoded image data",
		},
		{
			name: "assistant with multimodal content",
			msg: &OpenAIMessage{
				Role: "assistant",
				Content: []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "I can see a beautiful landscape with mountains and a lake.",
					},
				},
			},
			expectValid: true,
			description: "Assistant can also use multimodal format",
		},
		{
			name: "empty content array",
			msg: &OpenAIMessage{
				Role:    "user",
				Content: []interface{}{},
			},
			expectValid: true,
			description: "Empty array is technically valid (though not useful)",
		},
		{
			name: "numeric content (edge case)",
			msg: &OpenAIMessage{
				Role:    "user",
				Content: 12345,
			},
			expectValid: true,
			description: "interface{} allows any type - numbers should not break validation",
		},
		{
			name: "nil content (should be valid for assistant with tool calls)",
			msg: &OpenAIMessage{
				Role:    "assistant",
				Content: nil,
				ToolCalls: []ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location":"London"}`,
						},
					},
				},
			},
			expectValid: true,
			description: "nil content is valid when tool_calls are present",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateMessage(tt.msg)

			if tt.expectValid && err != nil {
				t.Errorf("Expected valid message but got error: %v", err)
				t.Logf("Description: %s", tt.description)
			}
			if !tt.expectValid && err == nil {
				t.Errorf("Expected validation error but message was valid")
				t.Logf("Description: %s", tt.description)
			}

			if tt.expectValid {
				t.Logf("✓ %s", tt.description)
			}
		})
	}
}

// TestMultimodalMessageSerialization tests JSON serialization of multimodal content.
// This ensures that Content as interface{} properly serializes to JSON.
func TestMultimodalMessageSerialization(t *testing.T) {
	tests := []struct {
		name     string
		msg      *OpenAIMessage
		wantJSON string
	}{
		{
			name: "simple text content",
			msg: &OpenAIMessage{
				Role:    "user",
				Content: "Hello",
			},
			wantJSON: `{"role":"user","content":"Hello"}`,
		},
		{
			name: "multimodal content array",
			msg: &OpenAIMessage{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "Test",
					},
					map[string]interface{}{
						"type": "image_url",
						"image_url": map[string]interface{}{
							"url": "https://example.com/img.jpg",
						},
					},
				},
			},
			wantJSON: `{"role":"user","content":[{"text":"Test","type":"text"},{"image_url":{"url":"https://example.com/img.jpg"},"type":"image_url"}]}`,
		},
		{
			name: "nil content omitted",
			msg: &OpenAIMessage{
				Role:    "assistant",
				Content: nil,
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: FunctionCall{
							Name:      "test",
							Arguments: `{}`,
						},
					},
				},
			},
			// content field should be omitted when nil
			wantJSON: `{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"test","arguments":"{}"}}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal to JSON
			jsonBytes, err := json.Marshal(tt.msg)
			if err != nil {
				t.Fatalf("Failed to marshal message: %v", err)
			}

			gotJSON := string(jsonBytes)

			// Unmarshal both expected and got to compare structure
			// (this handles field ordering differences)
			var expected, got interface{}
			if err := json.Unmarshal([]byte(tt.wantJSON), &expected); err != nil {
				t.Fatalf("Failed to unmarshal expected JSON: %v", err)
			}
			if err := json.Unmarshal(jsonBytes, &got); err != nil {
				t.Fatalf("Failed to unmarshal generated JSON: %v", err)
			}

			// Compare as strings for deep equality
			expectedStr := fmt.Sprintf("%v", expected)
			gotStr := fmt.Sprintf("%v", got)

			if expectedStr != gotStr {
				t.Errorf("JSON mismatch:\nExpected: %s\nGot:      %s", tt.wantJSON, gotJSON)
			} else {
				t.Logf("✓ JSON serialization correct: %s", gotJSON)
			}
		})
	}
}

// TestMultimodalContentInHistory tests that multimodal messages work with conversation history.
func TestMultimodalContentInHistory(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
		Logger:  log.New(os.Stdout, "[MULTIMODAL-TEST] ", log.LstdFlags),
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "multimodal-session"

	// Add a multimodal message to history
	multimodalMsg := &OpenAIMessage{
		Role: "user",
		Content: []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": "What's in this image?",
			},
			map[string]interface{}{
				"type": "image_url",
				"image_url": map[string]interface{}{
					"url": "https://example.com/test.jpg",
				},
			},
		},
	}

	om.addToHistory(sessionID, multimodalMsg)

	// Retrieve and verify
	history := om.getConversationHistory(sessionID)
	if history == nil || len(history) != 1 {
		t.Fatalf("Expected 1 message in history, got %v", history)
	}

	// Verify content is preserved
	if history[0].Role != "user" {
		t.Errorf("Expected role 'user', got '%s'", history[0].Role)
	}

	// Verify content can be type-asserted back to array
	contentArray, ok := history[0].Content.([]interface{})
	if !ok {
		t.Fatalf("Expected Content to be []interface{}, got %T", history[0].Content)
	}

	if len(contentArray) != 2 {
		t.Errorf("Expected 2 content parts, got %d", len(contentArray))
	}

	// Verify first part is text
	firstPart, ok := contentArray[0].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected first part to be map[string]interface{}, got %T", contentArray[0])
	}

	if firstPart["type"] != "text" {
		t.Errorf("Expected type 'text', got '%v'", firstPart["type"])
	}
	if firstPart["text"] != "What's in this image?" {
		t.Errorf("Text content mismatch: %v", firstPart["text"])
	}

	// Verify second part is image_url
	secondPart, ok := contentArray[1].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected second part to be map[string]interface{}, got %T", contentArray[1])
	}

	if secondPart["type"] != "image_url" {
		t.Errorf("Expected type 'image_url', got '%v'", secondPart["type"])
	}

	imageURL, ok := secondPart["image_url"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected image_url to be map, got %T", secondPart["image_url"])
	}

	if imageURL["url"] != "https://example.com/test.jpg" {
		t.Errorf("Image URL mismatch: %v", imageURL["url"])
	}

	t.Log("✓ Multimodal content properly stored and retrieved from conversation history")
}

// TestMultimodalMixedConversation tests a conversation with both text and multimodal messages.
func TestMultimodalMixedConversation(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "mixed-session"

	// Message 1: Regular text
	om.addToHistory(sessionID, &OpenAIMessage{
		Role:    "user",
		Content: "Hello",
	})

	// Message 2: Multimodal (text + image)
	om.addToHistory(sessionID, &OpenAIMessage{
		Role: "user",
		Content: []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": "Look at this",
			},
			map[string]interface{}{
				"type": "image_url",
				"image_url": map[string]interface{}{
					"url": "https://example.com/photo.jpg",
				},
			},
		},
	})

	// Message 3: Assistant response (text)
	om.addToHistory(sessionID, &OpenAIMessage{
		Role:    "assistant",
		Content: "I see a beautiful photo",
	})

	// Message 4: Multimodal with multiple images
	om.addToHistory(sessionID, &OpenAIMessage{
		Role: "user",
		Content: []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": "Compare these",
			},
			map[string]interface{}{
				"type": "image_url",
				"image_url": map[string]interface{}{
					"url": "https://example.com/img1.jpg",
				},
			},
			map[string]interface{}{
				"type": "image_url",
				"image_url": map[string]interface{}{
					"url": "https://example.com/img2.jpg",
				},
			},
		},
	})

	// Verify all messages are in history
	history := om.getConversationHistory(sessionID)
	if len(history) != 4 {
		t.Fatalf("Expected 4 messages, got %d", len(history))
	}

	// Verify message types
	// Message 1: string content
	if _, ok := history[0].Content.(string); !ok {
		t.Errorf("Message 1: expected string content, got %T", history[0].Content)
	}

	// Message 2: array content with 2 parts
	if arr, ok := history[1].Content.([]interface{}); !ok {
		t.Errorf("Message 2: expected []interface{} content, got %T", history[1].Content)
	} else if len(arr) != 2 {
		t.Errorf("Message 2: expected 2 content parts, got %d", len(arr))
	}

	// Message 3: string content
	if _, ok := history[2].Content.(string); !ok {
		t.Errorf("Message 3: expected string content, got %T", history[2].Content)
	}

	// Message 4: array content with 3 parts (1 text + 2 images)
	if arr, ok := history[3].Content.([]interface{}); !ok {
		t.Errorf("Message 4: expected []interface{} content, got %T", history[3].Content)
	} else if len(arr) != 3 {
		t.Errorf("Message 4: expected 3 content parts, got %d", len(arr))
	}

	t.Log("✓ Mixed text and multimodal messages coexist in conversation history")
}
