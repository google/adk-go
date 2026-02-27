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

