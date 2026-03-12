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
	"strings"
	"testing"
)

// Test JSON serialization of OpenAIMessage
func TestOpenAIMessage_JSONSerialization(t *testing.T) {
	tests := []struct {
		name     string
		msg      *OpenAIMessage
		expected string
	}{
		{
			name: "user message",
			msg: &OpenAIMessage{
				Role:    "user",
				Content: "Hello, world!",
			},
			expected: `{"role":"user","content":"Hello, world!"}`,
		},
		{
			name: "assistant message",
			msg: &OpenAIMessage{
				Role:    "assistant",
				Content: "Hello! How can I help you?",
			},
			expected: `{"role":"assistant","content":"Hello! How can I help you?"}`,
		},
		{
			name: "system message",
			msg: &OpenAIMessage{
				Role:    "system",
				Content: "You are a helpful assistant",
			},
			expected: `{"role":"system","content":"You are a helpful assistant"}`,
		},
		{
			name: "tool message",
			msg: &OpenAIMessage{
				Role:       "tool",
				Content:    `{"temperature":"20C"}`,
				ToolCallID: "call_abc123",
			},
			expected: `{"role":"tool","content":"{\"temperature\":\"20C\"}","tool_call_id":"call_abc123"}`,
		},
		{
			name: "assistant with tool calls",
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
			expected: `{"role":"assistant","tool_calls":[{"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"London\"}"}}]}`,
		},
		{
			name: "message with name field",
			msg: &OpenAIMessage{
				Role:    "user",
				Content: "Hello from John",
				Name:    "John",
			},
			expected: `{"role":"user","content":"Hello from John","name":"John"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Serialize
			data, err := json.Marshal(tt.msg)
			if err != nil {
				t.Fatalf("Failed to marshal message: %v", err)
			}

			if string(data) != tt.expected {
				t.Errorf("JSON mismatch:\nGot:      %s\nExpected: %s", string(data), tt.expected)
			}

			// Deserialize
			var decoded OpenAIMessage
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("Failed to unmarshal message: %v", err)
			}

			// Validate deserialized message
			if decoded.Role != tt.msg.Role {
				t.Errorf("Role mismatch: got %s, expected %s", decoded.Role, tt.msg.Role)
			}
		})
	}
}

// Test JSON deserialization from OpenAI API responses
func TestOpenAIMessage_JSONDeserialization(t *testing.T) {
	tests := []struct {
		name     string
		jsonData string
		validate func(*testing.T, *OpenAIMessage)
	}{
		{
			name:     "simple user message",
			jsonData: `{"role":"user","content":"Hello"}`,
			validate: func(t *testing.T, msg *OpenAIMessage) {
				if msg.Role != "user" {
					t.Errorf("Expected role 'user', got '%s'", msg.Role)
				}
				if msg.Content != "Hello" {
					t.Errorf("Expected content 'Hello', got '%v'", msg.Content)
				}
			},
		},
		{
			name:     "assistant with tool call",
			jsonData: `{"role":"assistant","tool_calls":[{"id":"call_abc","type":"function","function":{"name":"test","arguments":"{}"}}]}`,
			validate: func(t *testing.T, msg *OpenAIMessage) {
				if msg.Role != "assistant" {
					t.Errorf("Expected role 'assistant', got '%s'", msg.Role)
				}
				if len(msg.ToolCalls) != 1 {
					t.Fatalf("Expected 1 tool call, got %d", len(msg.ToolCalls))
				}
				if msg.ToolCalls[0].ID != "call_abc" {
					t.Errorf("Expected tool call ID 'call_abc', got '%s'", msg.ToolCalls[0].ID)
				}
			},
		},
		{
			name:     "tool response",
			jsonData: `{"role":"tool","content":"result","tool_call_id":"call_123"}`,
			validate: func(t *testing.T, msg *OpenAIMessage) {
				if msg.Role != "tool" {
					t.Errorf("Expected role 'tool', got '%s'", msg.Role)
				}
				if msg.ToolCallID != "call_123" {
					t.Errorf("Expected ToolCallID 'call_123', got '%s'", msg.ToolCallID)
				}
			},
		},
		{
			name:     "message with null content",
			jsonData: `{"role":"assistant","content":null}`,
			validate: func(t *testing.T, msg *OpenAIMessage) {
				if msg.Role != "assistant" {
					t.Errorf("Expected role 'assistant', got '%s'", msg.Role)
				}
				if msg.Content != nil {
					t.Errorf("Expected nil content, got '%v'", msg.Content)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var msg OpenAIMessage
			if err := json.Unmarshal([]byte(tt.jsonData), &msg); err != nil {
				t.Fatalf("Failed to unmarshal: %v", err)
			}
			tt.validate(t, &msg)
		})
	}
}

// Test validation for each role type
func TestValidateMessage_RoleSpecific(t *testing.T) {
	tests := []struct {
		name      string
		role      string
		msg       *OpenAIMessage
		shouldErr bool
		errSubstr string
	}{
		{
			name: "valid user role",
			role: "user",
			msg: &OpenAIMessage{
				Role:    "user",
				Content: "Test message",
			},
			shouldErr: false,
		},
		{
			name: "valid assistant role",
			role: "assistant",
			msg: &OpenAIMessage{
				Role:    "assistant",
				Content: "Test response",
			},
			shouldErr: false,
		},
		{
			name: "valid system role",
			role: "system",
			msg: &OpenAIMessage{
				Role:    "system",
				Content: "System instructions",
			},
			shouldErr: false,
		},
		{
			name: "valid tool role",
			role: "tool",
			msg: &OpenAIMessage{
				Role:       "tool",
				Content:    `{"result":"ok"}`,
				ToolCallID: "call_123",
			},
			shouldErr: false,
		},
		{
			name: "tool role without ToolCallID",
			role: "tool",
			msg: &OpenAIMessage{
				Role:    "tool",
				Content: `{"result":"ok"}`,
			},
			shouldErr: true,
			errSubstr: "must have ToolCallID",
		},
		{
			name: "tool role without content",
			role: "tool",
			msg: &OpenAIMessage{
				Role:       "tool",
				ToolCallID: "call_123",
			},
			shouldErr: true,
			errSubstr: "must have content",
		},
		{
			name: "tool role with empty string content",
			role: "tool",
			msg: &OpenAIMessage{
				Role:       "tool",
				Content:    "",
				ToolCallID: "call_123",
			},
			shouldErr: true,
			errSubstr: "must have content",
		},
		{
			name: "assistant with empty tool calls array",
			role: "assistant",
			msg: &OpenAIMessage{
				Role:      "assistant",
				Content:   "Hello",
				ToolCalls: []ToolCall{},
			},
			shouldErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateMessage(tt.msg)
			if tt.shouldErr {
				if err == nil {
					t.Errorf("Expected error containing '%s', got nil", tt.errSubstr)
				} else if !strings.Contains(err.Error(), tt.errSubstr) {
					t.Errorf("Expected error containing '%s', got '%s'", tt.errSubstr, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, got: %v", err)
				}
			}
		})
	}
}

// Test assistant messages with tool calls
func TestValidateMessage_AssistantToolCalls(t *testing.T) {
	tests := []struct {
		name      string
		msg       *OpenAIMessage
		shouldErr bool
		errSubstr string
	}{
		{
			name: "assistant with single tool call",
			msg: &OpenAIMessage{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location":"NYC"}`,
						},
					},
				},
			},
			shouldErr: false,
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
							Arguments: `{"location":"NYC"}`,
						},
					},
					{
						ID:   "call_2",
						Type: "function",
						Function: FunctionCall{
							Name:      "get_time",
							Arguments: `{"timezone":"EST"}`,
						},
					},
				},
			},
			shouldErr: false,
		},
		{
			name: "assistant with tool call and content",
			msg: &OpenAIMessage{
				Role:    "assistant",
				Content: "Let me check that for you.",
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: FunctionCall{
							Name:      "search",
							Arguments: `{"query":"test"}`,
						},
					},
				},
			},
			shouldErr: false,
		},
		{
			name: "assistant with tool call missing ID",
			msg: &OpenAIMessage{
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
			shouldErr: true,
			errSubstr: "must have an ID",
		},
		{
			name: "assistant with tool call missing type",
			msg: &OpenAIMessage{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{
						ID: "call_1",
						Function: FunctionCall{
							Name:      "test",
							Arguments: `{}`,
						},
					},
				},
			},
			shouldErr: true,
			errSubstr: "must have a type",
		},
		{
			name: "assistant with tool call missing function name",
			msg: &OpenAIMessage{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: FunctionCall{
							Arguments: `{}`,
						},
					},
				},
			},
			shouldErr: true,
			errSubstr: "must have a function name",
		},
		{
			name: "assistant with tool call empty arguments (valid)",
			msg: &OpenAIMessage{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: FunctionCall{
							Name:      "no_params_function",
							Arguments: "",
						},
					},
				},
			},
			shouldErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateMessage(tt.msg)
			if tt.shouldErr {
				if err == nil {
					t.Errorf("Expected error containing '%s', got nil", tt.errSubstr)
				} else if !strings.Contains(err.Error(), tt.errSubstr) {
					t.Errorf("Expected error containing '%s', got '%s'", tt.errSubstr, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, got: %v", err)
				}
			}
		})
	}
}

// Test tool messages validation
func TestValidateMessage_ToolMessages(t *testing.T) {
	tests := []struct {
		name      string
		msg       *OpenAIMessage
		shouldErr bool
		errSubstr string
	}{
		{
			name: "valid tool message with string content",
			msg: &OpenAIMessage{
				Role:       "tool",
				Content:    `{"temperature":"20C","condition":"sunny"}`,
				ToolCallID: "call_weather_123",
			},
			shouldErr: false,
		},
		{
			name: "valid tool message simple content",
			msg: &OpenAIMessage{
				Role:       "tool",
				Content:    "Simple string result",
				ToolCallID: "call_abc",
			},
			shouldErr: false,
		},
		{
			name: "tool message missing ToolCallID",
			msg: &OpenAIMessage{
				Role:    "tool",
				Content: `{"result":"ok"}`,
			},
			shouldErr: true,
			errSubstr: "must have ToolCallID",
		},
		{
			name: "tool message with nil content",
			msg: &OpenAIMessage{
				Role:       "tool",
				Content:    nil,
				ToolCallID: "call_123",
			},
			shouldErr: true,
			errSubstr: "must have content",
		},
		{
			name: "tool message with empty string content",
			msg: &OpenAIMessage{
				Role:       "tool",
				Content:    "",
				ToolCallID: "call_123",
			},
			shouldErr: true,
			errSubstr: "must have content",
		},
		{
			name: "tool message with empty ToolCallID",
			msg: &OpenAIMessage{
				Role:       "tool",
				Content:    "result",
				ToolCallID: "",
			},
			shouldErr: true,
			errSubstr: "must have ToolCallID",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateMessage(tt.msg)
			if tt.shouldErr {
				if err == nil {
					t.Errorf("Expected error containing '%s', got nil", tt.errSubstr)
				} else if !strings.Contains(err.Error(), tt.errSubstr) {
					t.Errorf("Expected error containing '%s', got '%s'", tt.errSubstr, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, got: %v", err)
				}
			}
		})
	}
}

// Test edge cases
func TestValidateMessage_EdgeCases(t *testing.T) {
	tests := []struct {
		name      string
		msg       *OpenAIMessage
		shouldErr bool
		errSubstr string
	}{
		{
			name:      "nil message",
			msg:       nil,
			shouldErr: true,
			errSubstr: "cannot be nil",
		},
		{
			name: "empty role",
			msg: &OpenAIMessage{
				Content: "Hello",
			},
			shouldErr: true,
			errSubstr: "role cannot be empty",
		},
		{
			name: "message with only role",
			msg: &OpenAIMessage{
				Role: "user",
			},
			shouldErr: false, // Content can be optional for some messages
		},
		{
			name: "user message with nil content",
			msg: &OpenAIMessage{
				Role:    "user",
				Content: nil,
			},
			shouldErr: false, // Content is optional for user messages
		},
		{
			name: "assistant with nil content and no tool calls",
			msg: &OpenAIMessage{
				Role:    "assistant",
				Content: nil,
			},
			shouldErr: false, // Valid - some models may return empty responses
		},
		{
			name: "message with numeric content",
			msg: &OpenAIMessage{
				Role:    "user",
				Content: 123, // interface{} can hold any type
			},
			shouldErr: false, // Validation allows any content type
		},
		{
			name: "unknown role",
			msg: &OpenAIMessage{
				Role:    "unknown_role",
				Content: "test",
			},
			shouldErr: false, // We don't restrict role values (future compatibility)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateMessage(tt.msg)
			if tt.shouldErr {
				if err == nil {
					t.Errorf("Expected error containing '%s', got nil", tt.errSubstr)
				} else if !strings.Contains(err.Error(), tt.errSubstr) {
					t.Errorf("Expected error containing '%s', got '%s'", tt.errSubstr, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, got: %v", err)
				}
			}
		})
	}
}

// Test JSON round-trip (serialize -> deserialize -> validate)
func TestOpenAIMessage_JSONRoundTrip(t *testing.T) {
	messages := []*OpenAIMessage{
		{
			Role:    "user",
			Content: "Hello",
		},
		{
			Role:    "assistant",
			Content: "Hi there!",
		},
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{
					ID:   "call_123",
					Type: "function",
					Function: FunctionCall{
						Name:      "test",
						Arguments: `{"key":"value"}`,
					},
				},
			},
		},
		{
			Role:       "tool",
			Content:    `{"result":"success"}`,
			ToolCallID: "call_123",
		},
	}

	for i, msg := range messages {
		t.Run(msg.Role, func(t *testing.T) {
			// Validate original
			if err := validateMessage(msg); err != nil {
				t.Fatalf("Original message invalid: %v", err)
			}

			// Serialize
			data, err := json.Marshal(msg)
			if err != nil {
				t.Fatalf("Failed to marshal: %v", err)
			}

			// Deserialize
			var decoded OpenAIMessage
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("Failed to unmarshal: %v", err)
			}

			// Validate decoded
			if err := validateMessage(&decoded); err != nil {
				t.Errorf("Decoded message invalid: %v", err)
			}

			// Check role preserved
			if decoded.Role != msg.Role {
				t.Errorf("Message %d: role mismatch after round-trip", i)
			}
		})
	}
}
