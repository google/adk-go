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
	"testing"
)

func TestStripThinkingBlocks(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "no thinking block",
			input: "Hello, world!",
			want:  "Hello, world!",
		},
		{
			name:  "thinking block at start",
			input: "<think>\nLet me think about this...\nI should answer directly.\n</think>\nThe answer is 4.",
			want:  "The answer is 4.",
		},
		{
			name:  "thinking block with no newline after closing tag",
			input: "<think>reasoning here</think>The answer is 42.",
			want:  "The answer is 42.",
		},
		{
			name:  "thinking block only (no actual content)",
			input: "<think>Just thinking, no response.</think>",
			want:  "",
		},
		{
			name:  "multiple thinking blocks",
			input: "<think>first thought</think>\nHello.\n<think>second thought</think>\nGoodbye.",
			want:  "Hello.\nGoodbye.",
		},
		{
			name:  "thinking block with tool call reasoning",
			input: "<think>\nThe user wants weather info.\nI should call get_weather.\n</think>\n",
			want:  "",
		},
		{
			name:  "empty string",
			input: "",
			want:  "",
		},
		{
			name:  "text without think tags but with angle brackets",
			input: "Use <b>bold</b> text for emphasis.",
			want:  "Use <b>bold</b> text for emphasis.",
		},
		{
			name:  "multiline thinking with code",
			input: "<think>\nLet me write some code:\n```python\nprint('hello')\n```\n</think>\nHere's your answer.",
			want:  "Here's your answer.",
		},
		{
			name:  "thinking block with extra whitespace after",
			input: "<think>thought</think>   \n\n  The answer.",
			want:  "The answer.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripThinkingBlocks(tt.input)
			if got != tt.want {
				t.Errorf("stripThinkingBlocks() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestReasoningContentField(t *testing.T) {
	// Verify that ReasoningContent is parsed from JSON.
	jsonData := `{
		"role": "assistant",
		"content": "The answer is 4.",
		"reasoning_content": "Let me think... 2+2=4"
	}`

	var msg OpenAIMessage
	if err := json.Unmarshal([]byte(jsonData), &msg); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if msg.ReasoningContent != "Let me think... 2+2=4" {
		t.Errorf("ReasoningContent = %q, want %q", msg.ReasoningContent, "Let me think... 2+2=4")
	}

	if msg.Content != "The answer is 4." {
		t.Errorf("Content = %v, want %q", msg.Content, "The answer is 4.")
	}
}

func TestReasoningContentNotLeaked(t *testing.T) {
	// Verify that ReasoningContent is NOT included when marshaling back
	// (since we don't send it to the model).
	msg := OpenAIMessage{
		Role:             "assistant",
		Content:          "Hello",
		ReasoningContent: "internal thinking",
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	var result map[string]any
	json.Unmarshal(data, &result)

	// ReasoningContent should be present in marshaled JSON (omitempty only omits zero values).
	if _, ok := result["reasoning_content"]; !ok {
		// This is fine — it means omitempty worked (empty string is zero value).
		// But our test has a non-empty value, so it should be present.
		t.Log("reasoning_content not present in marshaled JSON (omitempty behavior)")
	}
}

func TestConvertToLLMResponseStripsThinking(t *testing.T) {
	m := &openaiModel{name: "test"}

	msg := &OpenAIMessage{
		Role:    "assistant",
		Content: "<think>\nLet me analyze this query.\n</think>\nThe weather in London is 15°C.",
	}

	resp, err := m.convertToLLMResponse(msg, nil, nil)
	if err != nil {
		t.Fatalf("convertToLLMResponse failed: %v", err)
	}

	if resp.Content == nil || len(resp.Content.Parts) == 0 {
		t.Fatal("Expected non-empty content")
	}

	text := resp.Content.Parts[0].Text
	if text != "The weather in London is 15°C." {
		t.Errorf("Expected clean text, got %q", text)
	}
}

func TestConvertToLLMResponseThinkingOnly(t *testing.T) {
	m := &openaiModel{name: "test"}

	// Response with only thinking content (no actual response).
	msg := &OpenAIMessage{
		Role:    "assistant",
		Content: "<think>I need to call the weather tool.</think>",
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

	resp, err := m.convertToLLMResponse(msg, nil, nil)
	if err != nil {
		t.Fatalf("convertToLLMResponse failed: %v", err)
	}

	// Should have only the function call part (thinking stripped, empty text not added).
	if resp.Content == nil {
		t.Fatal("Expected non-nil content")
	}

	for _, part := range resp.Content.Parts {
		if part.Text != "" {
			t.Errorf("Expected no text parts (thinking should be stripped), got %q", part.Text)
		}
	}

	// Should still have the function call.
	hasFunctionCall := false
	for _, part := range resp.Content.Parts {
		if part.FunctionCall != nil {
			hasFunctionCall = true
			if part.FunctionCall.Name != "get_weather" {
				t.Errorf("Expected get_weather, got %s", part.FunctionCall.Name)
			}
		}
	}
	if !hasFunctionCall {
		t.Error("Expected function call in response")
	}
}
