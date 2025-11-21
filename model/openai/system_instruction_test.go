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
	"strings"
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// TestSystemInstruction tests that system instructions are properly converted
func TestSystemInstruction(t *testing.T) {
	cfg := &Config{
		BaseURL: "https://api.openai.com/v1",
	}

	m, err := NewModel("gpt-4", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	ctx := context.Background()

	tests := []struct {
		name              string
		systemInstruction *genai.Content
		wantSystemMsg     bool
		wantSystemText    string
	}{
		{
			name: "with system instruction",
			systemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: "You are a helpful assistant."},
				},
			},
			wantSystemMsg:  true,
			wantSystemText: "You are a helpful assistant.",
		},
		{
			name: "with multi-part system instruction",
			systemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: "You are a helpful assistant."},
					{Text: "Always be concise."},
				},
			},
			wantSystemMsg:  true,
			wantSystemText: "You are a helpful assistant.\nAlways be concise.",
		},
		{
			name:              "without system instruction",
			systemInstruction: nil,
			wantSystemMsg:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &model.LLMRequest{
				Contents: []*genai.Content{
					genai.NewContentFromText("Hello", "user"),
				},
				Config: &genai.GenerateContentConfig{
					SystemInstruction: tt.systemInstruction,
				},
			}

			msgs, err := om.convertToOpenAIMessages(ctx, req)
			if err != nil {
				t.Fatalf("convertToOpenAIMessages() error = %v", err)
			}

			// Check if system message exists
			hasSystemMsg := false
			var systemText string
			for _, msg := range msgs {
				if msg.Role == "system" {
					hasSystemMsg = true
					if text, ok := msg.Content.(string); ok {
						systemText = text
					}
					break
				}
			}

			if hasSystemMsg != tt.wantSystemMsg {
				t.Errorf("System message presence = %v, want %v", hasSystemMsg, tt.wantSystemMsg)
			}

			if tt.wantSystemMsg && systemText != tt.wantSystemText {
				t.Errorf("System text = %q, want %q", systemText, tt.wantSystemText)
			}
		})
	}
}

// TestJSONModeSafety tests that JSON mode automatically adds instruction
func TestJSONModeSafety(t *testing.T) {
	cfg := &Config{
		BaseURL: "https://api.openai.com/v1",
	}

	m, err := NewModel("gpt-4", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	ctx := context.Background()

	tests := []struct {
		name              string
		systemInstruction *genai.Content
		responseMIMEType  string
		wantJSONKeyword   bool
	}{
		{
			name:             "JSON mode without system instruction",
			responseMIMEType: "application/json",
			wantJSONKeyword:  true,
		},
		{
			name: "JSON mode with system instruction (no JSON keyword)",
			systemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: "You are a helpful assistant."},
				},
			},
			responseMIMEType: "application/json",
			wantJSONKeyword:  true,
		},
		{
			name: "JSON mode with system instruction (has JSON keyword)",
			systemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: "You are a helpful assistant. Respond in JSON format."},
				},
			},
			responseMIMEType: "application/json",
			wantJSONKeyword:  true,
		},
		{
			name: "No JSON mode",
			systemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: "You are a helpful assistant."},
				},
			},
			responseMIMEType: "",
			wantJSONKeyword:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &model.LLMRequest{
				Contents: []*genai.Content{
					genai.NewContentFromText("Hello", "user"),
				},
				Config: &genai.GenerateContentConfig{
					SystemInstruction: tt.systemInstruction,
					ResponseMIMEType:  tt.responseMIMEType,
				},
			}

			msgs, err := om.convertToOpenAIMessages(ctx, req)
			if err != nil {
				t.Fatalf("convertToOpenAIMessages() error = %v", err)
			}

			// Check system message for JSON keyword
			hasJSONKeyword := false
			for _, msg := range msgs {
				if msg.Role == "system" {
					if text, ok := msg.Content.(string); ok {
						hasJSONKeyword = strings.Contains(strings.ToUpper(text), "JSON")
					}
					break
				}
			}

			if tt.wantJSONKeyword && !hasJSONKeyword {
				t.Error("JSON mode enabled but system message doesn't contain 'JSON' keyword")
			}
		})
	}
}

// TestSystemInstructionOrder tests that system message is always first
func TestSystemInstructionOrder(t *testing.T) {
	cfg := &Config{
		BaseURL: "https://api.openai.com/v1",
	}

	m, err := NewModel("gpt-4", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	ctx := context.Background()

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello", "user"),
			genai.NewContentFromText("Hi there!", "model"),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: "You are a helpful assistant."},
				},
			},
		},
	}

	msgs, err := om.convertToOpenAIMessages(ctx, req)
	if err != nil {
		t.Fatalf("convertToOpenAIMessages() error = %v", err)
	}

	if len(msgs) == 0 {
		t.Fatal("No messages returned")
	}

	// First message should be system
	if msgs[0].Role != "system" {
		t.Errorf("First message role = %s, want system", msgs[0].Role)
	}

	// Verify text content
	if text, ok := msgs[0].Content.(string); ok {
		if !strings.Contains(text, "helpful assistant") {
			t.Errorf("System message doesn't contain expected text: %s", text)
		}
	} else {
		t.Error("System message content is not a string")
	}
}
