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

package anthropic

import (
	"context"
	"strings"
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestNewModel_ConfigBehavior(t *testing.T) {
	tests := []struct {
		name          string
		cfg           *Config
		wantMaxTokens int
		wantVariant   string
	}{
		{
			name: "explicit_max_tokens_and_variant",
			cfg: &Config{
				APIKey:           "test-api-key",
				DefaultMaxTokens: 2048,
				Variant:          VariantAnthropicAPI,
			},
			wantMaxTokens: 2048,
			wantVariant:   VariantAnthropicAPI,
		},
		{
			name: "default_max_tokens",
			cfg: &Config{
				APIKey:  "test-api-key",
				Variant: VariantAnthropicAPI,
			},
			wantMaxTokens: defaultMaxTokens,
			wantVariant:   VariantAnthropicAPI,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, err := NewModel(t.Context(), "claude-sonnet-4-20250514", tt.cfg)
			if err != nil {
				t.Fatalf("NewModel() error = %v", err)
			}

			if model.Name() != "claude-sonnet-4-20250514" {
				t.Errorf("Name() = %q, want %q", model.Name(), "claude-sonnet-4-20250514")
			}

			am := model.(*anthropicModel)
			if am.defaultMaxTokens != tt.wantMaxTokens {
				t.Errorf("defaultMaxTokens = %d, want %d", am.defaultMaxTokens, tt.wantMaxTokens)
			}
			if am.variant != tt.wantVariant {
				t.Errorf("variant = %q, want %q", am.variant, tt.wantVariant)
			}
		})
	}
}

func TestNewModel_VertexAI_MissingConfig(t *testing.T) {
	tests := []struct {
		name      string
		project   string
		location  string
		wantError string
	}{
		{"missing_project", "", "us-central1", "VertexProjectID is required"},
		{"missing_location", "test-project", "", "VertexLocation is required"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("GOOGLE_CLOUD_PROJECT", tt.project)
			t.Setenv("GOOGLE_CLOUD_LOCATION", tt.location)

			cfg := &Config{Variant: VariantVertexAI}
			_, err := NewModel(t.Context(), "claude-sonnet-4-20250514", cfg)
			if err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("NewModel() error = %v, want contains %q", err, tt.wantError)
			}
		})
	}
}

func TestConvertRequest_VertexAI_SkipsOutputConfig(t *testing.T) {
	m := &anthropicModel{
		name:             "claude-haiku-4-5-20251001",
		variant:          VariantVertexAI,
		defaultMaxTokens: defaultMaxTokens,
	}

	schema := &genai.Schema{
		Type:     genai.TypeObject,
		Required: []string{"name"},
		Properties: map[string]*genai.Schema{
			"name": {Type: genai.TypeString},
		},
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello", "user"),
		},
		Config: &genai.GenerateContentConfig{
			ResponseSchema: schema,
		},
	}

	params, err := m.convertRequest(req)
	if err != nil {
		t.Fatalf("convertRequest() error = %v", err)
	}

	// OutputConfig must not be set for Vertex AI
	if params.OutputConfig.Format.Schema != nil {
		t.Error("expected OutputConfig to be empty for Vertex AI, but it was set")
	}
}

func TestConvertRequest_DirectAPI_SetsOutputConfig(t *testing.T) {
	m := &anthropicModel{
		name:             "claude-haiku-4-5-20251001",
		variant:          VariantAnthropicAPI,
		defaultMaxTokens: defaultMaxTokens,
	}

	schema := &genai.Schema{
		Type:     genai.TypeObject,
		Required: []string{"name"},
		Properties: map[string]*genai.Schema{
			"name": {Type: genai.TypeString},
		},
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello", "user"),
		},
		Config: &genai.GenerateContentConfig{
			ResponseSchema: schema,
		},
	}

	params, err := m.convertRequest(req)
	if err != nil {
		t.Fatalf("convertRequest() error = %v", err)
	}

	if params.OutputConfig.Format.Schema == nil {
		t.Error("expected OutputConfig to be set for direct API, but it was empty")
	}
}

func TestEmbedSchemaAsSystemPrompt(t *testing.T) {
	schema := &genai.Schema{
		Type:     genai.TypeObject,
		Required: []string{"name"},
		Properties: map[string]*genai.Schema{
			"name": {Type: genai.TypeString},
		},
	}

	t.Run("no_existing_system_instruction", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Hello", "user"),
			},
			Config: &genai.GenerateContentConfig{
				ResponseSchema: schema,
			},
		}

		modified := embedSchemaAsSystemPrompt(req)

		// ResponseSchema should be cleared
		if modified.Config.ResponseSchema != nil {
			t.Error("expected ResponseSchema to be nil in modified request")
		}

		// Original request should be unchanged
		if req.Config.ResponseSchema == nil {
			t.Error("expected original request ResponseSchema to be unchanged")
		}

		// System instruction should contain the schema
		if modified.Config.SystemInstruction == nil {
			t.Fatal("expected SystemInstruction to be set")
		}
		text := modified.Config.SystemInstruction.Parts[0].Text
		if !strings.Contains(text, "JSON schema") {
			t.Errorf("expected system instruction to contain schema, got: %s", text)
		}
	})

	t.Run("with_existing_system_instruction", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Hello", "user"),
			},
			Config: &genai.GenerateContentConfig{
				SystemInstruction: &genai.Content{
					Parts: []*genai.Part{{Text: "You are a helpful assistant."}},
				},
				ResponseSchema: schema,
			},
		}

		modified := embedSchemaAsSystemPrompt(req)

		text := modified.Config.SystemInstruction.Parts[0].Text
		if !strings.Contains(text, "You are a helpful assistant.") {
			t.Error("expected original system instruction to be preserved")
		}
		if !strings.Contains(text, "JSON schema") {
			t.Error("expected schema instruction to be appended")
		}
	})
}

func TestStripMarkdownFromResponse(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		wantText string
	}{
		{
			name:     "no_fences",
			text:     `{"name": "test"}`,
			wantText: `{"name": "test"}`,
		},
		{
			name:     "json_fence",
			text:     "```json\n{\"name\": \"test\"}\n```",
			wantText: `{"name": "test"}`,
		},
		{
			name:     "plain_fence",
			text:     "```\n{\"name\": \"test\"}\n```",
			wantText: `{"name": "test"}`,
		},
		{
			name:     "fence_with_preamble",
			text:     "Here is the result:\n```json\n{\"name\": \"test\"}\n```",
			wantText: `{"name": "test"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := &model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: tt.text}},
				},
			}

			stripMarkdownFromResponse(context.Background(), resp)

			got := resp.Content.Parts[0].Text
			if got != tt.wantText {
				t.Errorf("stripMarkdownFromResponse() text = %q, want %q", got, tt.wantText)
			}
		})
	}
}

func TestStripMarkdownFromResponse_NilSafety(t *testing.T) {
	// Should not panic on nil inputs
	stripMarkdownFromResponse(context.Background(), nil)
	stripMarkdownFromResponse(context.Background(), &model.LLMResponse{})
	stripMarkdownFromResponse(context.Background(), &model.LLMResponse{Content: &genai.Content{}})
}
