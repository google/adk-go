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
	"strings"
	"testing"
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
		region    string
		wantError string
	}{
		{"missing_project", "", "us-central1", "VertexProjectID is required"},
		{"missing_region", "test-project", "", "VertexRegion is required"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("GOOGLE_CLOUD_PROJECT", tt.project)
			t.Setenv("GOOGLE_CLOUD_REGION", tt.region)

			cfg := &Config{Variant: VariantVertexAI}
			_, err := NewModel(t.Context(), "claude-sonnet-4-20250514", cfg)
			if err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("NewModel() error = %v, want contains %q", err, tt.wantError)
			}
		})
	}
}


