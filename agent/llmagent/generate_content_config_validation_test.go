// Copyright 2026 Google LLC
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

package llmagent_test

import (
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
)

// assertInvalidGenerateContentConfig fails t unless New returned no agent and
// an error that wraps agent.ErrInvalidAgentConfig and contains want.
func assertInvalidGenerateContentConfig(t *testing.T, a agent.Agent, err error, want string) {
	t.Helper()
	if a != nil {
		t.Errorf("New() agent = %T, want nil", a)
	}
	if !errors.Is(err, agent.ErrInvalidAgentConfig) {
		t.Fatalf("New() error = %v, want an error wrapping agent.ErrInvalidAgentConfig", err)
	}
	if !strings.Contains(err.Error(), want) {
		t.Errorf("New() error = %q, want it to contain %q", err, want)
	}
}

// TestNew_GenerateContentConfigRejectsTools fails if New accepts a
// GenerateContentConfig with a tool, or refuses it without wrapping
// agent.ErrInvalidAgentConfig or without naming llmagent.Config.Tools.
func TestNew_GenerateContentConfigRejectsTools(t *testing.T) {
	a, err := llmagent.New(llmagent.Config{
		Name:                  "a",
		GenerateContentConfig: &genai.GenerateContentConfig{Tools: []*genai.Tool{{}}},
	})
	assertInvalidGenerateContentConfig(t, a, err, "llmagent.Config.Tools")
}

// TestNew_GenerateContentConfigRejectsSystemInstruction fails if New accepts a
// GenerateContentConfig with a non-nil SystemInstruction, including one with
// no parts, or refuses it without naming llmagent.Config.Instruction.
func TestNew_GenerateContentConfigRejectsSystemInstruction(t *testing.T) {
	tests := []struct {
		name string
		si   *genai.Content
	}{
		{name: "non_empty_content", si: genai.NewContentFromText("system instruction", genai.RoleUser)},
		{name: "empty_content_struct", si: &genai.Content{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, err := llmagent.New(llmagent.Config{
				Name:                  "a",
				GenerateContentConfig: &genai.GenerateContentConfig{SystemInstruction: tc.si},
			})
			assertInvalidGenerateContentConfig(t, a, err, "llmagent.Config.Instruction")
		})
	}
}

// TestNew_GenerateContentConfigRejectsHTTPOptionsBaseURL fails if New accepts
// a GenerateContentConfig whose HTTPOptions set a BaseURL, or refuses it
// without pointing the caller to the model or its client.
func TestNew_GenerateContentConfigRejectsHTTPOptionsBaseURL(t *testing.T) {
	a, err := llmagent.New(llmagent.Config{
		Name: "a",
		GenerateContentConfig: &genai.GenerateContentConfig{
			HTTPOptions: &genai.HTTPOptions{BaseURL: "https://example.invalid"},
		},
	})
	assertInvalidGenerateContentConfig(t, a, err, "model or its client")
}

// TestNew_GenerateContentConfigValidationOrder fails if New, given several
// refused fields, reports any but the first of Tools, SystemInstruction and
// HTTPOptions.BaseURL, the order adk-python checks them in.
func TestNew_GenerateContentConfigValidationOrder(t *testing.T) {
	tests := []struct {
		name string
		cfg  *genai.GenerateContentConfig
		want string
	}{
		{
			name: "tools_before_system_instruction_and_base_url",
			cfg: &genai.GenerateContentConfig{
				Tools:             []*genai.Tool{{}},
				SystemInstruction: &genai.Content{},
				HTTPOptions:       &genai.HTTPOptions{BaseURL: "https://example.invalid"},
			},
			want: "llmagent.Config.Tools",
		},
		{
			name: "system_instruction_before_base_url",
			cfg: &genai.GenerateContentConfig{
				SystemInstruction: &genai.Content{},
				HTTPOptions:       &genai.HTTPOptions{BaseURL: "https://example.invalid"},
			},
			want: "llmagent.Config.Instruction",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, err := llmagent.New(llmagent.Config{Name: "a", GenerateContentConfig: tc.cfg})
			assertInvalidGenerateContentConfig(t, a, err, tc.want)
		})
	}
}

// TestNew_GenerateContentConfigAllowsEmptyReservedFields fails if New treats
// an empty Tools slice, a nil SystemInstruction or HTTPOptions with an empty
// BaseURL as set.
func TestNew_GenerateContentConfigAllowsEmptyReservedFields(t *testing.T) {
	_, err := llmagent.New(llmagent.Config{
		Name: "a",
		GenerateContentConfig: &genai.GenerateContentConfig{
			Tools:             []*genai.Tool{},
			SystemInstruction: nil,
			HTTPOptions:       &genai.HTTPOptions{BaseURL: ""},
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v, want nil", err)
	}
}

// TestNew_GenerateContentConfigPassesRequestTimeFields fails if New refuses
// the request-time HTTPOptions fields Timeout, Headers and ExtraBody, or if
// they do not reach the model request.
func TestNew_GenerateContentConfigPassesRequestTimeFields(t *testing.T) {
	m := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("ok", genai.RoleModel)}}
	a, err := llmagent.New(llmagent.Config{
		Name:  "a",
		Model: m,
		GenerateContentConfig: &genai.GenerateContentConfig{
			HTTPOptions: &genai.HTTPOptions{
				Timeout:   new(time.Duration(1000)),
				Headers:   http.Header{"X-Custom": {"val"}},
				ExtraBody: map[string]any{"tool_config": map[string]any{"mode": "AUTO"}},
			},
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v, want nil", err)
	}
	r := testutil.NewTestAgentRunner(t, a)
	if _, err := testutil.CollectEvents(r.Run(t, "s", "hi")); err != nil {
		t.Fatal(err)
	}

	if len(m.Requests) != 1 {
		t.Fatalf("model got %d requests, want 1", len(m.Requests))
	}
	want := &genai.HTTPOptions{
		Timeout:   new(time.Duration(1000)),
		Headers:   http.Header{"X-Custom": {"val"}},
		ExtraBody: map[string]any{"tool_config": map[string]any{"mode": "AUTO"}},
	}
	if diff := cmp.Diff(want, m.Requests[0].Config.HTTPOptions); diff != "" {
		t.Errorf("request HTTPOptions mismatch (-want +got):\n%s", diff)
	}
}

// TestNew_GenerateContentConfigAllowsResponseJsonSchema fails if New refuses
// a GenerateContentConfig.ResponseJsonSchema next to Config.OutputSchema.
// adk-python does not reserve response_json_schema either.
func TestNew_GenerateContentConfigAllowsResponseJsonSchema(t *testing.T) {
	_, err := llmagent.New(llmagent.Config{
		Name:                  "a",
		OutputSchema:          &genai.Schema{Type: genai.TypeObject},
		GenerateContentConfig: &genai.GenerateContentConfig{ResponseJsonSchema: map[string]any{"type": "string"}},
	})
	if err != nil {
		t.Fatalf("New() error = %v, want nil", err)
	}
}
