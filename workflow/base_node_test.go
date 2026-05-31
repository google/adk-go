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

package workflow

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/session"
)

// Compile-time assertions: every built-in workflow node must satisfy
// the Node interface.
var (
	_ Node = (*startNode)(nil)
	_ Node = (*FunctionNode)(nil)
	_ Node = (*AgentNode)(nil)
	_ Node = (*ToolNode)(nil)
	_ Node = (*JoinNode)(nil)
	_ Node = (*ParallelWorker)(nil)
	_ Node = (*WorkflowNode)(nil)
)

func TestNewBaseNode_RoundTrip(t *testing.T) {
	tTrue := true
	tFalse := false
	tests := []struct {
		name       string
		nameArg    string
		descArg    string
		cfg        NodeConfig
		wantConfig NodeConfig
	}{
		{
			name:    "zero config",
			nameArg: "n",
			descArg: "desc",
		},
		{
			name:       "WaitForOutput=true (JoinNode shape)",
			nameArg:    "join",
			descArg:    "fan-in",
			cfg:        NodeConfig{WaitForOutput: &tTrue},
			wantConfig: NodeConfig{WaitForOutput: &tTrue},
		},
		{
			name:       "ParallelWorker=true",
			nameArg:    "mapper",
			descArg:    "data parallel",
			cfg:        NodeConfig{ParallelWorker: true},
			wantConfig: NodeConfig{ParallelWorker: true},
		},
		{
			name:       "empty name and description",
			cfg:        NodeConfig{},
			wantConfig: NodeConfig{},
		},
		{
			name:    "fully populated configuration",
			nameArg: "full_node",
			descArg: "Node with all config fields set",
			cfg: NodeConfig{
				ParallelWorker: true,
				RerunOnResume:  &tFalse,
				WaitForOutput:  &tTrue,
				RetryConfig: &RetryConfig{
					MaxAttempts: 3,
				},
				Timeout: 10 * time.Second,
			},
			wantConfig: NodeConfig{
				ParallelWorker: true,
				RerunOnResume:  &tFalse,
				WaitForOutput:  &tTrue,
				RetryConfig: &RetryConfig{
					MaxAttempts: 3,
				},
				Timeout: 10 * time.Second,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := NewBaseNode(tt.nameArg, tt.descArg, tt.cfg)
			if got := b.Name(); got != tt.nameArg {
				t.Errorf("Name() = %q, want %q", got, tt.nameArg)
			}
			if got := b.Description(); got != tt.descArg {
				t.Errorf("Description() = %q, want %q", got, tt.descArg)
			}
			want := tt.wantConfig
			if (tt.cfg.WaitForOutput == nil) && (tt.cfg.ParallelWorker == false) {
				want = NodeConfig{}
			}
			if diff := cmp.Diff(want, b.Config()); diff != "" {
				t.Errorf("Config() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

type testSchemaInput struct {
	Value string `json:"value"`
}

func TestBaseNode_NilSchemas(t *testing.T) {
	b := NewBaseNode("nil_schema_node", "no validation node", NodeConfig{})
	if b.InputSchema() != nil {
		t.Errorf("InputSchema() = %v, want nil", b.InputSchema())
	}
	if b.OutputSchema() != nil {
		t.Errorf("OutputSchema() = %v, want nil", b.OutputSchema())
	}

	// ValidateInput with nil schema is a passthrough.
	input := map[string]any{"value": "hello"}
	got, err := b.ValidateInput(input)
	if err != nil {
		t.Fatalf("ValidateInput failed: %v", err)
	}
	if diff := cmp.Diff(input, got); diff != "" {
		t.Errorf("ValidateInput mismatch (-want +got):\n%s", diff)
	}

	// ValidateOutput with nil schema is a passthrough.
	output := map[string]any{"value": "world"}
	gotOut, err := b.ValidateOutput(output)
	if err != nil {
		t.Fatalf("ValidateOutput failed: %v", err)
	}
	if diff := cmp.Diff(output, gotOut); diff != "" {
		t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
	}
}

func TestBaseNode_WithSchemas(t *testing.T) {
	s, err := jsonschema.For[testSchemaInput](nil)
	if err != nil {
		t.Fatalf("jsonschema.For failed: %v", err)
	}
	resolvedInputSchema, err := s.Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve failed: %v", err)
	}

	b := NewBaseNodeWithSchemas("schema_node", "validation node", NodeConfig{}, resolvedInputSchema, resolvedInputSchema)
	if b.InputSchema() != resolvedInputSchema {
		t.Errorf("InputSchema() = %v, want %v", b.InputSchema(), resolvedInputSchema)
	}
	if b.OutputSchema() != resolvedInputSchema {
		t.Errorf("OutputSchema() = %v, want %v", b.OutputSchema(), resolvedInputSchema)
	}

	// Valid input is coerced/returned.
	input := map[string]any{"value": "hello"}
	got, err := b.ValidateInput(input)
	if err != nil {
		t.Fatalf("ValidateInput failed on valid input: %v", err)
	}
	// ValidateInput returns a coerced type (which is map[string]any because we coerce using ConvertToWithJSONSchema[any, any])
	// Let's verify it contains the expected value.
	gotMap, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("expected got to be map[string]any, got %T", got)
	}
	if gotMap["value"] != "hello" {
		t.Errorf("expected value %q, got %v", "hello", gotMap["value"])
	}

	// Invalid input fails validation.
	invalidInput := map[string]any{"value": 123}
	_, err = b.ValidateInput(invalidInput)
	if err == nil {
		t.Error("expected ValidateInput to fail on invalid input type, but succeeded")
	}

	// Valid output is returned unchanged.
	output := map[string]any{"value": "world"}
	gotOut, err := b.ValidateOutput(output)
	if err != nil {
		t.Fatalf("ValidateOutput failed on valid output: %v", err)
	}
	if diff := cmp.Diff(output, gotOut); diff != "" {
		t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
	}

	// Invalid output fails validation.
	invalidOutput := map[string]any{"value": 123}
	_, err = b.ValidateOutput(invalidOutput)
	if err == nil {
		t.Error("expected ValidateOutput to fail on invalid output type, but succeeded")
	}
}

// resolveTestSchema generates a *jsonschema.Resolved from a Go type
// for use in tests.
func resolveTestSchema[T any](t *testing.T) *jsonschema.Resolved {
	t.Helper()
	s, err := jsonschema.For[T](nil)
	if err != nil {
		t.Fatalf("jsonschema.For failed: %v", err)
	}
	resolved, err := s.Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve failed: %v", err)
	}
	return resolved
}

// TestDefaultValidateOutput_PassthroughTypes verifies that framework
// control values (*session.Event, *session.RequestInput) are returned
// unchanged by defaultValidateOutput even when a strict schema is
// configured: they are routed through Event.Output by some nodes but
// are not user output payloads.
func TestDefaultValidateOutput_PassthroughTypes(t *testing.T) {
	schema := resolveTestSchema[testSchemaInput](t)

	tests := []struct {
		name string
		in   any
	}{
		{
			name: "*session.Event",
			in:   &session.Event{Author: "node"},
		},
		{
			name: "*session.RequestInput",
			in:   &session.RequestInput{InterruptID: "approval"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := defaultValidateOutput(tc.in, schema)
			if err != nil {
				t.Fatalf("expected passthrough, got error: %v", err)
			}
			if got != tc.in {
				t.Errorf("expected identity passthrough, got different value")
			}
		})
	}
}

// TestDefaultValidateOutput_ContentFallback exercises the
// *genai.Content fallback path: extract text from parts, return it
// directly when the schema's root type is "string", otherwise
// JSON-parse and re-validate. When the fallback cannot produce a
// valid value the original validation error is surfaced.
func TestDefaultValidateOutput_ContentFallback(t *testing.T) {
	t.Run("string_schema_returns_text", func(t *testing.T) {
		schema := resolveTestSchema[string](t)
		content := &genai.Content{
			Parts: []*genai.Part{{Text: "hello "}, {Text: "world"}},
		}
		got, err := defaultValidateOutput(content, schema)
		if err != nil {
			t.Fatalf("defaultValidateOutput failed: %v", err)
		}
		if got != "hello world" {
			t.Errorf("got %q, want %q", got, "hello world")
		}
	})

	t.Run("object_schema_parses_json", func(t *testing.T) {
		schema := resolveTestSchema[testSchemaInput](t)
		content := &genai.Content{
			Parts: []*genai.Part{{Text: `{"value":"hello"}`}},
		}
		got, err := defaultValidateOutput(content, schema)
		if err != nil {
			t.Fatalf("defaultValidateOutput failed: %v", err)
		}
		gotMap, ok := got.(map[string]any)
		if !ok {
			t.Fatalf("expected map[string]any, got %T", got)
		}
		if gotMap["value"] != "hello" {
			t.Errorf("got %v, want value=hello", gotMap)
		}
	})

	t.Run("invalid_json_returns_original_error", func(t *testing.T) {
		schema := resolveTestSchema[testSchemaInput](t)
		content := &genai.Content{
			Parts: []*genai.Part{{Text: "not valid json"}},
		}
		_, err := defaultValidateOutput(content, schema)
		if err == nil {
			t.Fatal("expected validation error, got nil")
		}
	})

	t.Run("empty_text_returns_original_error", func(t *testing.T) {
		schema := resolveTestSchema[testSchemaInput](t)
		content := &genai.Content{
			Parts: []*genai.Part{{Text: "   "}},
		}
		_, err := defaultValidateOutput(content, schema)
		if err == nil {
			t.Fatal("expected validation error, got nil")
		}
	})
}
