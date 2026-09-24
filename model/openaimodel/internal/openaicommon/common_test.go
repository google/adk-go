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

package openaicommon

import (
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"
)

func TestCompletedContentSupersedes(t *testing.T) {
	call := func(id, city string) *genai.Part {
		return &genai.Part{FunctionCall: &genai.FunctionCall{
			ID: id, Name: "get_weather", Args: map[string]any{"city": city},
		}}
	}
	tests := []struct {
		name      string
		aggregate []*genai.Part
		completed []*genai.Part
		want      bool
	}{
		{
			name:      "reasoning containing the answer is not visible output",
			aggregate: []*genai.Part{{Text: "hello"}},
			completed: []*genai.Part{{Text: "hello is the answer", Thought: true}},
		},
		{
			name:      "reasoning alone does not justify replacement",
			aggregate: []*genai.Part{{Text: "Checking", Thought: true}},
			completed: []*genai.Part{{Text: "Checked", Thought: true}},
		},
		{
			name:      "completed text can precede and follow streamed refusal",
			aggregate: []*genai.Part{{Text: "I cannot help."}},
			completed: []*genai.Part{{Text: "Here is "}, {Text: "I cannot help."}, {Text: " Sorry."}},
			want:      true,
		},
		{
			name:      "shorter text does not replace the answer",
			aggregate: []*genai.Part{{Text: "hello"}},
			completed: []*genai.Part{{Text: "hel"}},
		},
		{
			name:      "same name with different arguments is a different call",
			aggregate: []*genai.Part{call("call_1", "SF")},
			completed: []*genai.Part{call("call_1", "NY")},
		},
		{
			name:      "different call IDs remain distinct",
			aggregate: []*genai.Part{call("call_1", "SF")},
			completed: []*genai.Part{call("call_2", "SF")},
		},
		{
			name:      "one completed call cannot replace two streamed calls",
			aggregate: []*genai.Part{call("call_1", "SF"), call("call_1", "SF")},
			completed: []*genai.Part{call("call_1", "SF")},
		},
		{
			name:      "matching calls allow recovery of completed text",
			aggregate: []*genai.Part{call("call_1", "SF")},
			completed: []*genai.Part{call("call_1", "SF"), {Text: "Checking the weather."}},
			want:      true,
		},
		{
			name:      "reordered calls can match one to one",
			aggregate: []*genai.Part{call("call_1", "SF"), call("call_2", "NY")},
			completed: []*genai.Part{call("call_2", "NY"), call("call_1", "SF")},
			want:      true,
		},
		{
			name:      "completed function call replaces reasoning alone",
			aggregate: []*genai.Part{{Text: "Checking", Thought: true}},
			completed: []*genai.Part{call("call_1", "SF")},
			want:      true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			aggregate := &genai.Content{Role: genai.RoleModel, Parts: tc.aggregate}
			completed := &genai.Content{Role: genai.RoleModel, Parts: tc.completed}
			if got := CompletedContentSupersedes(aggregate, completed); got != tc.want {
				t.Errorf("CompletedContentSupersedes() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestEnsureFunctionToolOnly(t *testing.T) {
	tests := []struct {
		name    string
		tool    *genai.Tool
		wantErr string
	}{
		{
			name:    "nil tool",
			tool:    nil,
			wantErr: "tool 0 is nil",
		},
		{
			name:    "non-function tool",
			tool:    &genai.Tool{GoogleSearch: &genai.GoogleSearch{}},
			wantErr: "non-function tools",
		},
		{
			name:    "no functions",
			tool:    &genai.Tool{},
			wantErr: "does not declare any functions",
		},
		{
			name: "valid",
			tool: &genai.Tool{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "fn1"}}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := EnsureFunctionToolOnly(0, tc.tool)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected no error, got %v", err)
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
				}
			}
		})
	}
}

func TestSchemaToMap(t *testing.T) {
	tests := []struct {
		name    string
		schema  *genai.Schema
		wantErr bool
		want    map[string]any
	}{
		{
			name:   "nil schema",
			schema: nil,
			want:   nil,
		},
		{
			name:   "string type",
			schema: &genai.Schema{Type: genai.TypeString},
			want:   map[string]any{"type": "string"}, // Marshals as "STRING" if using standard json, but we lower it
		},
		{
			name:    "invalid type",
			schema:  &genai.Schema{Example: make(chan int)},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := SchemaToMap(tc.schema)
			if (err != nil) != tc.wantErr {
				t.Fatalf("SchemaToMap() error = %v, wantErr %v", err, tc.wantErr)
			} else {
				if got["type"] != tc.want["type"] {
					t.Fatalf("unexpected map: %+v, want %+v", got, tc.want)
				}
			}
		})
	}
}

func TestFlattenContentText(t *testing.T) {
	tests := []struct {
		name    string
		content *genai.Content
		want    string
		wantErr bool
	}{
		{
			name:    "nil content",
			content: nil,
			want:    "",
		},
		{
			name: "valid text parts",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "part1"},
					nil,
					{Text: "part2"},
				},
			},
			want: "part1\npart2",
		},
		{
			name: "non-text part",
			content: &genai.Content{
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: "fn"}},
				},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			txt, err := FlattenContentText(tc.content)
			if (err != nil) != tc.wantErr {
				t.Fatalf("FlattenContentText() error = %v, wantErr %v", err, tc.wantErr)
			}
			if txt != tc.want {
				t.Fatalf("FlattenContentText() = %q, want %q", txt, tc.want)
			}
		})
	}
}

func TestNormalizeSchema(t *testing.T) {
	tests := []struct {
		name    string
		schema  any
		want    map[string]any
		wantErr bool
	}{
		{
			name:    "nil schema",
			schema:  nil,
			wantErr: true,
		},
		{
			name:   "map schema",
			schema: map[string]any{"type": "object"},
			want:   map[string]any{"type": "object"},
		},
		{
			name: "struct schema",
			schema: struct {
				Type string `json:"type"`
			}{Type: "array"},
			want: map[string]any{"type": "array"},
		},
		{
			name:    "invalid schema",
			schema:  func() {}, // unmarshalable
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NormalizeSchema(tc.schema)
			if (err != nil) != tc.wantErr {
				t.Fatalf("NormalizeSchema() error = %v, wantErr %v", err, tc.wantErr)
			}
			if !tc.wantErr && got["type"] != tc.want["type"] {
				t.Fatalf("NormalizeSchema() = %v, want %v", got, tc.want)
			}
		})
	}
}

// RequestTimeout must be safe on its own terms. applyGenerationConfig rejects a
// non-positive timeout before any request is built, so this guard is defense in
// depth — and it is tested directly rather than assumed unreachable, because a
// guard that only holds while callers keep the right order is not a guard.
func TestRequestTimeoutGuardsNonPositiveItself(t *testing.T) {
	for _, d := range []time.Duration{0, -time.Second} {
		timeout := d
		cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &timeout}}
		if got := RequestTimeout(cfg); got != 0 {
			t.Errorf("RequestTimeout(%v) = %v, want 0: a non-positive bound must not become a deadline", d, got)
		}
	}
	// The positive case still comes through, so the guard is not simply off.
	positive := time.Second
	cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &positive}}
	if got := RequestTimeout(cfg); got != positive {
		t.Errorf("RequestTimeout(%v) = %v, want it unchanged", positive, got)
	}
}
