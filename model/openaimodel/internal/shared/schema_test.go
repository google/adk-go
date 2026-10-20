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

package shared

import (
	"regexp"
	"strings"
	"testing"

	"google.golang.org/genai"
)

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

// openAIFormatName is the pattern OpenAI documents on the name field of both
// endpoints' structured-output format, quoted from the pinned openai-go:
// "Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum
// length of 64."
var openAIFormatName = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

func TestResponseFormatName(t *testing.T) {
	tests := []struct {
		name   string
		schema *genai.Schema
		want   string
	}{
		{
			name:   "nil schema",
			schema: nil,
			want:   DefaultResponseFormatName,
		},
		{
			name:   "no title",
			schema: &genai.Schema{Type: genai.TypeObject},
			want:   DefaultResponseFormatName,
		},
		{
			name:   "title already a legal name",
			schema: &genai.Schema{Title: "Custom-Title_2"},
			want:   "Custom-Title_2",
		},
		{
			name:   "spaces become underscores",
			schema: &genai.Schema{Title: "Weather Report"},
			want:   "Weather_Report",
		},
		{
			name:   "a run of illegal characters collapses to one underscore",
			schema: &genai.Schema{Title: "Weather  Report (v2)"},
			want:   "Weather_Report_v2",
		},
		{
			name:   "leading and trailing illegal characters are dropped",
			schema: &genai.Schema{Title: " .Weather. "},
			want:   "Weather",
		},
		{
			name:   "a title longer than 64 characters is truncated",
			schema: &genai.Schema{Title: strings.Repeat("Extraction", 8)},
			want:   strings.Repeat("Extraction", 8)[:64],
		},
		{
			name:   "a title with no legal character falls back to the default",
			schema: &genai.Schema{Title: "天気予報"},
			want:   DefaultResponseFormatName,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ResponseFormatName(tc.schema)
			if got != tc.want {
				t.Errorf("ResponseFormatName() = %q, want %q", got, tc.want)
			}
			if !openAIFormatName.MatchString(got) {
				t.Errorf("ResponseFormatName() = %q, want it to match %s", got, openAIFormatName)
			}
		})
	}
}

// A title made of multi-byte runes must not be truncated mid-rune into a name
// the pattern rejects, whatever the caller writes.
func TestResponseFormatNameAlwaysMatchesTheOpenAIPattern(t *testing.T) {
	titles := []string{
		"",
		" ",
		"_",
		"-",
		strings.Repeat("é", 100),
		strings.Repeat("a é", 100),
		strings.Repeat("天", 64) + "a",
		"\x00\n\t",
		strings.Repeat("-", 200),
	}
	for _, title := range titles {
		got := ResponseFormatName(&genai.Schema{Title: title})
		if !openAIFormatName.MatchString(got) {
			t.Errorf("ResponseFormatName(%q) = %q, want it to match %s", title, got, openAIFormatName)
		}
	}
}
