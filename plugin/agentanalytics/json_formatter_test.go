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

package agentanalytics

import (
	"encoding/json"
	"reflect"
	"testing"

	"google.golang.org/genai"
)

func TestFormatContentParts(t *testing.T) {
	tests := []struct {
		name      string
		content   *genai.Content
		maxLength int
		want      []map[string]any
	}{
		{
			name:      "nil content",
			content:   nil,
			maxLength: 10,
			want:      []map[string]any{},
		},
		{
			name: "empty parts",
			content: &genai.Content{
				Parts: nil,
			},
			maxLength: 10,
			want:      []map[string]any{},
		},
		{
			name: "various parts",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "this is a very long text"},
					{InlineData: &genai.Blob{MIMEType: "image/png"}},
					{FileData: &genai.FileData{MIMEType: "video/mp4", FileURI: "gs://bucket/file.mp4"}},
					{Text: "short"},
				},
			},
			maxLength: 10,
			want: []map[string]any{
				{
					"part_index":   0,
					"storage_mode": "INLINE",
					"mime_type":    "text/plain",
					"text":         "this is a ...[truncated]",
				},
				{
					"part_index":   1,
					"storage_mode": "INLINE",
					"mime_type":    "image/png",
					"text":         "[BINARY DATA]",
				},
				{
					"part_index":   2,
					"storage_mode": "EXTERNAL_URI",
					"mime_type":    "video/mp4",
					"uri":          "gs://bucket/file.mp4",
				},
				{
					"part_index":   3,
					"storage_mode": "INLINE",
					"mime_type":    "text/plain",
					"text":         "short",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FormatContentParts(tt.content, tt.maxLength)
			if len(got) != len(tt.want) {
				t.Fatalf("FormatContentParts() len = %v, want %v", len(got), len(tt.want))
			}
			for i := range got {
				if !reflect.DeepEqual(got[i], tt.want[i]) {
					t.Errorf("FormatContentParts() got[%d] = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

type testStruct struct {
	ExportedString      string `json:"exported_string"`
	IgnoredField        string `json:"-"`
	UnexportedField     string
	ExportedInt         int
	Nested              *nestedStruct `json:"nested,omitempty"`
	ExportedStringNoTag string
}

type nestedStruct struct {
	Value string
}

func TestSmartTruncate(t *testing.T) {
	tests := []struct {
		name      string
		obj       any
		maxLength int
		wantStr   string
		wantTrunc bool
	}{
		{
			name:      "nil",
			obj:       nil,
			maxLength: 10,
			wantStr:   `null`,
			wantTrunc: false,
		},
		{
			name:      "short string",
			obj:       "short",
			maxLength: 10,
			wantStr:   `"short"`,
			wantTrunc: false,
		},
		{
			name:      "simple string",
			obj:       "this is a very long string",
			maxLength: 10,
			wantStr:   `"this is a ...[truncated]"`,
			wantTrunc: true,
		},
		{
			name: "map with strings",
			obj: map[string]any{
				"key1": "long long long string",
				"key2": "short",
			},
			maxLength: 10,
			wantStr:   `{"key1":"long long ...[truncated]","key2":"short"}`,
			wantTrunc: true,
		},
		{
			name:      "slice of strings",
			obj:       []any{"long long long string", "short"},
			maxLength: 10,
			wantStr:   `["long long ...[truncated]","short"]`,
			wantTrunc: true,
		},
		{
			name: "complex struct",
			obj: &testStruct{
				ExportedString:      "long string here",
				IgnoredField:        "should not appear",
				UnexportedField:     "should not appear",
				ExportedInt:         42,
				ExportedStringNoTag: "another long string",
				Nested: &nestedStruct{
					Value: "nested long string",
				},
			},
			maxLength: 10,
			wantStr:   `{"ExportedInt":42,"ExportedStringNoTag":"another lo...[truncated]","UnexportedField":"should not...[truncated]","exported_string":"long strin...[truncated]","nested":{"Value":"nested lon...[truncated]"}}`,
			wantTrunc: true,
		},
		{
			name: "typed slice",
			obj: []string{
				"very long string 1",
				"short",
			},
			maxLength: 10,
			wantStr:   `["very long ...[truncated]","short"]`,
			wantTrunc: true,
		},
		{
			name: "map with non-string keys",
			obj: map[int]string{
				1: "very long string 1",
			},
			maxLength: 10,
			wantStr:   `{"1":"very long ...[truncated]"}`,
			wantTrunc: true,
		},
		{
			name: "array",
			obj: [1]string{
				"very long string",
			},
			maxLength: 10,
			wantStr:   `["very long ...[truncated]"]`,
			wantTrunc: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotBytes, truncated, err := SmartTruncate(tt.obj, tt.maxLength)
			if err != nil {
				t.Fatalf("SmartTruncate() error = %v", err)
			}
			if truncated != tt.wantTrunc {
				t.Errorf("SmartTruncate() truncated = %v, want %v", truncated, tt.wantTrunc)
			}

			// We use json.Unmarshal to canonicalize the JSON before comparison because map key order is non-deterministic
			var gotObj, wantObj any
			if string(gotBytes) != tt.wantStr {
				// To handle non-deterministic map ordering, unmarshal then compare
				if err := json.Unmarshal(gotBytes, &gotObj); err != nil {
					t.Fatalf("Failed to unmarshal gotBytes (%s): %v", string(gotBytes), err)
				}
				if err := json.Unmarshal([]byte(tt.wantStr), &wantObj); err != nil {
					t.Fatalf("Failed to unmarshal wantStr (%s): %v", tt.wantStr, err)
				}

				if gotObj == nil || wantObj == nil || !reflect.DeepEqual(gotObj, wantObj) {
					t.Errorf("SmartTruncate() = %s, want %s", string(gotBytes), tt.wantStr)
				}
			}
		})
	}
}
