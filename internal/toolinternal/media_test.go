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

package toolinternal_test

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/toolinternal"
)

func TestMediaPart(t *testing.T) {
	inlineAndFile := genai.Part{
		InlineData: &genai.Blob{Data: []byte("inline"), MIMEType: "image/png", DisplayName: "drop-me.png"},
		FileData:   &genai.FileData{FileURI: "gs://b/f", MIMEType: "image/jpeg", DisplayName: "drop-file.jpg"},
	}
	inlineWant := &genai.FunctionResponsePart{InlineData: &genai.FunctionResponseBlob{Data: []byte("inline"), MIMEType: "image/png"}}
	tests := []struct {
		name string
		v    any
		want *genai.FunctionResponsePart
	}{
		{
			name: "empty but non-nil inline data is media",
			v:    &genai.Part{InlineData: &genai.Blob{Data: []byte{}, MIMEType: "image/png"}},
			want: &genai.FunctionResponsePart{InlineData: &genai.FunctionResponseBlob{Data: []byte{}, MIMEType: "image/png"}},
		},
		{
			name: "inline data wins over file data and display name is dropped",
			v:    &inlineAndFile,
			want: inlineWant,
		},
		{
			name: "part value",
			v:    inlineAndFile,
			want: inlineWant,
		},
		{
			name: "file data without display name",
			v:    &genai.Part{FileData: &genai.FileData{FileURI: "gs://b/f", MIMEType: "image/jpeg", DisplayName: "chart.png"}},
			want: &genai.FunctionResponsePart{FileData: &genai.FunctionResponseFileData{FileURI: "gs://b/f", MIMEType: "image/jpeg"}},
		},
		{name: "text part", v: genai.NewPartFromText("hi")},
		{name: "inline data with nil bytes", v: &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png"}}},
		{name: "inline data without mime type", v: &genai.Part{InlineData: &genai.Blob{Data: []byte("x")}}},
		{name: "file data without uri", v: &genai.Part{FileData: &genai.FileData{MIMEType: "image/png"}}},
		{name: "file data without mime type", v: &genai.Part{FileData: &genai.FileData{FileURI: "gs://bucket/chart"}}},
		{name: "nil part pointer", v: (*genai.Part)(nil)},
		{name: "not a part", v: map[string]any{"a": 1}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, toolinternal.MediaPart(tc.v)); diff != "" {
				t.Errorf("MediaPart() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestExtractMedia(t *testing.T) {
	png := genai.NewPartFromBytes([]byte("png"), "image/png")
	jpeg := genai.NewPartFromBytes([]byte("jpeg"), "image/jpeg")
	pngPart := &genai.FunctionResponsePart{InlineData: &genai.FunctionResponseBlob{Data: []byte("png"), MIMEType: "image/png"}}
	jpegPart := &genai.FunctionResponsePart{InlineData: &genai.FunctionResponseBlob{Data: []byte("jpeg"), MIMEType: "image/jpeg"}}
	part := func(data string) *genai.FunctionResponsePart {
		return &genai.FunctionResponsePart{InlineData: &genai.FunctionResponseBlob{Data: []byte(data), MIMEType: "image/png"}}
	}
	tests := []struct {
		name          string
		result        any
		wantRemaining any
		wantParts     []*genai.FunctionResponsePart
	}{
		{
			name:          "single part",
			result:        png,
			wantRemaining: map[string]any{},
			wantParts:     []*genai.FunctionResponsePart{pngPart},
		},
		{
			name: "map keys are visited in sorted order",
			result: map[string]any{
				"z_img": genai.NewPartFromBytes([]byte("z"), "image/png"),
				"a_img": genai.NewPartFromBytes([]byte("a"), "image/png"),
				"m_img": genai.NewPartFromBytes([]byte("m"), "image/png"),
			},
			wantRemaining: map[string]any{},
			wantParts:     []*genai.FunctionResponsePart{part("a"), part("m"), part("z")},
		},
		{
			name:          "media next to plain data",
			result:        map[string]any{"chart": png, "summary": "up 3%"},
			wantRemaining: map[string]any{"summary": "up 3%"},
			wantParts:     []*genai.FunctionResponsePart{pngPart},
		},
		{
			name:          "typed map",
			result:        map[string]*genai.Part{"chart": png},
			wantRemaining: map[string]any{},
			wantParts:     []*genai.FunctionResponsePart{pngPart},
		},
		{
			name:          "slice emptied of media becomes an empty map",
			result:        []*genai.Part{png, jpeg},
			wantRemaining: map[string]any{},
			wantParts:     []*genai.FunctionResponsePart{pngPart, jpegPart},
		},
		{
			name:          "array",
			result:        [2]*genai.Part{png, jpeg},
			wantRemaining: map[string]any{},
			wantParts:     []*genai.FunctionResponsePart{pngPart, jpegPart},
		},
		{
			name:          "mixed slice keeps plain values in order",
			result:        []any{png, jpeg, "two charts"},
			wantRemaining: []any{"two charts"},
			wantParts:     []*genai.FunctionResponsePart{pngPart, jpegPart},
		},
		{
			name:          "bytes are not a container",
			result:        []byte("raw-bytes"),
			wantRemaining: []byte("raw-bytes"),
		},
		{
			name:          "raw JSON is not a container",
			result:        json.RawMessage(`{"a":1}`),
			wantRemaining: json.RawMessage(`{"a":1}`),
		},
		{
			name:          "inner container without media is kept unchanged",
			result:        map[string]any{"images": []any{}, "ok": true},
			wantRemaining: map[string]any{"images": []any{}, "ok": true},
		},
		{
			name:          "inner container keeps its plain values",
			result:        map[string]any{"items": []any{png, "keep"}, "ok": true},
			wantRemaining: map[string]any{"items": []any{"keep"}, "ok": true},
			wantParts:     []*genai.FunctionResponsePart{pngPart},
		},
		{
			name:          "inner container emptied of media is dropped",
			result:        map[string]any{"images": []*genai.Part{png, jpeg}, "summary": "s"},
			wantRemaining: map[string]any{"summary": "s"},
			wantParts:     []*genai.FunctionResponsePart{pngPart, jpegPart},
		},
		{
			name:          "media two containers deep stays in place",
			result:        map[string]any{"report": map[string]any{"charts": map[string]any{"first": png}}},
			wantRemaining: map[string]any{"report": map[string]any{"charts": map[string]any{"first": png}}},
		},
		{
			name:          "struct is not a container",
			result:        struct{ Chart *genai.Part }{Chart: png},
			wantRemaining: struct{ Chart *genai.Part }{Chart: png},
		},
		{
			name:          "non-media part stays",
			result:        genai.NewPartFromText("hi"),
			wantRemaining: genai.NewPartFromText("hi"),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			remaining, parts := toolinternal.ExtractMedia(tc.result)
			if diff := cmp.Diff(tc.wantRemaining, remaining); diff != "" {
				t.Errorf("ExtractMedia() remaining mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantParts, parts); diff != "" {
				t.Errorf("ExtractMedia() parts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestCarriesPart(t *testing.T) {
	png := genai.NewPartFromBytes([]byte("png"), "image/png")
	cyclic := map[string]any{"a": 1}
	cyclic["self"] = cyclic
	cyclicSlice := []any{"x", nil}
	cyclicSlice[1] = cyclicSlice
	tests := []struct {
		name string
		v    any
		want bool
	}{
		{name: "part pointer", v: png, want: true},
		{name: "part value", v: *png, want: true},
		{name: "non-media part", v: genai.NewPartFromText("hi"), want: true},
		{name: "nil part pointer", v: (*genai.Part)(nil), want: false},
		{name: "nil", v: nil, want: false},
		{name: "plain map", v: map[string]any{"a": 1}, want: false},
		{name: "part three containers deep", v: map[string]any{"report": map[string]any{"charts": map[string]any{"first": png}}}, want: true},
		{name: "part in a slice of maps", v: []any{map[string]any{"image": png}}, want: true},
		{name: "part in an array", v: [1]*genai.Part{png}, want: true},
		{name: "part inside a struct", v: struct{ Chart *genai.Part }{Chart: png}, want: false},
		{name: "cyclic map", v: cyclic, want: false},
		{name: "cyclic slice", v: cyclicSlice, want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := toolinternal.CarriesPart(tc.v); got != tc.want {
				t.Errorf("CarriesPart() = %v, want %v", got, tc.want)
			}
		})
	}
}
