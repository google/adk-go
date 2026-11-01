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

package completions

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"

	"google.golang.org/adk/v2/model/openaimodel/internal/shared"
)

// wireContent renders the content of the one user message a request builds, as
// the JSON the SDK would send.
func wireContent(t *testing.T, req *model.LLMRequest) string {
	t.Helper()
	msgs := wireMessages(t, requestWire(t, req))
	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1: %v", len(msgs), msgs)
	}
	if msgs[0]["role"] != "user" {
		t.Fatalf("role = %v, want user", msgs[0]["role"])
	}
	raw, err := json.Marshal(msgs[0]["content"])
	if err != nil {
		t.Fatalf("marshal content: %v", err)
	}
	return string(raw)
}

func TestBuildParams_MediaOnTheWire(t *testing.T) {
	png := []byte{0x89, 'P', 'N', 'G'}
	pdf := []byte("%PDF")
	const pngURL = "data:image/png;base64,iVBORw=="
	const pdfURL = "data:application/pdf;base64,JVBERg=="
	const image = `{"image_url":{"detail":"auto","url":"` + pngURL + `"},"type":"image_url"}`

	for _, tt := range []struct {
		name  string
		parts []*genai.Part
		want  string
	}{
		{
			name: "text then inline image keeps its order",
			parts: []*genai.Part{
				{Text: "describe this:"},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: png}},
			},
			want: `[{"text":"describe this:","type":"text"},` + image + `]`,
		},
		{
			name: "image then text keeps its order",
			parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: png}},
				{Text: "what is it?"},
			},
			want: `[` + image + `,{"text":"what is it?","type":"text"}]`,
		},
		{
			name:  "text and media on one part: text first",
			parts: []*genai.Part{{Text: "look", InlineData: &genai.Blob{MIMEType: "image/png", Data: png}}},
			want:  `[{"text":"look","type":"text"},` + image + `]`,
		},
		{
			name:  "image by url",
			parts: []*genai.Part{{FileData: &genai.FileData{MIMEType: "image/jpeg", FileURI: "https://example.com/a.jpg"}}},
			want:  `[{"image_url":{"detail":"auto","url":"https://example.com/a.jpg"},"type":"image_url"}]`,
		},
		{
			name:  "inline file with a display name",
			parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: pdf, DisplayName: "report.pdf"}}},
			want:  `[{"file":{"file_data":"` + pdfURL + `","filename":"report.pdf"},"type":"file"}]`,
		},
		{
			name:  "inline file without a display name gets the shared fallback",
			parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: pdf}}},
			want:  `[{"file":{"file_data":"` + pdfURL + `","filename":"inline_data"},"type":"file"}]`,
		},
		{
			name:  "uploaded file is a file id with no filename",
			parts: []*genai.Part{{FileData: &genai.FileData{MIMEType: "application/pdf", FileURI: "file-abc123", DisplayName: "report.pdf"}}},
			want:  `[{"file":{"file_id":"file-abc123"},"type":"file"}]`,
		},
		{
			name: "text between two media parts",
			parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: png}},
				{Text: "versus"},
				{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: pdf, DisplayName: "b.pdf"}},
			},
			want: `[` + image + `,{"text":"versus","type":"text"},{"file":{"file_data":"` + pdfURL + `","filename":"b.pdf"},"type":"file"}]`,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := wireContent(t, userReq(tt.parts...))
			if got != tt.want {
				t.Fatalf("content =\n%s\nwant\n%s", got, tt.want)
			}
		})
	}
}

func TestBuildParams_MediaRefused(t *testing.T) {
	const secretURL = "https://example.com/doc.pdf?X-Amz-Signature=SECRET"
	png := &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}}

	for _, tt := range []struct {
		name    string
		req     *model.LLMRequest
		wantIs  error
		wantMsg string
	}{
		{
			name: "audio is named by the type the part declared",
			req:  userReq(&genai.Part{InlineData: &genai.Blob{MIMEType: "audio/wav", Data: []byte{1}}}),
			// the MIME type, not the field it would have gone in
			wantIs:  shared.ErrUnsupportedMIMEType,
			wantMsg: "audio/wav",
		},
		{
			name:    "video",
			req:     userReq(&genai.Part{FileData: &genai.FileData{MIMEType: "video/mp4", FileURI: "https://example.com/v.mp4"}}),
			wantIs:  shared.ErrUnsupportedMIMEType,
			wantMsg: "video/mp4",
		},
		{
			name:    "uploaded image has no image_url field",
			req:     userReq(&genai.Part{FileData: &genai.FileData{MIMEType: "image/png", FileURI: "file-abc123"}}),
			wantMsg: "unsupported media source for an image on Chat Completions: file_id",
		},
		{
			name:    "file by url has no field and its uri stays out of the error",
			req:     userReq(&genai.Part{FileData: &genai.FileData{MIMEType: "application/pdf", FileURI: secretURL}}),
			wantMsg: "unsupported media source for a file on Chat Completions: url",
		},
		{
			name:    "media with no mime type",
			req:     userReq(&genai.Part{InlineData: &genai.Blob{Data: []byte{1}}}),
			wantIs:  shared.ErrMediaMIMETypeRequired,
			wantMsg: "inline data",
		},
		{
			name:    "media on a replayed model turn",
			req:     &model.LLMRequest{Contents: []*genai.Content{{Role: "model", Parts: []*genai.Part{png}}}},
			wantIs:  shared.ErrMediaOnAssistantTurn,
			wantMsg: "image/png",
		},
		{
			name:    "media in a system turn",
			req:     &model.LLMRequest{Contents: []*genai.Content{{Role: "system", Parts: []*genai.Part{png}}}},
			wantMsg: "media in a system message is not supported: image/png",
		},
		{
			name:    "media in a developer turn",
			req:     &model.LLMRequest{Contents: []*genai.Content{{Role: "developer", Parts: []*genai.Part{png}}}},
			wantMsg: "media in a developer message is not supported: image/png",
		},
		{
			name: "media beside a function response",
			req: &model.LLMRequest{Contents: []*genai.Content{
				{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "f"}}}},
				{Role: "user", Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{ID: "c1", Name: "f", Response: map[string]any{"ok": true}},
					InlineData:       &genai.Blob{MIMEType: "image/png", Data: []byte{1}},
				}}},
			}},
			wantMsg: "media on a function response part is not supported: image/png",
		},
		{
			name: "function call beside media in a user turn",
			req: &model.LLMRequest{Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "f"}}, png}},
				{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "c1", Name: "f", Response: map[string]any{"ok": true}}}}},
			}},
			wantMsg: "function call in a user turn",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := buildParams("m", tt.req)
			if err == nil {
				t.Fatal("buildParams() err = nil, want a refusal")
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("err = %v, want errors.Is %v", err, tt.wantIs)
			}
			if !strings.Contains(err.Error(), tt.wantMsg) {
				t.Errorf("err = %q, want it to contain %q", err, tt.wantMsg)
			}
			if strings.Contains(err.Error(), "SECRET") {
				t.Errorf("err = %q leaks the file uri", err)
			}
		})
	}
}

// TestBuildParams_TextOnlyTurnStaysAString pins that the media path left the
// plain-text request as it was: one string, joined with newlines.
func TestBuildParams_TextOnlyTurnStaysAString(t *testing.T) {
	msgs := wireMessages(t, requestWire(t, userReq(&genai.Part{Text: "a"}, &genai.Part{Text: "b"})))
	if got := msgs[0]["content"]; got != "a\nb" {
		t.Fatalf("content = %#v, want the joined string", got)
	}
}
