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
	"errors"
	"reflect"
	"strings"
	"testing"

	"google.golang.org/genai"
)

func TestClassifyMedia(t *testing.T) {
	tests := []struct {
		name string
		part *genai.Part
		want []Media
	}{
		{
			name: "nil part",
			part: nil,
		},
		{
			name: "part without media",
			part: &genai.Part{Text: "just words"},
		},
		{
			name: "inline image",
			part: &genai.Part{InlineData: &genai.Blob{
				MIMEType: "image/png",
				Data:     []byte{0x89, 'P', 'N', 'G'},
			}},
			want: []Media{{
				Kind:     MediaKindImage,
				Source:   MediaSourceData,
				MIMEType: "image/png",
				Data:     []byte{0x89, 'P', 'N', 'G'},
			}},
		},
		{
			name: "inline pdf keeps its display name",
			part: &genai.Part{InlineData: &genai.Blob{
				MIMEType:    "application/pdf",
				DisplayName: "invoice.pdf",
				Data:        []byte("%PDF-1.7"),
			}},
			want: []Media{{
				Kind:     MediaKindFile,
				Source:   MediaSourceData,
				MIMEType: "application/pdf",
				Filename: "invoice.pdf",
				Data:     []byte("%PDF-1.7"),
			}},
		},
		{
			name: "inline audio is labelled, not rejected here",
			part: &genai.Part{InlineData: &genai.Blob{
				MIMEType: "audio/mpeg",
				Data:     []byte("ID3"),
			}},
			want: []Media{{
				Kind:     MediaKindAudio,
				Source:   MediaSourceData,
				MIMEType: "audio/mpeg",
				Data:     []byte("ID3"),
			}},
		},
		{
			name: "inline video is labelled, not rejected here",
			part: &genai.Part{InlineData: &genai.Blob{
				MIMEType: "video/mp4",
				Data:     []byte("ftyp"),
			}},
			want: []Media{{
				Kind:     MediaKindVideo,
				Source:   MediaSourceData,
				MIMEType: "video/mp4",
				Data:     []byte("ftyp"),
			}},
		},
		{
			// The point of splitting kind from source: an uploaded image is an
			// image whose bytes are a file id, and the field an endpoint
			// reaches for follows the source, not the kind.
			name: "image by file id is a file id, not a url",
			part: &genai.Part{FileData: &genai.FileData{
				MIMEType: "image/jpeg",
				FileURI:  "file-abc123",
			}},
			want: []Media{{
				Kind:     MediaKindImage,
				Source:   MediaSourceFileID,
				MIMEType: "image/jpeg",
				FileID:   "file-abc123",
			}},
		},
		{
			name: "image by url",
			part: &genai.Part{FileData: &genai.FileData{
				MIMEType: "image/webp",
				FileURI:  "https://example.test/cat.webp",
			}},
			want: []Media{{
				Kind:     MediaKindImage,
				Source:   MediaSourceURL,
				MIMEType: "image/webp",
				URL:      "https://example.test/cat.webp",
			}},
		},
		{
			name: "pdf by url",
			part: &genai.Part{FileData: &genai.FileData{
				MIMEType:    "application/pdf",
				FileURI:     "https://example.test/spec.pdf",
				DisplayName: "spec.pdf",
			}},
			want: []Media{{
				Kind:     MediaKindFile,
				Source:   MediaSourceURL,
				MIMEType: "application/pdf",
				Filename: "spec.pdf",
				URL:      "https://example.test/spec.pdf",
			}},
		},
		{
			// Case, surrounding whitespace and trailing parameters are all
			// things a sender may vary without changing what the media is.
			name: "case, padding and parameters do not change the kind",
			part: &genai.Part{InlineData: &genai.Blob{
				MIMEType: "  IMAGE/PNG; charset=binary ",
				Data:     []byte{0x89},
			}},
			want: []Media{{
				Kind:   MediaKindImage,
				Source: MediaSourceData,
				// The declared type is kept verbatim: it is what goes on the
				// wire, and rewriting it would change the request.
				MIMEType: "  IMAGE/PNG; charset=binary ",
				Data:     []byte{0x89},
			}},
		},
		{
			// A part can hold both, and the order has to be the one the
			// request will carry.
			name: "inline data comes before file data",
			part: &genai.Part{
				InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{0x89}},
				FileData:   &genai.FileData{MIMEType: "application/pdf", FileURI: "file-xyz"},
			},
			want: []Media{
				{Kind: MediaKindImage, Source: MediaSourceData, MIMEType: "image/png", Data: []byte{0x89}},
				{Kind: MediaKindFile, Source: MediaSourceFileID, MIMEType: "application/pdf", FileID: "file-xyz"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ClassifyMedia(tt.part)
			if err != nil {
				t.Fatalf("ClassifyMedia() err = %v, want nil", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ClassifyMedia() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestClassifyMedia_Errors(t *testing.T) {
	tests := []struct {
		name    string
		part    *genai.Part
		wantErr error
		// wantNamed is a substring the message must carry, so a rejection says
		// which media it is about rather than only that one was rejected.
		wantNamed string
	}{
		{
			name:      "inline data without a mime type",
			part:      &genai.Part{InlineData: &genai.Blob{Data: []byte{0x89}}},
			wantErr:   ErrMediaMIMETypeRequired,
			wantNamed: "inline data",
		},
		{
			name: "file data without a mime type names the uri",
			part: &genai.Part{FileData: &genai.FileData{
				FileURI: "https://example.test/mystery",
			}},
			wantErr:   ErrMediaMIMETypeRequired,
			wantNamed: "https://example.test/mystery",
		},
		{
			name: "inline data with no bytes names the type",
			part: &genai.Part{InlineData: &genai.Blob{
				MIMEType: "image/png",
			}},
			wantErr:   ErrMediaDataRequired,
			wantNamed: "image/png",
		},
		{
			name:      "file data without a uri",
			part:      &genai.Part{FileData: &genai.FileData{MIMEType: "image/png"}},
			wantErr:   ErrMediaURIRequired,
			wantNamed: "file data",
		},
		{
			// The first field in request order decides the error, so a bad
			// blob is not masked by a good file reference behind it.
			name: "a bad blob is reported even with sound file data behind it",
			part: &genai.Part{
				InlineData: &genai.Blob{Data: []byte{0x89}},
				FileData:   &genai.FileData{MIMEType: "image/png", FileURI: "file-ok"},
			},
			wantErr:   ErrMediaMIMETypeRequired,
			wantNamed: "inline data",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ClassifyMedia(tt.part)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("ClassifyMedia() err = %v, want %v", err, tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantNamed) {
				t.Errorf("ClassifyMedia() err = %q, want it to name %q", err, tt.wantNamed)
			}
			if got != nil {
				t.Errorf("ClassifyMedia() = %+v on error, want nil: a partial list would be sent", got)
			}
		})
	}
}

func TestMediaDataURL(t *testing.T) {
	inline := Media{
		Source:   MediaSourceData,
		MIMEType: "image/png",
		Data:     []byte{0x89, 'P', 'N', 'G'},
	}
	if got, want := inline.DataURL(), "data:image/png;base64,iVBORw=="; got != want {
		t.Errorf("DataURL() = %q, want %q", got, want)
	}

	// Media held elsewhere has no bytes to render, and must not be handed a
	// data URL of the empty string, which a server would read as empty media.
	for _, m := range []Media{
		{Source: MediaSourceFileID, MIMEType: "image/png", FileID: "file-abc"},
		{Source: MediaSourceURL, MIMEType: "image/png", URL: "https://example.test/a.png"},
	} {
		if got := m.DataURL(); got != "" {
			t.Errorf("DataURL() = %q for source %s, want \"\"", got, m.Source)
		}
	}
}

// TestClassifyMedia_ReadsTheFieldsItNames guards against a rename upstream
// quietly emptying this classifier: both fields it reads must still exist on
// genai.Part.
//
// It does not guard the other direction. A media field genai adds later is
// caught by TestUnsupportedPayload_WalksEveryPartField, which reports any
// field the converters do not account for; duplicating that walk here would
// restate it without strengthening it.
func TestClassifyMedia_ReadsTheFieldsItNames(t *testing.T) {
	want := map[string]bool{"InlineData": true, "FileData": true}
	partType := reflect.TypeOf(genai.Part{})
	for i := range partType.NumField() {
		delete(want, partType.Field(i).Name)
	}
	if len(want) != 0 {
		t.Errorf("genai.Part no longer has the media fields %v this classifier reads", want)
	}
}
