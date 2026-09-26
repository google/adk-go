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

package imagegen

import (
	"bytes"
	"testing"

	"google.golang.org/genai"
)

func TestImageBytes(t *testing.T) {
	tests := []struct {
		name     string
		response *genai.GenerateImagesResponse
		want     []byte
		wantMIME string
		wantErr  string
	}{
		{
			name:    "nil response",
			wantErr: "image generation returned no images",
		},
		{
			name:     "empty response",
			response: &genai.GenerateImagesResponse{},
			wantErr:  "image generation returned no images",
		},
		{
			name: "nil generated image",
			response: &genai.GenerateImagesResponse{
				GeneratedImages: []*genai.GeneratedImage{nil},
			},
			wantErr: "image generation returned no usable image data",
		},
		{
			name: "first filtering reason",
			response: &genai.GenerateImagesResponse{
				GeneratedImages: []*genai.GeneratedImage{
					{
						Image:             &genai.Image{},
						RAIFilteredReason: "blocked",
					},
					{RAIFilteredReason: "later reason"},
				},
			},
			wantErr: "image generation returned no image: blocked",
		},
		{
			name: "whitespace filtering reason ignored",
			response: &genai.GenerateImagesResponse{
				GeneratedImages: []*genai.GeneratedImage{
					{RAIFilteredReason: " \t\n "},
				},
			},
			wantErr: "image generation returned no usable image data",
		},
		{
			name: "whitespace reason yields to a later real one",
			response: &genai.GenerateImagesResponse{
				GeneratedImages: []*genai.GeneratedImage{
					{RAIFilteredReason: "   "},
					{RAIFilteredReason: "  blocked  "},
				},
			},
			wantErr: "image generation returned no image: blocked",
		},
		{
			name: "empty image data",
			response: &genai.GenerateImagesResponse{
				GeneratedImages: []*genai.GeneratedImage{
					{Image: &genai.Image{}},
				},
			},
			wantErr: "image generation returned no usable image data",
		},
		{
			name: "empty but non-nil image bytes with a reason",
			response: &genai.GenerateImagesResponse{
				GeneratedImages: []*genai.GeneratedImage{
					{
						Image:             &genai.Image{ImageBytes: []byte{}},
						RAIFilteredReason: "blocked",
					},
				},
			},
			wantErr: "image generation returned no image: blocked",
		},
		{
			name: "GCS URI without inline image data",
			response: &genai.GenerateImagesResponse{
				GeneratedImages: []*genai.GeneratedImage{
					{Image: &genai.Image{GCSURI: "gs://bucket/image.png"}},
				},
			},
			wantErr: "image generation returned no usable image data",
		},
		{
			name: "first usable image",
			response: &genai.GenerateImagesResponse{
				GeneratedImages: []*genai.GeneratedImage{
					nil,
					{RAIFilteredReason: "filtered"},
					{Image: &genai.Image{ImageBytes: []byte("first image"), MIMEType: "image/jpeg"}},
					{Image: &genai.Image{ImageBytes: []byte("second image")}},
				},
			},
			want:     []byte("first image"),
			wantMIME: "image/jpeg",
		},
		{
			name: "MIME type omitted by the response",
			response: &genai.GenerateImagesResponse{
				GeneratedImages: []*genai.GeneratedImage{
					{Image: &genai.Image{ImageBytes: []byte("bytes")}},
				},
			},
			want:     []byte("bytes"),
			wantMIME: "",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, mime, err := ImageBytes(test.response)
			if test.wantErr != "" {
				if err == nil || err.Error() != test.wantErr {
					t.Fatalf("ImageBytes() error = %v, want %q", err, test.wantErr)
				}
				if got != nil || mime != "" {
					t.Errorf("ImageBytes() = (%q, %q), want (nil, \"\") on error", got, mime)
				}
				return
			}
			if err != nil {
				t.Fatalf("ImageBytes() unexpected error: %v", err)
			}
			if !bytes.Equal(got, test.want) {
				t.Errorf("ImageBytes() bytes = %q, want %q", got, test.want)
			}
			if mime != test.wantMIME {
				t.Errorf("ImageBytes() mime = %q, want %q", mime, test.wantMIME)
			}
		})
	}
}
