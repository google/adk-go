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
					{RAIFilteredReason: "blocked"},
					{
						Image:             &genai.Image{},
						RAIFilteredReason: "later reason",
					},
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
			name: "first usable image",
			response: &genai.GenerateImagesResponse{
				GeneratedImages: []*genai.GeneratedImage{
					nil,
					{RAIFilteredReason: "filtered"},
					{Image: &genai.Image{ImageBytes: []byte("first image")}},
					{Image: &genai.Image{ImageBytes: []byte("second image")}},
				},
			},
			want: []byte("first image"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := ImageBytes(test.response)
			if test.wantErr != "" {
				if err == nil || err.Error() != test.wantErr {
					t.Fatalf("ImageBytes() error = %v, want %q", err, test.wantErr)
				}
				if got != nil {
					t.Errorf("ImageBytes() = %q, want nil on error", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ImageBytes() unexpected error: %v", err)
			}
			if !bytes.Equal(got, test.want) {
				t.Errorf("ImageBytes() = %q, want %q", got, test.want)
			}
		})
	}
}
