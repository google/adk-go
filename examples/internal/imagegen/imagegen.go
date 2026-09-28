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

// Package imagegen contains helpers shared by image generation examples.
package imagegen

import (
	"errors"
	"fmt"
	"strings"

	"google.golang.org/genai"
)

// ImageBytes returns the bytes and MIME type of the first usable image in a
// generation response. If the response contains no generated images, it returns
// an error reporting that no images were returned. If the entries contain no
// usable image data but include RAI filtering reasons, the error includes the
// first reason that remains non-empty after trimming surrounding whitespace.
// Whitespace-only reasons are ignored. Otherwise, an error reports that the
// entries contain no usable image data.
//
// ImageBytes assumes the image is delivered as inline bytes (Image.ImageBytes);
// an entry carrying only a GCS URI is treated as having no usable image data.
//
// On error, ImageBytes returns nil bytes and an empty MIME type. On success,
// the returned bytes alias the image data in the SDK response; they are not
// copied. The MIME type is returned as-is from the response and may be empty,
// so callers should fall back to a default when it is empty.
func ImageBytes(response *genai.GenerateImagesResponse) ([]byte, string, error) {
	if response == nil || len(response.GeneratedImages) == 0 {
		return nil, "", errors.New("image generation returned no images")
	}

	var filteredReason string
	for _, generatedImage := range response.GeneratedImages {
		if generatedImage == nil {
			continue
		}
		if generatedImage.Image != nil && len(generatedImage.Image.ImageBytes) > 0 {
			return generatedImage.Image.ImageBytes, generatedImage.Image.MIMEType, nil
		}
		if reason := strings.TrimSpace(generatedImage.RAIFilteredReason); filteredReason == "" && reason != "" {
			filteredReason = reason
		}
	}

	if filteredReason != "" {
		return nil, "", fmt.Errorf("image generation returned no image: %s", filteredReason)
	}
	return nil, "", errors.New("image generation returned no usable image data")
}
