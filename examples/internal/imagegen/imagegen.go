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

	"google.golang.org/genai"
)

// ImageBytes returns the first usable image in a generation response.
func ImageBytes(response *genai.GenerateImagesResponse) ([]byte, error) {
	if response == nil || len(response.GeneratedImages) == 0 {
		return nil, errors.New("image generation returned no images")
	}

	var filteredReason string
	for _, generatedImage := range response.GeneratedImages {
		if generatedImage == nil {
			continue
		}
		if generatedImage.Image != nil && len(generatedImage.Image.ImageBytes) > 0 {
			return generatedImage.Image.ImageBytes, nil
		}
		if filteredReason == "" {
			filteredReason = generatedImage.RAIFilteredReason
		}
	}

	if filteredReason != "" {
		return nil, fmt.Errorf("image generation returned no image: %s", filteredReason)
	}
	return nil, errors.New("image generation returned no usable image data")
}
