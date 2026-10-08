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

package main

import (
	"errors"
	"io/fs"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
)

type saveImageContext struct {
	agent.StrictContextMock
	artifacts agent.Artifacts
}

func (c *saveImageContext) Artifacts() agent.Artifacts {
	return c.artifacts
}

func TestSaveImageRejectsMissingInlineData(t *testing.T) {
	tests := []struct {
		name string
		part *genai.Part
	}{
		{
			name: "nil inline data",
			part: genai.NewPartFromText("not image data"),
		},
		{
			name: "empty inline data",
			part: &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png"}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			artifacts := &artifactinternal.Artifacts{
				Service:   artifact.InMemoryService(),
				AppName:   "test-app",
				UserID:    "test-user",
				SessionID: "test-session",
			}
			if _, err := artifacts.Save(t.Context(), "image.png", test.part); err != nil {
				t.Fatalf("Save() error = %v", err)
			}
			ctx := &saveImageContext{
				StrictContextMock: agent.NewStrictContextMock(t.Context()),
				artifacts:         artifacts,
			}

			got, err := saveImage(ctx, saveImageInput{Filename: "image.png"})
			if err == nil || err.Error() != `artifact "image.png" has no inline data` {
				t.Fatalf("saveImage() error = %v, want %q", err, `artifact "image.png" has no inline data`)
			}
			if got != (saveImageResult{}) {
				t.Errorf("saveImage() result = %+v, want zero value", got)
			}
		})
	}
}

func TestSaveImageReturnsLoadError(t *testing.T) {
	artifacts := &artifactinternal.Artifacts{
		Service:   artifact.InMemoryService(),
		AppName:   "test-app",
		UserID:    "test-user",
		SessionID: "test-session",
	}
	ctx := &saveImageContext{
		StrictContextMock: agent.NewStrictContextMock(t.Context()),
		artifacts:         artifacts,
	}

	got, err := saveImage(ctx, saveImageInput{Filename: "missing.png"})
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("saveImage() error = %v, want an error matching fs.ErrNotExist", err)
	}
	if got != (saveImageResult{}) {
		t.Errorf("saveImage() result = %+v, want zero value", got)
	}
}
