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

package context

import (
	"context"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/artifact"
	"google.golang.org/adk/session"
)

type fakeArtifacts struct {
	version int64
}

func (f *fakeArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	return &artifact.SaveResponse{Version: f.version}, nil
}

func (f *fakeArtifacts) List(ctx context.Context) (*artifact.ListResponse, error) {
	return nil, nil
}

func (f *fakeArtifacts) Load(ctx context.Context, name string) (*artifact.LoadResponse, error) {
	return nil, nil
}

func (f *fakeArtifacts) LoadVersion(ctx context.Context, name string, version int) (*artifact.LoadResponse, error) {
	return nil, nil
}

func TestInternalArtifactsSaveKeepsNewestVersion(t *testing.T) {
	t.Parallel()

	actions := &session.EventActions{}
	fake := &fakeArtifacts{version: 1}
	ia := &internalArtifacts{
		Artifacts:    fake,
		eventActions: actions,
	}

	ctx := context.Background()
	if _, err := ia.Save(ctx, "file.txt", nil); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}
	if got := actions.ArtifactDelta["file.txt"]; got != 1 {
		t.Fatalf("expected version 1 after first save, got %d", got)
	}

	fake.version = 3
	if _, err := ia.Save(ctx, "file.txt", nil); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}
	if got := actions.ArtifactDelta["file.txt"]; got != 3 {
		t.Fatalf("expected version 3 after newer save, got %d", got)
	}

	fake.version = 2
	if _, err := ia.Save(ctx, "file.txt", nil); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}
	if got := actions.ArtifactDelta["file.txt"]; got != 3 {
		t.Fatalf("expected version 3 after older save, got %d", got)
	}
}
