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

package artifact

import (
	"errors"
	"os"
	"slices"
	"testing"

	"github.com/google/adk-go"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

func TestArtifactKey(t *testing.T) {
	key := artifactKey{
		AppName:   "testapp",
		UserID:    "testuser",
		SessionID: "testsession",
		FileName:  "testfile",
		Version:   123,
	}
	var key2 artifactKey
	key2.Decode(key.Encode())
	if diff := cmp.Diff(key, key2); diff != "" {
		t.Errorf("key mismatch (-want +got):\n%s", diff)
	}
}

func TestInMemoryArtifactService(t *testing.T) {
	ctx := t.Context()
	appName := "testapp"
	userID := "testuser"
	sessionID := "testsession"

	srv := &InMemoryArtifactService{}

	// Save these artifacts for later subtests.
	testData := []struct {
		fileName string
		version  int64
		artifact *genai.Part
	}{
		// file1.
		{"file1", 1, genai.NewPartFromText("file v1")},
		{"file1", 2, genai.NewPartFromText("file v2")},
		{"file1", 3, genai.NewPartFromText("file v3")},
		// file2.
		{"file2", 1, genai.NewPartFromText("file v3")},
	}

	t.Log("Save file1 and file2")
	for i, data := range testData {
		wantVersion := data.version
		gotVersion, err := srv.Save(ctx, appName, userID, sessionID, data.fileName, data.artifact)
		if err != nil || gotVersion != wantVersion {
			t.Errorf("[%d] Save() = (%v, %v), want (%v, nil)", i, gotVersion, err, wantVersion)
			continue
		}
	}

	t.Run("Load", func(t *testing.T) {
		fileName := "file1"
		for _, tc := range []struct {
			name string
			opt  *adk.ArtifactLoadOption
			want *genai.Part
		}{
			{"latest", nil, genai.NewPartFromText("file v3")},
			{"ver=1", &adk.ArtifactLoadOption{Version: 1}, genai.NewPartFromText("file v1")},
			{"ver=2", &adk.ArtifactLoadOption{Version: 2}, genai.NewPartFromText("file v2")},
		} {
			got, err := srv.Load(ctx, appName, userID, sessionID, fileName, tc.opt)
			if err != nil || !cmp.Equal(got, tc.want) {
				t.Errorf("Load(%v) = (%v, %v), want (%v, nil)", tc.opt, got, err, tc.want)
			}
		}
	})

	t.Run("List", func(t *testing.T) {
		got, err := srv.List(ctx, appName, userID, sessionID)
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
		slices.Sort(got)
		want := []string{"file1", "file2"} // testData has two files.
		if diff := cmp.Diff(got, want); diff != "" {
			t.Errorf("List() = %v, want %v", got, want)
		}
	})

	t.Run("Versions", func(t *testing.T) {
		got, err := srv.Versions(ctx, appName, userID, sessionID, "file1")
		if err != nil {
			t.Fatalf("Versions() failed: %v", err)
		}
		slices.Sort(got)
		want := []int64{1, 2, 3}
		if diff := cmp.Diff(got, want); diff != "" {
			t.Errorf("Versions('file1') = %v, want %v", got, want)
		}
	})

	t.Log("Delete file1")
	if err := srv.Delete(ctx, appName, userID, sessionID, "file1"); err != nil {
		t.Fatalf("Delete() failed: %v", err)
	}

	t.Run("LoadAfterDelete", func(t *testing.T) {
		got, err := srv.Load(ctx, appName, userID, sessionID, "file1", nil)
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("Load('file1') = (%v, %v, want error(%v)", got, err, os.ErrNotExist)
		}
	})

	t.Run("ListAfterDelete", func(t *testing.T) {
		got, err := srv.List(ctx, appName, userID, sessionID)
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
		slices.Sort(got)
		want := []string{"file2"} // testData has two files.
		if diff := cmp.Diff(got, want); diff != "" {
			t.Errorf("List() = %v, want %v", got, want)
		}
	})

	t.Run("VersionsAfterDelete", func(t *testing.T) {
		got, err := srv.Versions(ctx, appName, userID, sessionID, "file1")
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("Versions('file1') = (%v, %v), want error(%v)", got, err, os.ErrNotExist)
		}
	})
}

func TestInMemoryArtifactService_Empty(t *testing.T) {
	ctx := t.Context()
	srv := &InMemoryArtifactService{}
	t.Run("Load", func(t *testing.T) {
		got, err := srv.Load(ctx, "app", "user", "session", "file", nil)
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("List() = (%v, %v), want error(%v)", got, err, os.ErrNotExist)
		}
	})
	t.Run("List", func(t *testing.T) {
		_, err := srv.List(ctx, "app", "user", "session")
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
	})
	t.Run("Delete", func(t *testing.T) {
		err := srv.Delete(ctx, "app", "user", "session", "file1")
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("Delete() = %v, want error(%v)", err, os.ErrNotExist)
		}
	})
	t.Run("Versions", func(t *testing.T) {
		got, err := srv.Versions(ctx, "app", "user", "session", "file1")
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("Versions() = (%v, %v), want error(%v)", got, err, os.ErrNotExist)
		}
	})
}
