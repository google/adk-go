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
	"io/fs"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/types"
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
	_ = key2.Decode(key.Encode())
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
		got, err := srv.Save(ctx, &types.ArtifactSaveRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: data.fileName,
			Part: data.artifact,
		})
		if err != nil || got.Version != wantVersion {
			t.Errorf("[%d] Save() = (%v, %v), want (%v, nil)", i, got.Version, err, wantVersion)
			continue
		}
	}

	t.Run("Load", func(t *testing.T) {
		fileName := "file1"
		for _, tc := range []struct {
			name    string
			version int64
			want    *genai.Part
		}{
			{"latest", 0, genai.NewPartFromText("file v3")},
			{"ver=1", 1, genai.NewPartFromText("file v1")},
			{"ver=2", 2, genai.NewPartFromText("file v2")},
		} {
			got, err := srv.Load(ctx, &types.ArtifactLoadRequest{
				AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName,
				Version: tc.version,
			})
			if err != nil || !cmp.Equal(got.Part, tc.want) {
				t.Errorf("Load(%v) = (%v, %v), want (%v, nil)", tc.version, got.Part, err, tc.want)
			}
		}
	})

	t.Run("List", func(t *testing.T) {
		resp, err := srv.List(ctx, &types.ArtifactListRequest{
			AppName: appName, UserID: userID, SessionID: sessionID,
		})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
		got := resp.FileNames
		slices.Sort(got)
		want := []string{"file1", "file2"} // testData has two files.
		if diff := cmp.Diff(got, want); diff != "" {
			t.Errorf("List() = %v, want %v", got, want)
		}
	})

	t.Run("Versions", func(t *testing.T) {
		resp, err := srv.Versions(ctx, &types.ArtifactVersionsRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
		})
		if err != nil {
			t.Fatalf("Versions() failed: %v", err)
		}
		got := resp.Versions
		slices.Sort(got)
		want := []int64{1, 2, 3}
		if diff := cmp.Diff(got, want); diff != "" {
			t.Errorf("Versions('file1') = %v, want %v", got, want)
		}
	})

	t.Log("Delete file1 version 3")
	if err := srv.Delete(ctx, &types.ArtifactDeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
		Version: 3,
	}); err != nil {
		t.Fatalf("Delete(file1@v3) failed: %v", err)
	}

	t.Run("LoadAfterDeleteVersion3", func(t *testing.T) {
		resp, err := srv.Load(ctx, &types.ArtifactLoadRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
		})
		if err != nil {
			t.Fatalf("Load('file1') failed: %v", err)
		}
		got := resp.Part
		want := genai.NewPartFromText("file v2")
		if diff := cmp.Diff(got, want); diff != "" {
			t.Fatalf("Load('file1') = (%v, %v), want (%v, nil)", got, err, want)
		}
	})

	if err := srv.Delete(ctx, &types.ArtifactDeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
	}); err != nil {
		t.Fatalf("Delete(file1) failed: %v", err)
	}

	t.Run("LoadAfterDelete", func(t *testing.T) {
		got, err := srv.Load(ctx, &types.ArtifactLoadRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("Load('file1') = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})

	t.Run("ListAfterDelete", func(t *testing.T) {
		resp, err := srv.List(ctx, &types.ArtifactListRequest{
			AppName: appName, UserID: userID, SessionID: sessionID,
		})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
		got := resp.FileNames
		slices.Sort(got)
		want := []string{"file2"} // testData has two files.
		if diff := cmp.Diff(got, want); diff != "" {
			t.Errorf("List() = %v, want %v", got, want)
		}
	})

	t.Run("VersionsAfterDelete", func(t *testing.T) {
		got, err := srv.Versions(ctx, &types.ArtifactVersionsRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("Versions('file1') = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})
}

func TestInMemoryArtifactService_Empty(t *testing.T) {
	ctx := t.Context()
	srv := &InMemoryArtifactService{}
	t.Run("Load", func(t *testing.T) {
		got, err := srv.Load(ctx, &types.ArtifactLoadRequest{
			AppName: "app", UserID: "user", SessionID: "session", FileName: "file"})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("List() = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})
	t.Run("List", func(t *testing.T) {
		_, err := srv.List(ctx, &types.ArtifactListRequest{
			AppName: "app", UserID: "user", SessionID: "session"})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
	})
	t.Run("Delete", func(t *testing.T) {
		err := srv.Delete(ctx, &types.ArtifactDeleteRequest{
			AppName: "app", UserID: "user", SessionID: "sesion", FileName: "file1"})
		if err != nil {
			t.Fatalf("Delete() failed: %v", err)
		}
	})
	t.Run("Versions", func(t *testing.T) {
		got, err := srv.Versions(ctx, &types.ArtifactVersionsRequest{
			AppName: "app", UserID: "user", SessionID: "session", FileName: "file1"})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("Versions() = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})
}
