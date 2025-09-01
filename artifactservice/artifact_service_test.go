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

package artifactservice

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"slices"
	"testing"

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
	_ = key2.Decode(key.Encode())
	if diff := cmp.Diff(key, key2); diff != "" {
		t.Errorf("key mismatch (-want +got):\n%s", diff)
	}
}

func TestInMemoryArtifactService(t *testing.T) {
	ctx := t.Context()
	srv := Mem()
	_testArtifactService(ctx, t, srv, "_InMemory")
}

func TestGCSArtifactService(t *testing.T) {
	ctx := t.Context()
	srv, err := NewGcsArtifactServiceWithClient(ctx, NewFakeClient(), "new")
	if err != nil {
		t.Fatalf("TestGcsArtifactService() failed: %v", err)
		return
	}
	_testArtifactService(ctx, t, srv, "_GCS")
}

func _testArtifactService(ctx context.Context, t *testing.T, srv Service, testSuffix string) {
	appName := "testapp"
	userID := "testuser"
	sessionID := "testsession"

	// Save these artifacts for later subtests.
	testData := []struct {
		fileName string
		version  int64
		artifact *genai.Part
	}{
		// file1.
		{"file1", 1, genai.NewPartFromBytes([]byte("file v1"), "text/plain")},
		{"file1", 2, genai.NewPartFromBytes([]byte("file v2"), "text/plain")},
		{"file1", 3, genai.NewPartFromBytes([]byte("file v3"), "text/plain")},
		// file2.
		{"file2", 1, genai.NewPartFromBytes([]byte("file v3"), "text/plain")},
	}

	t.Log("Save file1 and file2")
	for i, data := range testData {
		wantVersion := data.version
		got, err := srv.Save(ctx, &SaveRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: data.fileName,
			Part: data.artifact,
		})
		if err != nil || got.Version != wantVersion {
			t.Errorf("[%d] Save() = (%v, %v), want (%v, nil)", i, got.Version, err, wantVersion)
			continue
		}
	}

	t.Run(fmt.Sprintf("Load%s", testSuffix), func(t *testing.T) {
		fileName := "file1"
		for _, tc := range []struct {
			name    string
			version int64
			want    *genai.Part
		}{
			{"latest", 0, genai.NewPartFromBytes([]byte("file v3"), "text/plain")},
			{"ver=1", 1, genai.NewPartFromBytes([]byte("file v1"), "text/plain")},
			{"ver=2", 2, genai.NewPartFromBytes([]byte("file v2"), "text/plain")},
		} {
			got, err := srv.Load(ctx, &LoadRequest{
				AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName,
				Version: tc.version,
			})
			if err != nil || !cmp.Equal(got.Part, tc.want) {
				t.Errorf("Load(%v) = (%v, %v), want (%v, nil)", tc.version, got.Part, err, tc.want)
			}
		}
	})

	t.Run(fmt.Sprintf("List%s", testSuffix), func(t *testing.T) {
		resp, err := srv.List(ctx, &ListRequest{
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

	t.Run(fmt.Sprintf("Versions%s", testSuffix), func(t *testing.T) {
		resp, err := srv.Versions(ctx, &VersionsRequest{
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
	if err := srv.Delete(ctx, &DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
		Version: 3,
	}); err != nil {
		t.Fatalf("Delete(file1@v3) failed: %v", err)
	}

	t.Run(fmt.Sprintf("LoadAfterDeleteVersion3%s", testSuffix), func(t *testing.T) {
		resp, err := srv.Load(ctx, &LoadRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
		})
		if err != nil {
			t.Fatalf("Load('file1') failed: %v", err)
		}
		got := resp.Part
		want := genai.NewPartFromBytes([]byte("file v2"), "text/plain")
		if diff := cmp.Diff(got, want); diff != "" {
			t.Fatalf("Load('file1') = (%v, %v), want (%v, nil)", got, err, want)
		}
	})

	if err := srv.Delete(ctx, &DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
	}); err != nil {
		t.Fatalf("Delete(file1) failed: %v", err)
	}

	t.Run(fmt.Sprintf("LoadAfterDelete%s", testSuffix), func(t *testing.T) {
		got, err := srv.Load(ctx, &LoadRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("Load('file1') = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})

	t.Run(fmt.Sprintf("ListAfterDelete%s", testSuffix), func(t *testing.T) {
		resp, err := srv.List(ctx, &ListRequest{
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

	t.Run(fmt.Sprintf("VersionsAfterDelete%s", testSuffix), func(t *testing.T) {
		got, err := srv.Versions(ctx, &VersionsRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("Versions('file1') = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})

	//Clean up
	if err := srv.Delete(ctx, &DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file2",
	}); err != nil {
		t.Fatalf("Delete(file2) failed: %v", err)
	}
}

func TestInMemoryArtifactService_Empty(t *testing.T) {
	ctx := t.Context()
	srv := Mem()
	_testArtifactService_Empty(ctx, t, srv, "_InMemory")
}

func TestGCSArtifactService_Empty(t *testing.T) {
	ctx := t.Context()
	srv, err := NewGcsArtifactServiceWithClient(ctx, NewFakeClient(), "new")
	if err != nil {
		t.Fatalf("TestGcsArtifactService() failed: %v", err)
		return
	}
	_testArtifactService_Empty(ctx, t, srv, "_GCS")
}

func _testArtifactService_Empty(ctx context.Context, t *testing.T, srv Service, testSuffix string) {
	t.Run(fmt.Sprintf("Load%s", testSuffix), func(t *testing.T) {
		got, err := srv.Load(ctx, &LoadRequest{
			AppName: "app", UserID: "user", SessionID: "session", FileName: "file"})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("List() = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})
	t.Run(fmt.Sprintf("List%s", testSuffix), func(t *testing.T) {
		_, err := srv.List(ctx, &ListRequest{
			AppName: "app", UserID: "user", SessionID: "session"})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
	})
	t.Run(fmt.Sprintf("Delete%s", testSuffix), func(t *testing.T) {
		err := srv.Delete(ctx, &DeleteRequest{
			AppName: "app", UserID: "user", SessionID: "sesion", FileName: "file1"})
		if err != nil {
			t.Fatalf("Delete() failed: %v", err)
		}
	})
	t.Run(fmt.Sprintf("Versions%s", testSuffix), func(t *testing.T) {
		got, err := srv.Versions(ctx, &VersionsRequest{
			AppName: "app", UserID: "user", SessionID: "session", FileName: "file1"})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("Versions() = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})
}
