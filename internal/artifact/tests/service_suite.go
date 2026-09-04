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

package tests

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
)

func TestArtifactService(t *testing.T, name string, factory func(t *testing.T) (artifact.Service, error)) {
	t.Run(fmt.Sprintf("Test%sArtifactService", name), func(t *testing.T) {
		ctx := t.Context()
		// Create the service using the factory for this sub-test
		srv, err := factory(t)
		if err != nil {
			t.Fatalf("Failed to set up service: %v", err)
		}
		testArtifactService(ctx, t, srv, name)
	})
	t.Run(fmt.Sprintf("Test%sArtifactService_Empty", name), func(t *testing.T) {
		ctx := t.Context()
		// Create the service using the factory for this sub-test
		srv, err := factory(t)
		if err != nil {
			t.Fatalf("Failed to set up service: %v", err)
		}
		testArtifactService_Empty(ctx, t, srv, name)
	})
	t.Run(fmt.Sprintf("Test%sArtifactService_UserScoped", name), func(t *testing.T) {
		ctx := t.Context()
		// Create the service using the factory for this sub-test
		srv, err := factory(t)
		if err != nil {
			t.Fatalf("Failed to set up service: %v", err)
		}
		testArtifactService_UserScoped(ctx, t, srv, name)
	})
	t.Run(fmt.Sprintf("Test%sArtifactService_GetArtifactVersion", name), func(t *testing.T) {
		ctx := t.Context()
		// Create the service using the factory for this sub-test.
		srv, err := factory(t)
		if err != nil {
			t.Fatalf("Failed to set up service: %v", err)
		}
		testArtifactService_GetArtifactVersion(ctx, t, srv, name)
	})
}

func testArtifactService(ctx context.Context, t *testing.T, srv artifact.Service, testSuffix string) {
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
		// file3.
		{"file3", 1, genai.NewPartFromText("file v1")},
	}

	t.Log("Save file1 and file2")
	for i, data := range testData {
		wantVersion := data.version
		got, err := srv.Save(ctx, &artifact.SaveRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: data.fileName,
			Part: data.artifact,
		})
		if err != nil || got.Version != wantVersion {
			t.Errorf("[%d] Save() = (%v, %v), want (%v, nil)", i, got.Version, err, wantVersion)
			continue
		}
	}

	t.Run(fmt.Sprintf("Load_%s", testSuffix), func(t *testing.T) {
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
			got, err := srv.Load(ctx, &artifact.LoadRequest{
				AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName,
				Version: tc.version,
			})
			if err != nil || !cmp.Equal(got.Part, tc.want) {
				t.Errorf("Load(%v) = (%v, %v), want (%v, nil)", tc.version, got.Part, err, tc.want)
			}
		}
	})

	t.Run(fmt.Sprintf("List_%s", testSuffix), func(t *testing.T) {
		resp, err := srv.List(ctx, &artifact.ListRequest{
			AppName: appName, UserID: userID, SessionID: sessionID,
		})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
		got := resp.FileNames
		slices.Sort(got)
		want := []string{"file1", "file2", "file3"} // testData has two files.
		if diff := cmp.Diff(got, want); diff != "" {
			t.Errorf("List() = %v, want %v", got, want)
		}
	})

	t.Run(fmt.Sprintf("Versions_%s", testSuffix), func(t *testing.T) {
		resp, err := srv.Versions(ctx, &artifact.VersionsRequest{
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
	if err := srv.Delete(ctx, &artifact.DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
		Version: 3,
	}); err != nil {
		t.Fatalf("Delete(file1@v3) failed: %v", err)
	}

	t.Run(fmt.Sprintf("LoadAfterDeleteVersion3_%s", testSuffix), func(t *testing.T) {
		resp, err := srv.Load(ctx, &artifact.LoadRequest{
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

	if err := srv.Delete(ctx, &artifact.DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
	}); err != nil {
		t.Fatalf("Delete(file1) failed: %v", err)
	}

	t.Run(fmt.Sprintf("LoadAfterDelete_%s", testSuffix), func(t *testing.T) {
		got, err := srv.Load(ctx, &artifact.LoadRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("Load('file1') = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})

	t.Run(fmt.Sprintf("ListAfterDelete_%s", testSuffix), func(t *testing.T) {
		resp, err := srv.List(ctx, &artifact.ListRequest{
			AppName: appName, UserID: userID, SessionID: sessionID,
		})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
		got := resp.FileNames
		slices.Sort(got)
		want := []string{"file2", "file3"} // testData has two files.
		if diff := cmp.Diff(got, want); diff != "" {
			t.Errorf("List() = %v, want %v", got, want)
		}
	})

	t.Run(fmt.Sprintf("VersionsAfterDelete_%s", testSuffix), func(t *testing.T) {
		got, err := srv.Versions(ctx, &artifact.VersionsRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file1",
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("Versions('file1') = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})

	// Clean up
	if err := srv.Delete(ctx, &artifact.DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file2",
	}); err != nil {
		t.Fatalf("Delete(file2) failed: %v", err)
	}
	if err := srv.Delete(ctx, &artifact.DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file3",
	}); err != nil {
		t.Fatalf("Delete(file3) failed: %v", err)
	}
}

func testArtifactService_UserScoped(ctx context.Context, t *testing.T, srv artifact.Service, testSuffix string) {
	appName := "testapp"
	userID := "testuser"
	sessionID := "testsession"

	// Save these artifacts for later subtests.
	testData := []struct {
		fileName  string
		sessionID string
		version   int64
		artifact  *genai.Part
		metadata  map[string]any
	}{
		// file1.
		{"user:file1", "save-session-1", 1, genai.NewPartFromBytes([]byte("file v1"), "text/plain"), map[string]any{"version": "v1"}},
		{"user:file1", "save-session-2", 2, genai.NewPartFromBytes([]byte("file v2"), "text/plain"), map[string]any{"version": "v2"}},
		{"user:file1", "save-session-3", 3, genai.NewPartFromBytes([]byte("file v3"), "text/plain"), map[string]any{"version": "v3"}},
		// file2.
		{"file2", sessionID, 1, genai.NewPartFromBytes([]byte("file v3"), "text/plain"), nil},
		// file3.
		{"user:file3", "another-save-session", 1, genai.NewPartFromText("file v1"), nil},
	}

	t.Log("Save file1 and file2")
	for i, data := range testData {
		wantVersion := data.version
		got, err := srv.Save(ctx, &artifact.SaveRequest{
			AppName: appName, UserID: userID, SessionID: data.sessionID, FileName: data.fileName,
			Part: data.artifact, CustomMetadata: data.metadata,
		})
		if err != nil || got.Version != wantVersion {
			t.Errorf("[%d] Save() = (%v, %v), want (%v, nil)", i, got.Version, err, wantVersion)
			continue
		}
	}

	t.Run(fmt.Sprintf("Load_%s", testSuffix), func(t *testing.T) {
		fileName := "user:file1"
		for _, tc := range []struct {
			name    string
			version int64
			want    *genai.Part
		}{
			{"latest", 0, genai.NewPartFromBytes([]byte("file v3"), "text/plain")},
			{"ver=1", 1, genai.NewPartFromBytes([]byte("file v1"), "text/plain")},
			{"ver=2", 2, genai.NewPartFromBytes([]byte("file v2"), "text/plain")},
		} {
			got, err := srv.Load(ctx, &artifact.LoadRequest{
				AppName: appName, UserID: userID, SessionID: "'user' should be used instead", FileName: fileName,
				Version: tc.version,
			})
			if err != nil || !cmp.Equal(got.Part, tc.want) {
				t.Errorf("Load(%v) = (%v, %v), want (%v, nil)", tc.version, got.Part, err, tc.want)
			}
		}
	})

	t.Run(fmt.Sprintf("List_%s", testSuffix), func(t *testing.T) {
		resp, err := srv.List(ctx, &artifact.ListRequest{
			AppName: appName, UserID: userID, SessionID: sessionID,
		})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
		got := resp.FileNames
		want := []string{"file2", "user:file1", "user:file3"} // testData has two files.
		if diff := cmp.Diff(got, want); diff != "" {
			t.Errorf("List() = %v, want %v", got, want)
		}
	})

	t.Run(fmt.Sprintf("Versions_%s", testSuffix), func(t *testing.T) {
		resp, err := srv.Versions(ctx, &artifact.VersionsRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "user:file1",
		})
		if err != nil {
			t.Fatalf("Versions() failed: %v", err)
		}
		got := resp.Versions
		slices.Sort(got)
		want := []int64{1, 2, 3}
		if diff := cmp.Diff(got, want); diff != "" {
			t.Errorf("Versions('user:file1') = %v, want %v", got, want)
		}
	})

	t.Run(fmt.Sprintf("GetArtifactVersion_%s", testSuffix), func(t *testing.T) {
		for _, tc := range []struct {
			name         string
			version      int64
			wantVersion  int64
			wantMetadata map[string]any
		}{
			{name: "latest", wantVersion: 3, wantMetadata: map[string]any{"version": "v3"}},
			{name: "specific", version: 2, wantVersion: 2, wantMetadata: map[string]any{"version": "v2"}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				resp, err := srv.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{
					AppName: appName, UserID: userID, SessionID: "'user' should be used instead", FileName: "user:file1",
					Version: tc.version,
				})
				if err != nil {
					t.Fatalf("GetArtifactVersion(%d) failed: %v", tc.version, err)
				}
				assertArtifactVersion(t, resp.ArtifactVersion, tc.wantVersion, "text/plain", tc.wantMetadata)
			})
		}
	})

	t.Log("Delete user:file1 version 3")
	if err := srv.Delete(ctx, &artifact.DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "user:file1",
		Version: 3,
	}); err != nil {
		t.Fatalf("Delete(user:file1@v3) failed: %v", err)
	}
	t.Run(fmt.Sprintf("GetArtifactVersionAfterDeleteLatest_%s", testSuffix), func(t *testing.T) {
		resp, err := srv.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{
			AppName: appName, UserID: userID, SessionID: "another-read-session", FileName: "user:file1",
		})
		if err != nil {
			t.Fatalf("GetArtifactVersion() failed: %v", err)
		}
		assertArtifactVersion(t, resp.ArtifactVersion, 2, "text/plain", map[string]any{"version": "v2"})
	})

	t.Run(fmt.Sprintf("LoadAfterDeleteVersion3_%s", testSuffix), func(t *testing.T) {
		resp, err := srv.Load(ctx, &artifact.LoadRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "user:file1",
		})
		if err != nil {
			t.Fatalf("Load('user:file1') failed: %v", err)
		}
		got := resp.Part
		want := genai.NewPartFromBytes([]byte("file v2"), "text/plain")
		if diff := cmp.Diff(got, want); diff != "" {
			t.Fatalf("Load('user:file1') = (%v, %v), want (%v, nil)", got, err, want)
		}
	})

	if err := srv.Delete(ctx, &artifact.DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "user:file1",
	}); err != nil {
		t.Fatalf("Delete(user:file1) failed: %v", err)
	}

	t.Run(fmt.Sprintf("LoadAfterDelete_%s", testSuffix), func(t *testing.T) {
		got, err := srv.Load(ctx, &artifact.LoadRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "user:file1",
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("Load('user:file1') = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})

	t.Run(fmt.Sprintf("ListAfterDelete_%s", testSuffix), func(t *testing.T) {
		resp, err := srv.List(ctx, &artifact.ListRequest{
			AppName: appName, UserID: userID, SessionID: sessionID,
		})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
		got := resp.FileNames
		slices.Sort(got)
		want := []string{"file2", "user:file3"} // testData has two files.
		if diff := cmp.Diff(got, want); diff != "" {
			t.Errorf("List() = %v, want %v", got, want)
		}
	})

	t.Run(fmt.Sprintf("VersionsAfterDelete_%s", testSuffix), func(t *testing.T) {
		got, err := srv.Versions(ctx, &artifact.VersionsRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "user:file1",
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("Versions('user:file1') = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})

	// Clean up
	if err := srv.Delete(ctx, &artifact.DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "file2",
	}); err != nil {
		t.Fatalf("Delete(file2) failed: %v", err)
	}
	if err := srv.Delete(ctx, &artifact.DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "user:file3",
	}); err != nil {
		t.Fatalf("Delete(user:file3) failed: %v", err)
	}
}

func testArtifactService_Empty(ctx context.Context, t *testing.T, srv artifact.Service, testSuffix string) {
	t.Run(fmt.Sprintf("Load_%s", testSuffix), func(t *testing.T) {
		got, err := srv.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("List() = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})
	t.Run(fmt.Sprintf("List_%s", testSuffix), func(t *testing.T) {
		_, err := srv.List(ctx, &artifact.ListRequest{
			AppName: "app", UserID: "user", SessionID: "session",
		})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
	})
	t.Run(fmt.Sprintf("Delete_%s", testSuffix), func(t *testing.T) {
		err := srv.Delete(ctx, &artifact.DeleteRequest{
			AppName: "app", UserID: "user", SessionID: "session", FileName: "file1",
		})
		if err != nil {
			t.Fatalf("Delete() failed: %v", err)
		}
	})
	t.Run(fmt.Sprintf("Versions_%s", testSuffix), func(t *testing.T) {
		got, err := srv.Versions(ctx, &artifact.VersionsRequest{
			AppName: "app", UserID: "user", SessionID: "session", FileName: "file1",
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("Versions() = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})
	t.Run(fmt.Sprintf("GetArtifactVersion_%s", testSuffix), func(t *testing.T) {
		got, err := srv.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{
			AppName: "app", UserID: "user", SessionID: "session", FileName: "file1",
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("GetArtifactVersion() = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})
}

func testArtifactService_GetArtifactVersion(ctx context.Context, t *testing.T, srv artifact.Service, testSuffix string) {
	appName := "testapp"
	userID := "testuser"
	sessionID := "testsession"
	fileName := "verfile"

	for i, data := range []struct {
		part     *genai.Part
		metadata map[string]any
	}{
		{genai.NewPartFromBytes([]byte("v1"), "text/plain"), nil},
		{genai.NewPartFromBytes([]byte("v2"), "image/png"), map[string]any{"version": "v2"}},
		{genai.NewPartFromText("v3"), map[string]any{"version": "v3"}},
	} {
		wantVersion := int64(i + 1)
		got, err := srv.Save(ctx, &artifact.SaveRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName,
			Part: data.part, CustomMetadata: data.metadata,
		})
		if err != nil || got.Version != wantVersion {
			t.Fatalf("[%d] Save() = (%v, %v), want (%v, nil)", i, got.Version, err, wantVersion)
		}
	}

	for _, tc := range []struct {
		name         string
		version      int64
		wantVersion  int64
		wantMimeType string
		wantMetadata map[string]any
	}{
		{name: "latest", wantVersion: 3, wantMimeType: "text/plain", wantMetadata: map[string]any{"version": "v3"}},
		{name: "specific", version: 2, wantVersion: 2, wantMimeType: "image/png", wantMetadata: map[string]any{"version": "v2"}},
		{name: "empty metadata", version: 1, wantVersion: 1, wantMimeType: "text/plain", wantMetadata: map[string]any{}},
	} {
		t.Run(fmt.Sprintf("GetArtifactVersion_%s_%s", tc.name, testSuffix), func(t *testing.T) {
			resp, err := srv.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{
				AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Version: tc.version,
			})
			if err != nil {
				t.Fatalf("GetArtifactVersion(%d) failed: %v", tc.version, err)
			}
			assertArtifactVersion(t, resp.ArtifactVersion, tc.wantVersion, tc.wantMimeType, tc.wantMetadata)
		})
	}

	t.Run(fmt.Sprintf("GetArtifactVersion_nonexistent-version_%s", testSuffix), func(t *testing.T) {
		got, err := srv.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Version: 99,
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("GetArtifactVersion(99) = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})
	t.Run(fmt.Sprintf("GetArtifactVersion_nonexistent-file_%s", testSuffix), func(t *testing.T) {
		got, err := srv.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: "does-not-exist",
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("GetArtifactVersion(missing file) = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})

	if err := srv.Delete(ctx, &artifact.DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Version: 3,
	}); err != nil {
		t.Fatalf("Delete(%s@v3) failed: %v", fileName, err)
	}
	t.Run(fmt.Sprintf("GetArtifactVersionAfterDeleteLatest_%s", testSuffix), func(t *testing.T) {
		resp, err := srv.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName,
		})
		if err != nil {
			t.Fatalf("GetArtifactVersion() failed: %v", err)
		}
		assertArtifactVersion(t, resp.ArtifactVersion, 2, "image/png", map[string]any{"version": "v2"})
	})

	if err := srv.Delete(ctx, &artifact.DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName,
	}); err != nil {
		t.Fatalf("Delete(%s) failed: %v", fileName, err)
	}
	t.Run(fmt.Sprintf("GetArtifactVersionAfterDelete_%s", testSuffix), func(t *testing.T) {
		got, err := srv.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName,
		})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("GetArtifactVersion() = (%v, %v), want error(%v)", got, err, fs.ErrNotExist)
		}
	})
}

func assertArtifactVersion(t *testing.T, got *artifact.ArtifactVersion, wantVersion int64, wantMimeType string, wantMetadata map[string]any) {
	t.Helper()
	if got == nil {
		t.Fatal("GetArtifactVersion().ArtifactVersion = nil, want non-nil")
	}
	if got.Version != wantVersion {
		t.Errorf("ArtifactVersion.Version = %d, want %d", got.Version, wantVersion)
	}
	if got.MimeType != wantMimeType {
		t.Errorf("ArtifactVersion.MimeType = %q, want %q", got.MimeType, wantMimeType)
	}
	if diff := cmp.Diff(wantMetadata, got.CustomMetadata); diff != "" {
		t.Errorf("ArtifactVersion.CustomMetadata mismatch (-want +got):\n%s", diff)
	}
	if got.CanonicalURI == "" {
		t.Error("ArtifactVersion.CanonicalURI is empty")
	}
	if got.CreateTime.IsZero() {
		t.Error("ArtifactVersion.CreateTime is zero")
	}
}
