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

package fileartifact

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/platform"
)

func newTestService(t *testing.T, root string) *service {
	t.Helper()
	svc, err := NewService(root)
	if err != nil {
		t.Fatalf("NewService() error = %v", err)
	}
	return svc.(*service)
}

func saveText(t *testing.T, svc artifact.Service, app, user, session, file, text string) *artifact.SaveResponse {
	t.Helper()
	resp, err := svc.Save(t.Context(), &artifact.SaveRequest{
		AppName: app, UserID: user, SessionID: session, FileName: file,
		Part: genai.NewPartFromText(text),
	})
	if err != nil {
		t.Fatalf("Save(%q) error = %v", file, err)
	}
	return resp
}

func TestSaveLoadVersionsAndMetadata(t *testing.T) {
	root := t.TempDir()
	svc := newTestService(t, root)
	createTime := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return createTime })

	v1, err := svc.Save(ctx, &artifact.SaveRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt",
		Part:           genai.NewPartFromText("hello"),
		CustomMetadata: map[string]any{"count": 3, "tag": "x"},
	})
	if err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	if v1.Version != 1 {
		t.Fatalf("Save() version = %d, want 1", v1.Version)
	}

	blob := genai.NewPartFromBytes([]byte{0, 1, 2}, "image/png")
	blob.InlineData.DisplayName = "picture.png"
	v2, err := svc.Save(ctx, &artifact.SaveRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt",
		Part: blob,
	})
	if err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	if v2.Version != 2 {
		t.Fatalf("Save() version = %d, want 2", v2.Version)
	}

	// A second service over the same root must see the saved versions.
	reopened := newTestService(t, root)
	load, err := reopened.Load(t.Context(), &artifact.LoadRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt",
	})
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if diff := cmp.Diff([]byte{0, 1, 2}, load.Part.InlineData.Data); diff != "" {
		t.Errorf("Load() data mismatch (-want +got):\n%s", diff)
	}
	if load.Part.InlineData.MIMEType != "image/png" {
		t.Errorf("Load() MIMEType = %q, want image/png", load.Part.InlineData.MIMEType)
	}
	if load.Part.InlineData.DisplayName != "picture.png" {
		t.Errorf("Load() DisplayName = %q, want picture.png", load.Part.InlineData.DisplayName)
	}

	loadV1, err := reopened.Load(t.Context(), &artifact.LoadRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt", Version: 1,
	})
	if err != nil {
		t.Fatalf("Load(version 1) error = %v", err)
	}
	if loadV1.Part.Text != "hello" {
		t.Errorf("Load(version 1) text = %q, want hello", loadV1.Part.Text)
	}

	versions, err := reopened.Versions(t.Context(), &artifact.VersionsRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt",
	})
	if err != nil {
		t.Fatalf("Versions() error = %v", err)
	}
	if diff := cmp.Diff([]int64{1, 2}, versions.Versions); diff != "" {
		t.Errorf("Versions() mismatch (-want +got):\n%s", diff)
	}

	meta, err := reopened.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt", Version: 1,
	})
	if err != nil {
		t.Fatalf("GetArtifactVersion() error = %v", err)
	}
	got := meta.ArtifactVersion
	if got.MimeType != "text/plain" {
		t.Errorf("MimeType = %q, want text/plain", got.MimeType)
	}
	if diff := cmp.Diff(map[string]any{"count": "3", "tag": "x"}, got.CustomMetadata); diff != "" {
		t.Errorf("CustomMetadata mismatch (-want +got):\n%s", diff)
	}
	if !got.CreateTime.Equal(createTime) {
		t.Errorf("CreateTime = %v, want %v", got.CreateTime, createTime)
	}
	if got.CanonicalURI == "" {
		t.Error("CanonicalURI is empty")
	}
}

func TestUserScopedArtifactsAreSharedAcrossSessions(t *testing.T) {
	svc := newTestService(t, t.TempDir())
	saveText(t, svc, "app", "user", "s1", "session.txt", "one")
	saveText(t, svc, "app", "user", "s1", "user:shared.txt", "shared")
	saveText(t, svc, "app", "user", "s2", "other.txt", "two")

	list, err := svc.List(t.Context(), &artifact.ListRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		t.Fatalf("List(s1) error = %v", err)
	}
	if diff := cmp.Diff([]string{"session.txt", "user:shared.txt"}, list.FileNames); diff != "" {
		t.Errorf("List(s1) mismatch (-want +got):\n%s", diff)
	}

	list, err = svc.List(t.Context(), &artifact.ListRequest{AppName: "app", UserID: "user", SessionID: "s2"})
	if err != nil {
		t.Fatalf("List(s2) error = %v", err)
	}
	if diff := cmp.Diff([]string{"other.txt", "user:shared.txt"}, list.FileNames); diff != "" {
		t.Errorf("List(s2) mismatch (-want +got):\n%s", diff)
	}

	load, err := svc.Load(t.Context(), &artifact.LoadRequest{
		AppName: "app", UserID: "user", SessionID: "s2", FileName: "user:shared.txt",
	})
	if err != nil {
		t.Fatalf("Load(user-scoped) error = %v", err)
	}
	if load.Part.Text != "shared" {
		t.Errorf("Load(user-scoped) text = %q, want shared", load.Part.Text)
	}
}

func TestDeleteVersionAndAllVersions(t *testing.T) {
	svc := newTestService(t, t.TempDir())
	saveText(t, svc, "app", "user", "session", "notes.txt", "v1")
	saveText(t, svc, "app", "user", "session", "notes.txt", "v2")
	saveText(t, svc, "app", "user", "session", "notes.txt", "v3")

	if err := svc.Delete(t.Context(), &artifact.DeleteRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt", Version: 2,
	}); err != nil {
		t.Fatalf("Delete(version 2) error = %v", err)
	}
	versions, err := svc.Versions(t.Context(), &artifact.VersionsRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt",
	})
	if err != nil {
		t.Fatalf("Versions() error = %v", err)
	}
	if diff := cmp.Diff([]int64{1, 3}, versions.Versions); diff != "" {
		t.Errorf("Versions() mismatch (-want +got):\n%s", diff)
	}
	if _, err := svc.Load(t.Context(), &artifact.LoadRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt", Version: 2,
	}); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("Load(deleted version) error = %v, want fs.ErrNotExist", err)
	}

	if err := svc.Delete(t.Context(), &artifact.DeleteRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt",
	}); err != nil {
		t.Fatalf("Delete(all versions) error = %v", err)
	}
	if _, err := svc.Versions(t.Context(), &artifact.VersionsRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt",
	}); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("Versions() error = %v, want fs.ErrNotExist", err)
	}
	if _, err := svc.Load(t.Context(), &artifact.LoadRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt",
	}); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("Load() error = %v, want fs.ErrNotExist", err)
	}
	list, err := svc.List(t.Context(), &artifact.ListRequest{AppName: "app", UserID: "user", SessionID: "session"})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(list.FileNames) != 0 {
		t.Errorf("List() = %v, want no files", list.FileNames)
	}
	if err := svc.Delete(t.Context(), &artifact.DeleteRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "notes.txt",
	}); err != nil {
		t.Errorf("Delete(missing) error = %v, want nil", err)
	}
}

func TestRequestValidation(t *testing.T) {
	svc := newTestService(t, t.TempDir())
	validPart := genai.NewPartFromText("x")

	tests := []struct {
		name string
		call func() error
	}{
		{
			name: "save missing fields",
			call: func() error {
				_, err := svc.Save(t.Context(), &artifact.SaveRequest{Part: validPart})
				return err
			},
		},
		{
			name: "save missing part",
			call: func() error {
				_, err := svc.Save(t.Context(), &artifact.SaveRequest{
					AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
				})
				return err
			},
		},
		{
			name: "save path separator",
			call: func() error {
				_, err := svc.Save(t.Context(), &artifact.SaveRequest{
					AppName: "app", UserID: "user", SessionID: "session", FileName: "a/b", Part: validPart,
				})
				return err
			},
		},
		{
			name: "load missing app",
			call: func() error {
				_, err := svc.Load(t.Context(), &artifact.LoadRequest{UserID: "user", SessionID: "session", FileName: "file"})
				return err
			},
		},
		{
			name: "list missing session",
			call: func() error {
				_, err := svc.List(t.Context(), &artifact.ListRequest{AppName: "app", UserID: "user"})
				return err
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.call(); err == nil {
				t.Fatal("error = nil, want a validation error")
			}
		})
	}
}

func TestConcurrentSavesGetUniqueVersions(t *testing.T) {
	svc := newTestService(t, t.TempDir())
	const n = 16
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := svc.Save(t.Context(), &artifact.SaveRequest{
				AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
				Part: genai.NewPartFromText(string(rune('a' + i))),
			})
			errs <- err
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("Save() error = %v", err)
		}
	}

	versions, err := svc.Versions(t.Context(), &artifact.VersionsRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
	})
	if err != nil {
		t.Fatalf("Versions() error = %v", err)
	}
	if len(versions.Versions) != n {
		t.Fatalf("Versions() = %v, want %d versions", versions.Versions, n)
	}
	for i, version := range versions.Versions {
		if version != int64(i+1) {
			t.Fatalf("Versions() = %v, want 1..%d", versions.Versions, n)
		}
	}
}

func TestNamesThatAreUnsafeOnDisk(t *testing.T) {
	svc := newTestService(t, t.TempDir())
	names := []string{".", "..", "CON", "with space.txt", "\u65e5\u672c\u8a9e.txt", "user:..", "user:a:b"}
	for _, name := range names {
		saveText(t, svc, "app", "user", "session", name, "content of "+name)
	}
	for _, name := range names {
		load, err := svc.Load(t.Context(), &artifact.LoadRequest{
			AppName: "app", UserID: "user", SessionID: "session", FileName: name,
		})
		if err != nil {
			t.Fatalf("Load(%q) error = %v", name, err)
		}
		if want := "content of " + name; load.Part.Text != want {
			t.Errorf("Load(%q) text = %q, want %q", name, load.Part.Text, want)
		}
	}
	list, err := svc.List(t.Context(), &artifact.ListRequest{AppName: "app", UserID: "user", SessionID: "session"})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	for _, name := range names {
		found := false
		for _, got := range list.FileNames {
			if got == name {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("List() = %v, missing %q", list.FileNames, name)
		}
	}
}

func TestUnpublishedVersionIsIgnored(t *testing.T) {
	root := t.TempDir()
	svc := newTestService(t, root)
	saveText(t, svc, "app", "user", "session", "file", "published")

	artifactDir := svc.artifactDir("app", "user", "session", "file")
	if err := os.MkdirAll(filepath.Join(artifactDir, versionsDirName, "2"), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}

	versions, err := svc.Versions(t.Context(), &artifact.VersionsRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
	})
	if err != nil {
		t.Fatalf("Versions() error = %v", err)
	}
	if diff := cmp.Diff([]int64{1}, versions.Versions); diff != "" {
		t.Errorf("Versions() mismatch (-want +got):\n%s", diff)
	}
	load, err := svc.Load(t.Context(), &artifact.LoadRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
	})
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if load.Part.Text != "published" {
		t.Errorf("Load() text = %q, want published", load.Part.Text)
	}

	// The unpublished directory keeps version 2 reserved, so the next save
	// claims 3 even though 2 was never readable.
	resp := saveText(t, svc, "app", "user", "session", "file", "next")
	if resp.Version != 3 {
		t.Errorf("Save() version = %d, want 3", resp.Version)
	}
	if _, err := svc.Load(t.Context(), &artifact.LoadRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "file", Version: 2,
	}); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("Load(unpublished version) error = %v, want fs.ErrNotExist", err)
	}
}
