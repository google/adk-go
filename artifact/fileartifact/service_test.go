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

package fileartifact_test

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"os/user"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/artifact/fileartifact"
	"google.golang.org/adk/v2/internal/artifact/tests"
	"google.golang.org/adk/v2/platform"
)

func TestFileArtifactService(t *testing.T) {
	factory := func(t *testing.T) (artifact.Service, error) {
		return fileartifact.NewService(t.Context(), t.TempDir())
	}
	tests.TestArtifactService(t, "File", factory)
}

func TestNewService_RootDirResolution(t *testing.T) {
	t.Run("tilde expands to the home directory", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("HOME", home)
		t.Chdir(t.TempDir())
		if _, err := fileartifact.NewService(t.Context(), "~/nested/store"); err != nil {
			t.Fatalf("NewService() failed: %v", err)
		}
		assertEmptyDir(t, filepath.Join(home, "nested", "store"))
	})
	t.Run("relative path resolves against the working directory", func(t *testing.T) {
		wd := t.TempDir()
		t.Chdir(wd)
		if _, err := fileartifact.NewService(t.Context(), "rel/store"); err != nil {
			t.Fatalf("NewService() failed: %v", err)
		}
		assertEmptyDir(t, filepath.Join(wd, "rel", "store"))
	})
	t.Run("empty path is the working directory", func(t *testing.T) {
		wd := t.TempDir()
		t.Chdir(wd)
		srv, err := fileartifact.NewService(t.Context(), "")
		if err != nil {
			t.Fatalf("NewService() failed: %v", err)
		}
		if _, err := srv.Save(t.Context(), &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.txt", Part: textPart("x")}); err != nil {
			t.Fatalf("Save() failed: %v", err)
		}
		if _, err := os.Stat(filepath.Join(wd, "apps", "app", "users", "u", "sessions", "s", "artifacts", "f.txt", "versions", "1", "f.txt")); err != nil {
			t.Errorf("payload not under the working directory: %v", err)
		}
	})
	t.Run("existing regular file is refused", func(t *testing.T) {
		file := filepath.Join(t.TempDir(), "file")
		if err := os.WriteFile(file, nil, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := fileartifact.NewService(t.Context(), file); err == nil {
			t.Error("NewService(regular file) succeeded, want an error")
		}
	})
	t.Run("tilde with an unknown user name is refused", func(t *testing.T) {
		wd := t.TempDir()
		t.Chdir(wd)
		_, err := fileartifact.NewService(t.Context(), "~adk_no_such_user_zz/store")
		if err == nil {
			t.Fatal("NewService(~unknown user) succeeded, want an error")
		}
		if !strings.Contains(err.Error(), "could not determine home directory") {
			t.Errorf("NewService(~unknown user) error = %q, want it to contain %q", err, "could not determine home directory")
		}
		if strings.Contains(err.Error(), "adk_no_such_user_zz") {
			t.Errorf("NewService(~unknown user) error = %q, want no user name in it", err)
		}
		assertEmptyDir(t, wd)
	})
	// The next two resolve to the real home directory, which exists, so
	// NewService creates nothing there.
	t.Run("tilde with the current user name expands to that user's home", func(t *testing.T) {
		u := currentUserWithHome(t)
		if strings.ContainsAny(u.Username, "/"+string(filepath.Separator)) {
			t.Skipf("user name %q holds a path separator", u.Username)
		}
		wd := t.TempDir()
		t.Chdir(wd)
		if _, err := fileartifact.NewService(t.Context(), "~"+u.Username); err != nil {
			t.Fatalf("NewService(~current user) failed: %v", err)
		}
		assertEmptyDir(t, wd)
	})
	t.Run("tilde without HOME expands to the current user's home", func(t *testing.T) {
		currentUserWithHome(t)
		t.Setenv("HOME", "")
		if err := os.Unsetenv("HOME"); err != nil {
			t.Fatal(err)
		}
		wd := t.TempDir()
		t.Chdir(wd)
		if _, err := fileartifact.NewService(t.Context(), "~"); err != nil {
			t.Fatalf("NewService(~) without HOME failed: %v", err)
		}
		assertEmptyDir(t, wd)
	})
}

// currentUserWithHome returns the current user, and skips the test when the
// password database has no entry for it or its home directory does not exist.
func currentUserWithHome(t *testing.T) *user.User {
	t.Helper()
	u, err := user.Current()
	if err != nil {
		t.Skipf("user.Current() failed: %v", err)
	}
	if info, err := os.Stat(u.HomeDir); err != nil || !info.IsDir() {
		t.Skipf("home directory %q of the current user is not a directory", u.HomeDir)
	}
	return u
}

// assertEmptyDir fails the test unless dir exists and holds nothing.
func assertEmptyDir(t *testing.T, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir(%s) failed: %v", dir, err)
	}
	if len(entries) != 0 {
		t.Errorf("ReadDir(%s) has %d entries, want none", dir, len(entries))
	}
}

func textPart(s string) *genai.Part { return genai.NewPartFromText(s) }

// fixedTime is the creation time the tests install with
// platform.WithTimeProvider; it is 1700000000.5 seconds after the epoch.
var fixedTime = time.Unix(1700000000, 500000000)

func fixedClock(t *testing.T) context.Context {
	return platform.WithTimeProvider(t.Context(), func() time.Time { return fixedTime })
}

// resolvedTempDir returns a new temporary directory with its symbolic links
// resolved, as the service resolves its root.
func resolvedTempDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestSave_OnDiskLayoutAndMetadataJSON(t *testing.T) {
	ctx := fixedClock(t)
	tmp := resolvedTempDir(t)
	root := filepath.Join(tmp, "root dir#1+2")
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}

	for _, tc := range []struct {
		name        string
		req         *artifact.SaveRequest
		versionDir  string
		payloadName string
		wantPayload string
		wantDoc     string
		wantPart    *genai.Part
		wantURI     string
		wantMIME    string
	}{
		{
			name: "session-scoped inline data",
			req: &artifact.SaveRequest{
				AppName: "myapp", UserID: "user123", SessionID: "sess789", FileName: " a b#c+d.bin ",
				Part: &genai.Part{InlineData: &genai.Blob{Data: []byte("raw-bytes"), DisplayName: "My Report"}},
			},
			versionDir:  "apps/myapp/users/user123/sessions/sess789/artifacts/a b#c+d.bin/versions/1",
			payloadName: "a b#c+d.bin",
			wantPayload: "raw-bytes",
			wantDoc: `{"version":1,"canonicalUri":"file://` + tmp + `/root%20dir%231%2B2/apps/myapp/users/user123/sessions/sess789/artifacts/a%20b%23c%2Bd.bin/versions/1/a%20b%23c%2Bd.bin",` +
				`"customMetadata":{},"createTime":1700000000.5,"mimeType":"application/octet-stream","fileName":" a b#c+d.bin ","displayName":"My Report"}`,
			wantPart: &genai.Part{InlineData: &genai.Blob{Data: []byte("raw-bytes"), MIMEType: "application/octet-stream", DisplayName: "My Report"}},
			wantURI:  "file://" + tmp + "/root%20dir%231%2B2/apps/myapp/users/user123/sessions/sess789/artifacts/a%20b%23c%2Bd.bin/versions/1/a%20b%23c%2Bd.bin",
			wantMIME: "application/octet-stream",
		},
		{
			name: "user-scoped text without a session",
			req: &artifact.SaveRequest{
				AppName: "myapp", UserID: "user123", FileName: "user:note.txt",
				Part: genai.NewPartFromText("hello"), CustomMetadata: map[string]any{"n": 1},
			},
			versionDir:  "apps/myapp/users/user123/artifacts/note.txt/versions/1",
			payloadName: "note.txt",
			wantPayload: "hello",
			wantDoc: `{"version":1,"canonicalUri":"file://` + tmp + `/root%20dir%231%2B2/apps/myapp/users/user123/artifacts/note.txt/versions/1/note.txt",` +
				`"customMetadata":{"n":"1"},"createTime":1700000000.5,"fileName":"user:note.txt"}`,
			wantPart: genai.NewPartFromText("hello"),
			wantURI:  "file://" + tmp + "/root%20dir%231%2B2/apps/myapp/users/user123/artifacts/note.txt/versions/1/note.txt",
			wantMIME: "text/plain",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			saved, err := srv.Save(ctx, tc.req)
			if err != nil {
				t.Fatalf("Save() failed: %v", err)
			}
			if saved.Version != 1 {
				t.Errorf("Save().Version = %d, want 1", saved.Version)
			}
			dir := filepath.Join(root, filepath.FromSlash(tc.versionDir))
			payloadPath := filepath.Join(dir, tc.payloadName)
			payload, err := os.ReadFile(payloadPath)
			if err != nil {
				t.Fatalf("payload: %v", err)
			}
			if diff := cmp.Diff(tc.wantPayload, string(payload)); diff != "" {
				t.Errorf("payload mismatch (-want +got):\n%s", diff)
			}
			doc, err := os.ReadFile(filepath.Join(dir, "metadata.json"))
			if err != nil {
				t.Fatalf("metadata.json: %v", err)
			}
			if diff := cmp.Diff(tc.wantDoc, string(doc)); diff != "" {
				t.Errorf("metadata.json mismatch (-want +got):\n%s", diff)
			}
			assertSameMode(t, payloadPath, filepath.Join(dir, "metadata.json"))

			load := &artifact.LoadRequest{AppName: tc.req.AppName, UserID: tc.req.UserID, SessionID: tc.req.SessionID, FileName: tc.req.FileName}
			loaded, err := srv.Load(ctx, load)
			if err != nil {
				t.Fatalf("Load() failed: %v", err)
			}
			if diff := cmp.Diff(tc.wantPart, loaded.Part); diff != "" {
				t.Errorf("Load() mismatch (-want +got):\n%s", diff)
			}
			got, err := srv.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{AppName: tc.req.AppName, UserID: tc.req.UserID, SessionID: tc.req.SessionID, FileName: tc.req.FileName})
			if err != nil {
				t.Fatalf("GetArtifactVersion() failed: %v", err)
			}
			if got.ArtifactVersion.CanonicalURI != tc.wantURI {
				t.Errorf("CanonicalURI = %q, want %q", got.ArtifactVersion.CanonicalURI, tc.wantURI)
			}
			if got.ArtifactVersion.MimeType != tc.wantMIME {
				t.Errorf("MimeType = %q, want %q", got.ArtifactVersion.MimeType, tc.wantMIME)
			}
			if !got.ArtifactVersion.CreateTime.Equal(fixedTime) {
				t.Errorf("CreateTime = %v, want %v", got.ArtifactVersion.CreateTime, fixedTime)
			}
		})
	}
}

// assertSameMode fails the test unless the files a and b have the same
// permission bits.
func assertSameMode(t *testing.T, a, b string) {
	t.Helper()
	sa, err := os.Stat(a)
	if err != nil {
		t.Fatal(err)
	}
	sb, err := os.Stat(b)
	if err != nil {
		t.Fatal(err)
	}
	if sa.Mode().Perm() != sb.Mode().Perm() {
		t.Errorf("mode of %s = %v, want %v as its payload", filepath.Base(b), sb.Mode().Perm(), sa.Mode().Perm())
	}
}

func TestSaveAndLoad_ContentValidationAndRewindTombstone(t *testing.T) {
	ctx := t.Context()
	root := resolvedTempDir(t)
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}

	t.Run("refused parts leave nothing on disk", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			part *genai.Part
			want string
		}{
			{name: "nil part", part: nil, want: "artifact part must not be nil"},
			{name: "inline data without data", part: &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png"}}, want: "artifact inline_data must contain data"},
			{name: "artifact reference", part: genai.NewPartFromURI("artifact://apps/app/users/u/sessions/s/artifacts/f/versions/1", "text/plain"), want: "artifact must have either inline_data or text content"},
			{name: "external file data", part: genai.NewPartFromURI("gs://bucket/f", "text/plain"), want: "artifact must have either inline_data or text content"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				_, err := srv.Save(ctx, &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "bad.png", Part: tc.part})
				if !errors.Is(err, artifact.ErrInputValidation) {
					t.Fatalf("Save() error = %v, want artifact.ErrInputValidation", err)
				}
				if !strings.Contains(err.Error(), tc.want) {
					t.Errorf("Save() error = %q, want it to contain %q", err, tc.want)
				}
			})
		}
		if _, err := os.Stat(filepath.Join(root, "apps")); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Stat(apps) error = %v, want fs.ErrNotExist: a refused save created directories", err)
		}
		list, err := srv.List(ctx, &artifact.ListRequest{AppName: "app", UserID: "u", SessionID: "s"})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
		if len(list.FileNames) != 0 {
			t.Errorf("List() = %q, want no names", list.FileNames)
		}
	})

	t.Run("empty inline data is stored", func(t *testing.T) {
		req := &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "empty.png", Part: &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{}}}}
		if _, err := srv.Save(ctx, req); err != nil {
			t.Fatalf("Save() failed: %v", err)
		}
		loaded, err := srv.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "empty.png"})
		if err != nil {
			t.Fatalf("Load() failed: %v", err)
		}
		if blob := loaded.Part.InlineData; blob == nil || len(blob.Data) != 0 || blob.MIMEType != "image/png" {
			t.Errorf("Load().Part = %+v, want empty image/png inline data", loaded.Part)
		}
	})

	t.Run("rewind tombstone hides its version", func(t *testing.T) {
		name := func(v int64) *artifact.LoadRequest {
			return &artifact.LoadRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "doc.txt", Version: v}
		}
		for i, part := range []*genai.Part{genai.NewPartFromText(""), artifact.NewRewindTombstone()} {
			saved, err := srv.Save(ctx, &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "doc.txt", Part: part})
			if err != nil {
				t.Fatalf("Save(%d) failed: %v", i, err)
			}
			if want := int64(i + 1); saved.Version != want {
				t.Errorf("Save(%d).Version = %d, want %d", i, saved.Version, want)
			}
		}
		for _, v := range []int64{0, 2} {
			if _, err := srv.Load(ctx, name(v)); !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("Load(version %d) error = %v, want fs.ErrNotExist", v, err)
			}
		}
		loaded, err := srv.Load(ctx, name(1))
		if err != nil {
			t.Fatalf("Load(version 1) failed: %v", err)
		}
		if diff := cmp.Diff(genai.NewPartFromText(""), loaded.Part); diff != "" {
			t.Errorf("Load(version 1) mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestSave_SkippedPendingAndConcurrentReservations(t *testing.T) {
	ctx := t.Context()
	root := resolvedTempDir(t)
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	versionsDir := filepath.Join(root, "apps", "app", "users", "u", "sessions", "s", "artifacts", "r.txt", "versions")
	// A save killed after reserving version 1, and an entry at 3 that is not
	// a version directory.
	if err := os.MkdirAll(filepath.Join(versionsDir, ".1.pending"), 0o777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(versionsDir, "3"), nil, 0o666); err != nil {
		t.Fatal(err)
	}
	save := func(srv artifact.Service) (int64, error) {
		resp, err := srv.Save(ctx, &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "r.txt", Part: genai.NewPartFromText("x")})
		if err != nil {
			return 0, err
		}
		return resp.Version, nil
	}

	var got []int64
	for range 2 {
		v, err := save(srv)
		if err != nil {
			t.Fatalf("Save() failed: %v", err)
		}
		got = append(got, v)
	}
	if diff := cmp.Diff([]int64{2, 4}, got); diff != "" {
		t.Errorf("Save() versions mismatch (-want +got):\n%s", diff)
	}
	if _, err := os.Stat(filepath.Join(versionsDir, ".1.pending")); err != nil {
		t.Errorf("abandoned reservation was touched: %v", err)
	}

	// A second service on the same root shares no lock with the first, so
	// only the reservation on disk keeps their versions apart.
	other, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	var (
		mu sync.Mutex
		wg sync.WaitGroup
	)
	for i := range 8 {
		wg.Go(func() {
			s := srv
			if i%2 == 1 {
				s = other
			}
			v, err := save(s)
			if err != nil {
				t.Errorf("concurrent Save() failed: %v", err)
				return
			}
			mu.Lock()
			got = append(got, v)
			mu.Unlock()
		})
	}
	wg.Wait()
	slices.Sort(got)
	if diff := cmp.Diff([]int64{2, 4, 5, 6, 7, 8, 9, 10, 11, 12}, got); diff != "" {
		t.Errorf("Save() versions mismatch (-want +got):\n%s", diff)
	}
	versions, err := srv.Versions(ctx, &artifact.VersionsRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "r.txt"})
	if err != nil {
		t.Fatalf("Versions() failed: %v", err)
	}
	if diff := cmp.Diff([]int64{2, 4, 5, 6, 7, 8, 9, 10, 11, 12}, versions.Versions); diff != "" {
		t.Errorf("Versions() mismatch (-want +got):\n%s", diff)
	}
}

func TestSave_StagingCleanupOnFailure(t *testing.T) {
	ctx := t.Context()
	root := resolvedTempDir(t)
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	req := &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.txt", Part: genai.NewPartFromText("v1")}
	if _, err := srv.Save(ctx, req); err != nil {
		t.Fatalf("Save(v1) failed: %v", err)
	}
	versionsDir := filepath.Join(root, "apps", "app", "users", "u", "sessions", "s", "artifacts", "f.txt", "versions")
	// A dangling link is not a version and does not exist for the reservation,
	// but a directory cannot be renamed onto it, so publishing version 2 fails.
	if err := os.Symlink("missing", filepath.Join(versionsDir, "2")); err != nil {
		t.Fatal(err)
	}

	req.Part = genai.NewPartFromText("v2")
	if _, err := srv.Save(ctx, req); err == nil {
		t.Fatal("Save(v2) succeeded, want an error from publishing")
	}

	pending, err := filepath.Glob(filepath.Join(versionsDir, ".*.pending"))
	if err != nil {
		t.Fatal(err)
	}
	if len(pending) != 0 {
		t.Errorf("staging directories left behind: %q", pending)
	}
	versions, err := srv.Versions(ctx, &artifact.VersionsRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.txt"})
	if err != nil {
		t.Fatalf("Versions() failed: %v", err)
	}
	if diff := cmp.Diff([]int64{1}, versions.Versions); diff != "" {
		t.Errorf("Versions() mismatch (-want +got):\n%s", diff)
	}
	loaded, err := srv.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.txt"})
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}
	if diff := cmp.Diff(genai.NewPartFromText("v1"), loaded.Part); diff != "" {
		t.Errorf("Load() mismatch (-want +got):\n%s", diff)
	}
}

// adk-python reads version directories with int(), which takes "01", "+2" and
// "1_0"; here only canonical positive decimal names are versions.
func TestFileService_StrictVersionParsing(t *testing.T) {
	ctx := t.Context()
	root := resolvedTempDir(t)
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	req := &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.txt", Part: genai.NewPartFromText("x")}
	if _, err := srv.Save(ctx, req); err != nil {
		t.Fatalf("Save() failed: %v", err)
	}
	versionsDir := filepath.Join(root, "apps", "app", "users", "u", "sessions", "s", "artifacts", "f.txt", "versions")
	for _, name := range []string{"01", "+2", "-1", "1_0", "0", " 5", ".3.pending", "99999999999999999999"} {
		if err := os.Mkdir(filepath.Join(versionsDir, name), 0o777); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(versionsDir, "4"), nil, 0o666); err != nil {
		t.Fatal(err)
	}

	versions, err := srv.Versions(ctx, &artifact.VersionsRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.txt"})
	if err != nil {
		t.Fatalf("Versions() failed: %v", err)
	}
	if diff := cmp.Diff([]int64{1}, versions.Versions); diff != "" {
		t.Errorf("Versions() mismatch (-want +got):\n%s", diff)
	}
	saved, err := srv.Save(ctx, req)
	if err != nil {
		t.Fatalf("Save() failed: %v", err)
	}
	if saved.Version != 2 {
		t.Errorf("Save().Version = %d, want 2", saved.Version)
	}
}

func TestReservedNames_SaveRejectsWhileReadAndDeleteSucceed(t *testing.T) {
	ctx := t.Context()
	root := resolvedTempDir(t)
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	const (
		metadataMsg  = "may not be named 'metadata.json'"
		versionsMsg  = "may not contain a 'versions' component"
		separatorMsg = "filename cannot contain path separators"
	)

	t.Run("save refuses reserved names before writing", func(t *testing.T) {
		for _, tc := range []struct {
			fileName string
			want     string
		}{
			{"metadata.json", metadataMsg},
			{"Metadata.JSON", metadataMsg},
			{"metadata.json.", metadataMsg},
			{"metadata.json ", metadataMsg},
			{"metadata.json. .", metadataMsg},
			{"user:metadata.json", metadataMsg},
			{"versions", versionsMsg},
			{"VeRsIoNs", versionsMsg},
			{"versions.", versionsMsg},
			{"user:versions ", versionsMsg},
			// Unicode case folding, as Python's casefold: U+017F LATIN SMALL
			// LETTER LONG S folds to "s".
			{"ver\u017fions", versionsMsg},
			{"metadata.j\u017fon", metadataMsg},
			// The reserved-name rules come before the separator rule, so a
			// nested name gets adk-python's message.
			{"nested/report.txt/versions", versionsMsg},
			{"nested/metadata.json ", metadataMsg},
			{"docs/readme.txt", separatorMsg},
		} {
			t.Run(tc.fileName, func(t *testing.T) {
				_, err := srv.Save(ctx, &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: tc.fileName, Part: genai.NewPartFromText("payload")})
				if !errors.Is(err, artifact.ErrInputValidation) {
					t.Fatalf("Save() error = %v, want artifact.ErrInputValidation", err)
				}
				if !strings.Contains(err.Error(), tc.want) {
					t.Errorf("Save() error = %q, want it to contain %q", err, tc.want)
				}
			})
		}
		if _, err := os.Stat(filepath.Join(root, "apps")); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Stat(apps) error = %v, want fs.ErrNotExist: a refused save wrote to disk", err)
		}
	})

	t.Run("reserved words elsewhere are allowed", func(t *testing.T) {
		other, err := fileartifact.NewService(ctx, filepath.Join(root, "versions"))
		if err != nil {
			t.Fatalf("NewService() failed: %v", err)
		}
		for _, fileName := range []string{"versions.txt", "reversions", "metadata.json.txt"} {
			req := &artifact.SaveRequest{AppName: "versions", UserID: "versions", SessionID: "versions", FileName: fileName, Part: genai.NewPartFromText("ok")}
			if _, err := other.Save(ctx, req); err != nil {
				t.Errorf("Save(%q) failed: %v", fileName, err)
			}
		}
	})

	t.Run("artifacts stored under reserved names stay readable and deletable", func(t *testing.T) {
		scope := filepath.Join(root, "apps", "app", "users", "u", "sessions", "s", "artifacts")
		for _, tc := range []struct {
			fileName, payload string
		}{
			{"metadata.json", `{"fileName":"metadata.json","version":1}`},
			{"versions", "legacy"},
		} {
			versionDir := filepath.Join(scope, tc.fileName, "versions", "1")
			if err := os.MkdirAll(versionDir, 0o777); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(versionDir, tc.fileName), []byte(tc.payload), 0o666); err != nil {
				t.Fatal(err)
			}
			loaded, err := srv.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: tc.fileName})
			if err != nil {
				t.Fatalf("Load(%q) failed: %v", tc.fileName, err)
			}
			if diff := cmp.Diff(genai.NewPartFromText(tc.payload), loaded.Part); diff != "" {
				t.Errorf("Load(%q) mismatch (-want +got):\n%s", tc.fileName, diff)
			}
			if err := srv.Delete(ctx, &artifact.DeleteRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: tc.fileName}); err != nil {
				t.Fatalf("Delete(%q) failed: %v", tc.fileName, err)
			}
			if _, err := os.Stat(filepath.Join(scope, tc.fileName)); !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("Stat(%q) after Delete() error = %v, want fs.ErrNotExist", tc.fileName, err)
			}
		}
	})
}

// callEveryOp calls each artifact.Service method that takes a file name, in
// the order Save, Load, Versions, GetArtifactVersion, Delete, and returns the
// error of each, by method name.
func callEveryOp(ctx context.Context, srv artifact.Service, appName, userID, sessionID, fileName string) map[string]error {
	_, saveErr := srv.Save(ctx, &artifact.SaveRequest{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Part: genai.NewPartFromText("x")})
	_, loadErr := srv.Load(ctx, &artifact.LoadRequest{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName})
	_, versionsErr := srv.Versions(ctx, &artifact.VersionsRequest{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName})
	_, getErr := srv.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName})
	deleteErr := srv.Delete(ctx, &artifact.DeleteRequest{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName})
	return map[string]error{"Save": saveErr, "Load": loadErr, "Delete": deleteErr, "Versions": versionsErr, "GetArtifactVersion": getErr}
}

func TestFileNameValidation_OutOfScopeAndSymlinkEscape(t *testing.T) {
	ctx := t.Context()
	root := resolvedTempDir(t)
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}

	const (
		rootedMsg    = "rooted or drive-qualified artifact filename is not permitted"
		traversalMsg = "artifact filename must not contain parent traversal"
		escapeMsg    = "artifact filename escapes storage directory"
	)
	for _, tc := range []struct {
		fileName string
		want     string
	}{
		{"/zq.txt", rootedMsg},
		{`\zq.txt`, rootedMsg},
		{"user: /zq.txt", rootedMsg},
		{`C:\zq.txt`, rootedMsg},
		{"C:/zq.txt", rootedMsg},
		{"C:zq.txt", rootedMsg},
		{`\\server\share\zq.txt`, rootedMsg},
		{"//server/share/zq.txt", rootedMsg},
		{"../zq.txt", traversalMsg},
		{`..\zq.txt`, traversalMsg},
		{"folder/../zq.txt", traversalMsg},
		{`folder\..\zq.txt`, traversalMsg},
		{"user:../zq.txt", traversalMsg},
		{" .. ", traversalMsg},
		{"dir/zq.txt", "filename cannot contain path separators"},
		{`dir\zq.txt`, "filename cannot contain path separators"},
	} {
		t.Run(tc.fileName, func(t *testing.T) {
			for op, err := range callEveryOp(ctx, srv, "app", "u", "s", tc.fileName) {
				if !errors.Is(err, artifact.ErrInputValidation) {
					t.Errorf("%s() error = %v, want artifact.ErrInputValidation", op, err)
					continue
				}
				if !strings.Contains(err.Error(), tc.want) || strings.Contains(err.Error(), "zq") {
					t.Errorf("%s() error = %q, want it to contain %q and not the file name", op, err, tc.want)
				}
			}
		})
	}

	t.Run("symbolic links that leave the scope", func(t *testing.T) {
		victim := &artifact.SaveRequest{AppName: "app", UserID: "u2", FileName: "user:secret.txt", Part: genai.NewPartFromText("u2 data")}
		if _, err := srv.Save(ctx, victim); err != nil {
			t.Fatalf("Save(victim) failed: %v", err)
		}
		outside := t.TempDir()
		scope := filepath.Join(root, "apps", "app", "users", "u1", "artifacts")
		if err := os.MkdirAll(scope, 0o777); err != nil {
			t.Fatal(err)
		}
		for name, dest := range map[string]string{
			// Another user's artifact, still inside the service root.
			"zq-other-user": "../../u2/artifacts/secret.txt",
			// A directory outside the root, and a missing path outside it.
			"zq-outside":  outside,
			"zq-dangling": filepath.Join(outside, "missing"),
			// The scope directory's parent itself.
			"zq-parent": "..",
		} {
			if err := os.Symlink(dest, filepath.Join(scope, name)); err != nil {
				t.Fatal(err)
			}
		}
		for _, name := range []string{"zq-other-user", "zq-outside", "zq-dangling", "zq-parent"} {
			for op, err := range callEveryOp(ctx, srv, "app", "u1", "", "user:"+name) {
				if !errors.Is(err, artifact.ErrInputValidation) {
					t.Errorf("%s(%s) error = %v, want artifact.ErrInputValidation", op, name, err)
					continue
				}
				if !strings.Contains(err.Error(), escapeMsg) || strings.Contains(err.Error(), "zq") {
					t.Errorf("%s(%s) error = %q, want it to contain %q and not the file name", op, name, err, escapeMsg)
				}
			}
		}
		versions, err := srv.Versions(ctx, &artifact.VersionsRequest{AppName: "app", UserID: "u2", FileName: "user:secret.txt"})
		if err != nil {
			t.Fatalf("Versions(victim) failed: %v", err)
		}
		if diff := cmp.Diff([]int64{1}, versions.Versions); diff != "" {
			t.Errorf("Versions(victim) mismatch (-want +got):\n%s", diff)
		}
		if entries, err := os.ReadDir(outside); err != nil || len(entries) != 0 {
			t.Errorf("ReadDir(outside) = %d entries, %v; want none", len(entries), err)
		}
	})
}

// adk-python's delete removes the versions directory under a save that is
// still writing into it, and the save fails with FileNotFoundError.
func TestFileService_SaveDeleteRace(t *testing.T) {
	ctx := t.Context()
	root := resolvedTempDir(t)
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	versionsDir := filepath.Join(root, "apps", "app", "users", "u", "sessions", "s", "artifacts", "race.txt", "versions")
	for i := range 50 {
		var wg sync.WaitGroup
		for range 4 {
			wg.Go(func() {
				if _, err := srv.Save(ctx, &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "race.txt", Part: genai.NewPartFromText("data")}); err != nil {
					t.Errorf("iteration %d: Save() failed: %v", i, err)
				}
			})
			wg.Go(func() {
				if err := srv.Delete(ctx, &artifact.DeleteRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "race.txt"}); err != nil {
					t.Errorf("iteration %d: Delete() failed: %v", i, err)
				}
			})
		}
		wg.Wait()
		if pending, _ := filepath.Glob(filepath.Join(versionsDir, ".*.pending")); len(pending) != 0 {
			t.Fatalf("iteration %d: staging directories left behind: %q", i, pending)
		}
		loaded, err := srv.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "race.txt"})
		switch {
		case errors.Is(err, fs.ErrNotExist):
		case err != nil:
			t.Fatalf("iteration %d: Load() failed: %v", i, err)
		case loaded.Part.Text != "data":
			t.Fatalf("iteration %d: Load() = %+v, want text %q", i, loaded.Part, "data")
		}
		if t.Failed() {
			return
		}
	}
}

// Delete holds the artifact's lock for writing, so a Load of the same
// artifact reads a version whole or finds none; it never reads the payload
// after its metadata is gone, which would turn the bytes into text.
func TestFileService_LoadDeleteRace(t *testing.T) {
	ctx := t.Context()
	srv, err := fileartifact.NewService(ctx, t.TempDir())
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	want := &genai.Part{InlineData: &genai.Blob{Data: []byte("bin"), MIMEType: "image/png"}}
	for i := range 300 {
		if _, err := srv.Save(ctx, &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "b.bin", Part: want}); err != nil {
			t.Fatalf("iteration %d: Save() failed: %v", i, err)
		}
		var wg sync.WaitGroup
		for range 3 {
			wg.Go(func() {
				if err := srv.Delete(ctx, &artifact.DeleteRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "b.bin"}); err != nil {
					t.Errorf("iteration %d: Delete() failed: %v", i, err)
				}
			})
			wg.Go(func() {
				loaded, err := srv.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "b.bin"})
				if errors.Is(err, fs.ErrNotExist) {
					return
				}
				if err != nil {
					t.Errorf("iteration %d: Load() failed: %v", i, err)
					return
				}
				if diff := cmp.Diff(want, loaded.Part); diff != "" {
					t.Errorf("iteration %d: Load() mismatch (-want +got):\n%s", i, diff)
				}
			})
		}
		wg.Wait()
		if t.Failed() {
			return
		}
	}
}

// Two services on one root share no lock, so a delete on one can remove the
// versions directory between the other's creation of it and its reservation
// of a version; the save then creates the directory again and goes on. Other
// failures of such a save are allowed, so only the reservation is checked.
// The four deletes of a round remove the directory at most four times, fewer
// than the retries a save allows.
func TestFileService_CrossServiceDeleteDuringReservation(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	saver, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	deleter, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	var (
		mu          sync.Mutex
		reserveErrs []error
	)
	for range 400 {
		var wg sync.WaitGroup
		for range 4 {
			wg.Go(func() {
				_, err := saver.Save(ctx, &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "r.txt", Part: genai.NewPartFromText("x")})
				if err != nil && strings.Contains(err.Error(), "reserve version") {
					mu.Lock()
					reserveErrs = append(reserveErrs, err)
					mu.Unlock()
				}
			})
			wg.Go(func() {
				// A delete may fail when the save writes into the directory it
				// removes; that is the documented cross-service behavior.
				_ = deleter.Delete(ctx, &artifact.DeleteRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "r.txt"})
			})
		}
		wg.Wait()
	}
	if len(reserveErrs) != 0 {
		t.Errorf("%d saves failed to reserve a version, first: %v", len(reserveErrs), reserveErrs[0])
	}
}

// An I/O error names the operation and keeps the cause, but not the path,
// which holds the caller's app, user, session and file names.
func TestSave_IOErrorOmitsCallerNames(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	userDir := filepath.Join(root, "apps", "zqapp", "users", "zquser")
	if err := os.MkdirAll(userDir, 0o777); err != nil {
		t.Fatal(err)
	}
	// A regular file where the user's sessions directory belongs.
	if err := os.WriteFile(filepath.Join(userDir, "sessions"), nil, 0o666); err != nil {
		t.Fatal(err)
	}
	_, err = srv.Save(ctx, &artifact.SaveRequest{AppName: "zqapp", UserID: "zquser", SessionID: "zqsession", FileName: "zqfile.txt", Part: genai.NewPartFromText("x")})
	if !errors.Is(err, syscall.ENOTDIR) {
		t.Fatalf("Save() error = %v, want syscall.ENOTDIR", err)
	}
	if strings.Contains(err.Error(), "zq") {
		t.Errorf("Save() error = %q, want none of the request's names", err)
	}
}

func TestList_MetadataFileNameFallbackAndPortedFlaws(t *testing.T) {
	ctx := t.Context()
	root := resolvedTempDir(t)
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	save := func(userID, sessionID, fileName, text string) int64 {
		t.Helper()
		resp, err := srv.Save(ctx, &artifact.SaveRequest{AppName: "app", UserID: userID, SessionID: sessionID, FileName: fileName, Part: genai.NewPartFromText(text)})
		if err != nil {
			t.Fatalf("Save(%q) failed: %v", fileName, err)
		}
		return resp.Version
	}
	list := func(userID, sessionID string) []string {
		t.Helper()
		resp, err := srv.List(ctx, &artifact.ListRequest{AppName: "app", UserID: userID, SessionID: sessionID})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
		return resp.FileNames
	}

	// adk-python strips surrounding white space from the path but keeps it in
	// the name, so the two names share one artifact listed under the last.
	t.Run("surrounding white space shares an artifact", func(t *testing.T) {
		if got := []int64{save("f1", "s", "a.txt", "one"), save("f1", "s", " a.txt", "two")}; !slices.Equal(got, []int64{1, 2}) {
			t.Errorf("Save() versions = %v, want [1 2]", got)
		}
		if diff := cmp.Diff([]string{" a.txt"}, list("f1", "s")); diff != "" {
			t.Errorf("List() mismatch (-want +got):\n%s", diff)
		}
		// Python's str.strip also removes the separators U+001C to U+001F.
		if got := []int64{save("f2", "s", "a.txt", "one"), save("f2", "s", "\x1fa.txt", "two")}; !slices.Equal(got, []int64{1, 2}) {
			t.Errorf("Save() versions with a leading U+001F = %v, want [1 2]", got)
		}
	})

	// adk-python lists a session-less save under its bare name, which a
	// session loads only with the "user:" prefix.
	t.Run("session-less save lists under the bare name", func(t *testing.T) {
		save("f4", "", "r.txt", "user-data")
		if diff := cmp.Diff([]string{"r.txt"}, list("f4", "s9")); diff != "" {
			t.Errorf("List() mismatch (-want +got):\n%s", diff)
		}
		if _, err := srv.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "f4", SessionID: "s9", FileName: "r.txt"}); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Load(r.txt) error = %v, want fs.ErrNotExist", err)
		}
		loaded, err := srv.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "f4", SessionID: "s9", FileName: "user:r.txt"})
		if err != nil {
			t.Fatalf("Load(user:r.txt) failed: %v", err)
		}
		if diff := cmp.Diff(genai.NewPartFromText("user-data"), loaded.Part); diff != "" {
			t.Errorf("Load(user:r.txt) mismatch (-want +got):\n%s", diff)
		}
	})

	// adk-python ignores the session ID of a "user:" name except in List.
	t.Run("user-scoped calls ignore the session ID", func(t *testing.T) {
		for op, err := range callEveryOp(ctx, srv, "app", "f3", "../../bad", "user:f.txt") {
			if err != nil {
				t.Errorf("%s() failed: %v", op, err)
			}
		}
		_, err := srv.List(ctx, &artifact.ListRequest{AppName: "app", UserID: "f3", SessionID: "../../bad"})
		if !errors.Is(err, artifact.ErrInputValidation) || !strings.Contains(err.Error(), "session_id must not contain traversal segments.") {
			t.Errorf("List() error = %v, want artifact.ErrInputValidation about traversal segments", err)
		}
	})

	t.Run("unreadable metadata falls back to the path", func(t *testing.T) {
		save("fb", "s1", "sfile", "s")
		save("fb", "s1", "user:ufile", "u")
		userDir := filepath.Join(root, "apps", "app", "users", "fb")
		if err := os.WriteFile(filepath.Join(userDir, "sessions", "s1", "artifacts", "sfile", "versions", "1", "metadata.json"), []byte("{corrupt"), 0o666); err != nil {
			t.Fatal(err)
		}
		shadowed := filepath.Join(userDir, "artifacts", "ufile", "versions", "1", "metadata.json")
		if err := os.Remove(shadowed); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(shadowed, 0o777); err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff([]string{"sfile", "user:ufile"}, list("fb", "s1")); diff != "" {
			t.Errorf("List() mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestLoadAndGetArtifactVersion_TamperedOrCorruptMetadataAndMissingPayload(t *testing.T) {
	ctx := t.Context()
	root := resolvedTempDir(t)
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	scope := filepath.Join(root, "apps", "app", "users", "u", "sessions", "s", "artifacts")
	writeVersion := func(fileName, metadata, payload string) {
		t.Helper()
		dir := filepath.Join(scope, fileName, "versions", "1")
		if err := os.MkdirAll(dir, 0o777); err != nil {
			t.Fatal(err)
		}
		if metadata != "" {
			if err := os.WriteFile(filepath.Join(dir, "metadata.json"), []byte(metadata), 0o666); err != nil {
				t.Fatal(err)
			}
		}
		if payload != "" {
			if err := os.WriteFile(filepath.Join(dir, fileName), []byte(payload), 0o666); err != nil {
				t.Fatal(err)
			}
		}
	}
	load := func(fileName string, version int64) (*artifact.LoadResponse, error) {
		return srv.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: fileName, Version: version})
	}
	getVersion := func(ctx context.Context, srv artifact.Service, fileName string) *artifact.ArtifactVersion {
		t.Helper()
		resp, err := srv.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: fileName})
		if err != nil {
			t.Fatalf("GetArtifactVersion(%q) failed: %v", fileName, err)
		}
		return resp.ArtifactVersion
	}

	t.Run("tampered canonical URI and missing payload", func(t *testing.T) {
		writeVersion("p.txt", `{"fileName":"p.txt","version":1,"mimeType":"text/plain","canonicalUri":"file:///etc/hostname","customMetadata":{}}`, "")
		if _, err := load("p.txt", 0); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Load() error = %v, want fs.ErrNotExist", err)
		}
		want := "file://" + root + "/apps/app/users/u/sessions/s/artifacts/p.txt/versions/1/p.txt"
		if got := getVersion(ctx, srv, "p.txt").CanonicalURI; got != want {
			t.Errorf("CanonicalURI = %q, want %q", got, want)
		}
	})

	t.Run("negative version", func(t *testing.T) {
		writeVersion("n.txt", "", "body")
		if _, err := load("n.txt", -1); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Load(version -1) error = %v, want fs.ErrNotExist", err)
		}
	})

	t.Run("unreadable metadata reads as none", func(t *testing.T) {
		fallback := time.Date(2030, time.January, 2, 3, 4, 5, 0, time.UTC)
		clock := platform.WithTimeProvider(ctx, func() time.Time { return fallback })
		writeVersion("corrupt.bin", "{corrupt", "body")
		writeVersion("shadowed.bin", "", "body")
		if err := os.Mkdir(filepath.Join(scope, "shadowed.bin", "versions", "1", "metadata.json"), 0o777); err != nil {
			t.Fatal(err)
		}
		for _, fileName := range []string{"corrupt.bin", "shadowed.bin"} {
			loaded, err := load(fileName, 0)
			if err != nil {
				t.Fatalf("Load(%q) failed: %v", fileName, err)
			}
			if diff := cmp.Diff(genai.NewPartFromText("body"), loaded.Part); diff != "" {
				t.Errorf("Load(%q) mismatch (-want +got):\n%s", fileName, diff)
			}
			got := getVersion(clock, srv, fileName)
			want := &artifact.ArtifactVersion{
				Version:        1,
				CanonicalURI:   "file://" + root + "/apps/app/users/u/sessions/s/artifacts/" + fileName + "/versions/1/" + fileName,
				CustomMetadata: map[string]any{},
				CreateTime:     fallback,
				MimeType:       "text/plain",
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("GetArtifactVersion(%q) mismatch (-want +got):\n%s", fileName, diff)
			}
		}
	})

	t.Run("create time survives a restart", func(t *testing.T) {
		if _, err := srv.Save(fixedClock(t), &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "kept.txt", Part: genai.NewPartFromText("x"), CustomMetadata: map[string]any{"k": true}}); err != nil {
			t.Fatalf("Save() failed: %v", err)
		}
		restarted, err := fileartifact.NewService(ctx, root)
		if err != nil {
			t.Fatalf("NewService() failed: %v", err)
		}
		later := platform.WithTimeProvider(ctx, func() time.Time { return fixedTime.Add(time.Hour) })
		got := getVersion(later, restarted, "kept.txt")
		if !got.CreateTime.Equal(fixedTime) {
			t.Errorf("CreateTime = %v, want %v", got.CreateTime, fixedTime)
		}
		if diff := cmp.Diff(map[string]any{"k": "true"}, got.CustomMetadata); diff != "" {
			t.Errorf("CustomMetadata mismatch (-want +got):\n%s", diff)
		}
	})
}

// File names may not hold separators yet, so the nested artifact here is
// written to disk directly, as a release that accepts nested names would.
func TestNestedArtifactLayout_ListVersionsAndDelete(t *testing.T) {
	ctx := t.Context()
	root := resolvedTempDir(t)
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	scope := filepath.Join(root, "apps", "app", "users", "u", "sessions", "s", "artifacts")
	name := func(fileName string, version int64) (*artifact.SaveRequest, *artifact.DeleteRequest, *artifact.VersionsRequest) {
		return &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: fileName, Part: genai.NewPartFromText(fileName)},
			&artifact.DeleteRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: fileName, Version: version},
			&artifact.VersionsRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: fileName}
	}

	t.Run("deleting a parent keeps the nested artifact", func(t *testing.T) {
		save, del, versions := name("doc", 0)
		if _, err := srv.Save(ctx, save); err != nil {
			t.Fatalf("Save() failed: %v", err)
		}
		nested := filepath.Join(scope, "doc", "nested", "versions", "1", "nested")
		if err := os.MkdirAll(filepath.Dir(nested), 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(nested, []byte("child"), 0o666); err != nil {
			t.Fatal(err)
		}
		list, err := srv.List(ctx, &artifact.ListRequest{AppName: "app", UserID: "u", SessionID: "s"})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
		if diff := cmp.Diff([]string{"doc", "doc/nested"}, list.FileNames); diff != "" {
			t.Errorf("List() mismatch (-want +got):\n%s", diff)
		}
		got, err := srv.Versions(ctx, versions)
		if err != nil {
			t.Fatalf("Versions() failed: %v", err)
		}
		if diff := cmp.Diff([]int64{1}, got.Versions); diff != "" {
			t.Errorf("Versions(doc) mismatch (-want +got):\n%s", diff)
		}
		if err := srv.Delete(ctx, del); err != nil {
			t.Fatalf("Delete() failed: %v", err)
		}
		if _, err := os.Stat(filepath.Join(scope, "doc", "versions")); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Stat(doc/versions) error = %v, want fs.ErrNotExist", err)
		}
		if _, err := os.Stat(nested); err != nil {
			t.Errorf("nested artifact removed with its parent: %v", err)
		}
	})

	t.Run("deleting the last version prunes the artifact directory", func(t *testing.T) {
		save, _, versions := name("two", 0)
		for range 2 {
			if _, err := srv.Save(ctx, save); err != nil {
				t.Fatalf("Save() failed: %v", err)
			}
		}
		_, delMissing, _ := name("two", 5)
		if err := srv.Delete(ctx, delMissing); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Delete(version 5) error = %v, want fs.ErrNotExist", err)
		}
		_, del2, _ := name("two", 2)
		if err := srv.Delete(ctx, del2); err != nil {
			t.Fatalf("Delete(version 2) failed: %v", err)
		}
		got, err := srv.Versions(ctx, versions)
		if err != nil {
			t.Fatalf("Versions() failed: %v", err)
		}
		if diff := cmp.Diff([]int64{1}, got.Versions); diff != "" {
			t.Errorf("Versions() mismatch (-want +got):\n%s", diff)
		}
		_, del1, _ := name("two", 1)
		if err := srv.Delete(ctx, del1); err != nil {
			t.Fatalf("Delete(version 1) failed: %v", err)
		}
		if _, err := os.Stat(filepath.Join(scope, "two")); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Stat(two) error = %v, want fs.ErrNotExist", err)
		}
		// The scope directory itself stays.
		if _, err := os.Stat(scope); err != nil {
			t.Errorf("Stat(scope) failed: %v", err)
		}
	})
}

func TestAppIsolationAndLegacyUnscopedLayoutIgnored(t *testing.T) {
	ctx := t.Context()
	root := resolvedTempDir(t)
	srv, err := fileartifact.NewService(ctx, root)
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	// Releases before the apps/{app} level wrote this tree, which names no
	// app.
	legacy := filepath.Join(root, "users", "u", "artifacts", "profile.txt", "versions", "1", "profile.txt")
	if err := os.MkdirAll(filepath.Dir(legacy), 0o777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(legacy, []byte("legacy"), 0o666); err != nil {
		t.Fatal(err)
	}
	assertMissing := func(appName string) {
		t.Helper()
		if _, err := srv.Load(ctx, &artifact.LoadRequest{AppName: appName, UserID: "u", FileName: "user:profile.txt"}); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Load(%s) error = %v, want fs.ErrNotExist", appName, err)
		}
		if _, err := srv.Versions(ctx, &artifact.VersionsRequest{AppName: appName, UserID: "u", FileName: "user:profile.txt"}); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Versions(%s) error = %v, want fs.ErrNotExist", appName, err)
		}
		if _, err := srv.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{AppName: appName, UserID: "u", FileName: "user:profile.txt"}); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("GetArtifactVersion(%s) error = %v, want fs.ErrNotExist", appName, err)
		}
		list, err := srv.List(ctx, &artifact.ListRequest{AppName: appName, UserID: "u"})
		if err != nil || len(list.FileNames) != 0 {
			t.Errorf("List(%s) = %v, %v; want no names", appName, list, err)
		}
	}
	saveIn := func(appName, text string) {
		t.Helper()
		resp, err := srv.Save(ctx, &artifact.SaveRequest{AppName: appName, UserID: "u", FileName: "user:profile.txt", Part: genai.NewPartFromText(text)})
		if err != nil {
			t.Fatalf("Save(%s) failed: %v", appName, err)
		}
		if resp.Version != 1 {
			t.Errorf("Save(%s).Version = %d, want 1", appName, resp.Version)
		}
	}
	assertLoads := func(appName, want string) {
		t.Helper()
		loaded, err := srv.Load(ctx, &artifact.LoadRequest{AppName: appName, UserID: "u", FileName: "user:profile.txt"})
		if err != nil {
			t.Fatalf("Load(%s) failed: %v", appName, err)
		}
		if loaded.Part.Text != want {
			t.Errorf("Load(%s) = %q, want %q", appName, loaded.Part.Text, want)
		}
	}

	assertMissing("app-a")
	saveIn("app-a", "secret-a")
	assertMissing("app-b")
	saveIn("app-b", "secret-b")
	assertLoads("app-a", "secret-a")
	if err := srv.Delete(ctx, &artifact.DeleteRequest{AppName: "app-b", UserID: "u", FileName: "user:profile.txt"}); err != nil {
		t.Fatalf("Delete(app-b) failed: %v", err)
	}
	assertMissing("app-b")
	assertLoads("app-a", "secret-a")
	if err := srv.Delete(ctx, &artifact.DeleteRequest{AppName: "app-a", UserID: "u", FileName: "user:profile.txt"}); err != nil {
		t.Fatalf("Delete(app-a) failed: %v", err)
	}
	if _, err := os.Stat(legacy); err != nil {
		t.Errorf("legacy payload touched: %v", err)
	}
}

// adk-python stores a name that resolves to the scope directory itself under
// "artifact", where it collides with an artifact of that name.
func TestFileService_RejectsEmptyFileName(t *testing.T) {
	ctx := t.Context()
	srv, err := fileartifact.NewService(ctx, t.TempDir())
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	real := &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "artifact", Part: genai.NewPartFromText("real")}
	if _, err := srv.Save(ctx, real); err != nil {
		t.Fatalf("Save(artifact) failed: %v", err)
	}
	for _, fileName := range []string{"", "   ", ".", "./.", "user:", "user:   ", "user:."} {
		for op, err := range callEveryOp(ctx, srv, "app", "u", "s", fileName) {
			if !errors.Is(err, artifact.ErrInputValidation) || !strings.Contains(err.Error(), "artifact filename must not be empty") {
				t.Errorf("%s(%q) error = %v, want artifact.ErrInputValidation about an empty file name", op, fileName, err)
			}
		}
	}
	versions, err := srv.Versions(ctx, &artifact.VersionsRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "artifact"})
	if err != nil {
		t.Fatalf("Versions(artifact) failed: %v", err)
	}
	if diff := cmp.Diff([]int64{1}, versions.Versions); diff != "" {
		t.Errorf("Versions(artifact) mismatch (-want +got):\n%s", diff)
	}
}
