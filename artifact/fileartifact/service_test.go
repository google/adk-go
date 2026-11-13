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
	"context"
	"errors"
	"io/fs"
	"math"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	artifacttests "google.golang.org/adk/v2/internal/artifact/tests"
	"google.golang.org/adk/v2/platform"
)

func TestServiceContract(t *testing.T) {
	artifacttests.TestArtifactService(t, "File", func(t *testing.T) (artifact.Service, error) {
		return New(Config{RootDir: t.TempDir()})
	})
}

func newService(t *testing.T, dir string) artifact.Service {
	t.Helper()
	s, err := New(Config{RootDir: dir})
	if err != nil {
		t.Fatalf("New failed: %T", err)
	}
	return s
}

func saveRequest(name string) *artifact.SaveRequest {
	return &artifact.SaveRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: name, Part: genai.NewPartFromText("fixture")}
}

func loadRequest(name string) *artifact.LoadRequest {
	return &artifact.LoadRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: name}
}

// savedArtifact returns a service containing version 1 of name and its root directory.
func savedArtifact(t *testing.T, name string) (artifact.Service, string) {
	t.Helper()
	dir := t.TempDir()
	s := newService(t, dir)
	if _, err := s.Save(t.Context(), saveRequest(name)); err != nil {
		t.Fatalf("Save fixture failed: %T", err)
	}
	return s, dir
}

func TestPersistence(t *testing.T) {
	fixed := time.Date(2026, 1, 2, 3, 4, 5, 123456789, time.UTC)
	for _, tc := range []struct {
		name    string
		part    *genai.Part
		mime    string
		created time.Time
	}{
		{"text", genai.NewPartFromText("persistent text"), "text/plain", fixed},
		{"binary", &genai.Part{InlineData: &genai.Blob{Data: []byte{0, 1, 255}, MIMEType: "image/png", DisplayName: "image.png"}}, "image/png", fixed},
		{"untyped blob", genai.NewPartFromBytes([]byte{1, 2}, ""), "", fixed},
		{"empty blob", genai.NewPartFromBytes([]byte{}, "application/octet-stream"), "application/octet-stream", fixed},
		{"zero creation time", genai.NewPartFromText("fixture"), "text/plain", time.Time{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "root with spaces # %", "nested")
			writer := newService(t, dir)
			ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return tc.created })
			req := saveRequest("report")
			req.Part = tc.part
			req.CustomMetadata = map[string]any{"count": 7, "ready": true}
			req.Version = 42 // SaveRequest.Version is ignored by the existing contract.
			saved, err := writer.Save(ctx, req)
			if err != nil || saved.Version != 1 {
				t.Fatalf("Save did not allocate the first version: %T", err)
			}
			req.CustomMetadata["count"] = 99
			reader := newService(t, dir)
			loaded, err := reader.Load(t.Context(), loadRequest("report"))
			if err != nil {
				t.Fatalf("Load after reopening failed: %T", err)
			}
			if !cmp.Equal(loaded.Part, tc.part) {
				t.Error("reopening changed the stored part")
			}
			metadataReq := &artifact.GetArtifactVersionRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "report"}
			info, err := reader.GetArtifactVersion(t.Context(), metadataReq)
			if err != nil {
				t.Fatalf("metadata after reopening failed: %T", err)
			}
			got := info.ArtifactVersion
			if got.Version != 1 || got.MimeType != tc.mime || !got.CreateTime.Equal(tc.created) {
				t.Error("version metadata did not survive reopening")
			}
			if !cmp.Equal(got.CustomMetadata, map[string]any{"count": "7", "ready": "true"}) {
				t.Error("custom metadata was not copied and stringified")
			}
			uri, err := url.Parse(got.CanonicalURI)
			if err != nil || uri.Scheme != "file" || uri.Host != "" || !strings.HasPrefix(uri.Path, "/") || strings.HasPrefix(uri.Path, "//") {
				t.Fatal("metadata did not return a file URI")
			}
			payloadPath := uri.Path
			if runtime.GOOS == "windows" {
				payloadPath = strings.TrimPrefix(payloadPath, "/")
			}
			payload, err := os.ReadFile(filepath.FromSlash(payloadPath))
			if err != nil {
				t.Fatalf("canonical URI is not a readable payload: %T", err)
			}
			want := []byte(tc.part.Text)
			if tc.part.InlineData != nil {
				want = tc.part.InlineData.Data
			}
			if !slices.Equal(payload, want) {
				t.Error("canonical URI does not identify the raw payload")
			}
			got.CustomMetadata["count"] = "changed"
			again, err := reader.GetArtifactVersion(t.Context(), metadataReq)
			if err != nil || again.ArtifactVersion.CustomMetadata["count"] != "7" {
				t.Error("returned metadata aliases storage")
			}
		})
	}
}

func TestScopeIsolation(t *testing.T) {
	dir := t.TempDir()
	s := newService(t, dir)
	cases := []struct{ app, user, session, name string }{
		{"app", "user", "session", "file"},
		{"App", "user", "session", "file"},
		{"app", "User", "session", "file"},
		{"app", "user", "Session", "file"},
		{"app", "user", "session", "File"},
		{"app", "user", "user", "file"},
		{"app", "user", "session", "user:file"},
		{"app", "user", "session", "metadata.json"},
		{"app", "user", "session", "versions"},
		{"app", "user", "session", ".."},
		{"app", "user", "session", "NUL"},
		{"app", "user", "session", strings.Repeat("long-name", 40)},
		{"../app", "a/b", "..", "文件.txt"},
	}
	for i, tc := range cases {
		saved, err := s.Save(t.Context(), &artifact.SaveRequest{AppName: tc.app, UserID: tc.user, SessionID: tc.session, FileName: tc.name, Part: genai.NewPartFromText(strconv.Itoa(i))})
		if err != nil || saved.Version != 1 {
			t.Fatalf("scope %d collided with another scope: %T", i, err)
		}
	}
	for i, tc := range cases {
		got, err := s.Load(t.Context(), &artifact.LoadRequest{AppName: tc.app, UserID: tc.user, SessionID: tc.session, FileName: tc.name})
		if err != nil || got.Part.Text != strconv.Itoa(i) {
			t.Errorf("scope %d returned another scope's payload: %T", i, err)
		}
	}
	for _, tc := range []struct{ app, user string }{{"other-app", "user"}, {"app", "other-user"}} {
		_, err := s.Load(t.Context(), &artifact.LoadRequest{AppName: tc.app, UserID: tc.user, SessionID: "session", FileName: "user:file"})
		if !errors.Is(err, fs.ErrNotExist) {
			t.Error("user-scoped file crossed app or user boundary")
		}
	}
}

func TestConcurrentSaves(t *testing.T) {
	const count = 24
	dir := t.TempDir()
	var wg sync.WaitGroup
	versions := make([]int64, count)
	for i := range count {
		// Construct separate services to exercise coordination on disk.
		service := newService(t, dir)
		wg.Go(func() {
			req := saveRequest("shared.bin")
			req.Part = genai.NewPartFromBytes([]byte{byte(i)}, "application/octet-stream")
			result, err := service.Save(t.Context(), req)
			if err != nil {
				t.Errorf("concurrent Save failed: %T", err)
				return
			}
			versions[i] = result.Version
		})
	}
	wg.Wait()
	s := newService(t, dir)
	seen := make(map[int64]bool)
	for i, version := range versions {
		if version <= 0 || seen[version] {
			t.Fatal("concurrent saves did not receive distinct positive versions")
		}
		seen[version] = true
		req := loadRequest("shared.bin")
		req.Version = version
		got, err := s.Load(t.Context(), req)
		if err != nil || !slices.Equal(got.Part.InlineData.Data, []byte{byte(i)}) {
			t.Errorf("saved version %d was overwritten: %T", version, err)
		}
	}
	got, err := s.Versions(t.Context(), &artifact.VersionsRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "shared.bin"})
	if err != nil || len(got.Versions) != count || !slices.IsSorted(got.Versions) {
		t.Fatal("version listing lost or misordered concurrent saves")
	}
}

func TestProcessPersistence(t *testing.T) {
	if dir := os.Getenv("ADK_FILEARTIFACT_TEST_ROOT"); dir != "" {
		s := newService(t, dir)
		req := saveRequest("process.txt")
		req.CustomMetadata = map[string]any{"source": "child"}
		if _, err := s.Save(t.Context(), req); err != nil {
			t.Fatalf("child Save failed: %T", err)
		}
		return
	}
	dir := t.TempDir()
	const count = 3
	commands := make([]*exec.Cmd, count)
	for i := range count {
		cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestProcessPersistence$")
		cmd.Env = append(os.Environ(), "ADK_FILEARTIFACT_TEST_ROOT="+dir)
		if err := cmd.Start(); err != nil {
			t.Fatalf("start child failed: %T", err)
		}
		commands[i] = cmd
	}
	for _, cmd := range commands {
		if err := cmd.Wait(); err != nil {
			t.Fatalf("child failed: %T", err)
		}
	}
	// All writer processes have exited before this reader is constructed.
	s := newService(t, dir)
	versions, err := s.Versions(t.Context(), &artifact.VersionsRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "process.txt"})
	if err != nil || len(versions.Versions) != count {
		t.Fatal("versions did not survive writer process exits")
	}
	for _, version := range versions.Versions {
		req := loadRequest("process.txt")
		req.Version = version
		loaded, err := s.Load(t.Context(), req)
		if err != nil || loaded.Part.Text != "fixture" {
			t.Error("payload did not survive writer process exit")
		}
		info, err := s.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "process.txt", Version: version})
		if err != nil || info.ArtifactVersion.CustomMetadata["source"] != "child" {
			t.Error("metadata did not survive writer process exit")
		}
	}
}

func TestIncompleteVersions(t *testing.T) {
	dir := t.TempDir()
	s := newService(t, dir)
	path := filepath.Join(dir, versionsPath("app", "user", "session", "file"))
	for _, name := range []string{".1.pending", "junk", "0", "-1", "01"} {
		if err := os.MkdirAll(filepath.Join(path, name), 0o700); err != nil {
			t.Fatal("create fixture failed")
		}
	}
	if err := os.WriteFile(filepath.Join(path, "2"), nil, 0o600); err != nil {
		t.Fatal("create fixture failed")
	}
	listReq := &artifact.ListRequest{AppName: "app", UserID: "user", SessionID: "session"}
	listed, err := s.List(t.Context(), listReq)
	if err != nil || len(listed.FileNames) != 0 {
		t.Fatal("an incomplete artifact was listed")
	}
	if _, err := s.Load(t.Context(), loadRequest("file")); !errors.Is(err, fs.ErrNotExist) {
		t.Error("an incomplete artifact was loaded")
	}
	if _, err := s.Versions(t.Context(), &artifact.VersionsRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "file"}); !errors.Is(err, fs.ErrNotExist) {
		t.Error("incomplete or noncanonical versions were listed")
	}
	req := saveRequest("file")
	saved, err := s.Save(t.Context(), req)
	if err != nil || saved.Version != 3 {
		t.Fatalf("Save did not skip an interrupted reservation: %T", err)
	}
	listed, err = s.List(t.Context(), listReq)
	if err != nil || !slices.Equal(listed.FileNames, []string{"file"}) {
		t.Error("published artifact missing from listing")
	}
	if err := s.Delete(t.Context(), &artifact.DeleteRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "file"}); err != nil {
		t.Fatal("Delete failed")
	}
	if _, err := os.Stat(filepath.Join(path, ".1.pending")); err != nil {
		t.Error("Delete removed an in-progress reservation")
	}
}

func TestFailedSaveIsNotPublished(t *testing.T) {
	// The platform clock runs before I/O and may cancel the request.
	dir := t.TempDir()
	s := newService(t, dir)
	ctx, cancel := context.WithCancel(t.Context())
	ctx = platform.WithTimeProvider(ctx, func() time.Time { cancel(); return time.Now() })
	if _, err := s.Save(ctx, saveRequest("file")); !errors.Is(err, context.Canceled) {
		t.Error("Save ignored cancellation")
	}
	listed, err := s.List(t.Context(), &artifact.ListRequest{AppName: "app", UserID: "user", SessionID: "session"})
	if err != nil || len(listed.FileNames) != 0 {
		t.Fatal("failed save became visible")
	}
	saved, err := s.Save(t.Context(), saveRequest("file"))
	if err != nil || saved.Version != 1 {
		t.Error("failed save consumed a version")
	}
}

func TestInvalidRequests(t *testing.T) {
	s := newService(t, t.TempDir())
	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"nil save", func() error { _, err := s.Save(t.Context(), nil); return err }},
		{"nil load", func() error { _, err := s.Load(t.Context(), nil); return err }},
		{"nil metadata", func() error { _, err := s.GetArtifactVersion(t.Context(), nil); return err }},
		{"nil list", func() error { _, err := s.List(t.Context(), nil); return err }},
		{"nil versions", func() error { _, err := s.Versions(t.Context(), nil); return err }},
		{"nil delete", func() error { return s.Delete(t.Context(), nil) }},
		{"empty save", func() error { _, err := s.Save(t.Context(), &artifact.SaveRequest{}); return err }},
		{"empty load", func() error { _, err := s.Load(t.Context(), &artifact.LoadRequest{}); return err }},
		{"empty metadata", func() error {
			_, err := s.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{})
			return err
		}},
		{"empty list", func() error { _, err := s.List(t.Context(), &artifact.ListRequest{}); return err }},
		{"empty versions", func() error { _, err := s.Versions(t.Context(), &artifact.VersionsRequest{}); return err }},
		{"empty delete", func() error { return s.Delete(t.Context(), &artifact.DeleteRequest{}) }},
		{"negative load version", func() error {
			req := loadRequest("file")
			req.Version = -1
			_, err := s.Load(t.Context(), req)
			return err
		}},
		{"negative metadata version", func() error {
			_, err := s.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "file", Version: -1})
			return err
		}},
		{"negative delete version", func() error {
			return s.Delete(t.Context(), &artifact.DeleteRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "file", Version: -1})
		}},
		{"filename with slash", func() error { _, err := s.Save(t.Context(), saveRequest("dir/file")); return err }},
		{"filename with backslash", func() error { _, err := s.Save(t.Context(), saveRequest(`dir\file`)); return err }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.call(); err == nil {
				t.Error("invalid request accepted")
			}
		})
	}
}

func TestCanceledOperations(t *testing.T) {
	s := newService(t, t.TempDir())
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	calls := []func() error{
		func() error { _, err := s.Save(ctx, saveRequest("file")); return err },
		func() error { _, err := s.Load(ctx, loadRequest("file")); return err },
		func() error {
			_, err := s.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "file"})
			return err
		},
		func() error {
			_, err := s.List(ctx, &artifact.ListRequest{AppName: "app", UserID: "user", SessionID: "session"})
			return err
		},
		func() error {
			_, err := s.Versions(ctx, &artifact.VersionsRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "file"})
			return err
		},
		func() error {
			return s.Delete(ctx, &artifact.DeleteRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "file"})
		},
	}
	for i, call := range calls {
		if !errors.Is(call(), context.Canceled) {
			t.Errorf("operation %d did not preserve cancellation", i)
		}
	}
}

func TestNew(t *testing.T) {
	if _, err := New(Config{}); !errors.Is(err, fs.ErrInvalid) {
		t.Error("empty root was accepted")
	}
	file := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal("create fixture failed")
	}
	if _, err := New(Config{RootDir: file}); err == nil {
		t.Error("regular file accepted as storage root")
	}
}

func TestCorruptMetadata(t *testing.T) {
	for _, data := range []string{"{", "null", "{}", `{"fileName":"file","customMetadata":{}}`, `{"createTime":"2026-01-01T00:00:00Z","customMetadata":{}}`, `{"fileName":"file","createTime":"2026-01-01T00:00:00Z","customMetadata":null}`, `{"fileName":"file","createTime":"2026-01-01T00:00:00Z","customMetadata":{"secret":"ok"},"text":"sensitive-value"}`} {
		s, dir := savedArtifact(t, "file")
		path := filepath.Join(dir, versionsPath("app", "user", "session", "file"), "1", "metadata.json")
		if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
			t.Fatal("corrupt fixture failed")
		}
		_, err := s.Load(t.Context(), loadRequest("file"))
		if !errors.Is(err, fs.ErrInvalid) {
			t.Error("corrupt metadata was accepted")
		}
		if err != nil && strings.Contains(err.Error(), "sensitive-value") {
			t.Error("error disclosed metadata contents")
		}
	}
}

func TestSymlinks(t *testing.T) {
	for _, target := range []string{"external", "other scope"} {
		t.Run(target, func(t *testing.T) {
			dir := t.TempDir()
			s := newService(t, dir)
			req := saveRequest("file")
			req.AppName = "private-app"
			if _, err := s.Save(t.Context(), req); err != nil {
				t.Fatal("Save failed")
			}
			source := filepath.Join(dir, "apps", encodeName("private-app"))
			dest := t.TempDir()
			if target == "other scope" {
				other := saveRequest("file")
				other.AppName = "other-app"
				if _, err := s.Save(t.Context(), other); err != nil {
					t.Fatal("Save failed")
				}
				dest = encodeName("other-app")
			}
			if err := os.RemoveAll(source); err != nil {
				t.Fatal("remove fixture failed")
			}
			if err := os.Symlink(dest, source); err != nil {
				t.Fatalf("symlink fixture failed: %T", err)
			}
			if _, err := s.Save(t.Context(), req); err == nil {
				t.Error("Save followed a symlinked app directory")
			}
			load := loadRequest("file")
			load.AppName = "private-app"
			if _, err := s.Load(t.Context(), load); err == nil {
				t.Error("Load followed a symlinked app directory")
			}
			err := s.Delete(t.Context(), &artifact.DeleteRequest{AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID, FileName: req.FileName})
			if err == nil {
				t.Error("Delete followed a symlinked app directory")
			}
		})
	}
	s, dir := savedArtifact(t, "file")
	outside := filepath.Join(t.TempDir(), "secret")
	if err := os.WriteFile(outside, []byte("secret"), 0o600); err != nil {
		t.Fatal("create outside fixture failed")
	}
	payload := filepath.Join(dir, versionsPath("app", "user", "session", "file"), "1", "data")
	if err := os.Remove(payload); err != nil {
		t.Fatal("remove payload failed")
	}
	if err := os.Symlink(outside, payload); err != nil {
		t.Fatal("create payload symlink failed")
	}
	if _, err := s.Load(t.Context(), loadRequest("file")); err == nil {
		t.Error("Load followed a payload symlink outside the version")
	}
}

func TestFilesystemErrorsAreRedacted(t *testing.T) {
	dir := t.TempDir()
	s := newService(t, dir)
	if err := os.RemoveAll(dir); err != nil {
		t.Fatal("remove root failed")
	}
	_, err := s.Load(t.Context(), loadRequest("file"))
	if !errors.Is(err, fs.ErrNotExist) {
		t.Error("missing root error lost its cause")
	}
	if err != nil && strings.Contains(err.Error(), dir) {
		t.Error("filesystem error disclosed the storage path")
	}
	linkErr := fileError("publish", &os.LinkError{Op: "rename", Old: "sensitive-old", New: "sensitive-new", Err: fs.ErrPermission})
	if !errors.Is(linkErr, fs.ErrPermission) || strings.Contains(linkErr.Error(), "sensitive") {
		t.Error("rename error was not redacted while preserving its cause")
	}
}

func TestVersionLimit(t *testing.T) {
	dir := t.TempDir()
	s := newService(t, dir)
	versionDir := filepath.Join(dir, versionsPath("app", "user", "session", "file"), strconv.FormatInt(math.MaxInt64, 10))
	if err := os.MkdirAll(versionDir, 0o700); err != nil {
		t.Fatal("create fixture failed")
	}
	if _, err := s.Save(t.Context(), saveRequest("file")); !errors.Is(err, fs.ErrInvalid) {
		t.Error("version overflow was accepted")
	}
}

// observingContext lets a test cancel at a filesystem boundary without timing
// assumptions or adding fault-injection hooks to the service.
type observingContext struct {
	context.Context
	observe func()
}

func (c observingContext) Err() error {
	c.observe()
	return c.Context.Err()
}

func TestCancellationBeforePublication(t *testing.T) {
	writer, dir := savedArtifact(t, "file")
	reader := newService(t, dir)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	observed := false
	pending := filepath.Join(dir, versionsPath("app", "user", "session", "file"), ".2.pending")
	ctx = observingContext{Context: ctx, observe: func() {
		if _, err := os.Stat(filepath.Join(pending, "metadata.json")); err != nil {
			return
		}
		observed = true
		loaded, err := reader.Load(t.Context(), loadRequest("file"))
		if err != nil || loaded.Part.Text != "fixture" {
			t.Error("unpublished version replaced the previous payload")
		}
		versions, err := reader.Versions(t.Context(), &artifact.VersionsRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "file"})
		if err != nil || !slices.Equal(versions.Versions, []int64{1}) {
			t.Error("unpublished version became visible")
		}
		cancel()
	}}
	req := saveRequest("file")
	req.Part = genai.NewPartFromText("replacement")
	if _, err := writer.Save(ctx, req); !errors.Is(err, context.Canceled) {
		t.Error("Save published after cancellation")
	}
	if !observed {
		t.Fatal("test did not reach staged payload and metadata")
	}
	if _, err := os.Stat(pending); !errors.Is(err, fs.ErrNotExist) {
		t.Error("failed Save left its staging directory")
	}
	saved, err := writer.Save(t.Context(), req)
	if err != nil || saved.Version != 2 {
		t.Error("failed Save left a version reservation")
	}
	loaded, err := reader.Load(t.Context(), loadRequest("file"))
	if err != nil || loaded.Part.Text != "replacement" {
		t.Error("successful save was not published")
	}
}

func TestMissingVersionFiles(t *testing.T) {
	for _, name := range []string{"data", "metadata.json"} {
		t.Run(name, func(t *testing.T) {
			s, dir := savedArtifact(t, "file")
			path := filepath.Join(dir, versionsPath("app", "user", "session", "file"), "1", name)
			if err := os.Remove(path); err != nil {
				t.Fatal("remove fixture failed")
			}
			if _, err := s.Load(t.Context(), loadRequest("file")); !errors.Is(err, fs.ErrNotExist) {
				t.Error("missing version file was not reported")
			}
			if name == "metadata.json" {
				_, err := s.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "file"})
				if !errors.Is(err, fs.ErrNotExist) {
					t.Error("missing metadata was fabricated")
				}
			}
		})
	}
}

func TestMetadataEncodingFailure(t *testing.T) {
	s := newService(t, t.TempDir())
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return time.Date(10000, 1, 1, 0, 0, 0, 0, time.UTC) })
	if _, err := s.Save(ctx, saveRequest("file")); !errors.Is(err, fs.ErrInvalid) {
		t.Error("invalid creation timestamp was accepted")
	}
	got, err := s.List(t.Context(), &artifact.ListRequest{AppName: "app", UserID: "user", SessionID: "session"})
	if err != nil || len(got.FileNames) != 0 {
		t.Error("invalid metadata left a visible artifact")
	}
}

func TestConcurrentDeleteAndSave(t *testing.T) {
	dir := t.TempDir()
	writer := newService(t, dir)
	deleter := newService(t, dir)
	reader := newService(t, dir)
	var wg sync.WaitGroup
	wg.Go(func() {
		for range 30 {
			if _, err := writer.Save(t.Context(), saveRequest("file")); err != nil {
				t.Errorf("Save failed during deletion: %T", err)
				return
			}
		}
	})
	wg.Go(func() {
		for range 30 {
			if err := deleter.Delete(t.Context(), &artifact.DeleteRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "file"}); err != nil {
				t.Errorf("concurrent Delete failed: %T", err)
				return
			}
		}
	})
	wg.Go(func() {
		for range 30 {
			got, err := reader.Load(t.Context(), loadRequest("file"))
			if err == nil && got.Part.Text != "fixture" {
				t.Error("concurrent Load returned incomplete contents")
				return
			}
			if err != nil && !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("concurrent Load failed unexpectedly: %T", err)
				return
			}
		}
	})
	wg.Wait()
	if _, err := writer.Save(t.Context(), saveRequest("file")); err != nil {
		t.Fatal("Save after concurrent operations failed")
	}
	if _, err := reader.Load(t.Context(), loadRequest("file")); err != nil {
		t.Fatal("Load after concurrent operations failed")
	}
}

func TestInvalidMetadataStrings(t *testing.T) {
	invalid := string([]byte{0xff})
	for _, tc := range []struct {
		name   string
		change func(*artifact.SaveRequest)
	}{
		{"filename", func(r *artifact.SaveRequest) { r.FileName = invalid }},
		{"MIME type", func(r *artifact.SaveRequest) { r.Part = genai.NewPartFromBytes([]byte{1}, invalid) }},
		{"display name", func(r *artifact.SaveRequest) {
			r.Part = &genai.Part{InlineData: &genai.Blob{Data: []byte{1}, DisplayName: invalid}}
		}},
		{"metadata key", func(r *artifact.SaveRequest) { r.CustomMetadata = map[string]any{invalid: "value"} }},
		{"metadata value", func(r *artifact.SaveRequest) { r.CustomMetadata = map[string]any{"key": invalid} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newService(t, t.TempDir())
			req := saveRequest("file")
			tc.change(req)
			if _, err := s.Save(t.Context(), req); !errors.Is(err, fs.ErrInvalid) {
				t.Error("invalid UTF-8 accepted for JSON metadata")
			}
			listed, err := s.List(t.Context(), &artifact.ListRequest{AppName: "app", UserID: "user", SessionID: "session"})
			if err != nil || len(listed.FileNames) != 0 {
				t.Error("rejected metadata affected artifact listing")
			}
		})
	}
}

func TestSaveReportsCleanupFailure(t *testing.T) {
	if runtime.GOOS == "windows" || os.Geteuid() == 0 {
		t.Skip("requires Unix directory permissions for fault injection")
	}
	dir := t.TempDir()
	s := newService(t, dir)
	versionsDir := filepath.Join(dir, versionsPath("app", "user", "session", "file"))
	t.Cleanup(func() { _ = os.Chmod(versionsDir, 0o700) })
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	observed := false
	ctx = observingContext{Context: ctx, observe: func() {
		if observed {
			return
		}
		if _, err := os.Stat(filepath.Join(versionsDir, ".1.pending", "metadata.json")); err != nil {
			return
		}
		observed = true
		if err := os.Chmod(versionsDir, 0o500); err != nil {
			t.Fatal("make cleanup unwritable failed")
		}
		cancel()
	}}
	_, err := s.Save(ctx, saveRequest("file"))
	if !observed {
		t.Fatal("test did not reach staged metadata")
	}
	if !errors.Is(err, context.Canceled) || !errors.Is(err, fs.ErrPermission) {
		t.Error("Save did not report both cancellation and failed cleanup")
	}
}

func TestListRejectsMisplacedMetadata(t *testing.T) {
	for _, tc := range []struct{ name, savedName, listedName string }{
		{"wrong filename", "file", "different-file"},
		{"wrong scope", "user:file", "user:file"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, dir := savedArtifact(t, tc.savedName)
			source := filepath.Dir(filepath.Join(dir, versionsPath("app", "user", "session", tc.savedName)))
			target := filepath.Join(dir, scopePath("app", "user", "session", false), encodeName(tc.listedName))
			if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
				t.Fatal("create destination scope failed")
			}
			// Move a complete artifact so its metadata no longer matches its location.
			if err := os.Rename(source, target); err != nil {
				t.Fatal("move fixture failed")
			}
			_, err := s.List(t.Context(), &artifact.ListRequest{AppName: "app", UserID: "user", SessionID: "session"})
			if !errors.Is(err, fs.ErrInvalid) {
				t.Error("List accepted metadata from a different filename or scope")
			}
		})
	}
}
