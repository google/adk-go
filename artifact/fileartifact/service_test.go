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
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/artifact/fileartifact"
	"google.golang.org/adk/v2/internal/artifact/tests"
	"google.golang.org/adk/v2/platform"
)

const (
	app     = "app"
	user    = "user"
	session = "session"
)

func TestServiceContract(t *testing.T) {
	tests.TestArtifactService(t, "File", func(t *testing.T) (artifact.Service, error) {
		return fileartifact.New(fileartifact.Config{RootDir: t.TempDir()})
	})
}

func newService(t *testing.T, root string) artifact.Service {
	t.Helper()
	service, err := fileartifact.New(fileartifact.Config{RootDir: root})
	must(t, err, "New()")
	return service
}

// newRoot returns an empty root directory with symbolic links resolved, so
// that paths built from it match the ones the service reports.
func newRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	must(t, err, "EvalSymlinks()")
	return root
}

// must stops the test when a step it depends on failed.
func must(t *testing.T, err error, step string) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s failed: %v", step, err)
	}
}

func saveRequest(name string, part *genai.Part) *artifact.SaveRequest {
	return &artifact.SaveRequest{AppName: app, UserID: user, SessionID: session, FileName: name, Part: part}
}

func loadRequest(name string) *artifact.LoadRequest {
	return &artifact.LoadRequest{AppName: app, UserID: user, SessionID: session, FileName: name}
}

func versionRequest(name string) *artifact.GetArtifactVersionRequest {
	return &artifact.GetArtifactVersionRequest{AppName: app, UserID: user, SessionID: session, FileName: name}
}

func deleteRequest(name string, version *int64) *artifact.DeleteRequest {
	return &artifact.DeleteRequest{AppName: app, UserID: user, SessionID: session, FileName: name, Version: version}
}

func mustSave(t *testing.T, service artifact.Service, name string, part *genai.Part) int64 {
	t.Helper()
	resp, err := service.Save(t.Context(), saveRequest(name, part))
	must(t, err, "Save()")
	return resp.Version
}

func mustList(t *testing.T, service artifact.Service) []string {
	t.Helper()
	resp, err := service.List(t.Context(), &artifact.ListRequest{AppName: app, UserID: user, SessionID: session})
	must(t, err, "List()")
	return resp.FileNames
}

// sessionDir is the artifacts directory of the test session and userDir the
// one user-scoped artifacts go to.
func sessionDir(root string) string {
	return filepath.Join(root, "apps", app, "users", user, "sessions", session, "artifacts")
}

func userDir(root string) string {
	return filepath.Join(root, "apps", app, "users", user, "artifacts")
}

func writeFile(t *testing.T, path string, data []byte) {
	t.Helper()
	must(t, os.WriteFile(path, data, 0o600), "WriteFile()")
}

func dirEntries(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	must(t, err, "ReadDir()")
	names := make([]string, len(entries))
	for i, entry := range entries {
		names[i] = entry.Name()
	}
	return names
}

// fileURI is the file URI of a local path.
func fileURI(path string) string {
	path = filepath.ToSlash(path)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return (&url.URL{Scheme: "file", Path: path}).String()
}

// refused reports whether err is the service declining to work on storage it
// does not trust, which is neither a bad request nor a missing artifact.
func refused(err error) bool {
	return err != nil && !errors.Is(err, fs.ErrInvalid) && !errors.Is(err, fs.ErrNotExist)
}

func TestSaveLayout(t *testing.T) {
	root := newRoot(t)
	service := newService(t, root)
	createTime := time.Date(2026, time.January, 2, 3, 4, 5, 123456789, time.UTC)
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return createTime })

	for _, tc := range []struct {
		name     string
		fileName string
		part     *genai.Part
		// wantDir is the artifact's directory and wantPayload its payload file.
		wantDir     string
		wantPayload string
		wantMeta    map[string]any
	}{
		{
			name:        "session-scoped text",
			fileName:    "notes.txt",
			part:        genai.NewPartFromText("hello"),
			wantDir:     filepath.Join(sessionDir(root), "notes.txt"),
			wantPayload: "notes.txt",
			wantMeta:    map[string]any{"fileName": "notes.txt"},
		},
		{
			name:        "nested inline data",
			fileName:    "reports/q1/chart.png",
			part:        &genai.Part{InlineData: &genai.Blob{Data: []byte("hello"), MIMEType: "image/png", DisplayName: "Chart"}},
			wantDir:     filepath.Join(sessionDir(root), "reports", "q1", "chart.png"),
			wantPayload: "chart.png",
			wantMeta:    map[string]any{"fileName": "reports/q1/chart.png", "mimeType": "image/png", "displayName": "Chart"},
		},
		{
			name:        "user-scoped",
			fileName:    "user:profile/avatar.bin",
			part:        genai.NewPartFromBytes([]byte("hello"), "application/octet-stream"),
			wantDir:     filepath.Join(userDir(root), "profile", "avatar.bin"),
			wantPayload: "avatar.bin",
			wantMeta:    map[string]any{"fileName": "user:profile/avatar.bin", "mimeType": "application/octet-stream"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := saveRequest(tc.fileName, tc.part)
			req.CustomMetadata = map[string]any{"count": 7}
			// The second version is the one inspected, so that its number
			// is not the zero value.
			for range 2 {
				_, err := service.Save(ctx, req)
				must(t, err, "Save()")
			}
			versionDir := filepath.Join(tc.wantDir, "versions", "1")
			payload, err := os.ReadFile(filepath.Join(versionDir, tc.wantPayload))
			if err != nil || string(payload) != "hello" {
				t.Fatalf("the payload is not at the adk-python location with the saved content: %v", err)
			}
			encoded, err := os.ReadFile(filepath.Join(versionDir, "metadata.json"))
			must(t, err, "reading metadata.json")
			var got map[string]any
			must(t, json.Unmarshal(encoded, &got), "parsing metadata.json")
			tc.wantMeta["version"] = 1.0
			tc.wantMeta["canonicalUri"] = fileURI(filepath.Join(versionDir, tc.wantPayload))
			tc.wantMeta["customMetadata"] = map[string]any{"count": "7"}
			// Seconds as a float, kept to the microsecond.
			tc.wantMeta["createTime"] = 1767323045.123456
			if !cmp.Equal(tc.wantMeta, got) {
				t.Errorf("metadata.json has fields %v with unexpected values or missing, among %v", differingKeys(tc.wantMeta, got), slices.Sorted(maps.Keys(got)))
			}
			if runtime.GOOS == "windows" {
				return // Windows has no permission bits to check.
			}
			for path, want := range map[string]fs.FileMode{
				filepath.Join(versionDir, tc.wantPayload):  0o600,
				filepath.Join(versionDir, "metadata.json"): 0o600,
				versionDir:                  0o700,
				tc.wantDir:                  0o700,
				filepath.Join(root, "apps"): 0o700,
			} {
				info, err := os.Stat(path)
				must(t, err, "Stat()")
				if got := info.Mode().Perm(); got != want {
					t.Errorf("mode of %s = %o, want %o", strings.TrimPrefix(path, root), got, want)
				}
			}
		})
	}
}

// differingKeys lists the keys whose values differ between two documents,
// without the values, which are payload.
func differingKeys(want, got map[string]any) []string {
	var keys []string
	for key := range want {
		if !cmp.Equal(want[key], got[key]) {
			keys = append(keys, key)
		}
	}
	for key := range got {
		if _, ok := want[key]; !ok {
			keys = append(keys, key)
		}
	}
	slices.Sort(keys)
	return keys
}

func TestRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name string
		part *genai.Part
		want *genai.Part
		// wantMIME is what GetArtifactVersion reports.
		wantMIME string
	}{
		{
			name:     "text",
			part:     genai.NewPartFromText("line one\r\nline two"),
			want:     genai.NewPartFromText("line one\r\nline two"),
			wantMIME: "text/plain",
		},
		{
			name:     "text that is not UTF-8",
			part:     genai.NewPartFromText("\xff\xfe"),
			want:     genai.NewPartFromText("\xff\xfe"),
			wantMIME: "text/plain",
		},
		{
			name:     "inline data with a display name",
			part:     &genai.Part{InlineData: &genai.Blob{Data: []byte{0xff, 0x00}, MIMEType: "image/png", DisplayName: "Chart (final).png"}},
			want:     &genai.Part{InlineData: &genai.Blob{Data: []byte{0xff, 0x00}, MIMEType: "image/png", DisplayName: "Chart (final).png"}},
			wantMIME: "image/png",
		},
		{
			name:     "inline data typed as text stays inline data",
			part:     genai.NewPartFromBytes([]byte("plain"), "text/plain"),
			want:     genai.NewPartFromBytes([]byte("plain"), "text/plain"),
			wantMIME: "text/plain",
		},
		{
			name:     "inline data without a MIME type",
			part:     &genai.Part{InlineData: &genai.Blob{Data: []byte("raw")}},
			want:     genai.NewPartFromBytes([]byte("raw"), "application/octet-stream"),
			wantMIME: "application/octet-stream",
		},
		{
			name:     "empty inline data",
			part:     &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png"}},
			want:     genai.NewPartFromBytes([]byte{}, "image/png"),
			wantMIME: "image/png",
		},
		{
			name:     "empty octet-stream is not a rewind marker",
			part:     genai.NewPartFromBytes([]byte{}, "application/octet-stream"),
			want:     genai.NewPartFromBytes([]byte{}, "application/octet-stream"),
			wantMIME: "application/octet-stream",
		},
		{
			name:     "inline data wins over text",
			part:     &genai.Part{Text: "ignored", InlineData: &genai.Blob{Data: []byte("kept"), MIMEType: "image/png"}},
			want:     genai.NewPartFromBytes([]byte("kept"), "image/png"),
			wantMIME: "image/png",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := newRoot(t)
			mustSave(t, newService(t, root), "file", tc.part)

			// A second instance shows that nothing is kept in memory.
			service := newService(t, root)
			loaded, err := service.Load(t.Context(), loadRequest("file"))
			must(t, err, "Load()")
			if !cmp.Equal(tc.want, loaded.Part) {
				t.Errorf("Load() returned a part of another kind, MIME type, display name or content than expected")
			}
			version, err := service.GetArtifactVersion(t.Context(), versionRequest("file"))
			must(t, err, "GetArtifactVersion()")
			if got := version.ArtifactVersion.MimeType; got != tc.wantMIME {
				t.Errorf("GetArtifactVersion().MimeType = %q, want %q", got, tc.wantMIME)
			}
		})
	}
}

func TestGetArtifactVersion(t *testing.T) {
	root := newRoot(t)
	createTime := time.Date(2026, time.January, 2, 3, 4, 5, 123456789, time.UTC)
	saveCtx := platform.WithTimeProvider(t.Context(), func() time.Time { return createTime })
	req := saveRequest("reports/q1.txt", genai.NewPartFromText("payload"))
	req.CustomMetadata = map[string]any{"count": 7, "ok": true, "label": "x", "none": nil}
	_, err := newService(t, root).Save(saveCtx, req)
	must(t, err, "Save()")

	// Read at another time through another instance: the stored time is
	// reported, not the time of the call.
	readCtx := platform.WithTimeProvider(t.Context(), func() time.Time { return createTime.Add(time.Hour) })
	resp, err := newService(t, root).GetArtifactVersion(readCtx, versionRequest("reports/q1.txt"))
	must(t, err, "GetArtifactVersion()")
	got := resp.ArtifactVersion
	if want := createTime.Truncate(time.Microsecond); !got.CreateTime.Equal(want) {
		t.Errorf("CreateTime = %v, want %v", got.CreateTime, want)
	}
	want := map[string]any{"count": "7", "ok": "true", "label": "x", "none": "<nil>"}
	if !cmp.Equal(want, got.CustomMetadata) {
		t.Errorf("CustomMetadata keys %v are not the saved values as strings", differingKeys(want, got.CustomMetadata))
	}

	// The URI names the payload file.
	wantPath := filepath.Join(sessionDir(root), "reports", "q1.txt", "versions", "0", "q1.txt")
	if got.CanonicalURI != fileURI(wantPath) {
		t.Errorf("CanonicalURI is not the file URI of the payload under versions/0")
	}
	if payload, err := os.ReadFile(wantPath); err != nil || string(payload) != "payload" {
		t.Errorf("the file CanonicalURI names does not hold the payload: %v", err)
	}
}

func TestNestedNames(t *testing.T) {
	root := newRoot(t)
	service := newService(t, root)
	for _, name := range []string{"doc", "doc/nested", "doc/nested/deeper.txt", "user:doc", "user:shared/notes.md"} {
		mustSave(t, service, name, genai.NewPartFromText(name))
	}
	want := []string{"doc", "doc/nested", "doc/nested/deeper.txt", "user:doc", "user:shared/notes.md"}
	if diff := cmp.Diff(want, mustList(t, service)); diff != "" {
		t.Fatalf("List() mismatch (-want +got):\n%s", diff)
	}

	// A parent artifact and the ones nested below it are independent.
	must(t, service.Delete(t.Context(), deleteRequest("doc", nil)), "Delete()")
	// "shared" is only a directory on the way to an artifact.
	must(t, service.Delete(t.Context(), deleteRequest("user:shared", nil)), "Delete() of a parent path")
	want = []string{"doc/nested", "doc/nested/deeper.txt", "user:doc", "user:shared/notes.md"}
	if diff := cmp.Diff(want, mustList(t, service)); diff != "" {
		t.Fatalf("List() after Delete mismatch (-want +got):\n%s", diff)
	}
	for _, name := range want {
		loaded, err := service.Load(t.Context(), loadRequest(name))
		if err != nil || loaded.Part.Text != name {
			t.Errorf("Load(%q), which was not deleted, did not return what was saved: %v", name, err)
		}
	}
	if _, err := service.Load(t.Context(), loadRequest("doc")); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("Load() of the deleted artifact: err = %v, want fs.ErrNotExist", err)
	}
}

// A name is a key: what looks alike is not the same artifact.
func TestNamesAreNotRewritten(t *testing.T) {
	service := newService(t, newRoot(t))
	names := []string{"pad ded", "padded", "Padded.", "a/ b", "a/b"}
	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" {
		// Their filesystems fold case or drop trailing dots and spaces.
		names = []string{"pad ded", "padded", "a/ b", "a/b"}
	}
	for _, name := range names {
		mustSave(t, service, name, genai.NewPartFromText(name))
	}
	want := slices.Sorted(slices.Values(names))
	if diff := cmp.Diff(want, mustList(t, service)); diff != "" {
		t.Fatalf("List() mismatch (-want +got):\n%s", diff)
	}
	for _, name := range names {
		loaded, err := service.Load(t.Context(), loadRequest(name))
		if err != nil || loaded.Part.Text != name {
			t.Errorf("Load(%q) did not return what was saved under exactly that name: %v", name, err)
		}
	}
}

func TestVersionOrder(t *testing.T) {
	service := newService(t, newRoot(t))
	// Enough versions for their names to sort differently than their numbers.
	const count = 12
	var want []int64
	for i := range count {
		want = append(want, mustSave(t, service, "file", genai.NewPartFromText(fmt.Sprint(i))))
	}
	resp, err := service.Versions(t.Context(), &artifact.VersionsRequest{AppName: app, UserID: user, SessionID: session, FileName: "file"})
	must(t, err, "Versions()")
	if !slices.Equal(resp.Versions, want) || !slices.IsSorted(want) {
		t.Errorf("Versions() = %v, want %v", resp.Versions, want)
	}
	loaded, err := service.Load(t.Context(), loadRequest("file"))
	if err != nil || loaded.Part.Text != fmt.Sprint(count-1) {
		t.Errorf("Load() did not return the last of %d versions: %v", count, err)
	}
	described, err := service.GetArtifactVersion(t.Context(), versionRequest("file"))
	if err != nil || described.ArtifactVersion.Version != count-1 {
		t.Errorf("GetArtifactVersion() did not describe the last of %d versions: %v", count, err)
	}
}

func TestDelete(t *testing.T) {
	t.Run("removes the directories it empties", func(t *testing.T) {
		root := newRoot(t)
		service := newService(t, root)
		mustSave(t, service, "a/b/c.txt", genai.NewPartFromText("one"))
		mustSave(t, service, "a/b/c.txt", genai.NewPartFromText("two"))
		mustSave(t, service, "a/keep.txt", genai.NewPartFromText("keep"))

		must(t, service.Delete(t.Context(), deleteRequest("a/b/c.txt", new(int64(1)))), "Delete() of one version")
		if got := dirEntries(t, filepath.Join(sessionDir(root), "a", "b", "c.txt", "versions")); !slices.Equal(got, []string{"0"}) {
			t.Fatalf("versions after deleting version 1 = %v, want [0]", got)
		}
		must(t, service.Delete(t.Context(), deleteRequest("a/b/c.txt", nil)), "Delete()")
		// "b" held nothing else and goes; "a" still holds an artifact.
		if got := dirEntries(t, filepath.Join(sessionDir(root), "a")); !slices.Equal(got, []string{"keep.txt"}) {
			t.Errorf("directory above the deleted artifact holds %v, want [keep.txt]", got)
		}
		must(t, service.Delete(t.Context(), deleteRequest("a/keep.txt", new(int64(0)))), "Delete() of the only version")
		// The scope's own directory stays.
		if got := dirEntries(t, sessionDir(root)); len(got) != 0 {
			t.Errorf("artifacts directory holds %v after deleting everything, want nothing", got)
		}
	})

	t.Run("keeps a save in progress", func(t *testing.T) {
		root := newRoot(t)
		service := newService(t, root)
		mustSave(t, service, "file", genai.NewPartFromText("zero"))
		pending := filepath.Join(sessionDir(root), "file", "versions", ".1.pending")
		must(t, os.Mkdir(pending, 0o700), "Mkdir()")
		must(t, service.Delete(t.Context(), deleteRequest("file", nil)), "Delete()")
		if got := dirEntries(t, filepath.Dir(pending)); !slices.Equal(got, []string{".1.pending"}) {
			t.Fatalf("versions after Delete = %v, want only the staging directory of the other save", got)
		}
		// The number of the deleted version is free again, the reserved one
		// is still taken.
		for _, want := range []int64{0, 2} {
			if got := mustSave(t, service, "file", genai.NewPartFromText("next")); got != want {
				t.Errorf("Save() after Delete = version %d, want %d", got, want)
			}
		}
	})

	t.Run("finishes what an earlier Delete left", func(t *testing.T) {
		root := newRoot(t)
		service := newService(t, root)
		mustSave(t, service, "file", genai.NewPartFromText("zero"))
		mustSave(t, service, "file", genai.NewPartFromText("one"))
		// A version a Delete had taken out of service when it died.
		versions := filepath.Join(sessionDir(root), "file", "versions")
		must(t, os.Rename(filepath.Join(versions, "1"), filepath.Join(versions, ".1.leftover.deleted")), "Rename()")

		loaded, err := service.Load(t.Context(), loadRequest("file"))
		if err != nil || loaded.Part.Text != "zero" {
			t.Errorf("Load() did not return the version that was not deleted: %v", err)
		}
		// Things that merely end the same way.
		must(t, os.Mkdir(filepath.Join(versions, "kept.deleted"), 0o700), "Mkdir()")
		writeFile(t, filepath.Join(versions, ".file.deleted"), nil)
		must(t, service.Delete(t.Context(), deleteRequest("file", new(int64(7)))), "Delete() of another version")
		if got := dirEntries(t, versions); !slices.Equal(got, []string{".file.deleted", "0", "kept.deleted"}) {
			t.Errorf("versions = %v, want only the version taken out of service removed", got)
		}
	})

	t.Run("what does not exist", func(t *testing.T) {
		root := newRoot(t)
		service := newService(t, root)
		mustSave(t, service, "file", genai.NewPartFromText("zero"))
		// Empty directories on the way to an artifact that is not there go
		// as well.
		must(t, os.MkdirAll(filepath.Join(sessionDir(root), "left", "over"), 0o700), "MkdirAll()")
		must(t, service.Delete(t.Context(), deleteRequest("left/over/never-saved", nil)), "Delete()")
		if got := dirEntries(t, sessionDir(root)); !slices.Equal(got, []string{"file"}) {
			t.Errorf("artifacts directory holds %v, want [file]", got)
		}
		for _, req := range []*artifact.DeleteRequest{
			deleteRequest("never-saved", nil),
			deleteRequest("never-saved", new(int64(0))),
			deleteRequest("file", new(int64(5))),
			deleteRequest("file", new(int64(-1))),
		} {
			if err := service.Delete(t.Context(), req); err != nil {
				t.Errorf("Delete() of a missing version failed: %v", err)
			}
		}
		if _, err := service.Load(t.Context(), loadRequest("file")); err != nil {
			t.Errorf("Load() after deleting other versions failed: %v", err)
		}
	})

	t.Run("latest version frees its number", func(t *testing.T) {
		service := newService(t, newRoot(t))
		mustSave(t, service, "file", genai.NewPartFromText("zero"))
		mustSave(t, service, "file", genai.NewPartFromText("one"))
		must(t, service.Delete(t.Context(), deleteRequest("file", new(int64(1)))), "Delete()")
		loaded, err := service.Load(t.Context(), loadRequest("file"))
		if err != nil || loaded.Part.Text != "zero" {
			t.Fatalf("Load() after deleting the latest version did not return the previous one: %v", err)
		}
		if got := mustSave(t, service, "file", genai.NewPartFromText("again")); got != 1 {
			t.Errorf("Save() after deleting version 1 = version %d, want 1", got)
		}
	})
}

// call runs every operation that takes the four identifiers and returns the
// errors by operation.
func call(ctx context.Context, service artifact.Service, appName, userID, sessionID, fileName string) map[string]error {
	errs := map[string]error{}
	_, errs["Save"] = service.Save(ctx, &artifact.SaveRequest{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Part: genai.NewPartFromText("x")})
	_, errs["Load"] = service.Load(ctx, &artifact.LoadRequest{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName})
	_, errs["Versions"] = service.Versions(ctx, &artifact.VersionsRequest{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName})
	_, errs["GetArtifactVersion"] = service.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName})
	errs["Delete"] = service.Delete(ctx, &artifact.DeleteRequest{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName})
	return errs
}

func TestInvalidIdentifiers(t *testing.T) {
	invalidIDs := map[string]string{
		"empty":                       "",
		"spaces only":                 "   ",
		"white space only":            "\t\n",
		"information separator only":  "\x1c\x1d\x1e\x1f",
		"no-break space only":         "\u00a0",
		"ideographic space only":      "a/\u3000",
		"parent":                      "..",
		"current":                     ".",
		"dots only":                   "...",
		"dots and spaces only":        ". .",
		"traversal":                   "a/../b",
		"backslash traversal":         `a\..\b`,
		"mixed traversal":             `./..\`,
		"rooted":                      "/abs",
		"rooted with backslash":       `\abs`,
		"trailing separator":          "a/",
		"empty segment":               "a//b",
		"blank segment":               "  /a",
		"drive":                       "C:data",
		"drive path":                  `c:\data`,
		"null byte":                   "a\x00b",
		"invalid UTF-8":               "a\xffb",
		"segment too long":            strings.Repeat("x", 256),
		"backslash segment too long":  strings.Repeat("a", 200) + `\` + strings.Repeat("b", 200),
		"too long":                    strings.Repeat(strings.Repeat("x", 255)+"/", 4) + "x",
		"too many segments":           strings.Repeat("a/", 64) + "a",
		"too many backslash segments": strings.Repeat(`a\`, 64) + "a",
		"apps directory":              "x/apps",
		"layout directory first":      "apps/x",
		"users directory":             `x\users`,
		"sessions directory":          "victim/sessions/s1",
		"artifacts directory":         "x/Artifacts",
		"versions directory":          "x/versions.",
	}
	for name, id := range invalidIDs {
		for _, field := range []string{"AppName", "UserID", "SessionID"} {
			t.Run(field+" "+name, func(t *testing.T) {
				root := newRoot(t)
				service := newService(t, root)
				ids := map[string]string{"AppName": app, "UserID": user, "SessionID": session}
				ids[field] = id
				errs := call(t.Context(), service, ids["AppName"], ids["UserID"], ids["SessionID"], "file")
				_, errs["List"] = service.List(t.Context(), &artifact.ListRequest{AppName: ids["AppName"], UserID: ids["UserID"], SessionID: ids["SessionID"]})
				// A user-scoped filename does not use the session, which is
				// validated all the same.
				_, errs["Save user-scoped"] = service.Save(t.Context(), &artifact.SaveRequest{AppName: ids["AppName"], UserID: ids["UserID"], SessionID: ids["SessionID"], FileName: "user:file", Part: genai.NewPartFromText("x")})
				// The message for another bad value of the same field: the
				// two are the same when neither repeats the value.
				ids[field] = "a//b"
				other := call(t.Context(), service, ids["AppName"], ids["UserID"], ids["SessionID"], "file")
				for op, err := range errs {
					if !errors.Is(err, fs.ErrInvalid) {
						t.Errorf("%s: err = %v, want fs.ErrInvalid", op, err)
					}
					if want, ok := other[op]; ok && err != nil && id != "" && err.Error() != want.Error() {
						t.Errorf("%s: the error depends on the identifier, so it gives part of it away", op)
					}
				}
				if got := dirEntries(t, root); len(got) != 0 {
					t.Errorf("rejected requests created %v", got)
				}
			})
		}
	}
}

func TestInvalidFileNames(t *testing.T) {
	for name, fileName := range map[string]string{
		"empty":                 "",
		"prefix only":           "user:",
		"spaces only":           "   ",
		"blank segment":         "a/ /b",
		"current":               ".",
		"parent":                "..",
		"dots only":             "a/...",
		"traversal":             "folder/../alias.txt",
		"user-scoped traversal": "user:../alias.txt",
		"rooted":                "/abs.txt",
		"trailing separator":    "dir/",
		"empty segment":         "a//b",
		"backslash":             `dir\file.txt`,
		"drive":                 "C:file.txt",
		"user-scoped drive":     "user:c:file.txt",
		"null byte":             "a\x00b",
		"invalid UTF-8":         "a\xffb",
		"segment too long":      "dir/" + strings.Repeat("x", 256),
		"too long":              strings.Repeat(strings.Repeat("x", 255)+"/", 4) + "x",
		"user-scoped too long":  "user:" + strings.Repeat(strings.Repeat("x", 255)+"/", 4) + "x",
		"too many segments":     strings.Repeat("a/", 64) + "a",
		// White space that adk-python would remove.
		"leading space":             " notes.txt",
		"trailing space":            "dir/notes.txt ",
		"trailing newline":          "notes.txt\n",
		"leading no-break space":    "\u00a0notes.txt",
		"information separator":     "notes.txt\x1f",
		"user-scoped leading space": "user: notes.txt",
		// The layout's own names.
		"metadata":                 "metadata.json",
		"metadata in another case": "MetaData.Json",
		"metadata with a dot":      "metadata.json.",
		"nested metadata":          "nested/metadata.json",
		"user-scoped metadata":     "user:metadata.json",
		"versions":                 "versions",
		"user-scoped versions":     "user:versions",
		"versions segment":         "docs/versions/v2.md",
		"versions in another case": "docs/VeRsIoNs/v2.md",
		"versions with a dot":      "docs/versions./v2.md",
		"versions with a space":    "docs/versions /v2.md",
		"versions as last segment": "nested/report.txt/versions",
	} {
		t.Run(name, func(t *testing.T) {
			root := newRoot(t)
			service := newService(t, root)
			// The message for another bad filename: the two are the same
			// when neither repeats the name.
			other := call(t.Context(), service, app, user, session, "a//b")
			for op, err := range call(t.Context(), service, app, user, session, fileName) {
				if !errors.Is(err, fs.ErrInvalid) {
					t.Errorf("%s: err = %v, want fs.ErrInvalid", op, err)
				}
				if err != nil && fileName != "" && err.Error() != other[op].Error() {
					t.Errorf("%s: the error depends on the filename, so it gives part of it away", op)
				}
			}
			if got := dirEntries(t, root); len(got) != 0 {
				t.Errorf("rejected requests created %v", got)
			}
		})
	}
}

// The longest names the rules allow are stored.
func TestLongestNames(t *testing.T) {
	service := newService(t, newRoot(t))
	longest := strings.Repeat("x", 255)
	// 64 segments in 1024 bytes.
	deepest := strings.Repeat("abcdefghijklmno/", 63) + "abcdefghijklmnop"
	for _, tc := range []struct{ name, id, fileName string }{
		{"segment", longest, longest},
		{"path", deepest, deepest},
		// The prefix is not part of the path.
		{"user-scoped path", deepest, "user:" + deepest},
	} {
		t.Run(tc.name, func(t *testing.T) {
			errs := call(t.Context(), service, tc.id, tc.id, tc.id, tc.fileName)
			for op, err := range errs {
				if err != nil {
					t.Errorf("%s with the longest allowed %s failed: %v", op, tc.name, err)
				}
			}
		})
	}
}

func TestInvalidSaveRequests(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*artifact.SaveRequest)
	}{
		{"no part", func(req *artifact.SaveRequest) { req.Part = nil }},
		{"empty part", func(req *artifact.SaveRequest) { req.Part = &genai.Part{} }},
		{"empty text", func(req *artifact.SaveRequest) { req.Part = genai.NewPartFromText("") }},
		{"file data", func(req *artifact.SaveRequest) { req.Part = genai.NewPartFromURI("gs://bucket/object", "image/png") }},
		{"MIME type is not UTF-8", func(req *artifact.SaveRequest) { req.Part = genai.NewPartFromBytes([]byte("x"), "image/\xff") }},
		{"display name is not UTF-8", func(req *artifact.SaveRequest) {
			req.Part = &genai.Part{InlineData: &genai.Blob{Data: []byte("x"), MIMEType: "image/png", DisplayName: "\xff"}}
		}},
		{"custom metadata key is not UTF-8", func(req *artifact.SaveRequest) { req.CustomMetadata = map[string]any{"\xff": "v"} }},
		{"custom metadata value is not UTF-8", func(req *artifact.SaveRequest) { req.CustomMetadata = map[string]any{"k": "\xff"} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := newRoot(t)
			req := saveRequest("file", genai.NewPartFromText("x"))
			tc.mutate(req)
			if _, err := newService(t, root).Save(t.Context(), req); !errors.Is(err, fs.ErrInvalid) {
				t.Errorf("Save() err = %v, want fs.ErrInvalid", err)
			}
			if got := dirEntries(t, root); len(got) != 0 {
				t.Errorf("rejected Save created %v", got)
			}
		})
	}
}

// One error names every required field that is missing, as the errors of the
// other artifact services do.
func TestMissingFields(t *testing.T) {
	service := newService(t, newRoot(t))
	errs := call(t.Context(), service, "", "", "", "")
	_, errs["List"] = service.List(t.Context(), &artifact.ListRequest{})
	for op, err := range errs {
		want := "missing required fields: AppName, UserID, SessionID, FileName"
		if op == "List" {
			want = "missing required fields: AppName, UserID, SessionID"
		}
		if err == nil || !strings.HasSuffix(err.Error(), want) {
			t.Errorf("%s with no field set: err = %v, want one ending in %q", op, err, want)
		}
	}
}

func TestNilRequests(t *testing.T) {
	service := newService(t, newRoot(t))
	errs := map[string]error{}
	_, errs["Save"] = service.Save(t.Context(), nil)
	_, errs["Load"] = service.Load(t.Context(), nil)
	_, errs["List"] = service.List(t.Context(), nil)
	_, errs["Versions"] = service.Versions(t.Context(), nil)
	_, errs["GetArtifactVersion"] = service.GetArtifactVersion(t.Context(), nil)
	errs["Delete"] = service.Delete(t.Context(), nil)
	for op, err := range errs {
		if !errors.Is(err, fs.ErrInvalid) {
			t.Errorf("%s(nil): err = %v, want fs.ErrInvalid", op, err)
		}
	}
}

func TestAcceptedIdentifiers(t *testing.T) {
	cases := map[string][3]string{
		"several segments":          {"org/app", "group/user123", "ns/sess0"},
		"backslashes":               {"app", `back\user`, `back\slash`},
		"layout names on their own": {"versions", "artifacts", "sessions"},
		"session named user":        {"app", "u", "user"},
		"spaces and punctuation":    {"my app", "alice@example.com", "'user' should be used instead"},
		"dots beside other text":    {"..app", "u...", "a..b"},
	}
	if runtime.GOOS != "windows" {
		// A colon is an ordinary character except after a drive letter.
		cases["colon"] = [3]string{"1:app", "ab:cd", "sess:"}
	}
	for name, ids := range cases {
		t.Run(name, func(t *testing.T) {
			service := newService(t, newRoot(t))
			for _, fileName := range []string{"file.txt", "user:file.txt"} {
				errs := call(t.Context(), service, ids[0], ids[1], ids[2], fileName)
				for op, err := range errs {
					if err != nil {
						t.Errorf("%s(%s): %v", op, fileName, err)
					}
				}
			}
		})
	}
}

func TestIdentifiersAreIsolated(t *testing.T) {
	service := newService(t, newRoot(t))
	for _, name := range []string{"private.txt", "user:shared.txt"} {
		mustSave(t, service, name, genai.NewPartFromText("secret"))
	}
	for _, tc := range []struct {
		name               string
		app, user, session string
		wantShared         bool
	}{
		{"another app", "app2", user, session, false},
		{"another user", app, "user2", session, false},
		{"another session", app, user, "session2", true},
		{"a longer app name", app + "/x", user, session, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := service.Load(t.Context(), &artifact.LoadRequest{AppName: tc.app, UserID: tc.user, SessionID: tc.session, FileName: "private.txt"}); !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("Load() of another scope's private artifact: err = %v, want fs.ErrNotExist", err)
			}
			_, err := service.Load(t.Context(), &artifact.LoadRequest{AppName: tc.app, UserID: tc.user, SessionID: tc.session, FileName: "user:shared.txt"})
			if shared := err == nil; shared != tc.wantShared {
				t.Errorf("Load() of the user-scoped artifact: err = %v, want shared = %v", err, tc.wantShared)
			}
		})
	}
}

// Names that only resemble the layout's own are ordinary: saved, listed and
// loaded like any other.
func TestAcceptedFileNames(t *testing.T) {
	service := newService(t, newRoot(t))
	names := []string{
		"metadata.json/report.txt", "a/Metadata.JSON /b", "user:metadata.json./c", "dir/ metadata.json",
		"versions.txt", "nested/versions.txt", "my-metadata.json", "a..b", ".hidden", "user:user:x",
	}
	if runtime.GOOS == "windows" {
		// No colon in a file name there, and no trailing dot or space.
		names = []string{"metadata.json/report.txt", "dir/ metadata.json", "versions.txt", "nested/versions.txt", "my-metadata.json", "a..b", ".hidden"}
	}
	for _, name := range names {
		mustSave(t, service, name, genai.NewPartFromText(name))
	}
	if diff := cmp.Diff(slices.Sorted(slices.Values(names)), mustList(t, service)); diff != "" {
		t.Errorf("List() mismatch (-want +got):\n%s", diff)
	}
	for _, name := range names {
		for op, err := range call(t.Context(), service, app, user, session, name) {
			if err != nil {
				t.Errorf("%s(%q) failed: %v", op, name, err)
			}
		}
	}
}

// The version directories of an artifact cannot be addressed as artifacts of
// their own, whatever is in them.
func TestNamesCannotReachVersions(t *testing.T) {
	root := newRoot(t)
	service := newService(t, root)
	mustSave(t, service, "file", genai.NewPartFromText("x"))
	versions := filepath.Join(sessionDir(root), "file", "versions")
	// The reservation of a save in progress, and a version with nothing in it.
	must(t, os.Mkdir(filepath.Join(versions, ".1.pending"), 0o700), "Mkdir()")
	must(t, os.Mkdir(filepath.Join(versions, "5"), 0o700), "Mkdir()")
	for _, name := range []string{"file/versions/0/file", "file/versions/0/metadata.json", "file/versions/.1.pending", "file/versions/5", "file/versions"} {
		for op, err := range call(t.Context(), service, app, user, session, name) {
			if !errors.Is(err, fs.ErrInvalid) {
				t.Errorf("%s(%q): err = %v, want fs.ErrInvalid", op, name, err)
			}
		}
	}
	if got := dirEntries(t, versions); !slices.Equal(got, []string{".1.pending", "0", "5"}) {
		t.Errorf("versions = %v, want them untouched", got)
	}
}

func TestDamagedMetadata(t *testing.T) {
	for _, tc := range []struct {
		name     string
		metadata string
	}{
		{"not UTF-8", "\xff"},
		{"not UTF-8 inside a string", "{\"fileName\":\"report.txt\",\"mimeType\":\"image/\xff\",\"version\":0,\"canonicalUri\":\"x\",\"customMetadata\":{},\"createTime\":1}"},
		{"not JSON", "{"},
		{"empty", ""},
		{"null", "null"},
		{"not an object", "[]"},
		{"wrong type", `{"fileName":"report.txt","mimeType":7,"createTime":1}`},
		{"custom metadata is not an object", `{"fileName":"report.txt","mimeType":"image/png","customMetadata":[],"createTime":1}`},
		{"no create time", `{"fileName":"report.txt","mimeType":"image/png"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := newRoot(t)
			service := newService(t, root)
			mustSave(t, service, "report.txt", genai.NewPartFromText("report"))
			mustSave(t, service, "user:report.txt", genai.NewPartFromText("report"))
			mustSave(t, service, "blob.bin", genai.NewPartFromBytes([]byte{0xff}, "image/png"))
			mustSave(t, service, "healthy.txt", genai.NewPartFromText("healthy"))
			blobVersion := filepath.Join(sessionDir(root), "blob.bin", "versions", "0")
			textVersion := filepath.Join(sessionDir(root), "report.txt", "versions", "0")
			for _, dir := range []string{textVersion, blobVersion, filepath.Join(userDir(root), "report.txt", "versions", "0")} {
				writeFile(t, filepath.Join(dir, "metadata.json"), []byte(tc.metadata))
			}

			// Listing does not depend on metadata.
			want := []string{"blob.bin", "healthy.txt", "report.txt", "user:report.txt"}
			if diff := cmp.Diff(want, mustList(t, service)); diff != "" {
				t.Errorf("List() mismatch (-want +got):\n%s", diff)
			}
			versions, err := service.Versions(t.Context(), &artifact.VersionsRequest{AppName: app, UserID: user, SessionID: session, FileName: "blob.bin"})
			if err != nil || !slices.Equal(versions.Versions, []int64{0}) {
				t.Errorf("Versions() = (%v, %v), want [0]", versions, err)
			}

			// A payload that is text still loads as text.
			loaded, err := service.Load(t.Context(), loadRequest("report.txt"))
			if err != nil || loaded.Part.Text != "report" {
				t.Errorf("Load() of a text payload did not return it as text: %v", err)
			}
			// One that cannot be text is reported as missing.
			if _, err := service.Load(t.Context(), loadRequest("blob.bin")); !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("Load() of a binary payload: err = %v, want fs.ErrNotExist", err)
			}

			// The version is described from what is left.
			payload := filepath.Join(blobVersion, "blob.bin")
			modTime := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
			must(t, os.Chtimes(payload, modTime, modTime), "Chtimes()")
			resp, err := service.GetArtifactVersion(t.Context(), versionRequest("blob.bin"))
			must(t, err, "GetArtifactVersion()")
			got := resp.ArtifactVersion
			if got.Version != 0 || got.MimeType != "" || len(got.CustomMetadata) != 0 || got.CustomMetadata == nil {
				t.Errorf("GetArtifactVersion() = version %d, MIME type %q, %d custom metadata entries (nil: %v); want 0, none, an empty map", got.Version, got.MimeType, len(got.CustomMetadata), got.CustomMetadata == nil)
			}
			if !got.CreateTime.Equal(modTime) {
				t.Errorf("CreateTime = %v, want the payload's modification time %v", got.CreateTime, modTime)
			}
			if got.CanonicalURI != fileURI(payload) {
				t.Error("CanonicalURI does not name the payload")
			}

			if runtime.GOOS != "windows" {
				// A link in place of the payload is not described either.
				must(t, os.Remove(payload), "Remove()")
				must(t, os.Symlink("metadata.json", payload), "Symlink()")
				if _, err := service.GetArtifactVersion(t.Context(), versionRequest("blob.bin")); !refused(err) {
					t.Errorf("GetArtifactVersion() of a link: err = %v, want a refusal", err)
				}
			}
			// With the payload gone too there is no version left to describe.
			must(t, os.Remove(payload), "Remove()")
			if _, err := service.GetArtifactVersion(t.Context(), versionRequest("blob.bin")); !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("GetArtifactVersion() without a payload: err = %v, want fs.ErrNotExist", err)
			}
		})
	}
}

// A version directory missing one of its two files is what a reader sees
// while Delete is removing it.
func TestMissingVersionFiles(t *testing.T) {
	for _, tc := range []struct {
		file        string
		wantVersion bool // Whether GetArtifactVersion still answers.
	}{
		{"metadata.json", false},
		{"report.txt", true},
	} {
		t.Run(tc.file, func(t *testing.T) {
			root := newRoot(t)
			service := newService(t, root)
			// Text, which is what a version without metadata would be taken for.
			mustSave(t, service, "report.txt", genai.NewPartFromText("report"))
			must(t, os.Remove(filepath.Join(sessionDir(root), "report.txt", "versions", "0", tc.file)), "Remove()")
			if _, err := service.Load(t.Context(), loadRequest("report.txt")); !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("Load() err = %v, want fs.ErrNotExist", err)
			}
			_, err := service.GetArtifactVersion(t.Context(), versionRequest("report.txt"))
			if tc.wantVersion && err != nil {
				t.Errorf("GetArtifactVersion() failed: %v", err)
			}
			if !tc.wantVersion && !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("GetArtifactVersion() err = %v, want fs.ErrNotExist", err)
			}
		})
	}
}

// A directory in adk-python's format is read as is.
func TestReadsPythonDirectory(t *testing.T) {
	root := newRoot(t)
	write := func(dir, payload, data, metadata string) {
		t.Helper()
		must(t, os.MkdirAll(dir, 0o755), "MkdirAll()")
		writeFile(t, filepath.Join(dir, payload), []byte(data))
		writeFile(t, filepath.Join(dir, "metadata.json"), []byte(metadata))
	}
	// A text artifact has no mimeType, and custom metadata keeps JSON types.
	write(filepath.Join(sessionDir(root), "docs", "notes.txt", "versions", "0"), "notes.txt", "from python",
		`{"version":0,"canonicalUri":"file:///elsewhere/notes.txt","customMetadata":{"count":7,"ok":true,"tags":["a","b"],"label":"x","quoted":"\"q\"","none":null},"createTime":1767323045.1234567,"fileName":"docs/notes.txt"}`)
	// Saved with session_id=None, so the name on record has no "user:".
	write(filepath.Join(userDir(root), "avatar.png", "versions", "3"), "avatar.png", "\xff\xd8",
		`{"version":3,"canonicalUri":"file:///elsewhere/avatar.png","customMetadata":{},"createTime":100,"mimeType":"image/png","fileName":"avatar.png","displayName":"Me"}`)
	// adk-python stores names this service does not accept. They are left
	// out of the listing rather than listed and then refused.
	write(filepath.Join(sessionDir(root), "a", " ", "b", "versions", "0"), "b", "x", `{"createTime":1}`)
	write(filepath.Join(sessionDir(root), "...", "versions", "0"), "...", "x", `{"createTime":1}`)
	write(filepath.Join(sessionDir(root), "versions.d", "metadata.json", "versions", "0"), "metadata.json", "x", `{"createTime":1}`)
	if runtime.GOOS != "windows" {
		write(filepath.Join(sessionDir(root), `back\slash`, "versions", "0"), `back\slash`, "x", `{"createTime":1}`)
		// In a session this would read as a user-scoped name.
		write(filepath.Join(sessionDir(root), "user:x", "versions", "0"), "user:x", "x", `{"createTime":1}`)
	}
	service := newService(t, root)

	if diff := cmp.Diff([]string{"docs/notes.txt", "user:avatar.png"}, mustList(t, service)); diff != "" {
		t.Errorf("List() mismatch (-want +got):\n%s", diff)
	}
	text, err := service.Load(t.Context(), loadRequest("docs/notes.txt"))
	if err != nil || text.Part.Text != "from python" {
		t.Errorf("Load() of a text artifact did not return its text: %v", err)
	}
	blob, err := service.Load(t.Context(), loadRequest("user:avatar.png"))
	must(t, err, "Load() of an inline-data artifact")
	wantBlob := &genai.Part{InlineData: &genai.Blob{Data: []byte("\xff\xd8"), MIMEType: "image/png", DisplayName: "Me"}}
	if !cmp.Equal(wantBlob, blob.Part) {
		t.Error("Load() of an inline-data artifact returned another kind, MIME type, display name or content than stored")
	}

	resp, err := service.GetArtifactVersion(t.Context(), versionRequest("docs/notes.txt"))
	must(t, err, "GetArtifactVersion()")
	got := resp.ArtifactVersion
	// A string is the string, anything else its JSON text.
	wantMetadata := map[string]any{"count": "7", "ok": "true", "tags": `["a","b"]`, "label": "x", "quoted": `"q"`, "none": "null"}
	if !cmp.Equal(wantMetadata, got.CustomMetadata) {
		t.Errorf("CustomMetadata keys %v are not the stored values as strings", differingKeys(wantMetadata, got.CustomMetadata))
	}
	if want := time.Date(2026, time.January, 2, 3, 4, 5, 123457000, time.UTC); !got.CreateTime.Equal(want) {
		t.Errorf("CreateTime = %v, want %v", got.CreateTime, want)
	}
	// The URI on record is not trusted.
	if strings.Contains(got.CanonicalURI, "elsewhere") {
		t.Error("CanonicalURI was taken from metadata.json")
	}

	// The next version continues after the ones already there.
	if got := mustSave(t, service, "user:avatar.png", genai.NewPartFromBytes([]byte("new"), "image/png")); got != 4 {
		t.Errorf("Save() = version %d, want 4", got)
	}
}

func TestSymlinksAreNotFollowed(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("creating symbolic links needs a privilege on Windows")
	}
	artifactDir := filepath.Join("apps", app, "users", user, "sessions", session, "artifacts", "file")

	// moveBehindLink moves the directory at rel, a path below root, to
	// another place and leaves a link to it, as someone would to redirect the
	// service. The link is relative when the new place is inside the root,
	// which os.Root alone would follow. It returns the new place.
	moveBehindLink := func(t *testing.T, root, rel string, outside bool) string {
		t.Helper()
		link := filepath.Join(root, rel)
		target := filepath.Join(filepath.Dir(link), "elsewhere")
		linkTo := "elsewhere"
		if outside {
			target = filepath.Join(newRoot(t), "elsewhere")
			linkTo = target
		}
		must(t, os.Rename(link, target), "Rename()")
		must(t, os.Symlink(linkTo, link), "Symlink()")
		return target
	}

	t.Run("directories", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			// link is the directory replaced by a link, below the root.
			link string
			// listed says that List still works, the link being inside the
			// scope it walks rather than on the way to it.
			listed bool
		}{
			{"apps", "apps", false},
			{"app", filepath.Join("apps", app), false},
			{"user", filepath.Join("apps", app, "users", user), false},
			{"session", filepath.Join("apps", app, "users", user, "sessions", session), false},
			{"scope", filepath.Dir(artifactDir), false},
			{"artifact", artifactDir, true},
			{"versions", filepath.Join(artifactDir, "versions"), true},
		} {
			for _, outside := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s outside=%v", tc.name, outside), func(t *testing.T) {
					root := newRoot(t)
					service := newService(t, root)
					mustSave(t, service, "file", genai.NewPartFromText("x"))
					target := moveBehindLink(t, root, tc.link, outside)

					errs := call(t.Context(), service, app, user, session, "file")
					for op, err := range errs {
						if !refused(err) {
							t.Errorf("%s through a link: err = %v, want a refusal", op, err)
						}
					}
					list, err := service.List(t.Context(), &artifact.ListRequest{AppName: app, UserID: user, SessionID: session})
					if tc.listed {
						if err != nil || slices.Contains(list.FileNames, "file") {
							t.Errorf("List() = (%v, %v), want no artifact behind a link", list, err)
						}
					} else if !refused(err) {
						t.Errorf("List() through a link: err = %v, want a refusal", err)
					}
					// Nothing was added or removed behind the link, which is
					// still in place.
					rest, err := filepath.Rel(tc.link, filepath.Join(artifactDir, "versions"))
					must(t, err, "Rel()")
					if got := dirEntries(t, filepath.Join(target, rest)); !slices.Equal(got, []string{"0"}) {
						t.Errorf("versions behind the link = %v, want [0]", got)
					}
					if info, err := os.Lstat(filepath.Join(root, tc.link)); err != nil || info.Mode()&fs.ModeSymlink == 0 {
						t.Errorf("the link was removed or replaced: %v", err)
					}
				})
			}
		}
	})

	// A link from one user's directory to another's must not share storage.
	t.Run("another tenant", func(t *testing.T) {
		root := newRoot(t)
		service := newService(t, root)
		mustSave(t, service, "file", genai.NewPartFromText("x"))
		must(t, os.Symlink(user, filepath.Join(root, "apps", app, "users", "mallory")), "Symlink()")
		for op, err := range call(t.Context(), service, app, "mallory", session, "file") {
			if !refused(err) {
				t.Errorf("%s as the user whose directory is a link: err = %v, want a refusal", op, err)
			}
		}
		if got := dirEntries(t, filepath.Join(root, artifactDir, "versions")); !slices.Equal(got, []string{"0"}) {
			t.Errorf("versions of the other user = %v, want [0]", got)
		}
	})

	// A link in place of a version directory is not a version: it is neither
	// read nor counted, and its number is not reused.
	t.Run("version", func(t *testing.T) {
		for _, outside := range []bool{false, true} {
			t.Run(fmt.Sprintf("outside=%v", outside), func(t *testing.T) {
				root := newRoot(t)
				service := newService(t, root)
				mustSave(t, service, "file", genai.NewPartFromText("x"))
				target := moveBehindLink(t, root, filepath.Join(artifactDir, "versions", "0"), outside)

				load := loadRequest("file")
				describe := versionRequest("file")
				for _, version := range []*int64{nil, new(int64(0))} {
					load.Version, describe.Version = version, version
					if _, err := service.Load(t.Context(), load); !errors.Is(err, fs.ErrNotExist) {
						t.Errorf("Load() through a link: err = %v, want fs.ErrNotExist", err)
					}
					if _, err := service.GetArtifactVersion(t.Context(), describe); !errors.Is(err, fs.ErrNotExist) {
						t.Errorf("GetArtifactVersion() through a link: err = %v, want fs.ErrNotExist", err)
					}
				}
				if got := mustList(t, service); len(got) != 0 {
					t.Errorf("List() = %v, want no artifact behind a link", got)
				}
				must(t, service.Delete(t.Context(), deleteRequest("file", nil)), "Delete()")
				if got := mustSave(t, service, "file", genai.NewPartFromText("y")); got != 1 {
					t.Errorf("Save() = version %d, want 1", got)
				}
				if got := dirEntries(t, target); !slices.Equal(got, []string{"file", "metadata.json"}) {
					t.Errorf("directory behind the link = %v, want it untouched", got)
				}
			})
		}
	})

	t.Run("files", func(t *testing.T) {
		for _, file := range []string{"file", "metadata.json"} {
			for _, tc := range []struct {
				name string
				// target creates what the link points to and returns it.
				target func(t *testing.T, versionDir string) string
			}{
				{"outside the root", func(t *testing.T, _ string) string {
					outside := filepath.Join(t.TempDir(), "secret")
					writeFile(t, outside, []byte(`{"createTime":1}`))
					return outside
				}},
				{"beside it", func(t *testing.T, versionDir string) string {
					writeFile(t, filepath.Join(versionDir, "other"), []byte(`{"createTime":1}`))
					return "other"
				}},
			} {
				t.Run(file+" "+tc.name, func(t *testing.T) {
					root := newRoot(t)
					service := newService(t, root)
					mustSave(t, service, "file", genai.NewPartFromText("x"))
					versionDir := filepath.Join(sessionDir(root), "file", "versions", "0")
					must(t, os.Remove(filepath.Join(versionDir, file)), "Remove()")
					must(t, os.Symlink(tc.target(t, versionDir), filepath.Join(versionDir, file)), "Symlink()")

					if _, err := service.Load(t.Context(), loadRequest("file")); !refused(err) {
						t.Errorf("Load() through a link: err = %v, want a refusal", err)
					}
					// Describing a version reads its metadata only.
					_, err := service.GetArtifactVersion(t.Context(), versionRequest("file"))
					if file == "metadata.json" && !refused(err) {
						t.Errorf("GetArtifactVersion() through a link: err = %v, want a refusal", err)
					}
					if file == "file" && err != nil {
						t.Errorf("GetArtifactVersion() failed: %v", err)
					}
				})
			}
		}
	})
}

func TestIncompleteVersions(t *testing.T) {
	root := newRoot(t)
	service := newService(t, root)
	mustSave(t, service, "file", genai.NewPartFromText("zero"))
	versions := filepath.Join(sessionDir(root), "file", "versions")
	// Left by saves that died, and things that are not versions at all.
	strays := []string{".1.pending", ".2.pending", "01", "+3", "-1", "9x", "tmp"}
	for _, dir := range strays {
		// With the files of a version, which still does not make it one.
		must(t, os.CopyFS(filepath.Join(versions, dir), os.DirFS(filepath.Join(versions, "0"))), "CopyFS()")
	}
	writeFile(t, filepath.Join(versions, "7"), nil)
	want := append(dirEntries(t, versions), "3")
	slices.Sort(want)

	resp, err := service.Versions(t.Context(), &artifact.VersionsRequest{AppName: app, UserID: user, SessionID: session, FileName: "file"})
	if err != nil || !slices.Equal(resp.Versions, []int64{0}) {
		t.Fatalf("Versions() = (%v, %v), want [0]", resp, err)
	}
	loaded, err := service.Load(t.Context(), loadRequest("file"))
	if err != nil || loaded.Part.Text != "zero" {
		t.Fatalf("Load() did not return the only published version: %v", err)
	}
	for _, version := range []int64{-1, 1, 7} {
		req := loadRequest("file")
		req.Version = &version
		if _, err := service.Load(t.Context(), req); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Load() of version %d, which was never published: err = %v, want fs.ErrNotExist", version, err)
		}
		// Deleting such a version deletes nothing.
		if err := service.Delete(t.Context(), deleteRequest("file", &version)); err != nil {
			t.Errorf("Delete() of version %d, which was never published, failed: %v", version, err)
		}
	}
	// Reserved numbers are skipped, not reused.
	if got := mustSave(t, service, "file", genai.NewPartFromText("next")); got != 3 {
		t.Errorf("Save() = version %d, want 3", got)
	}
	if diff := cmp.Diff(want, dirEntries(t, versions)); diff != "" {
		t.Errorf("versions directory mismatch (-want +got):\n%s", diff)
	}

	// An artifact with nothing published is not listed, and neither is the
	// directory of the scope, which is no artifact even with versions in it.
	must(t, os.MkdirAll(filepath.Join(sessionDir(root), "unpublished", "versions", ".0.pending"), 0o700), "MkdirAll()")
	must(t, os.MkdirAll(filepath.Join(sessionDir(root), "versions", "0"), 0o700), "MkdirAll()")
	if diff := cmp.Diff([]string{"file"}, mustList(t, service)); diff != "" {
		t.Errorf("List() mismatch (-want +got):\n%s", diff)
	}
	if _, err := service.Versions(t.Context(), &artifact.VersionsRequest{AppName: app, UserID: user, SessionID: session, FileName: "unpublished"}); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("Versions() of an unpublished artifact: err = %v, want fs.ErrNotExist", err)
	}
}

func TestVersionLimit(t *testing.T) {
	root := newRoot(t)
	service := newService(t, root)
	versions := filepath.Join(sessionDir(root), "file", "versions")
	must(t, os.MkdirAll(filepath.Join(versions, "9223372036854775807"), 0o700), "MkdirAll()")
	if _, err := service.Save(t.Context(), saveRequest("file", genai.NewPartFromText("x"))); err == nil {
		t.Error("Save() after the highest possible version succeeded")
	}
	if got := dirEntries(t, versions); !slices.Equal(got, []string{"9223372036854775807"}) {
		t.Errorf("versions = %v, want only the one that was there", got)
	}

	// The same when the last number is only reserved.
	reserved := filepath.Join(sessionDir(root), "reserved", "versions")
	must(t, os.MkdirAll(filepath.Join(reserved, "9223372036854775806"), 0o700), "MkdirAll()")
	must(t, os.Mkdir(filepath.Join(reserved, ".9223372036854775807.pending"), 0o700), "Mkdir()")
	if _, err := service.Save(t.Context(), saveRequest("reserved", genai.NewPartFromText("x"))); err == nil {
		t.Error("Save() after the highest possible reservation succeeded")
	}
	if got := dirEntries(t, reserved); !slices.Equal(got, []string{".9223372036854775807.pending", "9223372036854775806"}) {
		t.Errorf("versions = %v, want only what was there", got)
	}
}

// countingContext reports cancellation from its limit-th check on.
type countingContext struct {
	context.Context
	checks *int
	limit  int
}

func (c countingContext) Err() error {
	*c.checks++
	if *c.checks >= c.limit {
		return context.Canceled
	}
	return nil
}

func TestCanceledSaveLeavesNothing(t *testing.T) {
	root := newRoot(t)
	service := newService(t, root)
	mustSave(t, service, "other", genai.NewPartFromText("x"))

	// Cancel at each point Save looks at its context, until it gets through.
	canceled := 0
	for limit := 1; ; limit++ {
		ctx := countingContext{Context: t.Context(), checks: new(int), limit: limit}
		_, err := service.Save(ctx, saveRequest("dir/file", genai.NewPartFromText("x")))
		if err == nil {
			break
		}
		canceled++
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Save() canceled at check %d: err = %v, want context.Canceled", limit, err)
		}
		if got := dirEntries(t, sessionDir(root)); !slices.Equal(got, []string{"other"}) {
			t.Fatalf("Save() canceled at check %d left %v, want only [other]", limit, got)
		}
	}
	if canceled == 0 {
		t.Error("Save() never looked at its context")
	}
	if diff := cmp.Diff([]string{"dir/file", "other"}, mustList(t, service)); diff != "" {
		t.Errorf("List() mismatch (-want +got):\n%s", diff)
	}
}

// stagedContext is canceled from the moment the file at path exists.
type stagedContext struct {
	context.Context
	path string
}

func (c stagedContext) Err() error {
	if _, err := os.Stat(c.path); err == nil {
		return context.Canceled
	}
	return nil
}

// A cancellation that arrives while the files are being staged still stops
// the version from being published.
func TestCanceledBeforePublishing(t *testing.T) {
	root := newRoot(t)
	service := newService(t, root)
	staged := filepath.Join(sessionDir(root), "dir", "file", "versions", ".0.pending", "file")
	ctx := stagedContext{Context: t.Context(), path: staged}
	if _, err := service.Save(ctx, saveRequest("dir/file", genai.NewPartFromText("x"))); !errors.Is(err, context.Canceled) {
		t.Fatalf("Save() canceled after staging: err = %v, want context.Canceled", err)
	}
	if got := dirEntries(t, sessionDir(root)); len(got) != 0 {
		t.Errorf("canceled Save() left %v behind", got)
	}
	if got := mustSave(t, service, "dir/file", genai.NewPartFromText("x")); got != 0 {
		t.Errorf("Save() after a canceled one = version %d, want 0", got)
	}
}

// vanishingContext removes a directory the moment the file at path exists.
type vanishingContext struct {
	context.Context
	path, remove string
}

func (c vanishingContext) Err() error {
	if _, err := os.Stat(c.path); err == nil {
		_ = os.RemoveAll(c.remove)
	}
	return nil
}

// Storage removed under a save, as adk-python's delete does, fails the save.
// That is not an artifact that is missing.
func TestSaveIntoRemovedStorage(t *testing.T) {
	root := newRoot(t)
	service := newService(t, root)
	versions := filepath.Join(sessionDir(root), "dir", "file", "versions")
	ctx := vanishingContext{Context: t.Context(), path: filepath.Join(versions, ".0.pending", "file"), remove: versions}
	_, err := service.Save(ctx, saveRequest("dir/file", genai.NewPartFromText("x")))
	if err == nil || errors.Is(err, fs.ErrNotExist) || errors.Is(err, fs.ErrInvalid) {
		t.Errorf("Save() into removed storage: err = %v, want an error that is neither fs.ErrNotExist nor fs.ErrInvalid", err)
	}
}

func TestCanceledContext(t *testing.T) {
	service := newService(t, newRoot(t))
	mustSave(t, service, "file", genai.NewPartFromText("x"))
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	errs := call(ctx, service, app, user, session, "file")
	_, errs["List"] = service.List(ctx, &artifact.ListRequest{AppName: app, UserID: user, SessionID: session})
	for op, err := range errs {
		if !errors.Is(err, context.Canceled) {
			t.Errorf("%s: err = %v, want context.Canceled", op, err)
		}
	}
	if _, err := service.Load(t.Context(), loadRequest("file")); err != nil {
		t.Errorf("a canceled Delete removed the artifact: %v", err)
	}
}

func TestConcurrentSaves(t *testing.T) {
	root := newRoot(t)
	const savers = 16
	versions := make([]int64, savers)
	var group sync.WaitGroup
	for i := range savers {
		group.Go(func() {
			// Separate instances, as separate processes would have.
			service, err := fileartifact.New(fileartifact.Config{RootDir: root})
			if err != nil {
				t.Errorf("New() failed: %v", err)
				return
			}
			resp, err := service.Save(t.Context(), saveRequest("file", genai.NewPartFromText(fmt.Sprint(i))))
			if err != nil {
				t.Errorf("Save() failed: %v", err)
				return
			}
			versions[i] = resp.Version
		})
	}
	group.Wait()
	if t.Failed() {
		return
	}
	service := newService(t, root)
	for i, version := range versions {
		req := loadRequest("file")
		req.Version = &version
		loaded, err := service.Load(t.Context(), req)
		if err != nil || loaded.Part.Text != fmt.Sprint(i) {
			t.Errorf("version %d does not hold what its Save wrote: %v", version, err)
		}
	}
	slices.Sort(versions)
	want := make([]int64, savers)
	var wantDirs []string
	for i := range want {
		want[i] = int64(i)
		wantDirs = append(wantDirs, fmt.Sprint(i))
	}
	if !slices.Equal(versions, want) {
		t.Errorf("Save() returned versions %v, want each of 0..%d once", versions, savers-1)
	}
	// A reservation that lost its number to another save is given up.
	slices.Sort(wantDirs)
	if got := dirEntries(t, filepath.Join(sessionDir(root), "file", "versions")); !slices.Equal(got, wantDirs) {
		t.Errorf("versions directory holds %v, want only the %d versions", got, savers)
	}
}

// Delete removes the directories it empties while Save creates them and List
// walks them. None may fail another.
func TestConcurrentSaveAndDelete(t *testing.T) {
	root := newRoot(t)
	service := newService(t, root)
	const (
		workers = 4
		rounds  = 100
	)
	done := make(chan struct{})
	var saving, others sync.WaitGroup
	// Each worker saves and deletes an artifact of its own. They share the
	// directories above, which every Delete removes when it leaves them empty.
	for i := range workers {
		name := fmt.Sprintf("dir/sub/own-%d", i)
		saving.Go(func() {
			for range rounds {
				if _, err := service.Save(t.Context(), saveRequest(name, genai.NewPartFromText("x"))); err != nil {
					t.Errorf("Save() during Delete failed: %v", err)
					return
				}
				if loaded, err := service.Load(t.Context(), loadRequest(name)); err != nil || loaded.Part.Text != "x" {
					t.Errorf("Load() of what was just saved failed: %v", err)
					return
				}
				if err := service.Delete(t.Context(), deleteRequest(name, nil)); err != nil {
					t.Errorf("Delete() during Save failed: %v", err)
					return
				}
			}
		})
	}
	// Two more save one artifact that others delete.
	for range 2 {
		saving.Go(func() {
			for range rounds {
				if _, err := service.Save(t.Context(), saveRequest("dir/sub/shared", genai.NewPartFromText("x"))); err != nil {
					t.Errorf("Save() during Delete failed: %v", err)
					return
				}
			}
		})
	}
	// until runs step until the savers are done or it fails.
	until := func(step func() error) {
		others.Go(func() {
			for {
				select {
				case <-done:
					return
				default:
				}
				if err := step(); err != nil {
					t.Error(err)
					return
				}
			}
		})
	}
	for range 2 {
		until(func() error {
			if err := service.Delete(t.Context(), deleteRequest("dir/sub/shared", nil)); err != nil {
				return fmt.Errorf("Delete() during Save failed: %w", err)
			}
			return nil
		})
		until(func() error {
			// Readers see a whole version or none.
			loaded, err := service.Load(t.Context(), loadRequest("dir/sub/shared"))
			if err != nil && !errors.Is(err, fs.ErrNotExist) {
				return fmt.Errorf("Load() during Save and Delete failed: %w", err)
			}
			if err == nil && loaded.Part.Text != "x" {
				return errors.New("Load() during Save and Delete returned a partial version")
			}
			return nil
		})
		until(func() error {
			resp, err := service.List(t.Context(), &artifact.ListRequest{AppName: app, UserID: user, SessionID: session})
			if err != nil {
				return fmt.Errorf("List() during Save and Delete failed: %w", err)
			}
			for _, name := range resp.FileNames {
				if !strings.HasPrefix(name, "dir/sub/own-") && name != "dir/sub/shared" {
					return fmt.Errorf("List() during Save and Delete returned %q, which was never saved", name)
				}
			}
			return nil
		})
	}
	saving.Wait()
	close(done)
	others.Wait()
}

// Deletes of the same versions get in each other's way, and all succeed.
func TestConcurrentDeletes(t *testing.T) {
	root := newRoot(t)
	service := newService(t, root)
	for range 40 {
		mustSave(t, service, "dir/file", genai.NewPartFromText("x"))
	}
	start := make(chan struct{})
	var group sync.WaitGroup
	for range 8 {
		group.Go(func() {
			<-start
			if err := service.Delete(t.Context(), deleteRequest("dir/file", nil)); err != nil {
				t.Errorf("Delete() beside another Delete failed: %v", err)
			}
			// Whichever Delete took a version away, none is left when any
			// of them returns.
			if _, err := service.Versions(t.Context(), &artifact.VersionsRequest{AppName: app, UserID: user, SessionID: session, FileName: "dir/file"}); !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("Versions() after Delete returned: err = %v, want fs.ErrNotExist", err)
			}
		})
	}
	close(start)
	group.Wait()
	if got := dirEntries(t, sessionDir(root)); len(got) != 0 {
		t.Errorf("artifacts directory holds %v after deleting everything, want nothing", got)
	}
}

// While the latest version is deleted, the one before it is the latest. A
// reader never finds the artifact missing.
func TestLoadLatestDuringDelete(t *testing.T) {
	service := newService(t, newRoot(t))
	mustSave(t, service, "file", genai.NewPartFromText("x"))
	const (
		churners = 3
		rounds   = 150
	)
	done := make(chan struct{})
	var churn, readers sync.WaitGroup
	// Each saves a version and deletes it again, so that the latest version
	// keeps changing and its number keeps being reused.
	for range churners {
		churn.Go(func() {
			for range rounds {
				resp, err := service.Save(t.Context(), saveRequest("file", genai.NewPartFromText("x")))
				if err != nil {
					t.Errorf("Save() failed: %v", err)
					return
				}
				if err := service.Delete(t.Context(), deleteRequest("file", &resp.Version)); err != nil {
					t.Errorf("Delete() of the latest version failed: %v", err)
					return
				}
			}
		})
	}
	for range 4 {
		readers.Go(func() {
			for {
				select {
				case <-done:
					return
				default:
				}
				if loaded, err := service.Load(t.Context(), loadRequest("file")); err != nil || loaded.Part.Text != "x" {
					t.Errorf("Load() of the latest version while later ones are deleted failed: %v", err)
					return
				}
				if _, err := service.GetArtifactVersion(t.Context(), versionRequest("file")); err != nil {
					t.Errorf("GetArtifactVersion() of the latest version while later ones are deleted failed: %v", err)
					return
				}
			}
		})
	}
	churn.Wait()
	close(done)
	readers.Wait()
}

// List reports the artifacts and passes over what else is in the directory.
func TestListSkipsStrayEntries(t *testing.T) {
	root := newRoot(t)
	service := newService(t, root)
	for _, name := range []string{"a", "b/c", "d"} {
		mustSave(t, service, name, genai.NewPartFromText("x"))
	}
	writeFile(t, filepath.Join(sessionDir(root), "stray.txt"), nil)
	writeFile(t, filepath.Join(sessionDir(root), "b", "stray.txt"), nil)
	if runtime.GOOS != "windows" {
		must(t, os.Symlink("a", filepath.Join(sessionDir(root), "c")), "Symlink()")
	}
	if diff := cmp.Diff([]string{"a", "b/c", "d"}, mustList(t, service)); diff != "" {
		t.Errorf("List() mismatch (-want +got):\n%s", diff)
	}

	if runtime.GOOS == "windows" || os.Geteuid() == 0 {
		return // No permission bits, or none that bind.
	}
	// A directory that cannot be read may hold artifacts, so the listing is
	// not complete and says so.
	locked := filepath.Join(sessionDir(root), "b")
	must(t, os.Chmod(locked, 0), "Chmod()")
	t.Cleanup(func() { _ = os.Chmod(locked, 0o700) })
	if _, err := service.List(t.Context(), &artifact.ListRequest{AppName: app, UserID: user, SessionID: session}); !errors.Is(err, fs.ErrPermission) {
		t.Errorf("List() over an unreadable directory: err = %v, want fs.ErrPermission", err)
	}
}

// Without its root the service cannot store, which is not an artifact that
// is missing.
func TestRootRemoved(t *testing.T) {
	root := filepath.Join(newRoot(t), "root")
	service := newService(t, root)
	must(t, os.RemoveAll(root), "RemoveAll()")
	_, err := service.Save(t.Context(), saveRequest("file", genai.NewPartFromText("x")))
	if err == nil || errors.Is(err, fs.ErrNotExist) || errors.Is(err, fs.ErrInvalid) {
		t.Errorf("Save() without the root: err = %v, want an error that is neither fs.ErrNotExist nor fs.ErrInvalid", err)
	}
	if _, err := service.Load(t.Context(), loadRequest("file")); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("Load() without the root: err = %v, want fs.ErrNotExist", err)
	}
	if got := mustList(t, service); len(got) != 0 {
		t.Errorf("List() without the root = %v, want nothing", got)
	}
}

func TestNew(t *testing.T) {
	t.Run("empty root", func(t *testing.T) {
		if _, err := fileartifact.New(fileartifact.Config{}); !errors.Is(err, fs.ErrInvalid) {
			t.Errorf("New() err = %v, want fs.ErrInvalid", err)
		}
	})

	t.Run("creates missing directories", func(t *testing.T) {
		parent := filepath.Join(newRoot(t), "a")
		root := filepath.Join(parent, "b")
		mustSave(t, newService(t, root), "file", genai.NewPartFromText("x"))
		if _, err := os.Stat(filepath.Join(root, "apps")); err != nil {
			t.Errorf("artifacts were not stored under the new root: %v", err)
		}
		if runtime.GOOS == "windows" {
			return // Windows has no permission bits to check.
		}
		for _, dir := range []string{parent, root} {
			info, err := os.Stat(dir)
			must(t, err, "Stat()")
			if got := info.Mode().Perm(); got != 0o700 {
				t.Errorf("mode of a directory New created = %o, want 700", got)
			}
		}
	})

	t.Run("relative root", func(t *testing.T) {
		dir := newRoot(t)
		t.Chdir(dir)
		service := newService(t, "store")
		mustSave(t, service, "file", genai.NewPartFromText("x"))
		// The service keeps working from its directory when the process moves.
		t.Chdir(t.TempDir())
		if _, err := service.Load(t.Context(), loadRequest("file")); err != nil {
			t.Errorf("Load() after changing directory failed: %v", err)
		}
		if _, err := os.Stat(filepath.Join(dir, "store", "apps")); err != nil {
			t.Errorf("artifacts were not stored under the relative root: %v", err)
		}
	})

	t.Run("tilde is a directory name", func(t *testing.T) {
		dir := newRoot(t)
		t.Chdir(dir)
		newService(t, "~")
		if _, err := os.Stat(filepath.Join(dir, "~")); err != nil {
			t.Errorf("a root of ~ was not created as a directory of that name: %v", err)
		}
	})

	t.Run("file in the way", func(t *testing.T) {
		path := filepath.Join(newRoot(t), "file")
		writeFile(t, path, nil)
		for _, root := range []string{path, filepath.Join(path, "below")} {
			_, err := fileartifact.New(fileartifact.Config{RootDir: root})
			if err == nil {
				t.Fatal("New() over a file succeeded")
			}
			if strings.Contains(err.Error(), path) {
				t.Error("the error contains the path")
			}
		}
	})

	if runtime.GOOS == "windows" {
		return // Creating symbolic links needs a privilege there.
	}
	for _, tc := range []struct {
		name string
		// setup returns the root to pass to New and the directory it must
		// end up using.
		setup func(t *testing.T, dir string) (root, want string)
	}{
		{"link to a directory", func(t *testing.T, dir string) (string, string) {
			target := filepath.Join(dir, "target")
			must(t, os.Mkdir(target, 0o700), "Mkdir()")
			return symlink(t, target, filepath.Join(dir, "link")), target
		}},
		{"relative link", func(t *testing.T, dir string) (string, string) {
			target := filepath.Join(dir, "target")
			must(t, os.MkdirAll(filepath.Join(dir, "links"), 0o700), "MkdirAll()")
			must(t, os.Mkdir(target, 0o700), "Mkdir()")
			return symlink(t, filepath.Join("..", "target"), filepath.Join(dir, "links", "link")), target
		}},
		{"link above the root", func(t *testing.T, dir string) (string, string) {
			target := filepath.Join(dir, "target")
			must(t, os.Mkdir(target, 0o700), "Mkdir()")
			return filepath.Join(symlink(t, target, filepath.Join(dir, "link")), "a", "b"), filepath.Join(target, "a", "b")
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root, want := tc.setup(t, newRoot(t))
			service := newService(t, root)
			mustSave(t, service, "file", genai.NewPartFromText("x"))
			if _, err := os.Stat(filepath.Join(want, "apps")); err != nil {
				t.Fatalf("artifacts were not stored in the directory the link names: %v", err)
			}
			resp, err := service.GetArtifactVersion(t.Context(), versionRequest("file"))
			must(t, err, "GetArtifactVersion()")
			if !strings.HasPrefix(resp.ArtifactVersion.CanonicalURI, fileURI(want)+"/") {
				t.Error("CanonicalURI is not below the resolved root")
			}
		})
	}

	// New does not create what a link points to.
	t.Run("link to a missing directory", func(t *testing.T) {
		dir := newRoot(t)
		target := filepath.Join(dir, "target")
		link := symlink(t, target, filepath.Join(dir, "link"))
		for _, root := range []string{link, filepath.Join(link, "below")} {
			if _, err := fileartifact.New(fileartifact.Config{RootDir: root}); err == nil {
				t.Error("New() over a link to a missing directory succeeded")
			}
		}
		if _, err := os.Lstat(target); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("New() created the target of the link: Lstat() err = %v, want fs.ErrNotExist", err)
		}
	})
}

// symlink creates a symbolic link at path and returns path.
func symlink(t *testing.T, target, path string) string {
	t.Helper()
	must(t, os.Symlink(target, path), "Symlink()")
	return path
}

// Filesystem errors name the path, which holds the caller's identifiers.
func TestErrorsOmitPaths(t *testing.T) {
	root := newRoot(t)
	service := newService(t, root)
	const (
		secretUser = "customer-identifier"
		secretFile = "customer-file-name"
	)
	request := func(name string) (*artifact.SaveRequest, *artifact.LoadRequest) {
		return &artifact.SaveRequest{AppName: app, UserID: secretUser, SessionID: session, FileName: name, Part: genai.NewPartFromText("x")},
			&artifact.LoadRequest{AppName: app, UserID: secretUser, SessionID: session, FileName: name}
	}
	check := func(t *testing.T, err error) {
		t.Helper()
		if err == nil {
			t.Fatal("the operation succeeded, want a filesystem error")
		}
		for _, secret := range []string{secretUser, secretFile, root} {
			if strings.Contains(err.Error(), secret) {
				t.Errorf("the error message contains part of the path")
			}
		}
		if _, ok := errors.AsType[*fs.PathError](err); ok {
			t.Error("the error carries a path for errors.As")
		}
		if _, ok := errors.AsType[*os.LinkError](err); ok {
			t.Error("the error carries paths for errors.As")
		}
	}

	t.Run("missing directory", func(t *testing.T) {
		_, load := request(secretFile)
		_, err := service.Load(t.Context(), load)
		check(t, err)
	})
	t.Run("missing payload", func(t *testing.T) {
		save, load := request(secretFile)
		_, err := service.Save(t.Context(), save)
		must(t, err, "Save()")
		must(t, os.Remove(filepath.Join(root, "apps", app, "users", secretUser, "sessions", session, "artifacts", secretFile, "versions", "0", secretFile)), "Remove()")
		_, err = service.Load(t.Context(), load)
		check(t, err)
	})
	t.Run("file in place of a directory", func(t *testing.T) {
		// Creating a directory below a file fails in the filesystem.
		save, _ := request("blocked/" + secretFile)
		must(t, os.MkdirAll(filepath.Join(root, "apps", app, "users", secretUser, "sessions", session, "artifacts"), 0o700), "MkdirAll()")
		writeFile(t, filepath.Join(root, "apps", app, "users", secretUser, "sessions", session, "artifacts", "blocked"), nil)
		_, err := service.Save(t.Context(), save)
		check(t, err)
	})
	if runtime.GOOS != "windows" {
		t.Run("rename refused", func(t *testing.T) {
			// A directory without write permission accepts no rename, which
			// fails with both paths.
			save, _ := request("locked/" + secretFile)
			_, err := service.Save(t.Context(), save)
			must(t, err, "Save()")
			versions := filepath.Join(root, "apps", app, "users", secretUser, "sessions", session, "artifacts", "locked", secretFile, "versions")
			must(t, os.Chmod(versions, 0o500), "Chmod()")
			t.Cleanup(func() { _ = os.Chmod(versions, 0o700) })
			if os.Geteuid() == 0 {
				t.Skip("permissions do not bind the superuser")
			}
			check(t, service.Delete(t.Context(), &artifact.DeleteRequest{AppName: app, UserID: secretUser, SessionID: session, FileName: "locked/" + secretFile}))
		})
		t.Run("removal refused", func(t *testing.T) {
			// A version that cannot be emptied is taken out of service and
			// then fails to go away.
			save, _ := request("stuck/" + secretFile)
			_, err := service.Save(t.Context(), save)
			must(t, err, "Save()")
			versions := filepath.Join(root, "apps", app, "users", secretUser, "sessions", session, "artifacts", "stuck", secretFile, "versions")
			must(t, os.Chmod(filepath.Join(versions, "0"), 0o500), "Chmod()")
			t.Cleanup(func() {
				for _, name := range dirEntries(t, versions) {
					_ = os.Chmod(filepath.Join(versions, name), 0o700)
				}
			})
			if os.Geteuid() == 0 {
				t.Skip("permissions do not bind the superuser")
			}
			check(t, service.Delete(t.Context(), &artifact.DeleteRequest{AppName: app, UserID: secretUser, SessionID: session, FileName: "stuck/" + secretFile}))
		})
	}
}
