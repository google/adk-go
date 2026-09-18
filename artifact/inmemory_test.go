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

package artifact_test

import (
	"net/url"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/artifact/tests"
	"google.golang.org/adk/v2/platform"
)

func TestInMemoryArtifactService(t *testing.T) {
	factory := func(t *testing.T) (artifact.Service, error) {
		return artifact.InMemoryService(), nil
	}
	tests.TestArtifactService(t, "InMemory", factory)
}

func TestInMemoryArtifactVersionFields(t *testing.T) {
	firstCreateTime := time.Date(2026, time.August, 31, 12, 34, 56, 0, time.UTC)
	secondCreateTime := firstCreateTime.Add(time.Minute)
	readTime := secondCreateTime.Add(time.Hour)

	for _, tc := range []struct {
		name          string
		fileName      string
		saveSessionID string
		getSessionID  string
		uriPrefix     string
	}{
		{
			name:          "session scoped",
			fileName:      "file.txt",
			saveSessionID: "session",
			getSessionID:  "session",
			uriPrefix:     "memory://apps/app/users/user/sessions/session/artifacts/file.txt/versions/",
		},
		{
			name:          "user scoped",
			fileName:      "user:file.txt",
			saveSessionID: "save-session",
			getSessionID:  "irrelevant-session",
			uriPrefix:     "memory://apps/app/users/user/artifacts/user:file.txt/versions/",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := artifact.InMemoryService()
			metadata := map[string]any{"key": "value", "count": 42, "enabled": true}
			firstPart := genai.NewPartFromBytes([]byte("data"), "image/png")
			firstSaveCtx := platform.WithTimeProvider(t.Context(), func() time.Time { return firstCreateTime })
			if _, err := srv.Save(firstSaveCtx, &artifact.SaveRequest{
				AppName: "app", UserID: "user", SessionID: tc.saveSessionID, FileName: tc.fileName,
				Part: firstPart, CustomMetadata: metadata,
			}); err != nil {
				t.Fatalf("Save(v1) failed: %v", err)
			}
			metadata["key"] = "changed after save"
			firstPart.InlineData.MIMEType = "application/changed"

			secondSaveCtx := platform.WithTimeProvider(t.Context(), func() time.Time { return secondCreateTime })
			if _, err := srv.Save(secondSaveCtx, &artifact.SaveRequest{
				AppName: "app", UserID: "user", SessionID: tc.saveSessionID, FileName: tc.fileName,
				Part: genai.NewPartFromText("text"),
			}); err != nil {
				t.Fatalf("Save(v2) failed: %v", err)
			}

			readCtx := platform.WithTimeProvider(t.Context(), func() time.Time { return readTime })
			latest, err := srv.GetArtifactVersion(readCtx, &artifact.GetArtifactVersionRequest{
				AppName: "app", UserID: "user", SessionID: tc.getSessionID, FileName: tc.fileName,
			})
			if err != nil {
				t.Fatalf("GetArtifactVersion(latest) failed: %v", err)
			}
			wantLatest := &artifact.ArtifactVersion{
				Version:        2,
				CanonicalURI:   tc.uriPrefix + "2",
				CustomMetadata: map[string]any{},
				CreateTime:     secondCreateTime,
				MimeType:       "text/plain",
			}
			if diff := cmp.Diff(wantLatest, latest.ArtifactVersion); diff != "" {
				t.Errorf("GetArtifactVersion(latest) mismatch (-want +got):\n%s", diff)
			}

			first, err := srv.GetArtifactVersion(readCtx, &artifact.GetArtifactVersionRequest{
				AppName: "app", UserID: "user", SessionID: tc.getSessionID, FileName: tc.fileName, Version: 1,
			})
			if err != nil {
				t.Fatalf("GetArtifactVersion(v1) failed: %v", err)
			}
			wantFirst := &artifact.ArtifactVersion{
				Version:        1,
				CanonicalURI:   tc.uriPrefix + "1",
				CustomMetadata: map[string]any{"key": "value", "count": "42", "enabled": "true"},
				CreateTime:     firstCreateTime,
				MimeType:       "application/changed",
			}
			if diff := cmp.Diff(wantFirst, first.ArtifactVersion); diff != "" {
				t.Errorf("GetArtifactVersion(v1) mismatch (-want +got):\n%s", diff)
			}
			loaded, err := srv.Load(readCtx, &artifact.LoadRequest{
				AppName: "app", UserID: "user", SessionID: tc.getSessionID, FileName: tc.fileName, Version: 1,
			})
			if err != nil {
				t.Fatalf("Load(v1) failed: %v", err)
			}
			if got := loaded.Part.InlineData.MIMEType; got != first.ArtifactVersion.MimeType {
				t.Errorf("Load(v1).Part.InlineData.MIMEType = %q, GetArtifactVersion(v1).MimeType = %q", got, first.ArtifactVersion.MimeType)
			}

			first.ArtifactVersion.CustomMetadata["key"] = "changed after read"
			again, err := srv.GetArtifactVersion(readCtx, &artifact.GetArtifactVersionRequest{
				AppName: "app", UserID: "user", SessionID: tc.getSessionID, FileName: tc.fileName, Version: 1,
			})
			if err != nil {
				t.Fatalf("second GetArtifactVersion(v1) failed: %v", err)
			}
			if diff := cmp.Diff(wantFirst, again.ArtifactVersion); diff != "" {
				t.Errorf("second GetArtifactVersion(v1) mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestInMemoryCanonicalURIEscapesSegments(t *testing.T) {
	for _, tc := range []struct {
		name     string
		fileName string
		wantURI  string
	}{
		{
			name:     "session scoped",
			fileName: "report?old#draft",
			wantURI:  "memory://apps/app%2Fone/users/user%2Ftwo/sessions/session%2Fthree/artifacts/report%3Fold%23draft/versions/1",
		},
		{
			name:     "user scoped",
			fileName: "user:report?old#draft",
			wantURI:  "memory://apps/app%2Fone/users/user%2Ftwo/artifacts/user:report%3Fold%23draft/versions/1",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := artifact.InMemoryService()
			if _, err := srv.Save(t.Context(), &artifact.SaveRequest{
				AppName: "app/one", UserID: "user/two", SessionID: "session/three", FileName: tc.fileName,
				Part: genai.NewPartFromText("data"),
			}); err != nil {
				t.Fatalf("Save() failed: %v", err)
			}

			resp, err := srv.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{
				AppName: "app/one", UserID: "user/two", SessionID: "session/three", FileName: tc.fileName,
			})
			if err != nil {
				t.Fatalf("GetArtifactVersion() failed: %v", err)
			}
			if got := resp.ArtifactVersion.CanonicalURI; got != tc.wantURI {
				t.Errorf("CanonicalURI = %q, want %q", got, tc.wantURI)
			}
			parsed, err := url.Parse(resp.ArtifactVersion.CanonicalURI)
			if err != nil {
				t.Fatalf("url.Parse(CanonicalURI) failed: %v", err)
			}
			if parsed.RawQuery != "" || parsed.Fragment != "" {
				t.Errorf("CanonicalURI parsed with query %q and fragment %q, want neither", parsed.RawQuery, parsed.Fragment)
			}
		})
	}
}
