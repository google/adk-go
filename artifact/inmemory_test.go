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
				MimeType:       "image/png",
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
		{
			name:     "session scoped percent and space",
			fileName: "report 100%.txt",
			wantURI:  "memory://apps/app%2Fone/users/user%2Ftwo/sessions/session%2Fthree/artifacts/report%20100%25.txt/versions/1",
		},
		{
			name:     "user scoped percent and space",
			fileName: "user:report 100%.txt",
			wantURI:  "memory://apps/app%2Fone/users/user%2Ftwo/artifacts/user:report%20100%25.txt/versions/1",
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

// adk-python's InMemoryArtifactService deep-copies the part on save and on
// load (in_memory_artifact_service.py: `model_copy(deep=True)`), so a caller
// that edits a part it saved or loaded cannot rewrite the stored version.
func TestInMemoryService_DoesNotAliasStoredParts(t *testing.T) {
	ctx := t.Context()
	s := artifact.InMemoryService()
	req := func(p *genai.Part) *artifact.SaveRequest {
		return &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.txt", Part: p}
	}
	part := genai.NewPartFromText("v1")
	if _, err := s.Save(ctx, req(part)); err != nil {
		t.Fatal(err)
	}
	part.Text = "edited after save"

	load := func() *genai.Part {
		t.Helper()
		resp, err := s.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.txt"})
		if err != nil {
			t.Fatal(err)
		}
		return resp.Part
	}
	got := load()
	if got.Text != "v1" {
		t.Errorf("Load() after editing the saved part = %q, want %q", got.Text, "v1")
	}
	got.Text = "edited after load"
	if again := load(); again.Text != "v1" {
		t.Errorf("Load() after editing a loaded part = %q, want %q", again.Text, "v1")
	}
}

func fullPart() *genai.Part {
	fps := 24.0
	return &genai.Part{
		Text:             "x",
		InlineData:       &genai.Blob{Data: []byte("abc"), MIMEType: "text/plain"},
		FileData:         &genai.FileData{FileURI: "gs://b/o"},
		ThoughtSignature: []byte("sig"),
		VideoMetadata:    &genai.VideoMetadata{FPS: &fps, StartOffset: time.Second},
		PartMetadata:     map[string]any{"src": map[string]any{"name": "a"}, "list": []any{map[string]any{"n": "1"}}},
		MediaResolution:  &genai.PartMediaResolution{Level: genai.PartMediaResolutionLevelMediaResolutionLow},
		FunctionCall:     &genai.FunctionCall{Name: "fn", Args: map[string]any{"a": map[string]any{"b": "c"}}},
		FunctionResponse: &genai.FunctionResponse{Name: "fn", Response: map[string]any{"r": []any{"s"}}},
		ExecutableCode:   &genai.ExecutableCode{Code: "print(1)"},
	}
}

// mutateFullPart edits every member of a part built by fullPart in place.
func mutateFullPart(p *genai.Part) {
	p.InlineData.Data[0] = 'X'
	p.InlineData.MIMEType = "edited"
	p.FileData.FileURI = "edited"
	p.ThoughtSignature[0] = 'X'
	*p.VideoMetadata.FPS = 60
	p.VideoMetadata.StartOffset = time.Hour
	p.PartMetadata["src"].(map[string]any)["name"] = "b"
	p.PartMetadata["list"].([]any)[0].(map[string]any)["n"] = "2"
	p.MediaResolution.Level = genai.PartMediaResolutionLevelMediaResolutionHigh
	p.FunctionCall.Args["a"].(map[string]any)["b"] = "edited"
	p.FunctionResponse.Response["r"].([]any)[0] = "edited"
	p.ExecutableCode.Code = "edited"
}

func TestInMemoryService_DoesNotAliasStoredPartFields(t *testing.T) {
	ctx := t.Context()
	s := artifact.InMemoryService()
	load := func(version int64) *genai.Part {
		t.Helper()
		resp, err := s.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.bin", Version: version})
		if err != nil {
			t.Fatal(err)
		}
		return resp.Part
	}
	save := func(p *genai.Part) {
		t.Helper()
		if _, err := s.Save(ctx, &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.bin", Part: p}); err != nil {
			t.Fatal(err)
		}
	}

	t.Run("edit after save", func(t *testing.T) {
		part := fullPart()
		save(part)
		mutateFullPart(part)
		if diff := cmp.Diff(fullPart(), load(0)); diff != "" {
			t.Errorf("Load() after editing the saved part mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("edit after load", func(t *testing.T) {
		mutateFullPart(load(0))
		if diff := cmp.Diff(fullPart(), load(0)); diff != "" {
			t.Errorf("Load() after editing a loaded part mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("edit after versioned load", func(t *testing.T) {
		// Two versions, so Version: 1 takes the explicit-version path in Load.
		save(genai.NewPartFromText("v2"))
		mutateFullPart(load(1))
		if diff := cmp.Diff(fullPart(), load(1)); diff != "" {
			t.Errorf("Load(Version: 1) after editing a version-1 load mismatch (-want +got):\n%s", diff)
		}
	})
}

// The case from the review of the first version of the copy, where only some
// of the pointer members of genai.Part were detached.
func TestInMemoryService_ReviewRepro(t *testing.T) {
	ctx := t.Context()
	s := artifact.InMemoryService()
	fps := 24.0
	p := &genai.Part{
		Text:            "x",
		VideoMetadata:   &genai.VideoMetadata{FPS: &fps},
		PartMetadata:    map[string]any{"src": map[string]any{"name": "a"}},
		MediaResolution: &genai.PartMediaResolution{Level: genai.PartMediaResolutionLevelMediaResolutionLow},
	}
	if _, err := s.Save(ctx, &artifact.SaveRequest{AppName: "a", UserID: "u", SessionID: "s", FileName: "f", Part: p}); err != nil {
		t.Fatal(err)
	}
	fps = 60
	p.PartMetadata["src"].(map[string]any)["name"] = "b"
	p.MediaResolution.Level = genai.PartMediaResolutionLevelMediaResolutionHigh

	resp, err := s.Load(ctx, &artifact.LoadRequest{AppName: "a", UserID: "u", SessionID: "s", FileName: "f"})
	if err != nil {
		t.Fatal(err)
	}
	if got := *resp.Part.VideoMetadata.FPS; got != 24 {
		t.Errorf("VideoMetadata.FPS = %v, want 24", got)
	}
	if got := resp.Part.PartMetadata["src"].(map[string]any)["name"]; got != "a" {
		t.Errorf("PartMetadata[src][name] = %v, want a", got)
	}
	if got := resp.Part.MediaResolution.Level; got != genai.PartMediaResolutionLevelMediaResolutionLow {
		t.Errorf("MediaResolution.Level = %v, want %v", got, genai.PartMediaResolutionLevelMediaResolutionLow)
	}
}
