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
	"context"
	"errors"
	"fmt"
	"io/fs"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
)

// refPart returns a Part whose FileData references the named artifact.
func refPart(appName, userID, sessionID, fileName string, version int64) *genai.Part {
	uri := fmt.Sprintf("artifact://apps/%s/users/%s/sessions/%s/artifacts/%s/versions/%d", appName, userID, sessionID, fileName, version)
	return &genai.Part{FileData: &genai.FileData{FileURI: uri}}
}

func TestInMemoryArtifactService_References(t *testing.T) {
	ctx := context.Background()

	// saveAndLoad saves a target artifact, then an alias that references it, and
	// loads the alias back, asserting it resolves to the target's content.
	saveAndLoad := func(svc artifact.Service, fileName, targetName, content string) *genai.Part {
		target, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "sess", FileName: targetName,
			Part: genai.NewPartFromText(content),
		})
		if err != nil {
			t.Fatalf("Save(target %q) failed: %v", targetName, err)
		}
		if _, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "sess", FileName: fileName,
			Part: refPart("app", "user", "sess", targetName, target.Version),
		}); err != nil {
			t.Fatalf("Save(alias %q) failed: %v", fileName, err)
		}
		resp, err := svc.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "user", SessionID: "sess", FileName: fileName,
		})
		if err != nil {
			t.Fatalf("Load(%q) failed: %v", fileName, err)
		}
		return resp.Part
	}

	t.Run("session-scoped reference resolves on load", func(t *testing.T) {
		svc := artifact.InMemoryService()
		part := saveAndLoad(svc, "alias.txt", "target.txt", "target contents")
		if part.Text != "target contents" {
			t.Errorf("Load(alias) = %q, want %q", part.Text, "target contents")
		}
	})

	// A user-scoped URI (no sessions segment) may point at a user-scoped file,
	// which is stored under the shared user key. Both the alias and the target
	// are visible from every session of the user, so following the reference
	// must resolve from each of them.
	t.Run("user-scoped URI resolves a user-scoped target", func(t *testing.T) {
		svc := artifact.InMemoryService()
		target, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s1", FileName: "user:target.txt",
			Part: genai.NewPartFromText("shared contents"),
		})
		if err != nil {
			t.Fatalf("Save(user:target) failed: %v", err)
		}
		if _, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s1", FileName: "user:alias.txt",
			Part: &genai.Part{FileData: &genai.FileData{
				FileURI: fmt.Sprintf("artifact://apps/app/users/user/artifacts/user:target.txt/versions/%d", target.Version),
			}},
		}); err != nil {
			t.Fatalf("Save(user:alias) failed: %v", err)
		}
		for _, sessionID := range []string{"s1", "s2"} {
			resp, err := svc.Load(ctx, &artifact.LoadRequest{
				AppName: "app", UserID: "user", SessionID: sessionID, FileName: "user:alias.txt",
			})
			if err != nil {
				t.Fatalf("Load(user:alias) from %q failed: %v", sessionID, err)
			}
			if resp.Part.Text != "shared contents" {
				t.Errorf("Load(user:alias) from %q = %q, want %q", sessionID, resp.Part.Text, "shared contents")
			}
		}
	})

	t.Run("save rejects a malformed reference URI", func(t *testing.T) {
		for _, uri := range []string{
			"artifact://nonsense",
			"artifact://apps/app/users/user/sessions/s/artifacts/f.txt/versions/0",
		} {
			t.Run(uri, func(t *testing.T) {
				svc := artifact.InMemoryService()
				_, err := svc.Save(ctx, &artifact.SaveRequest{
					AppName: "app", UserID: "user", SessionID: "s", FileName: "alias.txt",
					Part: &genai.Part{FileData: &genai.FileData{FileURI: uri}},
				})
				if err == nil {
					t.Fatalf("Save(alias) with URI %q succeeded, want an error", uri)
				}
				if !strings.Contains(err.Error(), "invalid artifact reference URI") {
					t.Errorf("Save(alias) error = %v, want an invalid-reference-URI error", err)
				}
			})
		}
	})

	t.Run("save rejects out-of-scope references", func(t *testing.T) {
		cases := []struct {
			name                  string
			app, user, sess, file string
		}{
			{"cross-app", "other-app", "user", "sess", "target.txt"},
			{"cross-user", "app", "other-user", "sess", "target.txt"},
			{"cross-session", "app", "user", "other-sess", "target.txt"},
		}
		for _, c := range cases {
			t.Run(c.name, func(t *testing.T) {
				svc := artifact.InMemoryService()
				_, err := svc.Save(ctx, &artifact.SaveRequest{
					AppName: "app", UserID: "user", SessionID: "sess", FileName: "alias.txt",
					Part: refPart(c.app, c.user, c.sess, c.file, 1),
				})
				if err == nil {
					t.Fatalf("Save(alias) with %s reference succeeded, want error", c.name)
				}
			})
		}
	})

	// B1a: a user-scoped alias saved in a session that points at a session-scoped
	// target in that same session. Before the fix, Load rewrote the session to the
	// user storage key so the per-hop scope check compared the target's session
	// against "user" and failed; the owner must still be able to load it back.
	t.Run("user-scoped alias to a session-scoped target loads for the owner", func(t *testing.T) {
		svc := artifact.InMemoryService()
		target, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s1", FileName: "target.txt",
			Part: genai.NewPartFromText("hello"),
		})
		if err != nil {
			t.Fatalf("Save(target) failed: %v", err)
		}
		if _, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s1", FileName: "user:alias",
			Part: refPart("app", "user", "s1", "target.txt", target.Version),
		}); err != nil {
			t.Fatalf("Save(user:alias) failed: %v", err)
		}
		resp, err := svc.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "user", SessionID: "s1", FileName: "user:alias",
		})
		if err != nil {
			t.Fatalf("Load(user:alias) for owner failed: %v", err)
		}
		if resp.Part.Text != "hello" {
			t.Errorf("Load(user:alias) = %q, want %q", resp.Part.Text, "hello")
		}
	})

	// B1b: the same user-scoped alias is reachable from another session of the
	// same user, but its target is session-private to s1; following it from s2
	// must be rejected by the load-time scope check. This is also the case that
	// truly reaches the Load scope check (the alias lives in user scope).
	t.Run("user-scoped alias to a private target rejects a foreign session", func(t *testing.T) {
		svc := artifact.InMemoryService()
		target, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s1", FileName: "target.txt",
			Part: genai.NewPartFromText("secret"),
		})
		if err != nil {
			t.Fatalf("Save(target) failed: %v", err)
		}
		if _, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s1", FileName: "user:alias",
			Part: refPart("app", "user", "s1", "target.txt", target.Version),
		}); err != nil {
			t.Fatalf("Save(user:alias) failed: %v", err)
		}
		_, err = svc.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "user", SessionID: "s2", FileName: "user:alias",
		})
		if err == nil {
			t.Fatal("Load(user:alias) from a foreign session succeeded, want a scope error")
		}
		if errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Load(user:alias) from a foreign session returned not-found, want a scope error: %v", err)
		}
	})

	// A hop through a user-scoped URI carries no session, so a later
	// session-scoped reference must fail the scope check even when it names the
	// session that saved the chain. Otherwise another session could read a
	// private file through it.
	t.Run("user-scoped chain refuses a session-scoped hop", func(t *testing.T) {
		svc := artifact.InMemoryService()
		secret, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s1", FileName: "secret.txt",
			Part: genai.NewPartFromText("secret"),
		})
		if err != nil {
			t.Fatalf("Save(secret) failed: %v", err)
		}
		inner, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s1", FileName: "user:inner.txt",
			Part: refPart("app", "user", "s1", "secret.txt", secret.Version),
		})
		if err != nil {
			t.Fatalf("Save(user:inner) failed: %v", err)
		}
		if _, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s1", FileName: "user:alias.txt",
			Part: &genai.Part{FileData: &genai.FileData{
				FileURI: fmt.Sprintf("artifact://apps/app/users/user/artifacts/user:inner.txt/versions/%d", inner.Version),
			}},
		}); err != nil {
			t.Fatalf("Save(user:alias) failed: %v", err)
		}
		for _, sessionID := range []string{"s1", "s2"} {
			_, err := svc.Load(ctx, &artifact.LoadRequest{
				AppName: "app", UserID: "user", SessionID: sessionID, FileName: "user:alias.txt",
			})
			if err == nil {
				t.Fatalf("Load(user:alias) from %q succeeded, want a scope error", sessionID)
			}
			if errors.Is(err, fs.ErrNotExist) {
				t.Errorf("Load(user:alias) from %q returned not-found, want a scope error: %v", sessionID, err)
			}
			if !strings.Contains(err.Error(), "session-scoped artifact reference must stay within the same session scope") {
				t.Errorf("Load(user:alias) from %q error = %v, want the session-scope error", sessionID, err)
			}
		}
	})

	// makeChain stores f0..fhops in one session; fhops holds data and each earlier
	// file references the next, producing `hops` reference hops.
	makeChain := func(svc artifact.Service, hops int) {
		for i := hops; i >= 0; i-- {
			name := fmt.Sprintf("f%d", i)
			req := &artifact.SaveRequest{
				AppName: "app", UserID: "user", SessionID: "s", FileName: name,
			}
			if i == hops {
				req.Part = genai.NewPartFromText("chain end")
			} else {
				req.Part = refPart("app", "user", "s", fmt.Sprintf("f%d", i+1), 1)
			}
			if _, err := svc.Save(ctx, req); err != nil {
				t.Fatalf("Save(%q) failed: %v", name, err)
			}
		}
	}

	t.Run("a chain within the depth limit resolves", func(t *testing.T) {
		svc := artifact.InMemoryService()
		makeChain(svc, 5) // 5 hops, at the limit
		resp, err := svc.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "user", SessionID: "s", FileName: "f0",
		})
		if err != nil {
			t.Fatalf("Load(f0) within depth failed: %v", err)
		}
		if resp.Part.Text != "chain end" {
			t.Errorf("Load(f0) = %q, want %q", resp.Part.Text, "chain end")
		}
	})

	t.Run("a chain past the depth limit errors", func(t *testing.T) {
		svc := artifact.InMemoryService()
		makeChain(svc, 6) // 6 hops, one past the limit
		_, err := svc.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "user", SessionID: "s", FileName: "f0",
		})
		if err == nil {
			t.Fatal("Load over the depth limit succeeded, want the depth error")
		}
		if !strings.Contains(err.Error(), "maximum depth of 5") {
			t.Errorf("Load over the depth limit error = %v, want the maximum-depth error", err)
		}
	})

	t.Run("a self-referencing chain is bounded by the depth limit", func(t *testing.T) {
		svc := artifact.InMemoryService()
		if _, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s", FileName: "loop.txt",
			Part: refPart("app", "user", "s", "loop.txt", 1),
		}); err != nil {
			t.Fatalf("Save(loop) failed: %v", err)
		}
		_, err := svc.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "user", SessionID: "s", FileName: "loop.txt",
		})
		if err == nil {
			t.Fatal("Load of a self-referencing artifact succeeded, want the depth error")
		}
		if !strings.Contains(err.Error(), "maximum depth of 5") {
			t.Errorf("Load of a self-referencing artifact error = %v, want the maximum-depth error", err)
		}
	})
}

// TestInMemoryArtifactVersionReferenceMimeType pins that a stored reference has
// no media type of its own: the target's type is only known once Load follows
// the reference. adk-python's in-memory service leaves mime_type unset for a
// reference (in_memory_artifact_service.py), and Load still reports the
// target's type. The reference carries a FileData.MIMEType of its own so that
// the assertion cannot pass by accident when the field is empty.
func TestInMemoryArtifactVersionReferenceMimeType(t *testing.T) {
	ctx := context.Background()
	svc := artifact.InMemoryService()
	target, err := svc.Save(ctx, &artifact.SaveRequest{
		AppName: "app", UserID: "user", SessionID: "s", FileName: "target.png",
		Part: genai.NewPartFromBytes([]byte("png"), "image/png"),
	})
	if err != nil {
		t.Fatalf("Save(target) failed: %v", err)
	}
	aliasPart := refPart("app", "user", "s", "target.png", target.Version)
	aliasPart.FileData.MIMEType = "image/png"
	alias, err := svc.Save(ctx, &artifact.SaveRequest{
		AppName: "app", UserID: "user", SessionID: "s", FileName: "alias.txt",
		Part: aliasPart,
	})
	if err != nil {
		t.Fatalf("Save(alias) failed: %v", err)
	}

	got, err := svc.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{
		AppName: "app", UserID: "user", SessionID: "s", FileName: "alias.txt", Version: alias.Version,
	})
	if err != nil {
		t.Fatalf("GetArtifactVersion(alias) failed: %v", err)
	}
	if got.ArtifactVersion.MimeType != "" {
		t.Errorf("GetArtifactVersion(alias).MimeType = %q, want empty for a reference", got.ArtifactVersion.MimeType)
	}

	loaded, err := svc.Load(ctx, &artifact.LoadRequest{
		AppName: "app", UserID: "user", SessionID: "s", FileName: "alias.txt",
	})
	if err != nil {
		t.Fatalf("Load(alias) failed: %v", err)
	}
	if got := loaded.Part.InlineData.MIMEType; got != "image/png" {
		t.Errorf("Load(alias).Part.InlineData.MIMEType = %q, want %q", got, "image/png")
	}
}

// TestInMemoryArtifactService_ExternalFileData checks that the in-memory
// service stores file data that is not an artifact:// reference, loads it back
// as file data and reports its media type, as adk-python's in-memory service
// does (in_memory_artifact_service.py, save_artifact).
func TestInMemoryArtifactService_ExternalFileData(t *testing.T) {
	ctx := context.Background()
	testCases := []struct {
		name         string
		uri, mime    string
		wantMimeType string
	}{
		{name: "empty URI", uri: "", mime: "text/plain", wantMimeType: "text/plain"},
		{name: "gs URI", uri: "gs://my-bucket/file.pdf", mime: "application/pdf", wantMimeType: "application/pdf"},
		{name: "empty URI with a binary MIME type", uri: "", mime: "application/octet-stream", wantMimeType: "application/octet-stream"},
		{name: "gs URI without a MIME type", uri: "gs://my-bucket/file.bin", mime: "", wantMimeType: ""},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			svc := artifact.InMemoryService()
			saved, err := svc.Save(ctx, &artifact.SaveRequest{
				AppName: "app", UserID: "user", SessionID: "s", FileName: "external",
				Part: &genai.Part{FileData: &genai.FileData{FileURI: tc.uri, MIMEType: tc.mime}},
			})
			if err != nil {
				t.Fatalf("Save(external) failed: %v", err)
			}
			if saved.Version != 1 {
				t.Errorf("Save(external).Version = %d, want 1", saved.Version)
			}

			loaded, err := svc.Load(ctx, &artifact.LoadRequest{
				AppName: "app", UserID: "user", SessionID: "s", FileName: "external",
			})
			if err != nil {
				t.Fatalf("Load(external) failed: %v", err)
			}
			want := &genai.Part{FileData: &genai.FileData{FileURI: tc.uri, MIMEType: tc.mime}}
			if diff := cmp.Diff(want, loaded.Part); diff != "" {
				t.Errorf("Load(external).Part mismatch (-want +got):\n%s", diff)
			}

			got, err := svc.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{
				AppName: "app", UserID: "user", SessionID: "s", FileName: "external",
			})
			if err != nil {
				t.Fatalf("GetArtifactVersion(external) failed: %v", err)
			}
			if got.ArtifactVersion.MimeType != tc.wantMimeType {
				t.Errorf("GetArtifactVersion(external).MimeType = %q, want %q", got.ArtifactVersion.MimeType, tc.wantMimeType)
			}
		})
	}
}

// TestInMemoryArtifactService_ReferenceErrorsWrapSentinelAndPhrase checks that
// every refused artifact reference wraps artifact.ErrInputValidation, carries
// adk-python's phrase, and leaves the URI out of the message.
func TestInMemoryArtifactService_ReferenceErrorsWrapSentinelAndPhrase(t *testing.T) {
	ctx := context.Background()

	// assertRefused checks err against the sentinel and the wanted substrings,
	// and that it holds none of the unwanted ones.
	assertRefused := func(t *testing.T, err error, want, unwanted []string) {
		t.Helper()
		if err == nil {
			t.Fatal("got nil error, want a refused reference")
		}
		if !errors.Is(err, artifact.ErrInputValidation) {
			t.Errorf("errors.Is(%v, artifact.ErrInputValidation) = false, want true", err)
		}
		for _, s := range want {
			if !strings.Contains(err.Error(), s) {
				t.Errorf("error %q does not contain %q", err.Error(), s)
			}
		}
		for _, s := range unwanted {
			if strings.Contains(err.Error(), s) {
				t.Errorf("error %q contains %q, want it left out", err.Error(), s)
			}
		}
	}

	t.Run("save of a malformed URI", func(t *testing.T) {
		svc := artifact.InMemoryService()
		_, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s", FileName: "alias.txt",
			Part: &genai.Part{FileData: &genai.FileData{FileURI: "artifact://secret-app/secret-user/invalid"}},
		})
		assertRefused(t, err,
			[]string{"request validation failed", "invalid artifact reference URI", "Invalid artifact reference URI"},
			[]string{"secret"})

		list, err := svc.List(ctx, &artifact.ListRequest{AppName: "app", UserID: "user", SessionID: "s"})
		if err != nil {
			t.Fatalf("List() failed: %v", err)
		}
		if len(list.FileNames) != 0 {
			t.Errorf("List().FileNames = %v, want none", list.FileNames)
		}
		if _, err := svc.Versions(ctx, &artifact.VersionsRequest{
			AppName: "app", UserID: "user", SessionID: "s", FileName: "alias.txt",
		}); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Versions(alias) error = %v, want fs.ErrNotExist", err)
		}
	})

	t.Run("save of a cross-user reference", func(t *testing.T) {
		svc := artifact.InMemoryService()
		_, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s", FileName: "alias.txt",
			Part: refPart("app", "victim", "s", "target.txt", 1),
		})
		assertRefused(t, err,
			[]string{"request validation failed", "must stay within the same app and user scope"},
			[]string{"victim"})
	})

	t.Run("save of a user-scoped name without a session referencing a session file", func(t *testing.T) {
		svc := artifact.InMemoryService()
		_, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", FileName: "user:alias.txt",
			Part: refPart("app", "user", "s1", "file1", 1),
		})
		assertRefused(t, err,
			[]string{"request validation failed", "session-scoped artifact reference must stay within the same session scope"},
			[]string{"s1"})
	})

	t.Run("load hop into another session", func(t *testing.T) {
		svc := artifact.InMemoryService()
		if _, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s1", FileName: "target.txt",
			Part: genai.NewPartFromText("private"),
		}); err != nil {
			t.Fatalf("Save(target) failed: %v", err)
		}
		if _, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s1", FileName: "user:alias",
			Part: refPart("app", "user", "s1", "target.txt", 1),
		}); err != nil {
			t.Fatalf("Save(user:alias) failed: %v", err)
		}
		_, err := svc.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "user", SessionID: "s2", FileName: "user:alias",
		})
		assertRefused(t, err,
			[]string{"session-scoped artifact reference must stay within the same session scope"},
			[]string{"s1", "target.txt"})
	})

	t.Run("load past the depth limit", func(t *testing.T) {
		svc := artifact.InMemoryService()
		if _, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "s", FileName: "loop.txt",
			Part: refPart("app", "user", "s", "loop.txt", 1),
		}); err != nil {
			t.Fatalf("Save(loop) failed: %v", err)
		}
		_, err := svc.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "user", SessionID: "s", FileName: "loop.txt",
		})
		assertRefused(t, err,
			[]string{"maximum depth of 5", "Exceeded maximum recursion depth resolving artifact reference"},
			[]string{"artifact://", "loop.txt"})
	})
}
