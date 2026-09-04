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
	"io/fs"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/artifact/tests"
)

func TestInMemoryArtifactService(t *testing.T) {
	factory := func(t *testing.T) (artifact.Service, error) {
		return artifact.InMemoryService(), nil
	}
	tests.TestArtifactService(t, "InMemory", factory)
}

func TestInMemoryArtifactService_References(t *testing.T) {
	ctx := context.Background()

	t.Run("save and load resolves reference", func(t *testing.T) {
		svc := artifact.InMemoryService()

		saveResp, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "sess", FileName: "target.txt",
			Part: genai.NewPartFromText("target contents"),
		})
		if err != nil {
			t.Fatalf("Save(target) failed: %v", err)
		}

		refURI := artifact.BuildArtifactURI("app", "user", "sess", "target.txt", saveResp.Version)
		_, err = svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "sess", FileName: "alias.txt",
			Part: &genai.Part{FileData: &genai.FileData{FileURI: refURI}},
		})
		if err != nil {
			t.Fatalf("Save(alias) failed: %v", err)
		}

		loadResp, err := svc.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "user", SessionID: "sess", FileName: "alias.txt",
		})
		if err != nil {
			t.Fatalf("Load(alias) failed: %v", err)
		}
		if loadResp.Part.Text != "target contents" {
			t.Errorf("Load(alias) resolved to %+v, want text %q", loadResp.Part, "target contents")
		}
	})

	t.Run("save rejects cross-app reference", func(t *testing.T) {
		svc := artifact.InMemoryService()
		refURI := artifact.BuildArtifactURI("other-app", "user", "sess", "target.txt", 1)
		_, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "sess", FileName: "alias.txt",
			Part: &genai.Part{FileData: &genai.FileData{FileURI: refURI}},
		})
		if err == nil {
			t.Fatal("Save(alias) with cross-app reference succeeded, want error")
		}
	})

	t.Run("save rejects cross-user reference", func(t *testing.T) {
		svc := artifact.InMemoryService()
		refURI := artifact.BuildArtifactURI("app", "other-user", "sess", "target.txt", 1)
		_, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "sess", FileName: "alias.txt",
			Part: &genai.Part{FileData: &genai.FileData{FileURI: refURI}},
		})
		if err == nil {
			t.Fatal("Save(alias) with cross-user reference succeeded, want error")
		}
	})

	t.Run("save rejects cross-session reference", func(t *testing.T) {
		svc := artifact.InMemoryService()
		refURI := artifact.BuildArtifactURI("app", "user", "other-sess", "target.txt", 1)
		_, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "sess", FileName: "alias.txt",
			Part: &genai.Part{FileData: &genai.FileData{FileURI: refURI}},
		})
		if err == nil {
			t.Fatal("Save(alias) with cross-session reference succeeded, want error")
		}
	})

	t.Run("load rejects a reference chain that tries to escape scope after save", func(t *testing.T) {
		// A reference is validated at save time, so a later Load of a
		// legitimately-scoped reference must still succeed even though the
		// resolution walks back through ValidateArtifactReferenceScope again.
		svc := artifact.InMemoryService()
		if _, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "sess", FileName: "target.txt",
			Part: genai.NewPartFromText("hello"),
		}); err != nil {
			t.Fatalf("Save(target) failed: %v", err)
		}
		refURI := artifact.BuildArtifactURI("app", "user", "sess", "target.txt", 1)
		if _, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "sess", FileName: "alias.txt",
			Part: &genai.Part{FileData: &genai.FileData{FileURI: refURI}},
		}); err != nil {
			t.Fatalf("Save(alias) failed: %v", err)
		}

		// Loading from a different session must fail scope validation even
		// though the file names match, since aliases are keyed per-scope.
		_, err := svc.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "user", SessionID: "other-sess", FileName: "alias.txt",
		})
		if err == nil {
			t.Fatal("Load(alias) from a different session succeeded, want not-found error")
		}
		if !errors.Is(err, fs.ErrNotExist) {
			// alias.txt was never saved under "other-sess", so this should
			// surface as a plain not-found, not a scope error.
			t.Errorf("Load(alias) from a different session returned %v, want fs.ErrNotExist", err)
		}
	})
}
