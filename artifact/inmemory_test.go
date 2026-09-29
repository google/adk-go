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

// TestInMemoryArtifactService_SaveHonorVersion pins that Save stores an artifact
// at an explicitly requested version rather than always assigning the next one,
// as SaveRequest.Version documents. On a service that ignores the requested
// version, the returned version is the auto-assigned one and the targeted slot
// is left untouched.
func TestInMemoryArtifactService_SaveHonorVersion(t *testing.T) {
	ctx := context.Background()
	s := artifact.InMemoryService()
	const appName, userID, sessionID, fileName = "app", "user", "sess", "file1"

	// Seed two auto-assigned versions. A positive version must land in its own
	// slot, independent of the auto-assigned sequence.
	for _, data := range []string{"auto-1", "auto-2"} {
		got, err := s.Save(ctx, &artifact.SaveRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName,
			Part: genai.NewPartFromBytes([]byte(data), "text/plain"),
		})
		if err != nil {
			t.Fatalf("Save() error = %v", err)
		}
		if got.Version == 0 {
			t.Fatal("Save() returned a zero version")
		}
	}

	got, err := s.Save(ctx, &artifact.SaveRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName,
		Version: 2,
		Part:    genai.NewPartFromBytes([]byte("explicit-2"), "text/plain"),
	})
	if err != nil {
		t.Fatalf("Save() with explicit version error = %v", err)
	}
	if got.Version != 2 {
		t.Fatalf("Save() with Version:2 returned version %d, want 2", got.Version)
	}

	loaded, err := s.Load(ctx, &artifact.LoadRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Version: 2,
	})
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if loaded == nil || loaded.Part == nil || loaded.Part.InlineData == nil || string(loaded.Part.InlineData.Data) != "explicit-2" {
		t.Fatalf("Load(v=2) = %v, want inline content \"explicit-2\"", loaded)
	}
}
