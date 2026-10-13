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
	"slices"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/artifact"
	"google.golang.org/adk/internal/artifact/tests"
)

func TestInMemoryArtifactService(t *testing.T) {
	factory := func(t *testing.T) (artifact.Service, error) {
		return artifact.InMemoryService(), nil
	}
	tests.TestArtifactService(t, "InMemory", factory)
}

func TestInMemoryListSessionNamedUser(t *testing.T) {
	srv := artifact.InMemoryService()
	for _, file := range []struct {
		sessionID string
		fileName  string
	}{
		{"user", "private.txt"},
		{"user", "private.txt"}, // Multiple versions must not duplicate filenames.
		{"user", "user:shared.txt"},
		{"other-session", "user:shared.txt"},
		{"other-session", "other.txt"},
	} {
		if _, err := srv.Save(t.Context(), &artifact.SaveRequest{
			AppName: "app", UserID: "u", SessionID: file.sessionID,
			FileName: file.fileName, Part: genai.NewPartFromText("fixture"),
		}); err != nil {
			t.Fatalf("Save failed: %T", err)
		}
	}

	for _, tc := range []struct {
		name      string
		sessionID string
		want      []string
	}{
		{
			name:      "owner retains private and shared artifacts",
			sessionID: "user",
			want:      []string{"private.txt", "user:shared.txt"},
		},
		{
			name:      "other session sees only its own and shared artifacts",
			sessionID: "other-session",
			want:      []string{"other.txt", "user:shared.txt"},
		},
		{
			name:      "empty session sees only shared artifacts",
			sessionID: "empty-session",
			want:      []string{"user:shared.txt"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := srv.List(t.Context(), &artifact.ListRequest{
				AppName: "app", UserID: "u", SessionID: tc.sessionID,
			})
			if err != nil {
				t.Fatalf("List failed: %T", err)
			}
			if !slices.Equal(resp.FileNames, tc.want) {
				t.Error("List must return only the session's own and user-scoped filenames, sorted and without duplicates")
			}
		})
	}
}
