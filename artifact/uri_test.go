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

package artifact_test

import (
	"testing"

	"google.golang.org/adk/v2/artifact"
)

func TestBuildAndParseArtifactURI(t *testing.T) {
	roundTrips := []struct {
		name    string
		want    artifact.ParsedArtifactURI
		wantURI string
	}{
		{
			name:    "session-scoped with version 0",
			want:    artifact.ParsedArtifactURI{AppName: "app1", UserID: "user1", SessionID: "sess1", FileName: "file.txt", Version: 0},
			wantURI: "artifact://apps/app1/users/user1/sessions/sess1/artifacts/file.txt/versions/0",
		},
		{
			name:    "user-scoped with version 0",
			want:    artifact.ParsedArtifactURI{AppName: "app1", UserID: "user1", FileName: "file.txt", Version: 0},
			wantURI: "artifact://apps/app1/users/user1/artifacts/file.txt/versions/0",
		},
		{
			name: "namespaced segments",
			want: artifact.ParsedArtifactURI{
				AppName: "projects/123/locations/us-central1/reasoningEngines/456", UserID: "group/user123",
				SessionID: "team/sess123", FileName: "folder/file.txt", Version: 42,
			},
			wantURI: "artifact://apps/projects/123/locations/us-central1/reasoningEngines/456/users/group/user123/sessions/team/sess123/artifacts/folder/file.txt/versions/42",
		},
		{
			name:    "segments starting with a reserved word",
			want:    artifact.ParsedArtifactURI{AppName: "apps/sub", UserID: "users/u1", SessionID: "sessions/s1", FileName: "f.txt", Version: 1},
			wantURI: "artifact://apps/apps/sub/users/users/u1/sessions/sessions/s1/artifacts/f.txt/versions/1",
		},
		{
			name:    "session-scoped file name containing artifacts",
			want:    artifact.ParsedArtifactURI{AppName: "app1", UserID: "user1", SessionID: "sess1", FileName: "sub/artifacts/file.txt", Version: 1},
			wantURI: "artifact://apps/app1/users/user1/sessions/sess1/artifacts/sub/artifacts/file.txt/versions/1",
		},
		{
			name:    "session-scoped file name containing sessions",
			want:    artifact.ParsedArtifactURI{AppName: "app1", UserID: "user1", SessionID: "sess1", FileName: "a/sessions/b.txt", Version: 1},
			wantURI: "artifact://apps/app1/users/user1/sessions/sess1/artifacts/a/sessions/b.txt/versions/1",
		},
		{
			name:    "user-scoped file name containing artifacts",
			want:    artifact.ParsedArtifactURI{AppName: "app1", UserID: "user1", FileName: "sub/artifacts/file.txt", Version: 1},
			wantURI: "artifact://apps/app1/users/user1/artifacts/sub/artifacts/file.txt/versions/1",
		},
		{
			name:    "user-scoped file name starting with sessions",
			want:    artifact.ParsedArtifactURI{AppName: "app1", UserID: "user1", FileName: "sessions/sess1/artifacts/data.txt", Version: 123},
			wantURI: "artifact://apps/app1/users/user1/artifacts/sessions/sess1/artifacts/data.txt/versions/123",
		},
		{
			name:    "user-scoped file name containing sessions and artifacts",
			want:    artifact.ParsedArtifactURI{AppName: "app1", UserID: "user1", FileName: "a/sessions/b/artifacts/c.txt", Version: 1},
			wantURI: "artifact://apps/app1/users/user1/artifacts/a/sessions/b/artifacts/c.txt/versions/1",
		},
		{
			name:    "file name containing versions",
			want:    artifact.ParsedArtifactURI{AppName: "app1", UserID: "user1", SessionID: "sess1", FileName: "a/versions/7", Version: 3},
			wantURI: "artifact://apps/app1/users/user1/sessions/sess1/artifacts/a/versions/7/versions/3",
		},
	}
	for _, tc := range roundTrips {
		t.Run(tc.name, func(t *testing.T) {
			uri := artifact.BuildArtifactURI(tc.want.AppName, tc.want.UserID, tc.want.SessionID, tc.want.FileName, tc.want.Version)
			if uri != tc.wantURI {
				t.Errorf("BuildArtifactURI() = %q, want %q", uri, tc.wantURI)
			}
			got, ok := artifact.ParseArtifactURI(tc.wantURI)
			if !ok || got != tc.want {
				t.Errorf("ParseArtifactURI(%q) = (%+v, %v), want (%+v, true)", tc.wantURI, got, ok, tc.want)
			}
		})
	}

	rejections := []struct {
		name string
		uri  string
	}{
		{name: "empty", uri: ""},
		{name: "wrong scheme", uri: "https://example.com/file.txt"},
		{name: "missing apps", uri: "artifact://invalid"},
		{name: "missing artifacts", uri: "artifact://apps/a/users/u/versions/1"},
		{name: "empty app", uri: "artifact://apps//users/u/artifacts/f/versions/1"},
		{name: "empty user", uri: "artifact://apps/a/users//artifacts/f/versions/1"},
		{name: "empty session", uri: "artifact://apps/a/users/u/sessions//artifacts/f/versions/1"},
		{name: "empty file name", uri: "artifact://apps/a/users/u/artifacts//versions/1"},
		{name: "empty version", uri: "artifact://apps/a/users/u/artifacts/f/versions/"},
		{name: "negative version", uri: "artifact://apps/a/users/u/artifacts/f/versions/-1"},
		{name: "signed version", uri: "artifact://apps/a/users/u/artifacts/f/versions/+1"},
		{name: "non-numeric version", uri: "artifact://apps/a/users/u/artifacts/f/versions/latest"},
		{name: "version overflows int64", uri: "artifact://apps/a/users/u/artifacts/f/versions/99999999999999999999"},
		{name: "non-ASCII digits in version", uri: "artifact://apps/a/users/u/artifacts/f/versions/١٢"},
		{name: "trailing segment", uri: "artifact://apps/a/users/u/artifacts/f/versions/1/extra"},
		{name: "reserved users inside session", uri: "artifact://apps/a/users/u/sessions/s/users/u2/artifacts/f/versions/1"},
		{name: "reserved versions inside user", uri: "artifact://apps/a/users/u/versions/v/artifacts/f/versions/1"},
		{name: "reserved apps at end of app", uri: "artifact://apps/sub/apps/users/u/artifacts/f/versions/1"},
		{name: "newline in app", uri: "artifact://apps/app1\napp2/users/user1/artifacts/file.txt/versions/1"},
		{name: "newline in user", uri: "artifact://apps/app1/users/user1\nuser2/artifacts/file.txt/versions/1"},
		{name: "newline in session", uri: "artifact://apps/app1/users/user1/sessions/s1\ns2/artifacts/file.txt/versions/1"},
		{name: "newline in file name", uri: "artifact://apps/a/users/u/artifacts/line1\nline2/versions/1"},
	}
	for _, tc := range rejections {
		t.Run(tc.name, func(t *testing.T) {
			if got, ok := artifact.ParseArtifactURI(tc.uri); ok || got != (artifact.ParsedArtifactURI{}) {
				t.Errorf("ParseArtifactURI(%q) = (%+v, %v), want zero value and false", tc.uri, got, ok)
			}
		})
	}
}
