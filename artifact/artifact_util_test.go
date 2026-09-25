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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
)

func TestParseArtifactURI(t *testing.T) {
	testCases := []struct {
		name   string
		uri    string
		want   artifact.ParsedArtifactURI
		wantOK bool
	}{
		{
			name: "session-scoped",
			uri:  "artifact://apps/myapp/users/user-1/sessions/sess-1/artifacts/report.pdf/versions/3",
			want: artifact.ParsedArtifactURI{
				AppName:   "myapp",
				UserID:    "user-1",
				SessionID: "sess-1",
				FileName:  "report.pdf",
				Version:   3,
			},
			wantOK: true,
		},
		{
			name: "user-scoped",
			uri:  "artifact://apps/myapp/users/user-1/artifacts/user:profile.png/versions/1",
			want: artifact.ParsedArtifactURI{
				AppName:  "myapp",
				UserID:   "user-1",
				FileName: "user:profile.png",
				Version:  1,
			},
			wantOK: true,
		},
		{
			name:   "not an artifact URI",
			uri:    "https://example.com/file.txt",
			wantOK: false,
		},
		{
			name:   "empty",
			uri:    "",
			wantOK: false,
		},
		{
			name:   "missing version",
			uri:    "artifact://apps/myapp/users/user-1/sessions/sess-1/artifacts/report.pdf/versions/",
			wantOK: false,
		},
		{
			name:   "non-numeric version",
			uri:    "artifact://apps/myapp/users/user-1/sessions/sess-1/artifacts/report.pdf/versions/latest",
			wantOK: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := artifact.ParseArtifactURI(tc.uri)
			if ok != tc.wantOK {
				t.Fatalf("ParseArtifactURI(%q) ok = %v, want %v", tc.uri, ok, tc.wantOK)
			}
			if ok && got != tc.want {
				t.Errorf("ParseArtifactURI(%q) = %+v, want %+v", tc.uri, got, tc.want)
			}
		})
	}
}

func TestBuildArtifactURI(t *testing.T) {
	testCases := []struct {
		name      string
		appName   string
		userID    string
		sessionID string
		fileName  string
		version   int64
		want      string
	}{
		{
			name:      "session-scoped",
			appName:   "myapp",
			userID:    "user-1",
			sessionID: "sess-1",
			fileName:  "report.pdf",
			version:   3,
			want:      "artifact://apps/myapp/users/user-1/sessions/sess-1/artifacts/report.pdf/versions/3",
		},
		{
			name:     "user-scoped",
			appName:  "myapp",
			userID:   "user-1",
			fileName: "user:profile.png",
			version:  1,
			want:     "artifact://apps/myapp/users/user-1/artifacts/user:profile.png/versions/1",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := artifact.BuildArtifactURI(tc.appName, tc.userID, tc.sessionID, tc.fileName, tc.version)
			if got != tc.want {
				t.Errorf("BuildArtifactURI(...) = %q, want %q", got, tc.want)
			}
			parsed, ok := artifact.ParseArtifactURI(got)
			if !ok {
				t.Fatalf("ParseArtifactURI(%q) failed to parse output of BuildArtifactURI", got)
			}
			if parsed.AppName != tc.appName || parsed.UserID != tc.userID ||
				parsed.SessionID != tc.sessionID || parsed.FileName != tc.fileName ||
				parsed.Version != tc.version {
				t.Errorf("round trip mismatch: got %+v", parsed)
			}
		})
	}
}

func TestIsArtifactRef(t *testing.T) {
	testCases := []struct {
		name string
		part *genai.Part
		want bool
	}{
		{
			name: "artifact ref",
			part: &genai.Part{FileData: &genai.FileData{FileURI: "artifact://apps/a/users/u/artifacts/f/versions/1"}},
			want: true,
		},
		{
			name: "non-artifact file data",
			part: &genai.Part{FileData: &genai.FileData{FileURI: "gs://bucket/file.txt"}},
			want: false,
		},
		{
			name: "text part",
			part: genai.NewPartFromText("hello"),
			want: false,
		},
		{
			name: "nil part",
			part: nil,
			want: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := artifact.IsArtifactRef(tc.part); got != tc.want {
				t.Errorf("IsArtifactRef(%+v) = %v, want %v", tc.part, got, tc.want)
			}
		})
	}
}

func TestValidateArtifactReferenceScope(t *testing.T) {
	testCases := []struct {
		name      string
		appName   string
		userID    string
		sessionID string
		parsed    artifact.ParsedArtifactURI
		wantErr   bool
	}{
		{
			name:      "same scope, session-scoped",
			appName:   "app",
			userID:    "user",
			sessionID: "sess",
			parsed:    artifact.ParsedArtifactURI{AppName: "app", UserID: "user", SessionID: "sess", FileName: "f", Version: 1},
			wantErr:   false,
		},
		{
			name:      "same app/user, ref is user-scoped",
			appName:   "app",
			userID:    "user",
			sessionID: "sess",
			parsed:    artifact.ParsedArtifactURI{AppName: "app", UserID: "user", FileName: "user:f", Version: 1},
			wantErr:   false,
		},
		{
			name:      "cross-app escape",
			appName:   "app",
			userID:    "user",
			sessionID: "sess",
			parsed:    artifact.ParsedArtifactURI{AppName: "other-app", UserID: "user", SessionID: "sess", FileName: "f", Version: 1},
			wantErr:   true,
		},
		{
			name:      "cross-user escape",
			appName:   "app",
			userID:    "user",
			sessionID: "sess",
			parsed:    artifact.ParsedArtifactURI{AppName: "app", UserID: "other-user", SessionID: "sess", FileName: "f", Version: 1},
			wantErr:   true,
		},
		{
			name:      "cross-session escape",
			appName:   "app",
			userID:    "user",
			sessionID: "sess",
			parsed:    artifact.ParsedArtifactURI{AppName: "app", UserID: "user", SessionID: "other-sess", FileName: "f", Version: 1},
			wantErr:   true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := artifact.ValidateArtifactReferenceScope(tc.appName, tc.userID, tc.sessionID, tc.parsed)
			if (err != nil) != tc.wantErr {
				t.Errorf("ValidateArtifactReferenceScope(...) err = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}
