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

package artifact

import (
	"errors"
	"testing"

	"google.golang.org/genai"
)

func TestParseArtifactURI(t *testing.T) {
	testCases := []struct {
		name   string
		uri    string
		want   parsedArtifactURI
		wantOK bool
	}{
		{
			name: "session-scoped",
			uri:  "artifact://apps/myapp/users/user-1/sessions/sess-1/artifacts/report.pdf/versions/3",
			want: parsedArtifactURI{
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
			want: parsedArtifactURI{
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
		{
			// A reference must name a stored version; 0 is the "latest" sentinel.
			name:   "zero version rejected",
			uri:    "artifact://apps/myapp/users/user-1/sessions/sess-1/artifacts/report.pdf/versions/0",
			wantOK: false,
		},
		{
			// Segments are [^/]+; a path separator inside app/user/session is
			// not representable in a reference URI (stricter than adk-python).
			name:   "slash in app segment not representable",
			uri:    "artifact://apps/my/app/users/user-1/sessions/sess-1/artifacts/report.pdf/versions/1",
			wantOK: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := parseArtifactURI(tc.uri)
			if ok != tc.wantOK {
				t.Fatalf("parseArtifactURI(%q) ok = %v, want %v", tc.uri, ok, tc.wantOK)
			}
			if ok && got != tc.want {
				t.Errorf("parseArtifactURI(%q) = %+v, want %+v", tc.uri, got, tc.want)
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
			got := buildArtifactURI(tc.appName, tc.userID, tc.sessionID, tc.fileName, tc.version)
			if got != tc.want {
				t.Errorf("buildArtifactURI(...) = %q, want %q", got, tc.want)
			}
			parsed, ok := parseArtifactURI(got)
			if !ok {
				t.Fatalf("parseArtifactURI(%q) failed to parse output of buildArtifactURI", got)
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
			if got := isArtifactRef(tc.part); got != tc.want {
				t.Errorf("isArtifactRef(%+v) = %v, want %v", tc.part, got, tc.want)
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
		parsed    parsedArtifactURI
		wantErr   bool
	}{
		{
			name:      "same scope, session-scoped",
			appName:   "app",
			userID:    "user",
			sessionID: "sess",
			parsed:    parsedArtifactURI{AppName: "app", UserID: "user", SessionID: "sess", FileName: "f", Version: 1},
			wantErr:   false,
		},
		{
			name:      "same app/user, ref is user-scoped",
			appName:   "app",
			userID:    "user",
			sessionID: "sess",
			parsed:    parsedArtifactURI{AppName: "app", UserID: "user", FileName: "user:f", Version: 1},
			wantErr:   false,
		},
		{
			name:      "cross-app escape",
			appName:   "app",
			userID:    "user",
			sessionID: "sess",
			parsed:    parsedArtifactURI{AppName: "other-app", UserID: "user", SessionID: "sess", FileName: "f", Version: 1},
			wantErr:   true,
		},
		{
			name:      "cross-user escape",
			appName:   "app",
			userID:    "user",
			sessionID: "sess",
			parsed:    parsedArtifactURI{AppName: "app", UserID: "other-user", SessionID: "sess", FileName: "f", Version: 1},
			wantErr:   true,
		},
		{
			name:      "cross-session escape",
			appName:   "app",
			userID:    "user",
			sessionID: "sess",
			parsed:    parsedArtifactURI{AppName: "app", UserID: "user", SessionID: "other-sess", FileName: "f", Version: 1},
			wantErr:   true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateArtifactReferenceScope(tc.appName, tc.userID, tc.sessionID, tc.parsed)
			if (err != nil) != tc.wantErr {
				t.Errorf("validateArtifactReferenceScope(...) err = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func TestIsRewindTombstone(t *testing.T) {
	testCases := []struct {
		name string
		part *genai.Part
		want bool
	}{
		{name: "marker with empty data", part: genai.NewPartFromBytes([]byte{}, "application/octet-stream"), want: true},
		{name: "marker with nil data", part: &genai.Part{InlineData: &genai.Blob{MIMEType: "application/octet-stream"}}, want: true},
		{name: "non-empty data", part: genai.NewPartFromBytes([]byte("a"), "application/octet-stream"), want: false},
		{name: "other MIME type", part: genai.NewPartFromBytes([]byte{}, "application/pdf"), want: false},
		{
			name: "display name set",
			part: &genai.Part{InlineData: &genai.Blob{MIMEType: "application/octet-stream", Data: []byte{}, DisplayName: "empty.bin"}},
			want: false,
		},
		{
			name: "marker plus thought flag",
			part: &genai.Part{InlineData: &genai.Blob{MIMEType: "application/octet-stream"}, Thought: true},
			want: false,
		},
		{name: "empty text", part: genai.NewPartFromText(""), want: false},
		{name: "nil", part: nil, want: false},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsRewindTombstone(tc.part); got != tc.want {
				t.Errorf("IsRewindTombstone(%+v) = %v, want %v", tc.part, got, tc.want)
			}
		})
	}
}

func TestNewRewindTombstone(t *testing.T) {
	first := NewRewindTombstone()
	if !IsRewindTombstone(first) {
		t.Fatalf("IsRewindTombstone(NewRewindTombstone()) = false, want true")
	}
	req := &SaveRequest{AppName: "app", UserID: "user", SessionID: "session", FileName: "file.txt", Part: first}
	if err := req.Validate(); err != nil {
		t.Errorf("SaveRequest.Validate() with NewRewindTombstone() = %v, want nil", err)
	}

	// A caller that changes the part it got must not change the next marker.
	first.InlineData.MIMEType = "text/plain"
	first.InlineData.Data = []byte("changed")
	if second := NewRewindTombstone(); !IsRewindTombstone(second) {
		t.Errorf("NewRewindTombstone() after changing an earlier result = %+v, want a marker", second.InlineData)
	}
}

func TestValidatePathSegment(t *testing.T) {
	testCases := []struct {
		name      string
		value     string
		fieldName string
		// want is the exact error text; empty means value is valid.
		want string
	}{
		{name: "plain", value: "app1", fieldName: "app_name"},
		{name: "slash allowed", value: "group/user123", fieldName: "user_id"},
		{name: "resource name", value: "projects/123/locations/us-central1/reasoningEngines/456", fieldName: "session_id"},
		{name: "backslash allowed", value: "back\\slash", fieldName: "user_id"},
		{name: "bare users", value: "users", fieldName: "user_id"},
		{name: "bare apps", value: "apps", fieldName: "app_name"},
		{name: "bare sessions", value: "sessions", fieldName: "session_id"},
		{name: "bare artifacts", value: "artifacts", fieldName: "app_name"},
		{name: "bare versions", value: "versions", fieldName: "user_id"},
		{name: "dot segment inside a path", value: "a/.", fieldName: "user_id"},
		{name: "dots inside a segment", value: "a..b/c", fieldName: "user_id"},
		{name: "colon after two letters", value: "ab:1", fieldName: "user_id"},
		{name: "colon after a digit", value: "1:a", fieldName: "user_id"},
		{name: "colon after an underscore", value: "_:x", fieldName: "user_id"},
		{name: "session named user", value: "user", fieldName: "session_id"},
		{name: "session under user", value: "user/x", fieldName: "session_id"},
		{name: "folded word without a slash", value: "seßions", fieldName: "user_id"},
		{name: "dotted capital I does not fold to i", value: "x/versİons", fieldName: "user_id"},
		{name: "empty", value: "", fieldName: "user_id", want: "user_id must not be empty."},
		{name: "null byte", value: "null\x00byte", fieldName: "app_name", want: "app_name must not contain null bytes."},
		{name: "null byte before leading slash", value: "/\x00", fieldName: "app_name", want: "app_name must not contain null bytes."},
		{name: "leading slash", value: "/leading/slash", fieldName: "app_name", want: "app_name must not be an absolute path or start with a slash."},
		{name: "leading backslash", value: "\\x", fieldName: "user_id", want: "user_id must not be an absolute path or start with a slash."},
		{name: "leading slash before traversal", value: "/..", fieldName: "app_name", want: "app_name must not be an absolute path or start with a slash."},
		{name: "drive letter", value: "C:x", fieldName: "user_id", want: "user_id must not be drive-qualified."},
		{name: "lower-case drive letter", value: "z:\\abs", fieldName: "session_id", want: "session_id must not be drive-qualified."},
		{name: "drive before traversal", value: "C:/..", fieldName: "app_name", want: "app_name must not be drive-qualified."},
		{name: "bare drive", value: "C:", fieldName: "user_id", want: "user_id must not be drive-qualified."},
		{name: "drive letter a", value: "a:1", fieldName: "user_id", want: "user_id must not be drive-qualified."},
		{name: "drive letter A", value: "A:1", fieldName: "user_id", want: "user_id must not be drive-qualified."},
		{name: "drive letter Z", value: "Z:1", fieldName: "user_id", want: "user_id must not be drive-qualified."},
		{name: "dot", value: ".", fieldName: "user_id", want: "user_id must not contain traversal segments."},
		{name: "dot dot", value: "..", fieldName: "user_id", want: "user_id must not contain traversal segments."},
		{name: "leading traversal", value: "../../etc", fieldName: "session_id", want: "session_id must not contain traversal segments."},
		{name: "trailing traversal", value: "a/..", fieldName: "user_id", want: "user_id must not contain traversal segments."},
		{name: "traversal between backslashes", value: "mixed/..\\separators", fieldName: "user_id", want: "user_id must not contain traversal segments."},
		{name: "traversal before reserved", value: "../apps", fieldName: "app_name", want: "app_name must not contain traversal segments."},
		{name: "reserved apps", value: "victim/apps", fieldName: "session_id", want: "session_id must not contain reserved path segments."},
		{name: "reserved upper case", value: "a/APPS/b", fieldName: "user_id", want: "user_id must not contain reserved path segments."},
		{name: "reserved users", value: "victim/users", fieldName: "app_name", want: "app_name must not contain reserved path segments."},
		{name: "reserved sessions", value: "victim/sessions/1", fieldName: "user_id", want: "user_id must not contain reserved path segments."},
		{name: "reserved artifacts", value: "victim/artifacts", fieldName: "session_id", want: "session_id must not contain reserved path segments."},
		{name: "reserved versions", value: "victim/versions", fieldName: "session_id", want: "session_id must not contain reserved path segments."},
		{name: "reserved between backslashes", value: "back\\users\\x", fieldName: "user_id", want: "user_id must not contain reserved path segments."},
		{name: "long s folds to s", value: "a/ſessions", fieldName: "user_id", want: "user_id must not contain reserved path segments."},
		{name: "sharp s folds to ss", value: "x/seßions", fieldName: "user_id", want: "user_id must not contain reserved path segments."},
		{name: "long s in users", value: "a/uſers", fieldName: "user_id", want: "user_id must not contain reserved path segments."},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidatePathSegment(tc.value, tc.fieldName)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("ValidatePathSegment() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("ValidatePathSegment() = nil, want %q", tc.want)
			}
			if !errors.Is(err, ErrInputValidation) {
				t.Errorf("errors.Is(%v, ErrInputValidation) = false, want true", err)
			}
			if err.Error() != tc.want {
				t.Errorf("ValidatePathSegment() error = %q, want %q", err.Error(), tc.want)
			}
		})
	}
}

// TestParseAndValidateReferenceURI checks the one rule set that every service
// applies before following an artifact reference: the hop budget, the strict
// URI form and the caller's scope, in adk-python's order
// (artifact_util.resolve_artifact_reference).
func TestParseAndValidateReferenceURI(t *testing.T) {
	const (
		depthMsg   = "artifact reference resolution exceeded maximum depth of 5 (Exceeded maximum recursion depth resolving artifact reference)"
		invalidMsg = "invalid artifact reference URI (Invalid artifact reference URI)"
		appUserMsg = "invalid artifact reference: artifact reference must stay within the same app and user scope"
		sessionMsg = "invalid artifact reference: session-scoped artifact reference must stay within the same session scope"
	)
	testCases := []struct {
		name           string
		uri            string
		remainingDepth int
		want           ParsedArtifactURI
		wantMsg        string
	}{
		{
			name:           "session-scoped reference in scope",
			uri:            "artifact://apps/app/users/user1/sessions/sess1/artifacts/t.txt/versions/3",
			remainingDepth: 5,
			want:           ParsedArtifactURI{AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "t.txt", Version: 3},
		},
		{
			name:           "user-scoped reference from any session",
			uri:            "artifact://apps/app/users/user1/artifacts/user:n.txt/versions/1",
			remainingDepth: 5,
			want:           ParsedArtifactURI{AppName: "app", UserID: "user1", FileName: "user:n.txt", Version: 1},
		},
		{
			name:           "last hop of the budget",
			uri:            "artifact://apps/app/users/user1/sessions/sess1/artifacts/t.txt/versions/1",
			remainingDepth: 1,
			want:           ParsedArtifactURI{AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "t.txt", Version: 1},
		},
		{
			name:           "budget exhausted",
			uri:            "artifact://apps/app/users/user1/sessions/sess1/artifacts/zz-secret-zz/versions/1",
			remainingDepth: 0,
			wantMsg:        depthMsg,
		},
		{
			name:           "budget exhausted is reported before a malformed URI",
			uri:            "artifact://apps/app/invalid",
			remainingDepth: 0,
			wantMsg:        depthMsg,
		},
		{
			name:           "malformed URI",
			uri:            "artifact://apps/zz-secret-zz/invalid",
			remainingDepth: 5,
			wantMsg:        invalidMsg,
		},
		{
			name:           "not an artifact URI",
			uri:            "gs://zz-secret-zz/o",
			remainingDepth: 5,
			wantMsg:        invalidMsg,
		},
		{
			name:           "version 0",
			uri:            "artifact://apps/app/users/user1/sessions/sess1/artifacts/t.txt/versions/0",
			remainingDepth: 5,
			wantMsg:        invalidMsg,
		},
		{
			name:           "other app",
			uri:            "artifact://apps/zz-secret-zz/users/user1/sessions/sess1/artifacts/t.txt/versions/1",
			remainingDepth: 5,
			wantMsg:        appUserMsg,
		},
		{
			name:           "other user",
			uri:            "artifact://apps/app/users/zz-secret-zz/artifacts/user:n.txt/versions/1",
			remainingDepth: 5,
			wantMsg:        appUserMsg,
		},
		{
			name:           "other session",
			uri:            "artifact://apps/app/users/user1/sessions/zz-secret-zz/artifacts/t.txt/versions/1",
			remainingDepth: 5,
			wantMsg:        sessionMsg,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseAndValidateReferenceURI(tc.uri, "app", "user1", "sess1", tc.remainingDepth)
			if tc.wantMsg == "" {
				if err != nil {
					t.Fatalf("ParseAndValidateReferenceURI() error = %v, want nil", err)
				}
				if got != tc.want {
					t.Errorf("ParseAndValidateReferenceURI() = %+v, want %+v", got, tc.want)
				}
				return
			}
			if !errors.Is(err, ErrInputValidation) {
				t.Fatalf("ParseAndValidateReferenceURI() error = %v, want an error wrapping ErrInputValidation", err)
			}
			if err.Error() != tc.wantMsg {
				t.Errorf("ParseAndValidateReferenceURI() error = %q, want %q", err.Error(), tc.wantMsg)
			}
			if got != (ParsedArtifactURI{}) {
				t.Errorf("ParseAndValidateReferenceURI() = %+v on error, want the zero value", got)
			}
		})
	}
}

// TestParseAndValidateReferenceURISessionlessCaller checks that a caller with
// no session, as a "user:" artifact saved without one, may reference only
// user-scoped artifacts.
func TestParseAndValidateReferenceURISessionlessCaller(t *testing.T) {
	_, err := ParseAndValidateReferenceURI("artifact://apps/app/users/user1/sessions/s1/artifacts/t.txt/versions/1", "app", "user1", "", MaxArtifactReferenceDepth)
	if !errors.Is(err, ErrInputValidation) {
		t.Fatalf("ParseAndValidateReferenceURI() error = %v, want an error wrapping ErrInputValidation", err)
	}
	got, err := ParseAndValidateReferenceURI("artifact://apps/app/users/user1/artifacts/user:n.txt/versions/2", "app", "user1", "", MaxArtifactReferenceDepth)
	if err != nil {
		t.Fatalf("ParseAndValidateReferenceURI() error = %v, want nil", err)
	}
	if want := (ParsedArtifactURI{AppName: "app", UserID: "user1", FileName: "user:n.txt", Version: 2}); got != want {
		t.Errorf("ParseAndValidateReferenceURI() = %+v, want %+v", got, want)
	}
}
