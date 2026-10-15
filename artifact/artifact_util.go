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
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"google.golang.org/genai"
)

// artifactURIScheme is the URI scheme used to reference one artifact from
// another.
const artifactURIScheme = "artifact://"

// parsedArtifactURI is the result of parsing an artifact reference URI. It is
// unexported: the artifact package is the only consumer today, and a v2 module
// treats an exported name as a permanent commitment, so it can be exported later
// if a caller outside the package appears.
type parsedArtifactURI struct {
	AppName string
	UserID  string
	// SessionID is empty for a user-scoped artifact reference.
	SessionID string
	FileName  string
	Version   int64
}

// The artifact layout uses "/" as a delimiter, so a segment must not contain
// one. adk-python allows slashes inside a segment via a reserved-segment
// lookahead that its language's regex supports; Go's regexp does not, so the
// segments here match [^/]+ and are intentionally stricter than adk-python. An
// app/user/session ID that itself contains "/" (accepted elsewhere, e.g. the
// canonical-URI encoding) therefore cannot be named in a reference URI.
var (
	sessionScopedArtifactURIRE = regexp.MustCompile(
		`^artifact://apps/([^/]+)/users/([^/]+)/sessions/([^/]+)/artifacts/(.+)/versions/([1-9][0-9]*)$`,
	)
	userScopedArtifactURIRE = regexp.MustCompile(
		`^artifact://apps/([^/]+)/users/([^/]+)/artifacts/(.+)/versions/([1-9][0-9]*)$`,
	)
)

// parseArtifactURI parses an artifact reference URI of the form
// "artifact://apps/{app}/users/{user}/sessions/{session}/artifacts/{file}/versions/{version}"
// or its user-scoped variant (without the sessions segment). It reports whether
// uri was a well-formed artifact URI.
//
// A reference names a stored version, so the version segment requires a positive
// value ([1-9][0-9]*): in this package version 0 is the "latest" sentinel, and
// following a reference to "versions/0" would silently resolve to whatever the
// target's newest version is at load time. The regexes reject such a URI. The
// adk-python port accepts "versions/0" because Python has no such sentinel;
func parseArtifactURI(uri string) (parsedArtifactURI, bool) {
	if !strings.HasPrefix(uri, artifactURIScheme) {
		return parsedArtifactURI{}, false
	}

	if m := sessionScopedArtifactURIRE.FindStringSubmatch(uri); m != nil {
		version, err := strconv.ParseInt(m[5], 10, 64)
		if err != nil {
			return parsedArtifactURI{}, false
		}
		return parsedArtifactURI{
			AppName:   m[1],
			UserID:    m[2],
			SessionID: m[3],
			FileName:  m[4],
			Version:   version,
		}, true
	}

	if m := userScopedArtifactURIRE.FindStringSubmatch(uri); m != nil {
		version, err := strconv.ParseInt(m[4], 10, 64)
		if err != nil {
			return parsedArtifactURI{}, false
		}
		return parsedArtifactURI{
			AppName:  m[1],
			UserID:   m[2],
			FileName: m[3],
			Version:  version,
		}, true
	}

	return parsedArtifactURI{}, false
}

// buildArtifactURI constructs an artifact reference URI. If sessionID is empty,
// a user-scoped URI is constructed instead of a session-scoped one.
//
// The segments are not escaped, so an app, user, session, or file name that
// contains "/" is not round-trippable through parseArtifactURI for the same
// reason as above; call with segments that match [^/]+.
func buildArtifactURI(appName, userID, sessionID, fileName string, version int64) string {
	if sessionID == "" {
		return fmt.Sprintf("artifact://apps/%s/users/%s/artifacts/%s/versions/%d", appName, userID, fileName, version)
	}
	return fmt.Sprintf("artifact://apps/%s/users/%s/sessions/%s/artifacts/%s/versions/%d", appName, userID, sessionID, fileName, version)
}

// isArtifactRef reports whether part is an artifact reference, i.e. its
// FileData.FileURI starts with the "artifact://" scheme.
func isArtifactRef(part *genai.Part) bool {
	return part != nil && part.FileData != nil && strings.HasPrefix(part.FileData.FileURI, artifactURIScheme)
}

// validateArtifactReferenceScope ensures an artifact reference cannot escape the
// caller's app/user/session scope: parsed must belong to the same app and user
// as appName/userID, and if parsed is session-scoped, its session must match
// sessionID.
func validateArtifactReferenceScope(appName, userID, sessionID string, parsed parsedArtifactURI) error {
	if parsed.AppName != appName || parsed.UserID != userID {
		return fmt.Errorf("artifact reference must stay within the same app and user scope")
	}
	if parsed.SessionID != "" && parsed.SessionID != sessionID {
		return fmt.Errorf("session-scoped artifact reference must stay within the same session scope")
	}
	return nil
}
