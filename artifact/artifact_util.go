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
// another (see [ParseArtifactURI]).
const artifactURIScheme = "artifact://"

// ParsedArtifactURI is the result of parsing an artifact URI produced by
// [ParseArtifactURI].
type ParsedArtifactURI struct {
	AppName string
	UserID  string
	// SessionID is empty for a user-scoped artifact reference.
	SessionID string
	FileName  string
	Version   int64
}

var (
	sessionScopedArtifactURIRE = regexp.MustCompile(
		`^artifact://apps/([^/]+)/users/([^/]+)/sessions/([^/]+)/artifacts/(.+)/versions/(\d+)$`,
	)
	userScopedArtifactURIRE = regexp.MustCompile(
		`^artifact://apps/([^/]+)/users/([^/]+)/artifacts/(.+)/versions/(\d+)$`,
	)
)

// ParseArtifactURI parses an artifact reference URI of the form
// "artifact://apps/{app}/users/{user}/sessions/{session}/artifacts/{file}/versions/{version}"
// or its user-scoped variant (without the sessions segment). It reports
// whether uri was a well-formed artifact URI.
func ParseArtifactURI(uri string) (ParsedArtifactURI, bool) {
	if !strings.HasPrefix(uri, artifactURIScheme) {
		return ParsedArtifactURI{}, false
	}

	if m := sessionScopedArtifactURIRE.FindStringSubmatch(uri); m != nil {
		version, err := strconv.ParseInt(m[5], 10, 64)
		if err != nil {
			return ParsedArtifactURI{}, false
		}
		return ParsedArtifactURI{
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
			return ParsedArtifactURI{}, false
		}
		return ParsedArtifactURI{
			AppName:  m[1],
			UserID:   m[2],
			FileName: m[3],
			Version:  version,
		}, true
	}

	return ParsedArtifactURI{}, false
}

// BuildArtifactURI constructs an artifact reference URI. If sessionID is
// empty, a user-scoped URI is constructed instead of a session-scoped one.
func BuildArtifactURI(appName, userID, sessionID, fileName string, version int64) string {
	if sessionID == "" {
		return fmt.Sprintf("artifact://apps/%s/users/%s/artifacts/%s/versions/%d", appName, userID, fileName, version)
	}
	return fmt.Sprintf("artifact://apps/%s/users/%s/sessions/%s/artifacts/%s/versions/%d", appName, userID, sessionID, fileName, version)
}

// IsArtifactRef reports whether part is an artifact reference, i.e. its
// FileData.FileURI starts with the "artifact://" scheme.
func IsArtifactRef(part *genai.Part) bool {
	return part != nil && part.FileData != nil && strings.HasPrefix(part.FileData.FileURI, artifactURIScheme)
}

// ValidateArtifactReferenceScope ensures an artifact reference cannot escape
// the caller's app/user/session scope: parsed must belong to the same app
// and user as appName/userID, and if parsed is session-scoped, its session
// must match sessionID.
func ValidateArtifactReferenceScope(appName, userID, sessionID string, parsed ParsedArtifactURI) error {
	if parsed.AppName != appName || parsed.UserID != userID {
		return fmt.Errorf("artifact reference must stay within the same app and user scope")
	}
	if parsed.SessionID != "" && parsed.SessionID != sessionID {
		return fmt.Errorf("session-scoped artifact reference must stay within the same session scope")
	}
	return nil
}
