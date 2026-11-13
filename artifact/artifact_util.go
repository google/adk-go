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
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"golang.org/x/text/cases"
	"google.golang.org/genai"
)

// artifactURIScheme is the URI scheme used to reference one artifact from
// another.
const artifactURIScheme = "artifact://"

// parsedArtifactURI is the result of parseArtifactURI, the stricter parser that
// saving and loading an artifact reference use. [ParseArtifactURI] is the
// exported parser with adk-python's rules.
type parsedArtifactURI struct {
	AppName string
	UserID  string
	// SessionID is empty for a user-scoped artifact reference.
	SessionID string
	FileName  string
	Version   int64
}

// sessionScopedArtifactURIRE and userScopedArtifactURIRE are the patterns of
// parseArtifactURI, and so of the reference check in
// [ParseAndValidateReferenceURI]. Each app, user and session segment matches
// [^/]+. A reference therefore cannot name an app, user or session ID that
// contains "/", such as the user ID "team/alice": saving or following it fails
// with "Invalid artifact reference URI", although [ValidatePathSegment] accepts
// the ID, [BuildArtifactURI] builds a URI for it, and adk-python's resolver
// follows the same reference. [ParseArtifactURI] accepts such IDs, because it
// ends a segment only at "/" followed by a reserved word and "/".
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
// target's newest version is at load time. The regexes reject such a URI.
// adk-python accepts "versions/0" because Python has no such sentinel, so this
// parser is deliberately stricter.
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
// The app, user, and session segments are not escaped, so a value containing
// "/" cannot be represented in a reference URI for the reason above. The file
// name is matched by "(.+)", and the "/versions/{version}" suffix is anchored
// at the end of the string, so a file name containing "/" round-trips through
// parseArtifactURI even though SaveRequest.Validate rejects a path separator in
// a file name. A name containing a newline does not round-trip, because "."
// does not match "\n", although ValidateFileName accepts such a name.
func buildArtifactURI(appName, userID, sessionID, fileName string, version int64) string {
	return BuildArtifactURI(appName, userID, sessionID, fileName, version)
}

// ParsedArtifactURI holds parsed URI components. The zero value has empty strings and Version: 0.
type ParsedArtifactURI struct {
	AppName, UserID, SessionID, FileName string
	Version                              int64
}

// ParseArtifactURI parses an artifact URI, supporting slashes in app/user/session segments.
// Version is treated as an opaque non-negative integer. On failure, returns the zero value
// of ParsedArtifactURI and false. Safe for concurrent use. Returns ParsedArtifactURI{}, false
// for any URI that contains "\n", in any segment.
func ParseArtifactURI(uri string) (ParsedArtifactURI, bool) {
	// adk-python's patterns use "." without re.DOTALL, so no part of a URI may
	// hold a newline.
	if strings.Contains(uri, "\n") {
		return ParsedArtifactURI{}, false
	}
	rest, ok := strings.CutPrefix(uri, artifactURIScheme+"apps/")
	if !ok {
		return ParsedArtifactURI{}, false
	}
	appName, rest, next := cutSegment(rest)
	if appName == "" || next != "users" {
		return ParsedArtifactURI{}, false
	}
	userID, rest, next := cutSegment(rest)
	if userID == "" {
		return ParsedArtifactURI{}, false
	}
	var sessionID string
	if next == "sessions" {
		sessionID, rest, next = cutSegment(rest)
		if sessionID == "" {
			return ParsedArtifactURI{}, false
		}
	}
	if next != "artifacts" {
		return ParsedArtifactURI{}, false
	}
	// The file name may hold any reserved word, so it ends at the last
	// "/versions/", the only one a digits-only version can follow.
	i := strings.LastIndex(rest, "/versions/")
	if i <= 0 {
		return ParsedArtifactURI{}, false
	}
	fileName, digits := rest[:i], rest[i+len("/versions/"):]
	version, ok := parseASCIIDigits(digits)
	if !ok {
		return ParsedArtifactURI{}, false
	}
	return ParsedArtifactURI{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Version: version}, true
}

// reservedURISegments are the words that delimit the segments of an artifact
// URI. adk-python's _RESERVED_PATH_SEGMENTS.
var reservedURISegments = []string{"apps", "users", "sessions", "artifacts", "versions"}

// cutSegment splits s at the first "/{reserved}/" in it, returning the text
// before it as the segment, the text after it as rest, and the reserved word
// as next. When s holds no reserved delimiter, next is empty.
//
// This is adk-python's lazy segment pattern with its lookahead
// (?!/(?:apps|artifacts|sessions|users|versions)/), which RE2 lacks: a segment
// may hold "/" and may start with a reserved word ("apps/sub"), but may not
// contain "/{reserved}/" anywhere, its closing "/" included ("sub/apps").
func cutSegment(s string) (segment, rest, next string) {
	for i := 0; i < len(s); i++ {
		if s[i] != '/' {
			continue
		}
		for _, word := range reservedURISegments {
			if after, ok := strings.CutPrefix(s[i+1:], word+"/"); ok {
				return s[:i], after, word
			}
		}
	}
	return s, "", ""
}

// parseASCIIDigits parses s as a non-negative decimal int64 written in ASCII
// digits only. strconv.ParseInt alone would accept a sign, and adk-python's
// \d would accept non-ASCII digits.
func parseASCIIDigits(s string) (int64, bool) {
	if s == "" {
		return 0, false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return 0, false
		}
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// BuildArtifactURI builds an artifact URI. If sessionID is empty, returns the user-scoped variant.
// Treats version as an opaque integer string. Precondition: provided segments must round-trip
// through path validation. Does NOT escape components. Returns built string. Safe for concurrent use.
func BuildArtifactURI(appName, userID, sessionID, fileName string, version int64) string {
	if sessionID == "" {
		return fmt.Sprintf("artifact://apps/%s/users/%s/artifacts/%s/versions/%d", appName, userID, fileName, version)
	}
	return fmt.Sprintf("artifact://apps/%s/users/%s/sessions/%s/artifacts/%s/versions/%d", appName, userID, sessionID, fileName, version)
}

// IsRewindTombstone returns true exactly when part is non-nil, part.InlineData is non-nil with
// MIMEType == "application/octet-stream", len(Data) == 0 (nil or empty slice), DisplayName == "",
// and every other field of genai.Part and genai.Blob is zero.
func IsRewindTombstone(part *genai.Part) bool {
	if part == nil || part.InlineData == nil {
		return false
	}
	blob := *part.InlineData
	if blob.MIMEType != rewindTombstoneMIMEType || len(blob.Data) != 0 {
		return false
	}
	// Data may be nil or empty: clonePart keeps an empty slice non-nil.
	blob.MIMEType, blob.Data = "", nil
	rest := *part
	rest.InlineData = nil
	return reflect.ValueOf(blob).IsZero() && reflect.ValueOf(rest).IsZero()
}

// rewindTombstoneMIMEType is the media type of the rewind marker, as
// adk-python's _REWIND_TOMBSTONE.
const rewindTombstoneMIMEType = "application/octet-stream"

// NewRewindTombstone returns a new marker part that a session rewind saves to remove an artifact.
// The part is safe for the caller to store or modify. It returns a fresh part on each call that
// passes SaveRequest.Validate and that IsRewindTombstone accepts.
func NewRewindTombstone() *genai.Part {
	return &genai.Part{InlineData: &genai.Blob{MIMEType: rewindTombstoneMIMEType}}
}

// isArtifactRef reports whether part is an artifact reference, i.e. its
// FileData.FileURI starts with the "artifact://" scheme.
func isArtifactRef(part *genai.Part) bool {
	return part != nil && part.FileData != nil && strings.HasPrefix(part.FileData.FileURI, artifactURIScheme)
}

// ValidatePathSegment reports whether value is safe to use as the app name,
// user ID or session ID of an artifact, with the rules of adk-python's
// validate_path_segment. fieldName names the field in the error message, for
// example "session_id". The caller provides the value and the field name.
//
// The rules apply in this order and the first violation is returned:
//
//  1. value is not empty;
//  2. value contains no NUL byte;
//  3. value does not start with "/" or "\";
//  4. value does not start with an ASCII letter followed by ":" (drive-qualified);
//  5. value is not "." or "..", and no segment of value split on "/" and "\" is "..";
//  6. only when value contains "/" or "\": no segment, split on both, is one of
//     "apps", "users", "sessions", "artifacts" or "versions" after Unicode case
//     folding with golang.org/x/text/cases.Fold (Python's casefold), so "a/APPS/b",
//     "a/ſessions" and "x/seßions" are refused and "x/versİons" is accepted.
//     A value without a separator, such as "users" or "seßions", is valid.
//
// The empty value is refused: the error is "<fieldName> must not be empty.".
// The other messages are "<fieldName> must not contain null bytes.", "must not
// be an absolute path or start with a slash.", "must not be drive-qualified.",
// "must not contain traversal segments." and "must not contain reserved path
// segments.", each after "<fieldName> ". A violation returns an error that
// wraps [ErrInputValidation] and does not contain value. A "/" alone, as in
// "group/user123", is allowed. ValidatePathSegment is safe for concurrent use.
func ValidatePathSegment(value, fieldName string) error {
	// Unlike adk-python, the messages leave value out: it is caller data.
	switch {
	case value == "":
		return invalidInput(fieldName + " must not be empty.")
	case strings.Contains(value, "\x00"):
		return invalidInput(fieldName + " must not contain null bytes.")
	case strings.HasPrefix(value, "/") || strings.HasPrefix(value, "\\"):
		return invalidInput(fieldName + " must not be an absolute path or start with a slash.")
	case isDriveQualified(value):
		return invalidInput(fieldName + " must not be drive-qualified.")
	}
	segments := strings.Split(strings.ReplaceAll(value, "\\", "/"), "/")
	if value == "." || value == ".." || slices.Contains(segments, "..") {
		return invalidInput(fieldName + " must not contain traversal segments.")
	}
	if strings.ContainsAny(value, "/\\") && hasReservedSegment(segments) {
		return invalidInput(fieldName + " must not contain reserved path segments.")
	}
	return nil
}

// isDriveQualified reports whether value starts with an ASCII letter and a
// colon, as a Windows drive such as "C:" does (adk-python's _WINDOWS_DRIVE_RE).
func isDriveQualified(value string) bool {
	if len(value) < 2 || value[1] != ':' {
		return false
	}
	c := value[0]
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z'
}

// hasReservedSegment reports whether a segment, case-folded as Python's
// str.casefold does, is one of reservedURISegments.
func hasReservedSegment(segments []string) bool {
	// A Caser is not safe for concurrent use, so each call makes its own.
	fold := cases.Fold()
	for _, segment := range segments {
		if slices.Contains(reservedURISegments, fold.String(segment)) {
			return true
		}
	}
	return false
}

// MaxArtifactReferenceDepth is the maximum number of artifact references a
// service follows to load one artifact, as adk-python's
// _MAX_ARTIFACT_REFERENCE_DEPTH. It counts reference hops only: loading the
// requested artifact does not use one, so a chain of five references
// resolves and a sixth reference fails with [ErrInputValidation]. It bounds
// self-referencing and cyclic chains.
const MaxArtifactReferenceDepth = 5

// ParseAndValidateReferenceURI checks an artifact:// reference before a
// service saves or follows it, and returns its target. uri is the stored or
// submitted FileData.FileURI; appName, userID and sessionID are the scope of
// the caller (sessionID is empty for a caller without a session); and
// remainingDepth is how many more references the caller may follow. A service
// that follows a chain starts at [MaxArtifactReferenceDepth], passes one less
// at each hop and calls it with the scope of the artifact it just reached.
// A save checks a submitted reference with [MaxArtifactReferenceDepth].
//
// The checks run in adk-python's order (resolve_artifact_reference) and the
// first failure is returned:
//
//  1. remainingDepth is greater than zero;
//  2. uri has the session-scoped form
//     "artifact://apps/{app}/users/{user}/sessions/{session}/artifacts/{file}/versions/{version}"
//     or the user-scoped form without the sessions segment, where the app,
//     user and session segments hold no "/" and version is a positive decimal
//     number. "versions/0" is refused because version 0 asks for the latest
//     version elsewhere in this package, which a reference must not do;
//  3. the target has the caller's app and user, and a session-scoped target
//     has the caller's session.
//
// It does not check that the target exists or that its names pass
// [ValidatePathSegment]; the service does that when it looks the target up.
// On success it returns the parsed target, with an empty SessionID for a
// user-scoped one. On failure it returns the zero ParsedArtifactURI and an
// error that wraps [ErrInputValidation] and contains adk-python's phrase
// ("Exceeded maximum recursion depth", "Invalid artifact reference URI") or
// the scope rule that failed. The message never contains uri or its parts.
// It is safe for concurrent use.
func ParseAndValidateReferenceURI(uri, appName, userID, sessionID string, remainingDepth int) (ParsedArtifactURI, error) {
	if remainingDepth <= 0 {
		return ParsedArtifactURI{}, invalidInput(fmt.Sprintf("artifact reference resolution exceeded maximum depth of %d (Exceeded maximum recursion depth resolving artifact reference)", MaxArtifactReferenceDepth))
	}
	parsed, ok := parseArtifactURI(uri)
	if !ok {
		// adk-python's message holds the URI; it is caller data, so it is left out.
		return ParsedArtifactURI{}, invalidInput("invalid artifact reference URI (Invalid artifact reference URI)")
	}
	if err := validateArtifactReferenceScope(appName, userID, sessionID, parsed); err != nil {
		return ParsedArtifactURI{}, fmt.Errorf("invalid artifact reference: %w", err)
	}
	return ParsedArtifactURI(parsed), nil
}

// validateArtifactReferenceScope ensures an artifact reference cannot escape the
// caller's app/user/session scope: parsed must belong to the same app and user
// as appName/userID, and if parsed is session-scoped, its session must match
// sessionID. Its errors wrap [ErrInputValidation].
func validateArtifactReferenceScope(appName, userID, sessionID string, parsed parsedArtifactURI) error {
	if parsed.AppName != appName || parsed.UserID != userID {
		return invalidInput("artifact reference must stay within the same app and user scope")
	}
	if parsed.SessionID != "" && parsed.SessionID != sessionID {
		return invalidInput("session-scoped artifact reference must stay within the same session scope")
	}
	return nil
}
