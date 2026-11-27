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

package fileartifact

import (
	"fmt"
	"io/fs"
	"path"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Directory and file names of the storage layout.
const (
	appsDir      = "apps"
	usersDir     = "users"
	sessionsDir  = "sessions"
	artifactsDir = "artifacts"
	versionsDir  = "versions"
	metadataFile = "metadata.json"

	// userPrefix marks a filename as shared by all sessions of a user.
	userPrefix = "user:"
)

// Limits on an identifier or a filename. adk-python has none and leaves it to
// the operating system to fail. Only the first is a property of filesystems;
// the other two are this package's own choice.
const (
	// maxSegmentBytes is the longest name a directory entry can have: 255
	// bytes on ext4, XFS, Btrfs and APFS (NAME_MAX), and a name that short
	// also fits the 255 UTF-16 units of NTFS. A filesystem refuses a longer
	// name anyway, but only once Save gets to that segment, after it created
	// the directories above it. Checking here fails the request before
	// anything is written.
	maxSegmentBytes = 255

	// maxNameBytes bounds a whole identifier or filename. Without it the two
	// other limits would allow 64 segments of 255 bytes, 16 KB, for a name
	// that is copied into every metadata.json, List result and session
	// event. 1024 bytes is not derived from anything here: it is the limit
	// Cloud Storage puts on an object name, borrowed because it is a bound
	// that callers of the Cloud Storage artifact service already keep to.
	maxNameBytes = 1024

	// maxSegments bounds how deep one identifier or filename nests. Every
	// segment is a directory, and List and Delete hold one directory open
	// per level they descend, so depth costs file descriptors, as many
	// times over as there are calls running at once. Within maxNameBytes
	// alone a name could have 512 levels. 64 is not derived from anything
	// either: it is far deeper than a real name and still cheap.
	maxSegments = 64
)

// location is where the artifact of a request is stored. Its paths are
// slash-separated and relative to the root directory.
type location struct {
	// scope is the artifacts directory of the session or, for a user-scoped
	// filename, of the user.
	scope string
	// name is the artifact's directory below scope: the filename without its
	// "user:" prefix.
	name string
}

// versions is the directory holding the artifact's version directories.
func (l location) versions() string {
	return l.scope + "/" + l.name + "/" + versionsDir
}

// payload is the name of the payload file inside a version directory.
func (l location) payload() string {
	return path.Base(l.name)
}

// locate validates the identifiers of an op request, such as "save", and
// returns where its artifact is stored.
func locate(op, app, user, session, fileName string) (location, error) {
	if err := checkIDs(op, app, user, session, field{"FileName", fileName}); err != nil {
		return location{}, err
	}
	name, userScoped := strings.CutPrefix(fileName, userPrefix)
	if !validName(name) {
		return location{}, invalid(`FileName must be a relative path of "/"-separated segments: valid UTF-8, at most 1024 bytes after any "user:" prefix and 64 segments of at most 255 bytes, no white space at either end, no segment made only of dots and white space, no backslash, drive prefix or null byte, and neither "metadata.json" as its last segment nor a "versions" segment`)
	}
	return location{scope: scopePath(app, user, session, userScoped), name: name}, nil
}

// field is a request field and its value.
type field struct{ name, value string }

// checkIDs validates the identifiers every request of kind op carries. more
// holds the other fields the request must set, so that one error names all
// that are missing, as the other artifact services do.
func checkIDs(op, app, user, session string, more ...field) error {
	ids := []field{{"AppName", app}, {"UserID", user}, {"SessionID", session}}
	var missing []string
	for _, f := range append(ids, more...) {
		if f.value == "" {
			missing = append(missing, f.name)
		}
	}
	if len(missing) > 0 {
		return invalid(fmt.Sprintf("invalid %s request: missing required fields: %s", op, strings.Join(missing, ", ")))
	}
	for _, id := range ids {
		if !validID(id.value) {
			return invalid(id.name + ` must be a relative path: valid UTF-8, at most 1024 bytes and 64 segments of at most 255 bytes, no segment made only of dots and white space, no drive prefix or null byte, and with several segments none that is a directory name of the storage layout`)
		}
	}
	return nil
}

// scopePath is the artifacts directory of a session, or of a user when
// userScoped is set.
func scopePath(app, user, session string, userScoped bool) string {
	segments := []string{appsDir, app, usersDir, user}
	if !userScoped {
		segments = append(segments, sessionsDir, session)
	}
	return strings.Join(append(segments, artifactsDir), "/")
}

// validID reports whether an application name, user ID or session ID can be
// used as a relative directory path.
//
// An identifier may have several segments, such as "acme/alice", as it may
// for adk-python. Then none of them may be a directory name of the layout,
// because such a name in the middle of an identifier reads as the layout's
// own. The user ID "alice/sessions/s1" with a "user:" filename would be
// stored in
//
//	apps/{app}/users/alice/sessions/s1/artifacts/
//
// which is where session "s1" of user "alice" keeps its artifacts: whoever
// picks that user ID reads and overwrites them. The names are refused
// wherever they stand in the identifier, not only at its end, since the one
// in this example is in the middle. An identifier of one segment may be any
// name, also "sessions": it fills exactly one directory level and the path
// still reads one way only.
//
// A backslash separates segments on Windows and is an ordinary character
// elsewhere, so the identifier has to hold up read either way.
func validID(id string) bool {
	if !validPath(id) {
		return false
	}
	pieces := strings.Split(strings.ReplaceAll(id, `\`, "/"), "/")
	for _, piece := range pieces {
		if isBlank(piece) || len(pieces) > 1 && isLayoutDir(piece) {
			return false
		}
	}
	return len(pieces) <= maxSegments
}

// validName reports whether a filename, already without its "user:" prefix,
// can be used as the relative directory path of an artifact.
//
// It refuses the two names the layout uses inside an artifact's directory.
// The artifact "docs/chart.png" with two versions is stored as
//
//	artifacts/docs/chart.png/versions/0/chart.png      (payload)
//	artifacts/docs/chart.png/versions/0/metadata.json
//	artifacts/docs/chart.png/versions/1/chart.png
//	artifacts/docs/chart.png/versions/1/metadata.json
//
// The payload file is named after the last segment of the filename and lies
// beside metadata.json. A filename ending in "metadata.json", such as
// "docs/metadata.json", would need both to be one file:
//
//	artifacts/docs/metadata.json/versions/0/metadata.json   (payload and metadata)
//
// which cannot hold the two. Only the last segment matters:
// "metadata.json/chart.png" is stored in a directory of that name and is
// fine.
//
// A "versions" segment anywhere would put one artifact inside the version
// directories of another. Next to the "docs/chart.png" above, the filename
// "docs/chart.png/versions/1" would be stored as
//
//	artifacts/docs/chart.png/versions/1/versions/0/1
//
// inside version 1 of "docs/chart.png": deleting that version would delete
// this artifact with it. As "docs/chart.png/versions/7" it would create the
// directory versions/7 and so a version 7 of "docs/chart.png" that has no
// payload. A filename such as "versions/a" would be stored below
// artifacts/versions/, where List does not look, because it takes a directory
// of that name for the versions of an artifact.
//
// adk-python refuses these names on save only, so that an artifact stored
// under one before it had the rule can still be read and deleted. Here every
// operation refuses them: this package never stored such an artifact, and a
// Load or Delete with such a name would reach into the version directories of
// another artifact just as a Save would.
//
// A name may not begin or end with white space either. adk-python removes it
// before it makes the directory: it stores " notes.txt " in
//
//	artifacts/notes.txt/
//
// Stored as given, the name would be a second directory that adk-python can
// never reach, and the artifact adk-python stored would not be found under
// it. Refusing it leaves "notes.txt" as the one name of that directory.
func validName(name string) bool {
	return validDir(name) && !sameFile(path.Base(name), metadataFile) && strings.TrimFunc(name, isSpace) == name
}

// validDir reports whether name can be a directory on the way to an
// artifact: it is validName without the rule about the last segment, which
// only the artifact's own name has to keep.
func validDir(name string) bool {
	if !validPath(name) || strings.ContainsRune(name, '\\') {
		return false
	}
	for segment := range strings.SplitSeq(name, "/") {
		if sameFile(segment, versionsDir) {
			return false
		}
	}
	return true
}

// validPath reports whether s, an identifier or a filename, can serve as a
// relative path of "/"-separated segments.
func validPath(s string) bool {
	if len(s) > maxNameBytes || !utf8.ValidString(s) || strings.ContainsRune(s, 0) || driveQualified(s) {
		return false
	}
	segments := strings.Split(s, "/")
	for _, segment := range segments {
		if len(segment) > maxSegmentBytes || isBlank(segment) {
			return false
		}
	}
	return len(segments) <= maxSegments
}

// isBlank reports whether a segment is made of dots and white space only,
// or of nothing. That is no file name: "." and ".." are directories, and
// Windows strips both characters from the end of a name, which would leave
// nothing and so name the parent directory.
func isBlank(segment string) bool {
	return strings.TrimFunc(segment, func(r rune) bool { return r == '.' || isSpace(r) }) == ""
}

// isSpace reports whether r is white space as adk-python sees it, which
// counts the information separators as well.
func isSpace(r rune) bool {
	return unicode.IsSpace(r) || '\x1c' <= r && r <= '\x1f'
}

func isLayoutDir(segment string) bool {
	for _, dir := range []string{appsDir, usersDir, sessionsDir, artifactsDir, versionsDir} {
		if sameFile(segment, dir) {
			return true
		}
	}
	return false
}

// sameFile reports whether a filesystem could resolve segment to the file
// called name: case-insensitive filesystems fold case, and Windows drops
// trailing dots and spaces.
func sameFile(segment, name string) bool {
	return strings.EqualFold(strings.TrimRight(segment, " ."), name)
}

// driveQualified reports whether s starts with a Windows drive such as "C:",
// which would make it a path on that drive.
func driveQualified(s string) bool {
	if len(s) < 2 || s[1] != ':' {
		return false
	}
	return 'a' <= s[0] && s[0] <= 'z' || 'A' <= s[0] && s[0] <= 'Z'
}

// parseVersion returns the version a directory name stands for. Only the
// canonical decimal form counts, so that "01" or "+1" is not a second
// directory for version 1.
func parseVersion(name string) (int64, bool) {
	version, err := strconv.ParseInt(name, 10, 64)
	return version, err == nil && version >= 0 && strconv.FormatInt(version, 10) == name
}

// invalid returns the error for a request the service refuses. It wraps
// [fs.ErrInvalid] and names itself like the validation errors of the other
// artifact services. msg must not contain request values, which are caller
// data.
func invalid(msg string) error {
	return fmt.Errorf("%w: request validation failed: %s", fs.ErrInvalid, msg)
}
