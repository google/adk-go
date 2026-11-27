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

// Package fileartifact stores versioned artifacts in a directory on the local
// filesystem, so that they outlive the process without a cloud bucket.
//
// Create a service with [New] and pass it to runner.Config.ArtifactService:
//
//	service, err := fileartifact.New(fileartifact.Config{RootDir: "./artifacts"})
//	if err != nil {
//		return err
//	}
//
// # Layout
//
// The directory tree and the metadata file follow the format of adk-python's
// FileArtifactService:
//
//	RootDir/apps/{app}/users/{user}/sessions/{session}/artifacts/{name}/versions/{version}/
//	RootDir/apps/{app}/users/{user}/artifacts/{name}/versions/{version}/
//
// The second form holds the artifacts whose filename starts with "user:", which
// all sessions of a user share; {name} is the filename without that prefix.
// A version directory holds the payload, in a file named after the last
// segment of the filename, and a metadata.json describing it. The
// CanonicalURI of a version is the file URI of that payload file.
//
// # Names
//
// Application names, user IDs, session IDs and filenames become directory
// names, so each is a relative path of segments separated by "/". It must be
// valid UTF-8 of at most 1024 bytes (not counting a "user:" prefix) in at most
// 64 segments of at most 255 bytes, with no segment made only of dots and
// white space (which rules out "." and ".."), no drive prefix such as "C:"
// and no null byte. A request that breaks a rule fails with an error wrapping
// [fs.ErrInvalid] before anything is written. A filesystem can refuse a name
// these rules allow, which then fails as a storage error.
//
// A filename may be nested, such as "reports/q1.txt", which the in-memory and
// Cloud Storage services reject. It may not contain a backslash or begin or
// end with white space, and it may not use the two names the layout needs
// for itself: "metadata.json" as its last segment, whose payload file would
// have to be the metadata file beside it, and "versions" as any segment,
// which would place one artifact inside the version directories of another.
//
// An identifier is checked with "\" counting as a separator too, as it is one
// on Windows; elsewhere it is stored as an ordinary character. An identifier
// of several segments may not use the directory names of the layout ("apps",
// "users", "sessions", "artifacts", "versions"), which would let it name the
// storage of another user or session.
//
// The reserved and the layout names are matched without regard to case or to
// trailing dots and spaces, which some filesystems ignore. That is the only
// place such differences are evened out: two names the filesystem treats as
// one share storage. On a case-insensitive filesystem "Alice" and "alice" are
// the same user and "Notes.txt" and "notes.txt" the same artifact. Windows
// also ignores trailing dots and spaces, and where it keeps short names it
// opens the directory of "alexander" as "ALEXAN~1" too. Use identifiers that
// differ by more than that when the directory serves several tenants.
//
// # Versions
//
// Versions start at 0 and each Save takes the number after the highest one
// present, so deleting the latest version frees its number for the next Save.
// A Save stages its files in a hidden directory and publishes them with one
// rename, and Delete takes a version away with one rename before removing it:
// readers see a complete version or none. A process killed in between leaves
// a hidden directory behind. One left by Save is never read and keeps its
// number reserved; one left by Delete is removed by the next Delete. Save
// syncs the files of a version to disk before it publishes them, which
// adk-python does not, so a version found after the machine lost power is
// complete too. A save that returned just before that can be missing.
//
// A text part and an inline-data part are told apart by the MIME type in the
// metadata, which only inline data has. Inline data saved without a MIME type
// is therefore stored, and loaded, as "application/octet-stream".
// GetArtifactVersion reports "text/plain" for a text part. The creation time
// of a version is kept to the microsecond, and custom metadata values as
// strings.
//
// # Damaged storage
//
// A metadata.json that cannot be parsed, or has no creation time, costs its
// own version its details and nothing else. List and Versions do not read
// metadata and are unaffected. GetArtifactVersion reports the version with no
// MIME type, no custom metadata and the payload file's modification time, or
// [fs.ErrNotExist] if the payload is gone too. Load can no longer tell what
// the payload was, so it follows the rule above, as adk-python does: with no
// MIME type recorded the payload is text, and one that is not valid UTF-8,
// which cannot be text, is reported as missing with [fs.ErrNotExist].
//
// A version whose metadata.json is not there at all is missing as well,
// whatever its payload, where adk-python reads it as text. Save never
// publishes a version without the file, so such a directory is usually one
// that is being deleted at that moment, and its payload may be inline data
// that must not come back as text.
//
// Symbolic links below RootDir are not followed. One found where a
// directory or a file of the layout belongs is refused with an error that
// wraps neither [fs.ErrInvalid] nor [fs.ErrNotExist], as is any other entry
// of the wrong type. Two places pass over such an entry instead: under a
// version's number it is no version, and List skips it inside the scope it
// lists.
//
// # Differences from adk-python
//
// The request rules of [artifact.Service] hold here as in the other Go
// services: a session ID is required even for a "user:" filename, an empty
// text part is rejected while inline data without bytes is stored as empty,
// Delete can remove a single version, and a missing artifact is an error
// wrapping [fs.ErrNotExist]. Go has no session rewind, so an empty
// "application/octet-stream" artifact loads as what it is instead of as
// absent.
//
// Where adk-python rewrites a filename (removing white space around it,
// turning a backslash into "/", storing an empty name as "artifact"), this
// package rejects the name, so that no name is stored in another directory
// than the one adk-python uses for it. It rejects the layout's own names in
// every operation where adk-python does so on save only. List takes names
// from the directory tree instead of from metadata.json and leaves out
// directories whose name the other operations would reject, so a name it
// reports is the name the artifact is addressed by. It reports a user-scoped
// artifact adk-python saved without a session under its "user:" name.
//
// A directory in adk-python's format can be read, with these differences.
// Custom metadata stored as something other than a string is returned as the
// JSON text of the value. A version directory whose name is not the plain
// decimal number, such as "01", is not a version. A version without a
// metadata.json is missing instead of text. An artifact stored under a name
// or an identifier this package rejects cannot be reached.
//
// An artifact is found under the name of its directory, which is the name
// List reports. adk-python rewrites a name before it makes the directory,
// while its own listing and its session events keep the name as it was
// given. A name remembered from there is found here after the same
// rewriting: white space around it removed, "\" turned into "/", "." and
// empty segments dropped, "artifact" for an empty name, and "user:" put in
// front of a name that was saved without a session.
//
// adk-python releases before 2.6.0, and 1.x releases before 1.39.1, stored
// RootDir/users/... without the apps/{app} level. That tree is not read, as
// current adk-python does not read it either. Moving it to
// RootDir/apps/{app}/users makes it readable.
//
// adk-python follows a symbolic link that replaces a scope directory or a
// stored file, which reads from outside the root, and creates directories and
// files with the permissions the umask leaves. Here new directories get mode
// 0700 and files 0600 before the umask, and errors do not contain file paths,
// which are built from caller identifiers.
//
// adk-python expands a leading "~" in the root directory and creates the
// target of a symbolic link found there. [New] does neither.
package fileartifact
