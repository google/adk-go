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
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unicode/utf8"
)

// errUnexpectedEntry reports a symbolic link, or anything else that is not
// the directory or regular file the layout puts there. It is deliberately
// neither [fs.ErrInvalid] nor [fs.ErrNotExist]: the request is fine and the
// artifact may exist, but the storage is not in a state to be trusted.
var errUnexpectedEntry = errors.New("not a regular file or directory of the artifact storage")

// metadata is the metadata.json stored beside a payload. The fields and their
// JSON names are those of adk-python's FileArtifactVersion.
type metadata struct {
	// FileName is the filename the artifact was saved under, with its "user:"
	// prefix if it had one.
	FileName string `json:"fileName"`
	// MIMEType is the media type of inline data. It is absent for a text
	// part, which is how the two are told apart.
	MIMEType    string `json:"mimeType,omitempty"`
	DisplayName string `json:"displayName,omitempty"`
	// Version and CanonicalURI repeat what the location of the file says.
	// They are written because adk-python requires them and are never read.
	Version        int64          `json:"version"`
	CanonicalURI   string         `json:"canonicalUri"`
	CustomMetadata customMetadata `json:"customMetadata"`
	// CreateTime is in Unix seconds. A file without it counts as damaged.
	CreateTime *float64 `json:"createTime"`
}

// createTime is the time the version was saved.
func (m metadata) createTime() time.Time {
	return time.UnixMicro(int64(math.Round(*m.CreateTime * 1e6)))
}

// customMetadata is the caller's metadata, whose values Save stores as
// strings.
type customMetadata map[string]string

// UnmarshalJSON accepts the values adk-python stores as well, which can be
// any JSON: one that is not a string is kept as its JSON text.
func (c *customMetadata) UnmarshalJSON(data []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*c = make(customMetadata, len(raw))
	for key, value := range raw {
		text := string(value)
		if strings.HasPrefix(text, `"`) {
			if err := json.Unmarshal(value, &text); err != nil {
				return err
			}
		}
		(*c)[key] = text
	}
	return nil
}

// values returns the metadata in the form [artifact.ArtifactVersion] carries
// it: never nil.
func (c customMetadata) values() map[string]any {
	values := make(map[string]any, len(c))
	for key, value := range c {
		values[key] = value
	}
	return values
}

// readMetadata reads the metadata of the version directory dir.
//
// intact is false, with no error, when the file is there but is not a
// metadata document: it is not UTF-8, not the JSON object expected, or has no
// creation time. The zero metadata returned then stands for "nothing known".
//
// A missing file is an error wrapping [fs.ErrNotExist] instead, which makes
// the version count as absent. adk-python reads such a version as text: a
// text part is the kind stored without a MIME type, and no metadata means no
// MIME type. That guess is not made here, because a version with a payload
// and no metadata.json turns up in storage that is in good order: it is what
// a reader sees of a version that Delete is removing at that moment. The
// payload may be inline data, an image or a JSON document, and guessing would
// return it as text in ordinary operation. Reported as absent, a latest
// version caught like this makes readVersion look for the one before it.
//
// A damaged file is different: it only exists in storage that is damaged
// already, and there reading it the way adk-python does is the lesser harm.
func readMetadata(dir *os.Root) (meta metadata, intact bool, err error) {
	data, err := readFile(dir, metadataFile)
	if err != nil {
		return metadata{}, false, fileError("read artifact metadata", err)
	}
	// encoding/json accepts invalid UTF-8 and replaces it, so check first.
	if !utf8.Valid(data) || json.Unmarshal(data, &meta) != nil || meta.CreateTime == nil {
		return metadata{}, false, nil
	}
	return meta, true, nil
}

// readFile reads a file of a version directory.
func readFile(dir *os.Root, name string) ([]byte, error) {
	if _, err := statFile(dir, name); err != nil {
		return nil, err
	}
	return dir.ReadFile(name)
}

// statFile describes a file of a version directory. It refuses anything but a
// regular file, so that a symbolic link is not followed even to a file in the
// same directory.
func statFile(dir *os.Root, name string) (fs.FileInfo, error) {
	info, err := dir.Lstat(name)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errUnexpectedEntry
	}
	return info, nil
}

// writeFile creates the file name in dir with the given content and syncs it
// to disk. Syncing a file makes its content durable but not its name, which
// belongs to the directory: see syncDir.
//
// It fails if the file exists. A staging directory starts empty and its two
// files have different names, so that only happens on a filesystem that takes
// the two names for one, and there the second must not overwrite the first.
func writeFile(dir *os.Root, name string, data []byte) error {
	file, err := dir.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	_, err = file.Write(data)
	if err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	return err
}

// syncDir syncs the open directory dir to disk, which makes the names of the
// files created in it durable. Where a directory cannot be synced it does
// nothing.
func syncDir(dir *os.Root) error {
	if runtime.GOOS == "windows" {
		// Windows refuses to flush a directory opened for reading, and NTFS
		// journals the changes to a directory itself.
		return nil
	}
	handle, err := dir.Open(".")
	if err != nil {
		return err
	}
	err = handle.Sync()
	if errors.Is(err, syscall.EINVAL) || errors.Is(err, errors.ErrUnsupported) {
		// Some filesystems, shared folders of virtual machines among them,
		// answer like this because they cannot sync a directory. Saving
		// there has to keep working, without the protection.
		err = nil
	}
	if closeErr := handle.Close(); err == nil {
		err = closeErr
	}
	return err
}

// openDir opens the directory at rel, a slash-separated path below the root
// directory, creating its missing components when create is set.
//
// It goes down one component at a time and refuses one that is not a real
// directory. [os.Root] alone keeps a path from leaving the root, but follows
// a symbolic link that stays inside it, which could point one user's storage
// at another's.
func (s *service) openDir(ctx context.Context, rel string, create bool) (*os.Root, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return s.open(rel, create)
}

// open is openDir for work that goes on after the caller's context is done.
func (s *service) open(rel string, create bool) (*os.Root, error) {
	dir, err := os.OpenRoot(s.rootDir)
	if err != nil {
		return nil, fileError("open artifact root", err)
	}
	for name := range strings.SplitSeq(filepath.FromSlash(rel), string(filepath.Separator)) {
		if create {
			if err := dir.Mkdir(name, 0o700); err != nil && !errors.Is(err, fs.ErrExist) {
				_ = dir.Close()
				return nil, fileError("create artifact directory", err)
			}
		}
		child, err := openChild(dir, name)
		_ = dir.Close()
		if err != nil {
			return nil, fileError("open artifact directory", err)
		}
		dir = child
	}
	return dir, nil
}

// maxSwaps bounds how often openChild looks again at a directory that keeps
// being replaced while it is opened. Each time takes a removal and a creation
// between two system calls, so more than a few in a row do not happen by
// themselves.
const maxSwaps = 100

// openChild opens the directory name inside parent, refusing a symbolic link.
//
// It checks that what it opened is what it inspected, so that swapping the
// directory for a link in between does not get past the inspection. Delete
// and Save remove and recreate directories all the time, which looks the
// same, so after a mismatch the name is inspected again: a link is refused
// when it is seen, never assumed.
func openChild(parent *os.Root, name string) (*os.Root, error) {
	for range maxSwaps {
		info, err := parent.Lstat(name)
		if err != nil {
			return nil, err
		}
		if !info.IsDir() {
			return nil, errUnexpectedEntry
		}
		child, err := parent.OpenRoot(name)
		if err != nil {
			return nil, err
		}
		opened, err := child.Stat(".")
		if err != nil {
			_ = child.Close()
			return nil, err
		}
		if os.SameFile(info, opened) {
			return child, nil
		}
		_ = child.Close()
	}
	return nil, errUnexpectedEntry
}

// readVersion calls read with the directory of a version of the artifact at
// loc and its number: the latest version when version is nil. It returns an
// error wrapping [fs.ErrNotExist] when there is no such version.
//
// The latest version can be deleted while it is being read. The version that
// is then the latest is the answer, not "missing", so the read is repeated.
func (s *service) readVersion(ctx context.Context, loc location, version *int64, read func(dir *os.Root, version int64) error) error {
	if version != nil && *version < 0 {
		return errNotFound()
	}
	versions, err := s.openDir(ctx, loc.versions(), false)
	if err != nil {
		return notFound(err)
	}
	defer func() { _ = versions.Close() }()
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		var v int64
		if version != nil {
			v = *version
		} else {
			published, err := listVersions(versions)
			if err != nil {
				return notFound(err)
			}
			if len(published) == 0 {
				return errNotFound()
			}
			v = published[len(published)-1]
		}
		name := strconv.FormatInt(v, 10)
		dir, err := openChild(versions, name)
		if errors.Is(err, fs.ErrNotExist) || errors.Is(err, errUnexpectedEntry) {
			// Nothing under the number, or something that is not a
			// directory, is no version. For the latest that means the
			// listing is out of date.
			if version != nil {
				return errNotFound()
			}
			continue
		}
		if err != nil {
			return fileError("open artifact version", err)
		}
		err = read(dir, v)
		deleted := errors.Is(err, fs.ErrNotExist) && !isEntry(versions, name, dir)
		_ = dir.Close()
		if !deleted {
			return err
		}
	}
}

// isEntry reports whether the open directory dir is still what parent has
// under name. While dir is open nothing else can take its place and pass for
// it.
func isEntry(parent *os.Root, name string, dir *os.Root) bool {
	entry, err := parent.Lstat(name)
	if err != nil {
		return false
	}
	opened, err := dir.Stat(".")
	return err == nil && os.SameFile(entry, opened)
}

// listVersions returns the published versions in the versions directory dir,
// in ascending order. Staging directories and anything else that is not a
// directory named by a version number are skipped.
func listVersions(dir *os.Root) ([]int64, error) {
	entries, err := fs.ReadDir(dir.FS(), ".")
	if err != nil {
		return nil, fileError("list artifact versions", err)
	}
	var versions []int64
	for _, entry := range entries {
		if version, ok := parseVersion(entry.Name()); ok && entry.IsDir() {
			versions = append(versions, version)
		}
	}
	slices.Sort(versions)
	return versions, nil
}

// maxClaimAttempts bounds how often Save starts over after a directory it had
// just made was removed again. That takes a Delete, or the cleanup of a failed
// Save, emptying a directory on the artifact's path at the same moment, so the
// second attempt normally succeeds.
const maxClaimAttempts = 100

// claimVersion opens the versions directory at rel, creating it if needed, and
// reserves the next version in it. It returns the directory, the version and
// the name of the staging directory that holds the reservation.
//
// Directories that hold no version are removed by Delete, so one made here
// can be gone again before the reservation is in it. That shows as a missing
// directory, and the answer is to make it again. Once the staging directory
// exists nothing above it can be removed.
func (s *service) claimVersion(ctx context.Context, rel string) (dir *os.Root, version int64, pending string, err error) {
	for range maxClaimAttempts {
		if dir, err = s.openDir(ctx, rel, true); err == nil {
			if version, pending, err = reserveVersion(dir); err == nil {
				return dir, version, pending, nil
			}
			_ = dir.Close()
		}
		if !errors.Is(err, fs.ErrNotExist) {
			break
		}
	}
	return nil, 0, "", err
}

// reserveVersion claims the number after the highest published version in
// the versions directory dir by creating its staging directory, whose name it
// returns with the version.
func reserveVersion(dir *os.Root) (int64, string, error) {
	versions, err := listVersions(dir)
	if err != nil {
		return 0, "", err
	}
	var version int64
	if len(versions) > 0 {
		version = versions[len(versions)-1] + 1
	}
	// Past the highest number the counter wraps to a negative one.
	for ; version >= 0; version++ {
		name := strconv.FormatInt(version, 10)
		pending := "." + name + ".pending"
		if err := dir.Mkdir(pending, 0o700); errors.Is(err, fs.ErrExist) {
			continue // Reserved by another save, or left by one that died.
		} else if err != nil {
			return 0, "", fileError("reserve artifact version", err)
		}
		// Another save can have published this number since the listing.
		_, err := dir.Lstat(name)
		if errors.Is(err, fs.ErrNotExist) {
			return version, pending, nil
		}
		if removeErr := dir.Remove(pending); removeErr != nil {
			return 0, "", fileError("release artifact reservation", removeErr)
		}
		if err != nil {
			return 0, "", fileError("check artifact version", err)
		}
	}
	return 0, "", errors.New("artifact version limit reached")
}

// deletedSuffix ends the name of a version directory that Delete has taken
// out of service and is removing. Such a name also starts with a dot.
const deletedSuffix = ".deleted"

// removeVersions removes one published version from the versions directory
// dir, or all of them when version is nil, and whatever an earlier Delete
// took out of service and did not get to remove. Everything else in the
// directory stays: a staging directory belongs to a save still in progress,
// and an entry that only looks like the requested number is not a version.
//
// A version is renamed before it is removed. The rename takes it away in one
// step, so that a reader looking for the latest version never finds one that
// is half gone, and so that removing it cannot run into a version published
// under the same number meanwhile.
func removeVersions(dir *os.Root, version *int64) error {
	entries, err := fs.ReadDir(dir.FS(), ".")
	if err != nil {
		return fileError("list artifact versions", err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if !entry.IsDir() {
			continue
		}
		if v, ok := parseVersion(name); ok && (version == nil || v == *version) {
			deleted := "." + name + "." + rand.Text() + deletedSuffix
			if err := dir.Rename(name, deleted); errors.Is(err, fs.ErrNotExist) {
				continue // Another Delete got there first.
			} else if err != nil {
				return fileError("delete artifact version", err)
			}
			name = deleted
		}
		if strings.HasPrefix(name, ".") && strings.HasSuffix(name, deletedSuffix) {
			if err := dir.RemoveAll(name); err != nil {
				return fileError("delete artifact version", err)
			}
		}
	}
	return nil
}

// fileError wraps a filesystem error with the operation that failed.
//
// Such errors carry the path, which is built from caller identifiers. Those
// must not reach an error message, so the path is dropped and the cause kept
// for [errors.Is].
func fileError(operation string, err error) error {
	for {
		switch e := err.(type) {
		case *fs.PathError:
			err = e.Err
		case *os.LinkError:
			err = e.Err
		default:
			return fmt.Errorf("%s: %w", operation, err)
		}
	}
}

// errNotFound is the error for an artifact or version that does not exist.
func errNotFound() error {
	return fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
}

// notFound turns the error for a missing directory into the one for a missing
// artifact, which is what the directory stands for.
func notFound(err error) error {
	if errors.Is(err, fs.ErrNotExist) {
		return errNotFound()
	}
	return err
}
