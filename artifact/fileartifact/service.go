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
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/platform"
)

// Config configures a filesystem artifact service.
type Config struct {
	// RootDir is the directory that holds the artifacts. It must not be empty.
	// New creates it when it is missing and resolves it to an absolute path
	// with symbolic links followed. A symbolic link on the way has to lead to
	// a directory that exists: New does not create the target of a link, which
	// adk-python does.
	//
	// A leading "~" is not expanded as adk-python expands it: that is shell
	// syntax, and here it names a directory called "~".
	RootDir string
}

// New returns an artifact service that stores its data under cfg.RootDir.
//
// The service keeps no file open between calls. It is safe for concurrent
// use, also by other instances and processes that share RootDir on a local
// filesystem. Nothing else should write to RootDir.
//
// Delete does not wait for a Save in progress, whose version may or may not
// survive it.
func New(cfg Config) (artifact.Service, error) {
	if cfg.RootDir == "" {
		return nil, fmt.Errorf("%w: artifact root directory is required", fs.ErrInvalid)
	}
	rootDir, err := filepath.Abs(cfg.RootDir)
	if err != nil {
		return nil, fileError("resolve artifact root", err)
	}
	if err := os.MkdirAll(rootDir, 0o700); err != nil {
		return nil, fileError("create artifact root", err)
	}
	// A CanonicalURI is built from this path and should name the file itself,
	// not a link to it.
	if rootDir, err = filepath.EvalSymlinks(rootDir); err != nil {
		return nil, fileError("resolve artifact root", err)
	}
	return &service{rootDir: rootDir}, nil
}

type service struct{ rootDir string }

// defaultMIMEType is recorded for inline data saved without a MIME type. The
// metadata of a text part has none, so inline data needs one to be told apart.
const defaultMIMEType = "application/octet-stream"

// textMIMEType is the MIME type GetArtifactVersion reports for a text part, as
// the other artifact services do.
const textMIMEType = "text/plain"

// Save implements [artifact.Service]. A part that has both inline data and
// text is stored as inline data, as the Cloud Storage service does.
func (s *service) Save(ctx context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	if req == nil {
		return nil, invalid("save request is nil")
	}
	loc, err := locate("save", req.AppName, req.UserID, req.SessionID, req.FileName)
	if err != nil {
		return nil, err
	}
	if req.Part == nil {
		return nil, invalid("invalid save request: missing required fields: Part")
	}
	if req.Part.Text == "" && req.Part.InlineData == nil {
		return nil, invalid("invalid save request: Part.InlineData or Part.Text has to be set")
	}
	meta, data, err := describe(req)
	if err != nil {
		return nil, err
	}
	// metadata.json holds the time as seconds in a float, which is exact to
	// the microsecond.
	seconds := float64(platform.Now(ctx).UnixMicro()) / 1e6
	meta.CreateTime = &seconds

	// Everything a caller can get wrong was checked above, so a rejected
	// request creates no directory.
	dir, version, pending, err := s.claimVersion(ctx, loc.versions())
	if err != nil {
		s.prune(loc)
		return nil, storeError(err)
	}
	// adk-python requires both fields to parse the file. Neither is read back.
	meta.Version = version
	meta.CanonicalURI = s.canonicalURI(loc, version)
	err = publish(ctx, dir, pending, version, loc.payload(), data, meta)
	if err != nil {
		// A failed save leaves no trace of its version. A process killed
		// before this runs leaves the staging directory, which readers ignore.
		if removeErr := dir.RemoveAll(pending); removeErr != nil {
			err = errors.Join(err, fileError("remove artifact staging directory", removeErr))
		}
	}
	_ = dir.Close()
	if err != nil {
		s.prune(loc)
		return nil, storeError(err)
	}
	return &artifact.SaveResponse{Version: version}, nil
}

// storeError is the error Save returns for one that kept it from storing. A
// directory that vanished under a save, the root itself if it was removed, is
// a failure to store, which a caller must not take for a missing artifact, so
// such a cause is reported without [fs.ErrNotExist].
func storeError(err error) error {
	if errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("artifact storage changed during save: %v", err)
	}
	return err
}

// describe returns the metadata and the payload bytes Save stores for the
// part of req. The version fields of the metadata are left for the caller.
func describe(req *artifact.SaveRequest) (metadata, []byte, error) {
	meta := metadata{
		FileName:       req.FileName,
		CustomMetadata: make(customMetadata, len(req.CustomMetadata)),
	}
	data := []byte(req.Part.Text)
	if blob := req.Part.InlineData; blob != nil {
		data = blob.Data
		meta.MIMEType = cmp.Or(blob.MIMEType, defaultMIMEType)
		meta.DisplayName = blob.DisplayName
	}
	// encoding/json replaces invalid UTF-8 instead of failing, which would
	// store something other than what was saved.
	if !utf8.ValidString(meta.MIMEType) || !utf8.ValidString(meta.DisplayName) {
		return metadata{}, nil, invalid("Part.InlineData.MIMEType and Part.InlineData.DisplayName must be valid UTF-8")
	}
	for key, value := range req.CustomMetadata {
		text := fmt.Sprint(value)
		if !utf8.ValidString(key) || !utf8.ValidString(text) {
			return metadata{}, nil, invalid("CustomMetadata keys and values must be valid UTF-8")
		}
		meta.CustomMetadata[key] = text
	}
	return meta, data, nil
}

// publish writes a payload and its metadata into the staging directory
// pending of the versions directory dir, then makes them a version by renaming
// that directory.
//
// The two files and, where the system can do it, the staging directory are
// synced to disk before the rename. A filesystem may otherwise record the
// rename before it has written the contents of the files, and a machine that
// loses power in between comes back with a version whose payload or metadata
// is empty or cut short. With the sync, a version found after such a crash is
// whole. A save that returned just before the crash can still be missing.
func publish(ctx context.Context, dir *os.Root, pending string, version int64, payload string, data []byte, meta metadata) error {
	encoded, err := json.Marshal(meta)
	if err != nil {
		return fmt.Errorf("encode artifact metadata: %w", err)
	}
	if err := stage(dir, pending, payload, data, encoded); err != nil {
		return err
	}
	// The last moment a canceled save can still leave nothing behind.
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := dir.Rename(pending, strconv.FormatInt(version, 10)); err != nil {
		return fileError("publish artifact version", err)
	}
	return nil
}

// stage stores a payload and its encoded metadata durably in the staging
// directory pending of the versions directory dir.
func stage(dir *os.Root, pending, payload string, data, encoded []byte) error {
	staging, err := dir.OpenRoot(pending)
	if err != nil {
		return fileError("open artifact staging directory", err)
	}
	// Closed before the caller renames the directory: Windows may refuse to
	// rename one that is open.
	defer func() { _ = staging.Close() }()
	if err := writeFile(staging, payload, data); err != nil {
		return fileError("write artifact payload", err)
	}
	if err := writeFile(staging, metadataFile, encoded); err != nil {
		return fileError("write artifact metadata", err)
	}
	if err := syncDir(staging); err != nil {
		return fileError("sync artifact staging directory", err)
	}
	return nil
}

// Load implements [artifact.Service].
func (s *service) Load(ctx context.Context, req *artifact.LoadRequest) (*artifact.LoadResponse, error) {
	if req == nil {
		return nil, invalid("load request is nil")
	}
	loc, err := locate("load", req.AppName, req.UserID, req.SessionID, req.FileName)
	if err != nil {
		return nil, err
	}
	var part *genai.Part
	err = s.readVersion(ctx, loc, req.Version, func(dir *os.Root, _ int64) error {
		meta, intact, err := readMetadata(dir)
		if err != nil {
			return err
		}
		data, err := readFile(dir, loc.payload())
		if err != nil {
			return fileError("read artifact payload", err)
		}
		switch {
		case meta.MIMEType != "":
			part = &genai.Part{InlineData: &genai.Blob{Data: data, MIMEType: meta.MIMEType, DisplayName: meta.DisplayName}}
		case !intact && !utf8.Valid(data):
			// The metadata is damaged, so the MIME type is lost, and with it
			// whether the payload was text or inline data. Bytes that are
			// not valid UTF-8 cannot have been text. What they were is
			// unknown, so they are reported as absent, as adk-python does,
			// instead of being returned under a MIME type made up here.
			return fmt.Errorf("%w: artifact metadata is damaged and its payload is not text", fs.ErrNotExist)
		default:
			// No MIME type is how a text part is stored. With damaged
			// metadata this also returns, as text, a payload that is valid
			// UTF-8 but was saved as inline data, such as a JSON document:
			// adk-python does the same, and both then read a damaged
			// directory alike.
			part = genai.NewPartFromText(string(data))
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &artifact.LoadResponse{Part: part}, nil
}

// GetArtifactVersion implements [artifact.Service].
func (s *service) GetArtifactVersion(ctx context.Context, req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	if req == nil {
		return nil, invalid("get artifact version request is nil")
	}
	loc, err := locate("get artifact version", req.AppName, req.UserID, req.SessionID, req.FileName)
	if err != nil {
		return nil, err
	}
	var result *artifact.ArtifactVersion
	err = s.readVersion(ctx, loc, req.Version, func(dir *os.Root, version int64) error {
		meta, intact, err := readMetadata(dir)
		if err != nil {
			return err
		}
		result = &artifact.ArtifactVersion{
			Version: version,
			// Derived from where the payload is, never from the metadata
			// file, which could name any path.
			CanonicalURI:   s.canonicalURI(loc, version),
			CustomMetadata: meta.CustomMetadata.values(),
		}
		if intact {
			result.CreateTime = meta.createTime()
			result.MimeType = cmp.Or(meta.MIMEType, textMIMEType)
			return nil
		}
		// The payload's own timestamp is the closest record of when the
		// version was created, and it stays the same between calls. A payload
		// that is gone as well means the version is being deleted.
		info, err := statFile(dir, loc.payload())
		if err != nil {
			return fileError("inspect artifact payload", err)
		}
		result.CreateTime = info.ModTime()
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &artifact.GetArtifactVersionResponse{ArtifactVersion: result}, nil
}

// Versions implements [artifact.Service].
func (s *service) Versions(ctx context.Context, req *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	if req == nil {
		return nil, invalid("versions request is nil")
	}
	loc, err := locate("versions", req.AppName, req.UserID, req.SessionID, req.FileName)
	if err != nil {
		return nil, err
	}
	dir, err := s.openDir(ctx, loc.versions(), false)
	if err != nil {
		return nil, notFound(err)
	}
	defer func() { _ = dir.Close() }()
	versions, err := listVersions(dir)
	if err != nil {
		return nil, notFound(err)
	}
	if len(versions) == 0 {
		return nil, errNotFound()
	}
	return &artifact.VersionsResponse{Versions: versions}, nil
}

// Delete implements [artifact.Service]. Deleting a version that does not
// exist is not an error.
func (s *service) Delete(ctx context.Context, req *artifact.DeleteRequest) error {
	if req == nil {
		return invalid("delete request is nil")
	}
	loc, err := locate("delete", req.AppName, req.UserID, req.SessionID, req.FileName)
	if err != nil {
		return err
	}
	dir, err := s.openDir(ctx, loc.versions(), false)
	if err == nil {
		err = removeVersions(dir, req.Version)
		_ = dir.Close()
	}
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	s.prune(loc)
	return nil
}

// prune removes the directories of an artifact that hold nothing any more,
// from its versions directory up to, but not including, the artifacts
// directory of its scope. Each step removes an empty directory only, so a
// remaining version, a save in progress or a nested artifact ends it. Tidying
// up is not what the caller asked for, so it runs even if the caller's context
// is done and its errors are not reported.
func (s *service) prune(loc location) {
	// dirs[i] is the open directory that contains names[i].
	names := append(strings.Split(loc.name, "/"), versionsDir)
	dirs := make([]*os.Root, 0, len(names)+1)
	defer func() {
		for _, dir := range dirs {
			_ = dir.Close()
		}
	}()
	dir, err := s.open(loc.scope, false)
	for i := 0; err == nil; i++ {
		dirs = append(dirs, dir)
		if i == len(names) {
			break
		}
		dir, err = openChild(dir, names[i])
	}
	// Only what openChild has shown to be a directory is removed: Remove
	// would delete a file or a symbolic link as readily, which are not this
	// service's to delete. A directory is closed before its removal, which
	// Windows requires.
	for i := len(dirs) - 2; i >= 0; i-- {
		_ = dirs[i+1].Close()
		if dirs[i].Remove(names[i]) != nil {
			return
		}
	}
}

// List implements [artifact.Service]. It returns the session's artifacts and
// the user-scoped ones, sorted.
func (s *service) List(ctx context.Context, req *artifact.ListRequest) (*artifact.ListResponse, error) {
	if req == nil {
		return nil, invalid("list request is nil")
	}
	if err := checkIDs("list", req.AppName, req.UserID, req.SessionID); err != nil {
		return nil, err
	}
	var names []string
	for _, userScoped := range []bool{false, true} {
		dir, err := s.openDir(ctx, scopePath(req.AppName, req.UserID, req.SessionID, userScoped), false)
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		prefix := ""
		if userScoped {
			prefix = userPrefix
		}
		names, err = appendArtifacts(names, dir, prefix, "")
		_ = dir.Close()
		if err != nil {
			return nil, err
		}
	}
	slices.Sort(names)
	return &artifact.ListResponse{FileNames: names}, nil
}

// appendArtifacts appends to names the filename of every artifact stored
// below the open directory dir that has a published version. name is the
// artifact name dir itself stands for, empty for the directory of a scope,
// and prefix is put in front of every filename.
//
// A filename is its directory path, so the listing needs no metadata and a
// damaged metadata.json cannot hide an artifact or rename it.
func appendArtifacts(names []string, dir *os.Root, prefix, name string) ([]string, error) {
	entries, err := fs.ReadDir(dir.FS(), ".")
	if err != nil {
		return names, fileError("list artifacts", err)
	}
	for _, entry := range entries {
		names, err = appendEntry(names, dir, entry.Name(), prefix, name)
		// Directories come and go while Save and Delete run, and one that is
		// gone holds nothing to list. Files and symbolic links are not part
		// of the layout.
		if err != nil && !errors.Is(err, fs.ErrNotExist) && !errors.Is(err, errUnexpectedEntry) {
			return nil, err
		}
	}
	return names, nil
}

// appendEntry is appendArtifacts for the directory entry of parent, where
// name is the artifact name parent stands for.
func appendEntry(names []string, parent *os.Root, entry, prefix, name string) ([]string, error) {
	dir, err := openChild(parent, entry)
	if err != nil {
		return names, fileError("open artifact directory", err)
	}
	defer func() { _ = dir.Close() }()
	if entry == versionsDir {
		versions, err := listVersions(dir)
		// Only a filename this service accepts is listed, which the name of
		// a scope's own directory, the empty one, is not. Anything else was
		// not saved through it, and the other operations would refuse it.
		if len(versions) > 0 && validName(name) {
			names = append(names, prefix+name)
		}
		return names, err
	}
	if name != "" {
		entry = name + "/" + entry
	}
	// Nothing this service saved lies below a directory it would not create,
	// or below one in a session that reads as a user-scoped name, which the
	// other operations would look up elsewhere.
	if !validDir(entry) || prefix == "" && strings.HasPrefix(entry, userPrefix) {
		return names, nil
	}
	return appendArtifacts(names, dir, prefix, entry)
}

// canonicalURI is the file URI of the payload of a version.
func (s *service) canonicalURI(loc location, version int64) string {
	payload := filepath.Join(s.rootDir, filepath.FromSlash(loc.versions()), strconv.FormatInt(version, 10), loc.payload())
	uriPath := filepath.ToSlash(payload)
	if !strings.HasPrefix(uriPath, "/") {
		uriPath = "/" + uriPath // A Windows path starts with its drive.
	}
	return (&url.URL{Scheme: "file", Path: uriPath}).String()
}

var _ artifact.Service = (*service)(nil)
