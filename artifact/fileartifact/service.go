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

// Package fileartifact provides a local filesystem [artifact.Service].
//
// Artifacts survive process restarts and need no cloud dependency. A service is
// rooted at a directory and scopes artifacts by application, user, and session:
//
//	root/apps/{app}/users/{user}/sessions/{session}/artifacts/{file}/versions/{n}/
//	root/apps/{app}/users/{user}/artifacts/{file}/versions/{n}/
//
// The second layout holds user-scoped artifacts, whose file name starts with
// "user:". Each version directory contains a "payload" file with the artifact
// bytes and a "metadata.json" document describing them.
//
// Path components are base64url-encoded without padding, so any UTF-8 name is
// stored as one filesystem-safe component. This keeps names such as "..", "CON",
// or names containing a colon out of the path syntax and avoids collisions
// between the metadata document and an artifact named "metadata.json".
//
// Save reserves a version by creating its directory, writes the payload, and
// publishes the version by renaming a complete metadata document into place. A
// version whose metadata document is missing is ignored by reads, so a save
// interrupted mid-write never exposes a partial artifact.
package fileartifact

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/platform"
)

const (
	appsDir          = "apps"
	usersDir         = "users"
	sessionsDir      = "sessions"
	artifactsDir     = "artifacts"
	versionsDirName  = "versions"
	payloadFileName  = "payload"
	metadataFileName = "metadata.json"

	userNamespacePrefix = "user:"
)

// service is a local filesystem implementation of [artifact.Service].
type service struct {
	root string
}

// NewService returns an [artifact.Service] that stores artifacts beneath
// rootDir, creating rootDir if it does not exist. The returned service is safe
// for concurrent use by multiple goroutines and by separate processes sharing
// the directory.
func NewService(rootDir string) (artifact.Service, error) {
	root, err := filepath.Abs(rootDir)
	if err != nil {
		return nil, fmt.Errorf("resolve artifact root: %w", err)
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, fmt.Errorf("create artifact root: %w", err)
	}
	return &service{root: root}, nil
}

// artifactMetadata is the on-disk description of one artifact version.
type artifactMetadata struct {
	FileName       string            `json:"fileName"`
	MIMEType       string            `json:"mimeType,omitempty"`
	IsText         bool              `json:"isText,omitempty"`
	DisplayName    string            `json:"displayName,omitempty"`
	CustomMetadata map[string]string `json:"customMetadata,omitempty"`
	CreateTime     time.Time         `json:"createTime"`
}

// Save implements [artifact.Service].
func (s *service) Save(ctx context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}

	payload, metadata, err := partToDisk(req.Part)
	if err != nil {
		return nil, err
	}
	metadata.FileName = req.FileName
	metadata.CreateTime = platform.Now(ctx)
	if req.CustomMetadata != nil {
		metadata.CustomMetadata = make(map[string]string, len(req.CustomMetadata))
		for key, value := range req.CustomMetadata {
			metadata.CustomMetadata[key] = fmt.Sprint(value)
		}
	}

	artifactDir := s.artifactDir(req.AppName, req.UserID, req.SessionID, req.FileName)
	versionsDir := filepath.Join(artifactDir, versionsDirName)
	if err := os.MkdirAll(versionsDir, 0o755); err != nil {
		return nil, fmt.Errorf("create artifact versions directory: %w", err)
	}

	version, versionDir, err := reserveVersion(versionsDir)
	if err != nil {
		return nil, err
	}
	if err := publishVersion(versionDir, payload, metadata); err != nil {
		if removeErr := os.RemoveAll(versionDir); removeErr != nil && !errors.Is(removeErr, fs.ErrNotExist) {
			err = errors.Join(err, fmt.Errorf("discard unpublished version: %w", removeErr))
		}
		return nil, err
	}
	return &artifact.SaveResponse{Version: version}, nil
}

// partToDisk converts a validated part to its payload bytes and metadata. The
// caller fills in the file name and create time.
func partToDisk(part *genai.Part) ([]byte, *artifactMetadata, error) {
	if part.InlineData != nil {
		return part.InlineData.Data, &artifactMetadata{
			MIMEType:    part.InlineData.MIMEType,
			DisplayName: part.InlineData.DisplayName,
		}, nil
	}
	if part.Text != "" {
		return []byte(part.Text), &artifactMetadata{IsText: true}, nil
	}
	return nil, nil, fmt.Errorf("artifact part has neither text nor inline data")
}

// reserveVersion atomically claims the next numeric version directory under
// versionsDir. Unpublished directories left by an interrupted save keep their
// version reserved, so versions are not guaranteed to be contiguous.
func reserveVersion(versionsDir string) (int64, string, error) {
	next := int64(1)
	entries, err := os.ReadDir(versionsDir)
	if err != nil {
		return 0, "", fmt.Errorf("list artifact versions: %w", err)
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		version, err := strconv.ParseInt(entry.Name(), 10, 64)
		if err != nil {
			continue
		}
		if version >= next {
			next = version + 1
		}
	}

	for {
		versionDir := filepath.Join(versionsDir, strconv.FormatInt(next, 10))
		err := os.Mkdir(versionDir, 0o755)
		switch {
		case err == nil:
			return next, versionDir, nil
		case errors.Is(err, fs.ErrExist):
			next++
		default:
			return 0, "", fmt.Errorf("reserve artifact version: %w", err)
		}
	}
}

// publishVersion writes payload and metadata into the reserved versionDir. The
// metadata document is renamed into place last, so its presence marks the
// version as complete.
func publishVersion(versionDir string, payload []byte, metadata *artifactMetadata) error {
	if err := os.WriteFile(filepath.Join(versionDir, payloadFileName), payload, 0o644); err != nil {
		return fmt.Errorf("write artifact payload: %w", err)
	}

	encoded, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("encode artifact metadata: %w", err)
	}
	tmp, err := os.CreateTemp(versionDir, metadataFileName+".*.tmp")
	if err != nil {
		return fmt.Errorf("create artifact metadata: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(encoded); err != nil {
		tmp.Close()
		return fmt.Errorf("write artifact metadata: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close artifact metadata: %w", err)
	}
	if err := os.Rename(tmpName, filepath.Join(versionDir, metadataFileName)); err != nil {
		return fmt.Errorf("publish artifact metadata: %w", err)
	}
	return nil
}

// Load implements [artifact.Service].
func (s *service) Load(ctx context.Context, req *artifact.LoadRequest) (*artifact.LoadResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	artifactDir := s.artifactDir(req.AppName, req.UserID, req.SessionID, req.FileName)
	_, versionDir, err := resolveVersion(artifactDir, req.Version)
	if err != nil {
		return nil, err
	}
	metadata, err := readMetadata(versionDir)
	if err != nil {
		return nil, err
	}
	payload, err := os.ReadFile(filepath.Join(versionDir, payloadFileName))
	if err != nil {
		return nil, fmt.Errorf("read artifact payload: %w", err)
	}
	if metadata.IsText {
		return &artifact.LoadResponse{Part: &genai.Part{Text: string(payload)}}, nil
	}
	return &artifact.LoadResponse{Part: &genai.Part{InlineData: &genai.Blob{
		Data:        payload,
		MIMEType:    metadata.MIMEType,
		DisplayName: metadata.DisplayName,
	}}}, nil
}

// Delete implements [artifact.Service]. Deleting a missing artifact or version
// is not an error.
func (s *service) Delete(ctx context.Context, req *artifact.DeleteRequest) error {
	if err := req.Validate(); err != nil {
		return fmt.Errorf("request validation failed: %w", err)
	}
	artifactDir := s.artifactDir(req.AppName, req.UserID, req.SessionID, req.FileName)
	target := artifactDir
	if req.Version != 0 {
		target = filepath.Join(artifactDir, versionsDirName, strconv.FormatInt(req.Version, 10))
	}
	if err := os.RemoveAll(target); err != nil {
		return fmt.Errorf("delete artifact: %w", err)
	}
	pruneEmptyDirs(filepath.Dir(target), s.root)
	return nil
}

// pruneEmptyDirs removes leaf and its now-empty parents, stopping at root.
// os.Remove only succeeds on an empty directory, so an artifact that still
// holds versions or siblings is left alone.
func pruneEmptyDirs(leaf, root string) {
	for dir := leaf; dir != root && strings.HasPrefix(dir, root+string(filepath.Separator)); dir = filepath.Dir(dir) {
		if err := os.Remove(dir); err != nil {
			return
		}
	}
}

// List implements [artifact.Service]. It reports the session's artifacts and
// the user-scoped artifacts visible from that session, sorted by file name.
func (s *service) List(ctx context.Context, req *artifact.ListRequest) (*artifact.ListResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	userRoot := s.userRoot(req.AppName, req.UserID)
	files := map[string]bool{}
	if err := collectPublishedFiles(filepath.Join(userRoot, sessionsDir, encodeName(req.SessionID), artifactsDir), files); err != nil {
		return nil, err
	}
	if err := collectPublishedFiles(filepath.Join(userRoot, artifactsDir), files); err != nil {
		return nil, err
	}
	filenames := make([]string, 0, len(files))
	for name := range files {
		filenames = append(filenames, name)
	}
	sort.Strings(filenames)
	return &artifact.ListResponse{FileNames: filenames}, nil
}

// collectPublishedFiles adds the decoded names of published artifacts directly
// beneath artifactsDir. A name that is not valid base64url is not one this
// service wrote, so it is skipped.
func collectPublishedFiles(artifactsDir string, files map[string]bool) error {
	entries, err := os.ReadDir(artifactsDir)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("list artifacts: %w", err)
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name, err := decodeName(entry.Name())
		if err != nil {
			continue
		}
		versions, err := publishedVersions(filepath.Join(artifactsDir, entry.Name()))
		if err != nil {
			return err
		}
		if len(versions) > 0 {
			files[name] = true
		}
	}
	return nil
}

// Versions implements [artifact.Service] and returns fs.ErrNotExist when the
// artifact has no published version.
func (s *service) Versions(ctx context.Context, req *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	artifactDir := s.artifactDir(req.AppName, req.UserID, req.SessionID, req.FileName)
	versions, err := publishedVersions(artifactDir)
	if err != nil {
		return nil, err
	}
	if len(versions) == 0 {
		return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
	}
	return &artifact.VersionsResponse{Versions: versions}, nil
}

// GetArtifactVersion implements [artifact.Service] and returns fs.ErrNotExist
// when the artifact or requested version does not exist.
func (s *service) GetArtifactVersion(ctx context.Context, req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	artifactDir := s.artifactDir(req.AppName, req.UserID, req.SessionID, req.FileName)
	version, versionDir, err := resolveVersion(artifactDir, req.Version)
	if err != nil {
		return nil, err
	}
	metadata, err := readMetadata(versionDir)
	if err != nil {
		return nil, err
	}
	customMetadata := make(map[string]any, len(metadata.CustomMetadata))
	for key, value := range metadata.CustomMetadata {
		customMetadata[key] = value
	}
	mimeType := metadata.MIMEType
	if metadata.IsText {
		mimeType = "text/plain"
	}
	return &artifact.GetArtifactVersionResponse{
		ArtifactVersion: &artifact.ArtifactVersion{
			Version:        version,
			CanonicalURI:   fileURI(filepath.Join(versionDir, payloadFileName)),
			CustomMetadata: customMetadata,
			CreateTime:     metadata.CreateTime,
			MimeType:       mimeType,
		},
	}, nil
}

// resolveVersion returns the requested published version, or the latest one
// when requested is zero. A missing version is reported as fs.ErrNotExist.
func resolveVersion(artifactDir string, requested int64) (int64, string, error) {
	versions, err := publishedVersions(artifactDir)
	if err != nil {
		return 0, "", err
	}
	if len(versions) == 0 {
		return 0, "", fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
	}
	version := versions[len(versions)-1]
	if requested != 0 {
		index := sort.Search(len(versions), func(i int) bool { return versions[i] >= requested })
		if index == len(versions) || versions[index] != requested {
			return 0, "", fmt.Errorf("artifact version not found: %w", fs.ErrNotExist)
		}
		version = requested
	}
	return version, filepath.Join(artifactDir, versionsDirName, strconv.FormatInt(version, 10)), nil
}

// publishedVersions returns the numeric version directories under artifactDir
// that contain a metadata document, in ascending order.
func publishedVersions(artifactDir string) ([]int64, error) {
	versionsDir := filepath.Join(artifactDir, versionsDirName)
	entries, err := os.ReadDir(versionsDir)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("list artifact versions: %w", err)
	}
	var versions []int64
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		version, err := strconv.ParseInt(entry.Name(), 10, 64)
		if err != nil || version <= 0 {
			continue
		}
		versionDir := filepath.Join(versionsDir, entry.Name())
		info, err := os.Stat(filepath.Join(versionDir, metadataFileName))
		if err != nil || info.IsDir() {
			continue
		}
		versions = append(versions, version)
	}
	sort.Slice(versions, func(i, j int) bool { return versions[i] < versions[j] })
	return versions, nil
}

// readMetadata loads the metadata document for a published version.
func readMetadata(versionDir string) (*artifactMetadata, error) {
	data, err := os.ReadFile(filepath.Join(versionDir, metadataFileName))
	if err != nil {
		return nil, fmt.Errorf("read artifact metadata: %w", err)
	}
	var metadata artifactMetadata
	if err := json.Unmarshal(data, &metadata); err != nil {
		return nil, fmt.Errorf("decode artifact metadata: %w", err)
	}
	return &metadata, nil
}

// artifactDir returns the directory holding every version of one artifact.
func (s *service) artifactDir(appName, userID, sessionID, fileName string) string {
	userRoot := s.userRoot(appName, userID)
	if fileHasUserNamespace(fileName) {
		return filepath.Join(userRoot, artifactsDir, encodeName(fileName))
	}
	return filepath.Join(userRoot, sessionsDir, encodeName(sessionID), artifactsDir, encodeName(fileName))
}

// userRoot returns the directory holding one user's artifacts for one app.
func (s *service) userRoot(appName, userID string) string {
	return filepath.Join(s.root, appsDir, encodeName(appName), usersDir, encodeName(userID))
}

// fileHasUserNamespace reports whether a file name marks a user-scoped
// artifact, matching the other artifact services.
func fileHasUserNamespace(fileName string) bool {
	return strings.HasPrefix(fileName, userNamespacePrefix)
}

// encodeName renders a name as a single, filesystem-safe path component.
func encodeName(name string) string {
	return base64.RawURLEncoding.EncodeToString([]byte(name))
}

// decodeName reverses encodeName.
func decodeName(name string) (string, error) {
	decoded, err := base64.RawURLEncoding.DecodeString(name)
	if err != nil {
		return "", err
	}
	return string(decoded), nil
}

// fileURI returns a file:// URI for path, using a leading slash for Windows
// drive paths so the result has the usual file:///D:/... form.
func fileURI(path string) string {
	if abs, err := filepath.Abs(path); err == nil {
		path = abs
	}
	slash := filepath.ToSlash(path)
	if !strings.HasPrefix(slash, "/") {
		slash = "/" + slash
	}
	return (&url.URL{Scheme: "file", Path: slash}).String()
}

var _ artifact.Service = (*service)(nil)
