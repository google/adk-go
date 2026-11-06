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

// Package fileartifact stores versioned artifacts in a local directory.
//
// Create a service with New and pass it to runner.Config.ArtifactService:
//
//	service, err := fileartifact.New(fileartifact.Config{RootDir: "./artifacts"})
//	if err != nil {
//		return err
//	}
//	// Use service as the runner's ArtifactService.
package fileartifact

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/platform"
)

// Config configures a local artifact service.
type Config struct {
	// RootDir is the directory containing the service's files. It must be nonempty.
	// New creates it if necessary and resolves it to an absolute path.
	RootDir string
}

// New creates a filesystem-backed artifact service. The service retains no open
// files between calls and is safe for concurrent use, including by separate
// instances or processes sharing RootDir on a local filesystem.
//
// Saves publish complete versions by renaming a staging directory. Interrupted
// saves may leave staging directories; these are ignored by readers and their
// version numbers are skipped. Delete removes published versions; a concurrent
// Save can publish a new version after Delete returns.
//
// As in the other Go artifact services, versions start at 1, version 0 selects
// the latest version (or all versions for Delete), and filenames prefixed with
// "user:" are shared across that user's sessions. Unlike Python, filenames
// cannot contain path separators, and custom metadata values become strings.
// Filenames and metadata strings must be valid UTF-8.
//
// RootDir should be managed exclusively through this service. Its storage
// layout is private and is not interchangeable with Python's FileArtifactService.
// New directories have mode 0700 and files mode 0600, before applying the umask.
func New(cfg Config) (artifact.Service, error) {
	if cfg.RootDir == "" {
		return nil, fmt.Errorf("artifact root directory is required: %w", fs.ErrInvalid)
	}
	rootDir, err := filepath.Abs(cfg.RootDir)
	if err != nil {
		return nil, fileError("resolve artifact root", err)
	}
	if err := os.MkdirAll(rootDir, 0o700); err != nil {
		return nil, fileError("create artifact root", err)
	}
	rootDir, err = filepath.EvalSymlinks(rootDir)
	if err != nil {
		return nil, fileError("resolve artifact root", err)
	}
	return &fileService{rootDir: rootDir}, nil
}

type fileService struct{ rootDir string }

// Payload and metadata are published together. Text is stored separately from
// its MIME type so an inline blob with an empty MIME type remains a blob.
type metadata struct {
	FileName       string            `json:"fileName"`
	Text           bool              `json:"text"`
	MIMEType       string            `json:"mimeType"`
	DisplayName    string            `json:"displayName,omitempty"`
	CreateTime     *time.Time        `json:"createTime"`
	CustomMetadata map[string]string `json:"customMetadata"`
}

func (s *fileService) Save(ctx context.Context, req *artifact.SaveRequest) (response *artifact.SaveResponse, saveErr error) {
	if req == nil {
		return nil, fmt.Errorf("save request is required: %w", fs.ErrInvalid)
	}
	if err := req.Validate(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	createTime := platform.Now(ctx)
	m := metadata{
		FileName:       req.FileName,
		Text:           req.Part.InlineData == nil,
		MIMEType:       "text/plain",
		CreateTime:     &createTime,
		CustomMetadata: make(map[string]string, len(req.CustomMetadata)),
	}
	data := []byte(req.Part.Text)
	if blob := req.Part.InlineData; blob != nil {
		data = blob.Data
		m.MIMEType = blob.MIMEType
		m.DisplayName = blob.DisplayName
	}
	// encoding/json replaces invalid UTF-8; reject it before writing so names
	// still match their storage keys and metadata cannot silently change.
	if !utf8.ValidString(m.FileName) || !utf8.ValidString(m.MIMEType) || !utf8.ValidString(m.DisplayName) {
		return nil, fmt.Errorf("artifact filename and metadata must be valid UTF-8: %w", fs.ErrInvalid)
	}
	for key, value := range req.CustomMetadata {
		text := fmt.Sprint(value)
		if !utf8.ValidString(key) || !utf8.ValidString(text) {
			return nil, fmt.Errorf("artifact custom metadata must be valid UTF-8: %w", fs.ErrInvalid)
		}
		m.CustomMetadata[key] = text
	}
	encoded, err := json.Marshal(m)
	if err != nil {
		return nil, fmt.Errorf("encode artifact metadata: %w", fs.ErrInvalid)
	}
	path := versionsPath(req.AppName, req.UserID, req.SessionID, req.FileName)
	root, err := s.openDirectory(ctx, path, true)
	if err != nil {
		return nil, err
	}
	defer func() { _ = root.Close() }()
	version, pending, err := reserveVersion(ctx, root)
	if err != nil {
		return nil, err
	}
	// Leave no visible version after a failed save. A process killed before this
	// cleanup leaves only an ignored reservation, matching Python's staging scheme.
	defer func() {
		if saveErr != nil {
			if err := root.RemoveAll(pending); err != nil {
				saveErr = errors.Join(saveErr, fileError("remove artifact staging directory", err))
			}
		}
	}()
	stage, err := openChild(root, pending)
	if err != nil {
		return nil, fileError("open artifact staging directory", err)
	}
	defer func() { _ = stage.Close() }()
	if err := stage.WriteFile("data", data, 0o600); err != nil {
		return nil, fileError("write artifact payload", err)
	}
	if err := stage.WriteFile("metadata.json", encoded, 0o600); err != nil {
		return nil, fileError("write artifact metadata", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Close the directory handle before renaming, including on Windows.
	if err := stage.Close(); err != nil {
		return nil, fileError("close artifact staging directory", err)
	}
	if err := root.Rename(pending, strconv.FormatInt(version, 10)); err != nil {
		return nil, fileError("publish artifact version", err)
	}
	return &artifact.SaveResponse{Version: version}, nil
}

func (s *fileService) Load(ctx context.Context, req *artifact.LoadRequest) (*artifact.LoadResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("load request is required: %w", fs.ErrInvalid)
	}
	if err := req.Validate(); err != nil {
		return nil, err
	}
	root, _, err := s.openVersion(ctx, versionsPath(req.AppName, req.UserID, req.SessionID, req.FileName), req.Version)
	if err != nil {
		return nil, err
	}
	defer func() { _ = root.Close() }()
	m, err := readMetadata(root)
	if err != nil {
		return nil, err
	}
	data, err := root.ReadFile("data")
	if err != nil {
		return nil, fileError("read artifact payload", err)
	}
	if m.Text {
		return &artifact.LoadResponse{Part: genai.NewPartFromText(string(data))}, nil
	}
	return &artifact.LoadResponse{Part: &genai.Part{InlineData: &genai.Blob{Data: data, MIMEType: m.MIMEType, DisplayName: m.DisplayName}}}, nil
}

func (s *fileService) GetArtifactVersion(ctx context.Context, req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("get artifact version request is required: %w", fs.ErrInvalid)
	}
	if err := req.Validate(); err != nil {
		return nil, err
	}
	path := versionsPath(req.AppName, req.UserID, req.SessionID, req.FileName)
	root, version, err := s.openVersion(ctx, path, req.Version)
	if err != nil {
		return nil, err
	}
	defer func() { _ = root.Close() }()
	m, err := readMetadata(root)
	if err != nil {
		return nil, err
	}
	custom := make(map[string]any, len(m.CustomMetadata))
	for key, value := range m.CustomMetadata {
		custom[key] = value
	}
	// Derive the URI from the storage location, never from metadata read off disk.
	payloadPath := filepath.Join(s.rootDir, path, strconv.FormatInt(version, 10), "data")
	uriPath := filepath.ToSlash(payloadPath)
	if !strings.HasPrefix(uriPath, "/") {
		uriPath = "/" + uriPath
	}
	uri := (&url.URL{Scheme: "file", Path: uriPath}).String()
	return &artifact.GetArtifactVersionResponse{ArtifactVersion: &artifact.ArtifactVersion{
		Version:        version,
		CanonicalURI:   uri,
		CustomMetadata: custom,
		CreateTime:     *m.CreateTime,
		MimeType:       m.MIMEType,
	}}, nil
}

func (s *fileService) Versions(ctx context.Context, req *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("versions request is required: %w", fs.ErrInvalid)
	}
	if err := req.Validate(); err != nil {
		return nil, err
	}
	root, err := s.openDirectory(ctx, versionsPath(req.AppName, req.UserID, req.SessionID, req.FileName), false)
	if err != nil {
		return nil, err
	}
	defer func() { _ = root.Close() }()
	versions, err := listVersions(ctx, root)
	if err != nil {
		return nil, err
	}
	if len(versions) == 0 {
		return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
	}
	return &artifact.VersionsResponse{Versions: versions}, nil
}

func (s *fileService) Delete(ctx context.Context, req *artifact.DeleteRequest) error {
	if req == nil {
		return fmt.Errorf("delete request is required: %w", fs.ErrInvalid)
	}
	if err := req.Validate(); err != nil {
		return err
	}
	if req.Version < 0 {
		return fmt.Errorf("artifact version must be nonnegative: %w", fs.ErrInvalid)
	}
	root, err := s.openDirectory(ctx, versionsPath(req.AppName, req.UserID, req.SessionID, req.FileName), false)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer func() { _ = root.Close() }()
	versions := []int64{req.Version}
	if req.Version == 0 {
		versions, err = listVersions(ctx, root)
		if err != nil {
			return err
		}
	}
	for _, version := range versions {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := root.RemoveAll(strconv.FormatInt(version, 10)); err != nil {
			return fileError("delete artifact version", err)
		}
	}
	return nil
}

func (s *fileService) List(ctx context.Context, req *artifact.ListRequest) (*artifact.ListResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("list request is required: %w", fs.ErrInvalid)
	}
	if err := req.Validate(); err != nil {
		return nil, err
	}
	var names []string
	for _, userScoped := range []bool{false, true} {
		path := scopePath(req.AppName, req.UserID, req.SessionID, userScoped)
		root, err := s.openDirectory(ctx, path, false)
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		entries, err := fs.ReadDir(root.FS(), ".")
		_ = root.Close()
		if err != nil {
			return nil, fileError("list artifacts", err)
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			decoded, err := hex.DecodeString(entry.Name())
			if err != nil || len(decoded) != sha256.Size || hex.EncodeToString(decoded) != entry.Name() {
				continue
			}
			versionRoot, _, err := s.openVersion(ctx, filepath.Join(path, entry.Name(), "versions"), 0)
			if errors.Is(err, fs.ErrNotExist) {
				continue
			}
			if err != nil {
				return nil, err
			}
			m, err := readMetadata(versionRoot)
			_ = versionRoot.Close()
			if errors.Is(err, fs.ErrNotExist) {
				continue
			} // Concurrent deletion.
			if err != nil {
				return nil, err
			}
			if encodeName(m.FileName) != entry.Name() || strings.HasPrefix(m.FileName, "user:") != userScoped {
				return nil, fmt.Errorf("artifact metadata does not match its scope: %w", fs.ErrInvalid)
			}
			names = append(names, m.FileName)
		}
	}
	slices.Sort(names)
	return &artifact.ListResponse{FileNames: names}, nil
}

// Fixed-size hashes avoid path traversal, reserved names and case folding, and
// keep long logical names within filesystem component limits. The original
// filename lives in metadata for List; identifiers never become raw paths.
func encodeName(name string) string {
	sum := sha256.Sum256([]byte(name))
	return hex.EncodeToString(sum[:])
}

func scopePath(app, user, session string, userScoped bool) string {
	path := filepath.Join("apps", encodeName(app), "users", encodeName(user))
	if !userScoped {
		path = filepath.Join(path, "sessions", encodeName(session))
	}
	return filepath.Join(path, "artifacts")
}

func versionsPath(app, user, session, name string) string {
	return filepath.Join(scopePath(app, user, session, strings.HasPrefix(name, "user:")), encodeName(name), "versions")
}

func (s *fileService) openDirectory(ctx context.Context, path string, create bool) (*os.Root, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	root, err := os.OpenRoot(s.rootDir)
	if err != nil {
		return nil, fileError("open artifact root", err)
	}
	for _, component := range strings.Split(path, string(filepath.Separator)) {
		if err := ctx.Err(); err != nil {
			_ = root.Close()
			return nil, err
		}
		if create {
			if err := root.Mkdir(component, 0o700); err != nil && !errors.Is(err, fs.ErrExist) {
				_ = root.Close()
				return nil, fileError("create artifact directory", err)
			}
		}
		child, err := openChild(root, component)
		_ = root.Close()
		if err != nil {
			return nil, fileError("open artifact directory", err)
		}
		root = child
	}
	return root, nil
}

// Reject directory symlinks even within RootDir: they could alias another
// application's storage. Check the opened directory's identity as well, so a
// replacement between Lstat and OpenRoot cannot bypass that check.
func openChild(root *os.Root, name string) (*os.Root, error) {
	info, err := root.Lstat(name)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, fs.ErrInvalid
	}
	child, err := root.OpenRoot(name)
	if err != nil {
		return nil, err
	}
	opened, err := child.Stat(".")
	if err != nil {
		_ = child.Close()
		return nil, err
	}
	if !os.SameFile(info, opened) {
		_ = child.Close()
		return nil, fs.ErrInvalid
	}
	return child, nil
}

func (s *fileService) openVersion(ctx context.Context, path string, version int64) (*os.Root, int64, error) {
	if version < 0 {
		return nil, 0, fmt.Errorf("artifact version must be nonnegative: %w", fs.ErrInvalid)
	}
	root, err := s.openDirectory(ctx, path, false)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = root.Close() }()
	if version == 0 {
		versions, err := listVersions(ctx, root)
		if err != nil {
			return nil, 0, err
		}
		if len(versions) == 0 {
			return nil, 0, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
		}
		version = versions[len(versions)-1]
	}
	child, err := openChild(root, strconv.FormatInt(version, 10))
	if err != nil {
		return nil, 0, fileError("open artifact version", err)
	}
	return child, version, nil
}

func listVersions(ctx context.Context, root *os.Root) ([]int64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	entries, err := fs.ReadDir(root.FS(), ".")
	if err != nil {
		return nil, fileError("list artifact versions", err)
	}
	var versions []int64
	for _, entry := range entries {
		version, err := strconv.ParseInt(entry.Name(), 10, 64)
		if entry.IsDir() && err == nil && version > 0 && strconv.FormatInt(version, 10) == entry.Name() {
			versions = append(versions, version)
		}
	}
	slices.Sort(versions)
	return versions, nil
}

func reserveVersion(ctx context.Context, root *os.Root) (int64, string, error) {
	versions, err := listVersions(ctx, root)
	if err != nil {
		return 0, "", err
	}
	var previous int64
	if len(versions) > 0 {
		previous = versions[len(versions)-1]
	}
	for previous < math.MaxInt64 {
		if err := ctx.Err(); err != nil {
			return 0, "", err
		}
		previous++
		name := strconv.FormatInt(previous, 10)
		pending := "." + name + ".pending"
		if err := root.Mkdir(pending, 0o700); errors.Is(err, fs.ErrExist) {
			continue
		} else if err != nil {
			return 0, "", fileError("reserve artifact version", err)
		}
		// A writer can publish after our listing and before we reserve this number.
		_, err := root.Lstat(name)
		if errors.Is(err, fs.ErrNotExist) {
			return previous, pending, nil
		}
		if removeErr := root.Remove(pending); removeErr != nil {
			return 0, "", fileError("release artifact reservation", removeErr)
		}
		if err != nil {
			return 0, "", fileError("check artifact version", err)
		}
	}
	return 0, "", fmt.Errorf("artifact version limit reached: %w", fs.ErrInvalid)
}

func readMetadata(root *os.Root) (*metadata, error) {
	data, err := root.ReadFile("metadata.json")
	if err != nil {
		return nil, fileError("read artifact metadata", err)
	}
	var m metadata
	// Unlike Python's fallback, report corrupt metadata rather than guessing the
	// payload type or inventing a creation time for an existing version.
	if err := json.Unmarshal(data, &m); err != nil || m.FileName == "" || m.CreateTime == nil || m.CustomMetadata == nil {
		return nil, fmt.Errorf("invalid artifact metadata: %w", fs.ErrInvalid)
	}
	return &m, nil
}

// Filesystem errors include paths derived from customer identifiers. Preserve
// the cause for errors.Is without including those paths in diagnostics.
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

var _ artifact.Service = (*fileService)(nil)
