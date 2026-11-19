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

// Package fileartifact provides a file-backed artifact service.
//
// The service stores artifacts below a root directory in the layout of
// adk-python's FileArtifactService:
//
//	{root}/apps/{app}/users/{user}/sessions/{session}/artifacts/{name}/versions/{n}/{name}
//	{root}/apps/{app}/users/{user}/artifacts/{name}/versions/{n}/{name}
//
// The second form holds user-scoped artifacts: a file name with the "user:"
// prefix, or any call without a session ID. Beside each payload,
// metadata.json records the file name as the caller gave it, the media type,
// the custom metadata (as strings), the creation time and the display name of
// inline data. {name} is the file name without its "user:" prefix and
// surrounding white space. Text is stored as UTF-8 and loads as text; inline
// data loads with its media type, which defaults to
// "application/octet-stream". File data and artifact references are refused.
//
// Versions are numbered from 1, where adk-python numbers them from 0. A save
// writes into versions/.{n}.pending and publishes it with one rename, so a
// reader sees a version complete or not at all. A staging directory a killed
// save left behind keeps its number, so versions can have gaps.
//
// File names may not contain "/" or "\" (see [artifact.ValidateFileName]),
// may not be rooted, drive-qualified or contain "..", and may not lead out of
// their scope through a symbolic link. A save also refuses a name whose last
// element is "metadata.json" or that has a "versions" element, in any casing.
//
// Load reports a version that holds the rewind marker
// ([artifact.IsRewindTombstone]) as not found, as the in-memory service does,
// and a missing artifact or version as an error wrapping [fs.ErrNotExist].
// Versions returns version numbers in ascending order. To give agents
// file-backed artifacts, pass the service that [NewService] returns as the
// runner's ArtifactService.
package fileartifact

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"os/user"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode"

	"golang.org/x/text/cases"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/artifact/internal/pathquote"
	"google.golang.org/adk/v2/platform"
)

const (
	userNamespacePrefix = "user:"
	metadataFileName    = "metadata.json"
	versionsDirName     = "versions"
	defaultMIMEType     = "application/octet-stream"
)

// NewService creates a file-backed artifact service.
// The rootDir is the base path where artifacts are stored. A leading "~" path
// element expands to the current user's home directory ($HOME, or the
// password database entry when $HOME is not set), and a leading "~name"
// element to the home directory of the user name; when the home directory
// cannot be determined, NewService returns an error and creates nothing.
// The rootDir then resolves to an absolute path, creating the directory and
// its parents if missing.
// If rootDir is empty (""), it resolves to the current working directory at the
// time NewService is called. If the root exists as a regular file, it returns
// an error that wraps syscall.ENOTDIR. The ctx is not used.
// A symbolic link on the path from the root to a scope directory (the
// directory under "apps" that holds a user's or a session's artifacts) must
// be relative and stay inside the root; otherwise every call on that scope
// fails with an I/O error.
// The returned service is safe for concurrent use; concurrent use across
// different services sharing a root does not provide cross-process locking,
// but atomic renames provide baseline filesystem safety.
func NewService(ctx context.Context, rootDir string) (artifact.Service, error) {
	dir, err := expandHome(rootDir)
	if err != nil {
		return nil, err
	}
	dir, err = filepath.Abs(dir)
	if err != nil {
		return nil, ioError("resolve root directory", err)
	}
	if err := os.MkdirAll(dir, 0o777); err != nil {
		return nil, ioError("create root directory", err)
	}
	dir, err = filepath.EvalSymlinks(dir)
	if err != nil {
		return nil, ioError("resolve root directory", err)
	}
	return &service{rootDir: dir}, nil
}

// expandHome replaces a leading "~" or "~name" path element with the home
// directory of the current user or of the user name, as adk-python's
// Path.expanduser does. Any other "~" is left as it is.
func expandHome(dir string) (string, error) {
	if !strings.HasPrefix(dir, "~") {
		return dir, nil
	}
	name, rest := dir[1:], ""
	if i := strings.IndexAny(name, "/"+string(filepath.Separator)); i >= 0 {
		name, rest = name[:i], name[i:]
	}
	var home string
	var err error
	if name == "" {
		home, err = currentHomeDir()
	} else {
		home, err = userHomeDir(name)
	}
	if err != nil {
		return "", err
	}
	return filepath.Join(home, rest), nil
}

// currentHomeDir returns $HOME (or its platform equivalent), and the current
// user's home directory from the password database when that is not set.
func currentHomeDir() (string, error) {
	home, envErr := os.UserHomeDir()
	if envErr == nil {
		return home, nil
	}
	u, err := user.Current()
	if err != nil {
		return "", fmt.Errorf("fileartifact: could not determine home directory: %w", errors.Join(envErr, err))
	}
	return u.HomeDir, nil
}

// userHomeDir returns the home directory of the user name. Its error does not
// wrap the lookup's error, whose text holds the name.
func userHomeDir(name string) (string, error) {
	u, err := user.Lookup(name)
	if err != nil {
		return "", errors.New("fileartifact: could not determine home directory")
	}
	return u.HomeDir, nil
}

// service implements artifact.Service on a directory tree rooted at rootDir.
type service struct {
	// rootDir is absolute and has its symbolic links resolved.
	rootDir string
	// locks orders the calls of this service on one artifact: saves and
	// deletes take an artifact's lock for writing, the other calls for
	// reading. Services in other processes, or other services on the same
	// root, are not ordered; the atomic rename of a save is all they share.
	locks lockMap
}

// lockMap holds a read-write lock per artifact directory. An entry exists
// while a caller holds or waits for its lock, so the map does not grow with
// the number of artifacts ever touched.
type lockMap struct {
	mu    sync.Mutex // guards locks and the refs of every entry
	locks map[string]*artifactLock
}

type artifactLock struct {
	sync.RWMutex
	refs int
}

// acquire returns the lock of key and counts the caller as its user.
func (m *lockMap) acquire(key string) *artifactLock {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.locks == nil {
		m.locks = map[string]*artifactLock{}
	}
	l, ok := m.locks[key]
	if !ok {
		l = &artifactLock{}
		m.locks[key] = l
	}
	l.refs++
	return l
}

// release ends the caller's use of the lock of key, and drops the entry when
// no caller is left. Both happen under m.mu, so a caller that acquires the
// key afterwards gets a new entry that no one else holds.
func (m *lockMap) release(key string, l *artifactLock) {
	m.mu.Lock()
	defer m.mu.Unlock()
	l.refs--
	if l.refs == 0 {
		delete(m.locks, key)
	}
}

// lock locks the artifact directory key for writing and returns the function
// that unlocks it.
func (m *lockMap) lock(key string) (unlock func()) {
	l := m.acquire(key)
	l.Lock()
	return func() {
		l.Unlock()
		m.release(key, l)
	}
}

// rlock locks the artifact directory key for reading and returns the function
// that unlocks it.
func (m *lockMap) rlock(key string) (unlock func()) {
	l := m.acquire(key)
	l.RLock()
	return func() {
		l.RUnlock()
		m.release(key, l)
	}
}

var _ artifact.Service = (*service)(nil)

// target names one artifact on disk, after its request passed validation.
type target struct {
	// scopeRel is the scope directory, the one holding the artifact
	// directories, relative to the service root.
	scopeRel string
	// scopeAbs is the scope directory, absolute and resolved.
	scopeAbs string
	// rel is the artifact directory relative to the scope directory.
	rel string
}

// abs returns the artifact directory, absolute and resolved.
func (t target) abs() string { return filepath.Join(t.scopeAbs, t.rel) }

// baseName is the name the payload is stored under in a version directory.
func (t target) baseName() string { return filepath.Base(t.rel) }

func (t target) versionsRel() string { return filepath.Join(t.rel, versionsDirName) }

func (t target) versionRel(version int64) string {
	return filepath.Join(t.versionsRel(), strconv.FormatInt(version, 10))
}

// canonicalURI is the file:// URI of the payload of version, encoded as
// adk-python's Path.as_uri.
func (t target) canonicalURI(version int64) string {
	return fileURI(filepath.Join(t.abs(), versionsDirName, strconv.FormatInt(version, 10), t.baseName()))
}

// fileURI returns the file:// URI of the absolute path p.
func fileURI(p string) string {
	if vol := filepath.VolumeName(p); vol != "" {
		return "file:///" + vol + pathquote.Quote(filepath.ToSlash(p[len(vol):]))
	}
	return "file://" + pathquote.Quote(filepath.ToSlash(p))
}

// resolveTarget validates the names of a request that reads or deletes one
// artifact and returns where the artifact lives. The checks run in
// adk-python's order: app name, user ID, session ID (only when the artifact
// is session-scoped), then the file name. The separator rule shared with the
// other services comes last, so adk-python's own messages win.
func (s *service) resolveTarget(appName, userID, sessionID, fileName string) (target, error) {
	t, err := s.locate(appName, userID, sessionID, fileName)
	if err != nil {
		return target{}, err
	}
	if err := artifact.ValidateFileName(fileName); err != nil {
		return target{}, err
	}
	return t, nil
}

// resolveSaveTarget is resolveTarget for a save, which also refuses the names
// the layout reserves. Reads and deletes accept them, so an artifact stored
// under such a name stays readable and deletable.
func (s *service) resolveSaveTarget(appName, userID, sessionID, fileName string) (target, error) {
	t, err := s.locate(appName, userID, sessionID, fileName)
	if err != nil {
		return target{}, err
	}
	if err := checkReserved(t.rel); err != nil {
		return target{}, err
	}
	if err := artifact.ValidateFileName(fileName); err != nil {
		return target{}, err
	}
	return t, nil
}

// locate validates the IDs and the file name and returns where the artifact
// lives.
//
// As adk-python does, it resolves the symbolic links of the artifact path and
// refuses a path that leads out of the scope directory, before any write.
// Every later file operation goes through an os.Root at the scope directory
// as it is opened, so a link swapped in below the scope directory after this
// check cannot lead out of it; the os.Root then fails the operation with an
// I/O error. A link swapped in for the scope directory or one of its parents
// after this check is followed when the scope is opened, as in adk-python.
func (s *service) locate(appName, userID, sessionID, fileName string) (target, error) {
	scopeRel, err := scopeDir(appName, userID, sessionID, fileName)
	if err != nil {
		return target{}, err
	}
	rel, err := artifactPath(fileName)
	if err != nil {
		return target{}, err
	}
	scopeAbs, err := resolvePath(filepath.Join(s.rootDir, scopeRel))
	if err != nil {
		return target{}, err
	}
	dir, err := resolvePath(filepath.Join(scopeAbs, rel))
	if err != nil {
		return target{}, err
	}
	rel, err = filepath.Rel(scopeAbs, dir)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return target{}, invalidInput("artifact filename escapes storage directory")
	}
	if rel == "." {
		// adk-python stores a name that resolves to the scope directory under
		// "artifact", where it collides with an artifact of that name.
		return target{}, invalidInput("artifact filename must not be empty")
	}
	return target{scopeRel: scopeRel, scopeAbs: scopeAbs, rel: rel}, nil
}

// maxSymlinks bounds the dangling links resolvePath follows, as the kernel's
// ELOOP limit bounds the links in one lookup.
const maxSymlinks = 40

// resolvePath returns the absolute path p with its symbolic links resolved, as
// adk-python's Path.resolve(strict=False) does: the links in the part of p
// that exists are followed, dangling links included, and the rest of p is
// appended as it is.
func resolvePath(p string) (string, error) {
	return resolveLinks(p, maxSymlinks)
}

func resolveLinks(p string, budget int) (string, error) {
	resolved, err := filepath.EvalSymlinks(p)
	if err == nil {
		return resolved, nil
	}
	if !errors.Is(err, fs.ErrNotExist) && !errors.Is(err, syscall.ENOTDIR) {
		return "", ioError("resolve path", err)
	}
	parent := filepath.Dir(p)
	if parent == p {
		return p, nil
	}
	dir, err := resolveLinks(parent, budget)
	if err != nil {
		return "", err
	}
	candidate := filepath.Join(dir, filepath.Base(p))
	link, err := os.Readlink(candidate)
	switch {
	case errors.Is(err, syscall.EINVAL) || errors.Is(err, fs.ErrNotExist) || errors.Is(err, syscall.ENOTDIR):
		// Not a link, or nothing there: the rest of the path is literal.
		return candidate, nil
	case err != nil:
		return "", ioError("resolve path", err)
	case budget == 0:
		return "", ioError("resolve path", syscall.ELOOP)
	}
	if !filepath.IsAbs(link) {
		link = filepath.Join(dir, link)
	}
	return resolveLinks(link, budget-1)
}

// checkReserved refuses an artifact path whose last element is
// "metadata.json", where the payload would overwrite the metadata document,
// or with a "versions" element, which would nest a second version store. The
// elements are compared caselessly and without trailing dots and spaces,
// because case-insensitive file systems and Win32 resolve those forms to the
// same entry.
func checkReserved(rel string) error {
	fold := cases.Fold()
	normalize := func(elem string) string { return fold.String(strings.TrimRight(elem, " .")) }
	if normalize(filepath.Base(rel)) == metadataFileName {
		return invalidInput("artifact filename is reserved: an artifact may not be named 'metadata.json' (in any casing) because its payload is stored under the artifact's own name and would overwrite the metadata document")
	}
	for elem := range strings.SplitSeq(filepath.ToSlash(rel), "/") {
		if normalize(elem) == versionsDirName {
			return invalidInput("artifact filename is reserved: an artifact path may not contain a 'versions' component (in any casing) because that directory stores artifact versions")
		}
	}
	return nil
}

// scopeDir validates the app, user and session IDs and returns the scope
// directory relative to the service root. A "user:" file name or an empty
// session ID makes the artifact user-scoped, and the session ID is then
// neither used nor validated.
func scopeDir(appName, userID, sessionID, fileName string) (string, error) {
	base, err := userDir(appName, userID)
	if err != nil {
		return "", err
	}
	if sessionID == "" || strings.HasPrefix(fileName, userNamespacePrefix) {
		return filepath.Join(base, "artifacts"), nil
	}
	if err := artifact.ValidatePathSegment(sessionID, "session_id"); err != nil {
		return "", err
	}
	return filepath.Join(base, "sessions", sessionID, "artifacts"), nil
}

// userDir validates the app name and user ID and returns the directory that
// holds the user's artifacts, relative to the service root.
func userDir(appName, userID string) (string, error) {
	if err := artifact.ValidatePathSegment(appName, "app_name"); err != nil {
		return "", err
	}
	if err := artifact.ValidatePathSegment(userID, "user_id"); err != nil {
		return "", err
	}
	return filepath.Join("apps", appName, "users", userID), nil
}

// artifactPath checks fileName and returns the artifact directory relative to
// the scope directory: the name without its "user:" prefix and surrounding
// white space, with "\" read as a separator.
func artifactPath(fileName string) (string, error) {
	stripped := strings.TrimFunc(strings.TrimPrefix(fileName, userNamespacePrefix), isPythonSpace)
	if isRootedOrDriveQualified(stripped) {
		return "", invalidInput("rooted or drive-qualified artifact filename is not permitted; provide a path relative to the storage scope")
	}
	slashed := strings.ReplaceAll(stripped, `\`, "/")
	if slices.Contains(strings.Split(slashed, "/"), "..") {
		return "", invalidInput("artifact filename must not contain parent traversal")
	}
	return filepath.Clean(filepath.FromSlash(slashed)), nil
}

// isPythonSpace reports whether Python's str.strip removes r.
func isPythonSpace(r rune) bool {
	return unicode.IsSpace(r) || '\x1c' <= r && r <= '\x1f'
}

// isRootedOrDriveQualified reports whether name is rooted or drive-qualified in
// POSIX or Windows form: "/a", "\a", "//server/share", "\\server\share",
// "C:\a", "C:/a" or "C:a".
func isRootedOrDriveQualified(name string) bool {
	if strings.HasPrefix(name, "/") || strings.HasPrefix(name, `\`) {
		return true
	}
	if len(name) < 2 || name[1] != ':' {
		return false
	}
	c := name[0]
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z'
}

// openScope opens the scope directory of t. With create set it creates the
// directory first. The returned root confines every later path to the scope.
func (s *service) openScope(t target, create bool) (*os.Root, error) {
	root, err := os.OpenRoot(s.rootDir)
	if err != nil {
		return nil, ioError("open root directory", err)
	}
	defer closeDir(root)
	if create {
		if err := root.MkdirAll(t.scopeRel, 0o777); err != nil {
			return nil, ioError("create scope directory", err)
		}
	}
	scope, err := root.OpenRoot(t.scopeRel)
	if err != nil {
		return nil, ioError("open scope directory", err)
	}
	return scope, nil
}

// Save implements [artifact.Service].
func (s *service) Save(ctx context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	t, err := s.resolveSaveTarget(req.AppName, req.UserID, req.SessionID, req.FileName)
	if err != nil {
		return nil, err
	}
	payload, meta, err := encodePart(req.Part)
	if err != nil {
		return nil, err
	}
	meta.FileName = req.FileName
	meta.CustomMetadata = make(map[string]any, len(req.CustomMetadata))
	for k, v := range req.CustomMetadata {
		meta.CustomMetadata[k] = fmt.Sprint(v)
	}
	// Read the caller's clock before locking: a provider may call back into
	// the service.
	createTime := timeToSeconds(platform.Now(ctx))
	meta.CreateTime = &createTime

	defer s.locks.lock(t.abs())()
	scope, err := s.openScope(t, true)
	if err != nil {
		return nil, err
	}
	defer closeDir(scope)
	version, pending, err := reserveVersion(scope, t)
	if err != nil {
		return nil, err
	}
	meta.Version = version
	meta.CanonicalURI = t.canonicalURI(version)
	if err := publish(scope, t, pending, payload, meta); err != nil {
		// Readers never see a partial version: the staging directory goes,
		// and the earlier versions stay as they were.
		if rmErr := scope.RemoveAll(pending); rmErr != nil {
			return nil, errors.Join(err, ioError("remove staging directory", rmErr))
		}
		return nil, err
	}
	return &artifact.SaveResponse{Version: version}, nil
}

// publish writes the payload and the metadata of a version into its staging
// directory pending and renames that to the version directory.
func publish(scope *os.Root, t target, pending string, payload []byte, meta *metadata) error {
	doc, err := json.Marshal(meta)
	if err != nil {
		return fmt.Errorf("fileartifact: encode metadata: %w", err)
	}
	// Both files are created with the same mode, so the umask in effect for
	// this write gives them the same permission bits.
	if err := scope.WriteFile(filepath.Join(pending, t.baseName()), payload, 0o666); err != nil {
		return ioError("write payload", err)
	}
	if err := scope.WriteFile(filepath.Join(pending, metadataFileName), doc, 0o666); err != nil {
		return ioError("write metadata", err)
	}
	if err := scope.Rename(pending, t.versionRel(meta.Version)); err != nil {
		return ioError("publish version", err)
	}
	return nil
}

// maxReserveRetries bounds how often a save recreates the versions directory
// that a concurrent delete removed.
const maxReserveRetries = 5

// reserveVersion creates the staging directory of the next free version of t
// and returns the version and the directory, relative to the scope. It starts
// after the highest published version and moves past every number that
// another save has reserved or published. A staging directory a killed save
// left behind keeps its number for good, so versions can have gaps.
func reserveVersion(scope *os.Root, t target) (int64, string, error) {
	if err := scope.MkdirAll(t.versionsRel(), 0o777); err != nil {
		return 0, "", ioError("create versions directory", err)
	}
	versions, err := readVersions(scope, t)
	if err != nil {
		return 0, "", err
	}
	version := int64(1)
	if len(versions) > 0 {
		version = versions[len(versions)-1] + 1
	}
	for retries := 0; ; {
		pending := filepath.Join(t.versionsRel(), "."+strconv.FormatInt(version, 10)+".pending")
		err := scope.Mkdir(pending, 0o777)
		switch {
		case errors.Is(err, fs.ErrExist):
			version++
			continue
		case errors.Is(err, fs.ErrNotExist) && retries < maxReserveRetries:
			// A delete pruned the versions directory, or a parent shared with
			// another artifact, after this save created it.
			retries++
			if err := scope.MkdirAll(t.versionsRel(), 0o777); err != nil {
				return 0, "", ioError("create versions directory", err)
			}
			continue
		case err != nil:
			return 0, "", ioError("reserve version", err)
		}
		_, err = scope.Stat(t.versionRel(version))
		if errors.Is(err, fs.ErrNotExist) {
			return version, pending, nil
		}
		// Another writer published this version between the scan and the
		// reservation.
		if rmErr := scope.Remove(pending); rmErr != nil {
			return 0, "", ioError("release version", rmErr)
		}
		if err != nil {
			return 0, "", ioError("check version", err)
		}
		version++
	}
}

// encodePart checks part and returns the bytes to store for it and the
// metadata that describes them. The service stores inline data and text only:
// inline data needs data, except for the rewind marker, and a part whose every
// field is zero is empty text, since genai.Part cannot tell Text "" from an
// unset Text. File data and artifact references are refused.
func encodePart(part *genai.Part) ([]byte, *metadata, error) {
	switch {
	case part == nil:
		return nil, nil, invalidInput("artifact part must not be nil")
	case part.InlineData != nil:
		if part.InlineData.Data == nil && !artifact.IsRewindTombstone(part) {
			return nil, nil, invalidInput("artifact inline_data must contain data")
		}
		// Load reads a version without a media type as text, so inline data
		// always records one.
		mimeType := part.InlineData.MIMEType
		if mimeType == "" {
			mimeType = defaultMIMEType
		}
		return part.InlineData.Data, &metadata{MIMEType: mimeType, DisplayName: part.InlineData.DisplayName}, nil
	case part.Text != "" || reflect.ValueOf(*part).IsZero():
		return []byte(part.Text), &metadata{}, nil
	}
	return nil, nil, invalidInput("artifact must have either inline_data or text content")
}

// metadata is the metadata.json document of one version, in adk-python's
// camelCase form and key order.
type metadata struct {
	Version        int64          `json:"version"`
	CanonicalURI   string         `json:"canonicalUri"`
	CustomMetadata map[string]any `json:"customMetadata"`
	// CreateTime is in seconds since the Unix epoch.
	CreateTime  *float64 `json:"createTime,omitempty"`
	MIMEType    string   `json:"mimeType,omitempty"`
	FileName    string   `json:"fileName"`
	DisplayName string   `json:"displayName,omitempty"`
}

func timeToSeconds(t time.Time) float64 { return float64(t.UnixNano()) / 1e9 }

func secondsToTime(s float64) time.Time {
	sec, frac := math.Modf(s)
	return time.Unix(int64(sec), int64(math.Round(frac*1e9)))
}

// readMetadata returns the metadata of version, or nil when the document is
// missing, unreadable or malformed.
func readMetadata(scope *os.Root, t target, version int64) *metadata {
	doc, err := scope.ReadFile(filepath.Join(t.versionRel(version), metadataFileName))
	if err != nil {
		return nil
	}
	var meta metadata
	if err := json.Unmarshal(doc, &meta); err != nil {
		return nil
	}
	return &meta
}

// readVersions returns the published versions of t in ascending order, or none
// when the artifact has no versions directory.
func readVersions(scope *os.Root, t target) ([]int64, error) {
	dir, err := scope.Open(t.versionsRel())
	if errors.Is(err, fs.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) {
		return nil, nil
	}
	if err != nil {
		return nil, ioError("open versions directory", err)
	}
	defer closeDir(dir)
	entries, err := dir.ReadDir(-1)
	if err != nil {
		return nil, ioError("read versions directory", err)
	}
	var versions []int64
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if v, ok := parseVersion(e.Name()); ok {
			versions = append(versions, v)
		}
	}
	slices.Sort(versions)
	return versions, nil
}

// parseVersion parses the name of a version directory. Only canonical
// positive decimal numbers are versions: no sign, no leading zero, no other
// character, so two directories never name one version.
func parseVersion(name string) (int64, bool) {
	if name == "" || name[0] < '1' || name[0] > '9' {
		return 0, false
	}
	for i := 1; i < len(name); i++ {
		if name[i] < '0' || name[i] > '9' {
			return 0, false
		}
	}
	v, err := strconv.ParseInt(name, 10, 64)
	return v, err == nil
}

// openVersion opens the scope of t and picks the version a read asks for:
// the latest when version is 0. It returns an error wrapping fs.ErrNotExist
// when the artifact or the version does not exist.
func (s *service) openVersion(t target, version int64) (*os.Root, int64, error) {
	scope, err := s.openScope(t, false)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, 0, notFound()
	}
	if err != nil {
		return nil, 0, err
	}
	versions, err := readVersions(scope, t)
	if err != nil {
		closeDir(scope)
		return nil, 0, err
	}
	switch {
	case len(versions) == 0:
		closeDir(scope)
		return nil, 0, notFound()
	case version == 0:
		return scope, versions[len(versions)-1], nil
	case slices.Contains(versions, version):
		return scope, version, nil
	}
	closeDir(scope)
	return nil, 0, notFound()
}

// Load implements [artifact.Service].
func (s *service) Load(ctx context.Context, req *artifact.LoadRequest) (*artifact.LoadResponse, error) {
	t, err := s.resolveTarget(req.AppName, req.UserID, req.SessionID, req.FileName)
	if err != nil {
		return nil, err
	}
	defer s.locks.rlock(t.abs())()
	scope, version, err := s.openVersion(t, req.Version)
	if err != nil {
		return nil, err
	}
	defer closeDir(scope)
	meta := readMetadata(scope, t, version)
	// The payload path comes from the layout only, never from the metadata
	// document, which anyone who can write to the tree controls.
	data, err := scope.ReadFile(filepath.Join(t.versionRel(version), t.baseName()))
	if err != nil {
		return nil, ioError("read payload", err)
	}
	if meta == nil || meta.MIMEType == "" {
		return &artifact.LoadResponse{Part: &genai.Part{Text: string(data)}}, nil
	}
	part := &genai.Part{InlineData: &genai.Blob{Data: data, MIMEType: meta.MIMEType, DisplayName: meta.DisplayName}}
	if artifact.IsRewindTombstone(part) {
		// A rewind saved this version to remove the artifact; earlier
		// versions still load.
		return nil, notFound()
	}
	return &artifact.LoadResponse{Part: part}, nil
}

// Delete implements [artifact.Service].
func (s *service) Delete(ctx context.Context, req *artifact.DeleteRequest) error {
	t, err := s.resolveTarget(req.AppName, req.UserID, req.SessionID, req.FileName)
	if err != nil {
		return err
	}
	defer s.locks.lock(t.abs())()
	scope, err := s.openScope(t, false)
	if errors.Is(err, fs.ErrNotExist) {
		if req.Version != 0 {
			return notFound()
		}
		return nil
	}
	if err != nil {
		return err
	}
	defer closeDir(scope)
	if req.Version != 0 {
		versions, err := readVersions(scope, t)
		if err != nil {
			return err
		}
		if !slices.Contains(versions, req.Version) {
			return notFound()
		}
		if err := scope.RemoveAll(t.versionRel(req.Version)); err != nil {
			return ioError("delete version", err)
		}
		if len(versions) > 1 {
			return nil
		}
	} else if _, err := scope.Stat(t.versionsRel()); errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	// Only the artifact's own versions go: its directory may also hold
	// artifacts nested under it, so it is removed only when empty.
	if err := scope.RemoveAll(t.versionsRel()); err != nil {
		return ioError("delete versions", err)
	}
	pruneEmptyDirs(scope, t.rel)
	return nil
}

// pruneEmptyDirs removes dir and then each parent it leaves empty, stopping
// below the scope directory.
func pruneEmptyDirs(scope *os.Root, dir string) {
	for dir != "." {
		// Remove fails on a directory that is not empty, which ends the walk.
		if scope.Remove(dir) != nil {
			return
		}
		dir = filepath.Dir(dir)
	}
}

// List implements [artifact.Service].
func (s *service) List(ctx context.Context, req *artifact.ListRequest) (*artifact.ListResponse, error) {
	base, err := userDir(req.AppName, req.UserID)
	if err != nil {
		return nil, err
	}
	names := map[string]bool{}
	if req.SessionID != "" {
		if err := artifact.ValidatePathSegment(req.SessionID, "session_id"); err != nil {
			return nil, err
		}
		if err := s.collectNames(names, filepath.Join(base, "sessions", req.SessionID, "artifacts"), ""); err != nil {
			return nil, err
		}
	}
	if err := s.collectNames(names, filepath.Join(base, "artifacts"), userNamespacePrefix); err != nil {
		return nil, err
	}
	fileNames := make([]string, 0, len(names))
	for name := range names {
		fileNames = append(fileNames, name)
	}
	slices.Sort(fileNames)
	return &artifact.ListResponse{FileNames: fileNames}, nil
}

// collectNames adds to names the file name of every artifact under the scope
// directory scopeRel: the name its latest version was saved under or, when
// that metadata is unreadable, prefix and its slash-separated path in the
// scope.
func (s *service) collectNames(names map[string]bool, scopeRel, prefix string) error {
	scopeAbs, err := resolvePath(filepath.Join(s.rootDir, scopeRel))
	if err != nil {
		return err
	}
	scope, err := s.openScope(target{scopeRel: scopeRel}, false)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer closeDir(scope)
	var dirs []string
	err = fs.WalkDir(scope.FS(), ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			// A concurrent Delete can remove an artifact directory between
			// reading its parent and opening it.
			if errors.Is(err, fs.ErrNotExist) {
				return nil
			}
			return err
		}
		if d.Name() != versionsDirName || p == "." {
			return nil
		}
		// A versions entry marks its parent as an artifact directory. The
		// parent can also hold artifacts nested under it, so the walk goes on
		// and skips only the stored versions.
		dirs = append(dirs, filepath.Dir(filepath.FromSlash(p)))
		if d.IsDir() {
			return fs.SkipDir
		}
		return nil
	})
	if err != nil {
		return ioError("list artifacts", err)
	}
	for _, rel := range dirs {
		name, exists, err := s.artifactName(scope, target{scopeRel: scopeRel, scopeAbs: scopeAbs, rel: rel})
		if err != nil {
			return err
		}
		if !exists {
			continue
		}
		if name == "" {
			name = prefix + filepath.ToSlash(rel)
		}
		names[name] = true
	}
	return nil
}

// artifactName returns the file name the latest version of t was saved
// under, "" when the artifact has no version or its metadata is unreadable,
// and exists == false when a concurrent Delete removed its versions entry.
func (s *service) artifactName(scope *os.Root, t target) (name string, exists bool, err error) {
	defer s.locks.rlock(t.abs())()
	if _, statErr := scope.Stat(t.versionsRel()); errors.Is(statErr, fs.ErrNotExist) {
		return "", false, nil
	}
	versions, err := readVersions(scope, t)
	if err != nil || len(versions) == 0 {
		return "", true, err
	}
	if meta := readMetadata(scope, t, versions[len(versions)-1]); meta != nil {
		return meta.FileName, true, nil
	}
	return "", true, nil
}

// Versions implements [artifact.Service].
func (s *service) Versions(ctx context.Context, req *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	t, err := s.resolveTarget(req.AppName, req.UserID, req.SessionID, req.FileName)
	if err != nil {
		return nil, err
	}
	defer s.locks.rlock(t.abs())()
	scope, err := s.openScope(t, false)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, notFound()
	}
	if err != nil {
		return nil, err
	}
	defer closeDir(scope)
	versions, err := readVersions(scope, t)
	if err != nil {
		return nil, err
	}
	if len(versions) == 0 {
		return nil, notFound()
	}
	return &artifact.VersionsResponse{Versions: versions}, nil
}

// GetArtifactVersion implements [artifact.Service].
func (s *service) GetArtifactVersion(ctx context.Context, req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	t, err := s.resolveTarget(req.AppName, req.UserID, req.SessionID, req.FileName)
	if err != nil {
		return nil, err
	}
	// Read the caller's clock before locking: a provider may call back into
	// the service.
	now := platform.Now(ctx)
	defer s.locks.rlock(t.abs())()
	scope, version, err := s.openVersion(t, req.Version)
	if err != nil {
		return nil, err
	}
	defer closeDir(scope)
	av := &artifact.ArtifactVersion{
		Version:        version,
		CanonicalURI:   t.canonicalURI(version),
		CustomMetadata: map[string]any{},
		CreateTime:     now,
		// Load reads a version without a media type as text.
		MimeType: "text/plain",
	}
	if meta := readMetadata(scope, t, version); meta != nil {
		for k, v := range meta.CustomMetadata {
			av.CustomMetadata[k] = fmt.Sprint(v)
		}
		if meta.CreateTime != nil {
			av.CreateTime = secondsToTime(*meta.CreateTime)
		}
		if meta.MIMEType != "" {
			av.MimeType = meta.MIMEType
		}
	}
	return &artifact.GetArtifactVersionResponse{ArtifactVersion: av}, nil
}

// invalidInput returns an error with the text msg that wraps
// artifact.ErrInputValidation. msg carries adk-python's phrase and never the
// caller's values.
func invalidInput(msg string) error {
	return fmt.Errorf("%w: %s", artifact.ErrInputValidation, msg)
}

// closeDir closes a directory or an os.Root. Closing a directory loses no
// data, so its error carries nothing for the caller.
func closeDir(c io.Closer) { _ = c.Close() }

func notFound() error {
	return fmt.Errorf("fileartifact: artifact not found: %w", fs.ErrNotExist)
}

// ioError wraps the file system error err for the caller without the path it
// names, which holds the caller's app, user and file names.
func ioError(op string, err error) error {
	if pathErr, ok := errors.AsType[*fs.PathError](err); ok {
		err = pathErr.Err
	}
	if linkErr, ok := errors.AsType[*os.LinkError](err); ok {
		err = linkErr.Err
	}
	return fmt.Errorf("fileartifact: %s: %w", op, err)
}
