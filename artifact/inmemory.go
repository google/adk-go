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
	"context"
	"fmt"
	"io/fs"
	"iter"
	"maps"
	"math"
	"net/url"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"google.golang.org/genai"
	"rsc.io/omap"
	"rsc.io/ordered"

	"google.golang.org/adk/v2/platform"
)

type artifactEntry struct {
	part           *genai.Part
	createTime     time.Time
	customMetadata map[string]any
}

// inMemoryService is an in-memory implementation of the Service.
// It is primarily for testing and demonstration purposes.
type inMemoryService struct {
	mu sync.RWMutex
	// ordered(appName, userID, sessionID) -> session
	artifacts omap.Map[string, *artifactEntry]
}

// InMemoryService returns a new in-memory artifact service.
func InMemoryService() Service {
	return &inMemoryService{}
}

// fileHasUserNamespace checks if a filename indicates a user scoped artifact.
func fileHasUserNamespace(filename string) bool {
	return strings.HasPrefix(filename, "user:")
}

// userScopedArtifactKey defines the string for the part of the path used by user scope files
// this key is used to store user artifacts within user namespace (available to all sessions for given app+user_id)
const userScopedArtifactKey = "user"

type artifactKey struct {
	AppName   string
	UserID    string
	SessionID string
	FileName  string
	Version   int64
}

// Encode encodes the artifactKey into a string.
func (ak artifactKey) Encode() string {
	return string(ordered.Encode(ak.AppName, ak.UserID, ak.SessionID, ak.FileName, ordered.Rev(ak.Version)))
}

// Decode decodes the string key into an artifactKey.
func (ak *artifactKey) Decode(key string) error {
	var v ordered.Reverse[int64]
	err := ordered.Decode([]byte(key), &ak.AppName, &ak.UserID, &ak.SessionID, &ak.FileName, &v)
	if err != nil {
		return err
	}
	ak.Version = v.Value()
	return nil
}

// scan returns an iterator over all key-value pairs
// in the range begin ≤ key ≤ end.
// TODO: add a concurrent tests.
func (s *inMemoryService) scan(lo, hi string) iter.Seq2[artifactKey, *artifactEntry] {
	return func(yield func(key artifactKey, val *artifactEntry) bool) {
		for k, val := range s.artifacts.Scan(lo, hi) {
			var key artifactKey
			if err := key.Decode(k); err != nil {
				continue
			}

			if !yield(key, val) {
				return
			}
		}
	}
}

func (s *inMemoryService) find(appName, userID, sessionID, fileName string) (int64, *artifactEntry, bool) {
	lo := artifactKey{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Version: math.MaxInt64}.Encode()
	hi := artifactKey{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Version: 0}.Encode()
	for key, val := range s.scan(lo, hi) {
		// first key is the latest one.
		return key.Version, val, true
	}
	return 0, nil, false
}

func (s *inMemoryService) get(appName, userID, sessionID, fileName string, version int64) (*artifactEntry, bool) {
	key := artifactKey{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		FileName:  fileName,
		Version:   version,
	}.Encode()
	return s.artifacts.Get(key)
}

func (s *inMemoryService) set(appName, userID, sessionID, fileName string, version int64, entry *artifactEntry) {
	key := artifactKey{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		FileName:  fileName,
		Version:   version,
	}.Encode()
	s.artifacts.Set(key, entry)
}

func (s *inMemoryService) delete(appName, userID, sessionID, fileName string, version int64) {
	key := artifactKey{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		FileName:  fileName,
		Version:   version,
	}.Encode()
	s.artifacts.Delete(key)
}

// Save implements [artifact.Service]
func (s *inMemoryService) Save(ctx context.Context, req *SaveRequest) (*SaveResponse, error) {
	// Allow artifact reference parts, which carry neither text nor inline data
	// and are only understood by this service; the scope check below re-validates
	// that a reference stays within the caller's app/user/session.
	if err := req.validate(true); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	if isArtifactRef(req.Part) {
		parsed, ok := parseArtifactURI(req.Part.FileData.FileURI)
		if !ok {
			return nil, fmt.Errorf("invalid artifact reference URI: %s", req.Part.FileData.FileURI)
		}
		// Check the caller's session before the user-scope rewrite, so the scope
		// is validated against the caller, matching Load's per-hop rule.
		if err := validateArtifactReferenceScope(appName, userID, sessionID, parsed); err != nil {
			return nil, fmt.Errorf("invalid artifact reference: %w", err)
		}
	}
	// If file is user scoped, store it under user scope path
	if fileHasUserNamespace(fileName) {
		sessionID = userScopedArtifactKey
	}
	customMetadata := make(map[string]any, len(req.CustomMetadata))
	for key, value := range req.CustomMetadata {
		customMetadata[key] = fmt.Sprint(value)
	}
	// Read the caller-provided clock outside the service lock. A provider may
	// call back into the service, so invoking it while locked could deadlock.
	createTime := platform.Now(ctx)

	s.mu.Lock()
	defer s.mu.Unlock()

	nextVersion := int64(1)
	if internalVer, _, ok := s.find(appName, userID, sessionID, fileName); ok {
		nextVersion = internalVer + 1
	}
	s.set(appName, userID, sessionID, fileName, nextVersion, &artifactEntry{
		part:           req.Part,
		createTime:     createTime,
		customMetadata: customMetadata,
	})
	return &SaveResponse{Version: nextVersion}, nil
}

// Delete implements [artifact.Service]
func (s *inMemoryService) Delete(ctx context.Context, req *DeleteRequest) error {
	err := req.Validate()
	if err != nil {
		return fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	version := req.Version
	// If file is user scoped, adjust artifactKey part
	if fileHasUserNamespace(fileName) {
		sessionID = userScopedArtifactKey
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if version != 0 {
		s.delete(appName, userID, sessionID, fileName, version)
		return nil
	}

	// pick the latest version
	lo := artifactKey{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Version: math.MaxInt64}.Encode()
	hi := artifactKey{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName}.Encode()
	s.artifacts.DeleteRange(lo, hi)
	return nil
}

// Load implements [artifact.Service]
func (s *inMemoryService) Load(ctx context.Context, req *LoadRequest) (*LoadResponse, error) {
	err := req.Validate()
	if err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	version := req.Version

	s.mu.RLock()
	defer s.mu.RUnlock()

	// An artifact part may be a reference (its FileData.FileURI uses the
	// "artifact://" scheme), pointing at another artifact rather than storing
	// data. Resolve it here, re-validating scope at each hop so a reference can
	// never escape the app/user/session it was saved under. maxArtifactRefDepth
	// bounds resolution against cycles; the first load is the requested artifact
	// and is not a reference, while the error below counts reference hops. This
	// mirrors adk-python's cap of 5 (artifact_util._MAX_ARTIFACT_REFERENCE_DEPTH).
	const maxArtifactRefDepth = 5
	remainingDepth := maxArtifactRefDepth
	for {
		// User-scoped artifacts are stored under the shared user key, but the
		// scope check below must use the caller's scope (threaded through as
		// appName/userID/sessionID), never the storage key; keep the two distinct.
		lookupSession := sessionID
		if fileHasUserNamespace(fileName) {
			lookupSession = userScopedArtifactKey
		}

		var (
			entry *artifactEntry
			ok    bool
		)
		if version > 0 {
			entry, ok = s.get(appName, userID, lookupSession, fileName, version)
		} else {
			// pick the latest version
			_, entry, ok = s.find(appName, userID, lookupSession, fileName)
		}
		if !ok {
			return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
		}

		if !isArtifactRef(entry.part) {
			return &LoadResponse{Part: entry.part}, nil
		}

		parsed, ok := parseArtifactURI(entry.part.FileData.FileURI)
		if !ok {
			return nil, fmt.Errorf("invalid artifact reference URI: %s", entry.part.FileData.FileURI)
		}
		if err := validateArtifactReferenceScope(appName, userID, sessionID, parsed); err != nil {
			return nil, fmt.Errorf("invalid artifact reference: %w", err)
		}

		remainingDepth--
		if remainingDepth < 0 {
			return nil, fmt.Errorf("artifact reference resolution exceeded maximum depth of %d", maxArtifactRefDepth)
		}

		// Follow the reference: the next hop is validated against the session it
		// just followed, as adk-python resolves each hop against the URI it reached.
		appName, userID, sessionID, fileName, version = parsed.AppName, parsed.UserID, parsed.SessionID, parsed.FileName, parsed.Version
	}
}

// List implements [artifact.Service]
func (s *inMemoryService) List(ctx context.Context, req *ListRequest) (*ListResponse, error) {
	err := req.Validate()
	if err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID := req.AppName, req.UserID, req.SessionID
	s.mu.RLock()
	defer s.mu.RUnlock()

	files := map[string]bool{}
	lo := artifactKey{AppName: appName, UserID: userID, SessionID: sessionID}.Encode()
	hi := artifactKey{AppName: appName, UserID: userID, SessionID: sessionID + "\x00"}.Encode()
	// TODO(hyangah): extend omap to search key only and skip value decoding.
	for key := range s.scan(lo, hi) {
		if key.SessionID != sessionID { // scan includes key matching `hi`
			continue
		}
		files[key.FileName] = true
	}

	// Besides the session specific artifacts, also retrieve user scoped artifacts.
	userScopeLo := artifactKey{AppName: appName, UserID: userID, SessionID: userScopedArtifactKey}.Encode()
	userScopeHi := artifactKey{AppName: appName, UserID: userID, SessionID: userScopedArtifactKey + "\x00"}.Encode()
	// TODO: extend omap to search key only and skip value decoding.
	for key := range s.scan(userScopeLo, userScopeHi) {
		if key.SessionID != userScopedArtifactKey { // scan includes key matching `userScopeHi`
			continue
		}
		// Session ID "user" shares this key range with user-scoped artifacts,
		// so skip that session's private files. Python rejects the ID instead,
		// but Go already accepts it.
		if !fileHasUserNamespace(key.FileName) {
			continue
		}
		files[key.FileName] = true
	}

	filenames := slices.Collect(maps.Keys(files))
	sort.Strings(filenames)
	return &ListResponse{FileNames: filenames}, nil
}

// Versions implements [artifact.Service] and returns an error if no versions are found.
func (s *inMemoryService) Versions(ctx context.Context, req *VersionsRequest) (*VersionsResponse, error) {
	err := req.Validate()
	if err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	if fileHasUserNamespace(fileName) {
		sessionID = userScopedArtifactKey
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	var versions []int64
	lo := artifactKey{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Version: math.MaxInt64}.Encode()
	hi := artifactKey{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName}.Encode()
	// TODO(hyangah): extend omap to search key only and skip value decoding.
	for key := range s.scan(lo, hi) {
		versions = append(versions, key.Version)
	}
	if len(versions) == 0 {
		return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
	}
	return &VersionsResponse{Versions: versions}, nil
}

// GetArtifactVersion implements [artifact.Service] and returns the metadata for a specific version.
func (s *inMemoryService) GetArtifactVersion(ctx context.Context, req *GetArtifactVersionRequest) (*GetArtifactVersionResponse, error) {
	err := req.Validate()
	if err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID, fileName, version := req.AppName, req.UserID, req.SessionID, req.FileName, req.Version
	userScoped := fileHasUserNamespace(fileName)
	if userScoped {
		sessionID = userScopedArtifactKey
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	var entry *artifactEntry
	var ok bool
	if version > 0 {
		entry, ok = s.get(appName, userID, sessionID, fileName, version)
	} else {
		version, entry, ok = s.find(appName, userID, sessionID, fileName)
	}

	if !ok {
		return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
	}

	escapedAppName := url.PathEscape(appName)
	escapedUserID := url.PathEscape(userID)
	escapedFileName := url.PathEscape(fileName)
	canonicalURI := fmt.Sprintf("memory://apps/%s/users/%s/sessions/%s/artifacts/%s/versions/%d", escapedAppName, escapedUserID, url.PathEscape(sessionID), escapedFileName, version)
	if userScoped {
		canonicalURI = fmt.Sprintf("memory://apps/%s/users/%s/artifacts/%s/versions/%d", escapedAppName, escapedUserID, escapedFileName, version)
	}
	mimeType := "text/plain"
	if entry.part.InlineData != nil {
		mimeType = entry.part.InlineData.MIMEType
	}

	return &GetArtifactVersionResponse{
		ArtifactVersion: &ArtifactVersion{
			Version:        version,
			CanonicalURI:   canonicalURI,
			CustomMetadata: maps.Clone(entry.customMetadata),
			CreateTime:     entry.createTime,
			MimeType:       mimeType,
		},
	}, nil
}

var _ Service = (*inMemoryService)(nil)
