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
	"slices"
	"sync"

	"google.golang.org/adk"
	"google.golang.org/genai"
	"rsc.io/omap"
	"rsc.io/ordered"
)

// InMemorySessionService is an in-memory implementation of adk.SessionService.
// It is primarily for testing and demonstration purposes.
type InMemoryArtifactService struct {
	mu sync.RWMutex
	// ordered(appName, userID, sessionID) -> session
	artifacts omap.Map[string, *genai.Part]
}

type artifactKey struct {
	AppName   string
	UserID    string
	SessionID string
	FileName  string
	Version   int64
}

func (ak artifactKey) Encode() string {
	return string(ordered.Encode(ak.AppName, ak.UserID, ak.SessionID, ak.FileName, ordered.Rev(ak.Version)))
}

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
func (s *InMemoryArtifactService) scan(lo, hi string) iter.Seq2[artifactKey, *genai.Part] {
	return func(yield func(key artifactKey, val *genai.Part) bool) {
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

func (s *InMemoryArtifactService) find(appName, userID, sessionID, fileName string) (int64, *genai.Part, bool) {
	lo := artifactKey{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Version: math.MaxInt64}.Encode()
	hi := artifactKey{AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName, Version: 0}.Encode()
	for key, val := range s.scan(lo, hi) {
		// first key is the latest one.
		return key.Version, val, true
	}
	return 0, nil, false
}

func (s *InMemoryArtifactService) get(appName, userID, sessionID, fileName string, version int64) (*genai.Part, bool) {
	key := artifactKey{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		FileName:  fileName,
		Version:   version,
	}.Encode()
	return s.artifacts.Get(key)
}

func (s *InMemoryArtifactService) set(appName, userID, sessionID, fileName string, version int64, artifact *genai.Part) {
	key := artifactKey{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		FileName:  fileName,
		Version:   version,
	}.Encode()
	s.artifacts.Set(key, artifact)
}

func (s *InMemoryArtifactService) delete(appName, userID, sessionID, fileName string, version int64) {
	key := artifactKey{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		FileName:  fileName,
		Version:   version,
	}.Encode()
	s.artifacts.Delete(key)
}

func (s *InMemoryArtifactService) Save(ctx context.Context, req *adk.ArtifactSaveRequest) (*adk.ArtifactSaveResponse, error) {
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	artifact := req.Part

	if appName == "" || userID == "" || sessionID == "" || fileName == "" || artifact == nil {
		return nil, fmt.Errorf("invalid request: missing required fields")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	nextVersion := int64(1)
	if internalVer, _, ok := s.find(appName, userID, sessionID, fileName); ok {
		nextVersion = internalVer + 1
	}
	s.set(appName, userID, sessionID, fileName, nextVersion, artifact)
	return &adk.ArtifactSaveResponse{Version: nextVersion}, nil
}

func (s *InMemoryArtifactService) Delete(ctx context.Context, req *adk.ArtifactDeleteRequest) error {
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	version := req.Version
	if appName == "" || userID == "" || sessionID == "" || fileName == "" {
		return fmt.Errorf("invalid request: missing required fields")
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

func (s *InMemoryArtifactService) Load(ctx context.Context, req *adk.ArtifactLoadRequest) (*adk.ArtifactLoadResponse, error) {
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	if appName == "" || userID == "" || sessionID == "" || fileName == "" {
		return nil, fmt.Errorf("invalid request: missing required fields")
	}
	version := req.Version

	s.mu.RLock()
	defer s.mu.RUnlock()

	if version > 0 {
		artifact, ok := s.get(appName, userID, sessionID, fileName, version)
		if !ok {
			return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
		}
		return &adk.ArtifactLoadResponse{Part: artifact}, nil
	}
	// pick the latest version
	_, artifact, ok := s.find(appName, userID, sessionID, fileName)
	if !ok {
		return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
	}
	return &adk.ArtifactLoadResponse{Part: artifact}, nil
}

func (s *InMemoryArtifactService) List(ctx context.Context, req *adk.ArtifactListRequest) (*adk.ArtifactListResponse, error) {
	appName, userID, sessionID := req.AppName, req.UserID, req.SessionID
	if appName == "" || userID == "" || sessionID == "" {
		return nil, fmt.Errorf("invalid request: missing required fields")
	}
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
	return &adk.ArtifactListResponse{FileNames: slices.Collect(maps.Keys(files))}, nil
}

// Versions implements adk.ArtifactService.
func (s *InMemoryArtifactService) Versions(ctx context.Context, req *adk.ArtifactVersionsRequest) (*adk.ArtifactVersionsResponse, error) {
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	if appName == "" || userID == "" || sessionID == "" || fileName == "" {
		return nil, fmt.Errorf("invalid request: missing required fields")
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
	return &adk.ArtifactVersionsResponse{Versions: versions}, nil
}

var _ adk.ArtifactService = (*InMemoryArtifactService)(nil)
