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

package artifactservice

import (
	"context"

	"google.golang.org/genai"
)

// Service is the artifact storage service.
type Service interface {
	// Save saves an artifact to the artifact service storage.
	// The artifact is a file identified by the app name, user ID, session ID, and fileName.
	// After saving the artifact, a revision ID is returned to identify the artifact version.
	Save(ctx context.Context, req *SaveRequest) (*SaveResponse, error)
	// Load loads an artifact from the storage.
	// The artifact is a file indentified by the appName, userID, sessionID and fileName.
	Load(ctx context.Context, req *LoadRequest) (*LoadResponse, error)
	// Delete deletes an artifact. Deleting a non-existing entry is not an error.
	Delete(ctx context.Context, req *DeleteRequest) error
	// List lists all the artifact filenames within a session.
	List(ctx context.Context, req *ListRequest) (*ListResponse, error)
	// Versions lists all versions of an artifact.
	Versions(ctx context.Context, req *VersionsRequest) (*VersionsResponse, error)
}

// SaveRequest is the parameter for [ArtifactService.Save].
type SaveRequest struct {
	AppName, UserID, SessionID, FileName string
	// Part is the artifact to store.
	Part *genai.Part

	// Belows are optional fields.

	// If set, the artifact will be saved with this version.
	// If unset, a new version will be created.
	Version int64
}

// SaveResponse is the return type of [ArtifactService.Save].
type SaveResponse struct {
	Version int64
}

// LoadRequest is the parameter for [ArtifactService.Load].
type LoadRequest struct {
	AppName, UserID, SessionID, FileName string

	// Belows are optional fields.
	Version int64
}

// LoadResponse is the return type of [ArtifactService.Load].
type LoadResponse struct {
	// Part is the artifact stored.
	Part *genai.Part
}

// DeleteRequest is the parameter for [ArtifactService.Delete].
type DeleteRequest struct {
	AppName, UserID, SessionID, FileName string

	// Belows are optional fields.
	Version int64
}

// ListRequest is the parameter for [ArtifactService.List].
type ListRequest struct {
	AppName, UserID, SessionID string
}

// ListResponse is the return type of [ArtifactService.List].
type ListResponse struct {
	FileNames []string
}

// VersionsRequest is the parameter for [ArtifactService.Versions].
type VersionsRequest struct {
	AppName, UserID, SessionID, FileName string
}

// VersionsResponse is the parameter for [ArtifactService.Versions].
type VersionsResponse struct {
	Versions []int64
}
