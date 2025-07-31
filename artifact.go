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

package adk

import (
	"context"

	"google.golang.org/genai"
)

// ArtifactSaveRequest is the parameter for [ArtifactService.Save].
type ArtifactSaveRequest struct {
	AppName, UserID, SessionID, FileName string
	// Part is the artifact to store.
	Part *genai.Part

	// Belows are optional fields.

	// If set, the artifact will be saved with this version.
	// If unset, a new version will be created.
	Version int64
}

// ArtifactSaveResponse is the return type of [ArtifactService.Save].
type ArtifactSaveResponse struct {
	Version int64
}

// ArtifactLoadRequest is the parameter for [ArtifactService.Load].
type ArtifactLoadRequest struct {
	AppName, UserID, SessionID, FileName string

	// Belows are optional fields.
	Version int64
}

// ArtifactLoadResponse is the return type of [ArtifactService.Load].
type ArtifactLoadResponse struct {
	// Part is the artifact stored.
	Part *genai.Part
}

// ArtifactDeleteRequest is the parameter for [ArtifactService.Delete].
type ArtifactDeleteRequest struct {
	AppName, UserID, SessionID, FileName string

	// Belows are optional fields.
	Version int64
}

// ArtifactListRequest is the parameter for [ArtifactService.List].
type ArtifactListRequest struct {
	AppName, UserID, SessionID string
}

// ArtifactListResponse is the return type of [ArtifactService.List].
type ArtifactListResponse struct {
	FileNames []string
}

// ArtifactVersionsRequest is the parameter for [ArtifactService.Versions].
type ArtifactVersionsRequest struct {
	AppName, UserID, SessionID, FileName string
}

// ArtifactVersionsResponse is the parameter for [ArtifactService.Versions].
type ArtifactVersionsResponse struct {
	Versions []int64
}

// ArtifactService is the artifact storage service.
type ArtifactService interface {
	// Save saves an artifact to the artifact service storage.
	// The artifact is a file identified by the app name, user ID, session ID, and fileName.
	// After saving the artifact, a revision ID is returned to identify the artifact version.
	Save(ctx context.Context, req *ArtifactSaveRequest) (*ArtifactSaveResponse, error)
	// Load loads an artifact from the storage.
	// The artifact is a file indentified by the appName, userID, sessionID and fileName.
	Load(ctx context.Context, req *ArtifactLoadRequest) (*ArtifactLoadResponse, error)
	// Delete deletes an artifact. Deleting a non-existing entry is not an error.
	Delete(ctx context.Context, req *ArtifactDeleteRequest) error
	// List lists all the artifact filenames within a session.
	List(ctx context.Context, req *ArtifactListRequest) (*ArtifactListResponse, error)
	// Versions lists all versions of an artifact.
	Versions(ctx context.Context, req *ArtifactVersionsRequest) (*ArtifactVersionsResponse, error)
}
