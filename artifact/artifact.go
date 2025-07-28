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

	"google.golang.org/genai"
)

type ArtifactService interface {
	// Stores the artifact data and returns its assigned version number.
	SaveArtifact(context.Context, *SaveArtifactRequest) (*SaveArtifactResponse, error)

	// Retrieves a specific version (or the latest) of an artifact.
	LoadArtifact(context.Context, *LoadArtifactRequest) (*LoadArtifactResponse, error)

	// Lists the unique filenames of artifacts within a given scope.
	ListArtifactKeys(context.Context, *ListArtifactKeysRequest) (*ListArtifactResponse, error)

	// Removes an artifact (and potentially all its versions, depending on implementation).
	DeleteArtifact(context.Context, *DeleteArtifactRequest) (*DeleteArtifactResponse, error)

	// Lists all available version numbers for a specific artifact filename.
	ListVersions(context.Context, *ListVersionsRequest) (*ListVersionsResponse, error)
}

type SaveArtifactRequest struct {
	AppName, UserID, SessionID string
	Filename                   string
	Artifact                   *genai.Part
}

type SaveArtifactResponse struct {
	RevisionID int
}

type LoadArtifactRequest struct {
	AppName, UserID, SessionID string
	Filename                   string
	RevisionID                 *int // optional
}

type LoadArtifactResponse struct {
	Artifact *genai.Part
}

type ListArtifactKeysRequest struct {
	AppName, UserID, SessionID string
}

type ListArtifactResponse struct {
	Filenames []string
}

type DeleteArtifactRequest struct {
	AppName, UserID, SessionID string
	Filename                   string
}

// empty for now for backwards compatibility
type DeleteArtifactResponse struct{}

type ListVersionsRequest struct {
	AppName, UserID, SessionID string
	Filename                   string
}

type ListVersionsResponse struct {
	RevisionIDs []int
}
