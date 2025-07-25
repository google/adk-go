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

// ArtifactLoadOption is the option for [ArtifactService.Load].
type ArtifactLoadOption struct {
	Version int64
}

// ArtifactDeleteOption is the option for [ArtifactService.Delete].
type ArtifactDeleteOption struct {
	Version int64
}

type ArtifactService interface {
	// Save saves an artifact to the artifact service storage.
	// The artifact is a file identified by the app name, user ID, session ID, and fileName.
	// After saving the artifact, a revision ID is returned to identify the artifact version.
	Save(ctx context.Context, appName, userID, sessionID, fileName string, artifact *genai.Part) (int64, error)
	// Load loads an artifact from the storage.
	// The artifact is a file indentified by the appName, userID, sessionID and fileName.
	Load(ctx context.Context, appName, userID, sessionID, fileName string, opts *ArtifactLoadOption) (*genai.Part, error)
	// Delete deletes an artifact.
	Delete(ctx context.Context, appName, userID, sessionID, fileName string, opts *ArtifactDeleteOption) error
	// List lists all the artifact filenames within a session.
	List(ctx context.Context, appName, userID, sessionID string) ([]string, error)
	// Versions lists all versions of an artifact.
	Versions(ctx context.Context, appName, userID, sessionID, fileName string) ([]int64, error)
}
