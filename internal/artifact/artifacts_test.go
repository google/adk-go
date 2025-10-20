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

package artifact_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/artifact"
	artifactinternal "google.golang.org/adk/internal/artifact"
	"google.golang.org/genai"
)

func TestArtifacts(t *testing.T) {
	a := artifactinternal.Artifacts{
		Service:   artifact.InMemoryService(),
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	}

	// Save
	part := *genai.NewPartFromText("test data")
	err := a.Save("testArtifact", part)
	if err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Load
	loadedPart, err := a.Load("testArtifact")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if diff := cmp.Diff(part, loadedPart); diff != "" {
		t.Errorf("Loaded part differs from saved part (-want +got):\n%s", diff)
	}

	// List
	fileNames, err := a.List()
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}

	expectedFileNames := []string{"testArtifact"}
	if diff := cmp.Diff(expectedFileNames, fileNames); diff != "" {
		t.Errorf("List returned unexpected file names (-want +got):\n%s", diff)
	}
}

func TestArtifacts_WithLoadVersion(t *testing.T) {
	a := artifactinternal.Artifacts{
		Service:   artifact.InMemoryService(),
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	}

	part := *genai.NewPartFromText("test data")
	err := a.Save("testArtifact", part)
	if err != nil {
		t.Fatalf("Save failed: %v", err)
	}
	part2 := *genai.NewPartFromText("test data 2")
	err = a.Save("testArtifact", part2)
	if err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	loadedPart, err := a.LoadVersion("testArtifact", 0)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if diff := cmp.Diff(part2, loadedPart); diff != "" {
		t.Errorf("Loaded part differs from saved part (-want +got):\n%s", diff)
	}
}

func TestArtifacts_Errors(t *testing.T) {
	a := artifactinternal.Artifacts{
		Service:   artifact.InMemoryService(),
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	}

	// Attempt to Load non-existent artifact
	_, err := a.Load("nonExistentArtifact")
	if err == nil {
		t.Errorf("Load(\"nonExistentArtifact\") succeeded, want error")
	}

	// Attempt to LoadVersion non-existent artifact
	_, err = a.LoadVersion("nonExistentArtifact", 0)
	if err == nil {
		t.Errorf("LoadVersion(\"nonExistentArtifact\", 0) succeeded, want error")
	}

	// Save an artifact to test LoadVersion with an invalid version
	part := *genai.NewPartFromText("test data")
	if err := a.Save("existsArtifact", part); err != nil {
		t.Fatalf("Save(\"existsArtifact\") failed: %v", err)
	}

	// Attempt to LoadVersion with a version number that doesn't exist
	_, err = a.LoadVersion("existsArtifact", 99)
	if err == nil {
		t.Errorf("LoadVersion(\"existsArtifact\", 99) succeeded, want error")
	}
}
