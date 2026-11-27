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
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
)

func TestArtifacts(t *testing.T) {
	a := artifactinternal.Artifacts{
		Service:   artifact.InMemoryService(),
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	}

	// Save
	part := genai.NewPartFromText("test data")
	_, err := a.Save(t.Context(), "testArtifact", part)
	if err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Load
	loadResp, err := a.Load(t.Context(), "testArtifact")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if diff := cmp.Diff(part, loadResp.Part); diff != "" {
		t.Errorf("Loaded part differs from saved part (-want +got):\n%s", diff)
	}

	// List
	listResp, err := a.List(t.Context())
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}

	expectedFileNames := []string{"testArtifact"}
	if diff := cmp.Diff(expectedFileNames, listResp.FileNames); diff != "" {
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

	part0 := genai.NewPartFromText("test data")
	saved0, err := a.Save(t.Context(), "testArtifact", part0)
	if err != nil {
		t.Fatalf("Save failed: %v", err)
	}
	if saved0.Version != 0 {
		t.Fatalf("first Save().Version = %d, want 0", saved0.Version)
	}
	part1 := genai.NewPartFromText("test data 2")
	saved1, err := a.Save(t.Context(), "testArtifact", part1)
	if err != nil {
		t.Fatalf("Save failed: %v", err)
	}
	if saved1.Version != 1 {
		t.Fatalf("second Save().Version = %d, want 1", saved1.Version)
	}

	for _, tc := range []struct {
		name    string
		version *int64
		want    *genai.Part
	}{
		{name: "nil loads latest", version: nil, want: part1},
		{name: "version 0 loads first save", version: new(int64(0)), want: part0},
		{name: "version 1 loads second save", version: new(int64(1)), want: part1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			loadResp, err := a.LoadVersion(t.Context(), "testArtifact", tc.version)
			if err != nil {
				t.Fatalf("LoadVersion() failed: %v", err)
			}
			if diff := cmp.Diff(tc.want, loadResp.Part); diff != "" {
				t.Errorf("Loaded part differs from saved part (-want +got):\n%s", diff)
			}
		})
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
	_, err := a.Load(t.Context(), "nonExistentArtifact")
	if err == nil {
		t.Errorf("Load(\"nonExistentArtifact\") succeeded, want error")
	}

	// Attempt to LoadVersion non-existent artifact (latest and explicit version 0)
	for _, version := range []*int64{nil, new(int64(0))} {
		if _, err := a.LoadVersion(t.Context(), "nonExistentArtifact", version); err == nil {
			t.Errorf("LoadVersion(\"nonExistentArtifact\", %v) succeeded, want error", version)
		}
	}

	// Save an artifact once (version 0) to test out-of-range versions
	part := genai.NewPartFromText("test data")
	if _, err := a.Save(t.Context(), "existsArtifact", part); err != nil {
		t.Fatalf("Save(\"existsArtifact\") failed: %v", err)
	}

	for _, invalidVer := range []int64{-1, 1, 99} {
		if _, err := a.LoadVersion(t.Context(), "existsArtifact", new(invalidVer)); err == nil {
			t.Errorf("LoadVersion(\"existsArtifact\", %d) succeeded, want error", invalidVer)
		}
	}
}
