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

package types_test

import (
	"context"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/types"
	"google.golang.org/genai"
)

func TestListArtifacts_WithInMemoryService(t *testing.T) {
	tests := []struct {
		name                 string
		initialArtifacts     map[string]*genai.Part
		invocationContext    *types.InvocationContext
		expectedArtifacts    []string
		expectedError        bool
		expectedErrorMessage string
	}{
		{
			name: "successful_list",
			initialArtifacts: map[string]*genai.Part{
				"artifact1.txt": {Text: "content1"},
				"artifact2.log": {Text: "content2"},
			},
			invocationContext: &types.InvocationContext{
				Session: &types.Session{
					AppName: "MyCoolApp",
					UserID:  "Alice",
					ID:      "session-123",
				},
			},
			expectedArtifacts: []string{"artifact1.txt", "artifact2.log"},
			expectedError:     false,
		},
		{
			name:             "no_artifacts",
			initialArtifacts: map[string]*genai.Part{},
			invocationContext: &types.InvocationContext{
				Session: &types.Session{
					AppName: "MyCoolApp",
					UserID:  "Alice",
					ID:      "session-123",
				},
			},
			expectedArtifacts: nil,
			expectedError:     false,
		},
		{
			name:              "invocation_context_not_initialized",
			invocationContext: nil,
			expectedError:     true,
		},
		{
			name: "artifact_service_not_initialized",
			invocationContext: &types.InvocationContext{
				Session: &types.Session{
					AppName: "MyCoolApp",
					UserID:  "Alice",
					ID:      "session-123",
				},
				ArtifactService: nil,
			},
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inMemoryService := &artifact.InMemoryArtifactService{}
			ctx := context.Background()
			if tt.initialArtifacts != nil {
				for fileName, part := range tt.initialArtifacts {
					inMemoryService.Save(ctx, &types.ArtifactSaveRequest{
						AppName:   tt.invocationContext.Session.AppName,
						UserID:    tt.invocationContext.Session.UserID,
						SessionID: tt.invocationContext.Session.ID,
						FileName:  fileName,
						Part:      part,
					})
				}
			}

			toolCtx := &types.ToolContext{
				InvocationContext: tt.invocationContext,
			}
			if tt.invocationContext != nil && tt.initialArtifacts != nil {
				toolCtx.InvocationContext.ArtifactService = inMemoryService
			}

			artifacts, err := toolCtx.ListArtifacts()

			if err == nil {
				slices.Sort(artifacts)
			}

			if tt.expectedError {
				if err == nil {
					t.Errorf("ListArtifacts() expected an error but got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("ListArtifacts() unexpected error: %v", err)
				return
			}

			if diff := cmp.Diff(tt.expectedArtifacts, artifacts); diff != "" {
				t.Errorf("ListArtifacts() returned diff (-want +got):\n%s", diff)
			}
		})
	}
}
