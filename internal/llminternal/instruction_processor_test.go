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

package llminternal

import (
	"context"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	artifactinternal "google.golang.org/adk/internal/artifact"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/sessioninternal"
	"google.golang.org/adk/session"
)

func TestInjectSessionState(t *testing.T) {
	// Define the structure for our test cases
	testCases := []struct {
		name             string                 // Name of the sub-test
		template         string                 // Input template string
		state            map[string]any         // Initial session state
		artifacts        map[string]*genai.Part // Artifacts for the mock service
		expectNilService bool                   // Flag to test with a nil artifact service
		want             string                 // Expected successful output
		wantErr          bool                   // Whether we expect an error
		wantErrMsg       string                 // A substring of the expected error message
	}{
		// Corresponds to: test_inject_session_state
		{
			name:     "successful state injection",
			template: "Hello {user_name}, you are in {app_state} state.",
			state:    map[string]any{"user_name": "Foo", "app_state": "active"},
			want:     "Hello Foo, you are in active state.",
		},
		// Corresponds to: test_inject_session_state_with_artifact
		{
			name:     "successful artifact injection",
			template: "The artifact content is: {artifact.my_file}",
			artifacts: map[string]*genai.Part{
				"my_file": {Text: "This is my artifact content."},
			},
			want: "The artifact content is: This is my artifact content.",
		},
		// Corresponds to: test_inject_session_state_with_optional_state
		// and test_inject_session_state_with_optional_missing_state_returns_empty
		{
			name:     "optional missing state variable",
			template: "Optional value: {optional_value?}",
			state:    map[string]any{},
			want:     "Optional value: ",
		},
		// Corresponds to: test_inject_session_state_with_missing_state_raises_key_error
		{
			name:       "missing required state variable",
			template:   "Hello {missing_key}!",
			state:      map[string]any{"user_name": "Foo"},
			wantErr:    true,
			wantErrMsg: "failed to get key \"missing_key\" from state: state key does not exist",
		},
		// Corresponds to: test_inject_session_state_with_missing_artifact_raises_key_error
		{
			name:     "missing required artifact",
			template: "The artifact content is: {artifact.missing_file}",
			artifacts: map[string]*genai.Part{
				"my_file": {Text: "This is my artifact content."},
			},
			wantErr:    true,
			wantErrMsg: "failed to load artifact missing_file: artifact not found: file does not exist",
		},
		// Corresponds to: test_inject_session_state_with_invalid_state_name_returns_original
		{
			name:     "invalid state name is not replaced",
			template: "Hello {invalid-key}!",
			state:    map[string]any{"user_name": "Foo"},
			want:     "Hello {invalid-key}!",
		},
		// Corresponds to: test_inject_session_state_with_invalid_prefix_state_name_returns_original
		{
			name:     "invalid prefix state name is not replaced",
			template: "Hello {invalid:key}!",
			state:    map[string]any{"user_name": "Foo"},
			want:     "Hello {invalid:key}!",
		},
		// Corresponds to: test_inject_session_state_with_valid_prefix_state
		{
			name:     "valid prefixed state variable",
			template: "Hello {app:user_name}!",
			state:    map[string]any{"app:user_name": "Foo"},
			want:     "Hello Foo!",
		},
		// Corresponds to: test_inject_session_state_with_none_state_value_returns_empty
		{
			name:     "state value is nil",
			template: "Value: {test_key}",
			state:    map[string]any{"test_key": nil},
			want:     "Value: ",
		},
		// Corresponds to: test_inject_session_state_with_optional_missing_artifact_returns_empty
		{
			name:     "optional missing artifact",
			template: "Optional artifact: {artifact.missing_file?}",
			artifacts: map[string]*genai.Part{
				"my_file": {Text: "This is my artifact content."},
			},
			want: "Optional artifact: ",
		},
		// Corresponds to: test_inject_session_state_artifact_service_not_initialized_raises_value_error
		{
			name:             "artifact service not initialized",
			template:         "The artifact content is: {artifact.my_file}",
			expectNilService: true,
			wantErr:          true,
			wantErrMsg:       "artifact service is not initialized",
		},
		// Corresponds to: test_inject_session_state_with_empty_artifact_name_raises_key_error
		{
			name:       "empty artifact name",
			template:   "The artifact content is: {artifact.}",
			artifacts:  map[string]*genai.Part{},
			wantErr:    true,
			wantErrMsg: "failed to load artifact : request validation failed: invalid load request: missing required fields: FileName",
		},
		// Corresponds to: test_inject_session_state_with_multiple_variables_and_artifacts
		{
			name: "complex template with mixed variables and artifacts",
			template: `
Hello {user_name},
You are {user_age} years old.
Your favorite color is {favorite_color}.
The artifact says: {artifact.my_file}
And another optional artifact: {artifact.other_file?}
`,
			state: map[string]any{
				"user_name":      "Foo",
				"user_age":       30,
				"favorite_color": "blue",
			},
			artifacts: map[string]*genai.Part{
				"my_file": {Text: "This is my artifact content."},
			},
			want: `
Hello Foo,
You are 30 years old.
Your favorite color is blue.
The artifact says: This is my artifact content.
And another optional artifact: 
`,
		},
	}

	// Iterate over the test cases
	for _, tc := range testCases {
		// t.Run creates a sub-test, which makes test output cleaner and more organized.
		t.Run(tc.name, func(t *testing.T) {
			// Setup, create inMemorySessionService, inMemoryArtifactService and wrappers.
			sessionService := session.InMemoryService()
			createResp, err := sessionService.Create(t.Context(), &session.CreateRequest{
				AppName:   "testApp",
				UserID:    "testUser",
				SessionID: "testSession",
				State:     tc.state,
			})
			if err != nil {
				t.Fatalf("Failed to create session: %v", err)
			}
			sess := sessioninternal.NewMutableSession(sessionService, createResp.Session)

			// Setup Artifacts
			var artifacts agent.Artifacts
			if !tc.expectNilService {
				artifacts = &artifactinternal.Artifacts{
					Service:   artifact.InMemoryService(),
					AppName:   "testApp",
					UserID:    "testUser",
					SessionID: "testSession",
				}
			}
			for filename, part := range tc.artifacts {
				if _, err := artifacts.Save(t.Context(), filename, part); err != nil {
					t.Fatalf("Failed to save artifact: %v", err)
				}
			}
			// Create invocation context
			ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
				Artifacts: artifacts,
				Session:   sess,
			})

			// --- Execution ---
			got, err := InjectSessionState(ctx, tc.template)

			// --- Assertion ---
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected an error but got none")
				}
				if tc.wantErrMsg != "" && !strings.Contains(err.Error(), tc.wantErrMsg) {
					t.Errorf("expected error message to contain %q, but got %q", tc.wantErrMsg, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("did not expect an error but got: %v", err)
				}
				if got != tc.want {
					// Use %q to clearly show differences in strings, especially with whitespace.
					t.Errorf("got %q, want %q", got, tc.want)
				}
			}
		})
	}
}
