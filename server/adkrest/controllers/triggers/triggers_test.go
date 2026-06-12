// Copyright 2026 Google LLC
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

package triggers

import (
	"context"
	"iter"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/server/adkrest/internal/fakes"
	"google.golang.org/adk/session"
)

func TestRunAgent_UserIDTrimming(t *testing.T) {
	tests := []struct {
		name           string
		inputUserID    string
		expectedUserID string
	}{
		{
			name:           "NoSlashes",
			inputUserID:    "user123",
			expectedUserID: "user123",
		},
		{
			name:           "LeadingTrailingSlashes",
			inputUserID:    "/user123/",
			expectedUserID: "user123",
		},
		{
			name:           "InternalSlashes",
			inputUserID:    "users/123",
			expectedUserID: "users--123",
		},
		{
			name:           "ComplexSlashes",
			inputUserID:    "//users/123/profile/",
			expectedUserID: "users--123--profile",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sessionService := &fakes.FakeSessionService{Sessions: make(map[fakes.SessionKey]fakes.TestSession)}

			mockAgent, err := agent.New(agent.Config{
				Name: "test-agent",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						yield(&session.Event{ID: "success"}, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New failed: %v", err)
			}

			agentLoader := agent.NewSingleLoader(mockAgent)

			r := &RetriableRunner{
				sessionService: sessionService,
				agentLoader:    agentLoader,
				triggerConfig:  TriggerConfig{MaxRetries: 0},
			}

			_, err = r.RunAgent(context.Background(), "test-agent", tc.inputUserID, "hello")
			if err != nil {
				t.Fatalf("RunAgent failed: %v", err)
			}

			// Verify that the session was created with the expected UserID
			found := false
			for key := range sessionService.Sessions {
				if key.UserID == tc.expectedUserID {
					found = true
					break
				}
			}

			if !found {
				t.Errorf("Expected session with UserID %q not found in %v", tc.expectedUserID, sessionService.Sessions)
			}
		})
	}
}
