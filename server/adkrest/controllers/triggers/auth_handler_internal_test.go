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
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"iter"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/mux"
	"google.golang.org/api/idtoken"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/server/adkrest/internal/fakes"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
)

// These tests drive the exported handlers through a real gorilla/mux router
// with an Authorization header attached, which is the path a Pub/Sub push
// delivery actually takes. The rejection cases in pubsub_test.go and
// eventarc_test.go only cover requests with no token at all, so on their own
// they stay green even if the handler rejects every request. That would brick
// the feature for every real caller while looking correct.

// countingAgent returns an agent that records how many times it was run.
func countingAgent(t *testing.T, runs *int) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: "test-agent",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				*runs++
				yield(&session.Event{ID: "success-event"}, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New failed: %v", err)
	}
	return a
}

func authTestConfig(allowed ...string) TriggerConfig {
	return TriggerConfig{
		MaxConcurrentRuns: 10,
		MaxRetries:        3,
		BaseDelay:         1 * time.Millisecond,
		MaxDelay:          5 * time.Millisecond,
		OIDC: &OIDCConfig{
			ExpectedAudience:       testAudience,
			AllowedServiceAccounts: allowed,
		},
	}
}

func newSessionService() session.Service {
	return &fakes.FakeSessionService{Sessions: make(map[fakes.SessionKey]fakes.TestSession)}
}

// authCase is one request against a trigger endpoint with the OIDC feature on.
type authCase struct {
	name       string
	allowed    []string
	authHeader string
	// validate stands in for idtoken.Validate; nil means the handler must not
	// reach it.
	validate   tokenValidator
	wantStatus int
	wantRuns   int
}

func authCases() []authCase {
	validGoogleToken := func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
		return googlePayload(aud), nil
	}
	return []authCase{
		{
			name:       "valid token from the allowed service account runs the agent",
			allowed:    []string{testServiceAccount},
			authHeader: "Bearer valid-token",
			validate:   validGoogleToken,
			wantStatus: http.StatusOK,
			wantRuns:   1,
		},
		{
			name:       "valid token with no allow-list configured runs the agent",
			authHeader: "Bearer valid-token",
			validate:   validGoogleToken,
			wantStatus: http.StatusOK,
			wantRuns:   1,
		},
		{
			// A Google-signed token for this audience can belong to any
			// principal that can mint one, so the audience alone must not be
			// enough once an allow-list is configured.
			name:       "valid token from an unlisted principal is rejected",
			allowed:    []string{testServiceAccount},
			authHeader: "Bearer valid-token",
			validate: func(_ context.Context, _, aud string) (*idtoken.Payload, error) {
				p := googlePayload(aud)
				p.Claims["email"] = "unrelated@gmail.com"
				return p, nil
			},
			wantStatus: http.StatusForbidden,
		},
		{
			name:       "no Authorization header is rejected",
			allowed:    []string{testServiceAccount},
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "empty bearer credential is rejected",
			allowed:    []string{testServiceAccount},
			authHeader: "Bearer ",
			wantStatus: http.StatusUnauthorized,
		},
	}
}

func TestPubSubTriggerHandlerAuth(t *testing.T) {
	body, err := json.Marshal(models.PubSubTriggerRequest{
		Message: models.PubSubMessage{
			Data: []byte(base64.StdEncoding.EncodeToString([]byte("Hello agent"))),
		},
		Subscription: "test-sub",
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	for _, tc := range authCases() {
		t.Run(tc.name, func(t *testing.T) {
			runs := 0
			c, err := NewPubSubControllerWithConfig(ControllerConfig{
				SessionService: newSessionService(),
				AgentLoader:    agent.NewSingleLoader(countingAgent(t, &runs)),
				TriggerConfig:  authTestConfig(tc.allowed...),
			})
			if err != nil {
				t.Fatalf("NewPubSubControllerWithConfig() failed: %v", err)
			}
			c.runner.validateIDToken = orFail(t, tc.validate)

			router := mux.NewRouter()
			router.HandleFunc("/apps/{app_name}/trigger/pubsub", c.PubSubTriggerHandler).Methods(http.MethodPost)

			req := httptest.NewRequest(http.MethodPost, "/apps/test-agent/trigger/pubsub", bytes.NewReader(body))
			if tc.authHeader != "" {
				req.Header.Set("Authorization", tc.authHeader)
			}
			rr := httptest.NewRecorder()
			router.ServeHTTP(rr, req)

			checkAuthResult(t, rr, runs, tc)
		})
	}
}

func TestEventarcTriggerHandlerAuth(t *testing.T) {
	body, err := json.Marshal(models.EventarcTriggerRequest{
		ID:          "1234-5678",
		Source:      "//pubsub.googleapis.com/projects/test/topics/test",
		Type:        "google.cloud.pubsub.topic.v1.messagePublished",
		SpecVersion: "1.0",
		Data:        []byte(`{"message":{"data":"SGVsbG8gYWdlbnQ="}}`),
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	for _, tc := range authCases() {
		t.Run(tc.name, func(t *testing.T) {
			runs := 0
			c, err := NewEventarcControllerWithConfig(ControllerConfig{
				SessionService: newSessionService(),
				AgentLoader:    agent.NewSingleLoader(countingAgent(t, &runs)),
				TriggerConfig:  authTestConfig(tc.allowed...),
			})
			if err != nil {
				t.Fatalf("NewEventarcControllerWithConfig() failed: %v", err)
			}
			c.runner.validateIDToken = orFail(t, tc.validate)

			router := mux.NewRouter()
			router.HandleFunc("/apps/{app_name}/trigger/eventarc", c.EventarcTriggerHandler).Methods(http.MethodPost)

			req := httptest.NewRequest(http.MethodPost, "/apps/test-agent/trigger/eventarc", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/cloudevents+json")
			if tc.authHeader != "" {
				req.Header.Set("Authorization", tc.authHeader)
			}
			rr := httptest.NewRecorder()
			router.ServeHTTP(rr, req)

			checkAuthResult(t, rr, runs, tc)
		})
	}
}

// orFail returns validate, or a validator that fails the test if the handler
// reaches token verification when it should have rejected the request earlier.
func orFail(t *testing.T, validate tokenValidator) tokenValidator {
	t.Helper()
	if validate != nil {
		return validate
	}
	return func(context.Context, string, string) (*idtoken.Payload, error) {
		t.Error("token verification should not have been reached")
		return nil, nil
	}
}

func checkAuthResult(t *testing.T, rr *httptest.ResponseRecorder, runs int, tc authCase) {
	t.Helper()
	if rr.Code != tc.wantStatus {
		t.Errorf("status = %d, want %d. Body: %s", rr.Code, tc.wantStatus, rr.Body.String())
	}
	if runs != tc.wantRuns {
		t.Errorf("agent ran %d times, want %d", runs, tc.wantRuns)
	}
	if tc.wantStatus == http.StatusUnauthorized && rr.Header().Get("WWW-Authenticate") == "" {
		t.Error("401 response is missing a WWW-Authenticate header")
	}
}
