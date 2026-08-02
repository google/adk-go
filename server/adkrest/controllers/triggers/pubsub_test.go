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

package triggers_test

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/agent"

	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/server/adkrest/controllers/triggers"
	"google.golang.org/adk/v2/server/adkrest/internal/fakes"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
)

var defaultTriggerConfig = triggers.TriggerConfig{
	MaxConcurrentRuns: 10,
	MaxRetries:        3,
	BaseDelay:         1 * time.Millisecond,
	MaxDelay:          5 * time.Millisecond,
}

func TestPubSubTriggerHandler(t *testing.T) {
	tests := []struct {
		name               string
		mockAgentResults   []error
		expectedCode       int
		expectedRunCount   int
		requestAttributes  map[string]string
		expectedAttributes map[string]string
		requestData        string
	}{
		{
			name:             "Success_Immediate",
			mockAgentResults: nil,
			expectedCode:     http.StatusOK,
			expectedRunCount: 1,
			requestData:      "Hello agent",
		},
		{
			name:             "ResourceExhaustedRetry",
			mockAgentResults: []error{fmt.Errorf("429 ResourceExhausted"), fmt.Errorf("429 ResourceExhausted")},
			expectedCode:     http.StatusOK,
			expectedRunCount: 3,
			requestData:      "Hello agent",
		},
		{
			name:               "With_Attributes",
			mockAgentResults:   nil,
			expectedCode:       http.StatusOK,
			expectedRunCount:   1,
			requestAttributes:  map[string]string{"key1": "val1", "key2": "val2"},
			expectedAttributes: map[string]string{"key1": "val1", "key2": "val2"},
			requestData:        "Hello agent",
		},
		{
			name:             "Empty Data",
			mockAgentResults: nil,
			expectedCode:     http.StatusBadRequest,
			expectedRunCount: 0,
			requestData:      "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mockAgentRunCount := 0
			testAgent := createMockAgent(t, tc.mockAgentResults, &mockAgentRunCount, tc.expectedAttributes)

			apiController := setupTest(t, testAgent)

			reqObj := models.PubSubTriggerRequest{
				Message: models.PubSubMessage{
					Data:       []byte(base64.StdEncoding.EncodeToString([]byte(tc.requestData))),
					Attributes: tc.requestAttributes,
				},
				Subscription: "test-sub",
			}
			reqBytes, err := json.Marshal(reqObj)
			if err != nil {
				t.Fatalf("marshal request: %v", err)
			}

			req, err := http.NewRequest(http.MethodPost, "/apps/test-agent/triggers/pubsub", bytes.NewBuffer(reqBytes))
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			req = mux.SetURLVars(req, map[string]string{"app_name": "test-agent"})
			rr := httptest.NewRecorder()

			apiController.PubSubTriggerHandler(rr, req)

			if rr.Code != tc.expectedCode {
				t.Errorf("expected status %d, got %d. Body: %s", tc.expectedCode, rr.Code, rr.Body.String())
			}

			if mockAgentRunCount != tc.expectedRunCount {
				t.Errorf("expected %d run attempts, got %d", tc.expectedRunCount, mockAgentRunCount)
			}
		})
	}
}

func setupTest(t *testing.T, a agent.Agent) *triggers.PubSubController {
	t.Helper()
	sessionService := &fakes.FakeSessionService{Sessions: make(map[fakes.SessionKey]fakes.TestSession)}
	agentLoader := agent.NewSingleLoader(a)
	return triggers.NewPubSubController(sessionService, agentLoader, nil, nil, runner.PluginConfig{}, defaultTriggerConfig)
}

// TestPubSubTriggerHandler_RequiresAuthWhenAudienceConfigured guards against
// the endpoint accepting unauthenticated requests once an operator has opted
// in to OIDC verification via ExpectedAudience: without a bearer token at all
// (the realistic "reachable from the open internet" case) the request must be
// rejected before the agent is ever invoked.
func TestPubSubTriggerHandler_RequiresAuthWhenAudienceConfigured(t *testing.T) {
	mockAgentRunCount := 0
	testAgent := createMockAgent(t, nil, &mockAgentRunCount, nil)

	sessionService := &fakes.FakeSessionService{Sessions: make(map[fakes.SessionKey]fakes.TestSession)}
	agentLoader := agent.NewSingleLoader(testAgent)
	config := defaultTriggerConfig
	config.ExpectedAudience = "https://example-agent.example.com"
	apiController := triggers.NewPubSubController(sessionService, agentLoader, nil, nil, runner.PluginConfig{}, config)

	reqObj := models.PubSubTriggerRequest{
		Message: models.PubSubMessage{
			Data: []byte(base64.StdEncoding.EncodeToString([]byte("Hello agent"))),
		},
		Subscription: "test-sub",
	}
	reqBytes, err := json.Marshal(reqObj)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	req, err := http.NewRequest(http.MethodPost, "/apps/test-agent/triggers/pubsub", bytes.NewBuffer(reqBytes))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req = mux.SetURLVars(req, map[string]string{"app_name": "test-agent"})
	rr := httptest.NewRecorder()

	apiController.PubSubTriggerHandler(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected status %d for a request with no bearer token, got %d. Body: %s", http.StatusUnauthorized, rr.Code, rr.Body.String())
	}
	if mockAgentRunCount != 0 {
		t.Errorf("expected the agent not to run for an unauthenticated request, got %d run attempts", mockAgentRunCount)
	}
}

func createMockAgent(t *testing.T, results []error, runCount *int, expectedAttributes map[string]string) agent.Agent {
	t.Helper()
	testAgent, err := agent.New(agent.Config{
		Name: "test-agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				*runCount++

				userContent := ctx.UserContent()
				if len(expectedAttributes) > 0 {
					if userContent == nil || len(userContent.Parts) == 0 {
						t.Errorf("expected user content but got none")
					} else {
						var msgMap map[string]any
						err := json.Unmarshal([]byte(userContent.Parts[0].Text), &msgMap)
						if err != nil {
							t.Errorf("failed to unmarshal message content: %v", err)
						} else {
							gotAttrs, ok := msgMap["attributes"].(map[string]any)
							if !ok {
								t.Errorf("expected attributes map, got %T", msgMap["attributes"])
							} else {
								for k, v := range expectedAttributes {
									if gotAttrs[k] != v {
										t.Errorf("expected attribute %s=%s, got %s", k, v, gotAttrs[k])
									}
								}
							}
						}
					}
				}

				if *runCount <= len(results) {
					err := results[*runCount-1]
					if err != nil {
						yield(nil, err)
						return
					}
				}
				yield(&session.Event{ID: "success-event"}, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New failed: %v", err)
	}
	return testAgent
}
