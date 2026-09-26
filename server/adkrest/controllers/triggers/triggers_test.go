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
	"context"
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

func TestPubSubTriggerHandler_ContextCanceled(t *testing.T) {
	mockAgentRunCount := 0
	// Make agent return 429 so it triggers backoff retry loop
	mockResults := []error{fmt.Errorf("429 ResourceExhausted")}
	testAgent := createMockAgent(t, mockResults, &mockAgentRunCount, nil)

	sessionService := &fakes.FakeSessionService{Sessions: make(map[fakes.SessionKey]fakes.TestSession)}
	agentLoader := agent.NewSingleLoader(testAgent)

	longBackoffConfig := triggers.TriggerConfig{
		MaxConcurrentRuns: 10,
		MaxRetries:        3,
		BaseDelay:         10 * time.Second, // Long delay
		MaxDelay:          30 * time.Second,
	}

	controller := triggers.NewPubSubController(sessionService, agentLoader, nil, nil, runner.PluginConfig{}, longBackoffConfig)

	reqObj := models.PubSubTriggerRequest{
		Message: models.PubSubMessage{
			Data: []byte(base64.StdEncoding.EncodeToString([]byte("test data"))),
		},
		Subscription: "test-sub",
	}
	reqBytes, err := json.Marshal(reqObj)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "/apps/test-agent/triggers/pubsub", bytes.NewBuffer(reqBytes))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req = mux.SetURLVars(req, map[string]string{"app_name": "test-agent"})
	rr := httptest.NewRecorder()

	// Cancel the context after 50ms while inside the 10-second backoff delay
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	controller.PubSubTriggerHandler(rr, req)
	elapsed := time.Since(start)

	if elapsed > 1*time.Second {
		t.Errorf("expected immediate response on context cancel, took %v", elapsed)
	}

	if mockAgentRunCount != 1 {
		t.Errorf("expected 1 agent run attempt before cancel, got %d", mockAgentRunCount)
	}
}

func BenchmarkPubSubTriggerHandler_ContextCanceled(b *testing.B) {
	mockResults := []error{fmt.Errorf("429 ResourceExhausted")}

	longBackoffConfig := triggers.TriggerConfig{
		MaxConcurrentRuns: 100,
		MaxRetries:        3,
		BaseDelay:         5 * time.Second,
		MaxDelay:          10 * time.Second,
	}

	reqObj := models.PubSubTriggerRequest{
		Message: models.PubSubMessage{
			Data: []byte(base64.StdEncoding.EncodeToString([]byte("bench data"))),
		},
		Subscription: "bench-sub",
	}
	reqBytes, _ := json.Marshal(reqObj)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		mockRunCount := 0
		testAgent, err := agent.New(agent.Config{
			Name: "test-agent",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					mockRunCount++
					if mockRunCount <= len(mockResults) {
						yield(nil, mockResults[mockRunCount-1])
						return
					}
					yield(&session.Event{ID: "success-event"}, nil)
				}
			},
		})
		if err != nil {
			b.Fatalf("agent.New failed: %v", err)
		}

		sessionService := &fakes.FakeSessionService{Sessions: make(map[fakes.SessionKey]fakes.TestSession)}
		agentLoader := agent.NewSingleLoader(testAgent)
		controller := triggers.NewPubSubController(sessionService, agentLoader, nil, nil, runner.PluginConfig{}, longBackoffConfig)

		ctx, cancel := context.WithCancel(context.Background())
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, "/apps/test-agent/triggers/pubsub", bytes.NewBuffer(reqBytes))
		req = mux.SetURLVars(req, map[string]string{"app_name": "test-agent"})
		rr := httptest.NewRecorder()

		// Cancel immediately
		cancel()
		controller.PubSubTriggerHandler(rr, req)
	}
}
