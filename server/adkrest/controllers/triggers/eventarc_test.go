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
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/server/adkrest/controllers/triggers"
	"google.golang.org/adk/v2/server/adkrest/internal/fakes"
	"google.golang.org/adk/v2/session"
)

func TestEventarcTriggerHandler(t *testing.T) {
	storagePayload := `{
  "bucket": "sample-bucket",
  "contentType": "text/plain",
  "generation": "1587627537231057",
  "id": "sample-bucket/folder/Test.cs/1587627537231057",
  "kind": "storage#object",
  "size": "352",
  "storageClass": "MULTI_REGIONAL",
  "timeCreated": "2020-04-23T07:38:57.230Z",
  "timeStorageClassUpdated": "2020-04-23T07:38:57.230Z",
  "updated": "2020-04-23T07:38:57.230Z"
}`

	pubsubPayload := `{
  "subscription": "projects/test-project/subscriptions/my-subscription",
  "message": {
    "attributes": {
      "attr1":"attr1-value"
    },
    "data": "dGVzdCBtZXNzYWdlIDM=",
    "messageId": "message-id",
    "publishTime":"2021-02-05T04:06:14.109Z",
    "orderingKey": "ordering-key"
  }
}`

	jsonRawToMap := func(t *testing.T, data []byte) map[string]any {
		var m map[string]any
		if err := json.Unmarshal(data, &m); err != nil {
			t.Fatalf("failed to unmarshal JSON to map: %v", err)
		}
		return m
	}

	tests := []struct {
		name            string
		contentType     string
		headers         map[string]string
		body            []byte
		expectedPayload map[string]any
	}{
		{
			name:        "Storage_Structured_Mode",
			contentType: "application/cloudevents+json",
			body:        []byte(storagePayload),
			expectedPayload: map[string]any{
				"id":          "sample-bucket/folder/Test.cs/1587627537231057",
				"source":      "",
				"type":        "",
				"specversion": "",
			},
		},
		{
			name:        "Storage_Binary_Mode",
			contentType: "application/json",
			headers: map[string]string{
				"ce-id":          "1234-5678",
				"ce-type":        "google.storage.object.v1.finalized",
				"ce-source":      "//storage.googleapis.com/projects/_/buckets/sample-bucket",
				"ce-specversion": "1.0",
			},
			body: []byte(storagePayload),
			expectedPayload: map[string]any{
				"id":          "1234-5678",
				"type":        "google.storage.object.v1.finalized",
				"source":      "//storage.googleapis.com/projects/_/buckets/sample-bucket",
				"specversion": "1.0",
				"data":        jsonRawToMap(t, []byte(storagePayload)),
			},
		},
		{
			name:        "PubSub_Structured_Mode",
			contentType: "application/cloudevents+json",
			body: func() []byte {
				ce := map[string]any{
					"specversion": "1.0",
					"type":        "google.cloud.pubsub.topic.v1.messagePublished",
					"source":      "//pubsub.googleapis.com/projects/test-project/topics/test-topic",
					"id":          "1234-5678",
					"data":        json.RawMessage(pubsubPayload),
				}
				b, _ := json.Marshal(ce)
				return b
			}(),
			expectedPayload: map[string]any{
				"data": "test message 3",
				"attributes": map[string]any{
					"attr1": "attr1-value",
				},
			},
		},
		{
			name:        "PubSub_Binary_Mode",
			contentType: "application/json",
			headers: map[string]string{
				"ce-id":          "1234-5678",
				"ce-type":        "google.cloud.pubsub.topic.v1.messagePublished",
				"ce-source":      "//pubsub.googleapis.com/projects/test-project/topics/test-topic",
				"ce-specversion": "1.0",
			},
			body: []byte(pubsubPayload),
			expectedPayload: map[string]any{
				"data": "test message 3",
				"attributes": map[string]any{
					"attr1": "attr1-value",
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mockAgentRunCount := 0
			var receivedContent string
			testAgent, err := agent.New(agent.Config{
				Name: "test-agent",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						mockAgentRunCount++
						userContent := ctx.UserContent()
						if userContent != nil && len(userContent.Parts) > 0 {
							receivedContent = userContent.Parts[0].Text
						}
						yield(&session.Event{ID: "success-event"}, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New failed: %v", err)
			}

			sessionService := &fakes.FakeSessionService{Sessions: make(map[fakes.SessionKey]fakes.TestSession)}
			agentLoader := agent.NewSingleLoader(testAgent)
			controller := triggers.NewEventarcController(sessionService, agentLoader, nil, nil, runner.PluginConfig{}, defaultTriggerConfig)

			req, err := http.NewRequest(http.MethodPost, "/apps/test-agent/triggers/eventarc", bytes.NewBuffer(tc.body))
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			req.Header.Set("Content-Type", tc.contentType)
			for k, v := range tc.headers {
				req.Header.Set(k, v)
			}
			req = mux.SetURLVars(req, map[string]string{"app_name": "test-agent"})
			rr := httptest.NewRecorder()

			controller.EventarcTriggerHandler(rr, req)

			if rr.Code != http.StatusOK {
				t.Errorf("expected status 200, got %d. Body: %s", rr.Code, rr.Body.String())
			}

			if mockAgentRunCount != 1 {
				t.Errorf("expected 1 run attempt, got %d", mockAgentRunCount)
			}

			if tc.expectedPayload != nil {
				var gotPayload map[string]any
				if err := json.Unmarshal([]byte(receivedContent), &gotPayload); err != nil {
					t.Fatalf("failed to unmarshal received content: %v", err)
				}
				if diff := cmp.Diff(tc.expectedPayload, gotPayload); diff != "" {
					t.Errorf("payload mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

// TestEventarcBinaryModeRejectsOversizedBody verifies that a body over the
// configured limit is reported as a client error in binary mode, matching the
// structured branch and the Pub/Sub trigger, rather than as an internal error.
func TestEventarcBinaryModeRejectsOversizedBody(t *testing.T) {
	testAgent, err := agent.New(agent.Config{
		Name: "test-agent",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New failed: %v", err)
	}

	sessionService := &fakes.FakeSessionService{Sessions: make(map[fakes.SessionKey]fakes.TestSession)}
	controller := triggers.NewEventarcController(sessionService, agent.NewSingleLoader(testAgent), nil, nil, runner.PluginConfig{}, defaultTriggerConfig)

	const limit = 16
	rr := httptest.NewRecorder()
	req, err := http.NewRequest(http.MethodPost, "/apps/test-agent/triggers/eventarc", bytes.NewBufferString(strings.Repeat("a", int(limit)+1)))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req = mux.SetURLVars(req, map[string]string{"app_name": "test-agent"})
	req.Body = http.MaxBytesReader(rr, req.Body, limit)

	controller.EventarcTriggerHandler(rr, req)

	if got, want := rr.Code, http.StatusBadRequest; got != want {
		t.Fatalf("oversized binary-mode body: status %d, want %d (body: %s)", got, want, rr.Body.String())
	}
}

// buildEventarcRequest returns a CloudEvent POST request carrying a Pub/Sub
// messagePublished payload, ready for EventarcTriggerHandler.
func buildEventarcRequest(t *testing.T, data string) *http.Request {
	t.Helper()
	payload := `{"message": {"data": "` + base64.StdEncoding.EncodeToString([]byte(data)) + `"}}`
	ce := map[string]any{
		"specversion": "1.0",
		"type":        "google.cloud.pubsub.topic.v1.messagePublished",
		"source":      "//pubsub.googleapis.com/projects/test-project/topics/test-topic",
		"id":          "1234-5678",
		"data":        json.RawMessage(payload),
	}
	body, err := json.Marshal(ce)
	if err != nil {
		t.Fatalf("marshal cloud event: %v", err)
	}
	req, err := http.NewRequest(http.MethodPost, "/apps/test-agent/triggers/eventarc", bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/cloudevents+json")
	return mux.SetURLVars(req, map[string]string{"app_name": "test-agent"})
}

// blockingEventarcAgent returns an agent whose runs signal started and then
// hold until release is closed, so a test can count runs in flight.
func blockingEventarcAgent(t *testing.T, started, release chan struct{}) agent.Agent {
	t.Helper()
	blockingAgent, err := agent.New(agent.Config{
		Name: "test-agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				started <- struct{}{}
				<-release
				yield(&session.Event{ID: "success-event"}, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return blockingAgent
}

// TestEventarcTriggerHandler_NonPositiveMaxConcurrentRunsIsUnlimited mirrors
// the Pub/Sub guard for the same edge: MaxConcurrentRuns of zero or below must
// mean "no limit" (nil semaphore) — not a zero-capacity channel that blocks
// every request forever (zero), and not a constructor panic
// (negative: makechan size out of range). Requiring three concurrent requests
// in flight at once distinguishes "unlimited" from any small cap a wrong
// constructor might build.
func TestEventarcTriggerHandler_NonPositiveMaxConcurrentRunsIsUnlimited(t *testing.T) {
	for _, maxRuns := range []int{0, -1} {
		t.Run(fmt.Sprintf("MaxConcurrentRuns=%d", maxRuns), func(t *testing.T) {
			const requests = 3
			started := make(chan struct{}, requests)
			release := make(chan struct{})
			cfg := defaultTriggerConfig
			cfg.MaxConcurrentRuns = maxRuns
			sessionService := &fakes.FakeSessionService{Sessions: make(map[fakes.SessionKey]fakes.TestSession)}
			ctrl := triggers.NewEventarcController(sessionService, agent.NewSingleLoader(blockingEventarcAgent(t, started, release)), nil, nil, runner.PluginConfig{}, cfg)

			recorders := make([]*httptest.ResponseRecorder, requests)
			var wg sync.WaitGroup
			for i := range requests {
				recorders[i] = httptest.NewRecorder()
				wg.Add(1)
				go func() {
					defer wg.Done()
					ctrl.EventarcTriggerHandler(recorders[i], buildEventarcRequest(t, "hello"))
				}()
			}
			for i := range requests {
				select {
				case <-started:
				case <-time.After(2 * time.Second):
					t.Fatalf("only %d of %d runs in flight; MaxConcurrentRuns=%d must mean no limit", i, requests, maxRuns)
				}
			}
			close(release)
			wg.Wait()
			for i, rr := range recorders {
				if rr.Code != http.StatusOK {
					t.Errorf("request %d: status = %d, want 200. Body: %s", i, rr.Code, rr.Body.String())
				}
			}
		})
	}
}

// TestEventarcTriggerHandler_EnforcesMaxConcurrentRuns is the positive-limit
// sibling, mirroring TestPubSubTriggerHandler_EnforcesMaxConcurrentRuns: with
// MaxConcurrentRuns=N, firing N+1 concurrent requests at a blocking agent must
// leave exactly N runs in flight (the extra request waits on the semaphore).
func TestEventarcTriggerHandler_EnforcesMaxConcurrentRuns(t *testing.T) {
	const limit = 2
	started := make(chan struct{}, limit+1)
	release := make(chan struct{})
	cfg := defaultTriggerConfig
	cfg.MaxConcurrentRuns = limit
	sessionService := &fakes.FakeSessionService{Sessions: make(map[fakes.SessionKey]fakes.TestSession)}
	ctrl := triggers.NewEventarcController(sessionService, agent.NewSingleLoader(blockingEventarcAgent(t, started, release)), nil, nil, runner.PluginConfig{}, cfg)

	var wg sync.WaitGroup
	for range limit + 1 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctrl.EventarcTriggerHandler(httptest.NewRecorder(), buildEventarcRequest(t, "hello"))
		}()
	}

	for i := range limit {
		select {
		case <-started:
		case <-time.After(2 * time.Second):
			t.Fatalf("only %d of %d runs started", i, limit)
		}
	}
	// The (limit+1)th request must be blocked on the semaphore, not running.
	select {
	case <-started:
		t.Fatalf("a run beyond MaxConcurrentRuns=%d started; limit not enforced", limit)
	case <-time.After(250 * time.Millisecond):
	}
	close(release)
	wg.Wait()
}
