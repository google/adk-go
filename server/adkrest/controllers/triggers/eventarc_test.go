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
	"encoding/json"
	"iter"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/gorilla/mux"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/controllers/triggers"
	"google.golang.org/adk/server/adkrest/internal/fakes"
	"google.golang.org/adk/session"
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
