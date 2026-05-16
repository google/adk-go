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

package main

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"testing"
	"time"
)

type mockRoundTripper struct {
	handler func(req *http.Request) *http.Response
}

func (m *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return m.handler(req), nil
}

func newMockResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(bytes.NewBufferString(body)),
		Header:     make(http.Header),
	}
}

func setupMockHTTP(handler func(req *http.Request) *http.Response) {
	httpClient.Transport = &mockRoundTripper{handler: handler}
	GitHubToken = "test-token"
	Owner = "test-owner"
	Repo = "test-repo"
}

func TestGetEnvDefaults(t *testing.T) {
	os.Unsetenv("OWNER")
	val := getEnv("OWNER", "google")
	if val != "google" {
		t.Errorf("expected default 'google', got %s", val)
	}
}

func TestGetEnvInt_Invalid(t *testing.T) {
	os.Setenv("TEST_INT", "invalid")
	val := getEnvInt("TEST_INT", 5)
	if val != 5 {
		t.Errorf("expected fallback 5, got %d", val)
	}
}

func TestAPICallCounter(t *testing.T) {
	ResetAPICallCount()

	incrementAPICallCount()
	incrementAPICallCount()

	if GetAPICallCount() != 2 {
		t.Errorf("expected 2 API calls, got %d", GetAPICallCount())
	}

	ResetAPICallCount()
	if GetAPICallCount() != 0 {
		t.Errorf("expected reset to 0")
	}
}

func TestFormatDays_Whole(t *testing.T) {
	result := formatDays(168) // 7 days
	if result != "7" {
		t.Errorf("expected 7, got %s", result)
	}
}

func TestFormatDays_Decimal(t *testing.T) {
	result := formatDays(12) // 0.5 days
	if result != "0.5" {
		t.Errorf("expected 0.5, got %s", result)
	}
}

func TestIsMaintainer(t *testing.T) {
	maintainers := []string{"alice", "bob"}

	if !isMaintainer("alice", maintainers) {
		t.Errorf("alice should be maintainer")
	}

	if isMaintainer("charlie", maintainers) {
		t.Errorf("charlie should not be maintainer")
	}
}

func TestReplayHistory_LastAction(t *testing.T) {
	now := time.Now()

	history := []TimelineEvent{
		{
			Type:  "created",
			Actor: "user1",
			Time:  now.Add(-10 * time.Hour),
		},
		{
			Type:  "commented",
			Actor: "maintainer1",
			Time:  now,
			Data:  "please provide logs",
		},
	}

	maintainers := []string{"maintainer1"}

	state := replayHistoryToFindState(history, maintainers, "user1")

	if state.LastActionRole != "maintainer" {
		t.Errorf("expected maintainer, got %s", state.LastActionRole)
	}

	if state.LastActionType != "commented" {
		t.Errorf("expected commented, got %s", state.LastActionType)
	}

	if state.LastCommentText == nil {
		t.Errorf("expected comment text, got nil")
	}
}

func TestFormatPrompt(t *testing.T) {
	template := "Hello {NAME}"
	result := formatPrompt(template, map[string]string{
		"NAME": "Shiva",
	})

	if result != "Hello Shiva" {
		t.Errorf("unexpected result: %s", result)
	}
}

func TestGetIssueState_ActiveUserComment(t *testing.T) {
	now := time.Now().UTC().Format(time.RFC3339)

	mockGraphQL := `{
		"data": {
			"repository": {
				"issue": {
					"author": {"login": "user1"},
					"createdAt": "` + now + `",
					"labels": {"nodes": []},
					"comments": {
						"nodes": [{
							"author": {"login": "user1"},
							"body": "I have updated this",
							"createdAt": "` + now + `"
						}]
					},
					"userContentEdits": {"nodes": []},
					"timelineItems": {"nodes": []}
				}
			}
		}
	}`

	mockMaintainers := `[{"login":"maintainer1"}]`

	setupMockHTTP(func(req *http.Request) *http.Response {
		if req.URL.Path == "/graphql" {
			return newMockResponse(200, mockGraphQL)
		}
		if req.URL.Path == "/repos/test-owner/test-repo/collaborators" {
			return newMockResponse(200, mockMaintainers)
		}
		return newMockResponse(404, "{}")
	})

	res, err := getIssueState(nil, IssueTargetArgs{IssueNumber: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if res["status"] != "success" {
		t.Errorf("expected success")
	}

	if res["last_action_role"] != "author" {
		t.Errorf("expected author role")
	}

	if res["is_stale"] != false {
		t.Errorf("should not be stale")
	}
}

func TestGetIssueState_MaintainerQuestion(t *testing.T) {
	oldTime := time.Now().Add(-200 * time.Hour).UTC().Format(time.RFC3339)

	mockGraphQL := `{
		"data": {
			"repository": {
				"issue": {
					"author": {"login": "user1"},
					"createdAt": "` + oldTime + `",
					"labels": {"nodes": []},
					"comments": {
						"nodes": [{
							"author": {"login": "maintainer1"},
							"body": "Can you provide logs?",
							"createdAt": "` + oldTime + `"
						}]
					},
					"userContentEdits": {"nodes": []},
					"timelineItems": {"nodes": []}
				}
			}
		}
	}`

	mockMaintainers := `[{"login":"maintainer1"}]`

	setupMockHTTP(func(req *http.Request) *http.Response {
		if req.URL.Path == "/graphql" {
			return newMockResponse(200, mockGraphQL)
		}
		if req.URL.Path == "/repos/test-owner/test-repo/collaborators" {
			return newMockResponse(200, mockMaintainers)
		}
		return newMockResponse(404, "{}")
	})

	STALE_HOURS_THRESHOLD = 168 // 7 days

	res, _ := getIssueState(nil, IssueTargetArgs{IssueNumber: 2})

	if res["last_action_role"] != "maintainer" {
		t.Errorf("expected maintainer")
	}

	if res["days_since_activity"].(float64) < 7 {
		t.Errorf("expected stale candidate")
	}
}

func TestGetIssueState_SilentEditTriggersAlert(t *testing.T) {
	now := time.Now().UTC().Format(time.RFC3339)

	mockGraphQL := `{
		"data": {
			"repository": {
				"issue": {
					"author": {"login": "user1"},
					"createdAt": "` + now + `",
					"labels": {"nodes": []},
					"comments": {"nodes": []},
					"userContentEdits": {
						"nodes": [{
							"editor": {"login": "user1"},
							"editedAt": "` + now + `"
						}]
					},
					"timelineItems": {"nodes": []}
				}
			}
		}
	}`

	mockMaintainers := `[{"login":"maintainer1"}]`

	setupMockHTTP(func(req *http.Request) *http.Response {
		if req.URL.Path == "/graphql" {
			return newMockResponse(200, mockGraphQL)
		}
		if req.URL.Path == "/repos/test-owner/test-repo/collaborators" {
			return newMockResponse(200, mockMaintainers)
		}
		return newMockResponse(404, "{}")
	})

	res, _ := getIssueState(nil, IssueTargetArgs{IssueNumber: 3})

	if res["maintainer_alert_needed"] != true {
		t.Errorf("expected alert to be triggered")
	}
}

func TestAddStaleLabelAndComment(t *testing.T) {
	callCount := 0

	setupMockHTTP(func(req *http.Request) *http.Response {
		callCount++
		return newMockResponse(200, `{}`)
	})

	STALE_HOURS_THRESHOLD = 168
	CLOSE_HOURS_AFTER_STALE_THRESHOLD = 168

	res, err := addStaleLabelAndComment(nil, IssueTargetArgs{IssueNumber: 10})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if res.Status != "success" {
		t.Errorf("expected success")
	}

	if callCount != 2 {
		t.Errorf("expected 2 API calls (comment + label), got %d", callCount)
	}
}

func TestCloseAsStale(t *testing.T) {
	callCount := 0

	setupMockHTTP(func(req *http.Request) *http.Response {
		callCount++
		return newMockResponse(200, `{}`)
	})

	CLOSE_HOURS_AFTER_STALE_THRESHOLD = 168

	res, err := closeAsStale(nil, IssueTargetArgs{IssueNumber: 11})
	if err != nil {
		t.Fatalf("unexpected error")
	}

	if res.Status != "success" {
		t.Errorf("expected success")
	}

	if callCount != 2 {
		t.Errorf("expected 2 API calls (comment + close), got %d", callCount)
	}
}
