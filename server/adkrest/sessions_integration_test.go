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

package adkrest_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/session"
)

// unknownLengthReader hides the underlying reader's size so net/http uses chunked transfer encoding.
type unknownLengthReader struct {
	io.Reader
}

type createSessionResponse struct {
	State  map[string]any `json:"state"`
	Events []struct {
		ID           string `json:"id"`
		InvocationID string `json:"invocationId"`
		Author       string `json:"author"`
	} `json:"events"`
}

func TestCreateSessionPreservesChunkedBody(t *testing.T) {
	testServer := newChunkedSessionServer(t)
	requestBody := unknownLengthReader{strings.NewReader(`{
		"state":{"theme":"dark"},
		"events":[{"id":"event-1","invocationId":"invocation-1","author":"user"}]
	}`)}
	req, err := http.NewRequestWithContext(
		t.Context(),
		http.MethodPost,
		testServer.URL+"/apps/test-app/users/test-user/sessions/test-session",
		requestBody,
	)
	if err != nil {
		t.Fatalf("http.NewRequestWithContext() failed: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := testServer.Client().Do(req)
	if err != nil {
		t.Fatalf("client.Do() failed: %v", err)
	}
	respBody, readErr := io.ReadAll(resp.Body)
	closeErr := resp.Body.Close()
	if readErr != nil {
		t.Fatalf("io.ReadAll(response body) failed: %v", readErr)
	}
	if closeErr != nil {
		t.Fatalf("response body Close() failed: %v", closeErr)
	}
	if got, want := resp.StatusCode, http.StatusOK; got != want {
		t.Fatalf("create session status = %d, want %d; body: %s", got, want, respBody)
	}

	var got createSessionResponse
	if err := json.Unmarshal(respBody, &got); err != nil {
		t.Fatalf("json.Unmarshal(response body) failed: %v; body: %s", err, respBody)
	}
	if gotTheme, ok := got.State["theme"].(string); !ok || gotTheme != "dark" {
		t.Errorf("created session state theme = %v, want dark", got.State["theme"])
	}
	if gotCount, want := len(got.Events), 1; gotCount != want {
		t.Fatalf("created session event count = %d, want %d", gotCount, want)
	}
	if gotID, want := got.Events[0].ID, "event-1"; gotID != want {
		t.Errorf("created session event ID = %q, want %q", gotID, want)
	}
	if gotInvocationID, want := got.Events[0].InvocationID, "invocation-1"; gotInvocationID != want {
		t.Errorf("created session event invocation ID = %q, want %q", gotInvocationID, want)
	}
	if gotAuthor, want := got.Events[0].Author, "user"; gotAuthor != want {
		t.Errorf("created session event author = %q, want %q", gotAuthor, want)
	}
}

func TestCreateSessionAllowsEmptyChunkedBody(t *testing.T) {
	testServer := newChunkedSessionServer(t)
	req, err := http.NewRequestWithContext(
		t.Context(),
		http.MethodPost,
		testServer.URL+"/apps/test-app/users/test-user/sessions/test-session",
		unknownLengthReader{strings.NewReader("")},
	)
	if err != nil {
		t.Fatalf("http.NewRequestWithContext() failed: %v", err)
	}
	req.TransferEncoding = []string{"chunked"}

	resp, err := testServer.Client().Do(req)
	if err != nil {
		t.Fatalf("client.Do() failed: %v", err)
	}
	respBody, readErr := io.ReadAll(resp.Body)
	closeErr := resp.Body.Close()
	if readErr != nil {
		t.Fatalf("io.ReadAll(response body) failed: %v", readErr)
	}
	if closeErr != nil {
		t.Fatalf("response body Close() failed: %v", closeErr)
	}
	if got, want := resp.StatusCode, http.StatusOK; got != want {
		t.Fatalf("create session status = %d, want %d; body: %s", got, want, respBody)
	}

	var got createSessionResponse
	if err := json.Unmarshal(respBody, &got); err != nil {
		t.Fatalf("json.Unmarshal(response body) failed: %v; body: %s", err, respBody)
	}
	if len(got.State) != 0 {
		t.Errorf("created session state = %v, want empty state", got.State)
	}
	if len(got.Events) != 0 {
		t.Errorf("created session events = %v, want no events", got.Events)
	}
}

func TestCreateSessionRejectsMalformedChunkedBody(t *testing.T) {
	testServer := newChunkedSessionServer(t)
	req, err := http.NewRequestWithContext(
		t.Context(),
		http.MethodPost,
		testServer.URL+"/apps/test-app/users/test-user/sessions/test-session",
		unknownLengthReader{strings.NewReader(`{"state":`)},
	)
	if err != nil {
		t.Fatalf("http.NewRequestWithContext() failed: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := testServer.Client().Do(req)
	if err != nil {
		t.Fatalf("client.Do() failed: %v", err)
	}
	respBody, readErr := io.ReadAll(resp.Body)
	closeErr := resp.Body.Close()
	if readErr != nil {
		t.Fatalf("io.ReadAll(response body) failed: %v", readErr)
	}
	if closeErr != nil {
		t.Fatalf("response body Close() failed: %v", closeErr)
	}
	if got, want := resp.StatusCode, http.StatusBadRequest; got != want {
		t.Fatalf("create session status = %d, want %d; body: %s", got, want, respBody)
	}
}

func newChunkedSessionServer(t *testing.T) *httptest.Server {
	t.Helper()

	server, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService: session.InMemoryService(),
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() failed: %v", err)
	}
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ContentLength != -1 || !slices.Contains(r.TransferEncoding, "chunked") {
			http.Error(w, "test request was not chunked", http.StatusInternalServerError)
			return
		}
		server.ServeHTTP(w, r)
	}))
	t.Cleanup(testServer.Close)
	return testServer
}
