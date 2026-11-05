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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
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
	testServer := newChunkedSessionServer(t, "test-user")
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
	testServer := newChunkedSessionServer(t, "test-user")
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
	testServer := newChunkedSessionServer(t, "test-user")
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

func newChunkedSessionServer(t *testing.T, authenticatedUserID string) *httptest.Server {
	t.Helper()

	server, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService: session.InMemoryService(),
		Authenticator: authn.NewCustom(func(r *http.Request) (*authn.Caller, error) {
			return &authn.Caller{UserID: authenticatedUserID}, nil
		}),
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

// TestCreateSessionBodyIDRoundTrip pins that every session ID the create
// endpoint accepts in its request body can afterwards be read, updated and
// deleted through /sessions/{session_id} with the ID percent-encoded. An ID the
// session routes cannot address must be refused with a 400 and not stored,
// because a session created under it could only ever be listed.
func TestCreateSessionBodyIDRoundTrip(t *testing.T) {
	ids := []string{
		"plain",
		"with-dash_and.dot",
		"...",
		".hidden",
		"trailing.",
		"a b",
		"a+b",
		"a%b",
		"%2F",
		"a?b",
		"a#b",
		"a;b",
		"a\\b",
		"ünïcødé",
		"a\tb",
		"a\nb",
		"a\x00b",
		"/",
		"x/y",
		"a/",
		"/a",
		".",
		"..",
	}
	for _, id := range ids {
		t.Run(url.PathEscape(id), func(t *testing.T) {
			base := newSessionIDServer(t) + "/apps/app/users/u/sessions"
			body, err := json.Marshal(map[string]string{"sessionId": id})
			if err != nil {
				t.Fatalf("json.Marshal() failed: %v", err)
			}
			status, respBody := doSessionRequest(t, http.MethodPost, base, string(body))
			if status != http.StatusOK {
				if status != http.StatusBadRequest {
					t.Fatalf("POST %s with sessionId %q = %d, want 200 or 400; body: %s", base, id, status, respBody)
				}
				if listed := listSessionIDs(t, base); len(listed) != 0 {
					t.Errorf("sessionId %q was refused with 400 but the list holds %q", id, listed)
				}
				return
			}
			if got := decodeSessionID(t, respBody); got != id {
				t.Fatalf("POST returned session ID %q, want %q", got, id)
			}

			sessionURL := base + "/" + url.PathEscape(id)
			for _, step := range []struct {
				method string
				body   string
			}{
				{http.MethodGet, ""},
				{http.MethodPatch, `{"stateDelta":{"k":"v"}}`},
				{http.MethodGet, ""},
			} {
				status, respBody := doSessionRequest(t, step.method, sessionURL, step.body)
				if status != http.StatusOK {
					t.Fatalf("%s %s after creating sessionId %q = %d, want 200; body: %s", step.method, sessionURL, id, status, respBody)
				}
				if got := decodeSessionID(t, respBody); got != id {
					t.Fatalf("%s %s returned session ID %q, want %q", step.method, sessionURL, got, id)
				}
			}
			if status, respBody := doSessionRequest(t, http.MethodDelete, sessionURL, ""); status != http.StatusOK {
				t.Fatalf("DELETE %s after creating sessionId %q = %d, want 200; body: %s", sessionURL, id, status, respBody)
			}
			if listed := listSessionIDs(t, base); len(listed) != 0 {
				t.Errorf("DELETE %s left sessions %q, want none", sessionURL, listed)
			}
		})
	}
}

func newSessionIDServer(t *testing.T) string {
	t.Helper()
	server, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService: session.InMemoryService(),
		Authenticator: authn.NewCustom(func(r *http.Request) (*authn.Caller, error) {
			return &authn.Caller{UserID: "u"}, nil
		}),
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() failed: %v", err)
	}
	testServer := httptest.NewServer(server)
	t.Cleanup(testServer.Close)
	return testServer.URL
}

// doSessionRequest sends one request without following redirects, so a path
// the router rewrites shows up as the redirect it is rather than as whatever
// the rewritten path happens to serve.
func doSessionRequest(t *testing.T, method, target, body string) (int, []byte) {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), method, target, strings.NewReader(body))
	if err != nil {
		t.Fatalf("http.NewRequestWithContext(%s, %s) failed: %v", method, target, err)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("%s %s failed: %v", method, target, err)
	}
	respBody, readErr := io.ReadAll(resp.Body)
	closeErr := resp.Body.Close()
	if readErr != nil {
		t.Fatalf("io.ReadAll(response body) failed: %v", readErr)
	}
	if closeErr != nil {
		t.Fatalf("response body Close() failed: %v", closeErr)
	}
	return resp.StatusCode, respBody
}

func decodeSessionID(t *testing.T, body []byte) string {
	t.Helper()
	var got struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("json.Unmarshal(%s) failed: %v", body, err)
	}
	return got.ID
}

func listSessionIDs(t *testing.T, listURL string) []string {
	t.Helper()
	status, body := doSessionRequest(t, http.MethodGet, listURL, "")
	if status != http.StatusOK {
		t.Fatalf("GET %s = %d, want 200; body: %s", listURL, status, body)
	}
	var sessions []struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(body, &sessions); err != nil {
		t.Fatalf("json.Unmarshal(%s) failed: %v", body, err)
	}
	ids := make([]string, 0, len(sessions))
	for _, s := range sessions {
		ids = append(ids, s.ID)
	}
	return ids
}

// TestCreateSessionEventTimestamps pins that whatever number a client sends as
// an event timestamp, create-session either refuses it with a 400 and stores
// nothing, or stores a session that GET and the list endpoint still return.
//
// It runs against the in-memory and the SQLite session services because they
// fail differently. A timestamp in the first second of 1970 broke both, through
// the REST model's conversion, while an out-of-range one is stored by both but
// SQLite cannot read the session back.
func TestCreateSessionEventTimestamps(t *testing.T) {
	services := map[string]func(t *testing.T) session.Service{
		"inmemory": func(*testing.T) session.Service { return session.InMemoryService() },
		"sqlite":   newSQLiteSessionService,
	}
	tests := []struct {
		name      string
		timestamp string
		wantOK    bool
	}{
		{name: "sub-second after epoch", timestamp: "0.5", wantOK: true},
		{name: "sub-second before epoch", timestamp: "-0.5", wantOK: true},
		{name: "epoch seconds with microseconds", timestamp: "1733040000.123456", wantOK: true},
		{name: "first second of year 1", timestamp: "-62135596800", wantOK: true},
		{name: "last second of year 9999", timestamp: "253402300799", wantOK: true},
		{name: "epoch milliseconds", timestamp: "1733040000123", wantOK: false},
		{name: "overflows int64 microseconds", timestamp: "1e13", wantOK: false},
		{name: "year 10000", timestamp: "253402300800", wantOK: false},
		{name: "before year 1", timestamp: "-62135596801", wantOK: false},
	}
	for serviceName, newService := range services {
		for _, tt := range tests {
			t.Run(serviceName+"/"+tt.name, func(t *testing.T) {
				server, err := adkrest.NewServer(adkrest.ServerConfig{
					SessionService: newService(t),
					Authenticator: authn.NewCustom(func(r *http.Request) (*authn.Caller, error) {
						return &authn.Caller{UserID: "u"}, nil
					}),
				})
				if err != nil {
					t.Fatalf("adkrest.NewServer() failed: %v", err)
				}
				testServer := httptest.NewServer(server)
				t.Cleanup(testServer.Close)
				base := testServer.URL + "/apps/app/users/u/sessions"

				// The create route without a session ID: a 500 there would not
				// carry the generated ID, leaving the client no way to find the
				// session again.
				status, body := doTimestampRequest(t, http.MethodPost, base,
					`{"events":[{"author":"user","timestamp":`+tt.timestamp+`}]}`)
				if !tt.wantOK {
					if status != http.StatusBadRequest {
						t.Fatalf("create with timestamp %s = %d, want 400; body: %s", tt.timestamp, status, body)
					}
					if status, body := doTimestampRequest(t, http.MethodGet, base, ""); status != http.StatusOK || strings.TrimSpace(body) != "[]" {
						t.Errorf("list after refused create = %d %s, want 200 []", status, body)
					}
					return
				}
				if status != http.StatusOK {
					t.Fatalf("create with timestamp %s = %d, want 200; body: %s", tt.timestamp, status, body)
				}
				var created struct {
					ID string `json:"id"`
				}
				if err := json.Unmarshal([]byte(body), &created); err != nil {
					t.Fatalf("json.Unmarshal(%s) failed: %v", body, err)
				}
				if status, body := doTimestampRequest(t, http.MethodGet, base+"/"+created.ID, ""); status != http.StatusOK {
					t.Errorf("GET created session = %d, want 200; body: %s", status, body)
				}
				if status, body := doTimestampRequest(t, http.MethodGet, base, ""); status != http.StatusOK || !strings.Contains(body, created.ID) {
					t.Errorf("list after create = %d %s, want 200 listing %q", status, body, created.ID)
				}
			})
		}
	}
}

var sqliteDBCounter atomic.Int64

func newSQLiteSessionService(t *testing.T) session.Service {
	t.Helper()
	// A named shared-cache database per test, so tests do not see each
	// other's sessions.
	dsn := fmt.Sprintf("file:adkrest-timestamps-%d?mode=memory&cache=shared", sqliteDBCounter.Add(1))
	db, err := gorm.Open(sqlite.Open(dsn), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		t.Fatalf("gorm.Open() failed: %v", err)
	}
	t.Cleanup(func() {
		if sqlDB, err := db.DB(); err == nil {
			_ = sqlDB.Close()
		}
	})
	service, err := database.NewSessionServiceFromDB(db)
	if err != nil {
		t.Fatalf("database.NewSessionServiceFromDB() failed: %v", err)
	}
	if err := database.AutoMigrate(service); err != nil {
		t.Fatalf("database.AutoMigrate() failed: %v", err)
	}
	return service
}

func doTimestampRequest(t *testing.T, method, target, body string) (int, string) {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), method, target, strings.NewReader(body))
	if err != nil {
		t.Fatalf("http.NewRequestWithContext(%s, %s) failed: %v", method, target, err)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("%s %s failed: %v", method, target, err)
	}
	respBody, readErr := io.ReadAll(resp.Body)
	closeErr := resp.Body.Close()
	if readErr != nil {
		t.Fatalf("io.ReadAll(response body) failed: %v", readErr)
	}
	if closeErr != nil {
		t.Fatalf("response body Close() failed: %v", closeErr)
	}
	return resp.StatusCode, string(respBody)
}
