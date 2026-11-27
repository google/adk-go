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
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/platform"
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

// TestCreateSessionKeepsSeededStateDeltaOutOfSharedState checks that a state
// delta on a create-session event cannot reach "app:" state, which every user
// of the app reads, or "user:" state, which every session of that user reads.
// Two servers share one session service so the read can come from a different
// authenticated user.
func TestCreateSessionKeepsSeededStateDeltaOutOfSharedState(t *testing.T) {
	sessionService := session.InMemoryService()
	mallory := newSharedSessionServer(t, sessionService, "mallory")
	alice := newSharedSessionServer(t, sessionService, "alice")

	// The ID-less route is the one the ADK web UI's session import uses.
	for _, path := range []string{"/apps/app/users/mallory/sessions", "/apps/app/users/mallory/sessions/seeded"} {
		status, body := sendSessionRequest(t, mallory, http.MethodPost, path,
			`{"events":[{"author":"user","actions":{"stateDelta":{"app:prompt":"injected","user:is_admin":true}}}]}`)
		if status != http.StatusBadRequest {
			t.Errorf("POST %s with a scoped state delta: status = %d, want %d; body: %s", path, status, http.StatusBadRequest, body)
		}
	}
	if status, body := sendSessionRequest(t, mallory, http.MethodGet, "/apps/app/users/mallory/sessions", ""); status != http.StatusOK || body != "[]\n" {
		t.Errorf("list sessions after rejected creates: status = %d, body = %q; want %d, %q", status, body, http.StatusOK, "[]\n")
	}

	for _, tc := range []struct {
		name   string
		server *httptest.Server
		path   string
	}{
		{name: "same user, another session", server: mallory, path: "/apps/app/users/mallory/sessions/other"},
		{name: "another user", server: alice, path: "/apps/app/users/alice/sessions/other"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status, body := sendSessionRequest(t, tc.server, http.MethodPost, tc.path, "")
			if status != http.StatusOK {
				t.Fatalf("create session: status = %d, want %d; body: %s", status, http.StatusOK, body)
			}
			var got createSessionResponse
			if err := json.Unmarshal([]byte(body), &got); err != nil {
				t.Fatalf("json.Unmarshal(response body) failed: %v", err)
			}
			for _, key := range []string{"app:prompt", "user:is_admin"} {
				if _, ok := got.State[key]; ok {
					t.Errorf("session state has %q, written by another session's create request", key)
				}
			}
		})
	}
}

// newSharedSessionServer serves sessionService with every request
// authenticated as authenticatedUserID.
func newSharedSessionServer(t *testing.T, sessionService session.Service, authenticatedUserID string) *httptest.Server {
	t.Helper()

	server, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService: sessionService,
		Authenticator: authn.NewCustom(func(r *http.Request) (*authn.Caller, error) {
			return &authn.Caller{UserID: authenticatedUserID}, nil
		}),
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() failed: %v", err)
	}
	testServer := httptest.NewServer(server)
	t.Cleanup(testServer.Close)
	return testServer
}

// sendSessionRequest sends body, if any, as JSON and returns the response
// status and body.
func sendSessionRequest(t *testing.T, testServer *httptest.Server, method, path, body string) (int, string) {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), method, testServer.URL+path, strings.NewReader(body))
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
	return resp.StatusCode, string(respBody)
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

// TestCreateSessionStampsEventsWithoutTimestamp pins that an event sent to
// create-session without a timestamp is stored at the time it was stored, as
// adk-python's Event defaults it, while an event's own timestamp is kept.
//
// Stored with the zero time instead, the event moved the session's last update
// back to year 1: lastUpdateTime read -62135596800, and the session sorted
// ahead of every older session in the list.
func TestCreateSessionStampsEventsWithoutTimestamp(t *testing.T) {
	services := map[string]func(t *testing.T) session.Service{
		"inmemory": func(*testing.T) session.Service { return session.InMemoryService() },
		"sqlite":   newSQLiteSessionService,
	}
	for serviceName, newService := range services {
		t.Run(serviceName, func(t *testing.T) {
			server, err := adkrest.NewServer(adkrest.ServerConfig{
				SessionService: newService(t),
				Authenticator: authn.NewCustom(func(r *http.Request) (*authn.Caller, error) {
					return &authn.Caller{UserID: "u"}, nil
				}),
			})
			if err != nil {
				t.Fatalf("adkrest.NewServer() failed: %v", err)
			}
			// Every request reads the clock the test last set, so each
			// session's creation and each stamped event land at a known time.
			var now atomic.Pointer[time.Time]
			testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				clock := *now.Load()
				ctx := platform.WithTimeProvider(r.Context(), func() time.Time { return clock })
				server.ServeHTTP(w, r.WithContext(ctx))
			}))
			t.Cleanup(testServer.Close)
			base := testServer.URL + "/apps/app/users/u/sessions"

			older := time.Date(2026, 11, 17, 10, 0, 0, 0, time.UTC)
			newer := older.Add(time.Hour + 123456*time.Microsecond)
			now.Store(&older)
			if status, body := doTimestampRequest(t, http.MethodPost, base+"/older", `{}`); status != http.StatusOK {
				t.Fatalf("create older session = %d, want 200; body: %s", status, body)
			}
			now.Store(&newer)
			status, body := doTimestampRequest(t, http.MethodPost, base+"/newer", `{"events":[
				{"author":"user","timestamp":1733040000.5},
				{"author":"model"}
			]}`)
			if status != http.StatusOK {
				t.Fatalf("create newer session = %d, want 200; body: %s", status, body)
			}

			wantNewer := stampedSession{
				ID:             "newer",
				LastUpdateTime: newer.Unix(),
				// The first event keeps its own timestamp; the second, sent
				// without one, carries the time it was stored.
				EventTimestamps: []float64{1733040000.5, float64(newer.UnixMicro()) / 1e6},
			}
			if got := decodeStampedSession(t, body); !stampedSessionsEqual(got, wantNewer) {
				t.Errorf("create response = %+v, want %+v", got, wantNewer)
			}
			status, body = doTimestampRequest(t, http.MethodGet, base+"/newer", "")
			if status != http.StatusOK {
				t.Fatalf("GET newer session = %d, want 200; body: %s", status, body)
			}
			if got := decodeStampedSession(t, body); !stampedSessionsEqual(got, wantNewer) {
				t.Errorf("GET response = %+v, want %+v", got, wantNewer)
			}

			status, body = doTimestampRequest(t, http.MethodGet, base, "")
			if status != http.StatusOK {
				t.Fatalf("list sessions = %d, want 200; body: %s", status, body)
			}
			var listed []stampedSession
			if err := json.Unmarshal([]byte(body), &listed); err != nil {
				t.Fatalf("json.Unmarshal(%s) failed: %v", body, err)
			}
			// List is ordered by last update time, oldest first.
			want := []stampedSession{
				{ID: "older", LastUpdateTime: older.Unix()},
				{ID: "newer", LastUpdateTime: newer.Unix()},
			}
			if !slices.EqualFunc(listed, want, func(a, b stampedSession) bool {
				return a.ID == b.ID && a.LastUpdateTime == b.LastUpdateTime
			}) {
				t.Errorf("list = %+v, want %+v", listed, want)
			}
		})
	}
}

// stampedSession is the part of a session response that
// TestCreateSessionStampsEventsWithoutTimestamp checks.
type stampedSession struct {
	ID              string
	LastUpdateTime  int64
	EventTimestamps []float64
}

func (s *stampedSession) UnmarshalJSON(b []byte) error {
	var wire struct {
		ID             string `json:"id"`
		LastUpdateTime int64  `json:"lastUpdateTime"`
		Events         []struct {
			Timestamp float64 `json:"timestamp"`
		} `json:"events"`
	}
	if err := json.Unmarshal(b, &wire); err != nil {
		return err
	}
	s.ID, s.LastUpdateTime = wire.ID, wire.LastUpdateTime
	s.EventTimestamps = nil
	for _, e := range wire.Events {
		s.EventTimestamps = append(s.EventTimestamps, e.Timestamp)
	}
	return nil
}

func decodeStampedSession(t *testing.T, body string) stampedSession {
	t.Helper()
	var got stampedSession
	if err := json.Unmarshal([]byte(body), &got); err != nil {
		t.Fatalf("json.Unmarshal(%s) failed: %v", body, err)
	}
	return got
}

func stampedSessionsEqual(a, b stampedSession) bool {
	return a.ID == b.ID && a.LastUpdateTime == b.LastUpdateTime && slices.Equal(a.EventTimestamps, b.EventTimestamps)
}
