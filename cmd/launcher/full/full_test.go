// Copyright 2025 Google LLC
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

package full

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/cmd/launcher"
)

// freePort returns a port nothing is listening on. There is a race between
// closing the listener and the server binding it, which is why the caller polls
// rather than assuming the server is up.
func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen() failed: %v", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	if err := ln.Close(); err != nil {
		t.Fatalf("closing the probe listener failed: %v", err)
	}
	return port
}

// launchedServer is a full launcher running for the duration of one test.
type launchedServer struct {
	base string
	// runErr carries the result of Execute, which returns only once the
	// launcher has stopped. A value here before the test context is cancelled
	// means it stopped early.
	runErr chan error
}

// assertStillRunning fails the test with the launcher's own error when Execute
// has already returned.
//
// freePort hands out a port number rather than a held listener, so another
// process can take that port before this launcher binds it. ListenAndServe then
// fails while the port still answers, and the request the test was about to
// make comes back as a connection error or as a route that is not there.
// Calling this first names the bind failure instead of those.
func (s *launchedServer) assertStillRunning(t *testing.T) {
	t.Helper()
	select {
	case err := <-s.runErr:
		s.runErr <- err // Put it back, so the later checks and the cleanup see it too.
		t.Fatalf("the launcher on %s stopped while the test was running: %v", s.base, err)
	default:
	}
}

// startFullLauncher starts NewLauncher() on a free port with every sublauncher
// enabled. apiArgs follow the api keyword.
//
// This drives NewLauncher itself rather than assembling an equivalent launcher.
// The composition in full.go is the thing being pinned, and a test that builds
// its own launcher asserts only that its own argument order works.
func startFullLauncher(t *testing.T, apiArgs ...string) *launchedServer {
	t.Helper()

	rootAgent, err := agent.New(agent.Config{Name: "test_agent", Description: "root agent"})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}

	port := freePort(t)
	args := append([]string{"web", "--port", fmt.Sprint(port), "webui", "a2a", "pubsub", "eventarc", "api"}, apiArgs...)

	s := &launchedServer{
		base:   fmt.Sprintf("http://127.0.0.1:%d", port),
		runErr: make(chan error, 1),
	}
	go func() {
		s.runErr <- NewLauncher().Execute(t.Context(),
			&launcher.Config{AgentLoader: agent.NewSingleLoader(rootAgent)}, args)
	}()

	// The test context is cancelled just before cleanups run, so Execute is on
	// its way back with a nil error. Anything else is read here because the
	// assertions can miss it entirely: when a bind is lost to another process
	// running this same test, its server answers every request and the rows
	// pass against a launcher that is not the one under test.
	t.Cleanup(func() {
		select {
		case err := <-s.runErr:
			if err != nil {
				t.Errorf("the launcher on %s failed: %v", s.base, err)
			}
		case <-time.After(10 * time.Second):
			t.Logf("the launcher on %s did not return within 10s of shutdown", s.base)
		}
	})

	// Bounded per request, not only across the retry loop. Without this a
	// server that accepts and never answers hangs until the package timeout,
	// which CI leaves at the ten minute default.
	client := &http.Client{Timeout: 2 * time.Second}

	// The web launcher registers /health before any sublauncher, so readiness
	// does not depend on the order under test.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		s.assertStillRunning(t)
		resp, err := client.Get(s.base + "/health")
		if err == nil {
			_ = resp.Body.Close()
			// Every instance of this test answers /health the same way, so a
			// reply proves only that something holds the port. Re-check before
			// handing the server to the assertions.
			s.assertStillRunning(t)
			return s
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("GET %s/health never answered within the deadline", s.base)
	return nil
}

// TestAPIDoesNotShadowEarlierRoutes checks that the routes registered before
// api still serve their own content.
//
// Given an empty path prefix, api mounts a catch-all that matches every path,
// the web UI and the A2A paths included. pubsub and eventarc need no empty
// prefix to collide: they default to /api, the prefix api also defaults to, so
// api's mount covers their paths in the shipped configuration. gorilla/mux
// serves the first route that matches, so all four survive only because
// NewLauncher passes api last. Move api to the front and they 404, with no
// build or test failure to say so.
func TestAPIDoesNotShadowEarlierRoutes(t *testing.T) {
	emptyAPIPrefixServer := startFullLauncher(t, "--path_prefix", "")
	defaultAPIPrefixServer := startFullLauncher(t)

	tests := []struct {
		name string
		// emptyAPIPrefix selects the server whose api prefix is empty.
		emptyAPIPrefix bool
		method         string
		path           string
		headers        map[string]string
		body           string
		wantStatus     int
		// wantBody is lowercase and matched against the lowercased response
		// body. It separates an answer from the route under test from the
		// plain-text 404 the REST API gives for a path it does not know.
		wantBody string
	}{
		{
			name:           "webui",
			emptyAPIPrefix: true,
			method:         http.MethodGet,
			path:           "/ui/",
			wantStatus:     http.StatusOK,
			wantBody:       "<!doctype html",
		},
		{
			name:           "a2a",
			emptyAPIPrefix: true,
			method:         http.MethodGet,
			path:           "/.well-known/agent-card.json",
			wantStatus:     http.StatusOK,
			wantBody:       "test_agent",
		},
		{
			// POST, because the trigger routes match on method. A GET falls
			// through to the catch-all whatever the order is, so it would
			// prove nothing. The empty payload is rejected by the trigger
			// handler before it runs the agent, which keeps the row off the
			// model path.
			name:       "pubsub",
			method:     http.MethodPost,
			path:       "/api/apps/test_agent/trigger/pubsub",
			body:       "{}",
			wantStatus: http.StatusBadRequest,
			wantBody:   "failed to retrieve message content",
		},
		{
			name:   "eventarc",
			method: http.MethodPost,
			path:   "/api/apps/test_agent/trigger/eventarc",
			// A Pub/Sub event type sends the handler down the branch that
			// rejects an empty message, rather than the one that runs the
			// agent.
			headers:    map[string]string{"ce-type": "google.cloud.pubsub.topic.v1.messagePublished"},
			body:       "{}",
			wantStatus: http.StatusBadRequest,
			wantBody:   "failed to retrieve message content",
		},
	}

	client := &http.Client{Timeout: 5 * time.Second}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := defaultAPIPrefixServer
			if tc.emptyAPIPrefix {
				server = emptyAPIPrefixServer
			}
			url := server.base + tc.path

			req, err := http.NewRequestWithContext(t.Context(), tc.method, url, strings.NewReader(tc.body))
			if err != nil {
				t.Fatalf("http.NewRequestWithContext(%s, %s) failed: %v", tc.method, url, err)
			}
			for name, value := range tc.headers {
				req.Header.Set(name, value)
			}

			resp, err := client.Do(req)
			if err != nil {
				server.assertStillRunning(t)
				t.Fatalf("%s %s failed: %v", tc.method, url, err)
			}
			defer func() { _ = resp.Body.Close() }()

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("reading the response failed: %v", err)
			}
			if resp.StatusCode != tc.wantStatus {
				server.assertStillRunning(t)
				t.Fatalf("%s %s status = %d, want %d; api is shadowing the route. First 120 bytes: %.120q",
					tc.method, tc.path, resp.StatusCode, tc.wantStatus, body)
			}
			if !strings.Contains(strings.ToLower(string(body)), tc.wantBody) {
				t.Errorf("%s %s body does not contain %q; first 120 bytes: %.120q",
					tc.method, tc.path, tc.wantBody, body)
			}
		})
	}
}
