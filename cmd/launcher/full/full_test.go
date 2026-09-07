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

// TestWebUIOutranksTheAPICatchAll pins the sublauncher order in NewLauncher.
//
// With an empty path prefix the API sublauncher mounts a catch-all route that
// matches every path, including /ui/. gorilla/mux serves the first route that
// matches, so the web UI survives only because NewLauncher passes webui before
// api. Nothing else enforces that. Swap the two and the entire UI 404s, with no
// build or test failure to say so.
//
// This drives NewLauncher itself rather than assembling an equivalent launcher.
// The composition in full.go is the thing being pinned, and a test that builds
// its own launcher asserts only that its own argument order works.
func TestWebUIOutranksTheAPICatchAll(t *testing.T) {
	port := freePort(t)

	rootAgent, err := agent.New(agent.Config{Name: "test_agent", Description: "root agent"})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}

	// Surfaced rather than discarded. A bind failure would otherwise show up as
	// the 404 below, reporting a regression that did not happen.
	runErr := make(chan error, 1)
	go func() {
		// An empty API prefix is what makes the two collide. The default /api
		// does not overlap /ui/, which is why this has never bitten in practice.
		runErr <- NewLauncher().Execute(t.Context(),
			&launcher.Config{AgentLoader: agent.NewSingleLoader(rootAgent)},
			[]string{"web", "--port", fmt.Sprint(port), "webui", "api", "--path_prefix", ""})
	}()

	// Bounded per request, not only across the retry loop. Without this a
	// server that accepts and never answers hangs until the package timeout,
	// which CI leaves at the ten minute default.
	client := &http.Client{Timeout: 2 * time.Second}
	url := fmt.Sprintf("http://127.0.0.1:%d/ui/", port)

	var resp *http.Response
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		select {
		case err := <-runErr:
			t.Fatalf("launcher.Execute() returned before the server answered: %v", err)
		default:
		}
		r, err := client.Get(url)
		if err == nil {
			resp = r
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if resp == nil {
		t.Fatalf("GET %s never answered within the deadline", url)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading the response failed: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /ui/ status = %d, want %d; the API catch-all is shadowing the UI",
			resp.StatusCode, http.StatusOK)
	}
	// The UI serves index.html. The catch-all answers /ui/ with a plain-text
	// 404, so this cannot pass by accident.
	if !strings.Contains(strings.ToLower(string(body)), "<!doctype html") {
		t.Errorf("GET /ui/ did not return the UI document; first 120 bytes: %.120q", body)
	}
}
