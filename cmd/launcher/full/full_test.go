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
	"google.golang.org/adk/v2/cmd/launcher/web"
	"google.golang.org/adk/v2/cmd/launcher/web/api"
	"google.golang.org/adk/v2/cmd/launcher/web/webui"
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

// TestWebUIOutranksTheAPICatchAll pins the order the sublaunchers are passed to
// web.NewLauncher in NewLauncher.
//
// With an empty path prefix the API sublauncher mounts a catch-all route that
// matches every path, including /ui/. gorilla/mux serves the first route that
// matches, so the UI survives only because webui is registered before api.
// Nothing else enforces that. Swap the two arguments and the entire UI
// disappears behind the API, with no build or test failure to say so.
//
// The assertion is on the served response rather than on the argument order, so
// it still holds if the mounting changes shape.
func TestWebUIOutranksTheAPICatchAll(t *testing.T) {
	port := freePort(t)

	l := web.NewLauncher(webui.NewLauncher(), api.NewLauncher())
	// An empty API prefix is what makes the two collide. The default /api does
	// not overlap /ui/, which is why this has never bitten in practice.
	if _, err := l.Parse([]string{
		"--port", fmt.Sprint(port),
		"webui",
		"api", "--path_prefix", "",
	}); err != nil {
		t.Fatalf("Parse() failed: %v", err)
	}

	rootAgent, err := agent.New(agent.Config{Name: "test_agent", Description: "root agent"})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}

	go func() {
		// Run blocks until the context is done, which t.Context handles at the
		// end of the test.
		_ = l.Run(t.Context(), &launcher.Config{AgentLoader: agent.NewSingleLoader(rootAgent)})
	}()

	url := fmt.Sprintf("http://127.0.0.1:%d/ui/", port)
	var resp *http.Response
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		resp, err = http.Get(url)
		if err == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("GET %s never succeeded: %v", url, err)
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
	// The UI serves index.html. The REST catch-all answers with the API
	// router's own 404 body instead.
	if !strings.Contains(strings.ToLower(string(body)), "<!doctype html") {
		t.Errorf("GET /ui/ did not return the UI document; first 120 bytes: %.120q", body)
	}
}
