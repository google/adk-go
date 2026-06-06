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

package web_test

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/web"
	"google.golang.org/adk/session"
)

// minimalSublauncher is a stub Sublauncher for testing the web launcher.
type minimalSublauncher struct{}

func (m *minimalSublauncher) Keyword() string                             { return "stub" }
func (m *minimalSublauncher) Parse(args []string) ([]string, error)       { return args, nil }
func (m *minimalSublauncher) CommandLineSyntax() string                   { return "" }
func (m *minimalSublauncher) SimpleDescription() string                   { return "" }
func (m *minimalSublauncher) UserMessage(_ string, _ func(v ...any))      {}
func (m *minimalSublauncher) SetupSubrouters(router *mux.Router, _ *launcher.Config) error {
	router.HandleFunc("/ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	return nil
}

func getFreePort(t *testing.T) int {
	t.Helper()
	addr, err := net.ResolveTCPAddr("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("net.ResolveTCPAddr() error = %v", err)
	}
	l, err := net.ListenTCP("tcp", addr)
	if err != nil {
		t.Fatalf("net.ListenTCP() error = %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port
}

func TestHTTPMiddleware_AppliedOutermostFirst(t *testing.T) {
	var order []string

	makeMiddleware := func(name string) func(http.Handler) http.Handler {
		return func(next http.Handler) http.Handler {
			return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				order = append(order, name)
				next.ServeHTTP(w, r)
			})
		}
	}

	port := getFreePort(t)
	config := &launcher.Config{
		SessionService: session.InMemoryService(),
		HTTPMiddleware: []func(http.Handler) http.Handler{
			makeMiddleware("first"),
			makeMiddleware("second"),
		},
	}

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	webLauncher := web.NewLauncher(&minimalSublauncher{})
	errCh := make(chan error, 1)
	go func() {
		args := []string{fmt.Sprintf("--port=%d", port), "stub"}
		errCh <- webLauncher.Execute(ctx, config, args)
	}()

	// Wait for the server to start accepting connections.
	deadline := time.Now().Add(5 * time.Second)
	var pingErr error
	for time.Now().Before(deadline) {
		var resp *http.Response
		resp, pingErr = http.Get(fmt.Sprintf("http://localhost:%d/ping", port))
		if pingErr == nil {
			resp.Body.Close()
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if pingErr != nil {
		t.Fatalf("server did not start in time: %v", pingErr)
	}

	cancel()
	<-errCh

	if len(order) < 2 {
		t.Fatalf("middleware not invoked; got invocation order: %v", order)
	}
	if order[0] != "first" || order[1] != "second" {
		t.Errorf("middleware invocation order = %v, want [first second ...]", order)
	}
}
