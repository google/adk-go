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

package web

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/cmd/launcher"
)

type runnableTestSublauncher struct {
	keyword string
	route   string
}

func (s *runnableTestSublauncher) Keyword() string                       { return s.keyword }
func (s *runnableTestSublauncher) Parse(args []string) ([]string, error) { return args, nil }
func (s *runnableTestSublauncher) CommandLineSyntax() string             { return "" }
func (s *runnableTestSublauncher) SimpleDescription() string             { return "test sublauncher" }
func (s *runnableTestSublauncher) UserMessage(string, func(...any))      {}

func (s *runnableTestSublauncher) SetupSubrouters(router *mux.Router, _ *launcher.Config) error {
	router.HandleFunc(s.route, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	return nil
}

func TestNewRunnableUsesValuesAndActivatesSublaunchers(t *testing.T) {
	sub := &runnableTestSublauncher{keyword: "test", route: "/test"}
	r, err := NewRunnable(Config{Host: "127.0.0.1", Port: 9099, ReadTimeout: 3 * time.Second}, sub)
	if err != nil {
		t.Fatalf("NewRunnable() failed: %v", err)
	}

	w, ok := r.(*webLauncher)
	if !ok {
		t.Fatalf("NewRunnable() returned %T, want *webLauncher", r)
	}
	if w.config.port != 9099 {
		t.Errorf("port = %d, want 9099", w.config.port)
	}
	if w.config.host != "127.0.0.1" {
		t.Errorf("host = %q, want %q", w.config.host, "127.0.0.1")
	}
	if w.config.readTimeout != 3*time.Second {
		t.Errorf("readTimeout = %v, want 3s", w.config.readTimeout)
	}
	if w.config.writeTimeout != 15*time.Second {
		t.Errorf("writeTimeout = %v, want the 15s default", w.config.writeTimeout)
	}
	if _, ok := w.activeSublaunchers["test"]; !ok {
		t.Errorf("activeSublaunchers = %v, want the test sublauncher", w.activeSublaunchers)
	}

	router, err := w.buildRouter(&launcher.Config{})
	if err != nil {
		t.Fatalf("buildRouter() failed: %v", err)
	}
	rec := httptest.NewRecorder()
	// The bind host is loopback, so the request's Host must be too; otherwise
	// originguard refuses it as a possible DNS-rebinding request.
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "http://127.0.0.1/test", nil))
	if rec.Code != http.StatusNoContent {
		t.Errorf("GET /test = %d, want %d", rec.Code, http.StatusNoContent)
	}
}

func TestNewRunnableDefaults(t *testing.T) {
	r, err := NewRunnable(Config{})
	if err != nil {
		t.Fatalf("NewRunnable() failed: %v", err)
	}
	w := r.(*webLauncher)
	if w.config.port != 8080 {
		t.Errorf("port = %d, want 8080", w.config.port)
	}
	if w.config.host != defaultHost {
		t.Errorf("host = %q, want %q", w.config.host, defaultHost)
	}
	if w.config.shutdownTimeout != 15*time.Second {
		t.Errorf("shutdownTimeout = %v, want 15s", w.config.shutdownTimeout)
	}
}

func TestNewRunnableRejectsInvalidPort(t *testing.T) {
	if _, err := NewRunnable(Config{Port: 65536}); err == nil {
		t.Fatal("NewRunnable() succeeded for an out-of-range port, want an error")
	}
}

func TestNewRunnableRejectsDuplicateKeywords(t *testing.T) {
	sub := &runnableTestSublauncher{keyword: "test", route: "/test"}
	if _, err := NewRunnable(Config{}, sub, sub); err == nil {
		t.Fatal("NewRunnable() succeeded for duplicate keywords, want an error")
	}
}

func TestNewRunnableWithoutSublaunchersFailsRun(t *testing.T) {
	r, err := NewRunnable(Config{})
	if err != nil {
		t.Fatalf("NewRunnable() failed: %v", err)
	}
	if err := r.Run(context.Background(), &launcher.Config{}); err == nil {
		t.Fatal("Run() succeeded without a sublauncher, want an error")
	}
}
