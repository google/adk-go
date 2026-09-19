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
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/cmd/launcher"
)

// fakeTriggerSublauncher stands in for a trigger sublauncher (PubSub,
// Eventarc). It mounts a trigger endpoint that decodes the request body as
// JSON, mirroring how the real triggers register their routes via
// SetupSubrouters on the base router.
//
// The fake returns 400 when the body fails to decode, like the PubSub trigger;
// the Eventarc trigger returns 500 instead. The status code of any single
// trigger is not what this test asserts - it exists only so an oversize-body
// read failure is observable. What is under test is that the base-router
// body-limit middleware covers routes that sublaunchers mount.
type fakeTriggerSublauncher struct{}

func (f fakeTriggerSublauncher) Keyword() string                    { return "faketrigger" }
func (f fakeTriggerSublauncher) Parse(_ []string) ([]string, error) { return nil, nil }
func (f fakeTriggerSublauncher) CommandLineSyntax() string          { return "faketrigger" }
func (f fakeTriggerSublauncher) SimpleDescription() string {
	return "fake trigger sublauncher for body-limit test"
}
func (f fakeTriggerSublauncher) UserMessage(_ string, _ func(v ...any)) {}

func (f fakeTriggerSublauncher) SetupSubrouters(router *mux.Router, _ *launcher.Config) error {
	subrouter := router.PathPrefix("/api").Subrouter()
	subrouter.HandleFunc("/apps/{app_name}/trigger/pubsub", func(w http.ResponseWriter, r *http.Request) {
		var msg map[string]any
		if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
			http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}).Methods(http.MethodPost)
	return nil
}

// TestBuildRouterAppliesBodyLimitMiddlewareToSublauncherTriggerRoutes verifies
// that the body-limit middleware on the base router covers trigger routes
// mounted by sublaunchers, not just routes registered directly on the base
// router.
//
// It doubles as a regression test for the middleware itself: this test fails
// if the router.Use(MaxBytesMiddleware(...)) line is removed from buildRouter,
// because the oversized body below is valid JSON and would decode successfully
// (204) instead of tripping http.MaxBytesReader.
func TestBuildRouterAppliesBodyLimitMiddlewareToSublauncherTriggerRoutes(t *testing.T) {
	sub := fakeTriggerSublauncher{}
	w := &webLauncher{
		flags:              flag.NewFlagSet("web", flag.ContinueOnError),
		config:             &webConfig{maxPayloadSize: 1024},
		sublaunchers:       []Sublauncher{sub},
		activeSublaunchers: map[string]Sublauncher{sub.Keyword(): sub},
	}

	router, err := w.buildRouter(&launcher.Config{})
	if err != nil {
		t.Fatalf("buildRouter() error = %v", err)
	}

	// A request body well under the limit decodes successfully.
	smallRec := httptest.NewRecorder()
	router.ServeHTTP(smallRec, httptest.NewRequest(http.MethodPost, "/api/apps/my-app/trigger/pubsub", strings.NewReader(`{"message":{"data":"small"}}`)))
	if smallRec.Code != http.StatusNoContent {
		t.Fatalf("small request: got status %d, want %d (%s)", smallRec.Code, http.StatusNoContent, smallRec.Body.String())
	}

	// An oversized body trips http.MaxBytesReader, so this sublauncher's
	// handler (which reads the body) observes a decode error. Without the
	// base-router middleware, the same body would decode fine and this request
	// would return 204, so the test fails on regression.
	oversized := `{"message":{"data":"` + strings.Repeat("a", 8192) + `"}}`
	oversizedRec := httptest.NewRecorder()
	router.ServeHTTP(oversizedRec, httptest.NewRequest(http.MethodPost, "/api/apps/my-app/trigger/pubsub", strings.NewReader(oversized)))
	if oversizedRec.Code != http.StatusBadRequest {
		t.Fatalf("oversized request: got status %d, want %d (%s)", oversizedRec.Code, http.StatusBadRequest, oversizedRec.Body.String())
	}
	if !strings.Contains(oversizedRec.Body.String(), "http: request body too large") {
		t.Fatalf("oversized request: got body %q, want mention of %q", oversizedRec.Body.String(), "http: request body too large")
	}
}

// recordingSublauncher records the launcher.Config it receives during router
// setup so tests can assert on the configuration the web launcher threads
// through to sublaunchers.
type recordingSublauncher struct{ maxPayloadSize int64 }

func (r *recordingSublauncher) Keyword() string                        { return "recording" }
func (r *recordingSublauncher) Parse(_ []string) ([]string, error)     { return nil, nil }
func (r *recordingSublauncher) CommandLineSyntax() string              { return "recording" }
func (r *recordingSublauncher) SimpleDescription() string              { return "recording sublauncher" }
func (r *recordingSublauncher) UserMessage(_ string, _ func(v ...any)) {}

func (r *recordingSublauncher) SetupSubrouters(_ *mux.Router, config *launcher.Config) error {
	r.maxPayloadSize = config.MaxPayloadSize
	return nil
}

// TestBuildRouterPropagatesMaxPayloadSizeToSublaunchers verifies that
// buildRouter threads the web launcher's configured max_payload_size into the
// launcher.Config it hands to sublaunchers, so the ADK REST API sublauncher
// can apply the same limit as the base router instead of the default.
func TestBuildRouterPropagatesMaxPayloadSizeToSublaunchers(t *testing.T) {
	for _, tc := range []struct {
		name string
		web  int64
		want int64
	}{
		{name: "raised", web: int64(20 << 20), want: int64(20 << 20)},
		{name: "default", web: 0, want: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sub := &recordingSublauncher{}
			w := &webLauncher{
				flags:              flag.NewFlagSet("web", flag.ContinueOnError),
				config:             &webConfig{maxPayloadSize: tc.web},
				sublaunchers:       []Sublauncher{sub},
				activeSublaunchers: map[string]Sublauncher{sub.Keyword(): sub},
			}

			if _, err := w.buildRouter(&launcher.Config{}); err != nil {
				t.Fatalf("buildRouter() error = %v", err)
			}
			if got := sub.maxPayloadSize; got != tc.want {
				t.Fatalf("config.MaxPayloadSize = %d, want %d", got, tc.want)
			}
		})
	}
}
