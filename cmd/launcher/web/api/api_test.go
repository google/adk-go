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

package api

import (
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/session"
)

// TestAPIServerHonorsConfigMaxPayloadSize verifies that launcher.Config.
// MaxPayloadSize is passed through to the ADK REST API server, so the API
// server's body limit can be raised above its 10 MiB default.
//
// It fails without that plumbing: the API server would stay at
// adkrest.DefaultMaxPayloadSize and reject the raiseBody below (between the
// default and the configured limit) with 400.
func TestAPIServerHonorsConfigMaxPayloadSize(t *testing.T) {
	// 20 MiB - twice the adkrest default, clearly above the default limit.
	const configuredMax = int64(20 << 20)

	a := &apiLauncher{
		flags:  flag.NewFlagSet("api", flag.ContinueOnError),
		config: &apiConfig{frontendAddress: "localhost:8080"},
	}
	router := mux.NewRouter().StrictSlash(true)
	if err := a.SetupSubrouters(router, &launcher.Config{
		SessionService: session.InMemoryService(),
		MaxPayloadSize: configuredMax,
	}); err != nil {
		t.Fatalf("SetupSubrouters: %v", err)
	}

	sessionsURL := "/apps/my-app/users/u1/sessions"
	sendSessionBody := func(padding int) *httptest.ResponseRecorder {
		body := fmt.Sprintf(`{"state": {"padding": %q}}`, strings.Repeat("a", padding))
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, sessionsURL, strings.NewReader(body)))
		return rec
	}

	// A body between the 10 MiB default and the configured 20 MiB limit. It is
	// valid JSON, so it must succeed once the configured limit is honored; at
	// the default it would be rejected.
	raiseRec := sendSessionBody(int(adkrest.DefaultMaxPayloadSize) + 4096)
	if raiseRec.Code != http.StatusOK {
		t.Fatalf("raised-limit request: got status %d, want %d (%s)", raiseRec.Code, http.StatusOK, raiseRec.Body.String())
	}

	// The configured limit is still enforced above it.
	overRec := sendSessionBody(int(configuredMax) + 4096)
	if overRec.Code != http.StatusBadRequest {
		t.Fatalf("over-configured request: got status %d, want %d (%s)", overRec.Code, http.StatusBadRequest, overRec.Body.String())
	}
	if !strings.Contains(overRec.Body.String(), "http: request body too large") {
		t.Fatalf("over-configured request: got body %q, want mention of %q", overRec.Body.String(), "http: request body too large")
	}
}
