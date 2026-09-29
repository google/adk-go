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

package routers_test

import (
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/server/adkrest/internal/routers"
)

type staticRouter routers.Routes

func (r staticRouter) Routes() routers.Routes { return routers.Routes(r) }

// TestExperimentalRouteWarnsOnce pins adk-python's behavior for an
// experimental feature: one warning on first use, however many requests follow
// and however concurrently, and none for a route that is not experimental.
func TestExperimentalRouteWarnsOnce(t *testing.T) {
	// The standard logger serializes its writes, so a plain builder is safe.
	var logs strings.Builder
	prevOut, prevFlags := log.Writer(), log.Flags()
	log.SetOutput(&logs)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(prevOut)
		log.SetFlags(prevFlags)
	})

	ok := func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }
	router := mux.NewRouter()
	routers.SetupSubRouters(router, nil, staticRouter{
		{Name: "Trial", Methods: []string{http.MethodGet}, Pattern: "/trial/{id}", HandlerFunc: ok, Experimental: true},
		{Name: "Stable", Methods: []string{http.MethodGet}, Pattern: "/stable/{id}", HandlerFunc: ok},
	})

	var wg sync.WaitGroup
	for range 20 {
		for _, path := range []string{"/trial/1", "/stable/1"} {
			wg.Go(func() {
				rr := httptest.NewRecorder()
				router.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, path, nil))
				if rr.Code != http.StatusOK {
					t.Errorf("GET %s: status = %d, want %d", path, rr.Code, http.StatusOK)
				}
			})
		}
	}
	wg.Wait()

	const want = "[EXPERIMENTAL] /trial/{id}: This feature is experimental and may change or be removed in future versions without notice. It may introduce breaking changes at any time.\n"
	if got := logs.String(); got != want {
		t.Errorf("log = %q, want only the experimental route's warning, once: %q", got, want)
	}
}
