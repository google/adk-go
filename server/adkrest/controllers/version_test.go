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

package controllers_test

import (
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/internal/version"
	"google.golang.org/adk/v2/server/adkrest/controllers"
)

// versionBody serves GET /version and returns the recorder.
func versionBody(t *testing.T) *httptest.ResponseRecorder {
	t.Helper()
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/version", nil)
	controllers.NewVersionAPIController().VersionHandler(rr, req)
	return rr
}

// TestVersionHandlerFieldNames pins the JSON field names. They are a
// cross-implementation contract shared with adk-python: the same web UI reads
// this body from an ADK Go and an ADK Python server, so a struct-tag typo
// breaks the UI against one of them only.
func TestVersionHandlerFieldNames(t *testing.T) {
	rr := versionBody(t)

	if got, want := rr.Code, http.StatusOK; got != want {
		t.Fatalf("GET /version status = %d, want %d", got, want)
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatalf("json.Unmarshal(%q) failed: %v", rr.Body.String(), err)
	}

	want := []string{"language", "language_version", "version"}
	if diff := cmp.Diff(want, slices.Sorted(maps.Keys(body))); diff != "" {
		t.Errorf("GET /version JSON field names mismatch (-want +got):\n%s", diff)
	}
}

// TestVersionHandlerValues checks the decoded body, not the raw string. The
// language field is what tells a client which ADK implementation it reached; a
// Go server reporting "python" is the failure this guards.
func TestVersionHandlerValues(t *testing.T) {
	rr := versionBody(t)

	if got, want := rr.Code, http.StatusOK; got != want {
		t.Fatalf("GET /version status = %d, want %d", got, want)
	}
	if got, want := rr.Header().Get("Content-Type"), "application/json"; !strings.Contains(got, want) {
		t.Errorf("GET /version Content-Type = %q, want it to contain %q", got, want)
	}

	var got controllers.VersionResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("json.Unmarshal(%q) failed: %v", rr.Body.String(), err)
	}

	if want := "go"; got.Language != want {
		t.Errorf("GET /version language = %q, want %q", got.Language, want)
	}
	if want := version.Version; got.Version != want {
		t.Errorf("GET /version version = %q, want %q", got.Version, want)
	}
	if got.LanguageVersion == "" {
		t.Error("GET /version language_version is empty, want the running toolchain version")
	}
	if want := runtime.Version(); got.LanguageVersion != want {
		t.Errorf("GET /version language_version = %q, want the running toolchain %q", got.LanguageVersion, want)
	}
}
