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
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/mux"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/server/adkrest/controllers"
	"google.golang.org/adk/v2/server/authn"
)

const (
	testArtifactApp     = "test-app"
	testArtifactUser    = "test-user"
	testArtifactSession = "test-session"
	testArtifactName    = "report.txt"
)

// notFoundArtifactService reports every Delete as a missing artifact.
//
// The in-memory service treats deleting a non-existent artifact as success (see
// [artifact.Service.Delete]), so it cannot exercise the 404 path.
type notFoundArtifactService struct {
	artifact.Service
}

func (notFoundArtifactService) Delete(ctx context.Context, req *artifact.DeleteRequest) error {
	return fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
}

// artifactVars returns the mux route variables the artifact handlers read.
func artifactVars(name, version string) map[string]string {
	vars := map[string]string{
		"app_name":      testArtifactApp,
		"user_id":       testArtifactUser,
		"session_id":    testArtifactSession,
		"artifact_name": name,
	}
	if version != "" {
		vars["version"] = version
	}
	return vars
}

// artifactServiceWithFile returns an in-memory service holding two versions
// (0 = "v0 artifact", 1 = "v1 artifact") so tests can distinguish version 0
// from the latest version.
func artifactServiceWithFile(t *testing.T) artifact.Service {
	t.Helper()
	svc := artifact.InMemoryService()
	for _, text := range []string{"v0 artifact", "v1 artifact"} {
		if _, err := svc.Save(t.Context(), &artifact.SaveRequest{
			AppName:   testArtifactApp,
			UserID:    testArtifactUser,
			SessionID: testArtifactSession,
			FileName:  testArtifactName,
			Part:      genai.NewPartFromText(text),
		}); err != nil {
			t.Fatalf("save artifact: %v", err)
		}
	}
	return svc
}

func TestLoadArtifactHandler(t *testing.T) {
	tc := []struct {
		name         string
		artifactName string
		query        string
		wantStatus   int
		wantText     string
	}{
		{
			name:         "existing_artifact_latest",
			artifactName: testArtifactName,
			wantStatus:   http.StatusOK,
			wantText:     "v1 artifact",
		},
		{
			name:         "existing_artifact_version_0",
			artifactName: testArtifactName,
			query:        "?version=0",
			wantStatus:   http.StatusOK,
			wantText:     "v0 artifact",
		},
		{
			name:         "existing_artifact_version_1",
			artifactName: testArtifactName,
			query:        "?version=1",
			wantStatus:   http.StatusOK,
			wantText:     "v1 artifact",
		},
		{
			name:         "missing_artifact",
			artifactName: "no-such-file.txt",
			wantStatus:   http.StatusNotFound,
		},
		{
			name:         "missing_version_of_existing_artifact",
			artifactName: testArtifactName,
			query:        "?version=42",
			wantStatus:   http.StatusNotFound,
		},
		{
			name:         "int64_version_above_int32",
			artifactName: testArtifactName,
			query:        "?version=2147483648",
			wantStatus:   http.StatusNotFound,
		},
		{
			name:         "non_integer_version",
			artifactName: testArtifactName,
			query:        "?version=latest",
			wantStatus:   http.StatusBadRequest,
		},
		{
			name:         "overflow_version",
			artifactName: testArtifactName,
			query:        "?version=99999999999999999999",
			wantStatus:   http.StatusBadRequest,
		},
		{
			name:         "empty_artifact_name",
			artifactName: "",
			wantStatus:   http.StatusBadRequest,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			apiController := controllers.NewArtifactsAPIController(artifactServiceWithFile(t))
			req := httptest.NewRequest(http.MethodGet, "/artifacts/"+tt.artifactName+tt.query, nil)
			req = req.WithContext(authn.WithCaller(t.Context(), &authn.Caller{UserID: testArtifactUser}))
			req = mux.SetURLVars(req, artifactVars(tt.artifactName, ""))

			rr := httptest.NewRecorder()

			apiController.LoadArtifactHandler(rr, req)

			if got := rr.Code; got != tt.wantStatus {
				t.Fatalf("LoadArtifactHandler status = %d, want %d (body %q)", got, tt.wantStatus, rr.Body.String())
			}
			if tt.wantText != "" {
				var part genai.Part
				if err := json.Unmarshal(rr.Body.Bytes(), &part); err != nil {
					t.Fatalf("decode response part: %v", err)
				}
				if part.Text != tt.wantText {
					t.Errorf("LoadArtifactHandler text = %q, want %q", part.Text, tt.wantText)
				}
			}
		})
	}
}

func TestLoadArtifactVersionHandler(t *testing.T) {
	tc := []struct {
		name         string
		artifactName string
		version      string
		wantStatus   int
		wantText     string
	}{
		{
			name:         "existing_version_0",
			artifactName: testArtifactName,
			version:      "0",
			wantStatus:   http.StatusOK,
			wantText:     "v0 artifact",
		},
		{
			name:         "existing_version_1",
			artifactName: testArtifactName,
			version:      "1",
			wantStatus:   http.StatusOK,
			wantText:     "v1 artifact",
		},
		{
			name:         "missing_version",
			artifactName: testArtifactName,
			version:      "99",
			wantStatus:   http.StatusNotFound,
		},
		{
			name:         "int64_version_above_int32",
			artifactName: testArtifactName,
			version:      "2147483648",
			wantStatus:   http.StatusNotFound,
		},
		{
			name:         "missing_artifact",
			artifactName: "no-such-file.txt",
			version:      "0",
			wantStatus:   http.StatusNotFound,
		},
		{
			name:         "non_integer_version",
			artifactName: testArtifactName,
			version:      "latest",
			wantStatus:   http.StatusBadRequest,
		},
		{
			name:         "overflow_version",
			artifactName: testArtifactName,
			version:      "99999999999999999999",
			wantStatus:   http.StatusBadRequest,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			apiController := controllers.NewArtifactsAPIController(artifactServiceWithFile(t))
			req := httptest.NewRequest(http.MethodGet, "/artifacts/"+tt.artifactName+"/versions/"+tt.version, nil)
			req = req.WithContext(authn.WithCaller(t.Context(), &authn.Caller{UserID: testArtifactUser}))
			req = mux.SetURLVars(req, artifactVars(tt.artifactName, tt.version))
			rr := httptest.NewRecorder()

			apiController.LoadArtifactVersionHandler(rr, req)

			if got := rr.Code; got != tt.wantStatus {
				t.Fatalf("LoadArtifactVersionHandler status = %d, want %d (body %q)", got, tt.wantStatus, rr.Body.String())
			}
			if tt.wantText != "" {
				var part genai.Part
				if err := json.Unmarshal(rr.Body.Bytes(), &part); err != nil {
					t.Fatalf("decode response part: %v", err)
				}
				if part.Text != tt.wantText {
					t.Errorf("LoadArtifactVersionHandler text = %q, want %q", part.Text, tt.wantText)
				}
			}
		})
	}
}

func TestDeleteArtifactHandler(t *testing.T) {
	tc := []struct {
		name       string
		service    artifact.Service
		wantStatus int
	}{
		{
			name:       "existing_artifact",
			service:    artifactServiceWithFile(t),
			wantStatus: http.StatusOK,
		},
		{
			// The in-memory service reports a missing artifact as success, so a stub
			// is needed to reach the not-found branch.
			name:       "missing_artifact",
			service:    notFoundArtifactService{},
			wantStatus: http.StatusNotFound,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			apiController := controllers.NewArtifactsAPIController(tt.service)
			req := httptest.NewRequest(http.MethodDelete, "/artifacts/"+testArtifactName, nil)
			req = req.WithContext(authn.WithCaller(t.Context(), &authn.Caller{UserID: testArtifactUser}))
			req = mux.SetURLVars(req, artifactVars(testArtifactName, ""))
			rr := httptest.NewRecorder()

			apiController.DeleteArtifactHandler(rr, req)

			if got := rr.Code; got != tt.wantStatus {
				t.Errorf("DeleteArtifactHandler status = %d, want %d (body %q)", got, tt.wantStatus, rr.Body.String())
			}
		})
	}
}

func TestListArtifactVersionsHandlerWithoutAuthorizer(t *testing.T) {
	controller := controllers.NewArtifactsAPIController(artifact.InMemoryService())
	controller.WithAuthorizer(nil)
	req := httptest.NewRequestWithContext(t.Context(), http.MethodGet, "/artifacts/report.txt/versions", nil)
	req = mux.SetURLVars(req, artifactVars(testArtifactName, ""))
	rr := httptest.NewRecorder()
	controller.ListArtifactVersionsHandler(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("GET versions status = %d, want %d", rr.Code, http.StatusOK)
	}
}

// TestArtifactHandlersNilService covers the case where the server was built
// without an artifact service. Dereferencing the nil service panics, which drops
// the connection with no HTTP response at all, so each handler must answer 503.
func TestArtifactHandlersNilService(t *testing.T) {
	apiController := controllers.NewArtifactsAPIController(nil)

	tc := []struct {
		name    string
		method  string
		vars    map[string]string
		handler func(http.ResponseWriter, *http.Request)
	}{
		{
			name:    "ListArtifactsHandler",
			method:  http.MethodGet,
			vars:    artifactVars("", ""),
			handler: apiController.ListArtifactsHandler,
		},
		{
			name:    "LoadArtifactHandler",
			method:  http.MethodGet,
			vars:    artifactVars(testArtifactName, ""),
			handler: apiController.LoadArtifactHandler,
		},
		{
			name:    "LoadArtifactVersionHandler",
			method:  http.MethodGet,
			vars:    artifactVars(testArtifactName, "1"),
			handler: apiController.LoadArtifactVersionHandler,
		},
		{
			name:    "DeleteArtifactHandler",
			method:  http.MethodDelete,
			vars:    artifactVars(testArtifactName, ""),
			handler: apiController.DeleteArtifactHandler,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/artifacts/"+testArtifactName, nil)
			req = mux.SetURLVars(req, tt.vars)
			rr := httptest.NewRecorder()

			// Recover so a nil-service panic is a named failure rather than a dead
			// test binary that reports nothing.
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("%s panicked on a nil artifact service: %v", tt.name, r)
					}
				}()
				tt.handler(rr, req)
			}()

			if got := rr.Code; got != http.StatusServiceUnavailable {
				t.Errorf("%s status = %d, want %d (body %q)", tt.name, got, http.StatusServiceUnavailable, rr.Body.String())
			}
			if body := rr.Body.String(); body == "" {
				t.Errorf("%s wrote an empty body; want an explanatory message", tt.name)
			}
		})
	}
}
