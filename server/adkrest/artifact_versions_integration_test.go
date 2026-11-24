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

package adkrest_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/server/authz"
)

func TestListArtifactVersions(t *testing.T) {
	const (
		app      = "app"
		user     = "alice"
		session  = "session"
		filename = "notes.txt"
		base     = "/apps/" + app + "/users/" + user + "/sessions/" + session + "/artifacts/"
	)
	svc := artifact.InMemoryService()
	for i, text := range []string{"first", "second", "third"} {
		resp, err := svc.Save(t.Context(), &artifact.SaveRequest{
			AppName: app, UserID: user, SessionID: session, FileName: filename,
			Part: genai.NewPartFromText(text),
		})
		if err != nil {
			t.Fatalf("Save() failed: %v", err)
		}
		if resp.Version != int64(i) {
			t.Fatalf("Save() version = %d, want %d", resp.Version, i)
		}
	}
	if err := svc.Delete(t.Context(), &artifact.DeleteRequest{
		AppName: app, UserID: user, SessionID: session, FileName: filename, Version: new(int64(1)),
	}); err != nil {
		t.Fatalf("Delete() failed: %v", err)
	}
	srv, err := adkrest.NewServer(adkrest.ServerConfig{
		ArtifactService: svc,
		Authenticator: authn.NewCustom(func(*http.Request) (*authn.Caller, error) {
			return &authn.Caller{UserID: user}, nil
		}),
		Authorizer: authz.NewStrict(),
	})
	if err != nil {
		t.Fatalf("NewServer() failed: %v", err)
	}

	for _, tc := range []struct {
		name     string
		path     string
		wantCode int
		versions []int64
	}{
		{"stored IDs with a gap", base + filename + "/versions", http.StatusOK, []int64{0, 2}},
		{"missing artifact", base + "missing.txt/versions", http.StatusOK, []int64{}},
		{"invalid filename", base + "a%5Cb/versions", http.StatusBadRequest, nil},
		{"another user", "/apps/app/users/bob/sessions/session/artifacts/notes.txt/versions", http.StatusForbidden, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rr := httptest.NewRecorder()
			srv.ServeHTTP(rr, httptest.NewRequestWithContext(t.Context(), http.MethodGet, tc.path, nil))
			if rr.Code != tc.wantCode {
				t.Fatalf("GET versions status = %d, want %d", rr.Code, tc.wantCode)
			}
			if tc.wantCode != http.StatusOK {
				return
			}
			var versions []int64
			if err := json.Unmarshal(rr.Body.Bytes(), &versions); err != nil {
				t.Fatal("GET versions response is not a JSON array of integers")
			}
			if versions == nil || !slices.Equal(versions, tc.versions) {
				t.Errorf("GET versions = %v (nil: %t), want %v", versions, versions == nil, tc.versions)
			}
		})
	}
	for _, tc := range []struct {
		version int64
		text    string
	}{
		{0, "first"},
		{2, "third"},
	} {
		path := base + filename + "/versions/" + strconv.FormatInt(tc.version, 10)
		rr := httptest.NewRecorder()
		srv.ServeHTTP(rr, httptest.NewRequestWithContext(t.Context(), http.MethodGet, path, nil))
		if rr.Code != http.StatusOK {
			t.Fatalf("GET version %d status = %d, want %d", tc.version, rr.Code, http.StatusOK)
		}
		var part genai.Part
		if err := json.Unmarshal(rr.Body.Bytes(), &part); err != nil || part.Text != tc.text {
			t.Errorf("GET version %d did not return the corresponding artifact", tc.version)
		}
	}
}

// artifactVersionsService is a stub for the version-list operation.
type artifactVersionsService struct {
	artifact.Service
	response *artifact.VersionsResponse
	err      error
}

func (s artifactVersionsService) Versions(context.Context, *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	return s.response, s.err
}

func TestListArtifactVersionsServiceResponses(t *testing.T) {
	const errorDetail = "synthetic backend error detail"
	for _, tc := range []struct {
		name     string
		service  artifact.Service
		wantCode int
	}{
		{"disabled", nil, http.StatusServiceUnavailable},
		{"backend error", artifactVersionsService{err: errors.New(errorDetail)}, http.StatusInternalServerError},
		{"successful empty result", artifactVersionsService{response: &artifact.VersionsResponse{}}, http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, err := adkrest.NewServer(adkrest.ServerConfig{ArtifactService: tc.service})
			if err != nil {
				t.Fatalf("NewServer() failed: %v", err)
			}
			rr := httptest.NewRecorder()
			const path = "/apps/app/users/alice/sessions/session/artifacts/notes.txt/versions"
			srv.ServeHTTP(rr, httptest.NewRequestWithContext(t.Context(), http.MethodGet, path, nil))
			if rr.Code != tc.wantCode {
				t.Fatalf("GET versions status = %d, want %d", rr.Code, tc.wantCode)
			}
			if tc.wantCode == http.StatusOK && strings.TrimSpace(rr.Body.String()) != "[]" {
				t.Error("empty version list did not encode as []")
			}
			if strings.Contains(rr.Body.String(), errorDetail) {
				t.Error("GET versions exposed the backend error detail")
			}
		})
	}
}

func TestListArtifactVersionsPreservesServiceOrder(t *testing.T) {
	serviceVersions := []int64{2, 0}
	svc := artifactVersionsService{response: &artifact.VersionsResponse{Versions: serviceVersions}}
	srv, err := adkrest.NewServer(adkrest.ServerConfig{ArtifactService: svc})
	if err != nil {
		t.Fatalf("NewServer() failed: %v", err)
	}

	const path = "/apps/app/users/alice/sessions/session/artifacts/notes.txt/versions"
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, httptest.NewRequestWithContext(t.Context(), http.MethodGet, path, nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("GET versions status = %d, want %d", rr.Code, http.StatusOK)
	}
	var got []int64
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatal("GET versions response is not a JSON array of integers")
	}
	if want := []int64{0, 2}; !slices.Equal(got, want) {
		t.Errorf("GET versions = %v, want %v", got, want)
	}
	if want := []int64{2, 0}; !slices.Equal(serviceVersions, want) {
		t.Errorf("service versions after GET = %v, want %v", serviceVersions, want)
	}
}
