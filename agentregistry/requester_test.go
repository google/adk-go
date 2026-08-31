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

package agentregistry

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

// newTestRequester returns a restRequester pointing at srv with a fixed parent
// path and quota project.
func newTestRequester(srv *httptest.Server) *restRequester {
	return &restRequester{
		httpClient:  srv.Client(),
		baseURL:     srv.URL,
		basePath:    "projects/p/locations/l",
		userProject: "p",
	}
}

func TestRestRequester_Get_RelativePathAndHeaders(t *testing.T) {
	var gotPath, gotQuery, gotContentType, gotUserProject string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotContentType = r.Header.Get("Content-Type")
		gotUserProject = r.Header.Get("x-goog-user-project")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"agents":[{"displayName":"Foo"}],"nextPageToken":"next"}`))
	}))
	defer srv.Close()

	var page ListAgentsResponse
	params := url.Values{}
	params.Set("pageSize", "5")
	if err := newTestRequester(srv).Get(context.Background(), "agents", params, &page); err != nil {
		t.Fatalf("Get() error = %v", err)
	}

	if want := "/projects/p/locations/l/agents"; gotPath != want {
		t.Errorf("request path = %q, want %q", gotPath, want)
	}
	if want := "pageSize=5"; gotQuery != want {
		t.Errorf("request query = %q, want %q", gotQuery, want)
	}
	if gotContentType != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", gotContentType)
	}
	if gotUserProject != "p" {
		t.Errorf("x-goog-user-project = %q, want p", gotUserProject)
	}
	if len(page.Agents) != 1 || page.Agents[0].DisplayName != "Foo" || page.NextPageToken != "next" {
		t.Errorf("decoded page = %+v, want one agent Foo with nextPageToken", page)
	}
}

func TestRestRequester_Get_AbsoluteResourcePath(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_, _ = w.Write([]byte(`{"displayName":"Bar"}`))
	}))
	defer srv.Close()

	var got MCPServer
	name := "projects/p/locations/l/mcpServers/bar"
	if err := newTestRequester(srv).Get(context.Background(), name, nil, &got); err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if want := "/" + name; gotPath != want {
		t.Errorf("request path = %q, want %q (absolute resource name, not prefixed with parent)", gotPath, want)
	}
	if got.DisplayName != "Bar" {
		t.Errorf("decoded DisplayName = %q, want Bar", got.DisplayName)
	}
}

func TestRestRequester_Get_APIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"not found"}`))
	}))
	defer srv.Close()

	err := newTestRequester(srv).Get(context.Background(), "agents/missing", nil, &Agent{})
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("Get() error = %v, want *APIError", err)
	}
	if apiErr.StatusCode != http.StatusNotFound {
		t.Errorf("APIError.StatusCode = %d, want %d", apiErr.StatusCode, http.StatusNotFound)
	}
	if apiErr.Body != `{"error":"not found"}` {
		t.Errorf("APIError.Body = %q, want the raw response body", apiErr.Body)
	}
}

func TestRestRequester_Get_DecodeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{not valid json`))
	}))
	defer srv.Close()

	err := newTestRequester(srv).Get(context.Background(), "agents", nil, &ListAgentsResponse{})
	if err == nil {
		t.Fatal("Get() error = nil, want a decode error")
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		t.Errorf("Get() returned *APIError %v, want a decode error", apiErr)
	}
}

func TestRestRequester_Get_NilOutSkipsDecode(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`not json but ignored`))
	}))
	defer srv.Close()

	if err := newTestRequester(srv).Get(context.Background(), "agents", nil, nil); err != nil {
		t.Errorf("Get() with nil out error = %v, want nil", err)
	}
}

func TestRestRequester_Get_NoUserProjectHeaderWhenEmpty(t *testing.T) {
	var hadHeader bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, hadHeader = r.Header["X-Goog-User-Project"]
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	rt := &restRequester{httpClient: srv.Client(), baseURL: srv.URL, basePath: "projects/p/locations/l"}
	if err := rt.Get(context.Background(), "agents", nil, nil); err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if hadHeader {
		t.Error("x-goog-user-project header set, want it omitted when userProject is empty")
	}
}
