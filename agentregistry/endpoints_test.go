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
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestListEndpoints(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_, _ = w.Write([]byte(`{"endpoints":[{"displayName":"Gemini"}]}`))
	}))
	defer srv.Close()

	page, err := newTestClient(srv).ListEndpoints(context.Background())
	if err != nil {
		t.Fatalf("ListEndpoints() error = %v", err)
	}
	if want := "/projects/p/locations/l/endpoints"; gotPath != want {
		t.Errorf("request path = %q, want %q", gotPath, want)
	}
	if len(page.Endpoints) != 1 || page.Endpoints[0].DisplayName != "Gemini" {
		t.Errorf("page = %+v, want one endpoint Gemini", page)
	}
}

func TestGetEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"displayName":"Gemini","endpointId":"gemini"}`))
	}))
	defer srv.Close()

	got, err := newTestClient(srv).GetEndpoint(context.Background(), "projects/p/locations/l/endpoints/gemini")
	if err != nil {
		t.Fatalf("GetEndpoint() error = %v", err)
	}
	if got.DisplayName != "Gemini" || got.EndpointID != "gemini" {
		t.Errorf("endpoint = %+v, want Gemini/gemini", got)
	}
}

func TestModelName(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"interfaces":[{"url":"projects/p/locations/l/publishers/google/models/gemini-2.0:predict"}]}`))
	}))
	defer srv.Close()

	got, err := newTestClient(srv).ModelName(context.Background(), "projects/p/locations/l/endpoints/gemini")
	if err != nil {
		t.Fatalf("ModelName() error = %v", err)
	}
	if want := "projects/p/locations/l/publishers/google/models/gemini-2.0"; got != want {
		t.Errorf("ModelName() = %q, want %q", got, want)
	}
}

func TestModelName_NoURI(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"displayName":"Gemini"}`)) // no interfaces
	}))
	defer srv.Close()

	if _, err := newTestClient(srv).ModelName(context.Background(), "projects/p/locations/l/endpoints/gemini"); err == nil {
		t.Error("ModelName() error = nil, want an error when no connection URI is present")
	}
}

func TestParseModelName(t *testing.T) {
	tests := []struct {
		name string
		uri  string
		want string
	}{
		{
			name: "resource name with method suffix",
			uri:  "projects/p/locations/l/publishers/google/models/gemini-2.0:predict",
			want: "projects/p/locations/l/publishers/google/models/gemini-2.0",
		},
		{
			name: "already a resource name",
			uri:  "projects/p/locations/l/publishers/google/models/gemini-2.0",
			want: "projects/p/locations/l/publishers/google/models/gemini-2.0",
		},
		{
			name: "embedded resource name in a URL",
			uri:  "https://host/v1/projects/p/locations/l/models/foo",
			want: "projects/p/locations/l/models/foo",
		},
		{
			name: "no resource name returns input",
			uri:  "gemini-2.0",
			want: "gemini-2.0",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := parseModelName(tc.uri); got != tc.want {
				t.Errorf("parseModelName(%q) = %q, want %q", tc.uri, got, tc.want)
			}
		})
	}
}
