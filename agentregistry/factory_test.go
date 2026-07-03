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
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestIsGoogleAPI(t *testing.T) {
	tests := []struct {
		url  string
		want bool
	}{
		{url: "https://googleapis.com/v1", want: true},
		{url: "https://agentregistry.googleapis.com/v1", want: true},
		{url: "https://agentregistry.mtls.googleapis.com/v1", want: true},
		{url: "https://evil.com/googleapis.com", want: false},
		{url: "https://notgoogleapis.com", want: false},
		{url: "https://example.com", want: false},
		{url: "://bad", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.url, func(t *testing.T) {
			if got := isGoogleAPI(tc.url); got != tc.want {
				t.Errorf("isGoogleAPI(%q) = %t, want %t", tc.url, got, tc.want)
			}
		})
	}
}

func TestEgressClient(t *testing.T) {
	registryClient := &http.Client{}
	override := &http.Client{}
	c := &Client{httpClient: registryClient}

	tests := []struct {
		name    string
		url     string
		ec      egressConfig
		autoADC bool
		want    *http.Client
	}{
		{
			name:    "explicit override wins",
			url:     "https://x.googleapis.com",
			ec:      egressConfig{httpClient: override},
			autoADC: true,
			want:    override,
		},
		{
			name:    "google api with autoADC reuses registry client",
			url:     "https://x.googleapis.com/mcp",
			autoADC: true,
			want:    registryClient,
		},
		{
			name:    "google api without autoADC uses default",
			url:     "https://x.googleapis.com/mcp",
			autoADC: false,
			want:    http.DefaultClient,
		},
		{
			name:    "non-google uses default",
			url:     "https://third-party.example/mcp",
			autoADC: true,
			want:    http.DefaultClient,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := c.egressClient(tc.url, tc.ec, tc.autoADC); got != tc.want {
				t.Errorf("egressClient() = %p, want %p", got, tc.want)
			}
		})
	}
}

func TestEgressClient_HeadersProduceDistinctClient(t *testing.T) {
	registryClient := &http.Client{}
	c := &Client{httpClient: registryClient}
	got := c.egressClient("https://x.googleapis.com", egressConfig{headers: map[string]string{"X": "y"}}, true)
	if got == registryClient {
		t.Error("egressClient() with headers returned the shared registry client; want a distinct clone")
	}
}

func TestClientWithHeaders_NoHeadersReturnsBase(t *testing.T) {
	base := &http.Client{}
	if got := clientWithHeaders(base, nil); got != base {
		t.Errorf("clientWithHeaders(base, nil) = %p, want base %p", got, base)
	}
}

func TestClientWithHeaders_AppliesAndDoesNotMutateBase(t *testing.T) {
	var gotHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Test")
	}))
	defer srv.Close()

	base := &http.Client{}
	wrapped := clientWithHeaders(base, map[string]string{"X-Test": "v"})
	if wrapped == base {
		t.Fatal("clientWithHeaders returned base; want a clone")
	}
	if base.Transport != nil {
		t.Error("base client Transport was mutated; want it left nil")
	}

	resp, err := wrapped.Get(srv.URL)
	if err != nil {
		t.Fatalf("wrapped.Get() error = %v", err)
	}
	_ = resp.Body.Close()
	if gotHeader != "v" {
		t.Errorf("server saw X-Test = %q, want v", gotHeader)
	}
}
