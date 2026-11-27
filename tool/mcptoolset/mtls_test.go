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

package mcptoolset

import (
	"fmt"
	"net/http"
	"testing"

	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/internal/googlemtls/googlemtlstest"
)

// The client mcptoolset builds when the caller supplies no Transport presents
// the environment's client certificate to a Google API host, on its mTLS name.
func TestHTTPClientPresentsClientCertificateToGoogleAPIs(t *testing.T) {
	for _, tc := range []struct {
		name     string
		base     *http.Client
		provider auth.CredentialProvider
	}{
		{name: "no client"},
		{name: "client without Transport", base: &http.Client{}},
		{name: "no client, Auth", provider: auth.StaticToken("tok")},
		{name: "client without Transport, Auth", base: &http.Client{}, provider: auth.StaticToken("tok")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "")
			srv := googlemtlstest.NewServer(t)
			srv.Install(t)
			offline(t)
			c := httpClient(tc.base, tc.provider)
			resp, err := c.Get("https://mcp.googleapis.com/mcp")
			if err != nil {
				t.Fatalf("Get error = %v", err)
			}
			_ = resp.Body.Close()
			got := <-srv.Requests
			want := googlemtlstest.Request{Host: "mcp.mtls.googleapis.com", ServerName: "mcp.mtls.googleapis.com", PeerCerts: 1, Leaf: srv.Leaf()}
			if got != want {
				t.Errorf("server saw %+v, want %+v", got, want)
			}
		})
	}
}

// offline makes http.DefaultTransport fail every request for the rest of the
// test, so a request that misses the mTLS path fails here instead of going to
// the network. The mTLS transport then starts from the settings net/http
// creates http.DefaultTransport with.
func offline(t *testing.T) {
	orig := http.DefaultTransport
	http.DefaultTransport = offlineTransport{}
	t.Cleanup(func() { http.DefaultTransport = orig })
}

type offlineTransport struct{}

func (offlineTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return nil, fmt.Errorf("offline: request to %s reached http.DefaultTransport", req.URL.Host)
}

type recordingRoundTripper struct{ hosts []string }

func (r *recordingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	r.hosts = append(r.hosts, req.URL.Host)
	return &http.Response{StatusCode: http.StatusNoContent, Body: http.NoBody, Request: req}, nil
}

// A caller's Transport owns TLS, so it gets the request as addressed.
func TestHTTPClientKeepsCallerTransport(t *testing.T) {
	srv := googlemtlstest.NewServer(t)
	srv.Install(t)
	rec := &recordingRoundTripper{}
	c := httpClient(&http.Client{Transport: rec}, nil)
	resp, err := c.Get("https://mcp.googleapis.com/mcp")
	if err != nil {
		t.Fatalf("Get error = %v", err)
	}
	_ = resp.Body.Close()
	if len(rec.hosts) != 1 || rec.hosts[0] != "mcp.googleapis.com" {
		t.Errorf("caller transport got hosts %v, want [mcp.googleapis.com]", rec.hosts)
	}
}
