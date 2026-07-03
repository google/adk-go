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
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/oauth2/google"
)

func TestNew_Validation(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{name: "missing project", cfg: Config{Location: "us-central1", HTTPClient: &http.Client{}}},
		{name: "missing location", cfg: Config{ProjectID: "p", HTTPClient: &http.Client{}}},
		{name: "missing both", cfg: Config{HTTPClient: &http.Client{}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := New(context.Background(), tc.cfg); err == nil {
				t.Errorf("New(%+v) error = nil, want an error", tc.cfg)
			}
		})
	}
}

func TestNew_WiresRestTransport(t *testing.T) {
	t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", "never")
	t.Setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")

	c, err := New(context.Background(), Config{
		ProjectID:  "my-project",
		Location:   "us-central1",
		HTTPClient: &http.Client{},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	rt, ok := c.transport.(*restTransport)
	if !ok {
		t.Fatalf("client transport type = %T, want *restTransport", c.transport)
	}
	if rt.baseURL != baseURLProd {
		t.Errorf("baseURL = %q, want %q", rt.baseURL, baseURLProd)
	}
	if want := "projects/my-project/locations/us-central1"; rt.basePath != want {
		t.Errorf("basePath = %q, want %q", rt.basePath, want)
	}
	if rt.userProject != "my-project" {
		t.Errorf("userProject = %q, want my-project", rt.userProject)
	}
}

func TestSelectBaseURL(t *testing.T) {
	tests := []struct {
		name          string
		mode          MTLSMode
		useClientCert bool
		want          string
	}{
		{name: "always", mode: MTLSAlways, want: baseURLMTLS},
		{name: "always ignores cert", mode: MTLSAlways, useClientCert: false, want: baseURLMTLS},
		{name: "never", mode: MTLSNever, useClientCert: true, want: baseURLProd},
		{name: "auto without cert", mode: MTLSAuto, useClientCert: false, want: baseURLProd},
		{name: "auto with cert", mode: MTLSAuto, useClientCert: true, want: baseURLMTLS},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := selectBaseURL(tc.mode, tc.useClientCert); got != tc.want {
				t.Errorf("selectBaseURL(%v, %t) = %q, want %q", tc.mode, tc.useClientCert, got, tc.want)
			}
		})
	}
}

func TestMTLSModeFromEnv(t *testing.T) {
	tests := []struct {
		env  string
		want MTLSMode
	}{
		{env: "always", want: MTLSAlways},
		{env: "ALWAYS", want: MTLSAlways},
		{env: "never", want: MTLSNever},
		{env: "auto", want: MTLSAuto},
		{env: "", want: MTLSAuto},
		{env: "garbage", want: MTLSAuto},
	}
	for _, tc := range tests {
		t.Run(tc.env, func(t *testing.T) {
			t.Setenv("GOOGLE_API_USE_MTLS_ENDPOINT", tc.env)
			if got := mtlsModeFromEnv(); got != tc.want {
				t.Errorf("mtlsModeFromEnv() with %q = %v, want %v", tc.env, got, tc.want)
			}
		})
	}
}

func TestUseClientCertFromEnv(t *testing.T) {
	tests := []struct {
		env  string
		want bool
	}{
		{env: "true", want: true},
		{env: "TRUE", want: true},
		{env: "false", want: false},
		{env: "", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.env, func(t *testing.T) {
			t.Setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", tc.env)
			if got := useClientCertFromEnv(); got != tc.want {
				t.Errorf("useClientCertFromEnv() with %q = %t, want %t", tc.env, got, tc.want)
			}
		})
	}
}

func TestHasDefaultClientCertSource(t *testing.T) {
	writeFile := func(t *testing.T, path string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("{}"), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	t.Run("none present", func(t *testing.T) {
		t.Setenv("HOME", t.TempDir())
		t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", "")
		if hasDefaultClientCertSource() {
			t.Error("hasDefaultClientCertSource() = true, want false")
		}
	})

	t.Run("secure connect metadata", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("HOME", home)
		t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", "")
		writeFile(t, filepath.Join(home, ".secureConnect", "context_aware_metadata.json"))
		if !hasDefaultClientCertSource() {
			t.Error("hasDefaultClientCertSource() = false, want true")
		}
	})

	t.Run("default gcloud cert config", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("HOME", home)
		t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", "")
		writeFile(t, filepath.Join(home, ".config", "gcloud", "certificate_config.json"))
		if !hasDefaultClientCertSource() {
			t.Error("hasDefaultClientCertSource() = false, want true")
		}
	})

	t.Run("env cert config present", func(t *testing.T) {
		t.Setenv("HOME", t.TempDir())
		cfg := filepath.Join(t.TempDir(), "cert_config.json")
		writeFile(t, cfg)
		t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", cfg)
		if !hasDefaultClientCertSource() {
			t.Error("hasDefaultClientCertSource() = false, want true")
		}
	})

	t.Run("env cert config missing", func(t *testing.T) {
		t.Setenv("HOME", t.TempDir())
		t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", filepath.Join(t.TempDir(), "does-not-exist.json"))
		if hasDefaultClientCertSource() {
			t.Error("hasDefaultClientCertSource() = true, want false")
		}
	})
}

func TestQuotaProjectID(t *testing.T) {
	t.Run("env var takes precedence", func(t *testing.T) {
		t.Setenv("GOOGLE_CLOUD_QUOTA_PROJECT", "env-project")
		creds := &google.Credentials{JSON: []byte(`{"quota_project_id":"json-project"}`)}
		if got := quotaProjectID(creds); got != "env-project" {
			t.Errorf("quotaProjectID() = %q, want env-project", got)
		}
	})

	t.Run("from credentials JSON", func(t *testing.T) {
		t.Setenv("GOOGLE_CLOUD_QUOTA_PROJECT", "")
		creds := &google.Credentials{JSON: []byte(`{"quota_project_id":"json-project"}`)}
		if got := quotaProjectID(creds); got != "json-project" {
			t.Errorf("quotaProjectID() = %q, want json-project", got)
		}
	})

	t.Run("JSON without quota project", func(t *testing.T) {
		t.Setenv("GOOGLE_CLOUD_QUOTA_PROJECT", "")
		creds := &google.Credentials{JSON: []byte(`{"type":"service_account"}`)}
		if got := quotaProjectID(creds); got != "" {
			t.Errorf("quotaProjectID() = %q, want empty", got)
		}
	})

	t.Run("nil credentials", func(t *testing.T) {
		t.Setenv("GOOGLE_CLOUD_QUOTA_PROJECT", "")
		if got := quotaProjectID(nil); got != "" {
			t.Errorf("quotaProjectID(nil) = %q, want empty", got)
		}
	})
}

func TestListValues(t *testing.T) {
	got := listValues(WithFilter("type=A2A"), WithPageSize(25), WithPageToken("tok"))
	if got.Get("filter") != "type=A2A" {
		t.Errorf("filter = %q, want type=A2A", got.Get("filter"))
	}
	if got.Get("pageSize") != "25" {
		t.Errorf("pageSize = %q, want 25", got.Get("pageSize"))
	}
	if got.Get("pageToken") != "tok" {
		t.Errorf("pageToken = %q, want tok", got.Get("pageToken"))
	}

	// Zero/empty options should not set parameters.
	empty := listValues(WithFilter(""), WithPageSize(0), WithPageToken(""))
	if len(empty) != 0 {
		t.Errorf("listValues with empty options = %v, want no parameters", empty)
	}
}
