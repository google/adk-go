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
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/oauth2/google"
	"google.golang.org/api/option"
	htransport "google.golang.org/api/transport"
)

// cloudPlatformScope is the OAuth scope used for Application Default Credentials.
const cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

// Base URLs for the Agent Registry API (parity with adk-python).
const (
	baseURLProd = "https://agentregistry.googleapis.com/v1"
	baseURLMTLS = "https://agentregistry.mtls.googleapis.com/v1"
)

// MTLSMode controls selection of the mutual-TLS (mTLS) API endpoint.
type MTLSMode int

const (
	// MTLSAuto selects the mTLS endpoint only when client-certificate use is
	// opted in via GOOGLE_API_USE_CLIENT_CERTIFICATE and a default client
	// certificate source is actually present on the machine (parity with
	// adk-python). It is the zero value and also honors the
	// GOOGLE_API_USE_MTLS_ENDPOINT environment variable.
	MTLSAuto MTLSMode = iota
	// MTLSAlways always selects the mTLS endpoint.
	MTLSAlways
	// MTLSNever never selects the mTLS endpoint.
	MTLSNever
)

// Config configures a [Client].
type Config struct {
	// ProjectID is the Google Cloud project ID. Required.
	ProjectID string
	// Location is the Google Cloud location (region), e.g. "us-central1".
	// Required.
	Location string
	// HTTPClient is the HTTP client used for registry API calls. If nil, a
	// client authenticated with Application Default Credentials is created.
	HTTPClient *http.Client
	// MTLS controls mTLS endpoint selection. The zero value (MTLSAuto) honors
	// the GOOGLE_API_USE_MTLS_ENDPOINT environment variable.
	MTLS MTLSMode
}

// Client is a client for the Google Cloud Agent Registry.
type Client struct {
	transport Transport
	// httpClient is the client used for registry calls; the factory helpers
	// reuse it for egress to *.googleapis.com endpoints (parity with adk-python).
	httpClient *http.Client
}

// New creates a [Client]. By default it authenticates to
// agentregistry.googleapis.com using Application Default Credentials; provide
// Config.HTTPClient to supply a custom (e.g. pre-authenticated) client.
func New(ctx context.Context, cfg Config) (*Client, error) {
	if cfg.ProjectID == "" || cfg.Location == "" {
		return nil, fmt.Errorf("agentregistry: ProjectID and Location must be set")
	}

	// quotaProject is sent in the x-goog-user-project header. It prefers the
	// quota project configured in Application Default Credentials (parity with
	// adk-python's credentials.quota_project_id) and falls back to ProjectID.
	quotaProject := cfg.ProjectID

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		creds, err := google.FindDefaultCredentials(ctx, cloudPlatformScope)
		if err != nil {
			return nil, fmt.Errorf("agentregistry: loading Application Default Credentials: %w", err)
		}
		if qp := quotaProjectID(creds); qp != "" {
			quotaProject = qp
		}
		c, _, err := htransport.NewHTTPClient(ctx, option.WithCredentials(creds))
		if err != nil {
			return nil, fmt.Errorf("agentregistry: creating authenticated HTTP client: %w", err)
		}
		httpClient = c
	}

	mode := cfg.MTLS
	if mode == MTLSAuto {
		mode = mtlsModeFromEnv()
	}

	return &Client{
		transport: &restTransport{
			httpClient:  httpClient,
			baseURL:     selectBaseURL(mode, useClientCertFromEnv() && hasDefaultClientCertSource()),
			basePath:    fmt.Sprintf("projects/%s/locations/%s", cfg.ProjectID, cfg.Location),
			userProject: quotaProject,
		},
		httpClient: httpClient,
	}, nil
}

// get performs an authenticated GET against the registry, decoding the JSON
// response into v. It is the low-level primitive the discovery methods build on.
func (c *Client) get(ctx context.Context, resourcePath string, params url.Values, v any) error {
	return c.transport.Get(ctx, resourcePath, params, v)
}

// selectBaseURL returns the API base URL for the given mTLS mode and whether a
// client certificate is in use.
func selectBaseURL(mode MTLSMode, useClientCert bool) string {
	switch mode {
	case MTLSAlways:
		return baseURLMTLS
	case MTLSNever:
		return baseURLProd
	default: // MTLSAuto
		if useClientCert {
			return baseURLMTLS
		}
		return baseURLProd
	}
}

// mtlsModeFromEnv reads the mTLS endpoint mode from GOOGLE_API_USE_MTLS_ENDPOINT
// (auto|always|never), defaulting to MTLSAuto.
func mtlsModeFromEnv() MTLSMode {
	switch strings.ToLower(os.Getenv("GOOGLE_API_USE_MTLS_ENDPOINT")) {
	case "always":
		return MTLSAlways
	case "never":
		return MTLSNever
	default:
		return MTLSAuto
	}
}

// useClientCertFromEnv reports whether GOOGLE_API_USE_CLIENT_CERTIFICATE=="true".
func useClientCertFromEnv() bool {
	return strings.ToLower(os.Getenv("GOOGLE_API_USE_CLIENT_CERTIFICATE")) == "true"
}

// hasDefaultClientCertSource reports whether a default client certificate
// source is configured on the machine. It mirrors adk-python / google-auth's
// has_default_client_cert_source: a Secure Connect context-aware metadata file,
// the default gcloud certificate config, or a certificate config referenced by
// GOOGLE_API_CERTIFICATE_CONFIG.
func hasDefaultClientCertSource() bool {
	if home, err := os.UserHomeDir(); err == nil {
		if fileExists(filepath.Join(home, ".secureConnect", "context_aware_metadata.json")) {
			return true
		}
		if fileExists(filepath.Join(home, ".config", "gcloud", "certificate_config.json")) {
			return true
		}
	}
	if p := os.Getenv("GOOGLE_API_CERTIFICATE_CONFIG"); p != "" && fileExists(p) {
		return true
	}
	return false
}

// fileExists reports whether path names an existing file or directory.
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// quotaProjectID returns the quota project associated with creds, using the same
// precedence as Google API clients: the GOOGLE_CLOUD_QUOTA_PROJECT environment
// variable, then the "quota_project_id" field of the credentials JSON. It
// returns "" when none is configured (e.g. credentials sourced from the GCE
// metadata server, which carry no JSON).
func quotaProjectID(creds *google.Credentials) string {
	if q := os.Getenv("GOOGLE_CLOUD_QUOTA_PROJECT"); q != "" {
		return q
	}
	if creds == nil || len(creds.JSON) == 0 {
		return ""
	}
	var parsed struct {
		QuotaProjectID string `json:"quota_project_id"`
	}
	if err := json.Unmarshal(creds.JSON, &parsed); err != nil {
		return ""
	}
	return parsed.QuotaProjectID
}
