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
	"os"
	"strings"

	"golang.org/x/oauth2/google"
	"google.golang.org/api/option"
	"google.golang.org/api/option/internaloption"
	htransport "google.golang.org/api/transport"
)

// cloudPlatformScope is the OAuth scope used for Application Default Credentials.
const cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

// Base URLs for the Agent Registry API (parity with adk-python).
const (
	baseURLProd = "https://agentregistry.googleapis.com/v1"
	baseURLMTLS = "https://agentregistry.mtls.googleapis.com/v1"
)

// Config configures a [Client].
type Config struct {
	// ProjectID is the Google Cloud project ID. Required.
	ProjectID string
	// Location is the Google Cloud location (region), e.g. "us-central1".
	// Required.
	Location string
	// HTTPClient is used for registry API calls. If nil, an ADC-authenticated
	// client is created and the endpoint (incl. mTLS) is resolved from
	// GOOGLE_API_USE_MTLS_ENDPOINT / GOOGLE_API_USE_CLIENT_CERTIFICATE. A
	// supplied client manages its own mTLS.
	HTTPClient *http.Client
}

// Client is a client for the Google Cloud Agent Registry.
type Client struct {
	requester requester
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

	var creds *google.Credentials
	var baseURL string
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		c, err := google.FindDefaultCredentials(ctx, cloudPlatformScope)
		if err != nil {
			return nil, fmt.Errorf("agentregistry: loading Application Default Credentials: %w", err)
		}
		creds = c

		// Resolve the endpoint and client certificate from one source: the
		// transport honors GOOGLE_API_USE_MTLS_ENDPOINT / _CLIENT_CERTIFICATE and
		// returns the endpoint it dialed, so endpoint and mTLS can't disagree.
		client, endpoint, err := htransport.NewHTTPClient(ctx,
			option.WithCredentials(creds),
			internaloption.WithDefaultEndpoint(baseURLProd),
			internaloption.WithDefaultMTLSEndpoint(baseURLMTLS),
		)
		if err != nil {
			return nil, fmt.Errorf("agentregistry: creating authenticated HTTP client: %w", err)
		}
		httpClient = client
		baseURL = endpoint
	} else {
		baseURL = customClientBaseURL()
	}

	// x-goog-user-project quota header, adk-python precedence on the ADC path:
	// GOOGLE_CLOUD_QUOTA_PROJECT, then creds' quota_project_id, then ProjectID.
	// A caller-supplied client has no ADC creds, so only env var / ProjectID apply.
	quotaProject := cfg.ProjectID
	if qp := quotaProjectID(creds); qp != "" {
		quotaProject = qp
	}

	return &Client{
		requester: &restRequester{
			httpClient:  httpClient,
			baseURL:     baseURL,
			basePath:    fmt.Sprintf("projects/%s/locations/%s", cfg.ProjectID, cfg.Location),
			userProject: quotaProject,
		},
		httpClient: httpClient,
	}, nil
}

// customClientBaseURL picks the endpoint for a caller-supplied client (which
// carries no client cert): standard, unless GOOGLE_API_USE_MTLS_ENDPOINT=always.
func customClientBaseURL() string {
	if strings.EqualFold(os.Getenv("GOOGLE_API_USE_MTLS_ENDPOINT"), "always") {
		return baseURLMTLS
	}
	return baseURLProd
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
