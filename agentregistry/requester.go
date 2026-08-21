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
	"io"
	"net/http"
	"net/url"
	"strings"
)

// requester performs registry API GET requests: base-URL selection,
// authentication, routing, and JSON decoding. It is an internal seam so an
// alternative implementation (for example an internal, non-REST transport) can
// back the same [Client]; the default implementation ([restRequester]) talks to
// agentregistry.googleapis.com over REST.
type requester interface {
	// Get issues a GET for resourcePath — treated as an absolute resource name
	// when it begins with "projects/", otherwise resolved relative to the
	// configured parent (projects/<project>/locations/<location>) — applies
	// params as the query string, and decodes the JSON response body into v.
	// A non-2xx response is returned as an [*APIError].
	Get(ctx context.Context, resourcePath string, params url.Values, v any) error
}

// APIError is returned when the registry API responds with a non-2xx status.
type APIError struct {
	// StatusCode is the HTTP status code of the response.
	StatusCode int
	// Body is the raw response body, useful for diagnosing the failure.
	Body string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("agentregistry: API request failed with status %d: %s", e.StatusCode, e.Body)
}

// restRequester is the default requester: it issues authenticated GETs to the
// third-party (public) Agent Registry REST endpoint.
type restRequester struct {
	httpClient  *http.Client
	baseURL     string
	basePath    string // projects/<project>/locations/<location>
	userProject string // value for the x-goog-user-project quota header
}

var _ requester = (*restRequester)(nil)

func (r *restRequester) Get(ctx context.Context, resourcePath string, params url.Values, v any) error {
	var fullURL string
	if strings.HasPrefix(resourcePath, "projects/") {
		fullURL = r.baseURL + "/" + resourcePath
	} else {
		fullURL = r.baseURL + "/" + r.basePath + "/" + resourcePath
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fullURL, nil)
	if err != nil {
		return fmt.Errorf("agentregistry: building request: %w", err)
	}
	if len(params) > 0 {
		req.URL.RawQuery = params.Encode()
	}
	req.Header.Set("Content-Type", "application/json")
	if r.userProject != "" {
		req.Header.Set("x-goog-user-project", r.userProject)
	}

	resp, err := r.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("agentregistry: GET %s: %w", resourcePath, err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("agentregistry: reading response body: %w", err)
	}

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return &APIError{StatusCode: resp.StatusCode, Body: string(body)}
	}

	if v != nil {
		if err := json.Unmarshal(body, v); err != nil {
			return fmt.Errorf("agentregistry: decoding response: %w", err)
		}
	}
	return nil
}
