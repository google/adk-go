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

package spannertoolset

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
	"google.golang.org/api/option/internaloption"
	htransport "google.golang.org/api/transport/http"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/internal/version"
	"google.golang.org/adk/v2/session"
)

const (
	defaultEndpointTemplate = "https://spanner.UNIVERSE_DOMAIN/v1/"
	defaultMTLSEndpoint     = "https://spanner.mtls.googleapis.com/v1/"
	defaultUniverseDomain   = "googleapis.com"

	// userAgentPrefix matches adk-python's USER_AGENT for the Spanner tools,
	// so Spanner can attribute traffic from both to the ADK Spanner tools.
	userAgentPrefix = "adk-spanner-tool"
)

// scopes are adk-python's SPANNER_DEFAULT_SCOPE.
var scopes = []string{
	"https://www.googleapis.com/auth/spanner.admin",
	"https://www.googleapis.com/auth/spanner.data",
}

// client sends Spanner REST requests. It is safe for concurrent use.
type client struct {
	httpClient *http.Client
	endpoint   string
	userAgent  string
	tokenKey   string
}

func newClient(ctx context.Context, cfg Config) (*client, error) {
	opts := []option.ClientOption{
		option.WithScopes(scopes...),
		internaloption.WithDefaultEndpointTemplate(defaultEndpointTemplate),
		internaloption.WithDefaultMTLSEndpoint(defaultMTLSEndpoint),
		internaloption.WithDefaultUniverseDomain(defaultUniverseDomain),
	}
	if cfg.ExternalAccessTokenKey != "" {
		// Each call adds the token from the session state, so the shared
		// client must not authenticate on its own.
		opts = append(opts, option.WithoutAuthentication())
	}
	opts = append(opts, cfg.ClientOptions...)

	// The client outlives the call to New, and the credentials it holds
	// refresh their token with this context.
	httpClient, endpoint, err := htransport.NewClient(context.WithoutCancel(ctx), opts...)
	if err != nil {
		if cfg.ExternalAccessTokenKey != "" {
			return nil, fmt.Errorf("spannertoolset: creating the Spanner HTTP client for Config.ExternalAccessTokenKey, whose ClientOptions must not provide credentials: %w", err)
		}
		return nil, fmt.Errorf("spannertoolset: creating the Spanner HTTP client: %w", err)
	}
	if !strings.HasSuffix(endpoint, "/") {
		endpoint += "/"
	}
	if _, err := url.Parse(endpoint); err != nil {
		return nil, fmt.Errorf("spannertoolset: invalid Spanner endpoint: %w", withoutURL(err))
	}
	return &client{
		httpClient: httpClient,
		endpoint:   endpoint,
		userAgent:  userAgentPrefix + " " + version.TrackingHeader(),
		tokenKey:   cfg.ExternalAccessTokenKey,
	}, nil
}

// call sends a request for path, relative to the Spanner API endpoint, with
// body, if not nil, as JSON, and decodes the JSON response into out, if not
// nil.
func (c *client) call(ctx context.Context, httpClient *http.Client, method, path string, body, out any) error {
	resp, err := c.send(ctx, httpClient, method, path, body)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if out == nil {
		return nil
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("decoding the Spanner response: %w", err)
	}
	return nil
}

// send sends a request for path, relative to the Spanner API endpoint, with
// body, if not nil, as JSON. It returns the response of a successful request,
// whose body the caller closes.
func (c *client) send(ctx context.Context, httpClient *http.Client, method, path string, body any) (*http.Response, error) {
	var reqBody io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("encoding the Spanner request: %w", err)
		}
		reqBody = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.endpoint+path, reqBody)
	if err != nil {
		return nil, fmt.Errorf("building the Spanner request: %w", withoutURL(err))
	}
	req.Header.Set("User-Agent", c.userAgent)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending the Spanner request: %w", withoutURL(err))
	}
	if err := googleapi.CheckResponse(resp); err != nil {
		_ = resp.Body.Close()
		return nil, err
	}
	return resp, nil
}

// httpClientFor returns the HTTP client for one call: the shared one, or, with
// an external access token, one that sends the token from the session state
// over the shared client's connections.
func (c *client) httpClientFor(ctx agent.Context) (*http.Client, error) {
	if c.tokenKey == "" {
		return c.httpClient, nil
	}
	token, err := accessToken(ctx, c.tokenKey)
	if err != nil {
		return nil, err
	}
	withToken := *c.httpClient
	withToken.Transport = &auth.Transport{
		Provider: auth.StaticToken(token),
		Base:     c.httpClient.Transport,
	}
	return &withToken, nil
}

func accessToken(ctx agent.Context, key string) (string, error) {
	missing := fmt.Errorf("no access token found in the session state under the key %q set as ExternalAccessTokenKey", key)
	v, err := ctx.State().Get(key)
	if errors.Is(err, session.ErrStateKeyNotExist) {
		return "", missing
	}
	if err != nil {
		return "", fmt.Errorf("reading the access token from the session state: %w", err)
	}
	token, ok := v.(string)
	if !ok || token == "" {
		return "", missing
	}
	return token, nil
}

// databasePath returns the resource path of a database, with each id escaped
// as one path segment.
func databasePath(projectID, instanceID, databaseID string) (string, error) {
	var segments []string
	for _, id := range []struct{ arg, value string }{
		{"project_id", projectID},
		{"instance_id", instanceID},
		{"database_id", databaseID},
	} {
		s, err := pathSegment(id.arg, id.value)
		if err != nil {
			return "", err
		}
		segments = append(segments, s)
	}
	return "projects/" + segments[0] + "/instances/" + segments[1] + "/databases/" + segments[2], nil
}

// pathSegment escapes a model-supplied id for use as one segment of a
// request path. It rejects an empty id, which Spanner would read as a
// different resource, and "." and "..", which escaping leaves as they are and
// a server may resolve to a parent path.
func pathSegment(arg, id string) (string, error) {
	switch id {
	case "":
		return "", fmt.Errorf("%s must not be empty", arg)
	case ".", "..":
		return "", fmt.Errorf("%s is not a valid Spanner id", arg)
	}
	return url.PathEscape(id), nil
}

// withoutURL drops the URL a *url.Error carries, whose path holds the tool's
// arguments.
func withoutURL(err error) error {
	if urlErr, ok := errors.AsType[*url.Error](err); ok {
		return urlErr.Err
	}
	return err
}
