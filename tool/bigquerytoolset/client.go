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

package bigquerytoolset

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
	bigqueryScope           = "https://www.googleapis.com/auth/bigquery"
	defaultEndpointTemplate = "https://bigquery.UNIVERSE_DOMAIN/bigquery/v2/"
	defaultMTLSEndpoint     = "https://bigquery.mtls.googleapis.com/bigquery/v2/"
	defaultUniverseDomain   = "googleapis.com"

	// userAgentPrefix matches adk-python's BQ_USER_AGENT, so BigQuery can
	// attribute traffic from both to the ADK BigQuery tools.
	userAgentPrefix = "adk-bigquery-tool"
)

// client sends BigQuery REST requests. It is safe for concurrent use.
type client struct {
	httpClient *http.Client
	endpoint   string
	userAgent  string
	tokenKey   string
}

func newClient(ctx context.Context, cfg Config) (*client, error) {
	opts := []option.ClientOption{
		option.WithScopes(bigqueryScope),
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
			return nil, fmt.Errorf("bigquerytoolset: creating the BigQuery HTTP client for Config.ExternalAccessTokenKey, whose ClientOptions must not provide credentials: %w", err)
		}
		return nil, fmt.Errorf("bigquerytoolset: creating the BigQuery HTTP client: %w", err)
	}
	if !strings.HasSuffix(endpoint, "/") {
		endpoint += "/"
	}
	if _, err := url.Parse(endpoint); err != nil {
		return nil, fmt.Errorf("bigquerytoolset: invalid BigQuery endpoint: %w", withoutURL(err))
	}

	userAgent := userAgentPrefix + " " + version.TrackingHeader()
	if cfg.ApplicationName != "" {
		userAgent += " " + cfg.ApplicationName
	}
	return &client{
		httpClient: httpClient,
		endpoint:   endpoint,
		userAgent:  userAgent,
		tokenKey:   cfg.ExternalAccessTokenKey,
	}, nil
}

// get sends a GET request for path, relative to the BigQuery API endpoint,
// on behalf of the named tool and decodes the JSON response into out.
func (c *client) get(ctx agent.Context, toolName, path string, query url.Values, out any) error {
	httpClient, err := c.httpClientFor(ctx)
	if err != nil {
		return err
	}

	u := c.endpoint + path
	if len(query) > 0 {
		u += "?" + query.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return fmt.Errorf("building the BigQuery request: %w", withoutURL(err))
	}
	req.Header.Set("User-Agent", c.userAgent+" "+toolName)

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("sending the BigQuery request: %w", withoutURL(err))
	}
	defer func() { _ = resp.Body.Close() }()
	if err := googleapi.CheckResponse(resp); err != nil {
		return err
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("decoding the BigQuery response: %w", err)
	}
	return nil
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

// pathSegment escapes a model-supplied id for use as one segment of a
// request path. It rejects an empty id, which BigQuery would read as a
// different resource, and "." and "..", which escaping leaves as they are and
// a server may resolve to a parent path.
func pathSegment(arg, id string) (string, error) {
	switch id {
	case "":
		return "", fmt.Errorf("%s must not be empty", arg)
	case ".", "..":
		return "", fmt.Errorf("%s is not a valid BigQuery id", arg)
	}
	return url.PathEscape(id), nil
}

// withoutURL drops the URL a *url.Error carries, whose path and query hold
// the tool's arguments and page tokens.
func withoutURL(err error) error {
	if urlErr, ok := errors.AsType[*url.Error](err); ok {
		return urlErr.Err
	}
	return err
}

// errorResult is what a tool returns to the model when its BigQuery call
// fails, in adk-python's format.
func errorResult(err error) map[string]any {
	return map[string]any{
		"status":        "ERROR",
		"error_details": err.Error(),
	}
}
