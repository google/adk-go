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

package gcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	"google.golang.org/adk/v2/auth"
)

const (
	cloudPlatformScope      = "https://www.googleapis.com/auth/cloud-platform"
	defaultAgentIdentityURL = "https://agentidentitycredentials.googleapis.com"
	defaultConnectorURL     = "https://iamconnectorcredentials.googleapis.com"

	defaultPollTimeout    = 10 * time.Second
	defaultInitialBackoff = 500 * time.Millisecond
	maxBackoff            = 8 * time.Second
)

// connectorResourceRE matches an IAM Connector resource name; anything else is
// routed to the Agent Identity service (same split as adk-python).
var connectorResourceRE = regexp.MustCompile(`^projects/[^/]+/locations/[^/]+/connectors/[^/]+$`)

// Sentinel errors from [Client.RetrieveCredential]; callers test with errors.Is.
var (
	// ErrConsentRejected means the end user rejected the consent request.
	ErrConsentRejected = errors.New("gcp: user consent rejected")
	// ErrPollTimeout means polling exceeded the poll timeout while the credential
	// was still pending.
	ErrPollTimeout = errors.New("gcp: timed out waiting for credentials")
)

// Client retrieves end-user credentials from the Agent Identity / IAM Connector
// credential services and maps them to [auth.Credential].
type Client struct {
	httpClient       *http.Client
	agentIdentityURL string
	connectorURL     string
	pollTimeout      time.Duration
	initialBackoff   time.Duration
}

// Option configures a [Client].
type Option func(*Client)

// WithHTTPClient sets the HTTP client used to call the credential services.
// When unset, [NewClient] builds one from Application Default Credentials.
func WithHTTPClient(h *http.Client) Option { return func(c *Client) { c.httpClient = h } }

// WithAgentIdentityEndpoint overrides the Agent Identity base URL (scheme+host).
func WithAgentIdentityEndpoint(url string) Option {
	return func(c *Client) { c.agentIdentityURL = url }
}

// WithConnectorEndpoint overrides the IAM Connector base URL (scheme+host).
func WithConnectorEndpoint(url string) Option {
	return func(c *Client) { c.connectorURL = url }
}

// WithPollTimeout bounds the total time spent polling a pending retrieval.
func WithPollTimeout(d time.Duration) Option { return func(c *Client) { c.pollTimeout = d } }

// NewClient builds a Client. Unless [WithHTTPClient] is supplied, it discovers
// Application Default Credentials (cloud-platform scope) to authenticate calls
// to the credential services.
func NewClient(ctx context.Context, opts ...Option) (*Client, error) {
	c := &Client{
		agentIdentityURL: defaultAgentIdentityURL,
		connectorURL:     defaultConnectorURL,
		pollTimeout:      defaultPollTimeout,
		initialBackoff:   defaultInitialBackoff,
	}
	for _, opt := range opts {
		opt(c)
	}
	if c.httpClient == nil {
		creds, err := google.FindDefaultCredentials(ctx, cloudPlatformScope)
		if err != nil {
			return nil, fmt.Errorf("gcp: find default credentials: %w", err)
		}
		c.httpClient = oauth2.NewClient(ctx, creds.TokenSource)
	}
	return c, nil
}

// Request identifies the resource and acting user for a credential retrieval.
type Request struct {
	// Resource is a full resource name. A name matching
	// projects/*/locations/*/connectors/* is routed to the IAM Connector
	// service; anything else (e.g. .../authProviders/*) to Agent Identity.
	Resource string
	// UserID is the acting end user's identity. Required.
	UserID string
	// Scopes are the OAuth scopes requested for the credential.
	Scopes []string
	// ContinueURI is the developer-hosted URI used to finalize managed-OAuth
	// (3-legged) flows. Unused by non-interactive flows.
	ContinueURI string
}

// RetrieveCredential retrieves a credential for req, polling while the service
// reports a non-interactive pending state (up to the configured poll timeout).
// If interactive consent is required it returns an [auth.ConsentRequiredError].
func (c *Client) RetrieveCredential(ctx context.Context, req Request) (auth.Credential, error) {
	if req.Resource == "" {
		return nil, fmt.Errorf("gcp: RetrieveCredential requires a Resource")
	}
	if req.UserID == "" {
		return nil, fmt.Errorf("gcp: RetrieveCredential requires a UserID")
	}

	retrieve := c.retrieveAgentIdentity
	if connectorResourceRE.MatchString(req.Resource) {
		retrieve = c.retrieveConnector
	}

	deadline := time.Now().Add(c.pollTimeout)
	backoff := c.initialBackoff
	for {
		res, err := retrieve(ctx, req)
		if err != nil {
			return nil, err
		}
		switch res.status {
		case statusOK:
			return mapCredential(res.header, res.token)
		case statusConsentRequired:
			return nil, &auth.ConsentRequiredError{AuthURI: res.consentURI, Nonce: res.consentNonce}
		case statusRejected:
			return nil, fmt.Errorf("%w for %q", ErrConsentRejected, req.Resource)
		case statusPending:
			remaining := time.Until(deadline)
			if remaining <= 0 {
				return nil, fmt.Errorf("%w for %q", ErrPollTimeout, req.Resource)
			}
			wait := min(backoff, remaining)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(wait):
			}
			backoff = min(backoff*2, maxBackoff)
		}
	}
}

// retrieveStatus is the normalized outcome of a single retrieval call.
type retrieveStatus int

const (
	statusOK retrieveStatus = iota
	statusPending
	statusConsentRequired
	statusRejected
)

type retrieveResult struct {
	status       retrieveStatus
	token        string
	header       string
	consentURI   string
	consentNonce string
}

// mapCredential maps the service's {header, token} tuple to an [auth.Credential]:
// an "Authorization: Bearer" header becomes a bearer credential; any other header
// name becomes a header-based API key.
func mapCredential(header, token string) (auth.Credential, error) {
	if header == "" || token == "" {
		return nil, fmt.Errorf("gcp: credentials service returned an empty header or token")
	}
	name, hint, _ := strings.Cut(header, ":")
	name = strings.TrimSpace(name)
	if strings.EqualFold(name, "authorization") &&
		strings.HasPrefix(strings.ToLower(strings.TrimSpace(hint)), "bearer") {
		return auth.BearerCredential{Token: token}, nil
	}
	// Non-bearer header -> header-based API key. adk-python also mirrors the
	// token into X-Goog-Api-Key for custom headers (alongside the service's own
	// header), so match that behavior.
	key := auth.APIKeyCredential{Name: name, Value: token}
	return auth.WithHeaders(key, map[string]string{"X-Goog-Api-Key": token}), nil
}

// doPost sends body as JSON to url and decodes a JSON response into out.
func (c *Client) doPost(ctx context.Context, url string, body, out any) error {
	buf, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("gcp: marshal request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(buf))
	if err != nil {
		return fmt.Errorf("gcp: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("gcp: call credentials service: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("gcp: read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("gcp: credentials service returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("gcp: decode response: %w", err)
	}
	return nil
}
