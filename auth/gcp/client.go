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
	"unicode/utf8"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	"google.golang.org/adk/v2/auth"
)

const (
	cloudPlatformScope      = "https://www.googleapis.com/auth/cloud-platform"
	defaultAgentIdentityURL = "https://agentidentitycredentials.googleapis.com"
	defaultConnectorURL     = "https://iamconnectorcredentials.googleapis.com"

	defaultPollTimeout = 10 * time.Second
	// The credentials service documents an exponential polling backoff
	// (0.5, 1, 2, 4, 8s); these constants track it.
	defaultInitialBackoff = 500 * time.Millisecond
	maxBackoff            = 8 * time.Second
)

// connectorResourceRE matches an IAM Connector resource name; anything else is
// routed to the Agent Identity service (same split as adk-python).
var connectorResourceRE = regexp.MustCompile(`^projects/[^/]+/locations/[^/]+/connectors/[^/]+$`)

// resourceNameRE bounds a resource name to the characters GCP resource names
// use, so it can't inject extra path segments, a query, or a fragment into the
// request URL it is interpolated into. A separate ".." check blocks path
// traversal (dots are allowed so domain-style ids still pass).
var resourceNameRE = regexp.MustCompile(`^[A-Za-z0-9._~/-]+$`)

// Sentinel errors from [Client.RetrieveCredential]; callers test with errors.Is.
var (
	// ErrConsentRejected means the end user rejected the consent request.
	ErrConsentRejected = errors.New("gcp: user consent rejected")
	// ErrPollTimeout means polling exceeded the poll timeout while the credential
	// was still pending.
	ErrPollTimeout = errors.New("gcp: timed out waiting for credentials")
)

// APIError is returned when a credential service responds with a non-2xx
// status. Callers match it with errors.As to tell a fatal status (say 403) from
// a transient one (503) without matching on the message.
type APIError struct {
	// StatusCode is the HTTP status code of the response.
	StatusCode int
	// Body is the response body, truncated, useful for diagnosing the failure.
	Body string
}

func (e *APIError) Error() string {
	// %q, not %s: the body is service-controlled and can carry control bytes
	// that would otherwise forge lines in an operator's log.
	return fmt.Sprintf("gcp: credentials service returned status %d: %q", e.StatusCode, e.Body)
}

// Client retrieves end-user credentials from the Agent Identity / IAM Connector
// credential services and maps them to [auth.Credential].
type Client struct {
	httpClient       *http.Client
	agentIdentityURL string
	connectorURL     string
	pollTimeout      time.Duration
	initialBackoff   time.Duration
}

// Config configures a [Client]. A nil *Config, or any zero-valued field, uses
// the corresponding default.
type Config struct {
	// HTTPClient calls the credential services. If nil, [NewClient] builds one
	// from Application Default Credentials (cloud-platform scope). If set, it is
	// used verbatim and ADC is not applied, so it must carry its own credentials
	// and should refuse redirects for the reason [NewClient] describes.
	HTTPClient *http.Client
	// AgentIdentityEndpoint overrides the Agent Identity base URL (scheme+host).
	// It is used as given, not parsed: an http:// value would send the ADC token
	// in the clear, so keep it https outside tests.
	AgentIdentityEndpoint string
	// ConnectorEndpoint overrides the IAM Connector base URL (scheme+host), with
	// the same caveat as AgentIdentityEndpoint.
	ConnectorEndpoint string
	// PollTimeout bounds the wall-clock time spent retrying a pending retrieval.
	// It caps the retry loop, not an individual request; bound a single stalled
	// request via ctx (or an HTTPClient with its own Timeout).
	//
	// To bound requests without giving up ADC, put an [http.Client] carrying a
	// Timeout in the context passed to [NewClient] under [oauth2.HTTPClient]:
	// its Timeout is carried through to the ADC-backed client.
	PollTimeout time.Duration
}

// NewClient builds a Client from cfg; a nil cfg (or any zero field) uses
// defaults. Unless cfg.HTTPClient is set, it discovers Application Default
// Credentials (cloud-platform scope) to authenticate calls to the services.
//
// ctx is used for credential discovery only, and its cancellation is not
// honored: the token source backing the returned client is detached from ctx,
// so a Client built inside a request-scoped context keeps refreshing its token
// after that request ends.
//
// The ADC-backed client refuses redirects. A credentials:retrieve call has no
// reason to redirect, and following one would re-sign the request and hand the
// cloud-platform token to the redirect target.
func NewClient(ctx context.Context, cfg *Config) (*Client, error) {
	if cfg == nil {
		cfg = &Config{}
	}
	c := &Client{
		httpClient:       cfg.HTTPClient,
		agentIdentityURL: defaultAgentIdentityURL,
		connectorURL:     defaultConnectorURL,
		pollTimeout:      defaultPollTimeout,
		initialBackoff:   defaultInitialBackoff,
	}
	if cfg.AgentIdentityEndpoint != "" {
		c.agentIdentityURL = strings.TrimRight(cfg.AgentIdentityEndpoint, "/")
	}
	if cfg.ConnectorEndpoint != "" {
		c.connectorURL = strings.TrimRight(cfg.ConnectorEndpoint, "/")
	}
	if cfg.PollTimeout > 0 {
		c.pollTimeout = cfg.PollTimeout
	}
	if c.httpClient == nil {
		// The token source captures this context and reuses it for every later
		// refresh, so it must outlive the call; discovery itself needs no
		// cancellation (its only network probe bounds itself).
		creds, err := google.FindDefaultCredentials(context.WithoutCancel(ctx), cloudPlatformScope)
		if err != nil {
			return nil, fmt.Errorf("gcp: find default credentials: %w", err)
		}
		hc := oauth2.NewClient(ctx, creds.TokenSource)
		// oauth2.Transport re-signs every hop, below the layer where net/http
		// strips credentials on a cross-host redirect, so a redirect would leak
		// the token to whatever host it names.
		hc.CheckRedirect = func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		}
		c.httpClient = hc
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

// Retrieval is the result of [Client.RetrieveCredential].
type Retrieval struct {
	Credential auth.Credential
	// ExpiresAt is the credential's expiry, or the zero time when the service
	// reports no lifetime.
	ExpiresAt time.Time
}

// RetrieveCredential retrieves a credential for req, polling while the service
// reports a non-interactive pending state (up to the configured poll timeout).
// If interactive consent is required it returns an [auth.ConsentRequiredError].
func (c *Client) RetrieveCredential(ctx context.Context, req Request) (*Retrieval, error) {
	if req.Resource == "" {
		return nil, errors.New("gcp: RetrieveCredential requires a Resource")
	}
	if req.UserID == "" {
		return nil, errors.New("gcp: RetrieveCredential requires a UserID")
	}
	if !resourceNameRE.MatchString(req.Resource) || strings.Contains(req.Resource, "..") {
		return nil, fmt.Errorf("gcp: RetrieveCredential resource %q has invalid characters", req.Resource)
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
		switch o := res.(type) {
		case credOutcome:
			cred, err := mapCredential(o.header, o.token)
			if err != nil {
				return nil, err
			}
			return &Retrieval{Credential: cred, ExpiresAt: o.expiresAt}, nil
		case consentOutcome:
			return nil, &auth.ConsentRequiredError{AuthURI: o.authURI, Nonce: o.nonce}
		case rejectedOutcome:
			return nil, fmt.Errorf("%w for %q", ErrConsentRejected, req.Resource)
		case pendingOutcome:
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
		default:
			return nil, fmt.Errorf("gcp: unexpected retrieval outcome %T", res)
		}
	}
}

// outcome is the normalized result of one retrieval attempt — a closed sum type
// (one arm per state) that RetrieveCredential type-switches on.
type outcome interface{ isOutcome() }

type (
	// credOutcome carries a successfully retrieved {header, token} credential and
	// its expiry (zero when the service does not report one).
	credOutcome struct {
		header, token string
		expiresAt     time.Time
	}
	// pendingOutcome means retrieval is still pending; poll again.
	pendingOutcome struct{}
	// consentOutcome means interactive consent is required at authURI.
	consentOutcome struct {
		authURI string
		nonce   string
	}
	// rejectedOutcome means the end user rejected consent.
	rejectedOutcome struct{}
)

func (credOutcome) isOutcome()     {}
func (pendingOutcome) isOutcome()  {}
func (consentOutcome) isOutcome()  {}
func (rejectedOutcome) isOutcome() {}

// credentialPayload is the success shape shared by both services (under
// "success" for Agent Identity, "response" for the IAM Connector operation): the
// {header, token} pair plus the token's expiry. An empty expireTime means the
// service can't say when the token expires (possibly permanent), so callers must
// not treat it as "expires now".
type credentialPayload struct {
	Token      string `json:"token"`
	Header     string `json:"header"`
	ExpireTime string `json:"expireTime"` // RFC 3339; empty when the service reports no expiry
}

// parseExpireTime parses a proto Timestamp (RFC 3339) into a time.Time, or
// returns the zero time when empty or malformed.
func parseExpireTime(s string) time.Time {
	if s == "" {
		return time.Time{}
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return time.Time{}
	}
	return t
}

// retrieveRequest is the JSON body for both services' credentials:retrieve RPC
// (the auth provider / connector is bound to the URL path, not the body).
type retrieveRequest struct {
	UserID      string   `json:"userId,omitempty"`
	Scopes      []string `json:"scopes,omitempty"`
	ContinueURI string   `json:"continueUri,omitempty"`
}

// mapCredential maps the service's {header, token} tuple to an [auth.Credential]:
// an "Authorization: Bearer" header becomes a bearer credential; any other header
// name becomes a header-based API key.
func mapCredential(header, token string) (auth.Credential, error) {
	if header == "" || token == "" {
		return nil, errors.New("gcp: credentials service returned an empty header or token")
	}
	name, scheme, _ := strings.Cut(header, ":")
	if strings.EqualFold(strings.TrimSpace(name), "authorization") &&
		strings.HasPrefix(strings.ToLower(strings.TrimSpace(scheme)), "bearer") {
		return auth.BearerCredential{Token: token}, nil
	}
	// Non-bearer header -> header-based API key. Matches adk-python: key by the
	// full returned header, and mirror the token into X-Goog-Api-Key too.
	// Rejecting an unusable name here keeps the failure at the cause: net/http
	// would otherwise accept the credential and abort the eventual request.
	if !validHeaderFieldName(header) {
		return nil, fmt.Errorf("gcp: credentials service returned %q, which is not a usable HTTP header name", truncateForError(header))
	}
	key := auth.APIKeyCredential{Name: header, Value: token}
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
	httpReq.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("gcp: call credentials service: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Read one byte past the cap so an oversized body is caught explicitly rather
	// than fed to json.Unmarshal as silently truncated (and thus garbled) JSON.
	const maxBody = 1 << 20
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxBody+1))
	if err != nil {
		return fmt.Errorf("gcp: read response: %w", err)
	}
	// Classify the status before the size check, so an oversized error page still
	// reports the status — the most actionable field — instead of only its size.
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return &APIError{StatusCode: resp.StatusCode, Body: truncateForError(strings.TrimSpace(string(data)))}
	}
	if len(data) > maxBody {
		return fmt.Errorf("gcp: credentials service response exceeded %d bytes", maxBody)
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("gcp: decode response: %w", err)
	}
	return nil
}

// validHeaderFieldName reports whether s is an RFC 9110 field name (a token).
// Hand-rolled because the module depends on golang.org/x/net only indirectly.
func validHeaderFieldName(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case strings.ContainsRune("!#$%&'*+-.^_`|~", rune(c)):
		default:
			return false
		}
	}
	return true
}

// maxErrorBody caps service-controlled text carried into an error.
const maxErrorBody = 1024

// truncateForError caps an error body so a large (e.g. HTML gateway) response
// doesn't bloat the returned error.
func truncateForError(s string) string {
	const max = maxErrorBody
	if len(s) <= max {
		return s
	}
	// Back up to a rune boundary so a multi-byte rune straddling the cap isn't
	// sliced into a mangled partial rune. Bounded: the body need not be UTF-8 at
	// all, and an unbounded scan over continuation bytes would walk to 0 and
	// discard every byte of diagnostic context.
	cut := max
	for i := 0; i < utf8.UTFMax-1 && cut > 0 && !utf8.RuneStart(s[cut]); i++ {
		cut--
	}
	if !utf8.RuneStart(s[cut]) {
		cut = max
	}
	return s[:cut] + "..."
}
