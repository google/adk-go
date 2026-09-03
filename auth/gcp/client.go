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
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
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

// authProviderResourceRE matches an Agent Identity resource name. Together with
// connectorResourceRE it is the full set [NewProvider] accepts; the client
// itself is looser, routing any non-connector name to Agent Identity.
var authProviderResourceRE = regexp.MustCompile(`^projects/[^/]+/locations/[^/]+/authProviders/[^/]+$`)

// resourceNameRE bounds a resource name to the characters GCP resource names
// use. It cannot inject a query, a fragment, an authority or a percent-escape
// into the request URL the name is interpolated into; extra path segments are
// allowed, since a resource name is itself a path. The colon is allowed for
// domain-scoped project ids (projects/example.com:my-project/...) — the name is
// always appended after the endpoint and a /v1 segment, so it can never be read
// as a scheme.
var resourceNameRE = regexp.MustCompile(`^[A-Za-z0-9._~:/-]+$`)

// validateResource rejects a resource name that cannot be safely interpolated
// into a request URL, or that would not survive path normalization — an empty,
// "." or ".." segment blocks traversal, and also keeps the name the caller
// validated identical to the one connectorResourceRE routes on. [NewProvider]
// applies it at wiring time too, so a malformed name fails once rather than on
// every request.
func validateResource(name string) error {
	if !resourceNameRE.MatchString(name) {
		return fmt.Errorf("resource %q has invalid characters", name)
	}
	for seg := range strings.SplitSeq(name, "/") {
		if seg == "" || seg == "." || seg == ".." {
			return fmt.Errorf("resource %q has an empty or relative path segment", name)
		}
	}
	return nil
}

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
	// cacheSlot names this Client for the credential cache. What the services
	// return depends on the endpoint asked and on the identity the ask is
	// authenticated as, so two providers sharing one store must not share an
	// entry unless their Clients agree on both.
	//
	// It is one value per Client, not one per identity, because this package
	// cannot observe the identity a Client authenticates as. A caller-supplied
	// HTTPClient is opaque by construction. So is Application Default
	// Credentials: NewClient resolves it afresh on every call, so two ADC clients
	// in one process need not be the same principal — the credentials file can be
	// rewritten between them, and oauth2.HTTPClient in the context supplies the
	// transport underneath the ADC one, which this package's own PollTimeout doc
	// invites a caller to use. Erring towards a false miss costs a round trip;
	// erring the other way discloses one principal's token to another.
	//
	// The converse is a precondition this package cannot enforce and states on
	// Config.HTTPClient: one Client must not authenticate as several identities.
	//
	// The value is unique per process as well as within one, so a store that
	// outlives the process — or is shared by two — misses rather than serving an
	// entry written by a Client that no longer exists and cannot be identified.
	cacheSlot string
}

// clientSeq numbers Clients within this process, and clientNonce separates one
// process's numbering from another's. See the cacheSlot field.
var (
	clientSeq   atomic.Uint64
	clientNonce = newClientNonce()
)

// nonceBytes is the width of clientNonce. 128 bits makes a collision between two
// processes not worth reasoning about.
const nonceBytes = 16

func newClientNonce() string {
	var b [nonceBytes]byte
	// Documented never to fail, and it panics rather than returning short.
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// Config configures a [Client]. A nil *Config, or any zero-valued field, uses
// the corresponding default.
type Config struct {
	// HTTPClient calls the credential services. If nil, [NewClient] builds one
	// from Application Default Credentials (cloud-platform scope). If set, it is
	// used verbatim and ADC is not applied, so it must carry its own credentials
	// and should refuse redirects for the reason [NewClient] describes.
	//
	// It must authenticate as one identity for the Client's whole life. A Client
	// is a credential-cache dimension precisely because it is assumed to fix who
	// the credential service mints on behalf of, and a transport that picks its
	// credentials out of the request context breaks that assumption without this
	// package being able to see it: every tenant would share one cache entry.
	// Build a Client per identity instead.
	//
	// Two Clients are two cache dimensions even when built from one HTTPClient,
	// so build a Client once per identity and share it. One per request resolves
	// nothing from cache and leaves an entry behind per request.
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
	c.cacheSlot = joinFields(clientNonce, strconv.FormatUint(clientSeq.Add(1), 10))
	return c, nil
}

// CacheKey returns the key a credential for scheme, retrieved through c on
// behalf of userID under appName, is cached under — the key to hand
// [auth.CredentialStore.Delete] to invalidate that credential ahead of its
// expiry, on a consent revocation or a logout.
//
// Pass the same [ProviderScheme] value the provider was built with. A scheme
// differing in any field names a different entry, and Delete of a key nothing
// is filed under succeeds silently, so a mistake here reads as "already gone".
//
// It is meaningful only to the process that produced c, and only for as long as
// c is alive: a Client is one cache dimension, so rebuilding it strands whatever
// the old one cached.
func (c *Client) CacheKey(scheme ProviderScheme, appName, userID string) auth.CredentialKey {
	return auth.CredentialKey{AppName: appName, UserID: userID, Key: cacheSlot(c, scheme)}
}

// joinFields encodes fields as one unambiguous string, each prefixed with its
// byte length. No delimiter occurring inside a field can then make two different
// field lists encode alike, which joining on a separator alone allowed.
func joinFields(fields ...string) string {
	var b strings.Builder
	for _, f := range fields {
		b.WriteString(strconv.Itoa(len(f)))
		b.WriteByte(':')
		b.WriteString(f)
	}
	return b.String()
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
	// Credential authenticates outbound requests as the end user.
	Credential auth.Credential
	// ExpiresAt is the credential's expiry, or the zero time when the lifetime is
	// unknown: the service reports none when the token may be permanent or when
	// it cannot say, and one it reports that cannot be parsed arrives here the
	// same way. A caller must not read the zero value as "expires now", and must
	// not cache a credential carrying it.
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
	if err := validateResource(req.Resource); err != nil {
		return nil, fmt.Errorf("gcp: RetrieveCredential: %w", err)
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
			return nil, ErrConsentRejected
		case pendingOutcome:
			remaining := time.Until(deadline)
			if remaining <= 0 {
				return nil, ErrPollTimeout
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
	Token  string `json:"token"`
	Header string `json:"header"`
	// ExpireTime is RFC 3339, and empty when the service reports no expiry.
	//
	// For Agent Identity the name is the one in the published API surface
	// (https://agentidentitycredentials.googleapis.com/$discovery/rest?version=v1,
	// Success.expireTime), which also warns that the token may be revoked, or
	// expire slightly early through clock skew, before the time it names. The IAM
	// Connector service publishes no discovery document to anonymous callers, so
	// its field is assumed to be the same name and shape; if it is not, a
	// connector credential reports no lifetime and is never cached.
	ExpireTime lenientTime `json:"expireTime"`
}

// lenientTime is a JSON string that declines to fail. Only the cache reads this
// field, and the IAM Connector's shape for it is an assumption, so a value of an
// unexpected type must cost a cache entry rather than the whole retrieval — the
// credential itself is perfectly usable without an expiry.
type lenientTime string

func (t *lenientTime) UnmarshalJSON(b []byte) error {
	var s string
	if json.Unmarshal(b, &s) == nil {
		*t = lenientTime(s)
	}
	return nil
}

// parseExpireTime parses the service's expiry into a time.Time, collapsing an
// absent one and an unparseable one to the same zero time. Callers read that as
// "lifetime unknown" and decline to cache, which is the safe reading of both:
// the service omits the field when the token may be permanent or when it cannot
// say, and a value it sent but we cannot read tells us no more than silence.
func parseExpireTime(v lenientTime) time.Time {
	if v == "" {
		return time.Time{}
	}
	t, err := time.Parse(time.RFC3339, string(v))
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
