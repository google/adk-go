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
// use. It cannot inject a query, a fragment, an authority or a percent-escape
// into the request URL the name is interpolated into. Extra path segments are
// allowed, since a resource name is itself a path. The colon is allowed for
// domain-scoped project ids (projects/example.com:my-project/...) — the name is
// always appended after the endpoint and a version segment (/v1 for Agent
// Identity, /v1alpha for the connector), so it can never be read as a scheme.
var resourceNameRE = regexp.MustCompile(`^[A-Za-z0-9._~:/-]+$`)

// validateResource rejects a resource name that cannot be safely interpolated
// into a request URL, or that would not survive path normalization — an empty,
// "." or ".." segment blocks traversal, and also keeps the name the caller
// validated identical to the one connectorResourceRE routes on.
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
	// ErrMalformedResponse means a 2xx body failed JSON decoding — that arm and no
	// other. A 2xx that overruns the 1 MiB cap before the decoder sees it, or that
	// decodes cleanly but names an empty or unusable header, returns a plain error.
	// So this replaces a *json.SyntaxError check rather than answering the broader
	// question of whether the service sent back something unusable.
	//
	// It exists because the decode error is no longer wrapped with %w. The
	// decoder's message quotes the token it choked on, which is service-controlled
	// text that has to be scrubbed, and keeping the wrap would leave the unscrubbed
	// original reachable through Unwrap.
	//
	// Every encoding/json error loses its type this way, not only *json.SyntaxError.
	// TestDecodeErrorScrubsTheActingUser asserts on "cannot unmarshal number", which
	// is a *json.UnmarshalTypeError, so the second one a caller could match on is
	// demonstrated by this package's own tests.
	ErrMalformedResponse = errors.New("gcp: credentials service returned an undecodable response")
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
	// Body is the response body, prepared for an error rather than verbatim: the
	// request's own UserID and ContinueURI are removed WHERE THE SCRUB CAN MATCH
	// THEM and replaced with "[redacted]", the text is lowercased wherever
	// anything matched, and only the first kilobyte of the response is drawn on,
	// with "..." marking that the rest was dropped. A kilobyte is also the ceiling
	// on Body itself, which is not the same promise: one matched run becomes a
	// ten-byte marker whatever it replaced, so bounding the source alone left the
	// result several times larger.
	//
	// Removal is best effort, and the guarantee is narrower than removal: no value
	// this package was given is recoverable from Body by this package's own
	// decoder, unless the value is spelled entirely out of the characters of
	// "[redacted]" and laid out as a run of them.
	//
	// That carve-out is deliberate. The marker is text this package writes, so a
	// value found only inside one was never disclosed by the service, and counting
	// it would suppress every response — a UserID of "e" occurs in "[redacted]".
	// It cannot reach the values this path actually carries: the excused class is
	// the substrings of "[redacted][redacted]…", which is eight distinct letters
	// with no "@", ":" or "/" among them, so no address and no redirect URI is in
	// it. A service can put such a marker in its own body and be excused the same
	// way, since there is no telling its markers from ours.
	//
	// A value can also be partly legible. Where the UserID is a substring of the
	// ContinueURI and the service spells the URI with escapes the scrub cannot
	// match, the UserID's marker lands inside the URI and the rest of the URI
	// reads plainly around it, as "my-[redacted].test/oauth/callback".
	//
	// It can also be none of the response. Where the identifiers could not be
	// shown to be gone, Body is a fixed sentence saying so and bears no relation
	// to what the service sent. A response too long to examine whole lands here
	// too, since not looking is not the same as looking and finding nothing. There
	// is no supported way to tell that case apart, so branch on StatusCode and
	// treat Body as diagnostic text for a human rather than as something to match
	// on.
	//
	// Otherwise it is still service-controlled. Render it with %q, as
	// [APIError.Error] does — the service can put a newline in it directly, and an
	// escaped one in the body is decoded on the way here, so "%s" into a log
	// forges a second line.
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
	//
	// Validated before anything is sent, and rejected with an error naming the
	// reason. Letters, digits, and "._~/-:" are allowed, and every path segment
	// must be non-empty and be neither "." nor "..". So a trailing slash, a
	// doubled slash, and a relative segment are refused, while a domain-scoped
	// project id such as projects/example.com:my-project/... is accepted.
	//
	// The segment rule is what keeps routing and normalization from disagreeing:
	// the routing pattern above is matched on the name as given, and a name that
	// normalizes to a different one would be routed by one and served as the
	// other.
	//
	// Against v2.3.0, which took ^[A-Za-z0-9._~/-]+$ and refused any name
	// containing "..", four shapes moved and one did not. Newly refused: a
	// trailing slash, a doubled slash, and a "." segment. Newly accepted: the
	// colon, and ".." INSIDE a segment, so projects/example..com/... is a name
	// v2.3.0 refused and this one takes. Unchanged: a ".." segment was refused
	// there by that substring test and is refused here by the segment rule.
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
//
// Every error past validation names the resource. One client can serve several
// resources, so a caller holding only the error — including a direct caller,
// which has no provider to attribute it — must be able to tell which one failed.
// Wrapped with %w throughout, so [ErrConsentRejected], [ErrPollTimeout] and
// [auth.ConsentRequiredError] stay matchable.
//
// Matchable with [errors.Is] and [errors.As], and only with those. Naming the
// resource wraps every error past validation, so a direct == against any of those
// sentinels, or a bare type assertion to [*APIError], stops being true where it
// was true in v2.3.0. A decode failure is now [ErrMalformedResponse] and no
// longer carries the [encoding/json] error behind it, so [errors.As] against
// *json.SyntaxError stops finding one.
func (c *Client) RetrieveCredential(ctx context.Context, req Request) (_ auth.Credential, err error) {
	if req.Resource == "" {
		return nil, errors.New("gcp: RetrieveCredential requires a Resource")
	}
	if req.UserID == "" {
		return nil, errors.New("gcp: RetrieveCredential requires a UserID")
	}
	if err := validateResource(req.Resource); err != nil {
		return nil, fmt.Errorf("gcp: RetrieveCredential: %w", err)
	}
	// Named once here rather than at each return: the two sentinels and the
	// context error carried no resource at all, and the arms that did name it
	// then had it named twice over on the provider path. Appended rather than
	// prefixed, because the errors arriving here already open with the package
	// name and a second one reads as a stutter.
	//
	// The resource and nothing else. This error reaches a tool, which feeds it to
	// the model and persists it in the session, and every other id in scope comes
	// off the request — a user id is commonly an email, and a session id arrives
	// unvalidated from the request path. The resource is configuration.
	//
	// Adding nothing else is not sufficient on its own, because an [APIError]
	// carries up to a kilobyte of the service's own response, and a service that
	// rejects a request commonly quotes back what it rejected. That scrub happens
	// at the single place an APIError is built, which is the only one that can do
	// it correctly — see doPost.
	//
	// It deliberately does NOT happen again here. Re-running redact over an
	// already-scrubbed Body cannot find a real occurrence, because the first pass
	// removed them all, and it can find a spurious one: a user id of "e" matches
	// inside the "[redacted]" marker itself and rewrites it to
	// "[r[redacted]dact[redacted]d]", nesting once per pass and destroying the
	// operator's error along the way. A second scrub that can only corrupt is
	// worse than none, so this defer decorates and nothing else.
	defer func() {
		if err == nil {
			return
		}
		err = fmt.Errorf("%w (resource %q)", err, req.Resource)
	}()

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
			return mapCredential(o.header, o.token, req.UserID, req.ContinueURI)
		case consentOutcome:
			return nil, &auth.ConsentRequiredError{AuthURI: o.authURI, Nonce: o.nonce}
		case rejectedOutcome:
			return nil, ErrConsentRejected
		case pendingOutcome:
			remaining := time.Until(deadline)
			if remaining <= 0 {
				return nil, ErrPollTimeout
			}
			// A timer rather than time.After: the caller giving up is an ordinary
			// way out of this loop, and time.After holds the runtime timer until
			// it fires whether anyone is still waiting or not.
			timer := time.NewTimer(min(backoff, remaining))
			select {
			case <-ctx.Done():
				timer.Stop()
				return nil, ctx.Err()
			case <-timer.C:
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
	// credOutcome carries a successfully retrieved {header, token} credential.
	credOutcome struct{ header, token string }
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

// credentialPayload is the {header, token} success shape shared by both services
// (under "success" for Agent Identity, "response" for the IAM Connector operation).
type credentialPayload struct {
	Token  string `json:"token"`
	Header string `json:"header"`
}

// retrieveRequest is the JSON body for both services' credentials:retrieve RPC
// (the auth provider / connector is bound to the URL path, not the body).
type retrieveRequest struct {
	UserID      string   `json:"userId,omitempty"`
	Scopes      []string `json:"scopes,omitempty"`
	ContinueURI string   `json:"continueUri,omitempty"`
}

// mapCredential maps the service's {header, token} tuple to an [auth.Credential]:
// an "Authorization: Bearer" header becomes a bearer credential. Any other header
// name becomes a header-based API key.
//
// secrets are the caller-supplied values to scrub from the rejection below: the
// header name is service-controlled and reaches an error, so it gets the same
// treatment as a response body and an operation message.
//
// It is not the only other one. A consent URI reaches [auth.ConsentRequiredError]
// unscrubbed on purpose, because it is the URL the acting user must visit and an
// identifier in it is load-bearing rather than a leak — see that type's docs. So
// count the paths before assuming a new arm here is covered.
func mapCredential(header, token string, secrets ...string) (auth.Credential, error) {
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
		return nil, fmt.Errorf("gcp: credentials service returned %q, which is not a usable HTTP header name", redactedForError(header, secrets...))
	}
	key := auth.APIKeyCredential{Name: header, Value: token}
	return auth.WithHeaders(key, map[string]string{"X-Goog-Api-Key": token}), nil
}

// doPost sends body as JSON to url and decodes a JSON response into out.
func (c *Client) doPost(ctx context.Context, url string, body, out any, secrets ...string) error {
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
		return &APIError{StatusCode: resp.StatusCode, Body: redactedForError(strings.TrimSpace(string(data)), secrets...)}
	}
	if len(data) > maxBody {
		return fmt.Errorf("gcp: credentials service response exceeded %d bytes", maxBody)
	}
	if err := json.Unmarshal(data, out); err != nil {
		// Scrubbed and capped like the body above, and wrapped in a sentinel rather
		// than in the decoder's own error. A decoder message quotes the token it
		// choked on — a userId echoed back as a JSON number where a string was
		// expected lands in "cannot unmarshal number 1035…" — so it carries service
		// text, and keeping %w on the original would leave that text reachable
		// unscrubbed through Unwrap. [ErrMalformedResponse] is what a caller matches
		// instead of *json.SyntaxError, which this stops satisfying.
		//
		// %q like the other two service-text sites. A decoder message can carry a
		// byte the service chose, and unescapeJSON can turn an escape in it into a
		// real control character, so it is quoted rather than pasted.
		return fmt.Errorf("%w: %q", ErrMalformedResponse, redactedForError(err.Error(), secrets...))
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
