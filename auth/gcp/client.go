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
	"strconv"
	"strings"
	"time"
	"unicode/utf16"
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
// use. It cannot inject a query, a fragment, an authority or a percent-escape
// into the request URL the name is interpolated into. Extra path segments are
// allowed, since a resource name is itself a path. The colon is allowed for
// domain-scoped project ids (projects/example.com:my-project/...) — the name is
// always appended after the endpoint and a /v1 segment, so it can never be read
// as a scheme.
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
	// request's own UserID and ContinueURI are removed and replaced with
	// "[redacted]", the text is lowercased wherever anything matched, and the
	// result is capped at a kilobyte.
	//
	// It is still service-controlled. Render it with %q, as [APIError.Error] does
	// — the service can put a newline in it directly, and an escaped one in the
	// body is decoded on the way here, so "%s" into a log forges a second line.
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

// RetrieveCredential retrieves a credential for req, polling while the service
// reports a non-interactive pending state (up to the configured poll timeout).
// If interactive consent is required it returns an [auth.ConsentRequiredError].
//
// Every error past validation names the resource. One client can serve several
// resources, so a caller holding only the error — including a direct caller,
// which has no provider to attribute it — must be able to tell which one failed.
// Wrapped with %w throughout, so [ErrConsentRejected], [ErrPollTimeout] and
// [auth.ConsentRequiredError] stay matchable.
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
		return nil, fmt.Errorf("gcp: credentials service returned %q, which is not a usable HTTP header name", serviceText(header, secrets...))
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
		return &APIError{StatusCode: resp.StatusCode, Body: serviceText(strings.TrimSpace(string(data)), secrets...)}
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
		return fmt.Errorf("%w: %q", ErrMalformedResponse, serviceText(err.Error(), secrets...))
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

// withheldText replaces service text that could not be shown free of the caller's
// own identifiers. It names no secret and is constant, so it leaks nothing itself.
const withheldText = "[withheld: could not be shown free of the request's own identifiers]"

// redactedMarker is what redact writes in place of a removed value.
const redactedMarker = "[redacted]"

// serviceText prepares service-controlled text for an error message.
//
// Contract: no secret is recoverable from the returned string by this package's
// decoder, or nothing is returned. A caller may quote the result into an error a
// model reads and a session stores.
//
// Not covered, and the boundary is exact: an identifier a service mangles into a
// form no decoder reconstructs and a human reads anyway — split with a \n,
// percent-encoded, echoed in half. The guarantee is that WE add no identifier,
// not that we can launder one back out of arbitrary text.
//
// Two things here look like they could be simpler and cannot be.
//
// The choice between the two candidates is made on the OUTPUT, never on a property
// of s and u. The service writes the text both candidates are measured from, so any
// test over those inputs is one it controls: adding an occurrence the decode
// destroys costs it nothing and moves the comparison wherever it likes. Three
// revisions were broken that way before this one.
//
// redact runs before the cap. Capping first cuts an identifier in half whenever it
// straddles the boundary, and the surviving prefix matches nothing, so a long
// enough response smuggles out the acting user's leading bytes.
func serviceText(s string, secrets ...string) string {
	// Capped BEFORE the check, so what is examined is exactly what is returned.
	// The cap is not neutral: its ellipsis is appended text, and appended text can
	// finish a secret the untruncated string only started — an identifier ending
	// in a dot is completed by the first character of "...".
	if out := truncateForError(redact(s, secrets...)); !recoverable(out, secrets) {
		return out
	}
	if u := unescapeJSON(s); u != s {
		if out := truncateForError(redact(u, secrets...)); !recoverable(out, secrets) {
			return out
		}
	}
	// Both candidates still yield a secret. That needs an encoding this package
	// decodes wrapped in one it does not, so it is not a shape an ordinary service
	// produces — which is the reason to treat it as hostile and drop the text. The
	// status code, the resource and the sentinel all survive in the error around it.
	return withheldText
}

// recoverable reports whether any secret can be read out of x, either literally or
// once this package's decoder has been run over it to a fixpoint.
//
// Decoding x rather than trusting redact's own bookkeeping is the point: it asks
// what an attacker gets from the bytes being returned, so it cannot be steered by
// what the service put in the bytes that were measured.
func recoverable(x string, secrets []string) bool {
	for _, part := range strings.Split(x, redactedMarker) {
		lx := strings.ToLower(part)
		lu := strings.ToLower(decodeFully(part))
		for _, v := range secrets {
			if v == "" {
				continue
			}
			// Decoded spelling too, because decoding is what mangles a secret. An
			// identifier containing \/ survives a decoded copy's scrub as the slash
			// it decodes to, which is not the secret and is still the identity.
			for _, form := range []string{strings.ToLower(v), strings.ToLower(decodeFully(v))} {
				if form != "" && (strings.Contains(lx, form) || strings.Contains(lu, form)) {
					return true
				}
			}
		}
	}
	return false
}

// decodeFully applies unescapeJSON until the text stops changing.
//
// One pass is not enough. A doubly escaped identifier decodes to a singly escaped
// one, which still hides it from a substring scrub and which this same function
// will happily decode the rest of the way for anyone who asks twice — so checking
// only the first pass leaves an identifier the package's own decoder recovers.
//
// It terminates because every branch that changes anything writes fewer bytes than
// it consumed, so a pass that changes the text strictly shortens it. TestDecodeFully
// pins that, and it is the whole termination argument.
func decodeFully(s string) string {
	for {
		u := unescapeJSON(s)
		if u == s {
			return s
		}
		s = u
	}
}

// unescapeJSON decodes the JSON escapes that can hide a caller-supplied value
// from a substring scrub, and leaves every other one alone — a malformed escape
// included, which stays verbatim rather than being dropped.
//
// Three are decoded, and on valid UTF-8 built from those three plus ordinary
// text this agrees with encoding/json, which is what the differential test
// asserts. On a byte that is not valid UTF-8 the two part company on purpose:
// encoding/json replaces it with U+FFFD, and this copies it through untouched,
// because the input here is an arbitrary response body rather than a decoded
// string and substituting a byte the service sent would be the fabrication this
// scrub is trying to avoid.
//
// \uXXXX, because a JSON encoder commonly escapes non-ASCII, and a surrogate
// PAIR as one rune rather than two halves. A non-BMP identifier arrives as two
// \uXXXX escapes. Decoding each alone yields U+FFFD twice, so the identifier is
// in neither the decoded copy nor the original and survives the scrub whole.
// That is not an encoding this function declines to handle — it is two of the
// escapes it says it decodes.
//
// \/, because RFC 8259 permits it and several encoders emit it by default,
// which is enough to hide every slash in an echoed ContinueURI.
//
// A doubled backslash, because it is what stops a literal one from introducing
// either of the other two. Skipping it is not neutral: the second backslash then
// opens an escape JSON says is not there, so \\uXXXX decoded to a backslash
// followed by the rune rather than to the six literal characters.
func unescapeJSON(s string) string {
	if !strings.Contains(s, `\`) {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); {
		if i+2 <= len(s) && s[i] == '\\' && s[i+1] == '\\' {
			b.WriteByte('\\')
			i += 2
			continue
		}
		if i+2 <= len(s) && s[i] == '\\' && s[i+1] == '/' {
			b.WriteByte('/')
			i += 2
			continue
		}
		if i+6 <= len(s) && s[i] == '\\' && s[i+1] == 'u' {
			if n, err := strconv.ParseUint(s[i+2:i+6], 16, 32); err == nil {
				r := rune(n)
				if utf16.IsSurrogate(r) && i+12 <= len(s) && s[i+6] == '\\' && s[i+7] == 'u' {
					if lo, err := strconv.ParseUint(s[i+8:i+12], 16, 32); err == nil {
						if pair := utf16.DecodeRune(r, rune(lo)); pair != utf8.RuneError {
							b.WriteRune(pair)
							i += 12
							continue
						}
					}
				}
				b.WriteRune(r)
				i += 6
				continue
			}
		}
		b.WriteByte(s[i])
		i++
	}
	return b.String()
}

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

// redact removes caller-supplied values from a service-controlled string,
// ignoring case. Where it removes anything, the text it returns is lowercased.
//
// Case-insensitively because the two sides need not agree on it: a server taking
// session.UserID from an OIDC email claim keeps whatever case the provider sent,
// while the services lowercase an address before echoing it, so a literal match
// misses the echo entirely. Unlike an encoding, that is not something the
// best-effort caveat above covers — it is the plain identifier, spelled the same.
//
// Lowercased rather than spliced back into the original because there is no cheap
// way to map an offset in the lowered copy onto the original. Lowercasing changes
// byte length in both directions — Go folds U+0130 to a one-byte "i" and U+023A
// to a three-byte U+2C65 — so a body carrying equal numbers of each has the same
// total length lowered as unlowered while every offset inside it has moved. A
// guard comparing totals sees nothing wrong, the splice lands short, and the
// identifier survives in full. That was a real bug here, and keeping the
// service's capitalisation is not worth another.
//
// Every value is matched in ONE left-to-right pass rather than one pass each.
// Sequential passes let the second value match inside the marker the first
// inserted — a user id of "e" turns "[redacted]" into "[r[redacted]dact[redacted]d]"
// — and let a short value break a longer one that contains it, so the longer one
// then matches nothing and its remainder survives. Overlapping candidates are
// resolved earliest-first, then longest, which is what keeps a ContinueURI
// carrying the user id from being split by the user id.
//
// Text with no match is returned untouched, so an error carrying no secret keeps
// its case. Empty values are dropped, and that is load-bearing rather than
// cosmetic: the empty string matches at the cursor forever, so admitting one
// would leave the cursor where it is and the loop would never finish.
func redact(s string, values ...string) string {
	lowered := make([]string, 0, len(values))
	for _, v := range values {
		if v != "" {
			lowered = append(lowered, strings.ToLower(v))
		}
	}
	if len(lowered) == 0 {
		return s
	}
	ls := strings.ToLower(s)

	// next[i] is where lowered[i] matches at or after pos, or -1 once exhausted.
	// Refreshed only for values whose match pos has passed, and pos only moves
	// forward, so the whole scan stays linear in len(ls) per value.
	next := make([]int, len(lowered))
	for i, lv := range lowered {
		next[i] = strings.Index(ls, lv)
	}

	var b strings.Builder
	pos, hit := 0, false
	for {
		at, which := -1, -1
		for i, n := range next {
			if n < 0 {
				continue
			}
			if at < 0 || n < at || (n == at && len(lowered[i]) > len(lowered[which])) {
				at, which = n, i
			}
		}
		if at < 0 {
			break
		}
		// Extend the range while ANY occurrence starts inside it. Choosing the
		// earliest match covers a contained occurrence and an equal start, but not
		// one that straddles the far edge: the refresh below only looks forward
		// from the cursor, so a straddler is neither redacted nor found again and
		// its tail is copied straight out.
		//
		// Scanned position by position rather than off next[], which holds only
		// the FIRST occurrence of each value at or after the cursor. A second
		// occurrence of the same value can start inside the range and straddle it,
		// and next[] cannot see it. Measured with the earlier version:
		// redact("https://cb.test/u/alice@example.test/alice@example.test",
		//        "alice@example.test", "https://cb.test/u/alice@example.test/a")
		// returned "[redacted]lice@example.test", keeping 17 of the address's 18
		// bytes. The loop re-reads end each step, so an extension made at k is
		// picked up by the same walk.
		end := at + len(lowered[which])
		for k := at + 1; k < end; k++ {
			for _, lv := range lowered {
				if k+len(lv) > end && strings.HasPrefix(ls[k:], lv) {
					end = k + len(lv)
				}
			}
		}
		// One marker per redacted run, not per match. Adjacent ranges are the
		// common case for a short secret — a megabyte of the acting user's single
		// initial would otherwise emit a megabyte of markers, ten times the input,
		// to produce a kilobyte of error.
		//
		// hit doubles as "something is already emitted", which is what makes
		// at == pos mean "this range touches the last one" rather than "this is
		// the first range and it starts at zero".
		if at > pos || !hit {
			b.WriteString(ls[pos:at])
			b.WriteString(redactedMarker)
		}
		pos, hit = end, true
		for i, lv := range lowered {
			if next[i] >= 0 && next[i] < pos {
				if j := strings.Index(ls[pos:], lv); j < 0 {
					next[i] = -1
				} else {
					next[i] = pos + j
				}
			}
		}
	}
	if !hit {
		return s
	}
	b.WriteString(ls[pos:])
	return b.String()
}
