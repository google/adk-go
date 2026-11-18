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
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"cloud.google.com/go/auth/credentials"
	"cloud.google.com/go/auth/httptransport"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/internal/clientcert"
)

const (
	cloudPlatformScope      = "https://www.googleapis.com/auth/cloud-platform"
	defaultAgentIdentityURL = "https://agentidentitycredentials.googleapis.com"
	defaultConnectorURL     = "https://iamconnectorcredentials.googleapis.com"
	// Unlike the two above, these request a client certificate.
	defaultAgentIdentityMTLSURL = "https://agentidentitycredentials.mtls.googleapis.com"
	defaultConnectorMTLSURL     = "https://iamconnectorcredentials.mtls.googleapis.com"

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
// connectorResourceRE it is the full set [NewProvider] accepts. The client
// itself is looser, routing any non-connector name to Agent Identity.
var authProviderResourceRE = regexp.MustCompile(`^projects/[^/]+/locations/[^/]+/authProviders/[^/]+$`)

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
	// request's own UserID and ContinueURI, and its PriorToken on the route that
	// sends it, are removed WHERE THE SCRUB CAN MATCH THEM and replaced with
	// "[redacted]", the text is lowercased wherever anything matched, and only
	// the first kilobyte of the response is drawn on, with "..." marking that the
	// rest was dropped. A kilobyte is also the ceiling on Body itself, which is
	// not the same promise: one matched run becomes a ten-byte marker whatever it
	// replaced, so bounding the source alone left the result several times larger.
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
	// cacheSlot names this Client for the credential cache. What the services
	// return depends on the endpoint asked and on the identity the ask is
	// authenticated as, so two providers sharing one store must not share an
	// entry unless their Clients agree on both.
	//
	// It is one value per Client, not one per identity, because this package
	// cannot observe the identity a Client authenticates as. A caller-supplied
	// HTTPClient is opaque by construction. So is Application Default
	// Credentials: NewClient resolves it afresh on every call, so two ADC clients
	// in one process need not be the same principal — the credentials file, or
	// GOOGLE_APPLICATION_CREDENTIALS itself, can be rewritten between them.
	// Erring towards a false miss costs a round trip; erring the other way
	// discloses one principal's token to another.
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
	// To bound requests without giving up ADC, build a client with
	// [httptransport.NewClient] and pass it as HTTPClient: DetectOpts naming
	// the cloud-platform scope and a fresh, non-nil Headers map
	// per client (the transport writes the quota project into the map it is
	// given and drops it with a nil one, so a shared map carries the first
	// client's quota project to the next), a Logger that discards (the default
	// one writes token requests and responses to stderr when
	// GOOGLE_SDK_GO_LOGGING_LEVEL=debug), DisableTelemetry, CheckRedirect
	// returning [http.ErrUseLastResponse], and then the Timeout. That client
	// decides on a client certificate by the auth library's default, which is
	// wider than the rule [NewClient] documents, so set
	// GOOGLE_API_USE_CLIENT_CERTIFICATE where the two differ.
	PollTimeout time.Duration
}

// NewClient builds a Client from cfg; a nil cfg (or any zero field) uses
// defaults. Unless cfg.HTTPClient is set, it discovers Application Default
// Credentials (cloud-platform scope) to authenticate calls to the services,
// through the same transport the Google Cloud client libraries use: it sends the
// quota project, when there is one, as X-Goog-User-Project, caches tokens and
// refreshes them ahead of expiry, and presents the environment's client
// certificate when the rule below turns it on.
//
// The client certificate follows adk-python's rule.
// GOOGLE_API_USE_CLIENT_CERTIFICATE decides when it is set: true, TRUE or True
// turns the certificate on, and any other value, the empty string included,
// turns it off. When it is unset, the certificate is on only if
// GOOGLE_API_CERTIFICATE_CONFIG names a certificate config file with a workload
// section, and only in the googleapis.com universe. With the variable unset
// the Google Cloud client libraries for Go also use an enterprise certificate
// proxy, SecureConnect, or a certificate config at gcloud's default path. Here
// those take GOOGLE_API_USE_CLIENT_CERTIFICATE=true. A workload section that
// names no certificate fails NewClient.
//
// A certificate is only presented to an endpoint that requests one, so when
// the client has one the default endpoints become the services' mTLS
// hosts, agentidentitycredentials.mtls.googleapis.com and
// iamconnectorcredentials.mtls.googleapis.com. GOOGLE_API_USE_MTLS_ENDPOINT
// overrides that choice with the values the client libraries take: auto (the
// default) as described, always for the mTLS hosts with or without a
// certificate, and never for the plain hosts. Any other value fails NewClient.
// [Config.AgentIdentityEndpoint] and [Config.ConnectorEndpoint] each win over
// the choice, and with [Config.HTTPClient] set none is made.
// With a certificate, gcloud user credentials also fetch their tokens from
// oauth2.mtls.googleapis.com, which requests it. A service-account key keeps
// its token_uri.
// A credentials file naming a universe domain other than the client's —
// googleapis.com, or GOOGLE_CLOUD_UNIVERSE_DOMAIN when set — fails every request.
//
// The quota project is GOOGLE_CLOUD_QUOTA_PROJECT when set, and otherwise the
// credentials file's quota_project_id. The variable is not tied to the
// credentials, so the header can name a project they do not, as it does for
// metadata-server credentials, which name none.
//
// ctx does not bound the returned Client and nothing is read from it:
// credential discovery reads the environment, and tokens are minted as requests
// need them rather than on ctx, so a Client built inside a request-scoped
// context keeps working after that request ends. Through v2.8.0 the ADC-backed
// client took its transport and Timeout from an [http.Client] stored in ctx under
// oauth2.HTTPClient, as the PollTimeout doc then advised. That value is no longer
// read. Passing that same plain client as [Config.HTTPClient] is not the
// replacement: HTTPClient is used as given, so it would send no credentials.
// To keep a Timeout together with ADC, build the client the [Config.PollTimeout]
// doc describes.
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
	if cfg.PollTimeout > 0 {
		c.pollTimeout = cfg.PollTimeout
	}
	if c.httpClient == nil {
		mtlsMode, err := mtlsEndpointMode()
		if err != nil {
			return nil, err
		}
		certSource, err := clientcert.Source()
		if err != nil {
			return nil, fmt.Errorf("gcp: build default credentials client: %w", err)
		}
		// Discarded: the library's default logger answers
		// GOOGLE_SDK_GO_LOGGING_LEVEL=debug by writing each token request and
		// response to stderr, signed assertion and minted token included.
		logger := slog.New(slog.DiscardHandler)
		detect := &credentials.DetectOptions{Scopes: []string{cloudPlatformScope}, Logger: logger}
		// The auth library's transport rather than a hand-written bearer header,
		// so client certificates, the quota project header and token caching
		// come from it. See the NewClient doc.
		opts := &httptransport.Options{
			// Non-nil on purpose: the transport adds the quota project header to
			// this map, and with a nil map it allocates a private one that its
			// round-tripper never sees (cloud.google.com/go/auth through v0.24.4).
			Headers: http.Header{},
			Logger:  logger,
			// Off, as through v2.8.0. Left on, the library opens a client span for
			// every call, and with GOOGLE_SDK_GO_EXPERIMENTAL_TRACING=true records
			// on it the service's error message, which APIError.Body carries only
			// scrubbed.
			DisableTelemetry: true,
		}
		if certSource != nil {
			// The library finds the same source on its own, and with it also
			// sends the token exchange through the certificate.
			opts.DetectOpts = detect
		} else {
			// Left to its default the library would look for a certificate on
			// its own, by a wider rule than clientcert.Source's, and use it
			// for the token exchange as well. A source that presents nothing
			// stops the lookup, and credentials detected here keep the token
			// exchange off that source: detected by the library, they would
			// move to its mTLS token endpoint for any non-nil source.
			creds, err := credentials.DetectDefault(detect)
			if err != nil {
				return nil, fmt.Errorf("gcp: build default credentials client: %w", err)
			}
			opts.Credentials = creds
			opts.ClientCertProvider = noClientCertificate
		}
		hc, err := httptransport.NewClient(opts)
		if err != nil {
			return nil, fmt.Errorf("gcp: build default credentials client: %w", err)
		}
		// The transport re-signs every hop, below the layer where net/http
		// strips credentials on a cross-host redirect, so a redirect would leak
		// the token to whatever host it names.
		hc.CheckRedirect = func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		}
		c.httpClient = hc
		// One decision serves both services, since its inputs are environment
		// variables and certificate files, not the host.
		if mtlsMode == mtlsAlways || (mtlsMode == mtlsAuto && certSource != nil) {
			c.agentIdentityURL = defaultAgentIdentityMTLSURL
			c.connectorURL = defaultConnectorMTLSURL
		}
	}
	if cfg.AgentIdentityEndpoint != "" {
		c.agentIdentityURL = strings.TrimRight(cfg.AgentIdentityEndpoint, "/")
	}
	if cfg.ConnectorEndpoint != "" {
		c.connectorURL = strings.TrimRight(cfg.ConnectorEndpoint, "/")
	}
	c.cacheSlot = joinFields(clientNonce, strconv.FormatUint(clientSeq.Add(1), 10))
	return c, nil
}

// noClientCertificate answers a server's certificate request with no
// certificate.
func noClientCertificate(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
	return &tls.Certificate{}, nil
}

const (
	mtlsAuto   = "auto"
	mtlsAlways = "always"
	mtlsNever  = "never"
)

// mtlsEndpointMode reads GOOGLE_API_USE_MTLS_ENDPOINT. An unknown value is an
// error, as in the generated Python client adk-python uses. The Go client
// libraries fall back to the plain endpoint without saying so.
func mtlsEndpointMode() (string, error) {
	v := os.Getenv("GOOGLE_API_USE_MTLS_ENDPOINT")
	switch m := strings.ToLower(v); m {
	case "", mtlsAuto:
		return mtlsAuto, nil
	case mtlsAlways, mtlsNever:
		return m, nil
	default:
		return "", fmt.Errorf("gcp: GOOGLE_API_USE_MTLS_ENDPOINT is %q, want auto, always or never", v)
	}
}

// CacheKey returns the key a credential for scheme, retrieved through c on
// behalf of id, is cached under — the key to hand
// [auth.CredentialStore.Delete] to invalidate that credential ahead of its
// expiry, on a consent revocation or a logout. It is the key the provider
// itself files under, built from the same [agent.Identity] the provider reads
// off the invocation. Only id.AppName and id.UserID take part: a credential
// is shared across a user's sessions.
//
// Pass the same [ProviderScheme] the provider was built with, as it was then.
// [NewProvider] copies Scopes, so changing the caller's slice afterwards changes
// the key this computes but not the one the provider files under. A scheme
// differing in any field names a different entry, and Delete of a key nothing
// is filed under succeeds silently, so a mistake here reads as "already gone".
//
// It is meaningful only to the process that produced c, and only for as long as
// c is alive: a Client is one cache dimension, so rebuilding it strands whatever
// the old one cached. A Client that did not come from [NewClient] has no cache
// dimension at all, names an entry nothing ever writes, and must not be used.
func (c *Client) CacheKey(scheme ProviderScheme, id agent.Identity) auth.CredentialKey {
	return auth.CredentialKey{AppName: id.AppName, UserID: id.UserID, Key: cacheSlot(c, scheme)}
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
	// PriorToken is the previously issued, now rejected, token. Setting it asks
	// the service to mint a fresh credential rather than return the one it
	// already has. Empty on a normal fetch.
	//
	// The two services spell the ask differently on the wire and this one field
	// covers both — Agent Identity is sent the token, the IAM Connector a
	// boolean. See the request types below.
	PriorToken string
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
//
// Every error past validation names the resource. One client can serve several
// resources, so a caller holding only the error — including a direct caller,
// which has no provider to attribute it — must be able to tell which one failed.
// Wrapped with %w throughout, so [ErrConsentRejected], [ErrPollTimeout] and
// [auth.ConsentRequiredError] stay matchable.
//
// It returns a [*Retrieval] where v2.3.0 and v2.4.0 returned the
// [auth.Credential] alone, so a caller written against either release stops
// compiling and reads [Retrieval.Credential] instead.
//
// Matchable with [errors.Is] and [errors.As], and only with those. Naming the
// resource wraps every error past validation, so a direct == against any of those
// sentinels, or a bare type assertion to [*APIError], stops being true where it
// was true in v2.3.0. A decode failure is now [ErrMalformedResponse] and no
// longer carries the [encoding/json] error behind it, so [errors.As] against
// *json.SyntaxError stops finding one. On the ADC-backed client the error
// behind a rejected cloud-platform token request is whatever the
// cloud.google.com/go/auth credential for that source reports, where releases
// through v2.8.0 reported golang.org/x/oauth2's. For a service-account key
// that is a *cloud.google.com/go/auth.Error where it was an
// *oauth2.RetrieveError. Which type the other credential sources wrap, if any,
// is the library's choice, not this package's promise.
func (c *Client) RetrieveCredential(ctx context.Context, req Request) (_ *Retrieval, err error) {
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
			cred, err := mapCredential(o.header, o.token, req.UserID, req.ContinueURI, req.PriorToken)
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

// The two services agree on the retrieval body except for how a caller forces a
// refresh, so that one field is all these types do not share.
//
// Agent Identity takes the rejected token itself, and documents that a caller
// seeing a PERMISSION_DENIED should retry with it set
// (https://agentidentitycredentials.googleapis.com/$discovery/rest?version=v1,
// RetrieveCredentialsRequest.forceRefreshToken). The IAM Connector takes a
// boolean instead — it publishes no discovery document to anonymous callers, but
// adk-python's generated client fills the field in, as force_refresh
// (integrations/agent_identity/_iam_connector_credentials_provider.py). Sending
// one service the other's field would leave the refresh unforced at best, and
// rejected as an unknown field at worst.
type (
	// agentIdentityRequest is the JSON body for Agent Identity's
	// credentials:retrieve (the auth provider is bound to the URL path, not the
	// body).
	agentIdentityRequest struct {
		UserID            string   `json:"userId,omitempty"`
		Scopes            []string `json:"scopes,omitempty"`
		ContinueURI       string   `json:"continueUri,omitempty"`
		ForceRefreshToken string   `json:"forceRefreshToken,omitempty"`
	}
	// connectorRequest is the same for the IAM Connector.
	connectorRequest struct {
		UserID       string   `json:"userId,omitempty"`
		Scopes       []string `json:"scopes,omitempty"`
		ContinueURI  string   `json:"continueUri,omitempty"`
		ForceRefresh bool     `json:"forceRefresh,omitempty"`
	}
)

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
	for i := range len(s) {
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
