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

package mcptoolset

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/authconsent"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// consentContext records what the tool asked for and replays a consent response
// on the resume leg.
type consentContext struct {
	agent.ContextMock
	authResponse *authconsent.AuthConfig
	requested    []authconsent.AuthConfig
	requestErr   error
}

func (c *consentContext) AuthResponse() *authconsent.AuthConfig { return c.authResponse }

func (c *consentContext) RequestCredential(cfg authconsent.AuthConfig) error {
	if c.requestErr != nil {
		return c.requestErr
	}
	c.requested = append(c.requested, cfg)
	return nil
}

// countingClient records whether the MCP call was reached, which is what
// distinguishes "paused for consent" from "ran anyway".
type countingClient struct {
	calls int
}

func (c *countingClient) CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	c.calls++
	return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "done"}}}, nil
}

func (*countingClient) ListTools(context.Context) ([]*mcp.Tool, error) { return nil, nil }

type fakeCredential struct{}

func (fakeCredential) Apply(http.Header) error { return nil }

func providerReturning(cred auth.Credential, err error) auth.CredentialProvider {
	return auth.ProviderFunc(func(context.Context) (auth.Credential, error) { return cred, err })
}

func newCredTool(client MCPClient, provider auth.CredentialProvider) *mcpTool {
	return &mcpTool{name: "cred_tool", mcpClient: client, auth: provider}
}

// TestPreflightRaisesConsent is the headline path: a provider that needs
// interactive consent pauses the call instead of making it.
func TestPreflightRaisesConsent(t *testing.T) {
	client := &countingClient{}
	consent := &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth", Nonce: "n", Key: "k"}
	ctx := &consentContext{}

	_, err := newCredTool(client, providerReturning(nil, consent)).Run(ctx, map[string]any{})

	if !errors.Is(err, tool.ErrCredentialRequired) {
		t.Fatalf("Run() error = %v, want it to wrap tool.ErrCredentialRequired", err)
	}
	if client.calls != 0 {
		t.Errorf("MCP CallTool ran %d times, want 0: the call must not go out before consent", client.calls)
	}
	if len(ctx.requested) != 1 {
		t.Fatalf("RequestCredential called %d times, want 1", len(ctx.requested))
	}
	got := ctx.requested[0]
	if got.CredentialKey != "k" {
		t.Errorf("requested CredentialKey = %q, want %q", got.CredentialKey, "k")
	}
	if got.ExchangedAuthCredential == nil || got.ExchangedAuthCredential.OAuth2 == nil {
		t.Fatalf("requested config = %+v, want an OAuth2 credential — the client keys its consent prompt on that path", got)
	}
	if uri := got.ExchangedAuthCredential.OAuth2.AuthURI; uri != consent.AuthURI {
		t.Errorf("requested AuthURI = %q, want %q", uri, consent.AuthURI)
	}
	if nonce := got.ExchangedAuthCredential.OAuth2.Nonce; nonce != consent.Nonce {
		t.Errorf("requested Nonce = %q, want %q", nonce, consent.Nonce)
	}
}

// TestPreflightErrorKeepsURIOut pins that the consent URL never reaches the
// error. The error becomes the tool's result: it is fed to the model and
// persisted in the session, and the URL carries the state and nonce that bind
// the credential.
func TestPreflightErrorKeepsURIOut(t *testing.T) {
	const uri = "https://consent.example/auth?login_hint=someone%40example.com&state=s"
	consent := &auth.ConsentRequiredError{AuthURI: uri, Nonce: "secret-nonce", Key: "k"}

	_, err := newCredTool(&countingClient{}, providerReturning(nil, consent)).Run(&consentContext{}, map[string]any{})
	if err == nil {
		t.Fatal("Run() error = nil, want an error")
	}
	for _, leaked := range []string{uri, "consent.example", "login_hint", "secret-nonce"} {
		if strings.Contains(err.Error(), leaked) {
			t.Errorf("Run() error = %q, which contains %q. The error reaches the model and the session; the consent URL must not.", err, leaked)
		}
	}
}

// TestPreflightAfterConsentDoesNotLoop pins the resume guard. Once the user has
// consented, a provider that still cannot mint a credential must fail, not ask
// again — asking again produces the same consent request forever.
func TestPreflightAfterConsentDoesNotLoop(t *testing.T) {
	client := &countingClient{}
	consent := &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth", Key: "k"}
	ctx := &consentContext{authResponse: &authconsent.AuthConfig{CredentialKey: "k"}}

	_, err := newCredTool(client, providerReturning(nil, consent)).Run(ctx, map[string]any{})
	if err == nil {
		t.Fatal("Run() error = nil, want an error")
	}
	if errors.Is(err, tool.ErrCredentialRequired) {
		t.Errorf("Run() error = %v, want an error that is NOT ErrCredentialRequired: raising consent again after the user already consented loops forever", err)
	}
	if len(ctx.requested) != 0 {
		t.Errorf("RequestCredential called %d times after consent completed, want 0", len(ctx.requested))
	}
	if client.calls != 0 {
		t.Errorf("MCP CallTool ran %d times, want 0", client.calls)
	}
}

// TestPreflightAfterConsentProceeds is the other half of the resume: once the
// provider can mint, the call goes out and no second consent is raised.
func TestPreflightAfterConsentProceeds(t *testing.T) {
	client := &countingClient{}
	ctx := &consentContext{authResponse: &authconsent.AuthConfig{CredentialKey: "k"}}

	if _, err := newCredTool(client, providerReturning(fakeCredential{}, nil)).Run(ctx, map[string]any{}); err != nil {
		t.Fatalf("Run() error = %v, want nil", err)
	}
	if client.calls != 1 {
		t.Errorf("MCP CallTool ran %d times, want 1", client.calls)
	}
	if len(ctx.requested) != 0 {
		t.Errorf("RequestCredential called %d times, want 0", len(ctx.requested))
	}
}

// TestPreflightSurfacesNonConsentFailure pins that an ordinary resolution
// failure reports its cause here. The transport's RoundTripper would fail the
// same way on the call below, but the MCP SDK does not preserve the cause, so
// the caller would see a generic transport error instead.
func TestPreflightSurfacesNonConsentFailure(t *testing.T) {
	client := &countingClient{}
	sentinel := errors.New("credentials service unavailable")
	ctx := &consentContext{}

	_, err := newCredTool(client, providerReturning(nil, fmt.Errorf("gcp: %w", sentinel))).Run(ctx, map[string]any{})
	if !errors.Is(err, sentinel) {
		t.Fatalf("Run() error = %v, want it to wrap the provider's cause", err)
	}
	if errors.Is(err, tool.ErrCredentialRequired) {
		t.Errorf("Run() error = %v, want it not to look like a consent pause", err)
	}
	if client.calls != 0 {
		t.Errorf("MCP CallTool ran %d times, want 0: a call with no credential would only fail again", client.calls)
	}
	if len(ctx.requested) != 0 {
		t.Errorf("RequestCredential called %d times, want 0", len(ctx.requested))
	}
}

// TestPreflightReportsRequestFailure pins that a context which cannot enqueue
// the request — a callback context, which has no function call id — surfaces
// that rather than reporting a consent pause the framework will never honor.
func TestPreflightReportsRequestFailure(t *testing.T) {
	client := &countingClient{}
	sentinel := errors.New("no function call id")
	ctx := &consentContext{requestErr: sentinel}
	consent := &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth"}

	_, err := newCredTool(client, providerReturning(nil, consent)).Run(ctx, map[string]any{})
	if !errors.Is(err, sentinel) {
		t.Fatalf("Run() error = %v, want it to wrap the enqueue failure", err)
	}
	if errors.Is(err, tool.ErrCredentialRequired) {
		t.Errorf("Run() error = %v, want it not to look like a consent pause: nothing was enqueued", err)
	}
	if client.calls != 0 {
		t.Errorf("MCP CallTool ran %d times, want 0", client.calls)
	}
}

// TestPreflightSkippedWithoutProvider pins that a toolset with no Config.Auth
// behaves exactly as before: no probe, no consent, straight to the call.
func TestPreflightSkippedWithoutProvider(t *testing.T) {
	client := &countingClient{}
	ctx := &consentContext{}

	if _, err := newCredTool(client, nil).Run(ctx, map[string]any{}); err != nil {
		t.Fatalf("Run() error = %v, want nil", err)
	}
	if client.calls != 1 {
		t.Errorf("MCP CallTool ran %d times, want 1", client.calls)
	}
	if len(ctx.requested) != 0 {
		t.Errorf("RequestCredential called %d times, want 0", len(ctx.requested))
	}
}

// TestConfigAuthReachesEachTool pins the whole wiring, from Config.Auth through
// New and Tools to the tool that runs the pre-flight. Everything else in this
// file constructs the tool directly, so without this the provider could stop
// being passed anywhere along that path and no test would notice.
func TestConfigAuthReachesEachTool(t *testing.T) {
	provider := providerReturning(nil, &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth", Key: "k"})
	ts, err := New(Config{
		Endpoint: "https://mcp.example/",
		Client:   &mcp.Client{},
		Auth:     provider,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	s, ok := ts.(*set)
	if !ok {
		t.Fatalf("New() returned %T, want *set", ts)
	}
	// Point the toolset at an in-memory server, so Tools() lists a real tool
	// without a network. Config.Auth itself needs an HTTP endpoint, which is
	// why New gets one above and this swap happens after construction.
	s.mcpClient = newConnectionRefresher(nil, inMemoryServer(t, "cred_tool"))

	tools, err := s.Tools(readonlyTestContext(t))
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("Tools() returned %d tools, want 1", len(tools))
	}

	// Assert through behavior rather than the field: the provider is only wired
	// correctly if running the tool actually reaches it.
	runnable, ok := tools[0].(*mcpTool)
	if !ok {
		t.Fatalf("Tools() produced %T, want *mcpTool", tools[0])
	}
	ctx := &consentContext{}
	if _, err := runnable.Run(ctx, map[string]any{}); !errors.Is(err, tool.ErrCredentialRequired) {
		t.Fatalf("Run() error = %v, want tool.ErrCredentialRequired — Config.Auth did not reach the tool", err)
	}
	if len(ctx.requested) != 1 {
		t.Errorf("RequestCredential called %d times, want 1", len(ctx.requested))
	}
}

// TestPreflightToleratesTypedNilConsentError pins that a provider returning a
// typed-nil *auth.ConsentRequiredError produces an error, not a panic.
//
// `var e *auth.ConsentRequiredError; return nil, e` makes a non-nil error
// interface wrapping a nil pointer, and errors.As reports a match for it. The
// consent fields are then read off nil.
func TestPreflightToleratesTypedNilConsentError(t *testing.T) {
	var typedNil *auth.ConsentRequiredError
	client := &countingClient{}
	ctx := &consentContext{}

	_, err := newCredTool(client, providerReturning(nil, typedNil)).Run(ctx, map[string]any{})
	if err == nil {
		t.Fatal("Run() error = nil, want an error")
	}
	if errors.Is(err, tool.ErrCredentialRequired) {
		t.Errorf("Run() error = %v, want it not to report a consent pause: there is no consent URL to send anyone to", err)
	}
	if len(ctx.requested) != 0 {
		t.Errorf("RequestCredential called %d times, want 0", len(ctx.requested))
	}
	if client.calls != 0 {
		t.Errorf("MCP CallTool ran %d times, want 0", client.calls)
	}
}

// TestPreflightRejectsConfirmationPlusConsent pins that the unsupported
// combination reports itself instead of stalling.
//
// A tool that needs both a confirmation and a consent round-trip cannot work:
// ADK emits the request event from the model-response path only, so whichever
// round-trip is raised during the other's resume is recorded and never reaches
// the client. Verified by running both legs — the confirmation resume raises
// consent, and a consent resume re-raises confirmation.
func TestPreflightRejectsConfirmationPlusConsent(t *testing.T) {
	client := &countingClient{}
	consent := &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth", Key: "k"}
	tl := newCredTool(client, providerReturning(nil, consent))
	tl.requireConfirmation = true
	// Past the confirmation gate, which is where the consent probe runs.
	ctx := &confirmedContext{confirmation: &toolconfirmation.ToolConfirmation{Confirmed: true}}

	_, err := tl.Run(ctx, map[string]any{})
	if err == nil {
		t.Fatal("Run() error = nil, want an error naming the unsupported combination")
	}
	if errors.Is(err, tool.ErrCredentialRequired) {
		t.Errorf("Run() error = %v, want it not to report a consent pause: the request event is "+
			"never emitted from a resume, so nothing would answer it", err)
	}
	if !strings.Contains(err.Error(), "cannot be combined") {
		t.Errorf("Run() error = %v, want it to say the combination is unsupported", err)
	}
	if len(ctx.requested) != 0 {
		t.Errorf("RequestCredential called %d times, want 0", len(ctx.requested))
	}
	if client.calls != 0 {
		t.Errorf("MCP CallTool ran %d times, want 0", client.calls)
	}
}

// TestPreflightAllowsConfirmationWithNonInteractiveAuth is the control: a
// confirmation-gated tool with an ordinary provider is untouched by the rule
// above, since nothing raises a second round-trip.
func TestPreflightAllowsConfirmationWithNonInteractiveAuth(t *testing.T) {
	client := &countingClient{}
	tl := newCredTool(client, providerReturning(fakeCredential{}, nil))
	tl.requireConfirmation = true
	ctx := &confirmedContext{confirmation: &toolconfirmation.ToolConfirmation{Confirmed: true}}

	if _, err := tl.Run(ctx, map[string]any{}); err != nil {
		t.Fatalf("Run() error = %v, want nil", err)
	}
	if client.calls != 1 {
		t.Errorf("MCP CallTool ran %d times, want 1", client.calls)
	}
}

// confirmedContext is a consentContext that also carries a tool confirmation.
type confirmedContext struct {
	consentContext
	confirmation *toolconfirmation.ToolConfirmation
	actions      session.EventActions
}

func (c *confirmedContext) ToolConfirmation() *toolconfirmation.ToolConfirmation {
	return c.confirmation
}

func (c *confirmedContext) Actions() *session.EventActions { return &c.actions }

// TestToolsFailsWhenListingNeedsConsent pins the limitation Config.Auth
// documents. Tool listing runs before any tool call, so there is no call to
// pause and no id to key a consent request on; a server that authenticates
// listing behind an interactive provider therefore cannot be listed at all.
//
// It runs against a real streamable HTTP server so the request goes through
// the auth RoundTripper that Config.Auth installs, which is where the listing
// actually fails.
// TestToolsListsWhenProviderNeedsConsent pins the order the round-trip depends
// on over the real HTTP transport: an interactive provider that still needs
// consent must not break listing, because listing runs before any tool call and
// cannot pause. The request goes out unauthenticated, the tool is listed, and
// the consent is raised by the tool's own preflight on the first call.
func TestToolsListsWhenProviderNeedsConsent(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{Name: "listing", Version: "v1"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "cred_tool"}, func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, struct{}, error) {
		return &mcp.CallToolResult{}, struct{}{}, nil
	})
	var authHeaders []string
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authHeaders = append(authHeaders, r.Header.Get("Authorization"))
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)

	consent := &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth", Key: "k"}
	ts, err := New(Config{Endpoint: srv.URL, Auth: providerReturning(nil, consent)})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	t.Cleanup(func() { _ = ts.(*set).Close() })

	tools, err := ts.Tools(readonlyTestContext(t))
	if err != nil {
		t.Fatalf("Tools() error = %v, want listing to succeed: consent is raised per call, not at listing", err)
	}
	if len(tools) != 1 || tools[0].Name() != "cred_tool" {
		t.Fatalf("Tools() = %v, want [cred_tool]", tools)
	}
	for _, h := range authHeaders {
		if h != "" {
			t.Errorf("listing sent Authorization %q, want none while consent is pending", h)
		}
	}

	ctx := &consentContext{}
	_, err = tools[0].(*mcpTool).Run(ctx, map[string]any{})
	if !errors.Is(err, tool.ErrCredentialRequired) {
		t.Fatalf("Run() error = %v, want ErrCredentialRequired from the preflight", err)
	}
	if len(ctx.requested) != 1 || ctx.requested[0].ExchangedAuthCredential.OAuth2.AuthURI != consent.AuthURI {
		t.Errorf("RequestCredential got %+v, want one request carrying the provider's consent URL", ctx.requested)
	}
}

// TestToolsFailsWhenServerAuthenticatesListing is the residual limitation: a
// server that itself rejects an unauthenticated tools/list cannot be listed
// before the user has consented, because listing has nowhere to pause.
func TestToolsFailsWhenServerAuthenticatesListing(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{Name: "listing", Version: "v1"}, nil)
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)

	ts, err := New(Config{
		Endpoint: srv.URL,
		Auth:     providerReturning(nil, &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth"}),
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	t.Cleanup(func() { _ = ts.(*set).Close() })
	if _, err := ts.Tools(readonlyTestContext(t)); err == nil {
		t.Fatal("Tools() error = nil, want the server's 401 to surface: listing cannot pause for consent")
	}
}

// inMemoryServer starts an in-memory MCP server exposing the named tools and
// returns the client side of its transport.
func inMemoryServer(t *testing.T, names ...string) mcp.Transport {
	t.Helper()
	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	server := mcp.NewServer(&mcp.Implementation{Name: "in-memory", Version: "v1"}, nil)
	for _, name := range names {
		mcp.AddTool(server, &mcp.Tool{Name: name}, func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, struct{}, error) {
			return &mcp.CallToolResult{}, struct{}{}, nil
		})
	}
	if _, err := server.Connect(t.Context(), serverTransport, nil); err != nil {
		t.Fatalf("server.Connect() error = %v", err)
	}
	return clientTransport
}

// readonlyTestContext is a real readonly ADK context bound to the test's
// lifetime. Listing waits on ctx.Done, which agent.ContextMock does not provide.
func readonlyTestContext(t *testing.T) agent.ReadonlyContext {
	t.Helper()
	return icontext.NewReadonlyContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}))
}

// TestPreflightPausesWhenConfirmationProviderDeclines pins that the refusal of
// consent-plus-confirmation is decided per call, by what the confirmation
// provider answered for this call, not by the provider merely being configured.
// A toolset whose provider gates only destructive tools must still be able to
// pause a read-only call for consent.
func TestPreflightPausesWhenConfirmationProviderDeclines(t *testing.T) {
	client := &countingClient{}
	consent := &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth", Key: "k"}
	tl := newCredTool(client, providerReturning(nil, consent))
	tl.requireConfirmationProvider = func(string, any) bool { return false }
	ctx := &consentContext{}

	_, err := tl.Run(ctx, map[string]any{})
	if !errors.Is(err, tool.ErrCredentialRequired) {
		t.Fatalf("Run() error = %v, want ErrCredentialRequired: this call needs no confirmation, so consent alone is one round-trip", err)
	}
	if len(ctx.requested) != 1 {
		t.Errorf("RequestCredential called %d times, want 1", len(ctx.requested))
	}
	if client.calls != 0 {
		t.Errorf("MCP CallTool ran %d times, want 0", client.calls)
	}
}

// TestPreflightRefusesWhenConfirmationProviderRequires is the other half: when
// the provider says this call needs confirmation, the confirmation prompt goes
// out first and the consent preflight on the confirmed resume refuses the
// second round-trip, as with the static flag.
func TestPreflightRefusesWhenConfirmationProviderRequires(t *testing.T) {
	client := &countingClient{}
	consent := &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth", Key: "k"}
	tl := newCredTool(client, providerReturning(nil, consent))
	tl.requireConfirmationProvider = func(string, any) bool { return true }
	ctx := &confirmedContext{confirmation: &toolconfirmation.ToolConfirmation{Confirmed: true}}

	_, err := tl.Run(ctx, map[string]any{})
	if err == nil || !strings.Contains(err.Error(), "cannot be combined") {
		t.Fatalf("Run() error = %v, want the unsupported-combination error", err)
	}
	if len(ctx.requested) != 0 || client.calls != 0 {
		t.Errorf("RequestCredential called %d times and CallTool ran %d times, want 0 and 0", len(ctx.requested), client.calls)
	}
}

// TestPreflightUnwrapsConsentError pins that a provider may return the consent
// error wrapped, as the CredentialProvider doc allows, and the preflight still
// pauses rather than reporting a resolution failure.
func TestPreflightUnwrapsConsentError(t *testing.T) {
	client := &countingClient{}
	consent := &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth", Key: "k"}
	tl := newCredTool(client, providerReturning(nil, fmt.Errorf("provider: %w", consent)))
	ctx := &consentContext{}

	_, err := tl.Run(ctx, map[string]any{})
	if !errors.Is(err, tool.ErrCredentialRequired) {
		t.Fatalf("Run() error = %v, want ErrCredentialRequired for a wrapped consent error", err)
	}
	if len(ctx.requested) != 1 || ctx.requested[0].CredentialKey != "k" {
		t.Errorf("RequestCredential got %+v, want one request under key k", ctx.requested)
	}
	if client.calls != 0 {
		t.Errorf("MCP CallTool ran %d times, want 0", client.calls)
	}
}

// TestToolsFailsOnTypedNilConsentErrorAtListing pins that the transport treats
// a typed-nil *ConsentRequiredError the way the preflight does: as an ordinary
// resolution failure, not as consent pending. Listing fails with it.
func TestToolsFailsOnTypedNilConsentErrorAtListing(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{Name: "listing", Version: "v1"}, nil)
	srv := httptest.NewServer(mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil))
	t.Cleanup(srv.Close)
	var typedNil *auth.ConsentRequiredError
	ts, err := New(Config{Endpoint: srv.URL, Auth: providerReturning(nil, typedNil)})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	t.Cleanup(func() { _ = ts.(*set).Close() })
	if _, err := ts.Tools(readonlyTestContext(t)); err == nil {
		t.Fatal("Tools() error = nil, want a typed-nil consent error to fail listing rather than pass as pending consent")
	}
}

// refreshingProvider is a provider whose credential can be rejected and
// refreshed, recording the refreshes it was asked for.
type refreshingProvider struct {
	mu       sync.Mutex
	consent  bool
	token    string
	refreshs int
}

func (p *refreshingProvider) Credential(context.Context) (auth.Credential, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.consent {
		return nil, &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth"}
	}
	return auth.BearerCredential{Token: p.token}, nil
}

func (p *refreshingProvider) Refresh(context.Context, auth.Credential) (auth.Credential, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.refreshs++
	p.token = "fresh"
	return auth.BearerCredential{Token: p.token}, nil
}

// TestListingRefreshForwardsOnlyWithACredential pins the transport wrapper's
// refresh behavior: a 401 on a request that carried a stale credential is
// refreshed through the provider and retried, while a 401 on a request that
// went out with no credential, because consent is pending, is not: there is
// nothing to refresh, and the provider's Refresh must not be asked.
func TestListingRefreshForwardsOnlyWithACredential(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{Name: "listing", Version: "v1"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "cred_tool"}, func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, struct{}, error) {
		return &mcp.CallToolResult{}, struct{}{}, nil
	})
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer fresh" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)

	t.Run("stale credential is refreshed", func(t *testing.T) {
		p := &refreshingProvider{token: "stale"}
		ts, err := New(Config{Endpoint: srv.URL, Auth: p})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		t.Cleanup(func() { _ = ts.(*set).Close() })
		tools, err := ts.Tools(readonlyTestContext(t))
		if err != nil || len(tools) != 1 {
			t.Fatalf("Tools() = %v, %v; want the tool after one refresh", tools, err)
		}
		if p.refreshs == 0 {
			t.Error("provider.Refresh was never called, yet the stale credential was rejected")
		}
	})

	t.Run("pending consent is not refreshed", func(t *testing.T) {
		p := &refreshingProvider{consent: true}
		ts, err := New(Config{Endpoint: srv.URL, Auth: p})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		t.Cleanup(func() { _ = ts.(*set).Close() })
		if _, err := ts.Tools(readonlyTestContext(t)); err == nil {
			t.Fatal("Tools() error = nil, want the server's 401 to surface while consent is pending")
		}
		if p.refreshs != 0 {
			t.Errorf("provider.Refresh was called %d times with no credential to refresh", p.refreshs)
		}
	})
}
