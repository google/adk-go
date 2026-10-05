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
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
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

// listingClient serves a fixed tool list so Tools() can run without a server.
type listingClient struct {
	countingClient
	tools []*mcp.Tool
}

func (c *listingClient) ListTools(context.Context) ([]*mcp.Tool, error) { return c.tools, nil }

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
// that rather than reporting a consent pause the framework will never honour.
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
	// Swap in a client that lists one tool, so Tools() does not need a server.
	s.mcpClient = &listingClient{tools: []*mcp.Tool{{Name: "cred_tool"}}}

	tools, err := s.Tools(&agent.ContextMock{})
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

// TestPreflightTolerAtesTypedNilConsentError pins that a provider returning a
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
func TestToolsFailsWhenListingNeedsConsent(t *testing.T) {
	provider := providerReturning(nil, &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth"})
	s := &set{
		auth: provider,
		mcpClient: &authenticatingListClient{
			provider: provider,
			tools:    []*mcp.Tool{{Name: "cred_tool"}},
		},
	}

	_, err := s.Tools(&agent.ContextMock{})
	if err == nil {
		t.Fatal("Tools() error = nil, want the listing failure the provider produced")
	}
	var consent *auth.ConsentRequiredError
	if !errors.As(err, &consent) {
		t.Errorf("Tools() error = %v, want it to carry the consent requirement rather than hide it", err)
	}
}

// authenticatingListClient stands in for a server that authenticates the
// listing request: it resolves the credential before answering, the way the
// transport's RoundTripper would.
type authenticatingListClient struct {
	countingClient
	provider auth.CredentialProvider
	tools    []*mcp.Tool
}

func (c *authenticatingListClient) ListTools(ctx context.Context) ([]*mcp.Tool, error) {
	if _, err := c.provider.Credential(ctx); err != nil {
		return nil, err
	}
	return c.tools, nil
}
