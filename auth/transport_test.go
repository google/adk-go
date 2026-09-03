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

package auth_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"testing"

	"google.golang.org/adk/v2/auth"
)

func TestTransportAppliesCredential(t *testing.T) {
	base := &captureRT{}
	tr := &auth.Transport{Provider: auth.StaticToken("abc"), Base: base}

	req := newRequest(t)
	if _, err := tr.RoundTrip(req); err != nil {
		t.Fatalf("RoundTrip() error = %v", err)
	}

	if !base.called {
		t.Fatal("base transport was not called")
	}
	if base.gotAuth != "Bearer abc" {
		t.Errorf("Authorization = %q, want %q", base.gotAuth, "Bearer abc")
	}
	// RoundTrip must not mutate the caller's request.
	if got := req.Header.Get("Authorization"); got != "" {
		t.Errorf("original request was mutated: Authorization = %q", got)
	}
}

func TestTransportProviderError(t *testing.T) {
	base := &captureRT{}
	boom := errors.New("boom")
	p := auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
		return nil, boom
	})
	tr := &auth.Transport{Provider: p, Base: base}

	_, err := tr.RoundTrip(newRequest(t))
	if !errors.Is(err, boom) {
		t.Fatalf("RoundTrip() error = %v, want %v", err, boom)
	}
	if base.called {
		t.Error("base transport must not be called when the provider errors")
	}
}

func TestTransportConsentRequiredPropagates(t *testing.T) {
	base := &captureRT{}
	p := auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
		return nil, &auth.ConsentRequiredError{AuthURI: "https://consent.example", Key: "k"}
	})
	tr := &auth.Transport{Provider: p, Base: base}

	_, err := tr.RoundTrip(newRequest(t))
	var consent *auth.ConsentRequiredError
	if !errors.As(err, &consent) {
		t.Fatalf("RoundTrip() error = %v, want *auth.ConsentRequiredError", err)
	}
	if base.called {
		t.Error("base transport must not be called when consent is required")
	}
}

func TestTransportNilProvider(t *testing.T) {
	tr := &auth.Transport{Base: &captureRT{}}
	if _, err := tr.RoundTrip(newRequest(t)); err == nil {
		t.Fatal("RoundTrip() = nil error, want error for nil Provider")
	}
}

func TestTransportClosesBodyOnError(t *testing.T) {
	body := &closeTrackingBody{}
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://example.test/", body)
	if err != nil {
		t.Fatalf("NewRequestWithContext() error = %v", err)
	}
	tr := &auth.Transport{Provider: auth.ProviderFunc(func(context.Context) (auth.Credential, error) {
		return nil, errors.New("boom")
	})}

	if _, err := tr.RoundTrip(req); err == nil {
		t.Fatal("RoundTrip() = nil error, want error")
	}
	if !body.closed {
		t.Error("RoundTrip must close req.Body when it returns an error")
	}
}

// captureRT is a stub http.RoundTripper that records the Authorization header
// it received and whether it was invoked.
type captureRT struct {
	called  bool
	gotAuth string
}

func (c *captureRT) RoundTrip(req *http.Request) (*http.Response, error) {
	c.called = true
	c.gotAuth = req.Header.Get("Authorization")
	return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody, Header: http.Header{}}, nil
}

// newRequest builds a GET request bound to the test context.
func newRequest(t *testing.T) *http.Request {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://example.test/", nil)
	if err != nil {
		t.Fatalf("NewRequestWithContext() error = %v", err)
	}
	return req
}

// closeTrackingBody is an io.ReadCloser that records whether Close was called.
type closeTrackingBody struct {
	payload string
	read    int
	closed  bool
}

func (b *closeTrackingBody) Read(p []byte) (int, error) {
	if b.read >= len(b.payload) {
		return 0, io.EOF
	}
	n := copy(p, b.payload[b.read:])
	b.read += n
	return n, nil
}

func (b *closeTrackingBody) Close() error { b.closed = true; return nil }

// drained reports whether the body was read to the end and closed.
func (b *closeTrackingBody) drained() bool { return b.closed && b.read == len(b.payload) }

// sequenceRT returns the given status codes in order (200 once exhausted) and
// records the Authorization header and body of each request it receives.
type sequenceRT struct {
	statuses []int
	calls    int
	auths    []string
	bodies   []string
	reqs     []*http.Request
	// respBodies are the response bodies handed out, so a test can assert the
	// first one was drained and closed before the retry. http.NoBody cannot show
	// that: its Read is EOF and its Close a no-op.
	respBodies []*closeTrackingBody
}

func (s *sequenceRT) RoundTrip(req *http.Request) (*http.Response, error) {
	s.reqs = append(s.reqs, req)
	s.auths = append(s.auths, req.Header.Get("Authorization"))
	body := ""
	if req.Body != nil {
		b, _ := io.ReadAll(req.Body)
		body = string(b)
	}
	s.bodies = append(s.bodies, body)

	code := http.StatusOK
	if s.calls < len(s.statuses) {
		code = s.statuses[s.calls]
	}
	s.calls++
	rb := &closeTrackingBody{payload: "rejected"}
	s.respBodies = append(s.respBodies, rb)
	return &http.Response{StatusCode: code, Body: rb, Header: http.Header{}}, nil
}

// refreshProvider is a RefreshingProvider that yields cred first and fresh after
// Refresh is called. It records what it was told was rejected.
type refreshProvider struct {
	cred      auth.Credential
	fresh     auth.Credential
	resolves  int
	refreshes int
	rejected  auth.Credential
}

// Credential yields a different credential on every call, so a Transport that
// re-derived the rejected one instead of passing the one it applied hands
// Refresh a value no request ever carried — which is the round-1 defect, and is
// invisible against a fake that answers constantly.
func (p *refreshProvider) Credential(context.Context) (auth.Credential, error) {
	p.resolves++
	if p.resolves == 1 {
		return p.cred, nil
	}
	return p.fresh, nil
}

func (p *refreshProvider) Refresh(_ context.Context, rejected auth.Credential) (auth.Credential, error) {
	p.refreshes++
	p.rejected = rejected
	return p.fresh, nil
}

func TestTransportRefreshesAndRetriesOn401(t *testing.T) {
	base := &sequenceRT{statuses: []int{http.StatusUnauthorized, http.StatusOK}}
	p := &refreshProvider{
		cred:  auth.BearerCredential{Token: "stale"},
		fresh: auth.BearerCredential{Token: "fresh"},
	}
	tr := &auth.Transport{Provider: p, Base: base}

	resp, err := tr.RoundTrip(newRequest(t))
	if err != nil {
		t.Fatalf("RoundTrip() error = %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200 after refresh+retry", resp.StatusCode)
	}
	if base.calls != 2 {
		t.Errorf("base calls = %d, want 2", base.calls)
	}
	if p.refreshes != 1 {
		t.Errorf("refreshes = %d, want 1", p.refreshes)
	}
	if want := []string{"Bearer stale", "Bearer fresh"}; !slices.Equal(base.auths, want) {
		t.Errorf("auth headers = %v, want %v", base.auths, want)
	}
}

func TestTransportRetryReplaysBody(t *testing.T) {
	base := &sequenceRT{statuses: []int{http.StatusForbidden, http.StatusOK}}
	p := &refreshProvider{
		cred:  auth.BearerCredential{Token: "stale"},
		fresh: auth.BearerCredential{Token: "fresh"},
	}
	tr := &auth.Transport{Provider: p, Base: base}

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://example.test/", strings.NewReader("payload"))
	if err != nil {
		t.Fatalf("NewRequestWithContext() error = %v", err)
	}
	if _, err := tr.RoundTrip(req); err != nil {
		t.Fatalf("RoundTrip() error = %v", err)
	}
	if base.calls != 2 {
		t.Fatalf("base calls = %d, want 2", base.calls)
	}
	for i, b := range base.bodies {
		if b != "payload" {
			t.Errorf("call %d body = %q, want %q (body must be replayed)", i, b, "payload")
		}
	}
}

func TestTransportNoRefreshWithoutRefreshingProvider(t *testing.T) {
	base := &sequenceRT{statuses: []int{http.StatusUnauthorized}}
	tr := &auth.Transport{Provider: auth.StaticToken("x"), Base: base}

	resp, err := tr.RoundTrip(newRequest(t))
	if err != nil {
		t.Fatalf("RoundTrip() error = %v", err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401 (a non-refreshing provider is not retried)", resp.StatusCode)
	}
	if base.calls != 1 {
		t.Errorf("base calls = %d, want 1", base.calls)
	}
}

func TestTransportNoRetryNonReplayableBody(t *testing.T) {
	base := &sequenceRT{statuses: []int{http.StatusUnauthorized, http.StatusOK}}
	p := &refreshProvider{
		cred:  auth.BearerCredential{Token: "stale"},
		fresh: auth.BearerCredential{Token: "fresh"},
	}
	tr := &auth.Transport{Provider: p, Base: base}

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://example.test/", strings.NewReader("payload"))
	if err != nil {
		t.Fatalf("NewRequestWithContext() error = %v", err)
	}
	req.GetBody = nil // body cannot be replayed

	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip() error = %v", err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401 (non-replayable body is not retried)", resp.StatusCode)
	}
	if base.calls != 1 || p.refreshes != 0 {
		t.Errorf("base calls = %d, refreshes = %d; want 1 and 0", base.calls, p.refreshes)
	}
}

func TestTransportRetryStillRejected(t *testing.T) {
	// Both the original and the refreshed credential are rejected: after one
	// refresh+retry the second 401 is returned, with no further retries.
	base := &sequenceRT{statuses: []int{http.StatusUnauthorized, http.StatusUnauthorized}}
	p := &refreshProvider{
		cred:  auth.BearerCredential{Token: "stale"},
		fresh: auth.BearerCredential{Token: "fresh"},
	}
	tr := &auth.Transport{Provider: p, Base: base}

	resp, err := tr.RoundTrip(newRequest(t))
	if err != nil {
		t.Fatalf("RoundTrip() error = %v", err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401 (retry also rejected)", resp.StatusCode)
	}
	if base.calls != 2 {
		t.Errorf("base calls = %d, want 2 (one retry only)", base.calls)
	}
	if p.refreshes != 1 {
		t.Errorf("refreshes = %d, want 1", p.refreshes)
	}
}

// errRefreshProvider is a RefreshingProvider whose Refresh always fails.
type errRefreshProvider struct{ cred auth.Credential }

func (p *errRefreshProvider) Credential(context.Context) (auth.Credential, error) {
	return p.cred, nil
}

func (p *errRefreshProvider) Refresh(context.Context, auth.Credential) (auth.Credential, error) {
	return nil, errors.New("refresh boom")
}

func TestTransportRefreshErrorSurfacesOriginalAndClosesReplay(t *testing.T) {
	base := &sequenceRT{statuses: []int{http.StatusUnauthorized, http.StatusOK}}
	tr := &auth.Transport{Provider: &errRefreshProvider{cred: auth.BearerCredential{Token: "stale"}}, Base: base}

	replay := &closeTrackingBody{}
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://example.test/", strings.NewReader("payload"))
	if err != nil {
		t.Fatalf("NewRequestWithContext() error = %v", err)
	}
	req.GetBody = func() (io.ReadCloser, error) { return replay, nil }

	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip() error = %v", err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401 (refresh error surfaces the original rejection)", resp.StatusCode)
	}
	if base.calls != 1 {
		t.Errorf("base calls = %d, want 1 (no retry after refresh error)", base.calls)
	}
	if !replay.closed {
		t.Error("replay body was not closed after refresh error (leak)")
	}
}

func TestTransportRefreshNilCredentialNoPanic(t *testing.T) {
	// A RefreshingProvider that returns (nil, nil) must not panic on retry; the
	// original rejection is surfaced.
	base := &sequenceRT{statuses: []int{http.StatusUnauthorized, http.StatusOK}}
	p := &refreshProvider{cred: auth.BearerCredential{Token: "stale"}, fresh: nil}
	tr := &auth.Transport{Provider: p, Base: base}

	resp, err := tr.RoundTrip(newRequest(t))
	if err != nil {
		t.Fatalf("RoundTrip() error = %v", err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401 (nil refreshed credential surfaces the original)", resp.StatusCode)
	}
	if base.calls != 1 {
		t.Errorf("base calls = %d, want 1 (no retry with nil credential)", base.calls)
	}
}

// Refresh is told which credential the downstream rejected. Without it a
// provider has to guess from its own cache, which by then may hold a different
// credential entirely.
func TestTransportPassesTheRejectedCredentialToRefresh(t *testing.T) {
	base := &sequenceRT{statuses: []int{http.StatusUnauthorized, http.StatusOK}}
	stale := auth.BearerCredential{Token: "stale"}
	p := &refreshProvider{cred: stale, fresh: auth.BearerCredential{Token: "fresh"}}
	tr := &auth.Transport{Provider: p, Base: base}

	resp, err := tr.RoundTrip(mustRequest(t, http.MethodGet, "https://example.com/"))
	if err != nil {
		t.Fatalf("RoundTrip() error = %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if p.rejected != auth.Credential(stale) {
		t.Errorf("Refresh was told %+v was rejected, want %+v", p.rejected, stale)
	}
}

// consentRefreshProvider refuses to refresh because the end user must consent
// again.
type consentRefreshProvider struct{ cred auth.Credential }

func (p *consentRefreshProvider) Credential(context.Context) (auth.Credential, error) {
	return p.cred, nil
}

func (p *consentRefreshProvider) Refresh(context.Context, auth.Credential) (auth.Credential, error) {
	return nil, fmt.Errorf("gcp: %w", &auth.ConsentRequiredError{AuthURI: "https://consent.example", Nonce: "n"})
}

// A refresh that needs interactive consent is the one refusal worth surfacing:
// it is actionable, and Credential reports it the same way, so both paths
// through one Transport agree. Every other refresh failure degrades to the
// downstream's own rejection.
func TestTransportSurfacesConsentFromRefresh(t *testing.T) {
	base := &sequenceRT{statuses: []int{http.StatusUnauthorized, http.StatusOK}}
	tr := &auth.Transport{Provider: &consentRefreshProvider{cred: auth.BearerCredential{Token: "stale"}}, Base: base}

	resp, err := tr.RoundTrip(mustRequest(t, http.MethodGet, "https://example.com/"))
	if resp != nil {
		_ = resp.Body.Close()
		t.Error("RoundTrip() returned a response alongside the consent error")
	}
	var consent *auth.ConsentRequiredError
	if !errors.As(err, &consent) {
		t.Fatalf("RoundTrip() error = %v, want a *auth.ConsentRequiredError", err)
	}
	if consent.AuthURI != "https://consent.example" {
		t.Errorf("AuthURI = %q, want the one the provider reported", consent.AuthURI)
	}
	if base.calls != 1 {
		t.Errorf("base calls = %d, want 1 (no retry once consent is required)", base.calls)
	}
}

// The retry is built from the caller's request, not from the first attempt, so
// a header the first credential set cannot survive into it.
func TestTransportRetryDoesNotInheritTheFirstCredentialsHeader(t *testing.T) {
	base := &sequenceRT{statuses: []int{http.StatusUnauthorized, http.StatusOK}}
	p := &refreshProvider{
		cred:  auth.APIKeyCredential{Name: "X-First", Value: "one"},
		fresh: auth.APIKeyCredential{Name: "X-Second", Value: "two"},
	}
	tr := &auth.Transport{Provider: p, Base: base}

	resp, err := tr.RoundTrip(mustRequest(t, http.MethodGet, "https://example.com/"))
	if err != nil {
		t.Fatalf("RoundTrip() error = %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if len(base.reqs) != 2 {
		t.Fatalf("base calls = %d, want 2", len(base.reqs))
	}
	retry := base.reqs[1]
	if got := retry.Header.Get("X-Second"); got != "two" {
		t.Errorf("retry X-Second = %q, want the refreshed credential", got)
	}
	if got := retry.Header.Get("X-First"); got != "" {
		t.Errorf("retry still carries X-First = %q from the rejected credential", got)
	}
}

func mustRequest(t *testing.T, method, url string) *http.Request {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), method, url, nil)
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	return req
}

// The first response's body is read out and closed before the retry, or its
// connection is not returned to the pool.
func TestTransportDrainsTheRejectedResponseBeforeRetrying(t *testing.T) {
	base := &sequenceRT{statuses: []int{http.StatusUnauthorized, http.StatusOK}}
	p := &refreshProvider{cred: auth.BearerCredential{Token: "stale"}, fresh: auth.BearerCredential{Token: "fresh"}}
	tr := &auth.Transport{Provider: p, Base: base}

	resp, err := tr.RoundTrip(mustRequest(t, http.MethodGet, "https://example.com/"))
	if err != nil {
		t.Fatalf("RoundTrip() error = %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if len(base.respBodies) != 2 {
		t.Fatalf("responses = %d, want 2", len(base.respBodies))
	}
	if !base.respBodies[0].drained() {
		t.Error("the rejected response was not drained and closed before the retry")
	}
}

// The same holds on the consent path, which returns no response for the caller
// to close.
func TestTransportDrainsTheRejectedResponseOnConsent(t *testing.T) {
	base := &sequenceRT{statuses: []int{http.StatusUnauthorized, http.StatusOK}}
	tr := &auth.Transport{Provider: &consentRefreshProvider{cred: auth.BearerCredential{Token: "stale"}}, Base: base}

	if _, err := tr.RoundTrip(mustRequest(t, http.MethodGet, "https://example.com/")); err == nil {
		t.Fatal("RoundTrip() = nil error, want the consent error")
	}
	if len(base.respBodies) != 1 {
		t.Fatalf("responses = %d, want 1", len(base.respBodies))
	}
	if !base.respBodies[0].drained() {
		t.Error("the rejected response was not drained and closed, and nobody else can close it")
	}
}

// Only an auth rejection triggers the refresh. Anything else — success, a
// server error, a rate limit — is returned as it came, or a refresh a new token
// cannot help becomes a duplicate request and a needless re-mint.
func TestTransportRefreshesOnlyOnAuthRejection(t *testing.T) {
	for _, status := range []int{
		http.StatusOK,
		http.StatusBadRequest,
		http.StatusTooManyRequests,
		http.StatusInternalServerError,
	} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			base := &sequenceRT{statuses: []int{status, http.StatusOK}}
			p := &refreshProvider{cred: auth.BearerCredential{Token: "stale"}, fresh: auth.BearerCredential{Token: "fresh"}}
			tr := &auth.Transport{Provider: p, Base: base}

			resp, err := tr.RoundTrip(mustRequest(t, http.MethodGet, "https://example.com/"))
			if err != nil {
				t.Fatalf("RoundTrip() error = %v", err)
			}
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != status {
				t.Errorf("status = %d, want %d returned untouched", resp.StatusCode, status)
			}
			if base.calls != 1 {
				t.Errorf("base calls = %d, want 1 (no retry)", base.calls)
			}
			if p.refreshes != 0 {
				t.Errorf("refreshes = %d, want 0", p.refreshes)
			}
		})
	}
}
