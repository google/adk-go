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
type closeTrackingBody struct{ closed bool }

func (b *closeTrackingBody) Read([]byte) (int, error) { return 0, io.EOF }
func (b *closeTrackingBody) Close() error             { b.closed = true; return nil }

// sequenceRT returns the given status codes in order (200 once exhausted) and
// records the Authorization header and body of each request it receives.
type sequenceRT struct {
	statuses []int
	calls    int
	auths    []string
	bodies   []string
}

func (s *sequenceRT) RoundTrip(req *http.Request) (*http.Response, error) {
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
	return &http.Response{StatusCode: code, Body: http.NoBody, Header: http.Header{}}, nil
}

// refreshProvider is a RefreshingProvider that yields cred first and fresh after
// Refresh is called.
type refreshProvider struct {
	cred      auth.Credential
	fresh     auth.Credential
	refreshes int
}

func (p *refreshProvider) Credential(context.Context) (auth.Credential, error) { return p.cred, nil }

func (p *refreshProvider) Refresh(context.Context) (auth.Credential, error) {
	p.refreshes++
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

func (p *errRefreshProvider) Refresh(context.Context) (auth.Credential, error) {
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
