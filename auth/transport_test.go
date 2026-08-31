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
