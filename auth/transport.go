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

package auth

import (
	"errors"
	"fmt"
	"io"
	"net/http"
)

// Transport is an [http.RoundTripper] that resolves a credential per request
// via a [CredentialProvider] and applies it to the outgoing request headers.
//
// The provider receives the request context (req.Context()), which — for a
// request made during a tool call — descends from the ADK context that flowed
// into the call. The resolver runs on every request so that per-user
// credentials are never shared across users. Refresh and caching belong to the
// provider: a token-source-backed one refreshes itself, and a network-backed one
// can cache in a [CredentialStore].
//
// When the provider implements [RefreshingProvider] and the base response is a
// 401/403, Transport refreshes the credential and retries once — provided the
// request body can be replayed.
type Transport struct {
	// Provider resolves the credential to apply. Required.
	Provider CredentialProvider
	// Base is the underlying RoundTripper. When nil, [http.DefaultTransport].
	Base http.RoundTripper
}

// RoundTrip implements [http.RoundTripper].
func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Per the RoundTripper contract, req.Body must be closed on every path. On
	// success the base RoundTripper owns it; guard the early returns. Mirrors
	// golang.org/x/oauth2.Transport.
	reqBodyClosed := false
	if req.Body != nil {
		defer func() {
			if !reqBodyClosed {
				_ = req.Body.Close()
			}
		}()
	}

	if t.Provider == nil {
		return nil, fmt.Errorf("auth: Transport has no Provider")
	}
	base := t.Base
	if base == nil {
		base = http.DefaultTransport
	}

	cred, err := t.Provider.Credential(req.Context())
	if err != nil {
		return nil, fmt.Errorf("auth: resolve credential: %w", err)
	}
	if cred == nil {
		return nil, fmt.Errorf("auth: provider returned nil credential")
	}

	// From here the body is handed to the base RoundTripper (directly, or as a
	// replay), which owns closing it.
	reqBodyClosed = true

	resp, err := applyAndSend(base, req, req.Body, cred)
	if err != nil {
		return resp, err
	}

	// One refresh-and-retry on a downstream auth rejection, when the provider
	// supports refresh and the request body can be replayed.
	if !isAuthRejected(resp.StatusCode) {
		return resp, nil
	}
	rp, ok := t.Provider.(RefreshingProvider)
	if !ok {
		return resp, nil
	}
	body, ok := replayBody(req)
	if !ok {
		return resp, nil
	}
	fresh, err := rp.Refresh(req.Context(), cred)
	if err != nil {
		_ = body.Close()
		// One refusal is worth surfacing: interactive consent is something the
		// caller can act on, and Credential reports it the same way, so the two
		// paths through this Transport agree on it. Every other failure degrades
		// to the rejection the downstream actually sent, which is more
		// informative than "we could not refresh".
		var consent *ConsentRequiredError
		if errors.As(err, &consent) {
			drain(resp)
			return nil, err
		}
		return resp, nil
	}
	if fresh == nil {
		_ = body.Close()
		return resp, nil
	}
	drain(resp)
	return applyAndSend(base, req, body, fresh)
}

// applyAndSend sends a clone of req (with the given body) after applying cred,
// leaving the caller's request untouched. It closes body on an apply error,
// otherwise the base RoundTripper owns it.
//
// The clone starts from the caller's request, not from the first attempt, so a
// retry cannot inherit a header the first credential set. A credential that
// writes Authorization overwrites it either way; one that writes a different
// header would otherwise leave the first one's behind.
func applyAndSend(base http.RoundTripper, req *http.Request, body io.ReadCloser, cred Credential) (*http.Response, error) {
	out := req.Clone(req.Context())
	out.Body = body
	if err := cred.Apply(out.Header); err != nil {
		if body != nil {
			_ = body.Close()
		}
		return nil, fmt.Errorf("auth: apply credential: %w", err)
	}
	return base.RoundTrip(out)
}

func isAuthRejected(code int) bool {
	return code == http.StatusUnauthorized || code == http.StatusForbidden
}

// replayBody returns a fresh copy of req's body for a retry. It reports false
// when the body exists but cannot be replayed (no GetBody).
func replayBody(req *http.Request) (io.ReadCloser, bool) {
	if req.Body == nil || req.Body == http.NoBody {
		return http.NoBody, true
	}
	if req.GetBody == nil {
		return nil, false
	}
	body, err := req.GetBody()
	if err != nil {
		return nil, false
	}
	return body, true
}

// maxDrainBytes caps the pre-retry drain of a discarded response body.
const maxDrainBytes = 4 << 10

// drain reads and closes resp.Body so the connection can be reused before the
// retry. The read is capped: an auth-rejection body is small, and a pathological
// oversized one simply isn't drained (and so isn't reused) rather than read whole.
func drain(resp *http.Response) {
	if resp.Body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxDrainBytes))
	_ = resp.Body.Close()
}

var _ http.RoundTripper = (*Transport)(nil)
