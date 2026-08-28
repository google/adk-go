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
	"fmt"
	"net/http"
)

// Transport is an [http.RoundTripper] that resolves a credential per request
// via a [CredentialProvider] and applies it to the outgoing request headers.
//
// The provider receives the request context (req.Context()), which — for a
// request made during a tool call — descends from the ADK context that flowed
// into the call. The resolver runs on every request so that per-user
// credentials are never shared across users; refresh and caching are handled by
// the provider's underlying token source.
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

	// Clone before mutating: RoundTrip must not modify the caller's request.
	req2 := req.Clone(req.Context())
	if err := cred.Apply(req2.Header); err != nil {
		return nil, fmt.Errorf("auth: apply credential: %w", err)
	}

	reqBodyClosed = true // base RoundTripper now owns closing the body.
	return base.RoundTrip(req2)
}

var _ http.RoundTripper = (*Transport)(nil)
