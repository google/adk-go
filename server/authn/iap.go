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

package authn

import (
	"fmt"
	"net/http"
	"strings"
)

type identityAwareProxy struct{}

const (
	// UserEMailHeader is header name which contains user email (see https://docs.cloud.google.com/iap/docs/identity-howto).
	// Example value: accounts.google.com:example@gmail.com
	UserEMailHeader = "X-Goog-Authenticated-User-Email"

	// UserIDHeader is header name which contains user id (see https://docs.cloud.google.com/iap/docs/identity-howto).
	// Example value: accounts.google.com:userIDvalue
	UserIDHeader = "X-Goog-Authenticated-User-Id"
)

// NewIdentityAwareProxy returns an [Authenticator] to be used behind
// an Identity Aware-Proxy for Google Cloud (https://cloud.google.com/security/products/iap)
// - for instance for Cloud Run-based deployment.
// The proxy puts into the headers values for email and user-id ([UserEMailHeader] and [UserIDHeader] accordingly)
// IMPORTANT: Do not rely on those headers if IAP can be bypassed!
func NewIdentityAwareProxy() Authenticator {
	return &identityAwareProxy{}
}

// Authenticate implements [Authenticator].
func (iap *identityAwareProxy) Authenticate(r *http.Request) (*Caller, error) {
	email := r.Header.Get(UserEMailHeader)
	userID := r.Header.Get(UserIDHeader)

	if email == "" || userID == "" {
		return nil, ErrUnauthenticated
	}

	// strip prefixes
	email, found := strings.CutPrefix(email, "accounts.google.com:")
	if !found {
		return nil, fmt.Errorf("%w: email header is missing the expected prefix", ErrUnauthenticated)
	}
	userID, found = strings.CutPrefix(userID, "accounts.google.com:")
	if !found {
		return nil, fmt.Errorf("%w: user id header is missing the expected prefix", ErrUnauthenticated)
	}

	return &Caller{
		UserID: userID,
		Claims: map[string]any{
			"email": email,
		},
	}, nil
}

var _ Authenticator = &identityAwareProxy{}
