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
)

// header is an [Authenticator] that reads the caller's UserID from one request
// header. This is not HTTP Basic authentication; it trusts a header set by a
// trusted upstream (a proxy or gateway), so only use it when clients cannot set
// that header themselves.
type header struct {
	name string
}

// NewHeader returns an [Authenticator] that reads the UserID from the named
// request header. It rejects a request whose header is absent or carries more
// than one value.
func NewHeader(name string) Authenticator {
	return &header{name: name}
}

// Authenticate implements [Authenticator]. It returns the single non-empty value of the
// configured header, or an error when the header is absent or has multiple
// values.
func (h *header) Authenticate(r *http.Request) (*Caller, error) {
	vals := r.Header.Values(h.name)
	switch {
	case len(vals) == 0:
		return nil, fmt.Errorf("%w: header not found", ErrUnauthenticated)
	case len(vals) > 1:
		return nil, fmt.Errorf("%w: header found with multiple values", ErrUnauthenticated)
	default:
		userID := vals[0]
		if userID == "" {
			return nil, fmt.Errorf("%w: header found with empty value", ErrUnauthenticated)
		}
		return &Caller{
			UserID: userID,
		}, nil
	}
}

var _ Authenticator = &header{}
