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
	"net/http"
)

// AuthenticatorFunc is the function which can be used by [NewCustom] to create a custom [Authenticator]
type AuthenticatorFunc func(r *http.Request) (*Caller, error)

// custom is an Authenticator which allows you to implement your own logic for authentication
type custom struct {
	AuthenticatorFunc AuthenticatorFunc
}

// NewCustom returns a custom [Authenticator], using the provided function as a part of middleware
func NewCustom(authFunc AuthenticatorFunc) Authenticator {
	return &custom{
		AuthenticatorFunc: authFunc,
	}
}

// Authenticate implements [Authenticator].
func (c *custom) Authenticate(r *http.Request) (*Caller, error) {
	if c.AuthenticatorFunc == nil {
		return nil, ErrUnauthenticated
	}
	return c.AuthenticatorFunc(r)
}

var _ Authenticator = &custom{}
