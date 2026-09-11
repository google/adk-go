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

package authz

import (
	"context"
	"fmt"

	"google.golang.org/adk/v2/server/authn"
)

// strict is an [Authorizer] which ensures exact match between
// the authenticated user and the user from the payload
type strict struct{}

// NewStrict returns an [Authorizer] which ensures exact match between
// the authenticated user and the user from the payload
func NewStrict() Authorizer {
	return &strict{}
}

// CanActAsUser implements [Authorizer].
func (p *strict) CanActAsUser(ctx context.Context, userID string) error {
	identity, ok := authn.IdentityFromContext(ctx)
	if !ok {
		return fmt.Errorf("%w: no identity found in context", ErrUnauthorized)
	}
	if identity.UserID == userID {
		return nil
	}
	return fmt.Errorf("%w: identity doesn't match the provided userID", ErrUnauthorized)
}

var _ Authorizer = &strict{}
