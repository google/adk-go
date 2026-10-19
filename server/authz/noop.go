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
)

// noop is an [Authorizer] which allows any [authn.Caller] to act as any userID
type noop struct{}

// NewNoop creates a new Noop [Authorizer] - allows any [authn.Caller] to act as any userID.
func NewNoop() Authorizer {
	return &noop{}
}

// CanActAsUser implements [Authorizer].
func (p *noop) CanActAsUser(_ context.Context, userID string) error {
	return nil
}

var _ Authorizer = &noop{}
