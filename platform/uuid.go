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

package platform

import (
	"context"

	"github.com/google/uuid"
)

// UUIDProvider returns a new unique identifier. The default behavior is to
// return a random UUIDv4 (uuid.NewString); callers can install a custom
// provider on a context with WithUUIDProvider.
type UUIDProvider func() string

// uuidProviderKey is the context key under which a UUIDProvider is stored.
type uuidProviderKey struct{}

// WithUUIDProvider returns a copy of ctx that carries provider. Calls to
// NewUUID with the returned context, or any context derived from it, use
// provider instead of generating a random UUID. A nil provider is ignored by
// NewUUID, which then falls back to uuid.NewString.
func WithUUIDProvider(ctx context.Context, provider UUIDProvider) context.Context {
	// TODO(kdroste): move to invocationContext?
	return context.WithValue(ctx, uuidProviderKey{}, provider)
}

// NewUUID returns a new unique identifier. If ctx carries a UUIDProvider
// installed with WithUUIDProvider, that provider is used; otherwise NewUUID
// falls back to a random UUIDv4 (uuid.NewString).
//
// NewUUID is the analog of google.adk.platform.uuid.new_uuid in adk-python,
// with the provider read from ctx rather than from a contextvar.
func NewUUID(ctx context.Context) string {
	if ctx != nil {
		if p, ok := ctx.Value(uuidProviderKey{}).(UUIDProvider); ok && p != nil {
			return p()
		}
	}
	return uuid.NewString()
}
