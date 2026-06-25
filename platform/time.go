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
	"time"
)

// TimeProvider returns the current time. The default behavior is the wall
// clock (time.Now); callers can install a custom provider on a context
// with WithTimeProvider.
type TimeProvider func() time.Time

// timeProviderKey is the context key under which a TimeProvider is stored.
type timeProviderKey struct{}

// WithTimeProvider returns a copy of ctx that carries provider. Calls to Now
// with the returned context, or any context derived from it, use provider
// instead of the wall clock. A nil provider is ignored by Now, which then
// falls back to time.Now.
func WithTimeProvider(ctx context.Context, provider TimeProvider) context.Context {
	// TODO(kdroste): refactor underlying context
	return context.WithValue(ctx, timeProviderKey{}, provider)
}

// Now returns the current time. If ctx carries a TimeProvider installed with
// WithTimeProvider, that provider is used; otherwise Now falls back to
// time.Now.
//
// Now is the analog of google.adk.platform.time.get_time in adk-python.
func Now(ctx context.Context) time.Time {
	if ctx != nil {
		if p, ok := ctx.Value(timeProviderKey{}).(TimeProvider); ok && p != nil {
			return p()
		}
	}
	return time.Now()
}
