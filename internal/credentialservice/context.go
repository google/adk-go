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

// Package credentialservice carries a runner's credential service on a
// [context.Context], from the runner to the agent calls that load and save
// credentials. It is internal so that only the runner decides which service
// an invocation has: a service a user puts on the context cannot replace it.
package credentialservice

import (
	"context"

	api "google.golang.org/adk/v2/credentialservice"
)

type serviceKey struct{}

// WithService returns a context derived from ctx that carries s, in place of
// any service ctx carries. A nil s is carried too, and then [FromContext]
// reports no service for the derived context.
func WithService(ctx context.Context, s api.Service) context.Context {
	return context.WithValue(ctx, serviceKey{}, s)
}

// FromContext returns the service the closest [WithService] attached to ctx
// or to a context ctx derives from. It returns nil when ctx is nil, when no
// service was attached, or when the attached service is nil.
func FromContext(ctx context.Context) api.Service {
	if ctx == nil {
		return nil
	}
	s, _ := ctx.Value(serviceKey{}).(api.Service)
	return s
}
