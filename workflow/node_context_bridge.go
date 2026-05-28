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

package workflow

import "context"

// EXPERIMENTAL: temporary bridge that lets runnable tools recover
// the surrounding NodeContext from a tool.Context value chain
// without modifying the tool.Context interface. Will be removed
// once the CallbackContext / ToolContext unification (see TODO in
// node_context.go) lands.
type nodeContextKey struct{}

// WithNodeContext returns a derived Go context that carries nc under
// an opaque key, recoverable via NodeContextFromGoContext from any
// descendant context.Context. The scheduler calls this on every
// per-node activation.
func WithNodeContext(parent context.Context, nc NodeContext) context.Context {
	if parent == nil || nc == nil {
		return parent
	}
	return context.WithValue(parent, nodeContextKey{}, nc)
}

// NodeContextFromGoContext returns the NodeContext stashed by
// WithNodeContext, or (nil, false) if none is present on ctx.
func NodeContextFromGoContext(ctx context.Context) (NodeContext, bool) {
	if ctx == nil {
		return nil, false
	}
	nc, ok := ctx.Value(nodeContextKey{}).(NodeContext)
	return nc, ok
}
