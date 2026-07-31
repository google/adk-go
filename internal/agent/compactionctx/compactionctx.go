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

// Package compactionctx carries the context-compaction runtime from the runner
// down to the request processor that performs intra-invocation compaction.
//
// The processor needs both the compaction config and the session service, and
// [agent.InvocationContext] exposes neither. Adding them to that interface
// would break every external implementation of it, so the runtime rides on the
// context.Context instead, the same way parentmap, runconfig and plugininternal
// already do.
package compactionctx

import (
	"context"

	"google.golang.org/adk/v2/internal/compactioninternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// Runtime is everything intra-invocation compaction needs that the invocation
// context does not already provide.
type Runtime struct {
	// Config is the resolved compaction config, with its summarizer filled in.
	Config *compaction.Config
	// SessionService persists the summary event the compactor produces.
	SessionService session.Service
}

// Enabled reports whether rt can actually run a tail-retention compaction.
func (rt *Runtime) Enabled() bool {
	return rt != nil && rt.SessionService != nil && compactioninternal.HasTailRetention(rt.Config)
}

// ToContext returns a context carrying rt.
func ToContext(ctx context.Context, rt *Runtime) context.Context {
	return context.WithValue(ctx, runtimeCtxKey, rt)
}

// FromContext returns the [Runtime] carried by ctx, or nil when compaction is
// not configured.
func FromContext(ctx context.Context) *Runtime {
	rt, ok := ctx.Value(runtimeCtxKey).(*Runtime)
	if !ok {
		return nil
	}
	return rt
}

type ctxKey int

const runtimeCtxKey ctxKey = 0
