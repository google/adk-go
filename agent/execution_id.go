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

package agent

import (
	"context"

	"google.golang.org/adk/v2/internal/adkcontext"
)

// ExecutionIDFromContext returns the ID that the runner assigned to the
// [google.golang.org/adk/v2/runner.Runner.Run] or
// [google.golang.org/adk/v2/runner.Runner.RunLive] call that ctx belongs to,
// and whether ctx carries one. It is the same in every callback of one call,
// and it differs between two calls that share an invocation ID, such as a
// paused run and the HITL resume that answers it. A nested runner call, such
// as the one an agent tool starts, is its own call with its own ID. A context
// that no runner call derived, such as context.Background(), carries none.
//
// The ID is opaque and unique only within the process. It is meant for
// keeping the state of runs apart, as a plugin that tracks each run does,
// and not for storage: persist the invocation ID instead.
func ExecutionIDFromContext(ctx context.Context) (string, bool) {
	// Recovered, as in IdentityFromContext: ctx may be nil, and any context
	// between the caller and the runner's supplies Value.
	v, _ := adkcontext.Recovered(func() any { return ctx.Value(adkcontext.ExecutionIDKey) })
	id, ok := v.(string)
	if !ok || id == "" {
		return "", false
	}
	return id, true
}
