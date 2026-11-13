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

package runner

import (
	"context"
	"strconv"
	"sync/atomic"

	"google.golang.org/adk/v2/internal/adkcontext"
)

// executionCount numbers the runner calls of the process (see
// withExecutionID). A counter rather than platform.NewUUID leaves the UUIDs
// that a run draws, which tests pin with platform.WithUUIDProvider, as they
// were.
var executionCount atomic.Uint64

// withExecutionID returns ctx carrying a new ID of the runner call that ctx
// starts, which agent.ExecutionIDFromContext reads. Each Run, RunLive or
// node-path call sets it once, where it attaches the call's other values, so
// two calls that share an invocation ID, such as a paused run and the HITL
// resume that answers it, get different IDs.
func withExecutionID(ctx context.Context) context.Context {
	return context.WithValue(ctx, adkcontext.ExecutionIDKey, strconv.FormatUint(executionCount.Add(1), 10))
}
