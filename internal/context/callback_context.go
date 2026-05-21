// Copyright 2025 Google LLC
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

package context

import (
	"google.golang.org/adk/agent"
)

// NewCallbackContext returns a CallbackContext suitable for model, tool, and
// related callbacks. The returned context's Artifacts().Save tracks each saved
// artifact's version into the underlying EventActions.ArtifactDelta.
func NewCallbackContext(ctx agent.InvocationContext) agent.CallbackContext {
	return agent.NewCallbackContext(ctx, true, nil, nil)
}

// NewCallbackContextWithDelta returns a CallbackContext that uses the given
// stateDelta and artifactDelta maps as the initial backing storage for its
// EventActions. The returned context's Artifacts().Save tracks each saved
// artifact's version into artifactDelta.
func NewCallbackContextWithDelta(ctx agent.InvocationContext, stateDelta map[string]any, artifactDelta map[string]int64) agent.CallbackContext {
	return agent.NewCallbackContext(ctx, true, stateDelta, artifactDelta)
}
