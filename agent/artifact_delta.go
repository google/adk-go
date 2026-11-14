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
	"errors"
	"fmt"
)

// artifactDeltaMerger is implemented by the contexts that carry, or wrap a
// context that carries, the event actions of a callback or a tool.
type artifactDeltaMerger interface {
	mergeArtifactDelta(delta map[string]int64) error
}

// MergeArtifactDelta records delta, a map from artifact file name to version,
// in the ArtifactDelta of the event actions that ctx carries. A version
// already recorded for a name is replaced. It is for code that has saved an
// artifact through some other context and must report the version on the
// event of ctx: within a before- or after-agent callback that is the
// callback's event, which the agent then emits because its artifact delta is
// not empty; within a tool, the function response event. MergeArtifactDelta
// neither saves nor checks anything in the artifact service, and it does not
// keep delta: changing delta afterwards does not change what was recorded.
//
// The contexts that can record are those built by [NewCallbackContext],
// [NewCallbackContextWithArtifactTracking] and [NewToolContext], and the
// contexts derived from them with WithDelta, WithAgentCancel or
// WithAgentTimeout. ADK passes such contexts to agent callbacks, model
// callbacks and tools. For any other ctx, MergeArtifactDelta returns an error
// wrapping [errors.ErrUnsupported], whatever delta holds, nil and empty
// included. Such contexts include one built by [NewContext], one built by
// [Promote] from an InvocationContext that is not a Context of ADK's own, a
// Context implementation of the caller's own, and the context a streaming
// tool receives in a live session. On a context that can record, a nil or
// empty delta changes nothing and returns nil.
//
// MergeArtifactDelta is not safe for concurrent use with other calls that
// change the same event actions.
func MergeArtifactDelta(ctx Context, delta map[string]int64) error {
	m, ok := ctx.(artifactDeltaMerger)
	if !ok {
		return fmt.Errorf("context of type %T cannot record an artifact delta: %w", ctx, errors.ErrUnsupported)
	}
	return m.mergeArtifactDelta(delta)
}
