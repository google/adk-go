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

package llminternal

import (
	"fmt"
	"iter"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/compactionctx"
	"google.golang.org/adk/v2/internal/compactioninternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// CompactionRequestProcessor runs token-threshold tail-retention compaction
// before the conversation history is assembled for a model call.
//
// It must sit before [ContentsRequestProcessor] in the chain: the summary it
// appends only shrinks this request if contents are built afterwards.
//
// Unlike the runner's post-invocation sliding-window pass, this runs mid-turn,
// so it can react to a single long-running invocation that inflates the prompt
// rather than waiting for the turn to finish. It leaves the request itself
// untouched and emits no events.
func CompactionRequestProcessor(ctx agent.InvocationContext, _ *model.LLMRequest, _ *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		rt := compactionctx.FromContext(ctx)
		if !rt.Enabled() {
			return
		}
		sess := ctx.Session()
		if sess == nil {
			return
		}

		summary, err := compactioninternal.TailRetention(ctx, rt.Config, sess, promptTokenEstimator(ctx))
		if err != nil {
			// Surfaced rather than swallowed, unlike the post-invocation pass.
			// Compaction fires here precisely because the prompt is already
			// near the context limit, so continuing would most likely fail the
			// model call anyway, with a far less informative error.
			yield(nil, fmt.Errorf("token-threshold compaction failed: %w", err))
			return
		}
		if summary == nil {
			return
		}
		if err := rt.SessionService.AppendEvent(ctx, sess, summary); err != nil {
			yield(nil, fmt.Errorf("failed to append the compaction event to the session: %w", err))
		}
	}
}

// promptTokenEstimator returns a [compactioninternal.TokenCounter] that approximates
// the prompt size for ctx's agent.
//
// It is only consulted before any model response has reported a real token
// count. Building the contents the same way the request will is what makes the
// estimate meaningful: it sees branch and isolation-scope filtering, and any
// compaction already applied.
func promptTokenEstimator(ctx agent.InvocationContext) compactioninternal.TokenCounter {
	return func(events []*session.Event) int {
		llmAgent := asLLMAgent(ctx.Agent())
		if llmAgent == nil {
			return 0
		}
		state := llmAgent.internal()
		contents, err := buildContentsDefault(
			ctx.Agent().Name(),
			ctx.Branch(),
			ctx.IsolationScope(),
			events,
			state.Mode == ModeSingleTurn,
			ctx.UserContent(),
		)
		if err != nil {
			// An unbuildable history is the contents processor's problem to
			// report a moment from now; here it just means "no estimate".
			return 0
		}

		return compactioninternal.EstimateTokensFromContents(contents)
	}
}
