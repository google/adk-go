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
	"context"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// liveOptions parametrize one RunLive call. The zero value is what a caller
// of Flow.RunLive gets: the run's own live settings and no response dropped.
type liveOptions struct {
	// live holds the live settings of the connection. Nil means the run's
	// own: RunLive reads runconfig.FromContext(ctx).Live and fails with "live
	// run config not found" when there is none.
	live *agent.LiveRunConfig
	// dropEmptyResponses drops a live response that has no content and no
	// control field, after the response processors have seen it, as
	// adk-python's postprocess_live_flow does (live/_live_llm_flow.py).
	// Without it, such a response is dropped only as it arrives from the
	// connection, before the response processors run.
	dropEmptyResponses bool
}

// liveOptionsKey keys liveOptions in a context. Keying by the Flow means a
// Flow other than the one a CFC call started (a nested agent's, a transfer
// target's) never sees the options.
type liveOptionsKey struct{ flow *Flow }

// withLiveOptions returns a copy of ctx that carries opts for f.RunLive. ctx
// is not changed.
func withLiveOptions(ctx agent.InvocationContext, f *Flow, opts liveOptions) agent.InvocationContext {
	return ctx.WithContext(context.WithValue(ctx, liveOptionsKey{flow: f}, opts))
}

// liveOptionsFrom returns the options withLiveOptions stored in ctx for f, or
// the zero value.
func liveOptionsFrom(ctx context.Context, f *Flow) liveOptions {
	opts, _ := ctx.Value(liveOptionsKey{flow: f}).(liveOptions)
	return opts
}

// callLLMOverLive serves one model call of a unary run over the model's live
// connection, as adk-python's call_llm_async does when support_cfc is set
// (flows/llm_flows/core/_model_call.py). The caller has already run the
// before-model callbacks, stamped the agent-name label and counted the call
// on req; RunLive builds its own request by running the request processors
// again, and runs the response processors and the tools itself.
//
// It yields each live event, wrapped for runOneStep to pass on without
// post-processing it again, after the after-model callbacks have seen it. An
// event with Partial set is yielded only in SSE mode. The call ends after
// the first event with TurnComplete set or with Actions.TransferToAgent set,
// when the live events end (a task_completed function response ends them),
// on a failure, or when yield returns false; the connection is closed
// whichever way it ends.
//
// Each yielded event is a copy of the live event with its own content, so an
// after-model callback's edit of a function call changes the event the
// consumer gets and not the call the live loop runs.
func (f *Flow) callLLMOverLive(ctx agent.InvocationContext, req *model.LLMRequest, actions *session.EventActions, yield func(*responseWithEventID, error) bool) {
	sess, events, err := f.RunLive(withLiveOptions(ctx, f, liveOptions{live: &agent.LiveRunConfig{}, dropEmptyResponses: true}))
	if err != nil {
		f.yieldLiveError(ctx, req, actions, err, yield)
		return
	}
	// Close ends RunLive's goroutine and closes the connection; it never
	// fails.
	defer func() { _ = sess.Close() }()
	sse := streamingMode(ctx) == agent.StreamingModeSSE
	for ev, err := range events {
		if err != nil {
			f.yieldLiveError(ctx, req, actions, err, yield)
			return
		}
		if ev == nil {
			continue
		}
		out := liveEventCopy(ctx, ev)
		cbResp, cbErr := f.runAfterModelCallbacks(ctx, &out.LLMResponse, actions, nil)
		if cbErr != nil {
			yield(nil, cbErr)
			return
		}
		if cbResp != nil {
			out.LLMResponse = *inheritModelFields(&out.LLMResponse, cbResp)
		}
		if sse || !out.Partial {
			if !yield(&responseWithEventID{LLMResponse: &out.LLMResponse, eventID: out.ID, liveEvent: out}, nil) {
				return
			}
		}
		if out.TurnComplete || out.Actions.TransferToAgent != "" {
			return
		}
	}
}

// liveEventCopy returns a copy of ev that owns its content, with the branch
// of ctx when ev has none. RunLive's goroutine reads ev after handing it
// over, to run its function calls, so ev itself is never written to. Live
// responses carry no branch; adk-python's model response event has it.
func liveEventCopy(ctx agent.InvocationContext, ev *session.Event) *session.Event {
	out := *ev
	out.LLMResponse.Content = clone(ev.LLMResponse.Content)
	if out.Branch == "" {
		out.Branch = ctx.Branch()
	}
	return &out
}

// yieldLiveError ends a CFC model call that failed with err. It runs the
// on-model-error callbacks, as adk-python wraps the live call in
// _run_and_handle_error, in the order of callLLM's unary loop: a callback's
// replacement gets client function call IDs and goes through the after-model
// callbacks, then is yielded as an ordinary response, which runOneStep
// post-processes like a unary one. Without a replacement err is yielded. It
// yields exactly one value.
func (f *Flow) yieldLiveError(ctx agent.InvocationContext, req *model.LLMRequest, actions *session.EventActions, err error, yield func(*responseWithEventID, error) bool) {
	cbResp, cbErr := f.runOnModelErrorCallbacks(ctx, req, actions, err)
	if cbErr != nil {
		yield(nil, cbErr)
		return
	}
	if cbResp == nil {
		yield(nil, err)
		return
	}
	// Function call ID is optional in genai API and some models do not use
	// the field; the after-model callbacks may read it, as on the unary path.
	utils.PopulateClientFunctionCallID(ctx, cbResp.Content)
	callbackResp, callbackErr := f.runAfterModelCallbacks(ctx, cbResp, actions, nil)
	if callbackErr != nil {
		yield(nil, callbackErr)
		return
	}
	if callbackResp != nil {
		cbResp = inheritModelFields(cbResp, callbackResp)
	}
	yield(newResponseWithEventID(ctx, cbResp), nil)
}
