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
	"iter"
	"sync"

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
	// waitForYield holds pushEvent on each non-partial event until recvIter's
	// yield returns true, so a CFC step finishes AfterModelCallback,
	// OnEventCallback and AppendEvent before RunLive runs any function calls.
	waitForYield bool
}

type liveEventGateKey struct{}

// liveEventGate coordinates non-partial live events pushed by Flow.RunLive
// with Runner.RunLive, so plugins (such as toolcallintegrity) and session
// persistence finish handling a function call event before handleFunctionCalls
// starts, without blocking outer iter.Pull2 callers across yield.
type liveEventGate struct {
	mu    sync.Mutex
	gates map[*session.Event]chan bool
}

// WithLiveEventGate returns a copy of ctx carrying a gate that holds each
// non-partial Flow.RunLive event until ReleaseLiveEvent releases it.
func WithLiveEventGate(ctx context.Context) context.Context {
	return context.WithValue(ctx, liveEventGateKey{}, &liveEventGate{
		gates: make(map[*session.Event]chan bool),
	})
}

func liveEventGateFrom(ctx context.Context) *liveEventGate {
	g, _ := ctx.Value(liveEventGateKey{}).(*liveEventGate)
	return g
}

func (g *liveEventGate) hold(ev *session.Event) chan bool {
	if g == nil || ev == nil || ev.LLMResponse.Partial {
		return nil
	}
	ch := make(chan bool, 1)
	g.mu.Lock()
	g.gates[ev] = ch
	g.mu.Unlock()
	return ch
}

func (g *liveEventGate) release(ev *session.Event, ok bool) {
	if g == nil || ev == nil {
		return
	}
	g.mu.Lock()
	ch, exists := g.gates[ev]
	delete(g.gates, ev)
	g.mu.Unlock()
	if exists {
		ch <- ok
	}
}

// ReleaseLiveEvent releases ev on the gate attached by WithLiveEventGate. ok
// reports whether the runner accepted ev and RunLive may continue processing
// its function calls. Calling ReleaseLiveEvent more than once or on an untracked
// event is a no-op.
func ReleaseLiveEvent(ctx context.Context, ev *session.Event, ok bool) {
	liveEventGateFrom(ctx).release(ev, ok)
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
// post-processing it again. The after-model callbacks see every event except
// the ones a request or response processor yields, which are yielded as the
// processor made them: without CFC the callbacks never see a processor's
// event either; adk-python runs them on those events too. An event with
// Partial set is yielded only in SSE mode. The call ends after the first event
// with TurnComplete set or with Actions.TransferToAgent set, when the live
// events end (a task_completed function response ends them), on a failure, or
// when yield returns false; the connection is closed whichever way it ends.
//
// Each yielded event is a copy of the live event with its own content, so an
// after-model callback's edit of a function call changes the event the
// consumer gets and not the call the live loop runs.
func (f *Flow) callLLMOverLive(ctx agent.InvocationContext, req *model.LLMRequest, actions *session.EventActions, yield func(*responseWithEventID, error) bool) {
	marks := &processorEvents{set: make(map[*session.Event]struct{})}
	lf := f.withMarkedProcessors(marks)
	sess, events, err := lf.RunLive(withLiveOptions(ctx, lf, liveOptions{
		live:               &agent.LiveRunConfig{},
		dropEmptyResponses: true,
		waitForYield:       true,
	}))
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
		if !marks.take(ev) {
			cbResp, cbErr := f.runAfterModelCallbacks(ctx, &out.LLMResponse, actions, nil)
			if cbErr != nil {
				yield(nil, cbErr)
				return
			}
			if cbResp != nil {
				out.LLMResponse = *inheritModelFields(&out.LLMResponse, cbResp)
			}
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

// processorEvents is the set of events that the request and response
// processors of one CFC model call have yielded. RunLive's goroutine adds to
// it and callLLMOverLive takes from it, so mu guards set.
type processorEvents struct {
	mu  sync.Mutex
	set map[*session.Event]struct{}
}

// mark returns seq, adding each event to p before it is yielded.
func (p *processorEvents) mark(seq iter.Seq2[*session.Event, error]) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for ev, err := range seq {
			p.mu.Lock()
			p.set[ev] = struct{}{}
			p.mu.Unlock()
			if !yield(ev, err) {
				return
			}
		}
	}
}

// take reports whether ev is in p, and removes it.
func (p *processorEvents) take(ev *session.Event) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	_, ok := p.set[ev]
	delete(p.set, ev)
	return ok
}

// withMarkedProcessors returns a copy of f whose request and response
// processors add each event they yield to marks, so that a CFC model call can
// tell those events from the model's. RunLive pushes the processors' events
// themselves, so the pointer identifies them. f is not changed.
func (f *Flow) withMarkedProcessors(marks *processorEvents) *Flow {
	lf := *f
	lf.RequestProcessors = make([]func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error], len(f.RequestProcessors))
	for i, p := range f.RequestProcessors {
		lf.RequestProcessors[i] = func(ctx agent.InvocationContext, req *model.LLMRequest, pf *Flow) iter.Seq2[*session.Event, error] {
			return marks.mark(p(ctx, req, pf))
		}
	}
	lf.ResponseProcessors = make([]func(agent.InvocationContext, *model.LLMRequest, *model.LLMResponse) iter.Seq2[*session.Event, error], len(f.ResponseProcessors))
	for i, p := range f.ResponseProcessors {
		lf.ResponseProcessors[i] = func(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) iter.Seq2[*session.Event, error] {
			return marks.mark(p(ctx, req, resp))
		}
	}
	return &lf
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
