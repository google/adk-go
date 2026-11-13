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
	"fmt"
	"slices"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/abortevents"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

// watchAbort returns the engine context of a run: ictx with a context that is
// canceled when the run's abort signal fires, and stop, which ends the watch.
// The agent or workflow runs under the engine context, while the runner keeps
// persisting and calling plugins with ictx, which the abort never cancels.
// Without an abort state on ictx it returns ictx and a no-op stop. stop may be
// called more than once.
func watchAbort(ictx agent.Context) (engine agent.Context, stop func()) {
	state := icontext.AbortStateFrom(ictx)
	if state == nil {
		return ictx, func() {}
	}
	abortCtx, stop := state.Watch(ictx)
	return ictx.WithAgentContext(abortCtx), stop
}

// sealAborted appends to sess the events that seal the aborted invocation of
// ctx, at most once per run, and returns them as the OnEventCallback plugins
// left them. ctx must be the run's own invocation context, not the engine
// context. It returns no events and no error when the run has no abort state,
// has not been aborted, or has already been sealed.
//
// Each event takes the path of an agent event: run metadata and
// OnEventCallback plugins, then AppendEvent when shouldAppendEvent holds. It
// runs under ctx without its cancellation, so a run whose caller has canceled
// ctx is still sealed. On a plugin error it returns the error unchanged and on
// an append error it wraps it; the events appended before the failure stay in
// the session.
func (r *Runner) sealAborted(ctx agent.Context, sess session.Session) ([]*session.Event, error) {
	state := icontext.AbortStateFrom(ctx)
	if !state.Aborted() || !state.ClaimSeal() {
		return nil, nil
	}
	sealCtx := ctx.WithAgentContext(context.WithoutCancel(ctx))
	built := abortevents.BuildAbortEvents(sealCtx, slices.Collect(sess.Events().All()), ctx.InvocationID(), r.invocationRoot().Name(), ctx.Branch())
	sealed := make([]*session.Event, 0, len(built))
	for _, ev := range built {
		ev, err := r.processEventWithPlugins(sealCtx, ev)
		if err != nil {
			return nil, err
		}
		if shouldAppendEvent(ev, false) {
			if err := r.sessionService.AppendEvent(sealCtx, sess, ev); err != nil {
				return nil, fmt.Errorf("failed to add event to session: %w", err)
			}
			ev = withoutTempState(ev)
		}
		sealed = append(sealed, ev)
	}
	return sealed, nil
}

// sealOnStop seals an aborted run whose caller stopped ranging. It is deferred
// by the run loops, so it yields nothing, and it drops a sealing error: the
// caller has left and a deferred call cannot return one. adk-python logs that
// error (runners.py, Runner._exec_with_plugin); library code here writes to no
// process-wide log.
func (r *Runner) sealOnStop(ctx agent.Context, sess session.Session) {
	if !icontext.Aborted(ctx) {
		return
	}
	_, _ = r.sealAborted(ctx, sess)
}

// yieldSealed is the seal of a run loop that ended without the caller
// stopping: it seals an aborted run and yields the sealing events, or the
// sealing error. It reports whether the run may go on, which it may not after
// a sealing error or when yield returns false.
func (r *Runner) yieldSealed(ctx agent.Context, sess session.Session, yield func(*session.Event, error) bool) bool {
	sealed, err := r.sealAborted(ctx, sess)
	if err != nil {
		yield(nil, err)
		return false
	}
	for _, ev := range sealed {
		if !yield(ev, nil) {
			return false
		}
	}
	return true
}
