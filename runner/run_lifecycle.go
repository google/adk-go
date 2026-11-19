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
	"errors"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/plugininternal"
	"google.golang.org/adk/v2/session"
)

// runLifecycle is the run-level plugin lifecycle of one invocation. It decides
// whether the run ends because of a run error, calls the OnRunErrorCallback
// plugins once with that error, and otherwise runs the after-run callbacks
// once, as adk-python's Runner._exec_with_plugin does with its except and
// finally clauses (runners.py).
//
// A run ends because of a run error when the caller stops ranging at a
// yielded error, or when the runner itself stops on one (a stop site, which
// calls stop). A caller that ranges past an error where the runner goes on
// does not end the run with it: adk-go keeps running, and if nothing else
// fails the after-run callbacks run at the end. adk-python has no such caller.
//
// Each run has exactly one runLifecycle, created by the function that ranges
// the invocation's events (run, runNode or RunLive), and only the goroutine
// that runs the iterator body touches it, so it has no lock.
type runLifecycle struct {
	pm   *plugininternal.PluginManager
	ictx agent.InvocationContext // the context errors are reported with

	open         bool  // the run proper has started; set by start
	last         error // the last yielded item's error, when the caller ranged past it
	notified     bool  // OnRunErrorCallback ran: the run ended because of a run error
	keepAfterRun bool  // the before-run short-circuit, which runs after-run even when notified
	finished     bool  // finish ran the after-run callbacks, or decided not to
}

func newRunLifecycle(pm *plugininternal.PluginManager) *runLifecycle {
	return &runLifecycle{pm: pm}
}

// bind stores the invocation context that errors are reported with. It is
// called as soon as that context exists, so that a stage error raised before
// start is reported with it.
func (lc *runLifecycle) bind(ictx agent.InvocationContext) {
	lc.ictx = ictx
}

// start binds ictx and opens the run: from now on every yielded error is a
// run error, and finish runs the after-run callbacks.
func (lc *runLifecycle) start(ictx agent.InvocationContext) {
	lc.bind(ictx)
	lc.open = true
}

// yielded is called after the caller's yield returned continued for an item
// whose error is err. A caller that stopped at an error ends the run with it.
func (lc *runLifecycle) yielded(err error, continued bool) {
	lc.last = nil
	if err == nil {
		return
	}
	if !continued {
		lc.end(err)
		return
	}
	lc.last = err
}

// wrap returns a yield that calls yield and then yielded with its result.
func (lc *runLifecycle) wrap(yield func(*session.Event, error) bool) func(*session.Event, error) bool {
	return func(ev *session.Event, err error) bool {
		ok := yield(ev, err)
		lc.yielded(err, ok)
		return ok
	}
}

// stop is called at a stop site, after the runner yielded an error it does
// not go on from. If the caller ranged past that error, the run ends with it
// now; if the caller stopped at it, yielded already ended the run.
func (lc *runLifecycle) stop() {
	if lc.last != nil {
		lc.end(lc.last)
	}
}

// end records that the run ends because of err and calls every plugin's
// OnRunErrorCallback with it, at most once per run. A setup failure before
// start that is not a stage error, and the caller's cancellation, are not run
// errors and are ignored.
func (lc *runLifecycle) end(err error) {
	if lc.pm == nil || lc.notified {
		return
	}
	if !lc.open && !asStageError(err) {
		return
	}
	if agentinternal.IsCallerCancellation(lc.ictx, err) {
		return
	}
	lc.notified = true
	lc.pm.RunOnRunErrorCallback(lc.ictx, err)
}

// shortCircuit marks the run as ended at the before-run callback, where the
// after-run callbacks run even after a run error.
func (lc *runLifecycle) shortCircuit() {
	lc.keepAfterRun = true
}

// finish runs the after-run callbacks once, unless the run has not started or
// it ended because of a run error. It reports whether the run may go on,
// which it may not after an after-run failure.
//
// With a non-nil yield, which must be the run's yield wrapper, an after-run
// failure is yielded to the caller and ends the run. With a nil yield, as
// when deferred after the caller stopped ranging or while a panic unwinds,
// nothing can be yielded: the failure goes to the OnRunErrorCallback plugins
// and to the invocation span (adk-python logs it, runners.py; library code
// here writes to no process-wide log).
func (lc *runLifecycle) finish(yield func(*session.Event, error) bool) bool {
	if lc.finished || !lc.open {
		return true
	}
	lc.finished = true
	if lc.pm == nil || (lc.notified && !lc.keepAfterRun) {
		return true
	}
	err := lc.pm.RunAfterRunCallback(lc.ictx)
	if err == nil {
		return true
	}
	if yield != nil {
		yield(nil, err)
		lc.end(err)
		return false
	}
	lc.end(err)
	plugininternal.RecordDiagnostic(lc.ictx, err)
	return false
}

// runStageError marks an error from setup code that runs before the run
// starts but is a run error all the same: the on-user-message callback's
// failure and, on the node path, the user event's append failure (adk-python
// runs both inside the notified try on the node path,
// workflow/_node_runner_utils.py). Its text and Unwrap are those of err, so
// callers see no difference.
type runStageError struct{ err error }

func (e *runStageError) Error() string { return e.err.Error() }

func (e *runStageError) Unwrap() error { return e.err }

func asStageError(err error) bool {
	var stageErr *runStageError
	return errors.As(err, &stageErr)
}
