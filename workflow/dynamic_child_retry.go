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

package workflow

import (
	"errors"
	"fmt"

	"google.golang.org/adk/v2/agent"
)

// childCall holds what one leading RunNode call fixes before its first
// attempt: the child, its validated input, run id and path, branch and
// scope, the delegation claim's result and the resolved options. Read-only
// after construction, so attempts share it without a lock.
type childCall struct {
	child          Node
	input          any
	opts           runNodeOptions
	name           string
	runID          string
	childPath      string
	own            *agent.StateValidator
	childScope     string
	childAncestors []string
	delta          *agent.CommonContextDelta
}

// attemptResult is one attempt's outcome. The zero value is a completed
// attempt with a nil output and no recorded failure.
type attemptResult struct {
	out               any              // output of a completed attempt
	err               error            // what the call returns if this attempt is the last
	failure           error            // the child's own failure when RetryConfig may run it again; nil for success, pause and final failure
	skipSummarization bool             // the attempt's ctx.Actions().SkipSummarization
	inventory         failureInventory // the attempt's failed-children inventory, for finishRun
	failureRecorded   bool             // the failure record reached the consumer
}

// context returns the context of attempt n of the call: the parent's with the
// call's delta (AttemptCount n).
func (c *childCall) context(parent agent.Context, n int) agent.Context {
	d := *c.delta
	d.AttemptCount = &n
	return parent.WithDelta(&d)
}

// retryChild is the retry decision between attempts. It returns false when
// a.failure is nil (completed, paused or final failure) or ShouldRetry
// refuses it. Otherwise, unless the consumer stopped taking events while the
// dynamic node's context is alive, it applies the sub-children reset (when the
// failure is a failed sub-child), waits CalculateDelay(cfg, n) on
// s.parentCtx and returns true. When that wait ends because s.parentCtx is
// done it returns false and the error the call returns instead of the
// attempt's: a *NodeRunError wrapping ErrNodeFailed and the context's error.
func (s *dynamicSubScheduler) retryChild(c *childCall, a attemptResult, n int) (retry bool, cancelErr error) {
	cfg := c.child.Config().RetryConfig
	if a.failure == nil || !ShouldRetry(cfg, a.failure, n) {
		return false, nil
	}
	// A consumer that stopped taking events must not get another run.
	if !a.failureRecorded && s.parentCtx.Err() == nil {
		return false, nil
	}
	// A sub-child of the child failed the attempt: the retried child starts
	// its own children at run id 1 again, so completed ones replay from
	// history. Any other error keeps their count going, so the retry runs
	// them again under new run ids. Mirrors adk-python's
	// DynamicNodeFailError branch (workflow/_node_runner.py).
	var nre *NodeRunError
	if errors.As(a.failure, &nre) && errors.Is(nre, ErrNodeFailed) {
		s.counters.reset(c.childPath)
	}
	if err := waitRetryDelay(s.parentCtx, CalculateDelay(cfg, n)); err != nil {
		return false, &NodeRunError{
			ChildName: c.name, ChildPath: c.childPath, RunID: c.runID,
			Cause: fmt.Errorf("%w: %w", ErrNodeFailed, err),
		}
	}
	return true, nil
}
