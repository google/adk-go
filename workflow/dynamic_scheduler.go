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
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"go.opentelemetry.io/otel/codes"

	"google.golang.org/adk/session"
)

// dynamicSubScheduler runs the children of one dynamic-node activation.
type dynamicSubScheduler struct {
	parentPath string
	parentCtx  NodeContext
	emitUp     func(*session.Event) error

	// mu guards everything below. Never held across child.Run.
	mu sync.Mutex
	// runCountByChild seeds the auto-counter per child name; the
	// n-th invocation gets runID strconv.Itoa(n).
	runCountByChild map[string]int
	// resultByPath caches successful child outputs keyed by
	// childPath ("<parentPath>/<name>@<runID>"). Failures and HITL
	// interrupts are not cached.
	resultByPath map[string]any
}

func newDynamicSubScheduler(parent NodeContext, parentPath string, emitUp func(*session.Event) error) *dynamicSubScheduler {
	return &dynamicSubScheduler{
		parentPath:      parentPath,
		parentCtx:       parent,
		emitUp:          emitUp,
		runCountByChild: map[string]int{},
		resultByPath:    map[string]any{},
	}
}

// runNode executes child once and classifies the outcome: HITL →
// ErrNodeInterrupted, runtime failure → ErrNodeFailed. A child that
// fails after requesting input surfaces as ErrNodeFailed. A repeated
// call with the same stable WithRunID returns the cached output
// without re-running the child; auto-counter ids never collide so
// the cache is effectively bypassed for them.
//
// Session, invocation metadata, and cancellation come from
// s.parentCtx. opts carries the resolved RunNodeOption arguments.
func (s *dynamicSubScheduler) runNode(child Node, input any, opts runNodeOptions) (result any, err error) {
	name := child.Name()
	runID, err := s.resolveRunID(name, opts.customRunID)
	if err != nil {
		return nil, &NodeRunError{ChildName: name, Cause: err}
	}
	childPath := s.parentPath + "/" + name + "@" + runID

	if cached, ok := s.lookupCachedOutput(childPath); ok {
		return cached, nil
	}

	childBranch := deriveChildBranch(s.parentCtx.Branch(), name, runID, opts.useSubBranch, opts.overrideBranch)
	childCtx := newDynamicNodeContext(s.parentCtx.WithBranch(childBranch), childPath, runID, s)

	// Emit an "invoke_node <name>" span nested under the dynamic
	// node's span (carried in s.parentCtx), so RunNode-driven
	// children appear in the trace tree. startNodeSpan returns a
	// *nodeContext because childCtx is one.
	span, spanCtx := startNodeSpan(childCtx, child)
	defer span.End()
	childCtx = spanCtx.(*nodeContext)

	// Record genuine runtime failures on the span. HITL pauses
	// (ErrNodeInterrupted) and parent cancellation (context.Canceled)
	// are expected control flow, not span errors — matching the top
	// scheduler's runNode, which skips context.Canceled.
	defer func() {
		if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, ErrNodeInterrupted) {
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
		}
	}()

	// EXPERIMENTAL: stash childCtx (a *nodeContext with non-nil
	// subScheduler) in the embedded context.Context so tools running
	// inside an LlmAgent that is itself running as this dynamic
	// child can recover the NodeContext via
	// workflow.NodeContextFromGoContext. See
	// scheduleResumedNode for the static-node equivalent.
	childCtx.InvocationContext = childCtx.InvocationContext.WithContext(
		WithNodeContext(childCtx.InvocationContext, childCtx),
	)

	var (
		out         any
		interrupted bool
	)
	for ev, evErr := range child.Run(childCtx, input) {
		if evErr != nil {
			// Child error wins over any prior interrupt.
			return nil, &NodeRunError{
				ChildName: name, ChildPath: childPath, RunID: runID,
				Cause: fmt.Errorf("%w: %v", ErrNodeFailed, evErr),
			}
		}
		if ev == nil {
			continue
		}
		// Stamp NodeInfo.Path so the top scheduler scopes the
		// child's Output/Routes to the child (not the parent's
		// accumulator). RequestedInput is promoted to the parent —
		// see scheduler.handleEvent. Skip if the child already
		// stamped NodeInfo (nested dynamic node yielding its own
		// terminal event, dynamic_node.go).
		if ev.NodeInfo == nil {
			ev.NodeInfo = &session.NodeInfo{Path: childPath}
		}
		if ev.RequestedInput != nil {
			interrupted = true
		}
		if ev.Output != nil {
			out = ev.Output
		}
		if err := s.emitUp(ev); err != nil {
			return nil, &NodeRunError{
				ChildName: name, ChildPath: childPath, RunID: runID,
				Cause: fmt.Errorf("%w: emitUp: %v", ErrNodeFailed, err),
			}
		}
	}

	if interrupted {
		// HITL is not terminal — parent re-runs on resume and is
		// expected to re-invoke RunNode. Do not cache.
		return nil, &NodeRunError{
			ChildName: name, ChildPath: childPath, RunID: runID,
			Cause: ErrNodeInterrupted,
		}
	}

	s.storeCachedOutput(childPath, out)
	return out, nil
}

func (s *dynamicSubScheduler) lookupCachedOutput(childPath string) (any, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out, ok := s.resultByPath[childPath]
	return out, ok
}

func (s *dynamicSubScheduler) storeCachedOutput(childPath string, out any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.resultByPath[childPath] = out
}

// resolveRunID validates a user-supplied id, or returns the next
// per-child-name counter value under lock.
func (s *dynamicSubScheduler) resolveRunID(childName, custom string) (string, error) {
	if custom != "" {
		if err := validateCustomRunID(custom); err != nil {
			return "", err
		}
		return custom, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.runCountByChild[childName]++
	return strconv.Itoa(s.runCountByChild[childName]), nil
}

// validateCustomRunID rejects empty ids, purely-numeric ids (collide
// with the auto-counter), and ids containing the composite-path
// separators / and @.
func validateCustomRunID(id string) error {
	if id == "" {
		return fmt.Errorf("%w: empty", ErrInvalidRunID)
	}
	if isAllDigits(id) {
		return fmt.Errorf("%w: %q is purely numeric (reserved for auto-counter)", ErrInvalidRunID, id)
	}
	if strings.ContainsAny(id, "/@") {
		return fmt.Errorf("%w: %q contains reserved separator (/ or @)", ErrInvalidRunID, id)
	}
	return nil
}

// isAllDigits checks ASCII digits only by design: the auto-counter
// emits ASCII digits, so collision is only possible with ASCII numeric
// ids.
func isAllDigits(s string) bool {
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
