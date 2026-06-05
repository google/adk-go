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
	"fmt"
	"strconv"
	"strings"
	"sync"

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
	delegation   outputDelegation
}

// outputDelegation is the at-most-one WithUseAsOutput delegation for a
// parent activation. claim is set eagerly on the first delegating child
// and never cleared within the activation (matching adk-python's
// _output_delegated); a second delegating child is rejected. A fresh
// sub-scheduler is built per activation, so there is nothing to reset
// across turns. hasValue is the source of truth for readability because
// nil is a valid delegated value.
//
// Methods require the enclosing scheduler's mu to be held.
type outputDelegation struct {
	claimed   bool
	childPath string
	childName string
	value     any
	hasValue  bool
}

// reserve claims the delegation for childPath. Re-claiming the same
// childPath is a no-op (supports WithRunID replay). On conflict the
// existing holder's name is returned for error reporting.
func (d *outputDelegation) reserve(childPath, childName string) (existingName string, ok bool) {
	if d.claimed && d.childPath != childPath {
		return d.childName, false
	}
	d.claimed = true
	d.childPath = childPath
	d.childName = childName
	return "", true
}

// commit publishes value for the claiming child. Mismatched childPath is
// silently dropped rather than clobbering another child's delegation.
func (d *outputDelegation) commit(childPath string, value any) {
	if !d.claimed || d.childPath != childPath {
		return
	}
	d.value = value
	d.hasValue = true
}

func (d *outputDelegation) output() (any, bool) {
	return d.value, d.hasValue
}

func newDynamicSubScheduler(parent NodeContext, parentPath string, emitUp func(*session.Event) error) *dynamicSubScheduler {
	s := &dynamicSubScheduler{
		parentPath:      parentPath,
		parentCtx:       parent,
		emitUp:          emitUp,
		runCountByChild: map[string]int{},
		resultByPath:    map[string]any{},
	}
	s.rehydrateCache()
	return s
}

// rehydrateCache repopulates resultByPath from session history so a
// resumed orchestrator (which re-runs from the top) serves already
// completed children from cache instead of re-executing them. Each
// child's terminal event carries its childPath in NodeInfo.Path and a
// non-nil Output; keyed by childPath, so only stable WithRunID calls
// hit (auto-counter ids regenerate per activation and miss).
func (s *dynamicSubScheduler) rehydrateCache() {
	sess := s.parentCtx.Session()
	if sess == nil {
		return
	}
	prefix := s.parentPath + "/"
	s.mu.Lock()
	defer s.mu.Unlock()
	for ev := range sess.Events().All() {
		if ev == nil || ev.Output == nil || ev.NodeInfo == nil {
			continue
		}
		if !strings.HasPrefix(ev.NodeInfo.Path, prefix) {
			continue
		}
		// Last write wins, matching live execution order.
		s.resultByPath[ev.NodeInfo.Path] = ev.Output
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
func (s *dynamicSubScheduler) runNode(child Node, input any, opts runNodeOptions) (any, error) {
	name := child.Name()
	runID, err := s.resolveRunID(name, opts.customRunID)
	if err != nil {
		return nil, &NodeRunError{ChildName: name, Cause: err}
	}
	childPath := s.parentPath + "/" + name + "@" + runID

	// Claim before child.Run so a sibling WithUseAsOutput fails fast
	// rather than after the child finishes. The claim is not rolled back
	// on interrupt/failure: the orchestrator body ends on the sentinel
	// and resume rebuilds a fresh sub-scheduler.
	if err := s.claimDelegation(childPath, name, opts.useAsOutput); err != nil {
		return nil, err
	}

	// Cached (WithRunID replay): the child already ran, so publish its
	// output for the delegation immediately.
	if cached, ok := s.lookupCachedOutput(childPath); ok {
		s.commitDelegation(childPath, cached)
		return cached, nil
	}

	childBranch := deriveChildBranch(s.parentCtx.Branch(), name, runID, opts.useSubBranch, opts.overrideBranch)
	childCtx := newDynamicNodeContext(s.parentCtx.WithBranch(childBranch), childPath, runID, s)

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
		// A delegated child's output is re-emitted on the parent's
		// terminal event, so drop it here to avoid a duplicate.
		if childOut, ok := childEventOutput(ev); ok {
			out = childOut
			if opts.useAsOutput {
				continue
			}
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
	s.commitDelegation(childPath, out) // no-op unless this child claimed the delegation
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

// claimDelegation reserves the at-most-one output delegation when
// useAsOutput is set, mapping a conflict to NodeRunError. It is a no-op
// (nil) when useAsOutput is false.
func (s *dynamicSubScheduler) claimDelegation(childPath, childName string, useAsOutput bool) error {
	if !useAsOutput {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	existing, ok := s.delegation.reserve(childPath, childName)
	if !ok {
		return &NodeRunError{
			ChildName: childName,
			ChildPath: childPath,
			Cause: fmt.Errorf("%w: %s already delegates to %s",
				ErrOutputAlreadyDelegated, s.parentPath, existing),
		}
	}
	return nil
}

func (s *dynamicSubScheduler) commitDelegation(childPath string, value any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.delegation.commit(childPath, value)
}

func (s *dynamicSubScheduler) delegatedOutput() (any, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.delegation.output()
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
