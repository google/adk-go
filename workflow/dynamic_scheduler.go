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

	// mu protects runCountByChild.
	mu sync.Mutex
	// runCountByChild seeds the auto-counter for each child name —
	// the n-th invocation of any given name gets runID
	// strconv.Itoa(n).
	runCountByChild map[string]int
}

func newDynamicSubScheduler(parent NodeContext, parentPath string, emitUp func(*session.Event) error) *dynamicSubScheduler {
	return &dynamicSubScheduler{
		parentPath:      parentPath,
		parentCtx:       parent,
		emitUp:          emitUp,
		runCountByChild: map[string]int{},
	}
}

// runNode executes child once, forwards its events upstream, and
// classifies the outcome. HITL surfaces as ErrNodeInterrupted; runtime
// failure as ErrNodeFailed; a child that fails after requesting input
// surfaces as ErrNodeFailed.
//
// Session, invocation metadata, and cancellation come from s.parentCtx
// captured at sub-scheduler construction. opts carries the resolved
// per-call configuration assembled from RunNode's variadic
// RunNodeOption arguments: opts.customRunID is empty to use the
// auto-counter or a user-supplied stable id (validated against the
// rules in validateCustomRunID); opts.useSubBranch and
// opts.overrideBranch derive the child's Branch() via
// deriveChildBranch.
func (s *dynamicSubScheduler) runNode(child Node, input any, opts runNodeOptions) (any, error) {
	name := child.Name()
	runID, err := s.resolveRunID(name, opts.customRunID)
	if err != nil {
		return nil, &NodeRunError{ChildName: name, Cause: err}
	}
	childPath := s.parentPath + "/" + name + "@" + runID
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
		return nil, &NodeRunError{
			ChildName: name, ChildPath: childPath, RunID: runID,
			Cause: ErrNodeInterrupted,
		}
	}
	return out, nil
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
