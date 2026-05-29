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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/session"
)

// dynamicSubScheduler runs the children of one dynamic-node activation.
type dynamicSubScheduler struct {
	parentPath string
	parentCtx  NodeContext
	emitUp     func(*session.Event) error

	// outputForAncestors are the delegating-ancestor paths this
	// activation's output also counts for, set when this dynamic node is
	// itself a WithUseAsOutput child. Mirrors adk-python's
	// Context._output_for_ancestors.
	outputForAncestors []string

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

// ResolveByRunID implements [agent.DynamicSubScheduler].
func (s *dynamicSubScheduler) ResolveByRunID(childName, custom string) (string, error) {
	return s.resolveRunID(childName, custom)
}

// DelegatedOutput implements [agent.DynamicSubScheduler].
func (s *dynamicSubScheduler) DelegatedOutput() (any, bool) {
	return s.delegatedOutput()
}

// OutputForAncestors implements [agent.DynamicSubScheduler].
func (s *dynamicSubScheduler) OutputForAncestors() []string {
	return s.outputForAncestors
}

// ParentPath implements [agent.DynamicSubScheduler].
func (s *dynamicSubScheduler) ParentPath() string {
	return s.parentPath
}

// RunNode implements [agent.DynamicSubScheduler].
func (s *dynamicSubScheduler) RunNode(child, input, opts any) (any, error) {
	childNode, ok := child.(Node)
	if !ok {
		return nil, fmt.Errorf("got child %T, want Node", child)
	}
	options, ok := opts.(runNodeOptions)
	if !ok {
		return nil, fmt.Errorf("got opts %T, want runNodeOptions", opts)
	}
	return s.runNode(childNode, input, options)
}

var _ agent.DynamicSubScheduler = (*dynamicSubScheduler)(nil)

// outputDelegation is the at-most-one WithUseAsOutput delegation for a
// parent activation. claim is set eagerly on the first delegating child
// and never cleared within the activation (matching adk-python's
// _output_delegated); a second delegating child is rejected. hasValue
// (not value != nil) is the source of truth, since nil is a valid
// delegated value.
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

func newDynamicSubScheduler(parent agent.Context, parentPath string, emitUp func(*session.Event) error) agent.DynamicSubScheduler {
	ancestors := []string{}
	if parent != nil {
		ancestors = parent.OutputForAncestors()
	}
	s := &dynamicSubScheduler{
		parentPath:         parentPath,
		parentCtx:          parent,
		emitUp:             emitUp,
		outputForAncestors: ancestors,
		runCountByChild:    map[string]int{},
		resultByPath:       map[string]any{},
	}
	s.rehydrateCache()
	return s
}

// rehydrateCache repopulates resultByPath from session history so a
// resumed orchestrator (which re-runs from the top) serves already
// completed children from cache instead of re-executing them. Each
// child's terminal event carries its childPath in NodeInfo.Path and a
// non-nil Output; keyed by childPath.
//
// Only events from the current invocation are considered. Auto-counter
// run-ids reset to 1 on every fresh activation, so a later user turn
// reuses the same childPath ("<parent>/<child>@1") as a prior turn;
// without the invocation filter those stale outputs would be served
// from cache and the child would never re-run on the new turn. Mirrors
// adk-python, which gates rehydration on event.invocation_id (see
// _reconstruct_node_states / _scan_parent_child_sequence).
func (s *dynamicSubScheduler) rehydrateCache() {
	sess := s.parentCtx.Session()
	if sess == nil {
		return
	}
	invocationID := s.parentCtx.InvocationID()
	prefix := s.parentPath + "/"
	s.mu.Lock()
	defer s.mu.Unlock()
	for ev := range sess.Events().All() {
		if ev == nil || ev.Output == nil || ev.NodeInfo == nil {
			continue
		}
		if invocationID != "" && ev.InvocationID != invocationID {
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
func (s *dynamicSubScheduler) runNode(child Node, input any, opts runNodeOptions) (out any, err error) {
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

	childBranch := deriveChildBranch(s.parentCtx.Branch(), name, runID, opts.useSubBranch, opts.overrideBranch)
	// A delegating child extends the chain: its own delegating children
	// must count their output for this parent and its ancestors too.
	var childAncestors []string
	if opts.useAsOutput {
		childAncestors = append([]string{s.parentPath}, s.outputForAncestors...)
	}
	childCtx := agent.NewDynamicNodeContext(s.parentCtx.WithBranch(childBranch), childPath, runID, s, childAncestors)
	// logContext(childCtx, "childCtx after newDynamicNodeContext", 0)

	// Explicit scope wins over the node-path default; absent both,
	// inherit. Matches adk-python _compute_isolation_scope_for_node.
	childScope := childCtx.IsolationScope()
	if opts.overrideIsolationScope != "" {
		childScope = opts.overrideIsolationScope
	} else if opts.scopeFromNodePath {
		childScope = childPath
	}
	childCtx = withIsolationScope(childCtx, childScope)
	///childCtx.SetInvocationContext(iCtx)
	// logContext(childCtx, "childCtx after withIsolationScope", 0)
	//	log.Printf("childCtx: %+v branch: %v", childCtx, childCtx.Branch())

	// Emit an "invoke_node <name>" span nested under the dynamic
	// node's span (carried in s.parentCtx), so RunNode-driven
	// children appear in the trace tree. The span is opened before the
	// cache lookup so a cached (WithRunID replay) hit is still emitted
	// as its own span. startNodeSpan returns a context carrying the span.
	span, spanCtx := startNodeSpan(childCtx, child)
	defer span.End()
	childCtx = spanCtx

	// Record genuine runtime failures on the span. Pauses (HITL
	// ErrNodeInterrupted, WaitForOutput ErrNodeWaitingForOutput) and
	// parent cancellation (context.Canceled) are expected control
	// flow, not span errors — matching the top scheduler's runNode.
	defer func() {
		if err != nil &&
			!errors.Is(err, context.Canceled) &&
			!errors.Is(err, ErrNodeInterrupted) &&
			!errors.Is(err, ErrNodeWaitingForOutput) {
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
		}
	}()

	// Cached (WithRunID replay): the child already ran, so publish its
	// output for the delegation immediately. The span opened above still
	// records the cache hit.
	if cached, ok := s.lookupCachedOutput(childPath); ok {
		s.commitDelegation(childPath, cached)
		return cached, nil
	}

	// EXPERIMENTAL: stash childCtx (a *nodeContext with non-nil
	// subScheduler) in the embedded context.Context so tools running
	// inside an LlmAgent that is itself running as this dynamic
	// child can recover the NodeContext via
	// workflow.NodeContextFromGoContext. See
	// scheduleResumedNode for the static-node equivalent.

	// ctxWithValue := WithNodeContext(childCtx.InvocationContext(), childCtx)
	// logContext(ctxWithValue, "iCtx3", 0)

	// log.Printf("iCtx3: %+v branch: n/a", ctxWithValue)
	// iCtx2 := childCtx.WithContext(ctxWithValue)
	// logContext(iCtx2, "iCtx2", 0)
	// log.Printf("iCtx2: %+v branch: %v", iCtx2, iCtx2.Branch())
	// childCtx.SetInvocationContext(iCtx2)
	// log.Printf("final childCtx: %+v branch: %v", childCtx, childCtx.Branch())
	// // childCtx= iCtx3

	var (
		hasOutput   bool
		interrupted bool
		// pendingLongRunningIDs collects FunctionCall IDs the child
		// emitted as long-running (listed in the emitting event's
		// LongRunningToolIDs). Each is removed when we later see a
		// matching FunctionResponse from the child. Any IDs left
		// at the end of the child's iteration represent tools the
		// child is still waiting on — used by WithRaiseOnWait to
		// distinguish "child paused for HITL" from "child finished
		// cleanly with no output".
		pendingLongRunningIDs map[string]struct{}
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
		// see scheduler.handleEvent. A child may set NodeInfo without
		// a Path (e.g. MessageAsOutput), so fill the Path when empty
		// rather than only when NodeInfo is nil; a nested dynamic node
		// that already set its own Path keeps it.
		if ev.NodeInfo == nil {
			ev.NodeInfo = &session.NodeInfo{Path: childPath}
		} else if ev.NodeInfo.Path == "" {
			ev.NodeInfo.Path = childPath
		}
		// Tag the event for scope filtering; mirrors adk-python
		// NodeRunner._enrich_event.
		if childScope != "" && ev.IsolationScope == "" {
			ev.IsolationScope = childScope
		}
		if ev.RequestedInput != nil {
			interrupted = true
		}
		// Track LongRunningToolIDs vs FunctionResponses for
		// WithRaiseOnWait. The check runs unconditionally so this
		// stays cheap even when the option is off — only one map
		// lookup per FC / FR and the maps stay empty when there
		// are no long-running tools at play.
		if opts.raiseOnWait {
			if len(ev.LongRunningToolIDs) > 0 {
				lrtSet := make(map[string]struct{}, len(ev.LongRunningToolIDs))
				for _, id := range ev.LongRunningToolIDs {
					lrtSet[id] = struct{}{}
				}
				for _, fc := range utils.FunctionCalls(ev.Content) {
					if fc == nil || fc.ID == "" {
						continue
					}
					if _, isLR := lrtSet[fc.ID]; !isLR {
						continue
					}
					if pendingLongRunningIDs == nil {
						pendingLongRunningIDs = map[string]struct{}{}
					}
					pendingLongRunningIDs[fc.ID] = struct{}{}
				}
			}
			if len(pendingLongRunningIDs) > 0 {
				for _, fr := range utils.FunctionResponses(ev.Content) {
					if fr == nil {
						continue
					}
					delete(pendingLongRunningIDs, fr.ID)
				}
			}
		}
		if childOut, ok := childEventOutput(ev); ok {
			validated, err := validateAndStampOutput(child, childOut, ev)
			if err != nil {
				return nil, &NodeRunError{
					ChildName: name, ChildPath: childPath, RunID: runID,
					Cause: err,
				}
			}
			out = validated
			hasOutput = true
			// Stamp OutputFor so resume can attribute the output: the
			// emitter's own path plus, under delegation, this parent and
			// its ancestors (the parent then suppresses its own terminal
			// event). Mirrors adk-python _enrich_event. A nested child
			// that already stamped its chain keeps it.
			if ev.NodeInfo.OutputFor == nil {
				outputFor := []string{ev.NodeInfo.Path}
				if opts.useAsOutput {
					outputFor = append(outputFor, s.parentPath)
					outputFor = append(outputFor, s.outputForAncestors...)
				}
				ev.NodeInfo.OutputFor = outputFor
			}
		}
		if err := s.emitUp(ev); err != nil {
			return nil, &NodeRunError{
				ChildName: name, ChildPath: childPath, RunID: runID,
				Cause: fmt.Errorf("%w: emitUp: %v", ErrNodeFailed, err),
			}
		}
	}

	if opts.raiseOnWait && len(pendingLongRunningIDs) > 0 {
		interrupted = true
	}

	// HITL (and raise-on-wait above): not cached, so resume re-runs and
	// re-invokes RunNode.
	if interrupted {
		return nil, s.pause(name, childPath, runID, ErrNodeInterrupted)
	}

	// A WaitForOutput child that produced no output is not done; park so
	// the parent re-runs. Mirrors adk-python's wait_for_output node field
	// (parks on missing output), independent of the WithRaiseOnWait gate.
	if !hasOutput && waitsForOutput(child) {
		return nil, s.pause(name, childPath, runID, ErrNodeWaitingForOutput)
	}

	s.storeCachedOutput(childPath, out)
	s.commitDelegation(childPath, out) // no-op unless this child claimed the delegation
	return out, nil
}

// pause reports that a child did not finish this turn so the parent must
// re-run later. cause is a pause sentinel, not a failure: ErrNodeInterrupted
// for HITL, ErrNodeWaitingForOutput for a WaitForOutput child with no output.
func (s *dynamicSubScheduler) pause(name, childPath, runID string, cause error) error {
	return &NodeRunError{
		ChildName: name, ChildPath: childPath, RunID: runID,
		Cause: cause,
	}
}

// waitsForOutput reports whether node opts into WaitForOutput (tri-state
// pointer; nil means the engine default of false).
func waitsForOutput(node Node) bool {
	w := node.Config().WaitForOutput
	return w != nil && *w
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
