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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// dynamicSubScheduler runs the children of one dynamic-node activation.
// It implements agent.NodeScheduler so the unified Context can schedule
// dynamic children without the agent package importing workflow.
type dynamicSubScheduler struct {
	parentPath string
	parentCtx  agent.Context
	emitUp     func(*session.Event) error

	// outputForAncestors are the extra node paths this activation's
	// output counts for, set when this activation is itself a
	// WithUseAsOutput child. A delegating child's event is stamped
	// OutputFor=[childPath, parentPath, ...outputForAncestors].
	outputForAncestors []string

	// resumeInputs are the re-entry resume payloads (keyed by
	// InterruptID) to propagate to children so HITL responses reach
	// dynamic children. Nil on fresh activations and handoff resume.
	resumeInputs map[string]any

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

// outputDelegation is the at-most-one WithUseAsOutput claim for a parent
// activation. claimed is set eagerly on the first delegating child and
// never cleared within the activation (matching adk-python's
// _output_delegated); a second delegating child is rejected. The output
// value flows up on the child's event (stamped OutputFor), so the claim
// only needs to record that delegation happened, not the value. A fresh
// sub-scheduler is built per activation, so nothing resets across turns.
//
// Methods require the enclosing scheduler's mu to be held.
type outputDelegation struct {
	claimed   bool
	childPath string
	childName string
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

func newDynamicSubScheduler(parent agent.Context, parentPath string, emitUp func(*session.Event) error) *dynamicSubScheduler {
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

// ScheduleNode implements agent.NodeScheduler. It adapts the
// agent-level call (child as any, agent.NodeRunOptions) onto the
// internal runNode, so the typed workflow.RunNode helper and any
// agent-level caller can schedule a dynamic child.
func (s *dynamicSubScheduler) ScheduleNode(_ agent.Context, child any, input any, opts agent.NodeRunOptions) (any, error) {
	node, ok := child.(Node)
	if !ok {
		return nil, fmt.Errorf("%w: child is %T, not a workflow.Node", ErrInvalidRunNodeContext, child)
	}
	return s.runNode(node, input, runNodeOptions{
		customRunID:    opts.RunID,
		useSubBranch:   opts.UseSubBranch,
		overrideBranch: opts.OverrideBranch,
		useAsOutput:    opts.UseAsOutput,
	})
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

	// Cached (WithRunID replay): the child already ran.
	if cached, ok := s.lookupCachedOutput(childPath); ok {
		return cached, nil
	}

	childBranch := deriveChildBranch(s.parentCtx.Branch(), name, runID, opts.useSubBranch, opts.overrideBranch)
	// A delegating child extends the OutputFor chain: its own delegating
	// children must also count their output for this parent and its
	// ancestors.
	var childAncestors []string
	if opts.useAsOutput {
		childAncestors = append([]string{s.parentPath}, s.outputForAncestors...)
	}
	// Build the child's unified context: rebrand the branch on the
	// parent's InvocationContext, then wrap as a node context carrying
	// the child path/runID, inherited resume inputs, and this
	// sub-scheduler (so a nested RunNode inside the child reaches it).
	childIC := withBranch(s.parentCtx.InvocationContext(), childBranch)
	childCtx := agent.NewNodeContext(childIC, childPath, runID, s.resumeInputs, s, nil)

	// EXPERIMENTAL: stash the child context (and its outputForAncestors
	// chain) in the embedded context.Context so tools running inside an
	// LlmAgent that is itself running as this dynamic child can recover
	// the node context via workflow.NodeContextFromGoContext, and a
	// nested dynamic node can recover the OutputFor chain. See
	// scheduleResumedNode for the static-node equivalent.
	bridge := &nodeBridge{outputForAncestors: childAncestors, resumeInputs: s.resumeInputs}
	childCtx = childCtx.WithContext(
		withNodeBridge(childCtx, bridge),
	).(agent.Context)
	bridge.ctx = childCtx

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
		if childOut, ok := childEventOutput(ev); ok {
			out = childOut
			// Every output event records the paths its output counts
			// for, starting with the emitter (mirrors adk-python
			// _node_runner._enrich_event). Under delegation the same
			// event also stands in for the parent and its ancestors, so
			// the parent skips re-emitting a duplicate.
			if ev.NodeInfo.OutputFor == nil {
				ev.NodeInfo.OutputFor = []string{ev.NodeInfo.Path}
			}
			if opts.useAsOutput {
				ev.Output = childOut // may have been carried as message text
				ev.NodeInfo.OutputFor = appendUnique(ev.NodeInfo.OutputFor, s.parentPath)
				for _, a := range s.outputForAncestors {
					ev.NodeInfo.OutputFor = appendUnique(ev.NodeInfo.OutputFor, a)
				}
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

// isDelegated reports whether a child claimed this activation's output
// via WithUseAsOutput. When true, that child's event already carried the
// output up (stamped with OutputFor), so the parent must not emit its own
// terminal output event.
func (s *dynamicSubScheduler) isDelegated() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.delegation.claimed
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

// appendUnique appends s to paths if not already present.
func appendUnique(paths []string, s string) []string {
	for _, p := range paths {
		if p == s {
			return paths
		}
	}
	return append(paths, s)
}
