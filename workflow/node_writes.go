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
	"maps"
	"slices"
	"sync"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// nodeWrites serves one node activation: it validates the events the
// activation yields against the state validator of its context and moves the
// activation's pending state and artifact deltas onto them, as adk-python's
// NodeRunner does (workflow/_node_runner.py _track_event_in_context,
// _flush_deltas, _flush_output_and_deltas).
//
// It is not safe for concurrent use: admit, forwardUp, pending and drainInto
// are called only from the goroutine that drives the activation's iterator.
type nodeWrites struct {
	// actions is ctx.Actions() of the activation; nil when its context has
	// none, and then nothing is merged or flushed.
	actions *session.EventActions
	// validator is inherited from the context; nil checks nothing.
	validator agent.StateValidator
	// validatesPending is set for a *FunctionNode, whose events carry its
	// pending delta in adk-python (_function_node.py _to_event), so a direct
	// write to ctx.Actions().StateDelta is checked with the event.
	validatesPending bool
	// mine holds the events deeper runners checked and forwarded to this
	// activation.
	mine *forwardedEvents
	// up is the enclosing activation's registry; nil at the root.
	up *forwardedEvents
}

// forwardedKey is the context key of an activation's forwardedEvents.
type forwardedKey struct{}

// activationActionsKey is the context key of the pending event actions
// beginActivation gave an activation's context.
type activationActionsKey struct{}

// isActivationContext reports whether ctx is the context a node activation
// runs under, or a copy of it that keeps its pending event actions. A tool
// context derived from it has actions of its own, and a callback context or
// agent.NewContext(ic) has none, so neither is one.
func isActivationContext(ctx agent.Context) bool {
	actions, ok := ctx.Value(activationActionsKey{}).(*session.EventActions)
	return ok && actions == ctx.Actions()
}

// beginActivation derives the context n runs under, with fresh pending event
// actions and a fresh registry of forwarded events, and returns the
// activation's nodeWrites. A context that ignores the delta yields a
// nodeWrites that checks and merges nothing.
//
// own is the node's own validator from nodeStateValidator; nil keeps the
// validator ctx already carries.
func beginActivation(ctx agent.Context, n Node, own *agent.StateValidator) (agent.Context, *nodeWrites) {
	w := &nodeWrites{up: forwardedFrom(ctx), mine: &forwardedEvents{}}
	_, w.validatesPending = n.(*FunctionNode)
	actions := &session.EventActions{}
	base := context.WithValue(context.WithValue(ctx, forwardedKey{}, w.mine), activationActionsKey{}, actions)
	derived := ctx.WithDelta(&agent.CommonContextDelta{
		EventActions:           actions,
		InvocationContextDelta: &agent.InvocationContextDelta{Context: &base},
		StateValidator:         own,
	})
	w.actions = derived.Actions()
	if sv, ok := derived.(interface{ StateValidator() agent.StateValidator }); ok {
		w.validator = sv.StateValidator()
	}
	return derived, w
}

// admit checks ev and moves the pending delta onto it. An event a deeper
// runner has already checked is left untouched. It returns the validator's
// error, unwrapped, for the first refused key in sorted order, and then
// leaves ev and the pending delta as they were. A partial event is checked
// and carries no pending delta.
func (w *nodeWrites) admit(ev *session.Event) error {
	if ev == nil || w.mine.take(ev) {
		return nil
	}
	partial := ev.Partial
	if w.validator != nil {
		checked := ev.Actions.StateDelta
		if w.validatesPending && !partial && w.actions != nil && len(w.actions.StateDelta) > 0 {
			checked = maps.Clone(ev.Actions.StateDelta)
			if checked == nil {
				checked = map[string]any{}
			}
			maps.Copy(checked, w.actions.StateDelta)
		}
		if err := validateEntries(w.validator, checked); err != nil {
			return err
		}
	}
	if partial || w.actions == nil {
		return nil
	}
	mergeDelta(&ev.Actions, w.actions)
	clear(w.actions.StateDelta)
	clear(w.actions.ArtifactDelta)
	return nil
}

// snapshotPending returns a copy of the pending state and artifact deltas
// that admit(ev) is about to move onto ev, or nil when admit moves nothing:
// ev is nil, partial or already checked by a deeper runner (registered in
// w.mine, not consumed here), w.actions is nil, or nothing is pending. Call
// it immediately before admit(ev), from the goroutine that drives the
// iterator.
func (w *nodeWrites) snapshotPending(ev *session.Event) *session.EventActions {
	if ev == nil || ev.Partial || w.actions == nil || w.mine.has(ev) {
		return nil
	}
	if len(w.actions.StateDelta) == 0 && len(w.actions.ArtifactDelta) == 0 {
		return nil
	}
	snapshot := &session.EventActions{}
	mergeDelta(snapshot, w.actions)
	return snapshot
}

// forwardUp registers ev as checked in the enclosing activation's registry.
// It does nothing at the root, nor when the activation has no validator: a
// runner that checked nothing leaves ev to the enclosing activation's admit.
func (w *nodeWrites) forwardUp(ev *session.Event) {
	if w.validator == nil {
		return
	}
	w.up.mark(ev)
}

// pending builds the activation's trailing event, which carries the pending
// state and artifact deltas and nothing else, and empties them. It returns nil
// when nothing is pending. The event is not validated, as adk-python's
// trailing flush is not.
func (w *nodeWrites) pending(ctx agent.Context) *session.Event {
	if w.actions == nil || (len(w.actions.StateDelta) == 0 && len(w.actions.ArtifactDelta) == 0) {
		return nil
	}
	ev := session.NewEvent(ctx, ctx.InvocationID())
	w.drainInto(&ev.Actions)
	return ev
}

// drainInto moves the pending state and artifact deltas into dst.
func (w *nodeWrites) drainInto(dst *session.EventActions) {
	if w.actions == nil {
		return
	}
	mergeDelta(dst, w.actions)
	clear(w.actions.StateDelta)
	clear(w.actions.ArtifactDelta)
}

// validateEntries checks every entry of delta with v, in sorted key order, and
// returns the first error. A nil v checks nothing.
func validateEntries(v agent.StateValidator, delta map[string]any) error {
	if v == nil {
		return nil
	}
	for _, key := range slices.Sorted(maps.Keys(delta)) {
		if err := v.ValidateStateEntry(key, delta[key]); err != nil {
			return err
		}
	}
	return nil
}

// mergeDelta copies the state and artifact deltas of src into dst; an entry of
// src replaces one of dst with the same key. A nil src changes nothing.
func mergeDelta(dst, src *session.EventActions) {
	if src == nil {
		return
	}
	if len(src.StateDelta) > 0 {
		if dst.StateDelta == nil {
			dst.StateDelta = make(map[string]any, len(src.StateDelta))
		}
		maps.Copy(dst.StateDelta, src.StateDelta)
	}
	if len(src.ArtifactDelta) > 0 {
		if dst.ArtifactDelta == nil {
			dst.ArtifactDelta = make(map[string]int64, len(src.ArtifactDelta))
		}
		maps.Copy(dst.ArtifactDelta, src.ArtifactDelta)
	}
}

// flushesPending reports whether an activation that ended with cause sends its
// pending deltas. It does not when it was canceled, nor when it failed because
// a dynamic child failed, as adk-python's NodeRunner.run returns before its
// flush in both cases (workflow/_node_runner.py, the asyncio.CancelledError and
// DynamicNodeFailError branches). Every other ending flushes: success, a pause,
// a timeout, a panic and any other error.
func flushesPending(cause error) bool {
	if errors.Is(cause, context.Canceled) {
		return false
	}
	var childFailure *NodeRunError
	return !errors.As(cause, &childFailure) || !errors.Is(cause, ErrNodeFailed)
}

// forwardedEvents records the events that deeper runners checked and
// forwarded to one activation, so that the activation's admit neither checks
// them again nor merges its own pending delta into them. It is safe for
// concurrent use.
type forwardedEvents struct {
	mu   sync.Mutex
	seen map[*session.Event]struct{}
}

// mark registers ev. A nil receiver or a nil ev does nothing.
func (f *forwardedEvents) mark(ev *session.Event) {
	if f == nil || ev == nil {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.seen == nil {
		f.seen = map[*session.Event]struct{}{}
	}
	f.seen[ev] = struct{}{}
}

// take reports whether ev is registered and forgets it. A nil receiver
// reports false.
func (f *forwardedEvents) take(ev *session.Event) bool {
	if f == nil || ev == nil {
		return false
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	_, ok := f.seen[ev]
	delete(f.seen, ev)
	return ok
}

// has reports whether ev is registered, without forgetting it. A nil
// receiver reports false.
func (f *forwardedEvents) has(ev *session.Event) bool {
	if f == nil || ev == nil {
		return false
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	_, ok := f.seen[ev]
	return ok
}

// forwardedFrom returns the registry of the activation ctx belongs to, or nil
// when there is none or ctx is nil.
func forwardedFrom(ctx context.Context) *forwardedEvents {
	if ctx == nil {
		return nil
	}
	f, _ := ctx.Value(forwardedKey{}).(*forwardedEvents)
	return f
}
