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
	"iter"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/session"
)

// ErrInvalidResumeResponse is returned by Workflow.Resume when a
// response payload does not match its corresponding
// RequestInput.ResponseSchema. The waiting node is left in
// NodeWaiting with PendingRequest intact so the caller can retry
// with a corrected payload.
var ErrInvalidResumeResponse = errors.New("workflow: resume response does not match request schema")

// ErrNothingToResume is returned by Workflow.Resume when the
// caller supplied a non-empty responses map but no waiting node
// matched any of the InterruptIDs in it. Lets the caller
// distinguish "successful resume" from "submitted response had no
// effect" — typically a sign that the response targets a stale or
// already-consumed request, or that the workflow graph has
// evolved out of the node that was waiting.
var ErrNothingToResume = errors.New("workflow: no waiting node matched the supplied responses")

// Resume continues a previously paused workflow run. state is the
// RunState loaded from session storage; responses maps
// RequestInput.InterruptID to the user-supplied response payload.
//
// For each waiting node whose InterruptID has a matching entry in
// responses, Resume:
//
//  1. Validates the payload against PendingRequest.ResponseSchema,
//     if non-nil. A mismatch surfaces as ErrInvalidResumeResponse
//     via the iterator and leaves the node in NodeWaiting with
//     PendingRequest intact.
//
//  2. Consumes the pending request (clears PendingRequest, sets
//     Status = NodePending) before re-scheduling, so a duplicate
//     Resume call with the same InterruptID becomes a no-op.
//
//  3. Routes the response to the asker's successors as if the
//     asker had emitted it as its output (handoff mode). The
//     asker itself does NOT re-execute.
//
// Waiting nodes whose InterruptID is absent from responses remain
// in NodeWaiting unchanged.
//
// If responses is non-empty but no waiting node matches any
// InterruptID in it, Resume yields ErrNothingToResume so the
// caller can distinguish a successful resume from a stale or
// mistargeted submission. An empty responses map (or nil state)
// is treated as a clean no-op with no error.
func (w *Workflow) Resume(
	ctx agent.InvocationContext,
	state *RunState,
	responses map[string]any,
) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if state == nil || len(responses) == 0 {
			return
		}

		s := newScheduler(ctx, w.graph)
		s.state = state

		scheduled := 0
		for name, ns := range state.Nodes {
			if ns.Status != NodeWaiting || ns.PendingRequest == nil {
				continue
			}
			resp, ok := responses[ns.PendingRequest.InterruptID]
			if !ok {
				continue
			}

			// Schema validation: surface a typed error and leave
			// the node parked so the caller can retry.
			if ns.PendingRequest.ResponseSchema != nil {
				validated, err := validateResumeResponse(resp, ns.PendingRequest.ResponseSchema)
				if err != nil {
					if !yield(nil, fmt.Errorf("%w: node %q: %w", ErrInvalidResumeResponse, name, err)) {
						return
					}
					continue
				}
				resp = validated
			}

			node := s.nodesByName[name]
			if node == nil {
				continue
			}

			// Snapshot InterruptID before consuming PendingRequest;
			// re-entry mode passes it through resumeInputs.
			interruptID := ns.PendingRequest.InterruptID

			// Consume PendingRequest before scheduling. A duplicate
			// Resume with the same InterruptID will skip this node
			// because PendingRequest is now nil.
			ns.PendingRequest = nil
			ns.Status = NodePending

			if r := node.Config().RerunOnResume; r != nil && *r {
				// Re-entry mode: re-activate the asker with its
				// original input; the response is delivered via
				// ctx.ResumedInput(InterruptID), not via the
				// input parameter. Successors fire only when the
				// re-entry activation produces an output.
				resumeInputs := map[string]any{interruptID: resp}
				s.scheduleResumedNode(node, ns.Input, ns.TriggeredBy, resumeInputs)
			} else {
				// Handoff mode: schedule each successor with the
				// response as its input, exactly as if the asker
				// had emitted it as output. Reuses findSuccessors
				// so routing, fan-out and fan-in invariants apply
				// uniformly. findSuccessors is called with
				// event=nil, so successors reached only via a
				// concrete Route (StringRoute etc.) do not fire —
				// the response is opaque to the routing layer.
				// Successors reached via an unconditional edge or
				// via the Default route fire as usual.
				for _, succ := range findSuccessors(s.graph, node, resp, nil) {
					s.scheduleNode(succ.node, succ.input, succ.triggeredBy)
				}
			}
			scheduled++
		}

		if scheduled == 0 {
			yield(nil, ErrNothingToResume)
			return
		}

		s.run(yield)
		s.wg.Wait()

		// Persist the post-resume state via a session.Event with
		// StateDelta. If new nodes paused during this Resume the
		// next turn will see them; if the run completed the state
		// reflects that too.
		yieldRunStateEvent(ctx, w.name, s.state, yield)
	}
}

// validateResumeResponse coerces resp into the type described by
// schema, returning the validated value or an error. When schema
// is nil, resp passes through unchanged.
//
// The schema is resolved on each call rather than cached: persisted
// RunState round-trips schemas through JSON, which does not
// preserve any pre-resolved form.
func validateResumeResponse(resp any, schema *jsonschema.Schema) (any, error) {
	if schema == nil {
		return resp, nil
	}
	resolved, err := schema.Resolve(nil)
	if err != nil {
		return nil, fmt.Errorf("resolve schema: %w", err)
	}
	return typeutil.ConvertToWithJSONSchema[any, any](resp, resolved)
}
