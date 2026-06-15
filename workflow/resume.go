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
	ctx agent.Context,
	state *RunState,
	responses map[string]any,
) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if state == nil || len(responses) == 0 {
			return
		}

		s := newScheduler(ctx, w.graph, w.maxConcurrency)
		s.state = state

		// Resume runs in two passes so that when one call
		// satisfies several askers feeding a JoinNode, the
		// barrier sees every predecessor as NodeCompleted
		// before it is evaluated. Re-entry-mode askers are
		// unaffected — they re-run rather than complete here.
		type deferredHandoff struct {
			node Node
			resp any
		}
		var deferredHandoffs []deferredHandoff
		scheduled := 0

		// Act on each node the rehydration reconstructed, but only
		// for interrupts answered in THIS turn (present in responses).
		// Gating on the current turn's responses keeps Resume
		// idempotent: a duplicate turn whose responses target an
		// already-consumed interrupt reschedules nothing. Mirrors
		// adk-python gating _extract_resume_output on ctx.resume_inputs.
		for name, ns := range state.Nodes {
			node := s.nodesByName[name]
			if node == nil {
				continue
			}

			// Which of this node's interrupts were answered this turn?
			answeredNow := false
			for id := range ns.ResumedInputs {
				if _, ok := responses[id]; ok {
					answeredNow = true
					break
				}
			}
			// WAITING nodes whose response arrived this turn but is not
			// yet in history (the runner node path passes responses
			// directly): fold it into ResumedInputs after validation.
			freshMatched := map[string]any{}
			if ns.Status == NodeWaiting {
				schemaErr := false
				for _, id := range ns.Interrupts {
					resp, ok := responses[id]
					if !ok {
						continue
					}
					if sc := ns.interruptSchemas[id]; sc != nil {
						validated, err := validateResumeResponse(resp, sc)
						if err != nil {
							if !yield(nil, fmt.Errorf("%w: node %q: %w", ErrInvalidResumeResponse, name, err)) {
								return
							}
							schemaErr = true
							break
						}
						resp = validated
					}
					freshMatched[id] = resp
				}
				if schemaErr {
					continue
				}
			}
			if !answeredNow && len(freshMatched) == 0 {
				continue
			}

			reenter := false
			if r := node.Config().RerunOnResume; r != nil && *r {
				reenter = true
			}

			if reenter || ns.Status == NodePending {
				// Re-entry: re-activate with the resolved responses
				// delivered via ctx.ResumedInput.
				if ns.ResumedInputs == nil {
					ns.ResumedInputs = map[string]any{}
				}
				for id, resp := range freshMatched {
					ns.ResumedInputs[id] = resp
				}
				ns.Status = NodePending
				s.scheduleResumedNode(node, ns.Input, ns.TriggeredBy, ns.Branch, ns.ResumedInputs)
				scheduled++
			} else {
				// Handoff: the response is the asker's output for its
				// successors; the asker does not re-run.
				out := ns.Output
				if len(freshMatched) > 0 {
					out = resumeOutput(freshMatched)
				}
				ns.Status = NodeCompleted
				ns.Output = out
				ns.Interrupts = nil
				deferredHandoffs = append(deferredHandoffs, deferredHandoff{
					node: node, resp: out,
				})
				// A matched asker is itself an effective resume even
				// when terminal (no successors to count in Pass 2):
				// without this a single-asker workflow would wrongly
				// report ErrNothingToResume. answeredThisTurn gates on
				// the response being new this turn (rehydration sets it
				// from resolvedCount), so a duplicate resume stays a
				// no-op. freshMatched covers the runner-direct path
				// where the response is not yet in history.
				if ns.answeredThisTurn || len(freshMatched) > 0 {
					scheduled++
				}
			}
		}

		// Pass 2: walk successors of the deferred handoffs.
		// All matched askers are now NodeCompleted, so any
		// downstream JoinNode sees a settled predecessor set.
		for _, h := range deferredHandoffs {
			// findSuccessors is called with event=nil, so
			// successors reached only via a concrete Route
			// (StringRoute etc.) do not fire — the response is
			// opaque to the routing layer. Successors reached via
			// an unconditional edge or via the Default route fire
			// as usual.
			// Handoff successors inherit the asker's branch so the
			// downstream LLM history filter still scopes correctly
			// when a parallel branch resumes via handoff.
			parentBranch := ""
			if ns := s.state.Nodes[h.node.Name()]; ns != nil {
				parentBranch = ns.Branch
			}
			for _, succ := range findSuccessors(s.graph, s.state, h.node, h.resp, nil, parentBranch) {
				// Skip a successor that already produced output on a
				// prior turn: re-triggering it would re-run completed
				// work (a duplicate resume). Keeps Resume idempotent.
				if state.completed[succ.node.Name()] {
					continue
				}
				s.scheduleNode(succ.node, succ.input, succ.triggeredBy, succ.branch)
				scheduled++
			}
		}

		if scheduled == 0 {
			yield(nil, ErrNothingToResume)
			return
		}

		s.run(yield)
		s.wg.Wait()
	}
}

// resumeOutput collapses a node's matched interrupt responses into a
// single handoff output: one response forwards its value directly,
// several forward the whole map. Mirrors adk-python
// _extract_resume_output.
func resumeOutput(matched map[string]any) any {
	if len(matched) == 1 {
		for _, v := range matched {
			return v
		}
	}
	return matched
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
