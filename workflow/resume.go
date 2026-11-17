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
	"maps"
	"slices"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/typeutil"
	"google.golang.org/adk/v2/session"
)

// ErrInvalidResumeResponse is returned by Workflow.Resume when a
// response payload does not match its corresponding
// RequestInput.ResponseSchema. The waiting node is left in
// NodeWaiting with its interrupt still recorded so the caller can
// retry with a corrected payload.
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
// RunState reconstructed from session event history (see
// ReconstructRunState); responses maps
// RequestInput.InterruptID to the user-supplied response payload.
//
// For each waiting node whose InterruptID has a matching entry in
// responses, Resume:
//
//  1. Validates the payload against the interrupt's declared
//     response schema, if it declared one. A mismatch surfaces as
//     ErrInvalidResumeResponse via the iterator and leaves the node
//     in NodeWaiting with its interrupt still recorded.
//
//  2. Records the response and re-activates the node. What that means
//     depends on how the node asked:
//
//     - A re-entry node (RerunOnResume) is set to NodePending and
//     scheduled again with the response delivered via
//     ctx.ResumedInput. Successful completion clears the consumed
//     responses; reconstructed history retains the completed result.
//     - A handoff node is set to NodeCompleted and its response is
//     routed to its successors without re-running the asker; a
//     duplicate Resume then matches no waiting node and yields
//     ErrNothingToResume.
//
//     A caller that must treat a double-submit as success therefore
//     has to tolerate ErrNothingToResume after completion. A failed
//     re-entry can be retried with a matching response.
//
// Waiting nodes whose InterruptID is absent from responses remain
// in NodeWaiting unchanged.
// After a response matches an interrupted node, siblings whose latest
// activation failed are re-run before they can satisfy a downstream join.
//
// If responses is non-empty but no waiting node matches any
// InterruptID in it, Resume yields ErrNothingToResume so the
// caller can distinguish a successful resume from a stale or
// mistargeted submission. An empty responses map (or nil state)
// is treated as a clean no-op with no error.
//
// A node that Resume re-activates keeps the run id of the activation that
// paused or failed, and the next new activation of that node continues the
// count. Events carry the path "<ctx path>/<name>@<run id>", where ctx path
// is the path of the context passed to Resume. The runner's resume turn
// passes an empty path, so there a re-activated node's events carry
// "loop@2", not "wf@1/loop@2". Triggers that the node's predecessors
// produced while it was waiting run, in arrival order, once the node is free.
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
		s.workflowName = w.name
		s.state = state
		for name, n := range state.runCounts {
			s.counters.seed(ctx.Path(), name, n)
		}
		defer func() {
			s.cancelAll()
			if len(s.runsByName) > 0 {
				s.run(nil, true)
			}
			s.wg.Wait()
		}()

		// Resume runs in two passes so that when one call
		// satisfies several askers feeding a JoinNode, the
		// barrier sees every predecessor as NodeCompleted
		// before it is evaluated. Re-entry-mode askers are
		// unaffected — they re-run rather than complete here.
		type deferredHandoff struct {
			node Node
			resp any
			// answeredEarlier: the asker's response was already in
			// history before this turn.
			answeredEarlier bool
		}
		var deferredHandoffs []deferredHandoff
		scheduled := 0
		invalidResponses := map[string]bool{}

		// Act on each node the rehydration reconstructed, but only
		// for interrupts answered in THIS turn (present in responses).
		// Gating on the current turn's responses keeps a duplicate
		// Resume from acting on an already-consumed interrupt. For a
		// handoff node that leaves nothing to reschedule; a re-entry
		// node is rescheduled regardless, as step 2 describes. Mirrors
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
			// Direct Resume callers may supply a new or corrected response
			// not yet in history. Validate it before a waiting handoff or
			// merging it into a pending/failed node's re-entry inputs.
			freshMatched := map[string]any{}
			if ns.Status == NodeWaiting || ns.Status == NodePending || ns.Status == NodeFailed {
				ids := append([]string(nil), ns.Interrupts...)
				if ns.Status != NodeWaiting {
					for id := range ns.ResumedInputs {
						ids = append(ids, id)
					}
				}
				schemaErr := false
				for _, id := range ids {
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
					invalidResponses[name] = true
					continue
				}
			}
			if !answeredNow && len(freshMatched) == 0 {
				continue
			}

			reenter := s.graph.rerunsOnResume(node)

			if reenter || ns.Status == NodePending || ns.Status == NodeFailed {
				// Re-entry: re-activate with the resolved responses
				// delivered via ctx.ResumedInput.
				if ns.ResumedInputs == nil {
					ns.ResumedInputs = map[string]any{}
				}
				maps.Copy(ns.ResumedInputs, freshMatched)
				ns.Status = NodePending
				s.enqueue(pendingActivation{
					node: node, input: ns.Input, triggeredBy: ns.TriggeredBy,
					branch: ns.Branch, runID: ns.runID, resumeInputs: ns.ResumedInputs,
				})
				// Start it now, so the node is NodeRunning while the
				// rest of this pass looks at the other nodes.
				s.scheduleReady()
				scheduled++
			} else {
				// Handoff: the response is the asker's output for its
				// successors; the asker does not re-run.
				out := ns.Output
				if len(freshMatched) > 0 {
					out = resumeOutput(freshMatched)
				}
				answeredThisTurn := ns.answeredThisTurn || len(freshMatched) > 0
				ns.Status = NodeCompleted
				ns.Output = out
				ns.Interrupts = nil
				deferredHandoffs = append(deferredHandoffs, deferredHandoff{
					node: node, resp: out, answeredEarlier: !answeredThisTurn,
				})
				// A matched asker is itself an effective resume even
				// when terminal (no successors to count in Pass 2):
				// without this a single-asker workflow would wrongly
				// report ErrNothingToResume. answeredThisTurn gates on
				// the response being new this turn (rehydration sets it
				// from resolvedCount), so a duplicate resume stays a
				// no-op. freshMatched covers the runner-direct path
				// where the response is not yet in history.
				if answeredThisTurn {
					scheduled++
				}
			}
		}

		// Pass 2: walk successors of the deferred handoffs.
		// All matched askers are now NodeCompleted, so any
		// downstream JoinNode sees a settled predecessor set, and
		// each JoinNode is buffered at most once.
		joinsBuffered := map[string]bool{}
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
			from := h.node.Name()
			// Skip a successor of an asker answered on an earlier turn
			// when that successor already produced output: the earlier
			// turn ran it, and re-triggering it would re-run completed
			// work (a duplicate resume). Keeps Resume idempotent. An
			// asker answered this turn triggers every successor, even
			// one another predecessor ran before. An earlier answer
			// whose successor never ran (a rejected response) does not
			// skip it.
			succs := slices.DeleteFunc(findSuccessors(s.graph, h.node, h.resp, nil, parentBranch), func(sc successor) bool {
				return h.answeredEarlier && state.completed[sc.node.Name()]
			})
			// Non-join triggers first, then the barriers, as
			// bufferSuccessors orders them.
			scheduled += s.bufferSuccessors(from, slices.DeleteFunc(slices.Clone(succs), func(sc successor) bool { return sc.join }))
			for _, sc := range succs {
				name := sc.node.Name()
				if sc.join && !joinsBuffered[name] && s.bufferBarrier(from, sc.node) {
					joinsBuffered[name] = true
					scheduled++
				}
			}
		}

		if scheduled == 0 {
			yield(nil, ErrNothingToResume)
			return
		}
		// A valid resume also retries siblings whose latest attempt
		// failed. They cannot satisfy a join with their discarded partial
		// output, and need not have raised an interrupt themselves.
		for name, ns := range state.Nodes {
			if ns.Status != NodeFailed || invalidResponses[name] {
				continue
			}
			if node := s.nodesByName[name]; node != nil {
				s.enqueue(pendingActivation{
					node: node, input: ns.Input, triggeredBy: ns.TriggeredBy,
					branch: ns.Branch, runID: ns.runID, resumeInputs: ns.ResumedInputs,
				})
			}
		}
		// Restore the triggers that history shows were produced but had
		// not started, behind the re-activations, so each runs once its
		// node is free. A map has no history order, so nodes are restored
		// by name; each node's own triggers keep their order.
		for _, name := range slices.Sorted(maps.Keys(state.queued)) {
			node := s.nodesByName[name]
			if node == nil || invalidResponses[name] {
				continue
			}
			for _, t := range state.queued[name] {
				t.node = node
				s.enqueue(t)
			}
		}
		s.scheduleReady()

		s.run(yield, false)
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
