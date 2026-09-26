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
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// nodeScanState accumulates, per node, what the session history says
// about a paused run. Mirrors adk-python's _ChildScanState.
type nodeScanState struct {
	// interrupts are the long-running tool IDs the node raised
	// (insertion-ordered for stable reconstruction).
	interrupts []string
	seen       map[string]struct{}
	// resolved maps an interrupt ID to the (last) user response.
	resolved map[string]any
	// resolvedCount maps an interrupt ID to how many user
	// FunctionResponse events in history resolved it. 1 means the
	// response arrived this turn for the first time; >1 means a
	// duplicate resume replayed an already-consumed response. Lets
	// Resume tell a genuine first resume from an idempotent no-op.
	resolvedCount map[string]int
	// consumed marks an interrupt whose answer the node has already
	// acted on: it emitted an event after that answer first appeared.
	// Counting responses cannot express this — a retry after a
	// rejected payload and a duplicate replay both leave two responses
	// in history, but only the replay follows a run of the node.
	consumed map[string]bool
	// schemas maps an interrupt ID to its declared response schema,
	// re-extracted from the pause FunctionCall args.
	schemas map[string]*jsonschema.Schema
	branch  string
}

// reopen drops an interrupt from the seen set so addInterrupt records it again
// as a fresh, unanswered request.
func (s *nodeScanState) reopen(id string) {
	delete(s.seen, id)
	s.interrupts = slices.DeleteFunc(s.interrupts, func(v string) bool { return v == id })
}

func (s *nodeScanState) addInterrupt(id string) {
	if s.seen == nil {
		s.seen = map[string]struct{}{}
	}
	if _, ok := s.seen[id]; ok {
		return
	}
	s.seen[id] = struct{}{}
	s.interrupts = append(s.interrupts, id)
}

// ReconstructRunState rebuilds the paused RunState by scanning session
// history instead of loading a persisted blob, mirroring adk-python's
// rehydration (workflow/utils/_rehydration_utils.py:
// _reconstruct_node_states + _workflow.py:_infer_node_state).
//
// For each node it collects the long-running interrupts it raised
// (Event.LongRunningToolIDs, attributed by event node path), the user
// FunctionResponses that resolved them, and each interrupt's declared
// response schema. inferNodeState then maps that scan to a NodeState
// (WAITING / PENDING+ResumedInputs / COMPLETED+Output). Returns
// (nil, nil) when no node has interrupt history.
//
// invocationID scopes the scan to a single logical run: events from
// other invocations are skipped, so a fresh run started in a session
// that already holds a completed run does not collide with it (a stable
// InterruptID reused across runs would otherwise rehydrate against the
// prior, already-resolved interrupt). The runner reuses the paused
// run's invocation ID for the resume turn, so the pause and its reply
// share it. Empty invocationID disables the filter (scan all history).
// Mirrors adk-python _reconstruct_node_states' invocation_id gate.
func (w *Workflow) ReconstructRunState(sess session.Session, invocationID string) (*RunState, error) {
	if sess == nil {
		return nil, nil
	}
	nodesByName := buildNodesByName(w.graph)
	events := sess.Events()

	// Stage 1: scan history into a per-node view of the pause
	// (interrupts raised, responses that resolved them, schemas).
	scans := scanHistory(events, nodesByName, invocationID)

	// Stage 2: gather the inputs inferNodeState needs to rebuild a
	// re-entry node's input: every node's cached output, the set of
	// nodes that ran, and the workflow's seed input.
	nodeOutputs, completed := collectNodeOutputs(events, nodesByName, invocationID)
	workflowInput := firstUserInput(events, invocationID)

	// Stage 3: turn each interrupted node's scan into a NodeState.
	state, err := w.buildRunState(scans, nodesByName, nodeOutputs, workflowInput)
	if err != nil {
		return nil, err
	}
	if state == nil {
		return nil, nil
	}

	// WAITING nodes have not finished, so Resume must not treat them
	// as already-run; the rest stay in completed to skip their
	// successors.
	for name, ns := range state.Nodes {
		if ns.Status == NodeWaiting {
			delete(completed, name)
		}
	}
	state.completed = completed
	return state, nil
}

// scanHistory walks session events once and returns, per static graph
// node, what history says about a paused run: the long-running
// interrupts it raised, the user responses that resolved them, and
// each interrupt's declared response schema. Only nodes with
// interrupt history are returned. invocationID, when non-empty,
// restricts the scan to that invocation's events.
func scanHistory(events session.Events, nodesByName map[string]Node, invocationID string) map[string]*nodeScanState {
	scans := map[string]*nodeScanState{}
	interruptOwner := map[string]string{} // interrupt ID -> node name
	scanFor := func(name string) *nodeScanState {
		s := scans[name]
		if s == nil {
			s = &nodeScanState{resolved: map[string]any{}, resolvedCount: map[string]int{}, consumed: map[string]bool{}, schemas: map[string]*jsonschema.Schema{}}
			scans[name] = s
		}
		return s
	}

	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		if ev == nil {
			continue
		}
		if invocationID != "" && ev.InvocationID != invocationID {
			continue
		}

		// A user FunctionResponse resolves an interrupt — not the
		// tool's own initial "pending" response (authored by the
		// node). Mirrors adk-python's event.author == 'user' gate.
		// Last response per interrupt wins, so a retry after a
		// rejected payload supersedes the earlier one.
		if ev.Author == "user" && ev.Content != nil {
			for _, p := range ev.Content.Parts {
				fr := frPart(p)
				if fr == nil {
					continue
				}
				owner, ok := interruptOwner[fr.ID]
				if !ok {
					continue
				}
				sf := scanFor(owner)
				// consumed is deliberately not reset here. An answer is
				// consumed once the node settles on it, and a later
				// answer to the same interrupt does not un-settle that —
				// the node either acted after the first one or it did
				// not. (Resetting would also be a no-op: consumed is
				// written only for IDs already in resolved.)
				sf.resolved[fr.ID] = utils.UnwrapResponse(fr.Response)
				sf.resolvedCount[fr.ID]++
			}
			continue
		}

		// Interrupts the node raised, attributed to the static graph
		// node that emitted the event (NodeInfo.Path; dynamic children
		// fold into their static ancestor — see eventNodeName).
		owner := eventNodeName(ev, nodesByName)
		if _, ok := nodesByName[owner]; !ok {
			continue
		}
		s := scanFor(owner)
		// The node SETTLED on every answer already in history: it either
		// produced its output or paused again on a new interrupt.
		//
		// Any other event is work in progress, and an activation that
		// emitted one and then failed must stay resumable. History never
		// un-answers an interrupt, so marking those answers consumed
		// wedges the run for good — every later retry is skipped as a
		// replay and the approved side effect never settles.
		//
		// Recorded per interrupt, not per node: a node that re-entered
		// on one answer and paused again must still act on the answer to
		// the new interrupt.
		//
		// Only the node's OWN activation settles it. eventNodeName folds
		// a delegated child into its static ancestor, so without the
		// settlesOwner test a child completing would read as the
		// orchestrator completing — and an orchestrator that delegates
		// successfully and then fails could never be retried.
		if settlesOwner(ev, owner, nodesByName) && (ev.Output != nil || len(ev.LongRunningToolIDs) > 0) {
			for id := range s.resolved {
				s.consumed[id] = true
			}
		}
		if ev.Output != nil {
			s.branch = ev.Branch
		}
		for _, id := range ev.LongRunningToolIDs {
			if id == "" {
				continue
			}
			// Raising an ID the node has already been answered on re-opens
			// it: the node looked at the answer and asked again, typically
			// because it rejected the payload.
			//
			// Two things go wrong without it, one per node kind. A handoff
			// node rehydrates NodeCompleted, because addInterrupt dedupes
			// and the ID never returns to the unresolved set, and hands the
			// rejected answer to its successors. A re-entry node is skipped
			// as a replay instead, because the re-raise settles it and so
			// marks the answer consumed. Either way the corrected answer can
			// never reach the node.
			//
			// Whoever raised it. A delegated child re-asking under this node
			// is the node asking again as far as the engine is concerned:
			// that is the pause it parks and re-enters on.
			if _, answered := s.resolved[id]; answered {
				delete(s.resolved, id)
				delete(s.resolvedCount, id)
				delete(s.consumed, id)
				s.reopen(id)
			}
			s.addInterrupt(id)
			if s.branch == "" {
				s.branch = ev.Branch
			}
			interruptOwner[id] = owner
			if sc := schemaFromEvent(ev, id); sc != nil {
				s.schemas[id] = sc
			}
		}
	}
	return scans
}

// collectNodeOutputs walks history once and returns each graph node's
// last cached output plus the set of nodes that emitted any event.
// The outputs feed predecessor-input reconstruction for re-entry
// nodes; completed lets Resume skip already-run successors. When
// invocationID is non-empty, events from other invocations are skipped
// so a prior run's completed nodes do not suppress the current run.
func collectNodeOutputs(events session.Events, nodesByName map[string]Node, invocationID string) (outputs map[string]any, completed map[string]bool) {
	outputs = map[string]any{}
	completed = map[string]bool{}
	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		if ev == nil {
			continue
		}
		if invocationID != "" && ev.InvocationID != invocationID {
			continue
		}
		name := eventNodeName(ev, nodesByName)
		if _, ok := nodesByName[name]; !ok {
			continue
		}
		completed[name] = true
		// Prefer an explicit Output; otherwise derive it from the
		// model message when the event is flagged MessageAsOutput,
		// so a message-as-output node recovers its output on resume
		// (mirrors adk-python _reconstruct_node_states'
		// use_message_as_output branch).
		out, ok := childEventOutput(ev)
		if !ok {
			continue
		}
		outputs[name] = out
		// A delegated output also counts for the static owners of the
		// OutputFor paths, so a delegating ancestor recovers it on resume
		// without re-emitting. Mirrors adk-python's output_for.
		if ev.NodeInfo != nil {
			for _, p := range ev.NodeInfo.OutputFor {
				owner := staticNodeName(p)
				if owner == name {
					continue
				}
				if _, known := nodesByName[owner]; known {
					outputs[owner] = out
				}
			}
		}
	}
	return outputs, completed
}

// buildRunState maps each interrupted node's scan to a NodeState via
// inferNodeState. Returns (nil, nil) when no node has interrupt
// history, matching the "nothing to resume" case.
func (w *Workflow) buildRunState(scans map[string]*nodeScanState, nodesByName map[string]Node, nodeOutputs map[string]any, workflowInput any) (*RunState, error) {
	var state *RunState
	for nodeName, scan := range scans {
		if len(scan.interrupts) == 0 {
			continue
		}
		ns, err := w.inferNodeState(nodesByName[nodeName], scan, nodeOutputs, workflowInput)
		if err != nil {
			return nil, err
		}
		if ns == nil {
			continue
		}
		if state == nil {
			state = NewRunState()
		}
		state.Nodes[nodeName] = ns
	}
	return state, nil
}

// unresolvedInterrupts returns the interrupts the node raised that no
// user response has resolved yet, preserving insertion order.
func unresolvedInterrupts(scan *nodeScanState) []string {
	unresolved := make([]string, 0, len(scan.interrupts))
	for _, id := range scan.interrupts {
		if _, done := scan.resolved[id]; !done {
			unresolved = append(unresolved, id)
		}
	}
	return unresolved
}

// rerunsOnResume reports whether the node opted into re-entry mode
// (NodeConfig.RerunOnResume), in which Resume re-runs the node with
// the user responses rather than handing off to its successors.
func rerunsOnResume(node Node) bool {
	if node == nil {
		return false
	}
	r := node.Config().RerunOnResume
	return r != nil && *r
}

// validateResolved validates each surviving (last-wins) response
// against its declared schema and returns the responses keyed by
// interrupt ID. A superseded invalid payload never reaches here.
func validateResolved(scan *nodeScanState) (map[string]any, error) {
	resumed := map[string]any{}
	for id, resp := range scan.resolved {
		if sc := scan.schemas[id]; sc != nil {
			validated, err := validateResumeResponse(resp, sc)
			if err != nil {
				return nil, fmt.Errorf("%w: interrupt %q: %w", ErrInvalidResumeResponse, id, err)
			}
			resp = validated
		}
		resumed[id] = resp
	}
	return resumed, nil
}

// inferNodeState maps a node's scan to a NodeState, mirroring
// adk-python _infer_node_state.
//
// Status priority:
//   - unresolved interrupts, re-run + some resolved -> NodePending
//     (partial resume: re-run with the resolved responses)
//   - unresolved interrupts otherwise               -> NodeWaiting
//   - all resolved, re-run                           -> NodePending (re-entry)
//   - all resolved, handoff                          -> NodeCompleted
//     with Output = the response (forwarded to successors by Resume)
func (w *Workflow) inferNodeState(node Node, scan *nodeScanState, nodeOutputs map[string]any, workflowInput any) (*NodeState, error) {
	unresolved := unresolvedInterrupts(scan)
	reenter := rerunsOnResume(node)

	resumed, err := validateResolved(scan)
	if err != nil {
		return nil, err
	}

	ns := &NodeState{Branch: scan.branch, interruptSchemas: scan.schemas}

	// A response seen for the first time this turn (count == 1) marks a
	// genuine first resume; a duplicate turn replays an already-counted
	// response (>= 2) and must stay a no-op. Every arm needs this, not just
	// the completed one: a node left waiting on its other interrupts still
	// took delivery of the answer that did arrive, so the turn is not the
	// empty no-op ErrNothingToResume reports.
	//
	// Recorded per ID. Collapsed to one bool per node it read true for the
	// rest of the session as soon as the node held a single never-replayed
	// answer, so a replay of any of its OTHER answers counted as new work
	// and re-triggered its successors.
	for id := range resumed {
		if scan.resolvedCount[id] == 1 {
			if ns.freshAnswers == nil {
				ns.freshAnswers = map[string]bool{}
			}
			ns.freshAnswers[id] = true
		}
	}

	switch {
	case len(unresolved) > 0 && reenter && len(resumed) > 0:
		// Partial resume: re-run with resolved responses so the node
		// can proceed or re-interrupt.
		ns.Status = NodePending
		ns.ResumedInputs = resumed
		ns.Interrupts = unresolved
		ns.Input, ns.TriggeredBy = w.predecessorInput(node, nodeOutputs, workflowInput)
		// Same replay guard as the all-resolved arm below: a node still
		// holding an open interrupt is no less exposed to a duplicate
		// answer than one that has none.
		ns.reentryConsumed = allConsumed(scan, resumed)
	case len(unresolved) > 0:
		// Still waiting for the remaining interrupts.
		ns.Status = NodeWaiting
		ns.Interrupts = unresolved
		if len(resumed) > 0 {
			ns.ResumedInputs = resumed
		}
	case reenter:
		// All resolved, re-entry: re-run with the responses.
		ns.Status = NodePending
		ns.ResumedInputs = resumed
		ns.Input, ns.TriggeredBy = w.predecessorInput(node, nodeOutputs, workflowInput)
		// Unless the node already ran on every one of them, in which
		// case this turn is a replay and re-running would re-do
		// whatever the human approved.
		ns.reentryConsumed = len(resumed) > 0 && allConsumed(scan, resumed)
	default:
		// All resolved, handoff: the node is done; its output is the
		// response, which Resume forwards to successors. Keep the
		// resolved responses so Resume can gate the idempotent
		// successor trigger on this turn's responses.
		ns.Status = NodeCompleted
		ns.Output = resumeOutput(resumed)
		ns.ResumedInputs = resumed
	}
	return ns, nil
}

// allConsumed reports whether the node has already acted on every one of the
// given resolved interrupts.
func allConsumed(scan *nodeScanState, resumed map[string]any) bool {
	for id := range resumed {
		if !scan.consumed[id] {
			return false
		}
	}
	return true
}

// predecessorInput walks incoming edges backward to find a resuming
// node's input: a predecessor's cached output, else the workflow seed
// input for a START successor. Mirrors adk-python
// _find_predecessor_input.
func (w *Workflow) predecessorInput(node Node, nodeOutputs map[string]any, workflowInput any) (any, string) {
	if node == nil {
		return nil, ""
	}
	incoming := w.graph.predecessorsOf(node)
	if len(incoming) == 0 {
		return nil, ""
	}
	for _, e := range incoming {
		from := e.From.Name()
		if from != Start.Name() {
			if out, ok := nodeOutputs[from]; ok {
				return out, from
			}
		}
	}
	for _, e := range incoming {
		if e.From.Name() == Start.Name() {
			return workflowInput, Start.Name()
		}
	}
	return nodeOutputs[incoming[0].From.Name()], incoming[0].From.Name()
}

// firstUserInput returns the seed workflow input: the text of the
// first user event in history (the original prompt), used as the
// START successor's input on re-entry. Resume turns (user
// FunctionResponses) are skipped. When invocationID is non-empty, only
// that invocation's user events are considered, so the seed is the
// current run's prompt rather than an earlier run's.
func firstUserInput(events session.Events, invocationID string) any {
	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		if ev == nil || ev.Author != "user" || ev.Content == nil {
			continue
		}
		if invocationID != "" && ev.InvocationID != invocationID {
			continue
		}
		var text string
		hasFR := false
		for _, p := range ev.Content.Parts {
			if p == nil {
				continue
			}
			if p.FunctionResponse != nil {
				hasFR = true
			}
			text += p.Text
		}
		if hasFR {
			continue
		}
		if text != "" {
			return text
		}
	}
	return nil
}

// eventNodeName returns the name of the static graph node that owns
// ev, for attribution during rehydration.
//
// Static node events are stamped with NodeInfo.Path == node name. A
// dynamic child invoked via RunNode carries a hierarchical path like
// "parent/child@1"; its interrupt is owned by the nearest static
// ancestor (the first path segment). Falls back to Author for the
// LlmAgent node path, where Author == node name and no path is set.
func eventNodeName(ev *session.Event, nodesByName map[string]Node) string {
	if ev.NodeInfo != nil && ev.NodeInfo.Path != "" {
		for _, seg := range strings.Split(ev.NodeInfo.Path, "/") {
			name := seg
			if idx := strings.IndexByte(name, '@'); idx >= 0 {
				name = name[:idx]
			}
			if _, ok := nodesByName[name]; ok {
				return name
			}
		}
	}
	return ev.Author
}

// settlesOwner reports whether ev shows owner's own activation finishing —
// producing its output or parking on a new interrupt — as opposed to something
// owner delegated to finishing under it. The distinction decides whether the
// answers owner holds count as acted upon: a child completing while the
// orchestrator goes on to fail must leave the orchestrator retryable.
//
// Two ways to qualify. The event came from owner's own activation, or its
// Output is attributed to that activation through OutputFor — which is how a
// WithUseAsOutput child's output becomes the orchestrator's, the orchestrator
// then emitting no terminal event of its own.
func settlesOwner(ev *session.Event, owner string, nodesByName map[string]Node) bool {
	if ownActivation(ev, owner, nodesByName) {
		return true
	}
	if ev.NodeInfo == nil {
		return false
	}
	for _, p := range ev.NodeInfo.OutputFor {
		if lastSegmentName(p) == owner {
			return true
		}
	}
	return false
}

// ownActivation reports whether ev came from owner's own activation rather
// than from something owner delegated to.
//
// eventNodeName attributes an event to the FIRST path segment naming a static
// graph node, so a delegated child's events carry their ancestor's name. The
// event is therefore the attributed node's own only when that first matching
// segment is also the last one — nothing is nested below it. Comparing only
// the last segment would get a self-recursive orchestrator ("orch@1/orch@2")
// backwards, calling the child's events the ancestor's.
//
// When no segment names a graph node, eventNodeName fell back to Author, which
// names a node and never a delegated child, so the event counts as the node's
// own.
func ownActivation(ev *session.Event, owner string, nodesByName map[string]Node) bool {
	if ev.NodeInfo == nil || ev.NodeInfo.Path == "" {
		return true
	}
	segs := strings.Split(ev.NodeInfo.Path, "/")
	for i, seg := range segs {
		name := segmentName(seg)
		if _, ok := nodesByName[name]; ok {
			return i == len(segs)-1 && name == owner
		}
	}
	return true
}

// segmentName strips the "@runID" suffix from one node-path segment.
func segmentName(seg string) string {
	if i := strings.IndexByte(seg, '@'); i >= 0 {
		return seg[:i]
	}
	return seg
}

// lastSegmentName returns the node name of a path's deepest segment.
func lastSegmentName(path string) string {
	if i := strings.LastIndexByte(path, '/'); i >= 0 {
		path = path[i+1:]
	}
	return segmentName(path)
}

// staticNodeName returns the static graph node owning a node path: the
// first segment of a composite "parent/child@run" path.
func staticNodeName(path string) string {
	if i := strings.IndexByte(path, '/'); i >= 0 {
		return path[:i]
	}
	return path
}

// frPart returns the FunctionResponse on a part if present and keyed.
func frPart(p *genai.Part) *genai.FunctionResponse {
	if p == nil || p.FunctionResponse == nil || p.FunctionResponse.ID == "" {
		return nil
	}
	return p.FunctionResponse
}

// schemaFromEvent re-extracts the response schema for interrupt id
// from the pause event (RequestedInput or the adk_request_input
// FunctionCall args), mirroring adk-python _extract_schema_from_event.
// The schema lives only in the events; it is not persisted.
func schemaFromEvent(ev *session.Event, id string) *jsonschema.Schema {
	if ev.RequestedInput != nil && ev.RequestedInput.InterruptID == id {
		return ev.RequestedInput.ResponseSchema
	}
	if ev.Content == nil {
		return nil
	}
	for _, p := range ev.Content.Parts {
		if p == nil {
			continue
		}
		fc := p.FunctionCall
		if fc == nil || fc.Name != WorkflowInputFunctionCallName || fc.ID != id {
			continue
		}
		if raw, ok := fc.Args["responseSchema"]; ok {
			if sc, ok := raw.(*jsonschema.Schema); ok {
				return sc
			}
		}
	}
	return nil
}
