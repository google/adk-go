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
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/session"
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
	// schemas maps an interrupt ID to its declared response schema,
	// re-extracted from the pause FunctionCall args.
	schemas map[string]*jsonschema.Schema
	branch  string
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
func (w *Workflow) ReconstructRunState(sess session.Session) (*RunState, error) {
	if sess == nil {
		return nil, nil
	}
	nodesByName := buildNodesByName(w.graph)
	events := sess.Events()

	// Stage 1: scan history into a per-node view of the pause
	// (interrupts raised, responses that resolved them, schemas).
	scans := scanHistory(events, nodesByName)

	// Stage 2: gather the inputs inferNodeState needs to rebuild a
	// re-entry node's input: every node's cached output, the set of
	// nodes that ran, and the workflow's seed input.
	nodeOutputs, completed := collectNodeOutputs(events, nodesByName)
	workflowInput := firstUserInput(events)

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
// interrupt history are returned.
func scanHistory(events session.Events, nodesByName map[string]Node) map[string]*nodeScanState {
	scans := map[string]*nodeScanState{}
	interruptOwner := map[string]string{} // interrupt ID -> node name
	scanFor := func(name string) *nodeScanState {
		s := scans[name]
		if s == nil {
			s = &nodeScanState{resolved: map[string]any{}, schemas: map[string]*jsonschema.Schema{}}
			scans[name] = s
		}
		return s
	}

	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		if ev == nil {
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
				scanFor(owner).resolved[fr.ID] = unwrapResponse(fr.Response)
			}
			continue
		}

		// Interrupts the node raised, attributed to the static graph
		// node that emitted the event (NodeInfo.Path; dynamic children
		// fold into their static ancestor — see eventNodeName).
		owner := eventNodeName(ev)
		if _, ok := nodesByName[owner]; !ok {
			continue
		}
		s := scanFor(owner)
		if ev.Output != nil {
			s.branch = ev.Branch
		}
		for _, id := range ev.LongRunningToolIDs {
			if id == "" {
				continue
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
// nodes; completed lets Resume skip already-run successors.
func collectNodeOutputs(events session.Events, nodesByName map[string]Node) (outputs map[string]any, completed map[string]bool) {
	outputs = map[string]any{}
	completed = map[string]bool{}
	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		if ev == nil {
			continue
		}
		name := eventNodeName(ev)
		if _, ok := nodesByName[name]; !ok {
			continue
		}
		completed[name] = true
		if ev.Output != nil {
			outputs[name] = ev.Output
			// A delegated output (WithUseAsOutput) also counts for the
			// static owners of the ancestor paths in OutputFor, so they
			// recover their output without a re-emitted event.
			for _, p := range ev.NodeInfo.OutputFor {
				if owner := staticOwner(p); owner != name {
					if _, ok := nodesByName[owner]; ok {
						outputs[owner] = ev.Output
					}
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

	switch {
	case len(unresolved) > 0 && reenter && len(resumed) > 0:
		// Partial resume: re-run with resolved responses so the node
		// can proceed or re-interrupt.
		ns.Status = NodePending
		ns.ResumedInputs = resumed
		ns.Interrupts = unresolved
		ns.Input, ns.TriggeredBy = w.predecessorInput(node, nodeOutputs, workflowInput)
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
// FunctionResponses) are skipped.
func firstUserInput(events session.Events) any {
	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		if ev == nil || ev.Author != "user" || ev.Content == nil {
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
func eventNodeName(ev *session.Event) string {
	if ev.NodeInfo != nil && ev.NodeInfo.Path != "" {
		return staticOwner(ev.NodeInfo.Path)
	}
	return ev.Author
}

// staticOwner returns the static graph node name owning a node path:
// the first segment of a hierarchical "parent/child@1" path.
func staticOwner(path string) string {
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

// unwrapResponse extracts the original value from a FunctionResponse
// payload. A sole single-key wrapper — {"result": v} (adk-python),
// {"response": v} or {"payload": v} (adk-go) — is unwrapped, with
// string values JSON-parsed when possible; anything else passes
// through. Mirrors adk-python _unwrap_response, extended with the
// adk-go keys for cross-runtime sessions.
func unwrapResponse(data map[string]any) any {
	if len(data) != 1 {
		return data
	}
	for _, key := range []string{"result", "response", "payload"} {
		v, ok := data[key]
		if !ok {
			continue
		}
		if s, isStr := v.(string); isStr {
			var parsed any
			if err := json.Unmarshal([]byte(s), &parsed); err == nil {
				return parsed
			}
			return s
		}
		return v
	}
	return data
}
