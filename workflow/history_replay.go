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
	"cmp"
	"slices"
	"strconv"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/session"
)

// triggerReplay is what replayHistory learns about the triggers of one
// invocation's static activations.
type triggerReplay struct {
	maxRun  map[string]int                       // highest run id seen per node name
	trigger map[string]map[int]pendingActivation // the trigger each seen activation consumed: input and triggeredBy
	queued  map[string][]pendingActivation       // triggers produced and not consumed, oldest first
	bad     map[string]bool                      // nodes whose history does not replay
}

// segmentKey names one activation in history: "<name>@<run>".
type segmentKey struct {
	name string
	run  int
}

// segmentState is what history says about one activation.
type segmentState struct {
	terminal  NodeStatus // kind of the latest terminal event; NodeRunning when there is none
	output    any
	hasOutput bool
	routingEv *session.Event
	branch    string
	open      []string                      // interrupts raised and not answered
	answers   map[string]any                // accepted answer per interrupt
	schemas   map[string]*jsonschema.Schema // declared response schema per interrupt
	pos       int                           // index of the latest terminal event, else of the first event
}

func (st *segmentState) clearResult() {
	st.output, st.hasOutput, st.routingEv = nil, false, nil
}

// status is the activation's state as the live scheduler would have
// left it: an open interrupt wins over any result.
func (st *segmentState) status() NodeStatus {
	if len(st.open) > 0 {
		return NodeWaiting
	}
	return st.terminal
}

// eventSegment returns the activation that owns path: the first segment
// that names a node of the graph, when it carries "@<run id>". own is
// true when that segment is the last one (the activation's own event,
// not a dynamic descendant's). ok is false when the first graph-node
// segment has no run id (a root wrapper, older history) or no segment
// names a graph node.
func eventSegment(path string, nodesByName map[string]Node) (key segmentKey, own, ok bool) {
	if path == "" {
		return segmentKey{}, false, false
	}
	segments := strings.Split(path, "/")
	for i, segment := range segments {
		name, runID, hasRun := strings.Cut(segment, "@")
		if _, known := nodesByName[name]; !known {
			continue
		}
		run, err := strconv.Atoi(runID)
		if !hasRun || err != nil || run < 1 {
			return segmentKey{}, false, false
		}
		return segmentKey{name: name, run: run}, i == len(segments)-1, true
	}
	return segmentKey{}, false, false
}

// eventPath returns ev's node path, "" when it has none.
func eventPath(ev *session.Event) string {
	if ev.NodeInfo == nil {
		return ""
	}
	return ev.NodeInfo.Path
}

// eventActivation returns the activation that owns ev (see eventSegment).
func eventActivation(ev *session.Event, nodesByName map[string]Node) (segmentKey, bool) {
	key, _, ok := eventSegment(eventPath(ev), nodesByName)
	return key, ok
}

// replayHistory dry-runs trigger generation over the invocation's events. nil
// when events hold no segment.
//
// It reads the events scanHistory reads, groups them by activation
// (eventSegment), and takes each activation's outcome from its latest
// terminal event: an own output, route, completion record or failure
// outcome, or an own or descendant interrupt. A user response its schema
// accepts reopens the activation and discards its result, as scanHistory
// does; when it answers the last open interrupt of a handoff node, the
// response is the activation's output, as Resume makes it, and the user
// event is its terminal event. It then replays the completions in the
// order of their latest terminal events through the scheduler's own trigger
// logic (findSuccessors, bufferSuccessors), with START seeded as
// Workflow.RunNode seeds it. Activation k of a node consumes that
// node's k-th trigger; a node whose history needs a trigger the replay
// did not produce is marked bad, and callers fall back for it.
func replayHistory(g *graph, events session.Events, nodesByName map[string]Node,
	invocationID, workflowName string, workflowInput any,
) *triggerReplay {
	segs := map[segmentKey]*segmentState{}
	interruptOwner := map[string]segmentKey{}
	maxRun := map[string]int{}

	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		if ev == nil || ev.Partial || (invocationID != "" && ev.InvocationID != invocationID) {
			continue
		}
		ev = graphHistoryEvent(ev, workflowName)

		if ev.Author == "user" && ev.Content != nil {
			for _, p := range ev.Content.Parts {
				fr := frPart(p)
				if fr == nil {
					continue
				}
				key := interruptOwner[fr.ID]
				if st := segs[key]; st != nil && replayAnswer(st, fr.ID, unwrapResponse(fr.Response)) &&
					len(st.open) == 0 && !g.rerunsOnResume(nodesByName[key.name]) {
					st.terminal = NodeCompleted
					st.output, st.hasOutput = resumeOutput(st.answers), true
					st.pos = i
				}
			}
			continue
		}

		key, own, ok := eventSegment(eventPath(ev), nodesByName)
		if !ok || key.name == Start.Name() {
			continue
		}
		maxRun[key.name] = max(maxRun[key.name], key.run)
		st := segs[key]
		if st == nil {
			st = &segmentState{terminal: NodeRunning, answers: map[string]any{}, schemas: map[string]*jsonschema.Schema{}, pos: i}
			segs[key] = st
		}
		terminal := false
		for _, id := range ev.LongRunningToolIDs {
			if id == "" {
				continue
			}
			terminal = true
			interruptOwner[id] = key
			if sc := schemaFromEvent(ev, id); sc != nil {
				st.schemas[id] = sc
			}
			if _, answered := st.answers[id]; !answered && !slices.Contains(st.open, id) {
				st.open = append(st.open, id)
			}
		}
		if own {
			if ev.Branch != "" {
				st.branch = ev.Branch
			}
			if replayOwnEvent(st, ev) {
				terminal = true
			}
		}
		if terminal {
			st.pos = i
		}
	}
	if len(segs) == 0 {
		return nil
	}

	order := make([]segmentKey, 0, len(segs))
	for key := range segs {
		order = append(order, key)
	}
	// One user event can complete several handoff askers; order them by
	// name so the replay is deterministic.
	slices.SortFunc(order, func(a, b segmentKey) int {
		return cmp.Or(segs[a].pos-segs[b].pos, strings.Compare(a.name, b.name), a.run-b.run)
	})

	r := &triggerReplay{
		maxRun:  maxRun,
		trigger: map[string]map[int]pendingActivation{},
		queued:  map[string][]pendingActivation{},
		bad:     map[string]bool{},
	}
	dry := &scheduler{state: NewRunState(), graph: g, nodesByName: nodesByName, lastOutputs: map[string]any{}}
	dry.state.Nodes[Start.Name()] = &NodeState{Status: NodeCompleted, Input: workflowInput}
	dry.bufferSuccessors(Start.Name(), findSuccessors(g, Start, workflowInput, nil, ""))

	consumed := map[string]int{}
	for _, key := range order {
		if r.bad[key.name] {
			continue
		}
		t, ok := consumeTrigger(dry, consumed, key)
		if !ok {
			r.bad[key.name] = true
			continue
		}
		if r.trigger[key.name] == nil {
			r.trigger[key.name] = map[int]pendingActivation{}
		}
		r.trigger[key.name][key.run] = t
		replayCompletion(dry, nodesByName[key.name], segs[key])
	}
	for _, name := range dry.triggers.names() {
		r.queued[name] = dry.queuedFor(name)
	}
	return r
}

// replayAnswer applies a user response to interrupt id of st and reports
// whether it was consumed: the first answer its schema accepts. It
// closes the interrupt and discards the activation's result. A repeated
// answer does not reopen work that completed after the first one, and a
// rejected answer leaves the interrupt open, as Resume does.
func replayAnswer(st *segmentState, id string, resp any) bool {
	if _, answered := st.answers[id]; answered {
		return false
	}
	if sc := st.schemas[id]; sc != nil {
		validated, err := validateResumeResponse(resp, sc)
		if err != nil {
			return false
		}
		resp = validated
	}
	st.answers[id] = resp
	st.open = slices.DeleteFunc(st.open, func(open string) bool { return open == id })
	st.clearResult()
	return true
}

// replayOwnEvent applies an activation's own event to st, as handleEvent
// and handleCompletion would, and reports whether it is terminal.
func replayOwnEvent(st *segmentState, ev *session.Event) bool {
	if ev.CustomMetadata[workflowNodeCompletedKey] == true {
		st.terminal = NodeCompleted
		if value, delegated := completedDelegatedOutput(ev); delegated && value != nil {
			st.output, st.hasOutput = value, true
		}
		return true
	}
	out, hasOutput := childEventOutput(ev)
	if hasOutput || ev.Routes != nil {
		st.terminal = NodeCompleted
		if hasOutput {
			st.output, st.hasOutput = out, true
		}
		if ev.Routes != nil {
			st.routingEv = ev
		}
		return true
	}
	if workflowNodeOutcome(ev) != "" {
		// A failed or canceled attempt; a retry starts the
		// activation's result again.
		st.terminal = NodeFailed
		st.clearResult()
		return true
	}
	return false
}

// consumeTrigger takes the triggers of key.name from the dry buffer
// until key.run of them have been taken in total and returns the last
// one, the trigger of activation key.run. The ones before it belong to
// activations that left no event. ok is false when the buffer runs out
// or an earlier activation was already replayed after a later one.
func consumeTrigger(dry *scheduler, consumed map[string]int, key segmentKey) (pendingActivation, bool) {
	if consumed[key.name] >= key.run {
		return pendingActivation{}, false
	}
	var t pendingActivation
	for consumed[key.name] < key.run {
		next, ok := dry.dequeue(key.name)
		if !ok {
			return pendingActivation{}, false
		}
		t = next
		consumed[key.name]++
	}
	return t, true
}

// replayCompletion sets node's dry state from st and, for a completion
// that triggers successors, buffers them as handleCompletion does.
func replayCompletion(dry *scheduler, node Node, st *segmentState) {
	name := node.Name()
	ns := dry.state.EnsureNode(name)
	ns.Branch = st.branch
	ns.Output = nil
	ns.Interrupts = slices.Clone(st.open)
	ns.Status = st.status()
	if ns.Status != NodeCompleted {
		return
	}
	if st.hasOutput {
		ns.Output = st.output
		if st.output != nil {
			dry.lastOutputs[name] = st.output
		}
	}
	dry.bufferSuccessors(name, findSuccessors(dry.graph, node, st.output, st.routingEv, st.branch))
}
