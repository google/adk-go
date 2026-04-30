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
	"strings"

	"google.golang.org/adk/agent"
)

// rehydrate reconstructs the run state from session events so a workflow
// resuming a prior invocation skips already-completed nodes and continues
// from the first non-completed point.
//
// Phase 4 scope (mirrors a subset of adk-python _restore_static_nodes_from_events):
//   - Direct children of this workflow path that have a recorded Output are
//     marked COMPLETED. Their cached outputs are replayed through
//     state.complete so successor nodes get the right input.
//   - Already-completed nodes are then stripped from the queue so they
//     don't re-run.
//   - WAITING nodes (unresolved interrupts), dynamic-node rehydration, and
//     RerunOnResume=false fast-path land in Phase 5 alongside the HITL
//     plumbing they depend on.
//
// The function is a no-op when:
//   - ic.Session() returns nil
//   - the session has no events for this workflow path
func (w *Workflow) rehydrate(state *runState, ic agent.InvocationContext, workflowPath string) {
	if ic == nil {
		return
	}
	sess := ic.Session()
	if sess == nil {
		return
	}
	evs := sess.Events()
	if evs == nil || evs.Len() == 0 {
		return
	}

	type repl struct {
		name   string
		runID  int
		output any
	}

	var ordered []repl
	prefix := workflowPath + "/"
	for ev := range evs.All() {
		if ev == nil || ev.Actions.NodeInfo == nil {
			continue
		}
		ni := ev.Actions.NodeInfo
		if !strings.HasPrefix(ni.Path, prefix) {
			continue
		}
		rest := strings.TrimPrefix(ni.Path, prefix)
		// Direct child only: no nested slashes.
		if strings.Contains(rest, "/") {
			continue
		}
		name, _ := splitNameAndRun(rest)
		if name == "" {
			continue
		}
		// Phase 4: only completed-output events drive replay.
		if ni.Output == nil {
			continue
		}
		// Honor RerunOnResume: nodes flagged for rerun re-execute every
		// resume, so we don't replay their cached output.
		if node, ok := w.graph.nodes[name]; ok && node.Spec().RerunOnResume {
			continue
		}
		ordered = append(ordered, repl{name: name, output: ni.Output})
	}

	if len(ordered) == 0 {
		return
	}

	// Replay completions in event order. state.complete queues successors;
	// they'll be filtered out below if they were also completed.
	for _, r := range ordered {
		state.complete(r.name, []any{r.output}, nil)
	}

	// Strip already-completed nodes from the queue so they don't re-run.
	completedSet := map[string]bool{}
	for _, r := range ordered {
		completedSet[r.name] = true
	}
	state.mu.Lock()
	keep := state.queue[:0]
	for _, ref := range state.queue {
		if completedSet[ref.name] {
			continue
		}
		keep = append(keep, ref)
	}
	state.queue = keep
	state.mu.Unlock()
}

// splitNameAndRun parses a "name@runID" segment, returning ("", 0) on
// malformed input.
func splitNameAndRun(seg string) (string, int) {
	idx := strings.LastIndexByte(seg, '@')
	if idx < 0 || idx == 0 || idx == len(seg)-1 {
		return "", 0
	}
	name := seg[:idx]
	// We don't strictly need the run ID for Phase 4 replay.
	return name, 0
}
