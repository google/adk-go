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

package llminternal

import (
	"slices"
	"strings"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// FunctionCallIDs returns the non-empty IDs of the function calls in events.
// Nil events and events without content are skipped. The result is never
// nil; it is empty when events holds no call with an ID.
func FunctionCallIDs(events []*session.Event) map[string]struct{} {
	ids := make(map[string]struct{})
	for _, ev := range events {
		for _, call := range utils.FunctionCalls(utils.Content(ev)) {
			if call.ID != "" {
				ids[call.ID] = struct{}{}
			}
		}
	}
	return ids
}

// isToolSubBranchEvent reports whether ev was published by a tool on a
// sub-branch below agentBranch, such as a single_turn sub-agent run on
// "<agent>@<function_call_id>". The calling agent gets the tool's function
// response, so these events are not part of its context.
//
// An agent with a non-empty branch already does not see them, because a
// branch only sees itself and its ancestors. An agent with an empty branch
// sees every branch, so the tool's events would otherwise come back as
// another agent's messages on every later turn.
//
// Every segment below agentBranch is checked, so events nested further
// down a tool's sub-branch are skipped too. A segment naming agentName is
// the agent's own branch and is not skipped. A segment keyed by a run counter
// is a workflow node's branch and is not skipped either, unless a function
// call in the session has an id equal to that counter, since only the suffix
// is looked up in callIDs.
//
// Mirrors _is_tool_sub_branch_event in adk-python's
// flows/llm_flows/context/_contents.py, which has the same counter caveat.
func isToolSubBranchEvent(agentBranch string, ev *session.Event, agentName string, callIDs map[string]struct{}) bool {
	if ev.Branch == "" {
		return false
	}
	var base []string
	if agentBranch != "" {
		base = strings.Split(agentBranch, ".")
	}
	segments := strings.Split(ev.Branch, ".")
	if len(segments) <= len(base) || !slices.Equal(segments[:len(base)], base) {
		return false
	}
	for _, segment := range segments[len(base):] {
		i := strings.LastIndex(segment, "@")
		if i < 0 {
			continue
		}
		if _, ok := callIDs[segment[i+1:]]; ok && segment[:i] != agentName {
			return true
		}
	}
	return false
}

// IsToolBranch reports whether branch is the sub-branch a tool publishes its
// user-facing message on, rather than a branch the node nodeName runs on.
// callIDs holds the IDs of the function calls recorded in the session (see
// FunctionCallIDs).
//
// branch is "" or a dot-separated path of segments. It is a tool branch when
// its last segment is "<name>@<id>", split at the last "@", where <id> is in
// callIDs and <name> is not nodeName. An agent run as a tool is scoped on
// "<parent>.<agent>@<function call id>", which names the agent itself, so it
// is that agent's own branch. An empty branch, a last segment without "@",
// and an ID not in callIDs all give false. Port of adk-python's
// _BranchPath.is_tool_branch (events/_branch_path.py).
func IsToolBranch(branch, nodeName string, callIDs map[string]struct{}) bool {
	segments := strings.Split(branch, ".")
	leaf := segments[len(segments)-1]
	i := strings.LastIndex(leaf, "@")
	if i < 0 {
		return false
	}
	if _, ok := callIDs[leaf[i+1:]]; !ok {
		return false
	}
	return leaf[:i] != nodeName
}
