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

import "strconv"

// resumeBranch is the branch node resumes on: that of its first own event
// when it has one, else the branch of the trigger of its latest activation
// (deriveSubBranch(t.branch, name+"@"+run) when the trigger puts the node
// on a sub-branch), else, when history does not replay, the scan's branch.
//
// node and replay may be nil; scan must not be.
func resumeBranch(node Node, scan *nodeScanState, replay *triggerReplay) string {
	if scan.branchKnown || replay == nil || node == nil {
		return scan.branch
	}
	name := node.Name()
	if replay.bad[name] {
		return scan.branch
	}
	run := replay.maxRun[name]
	t, ok := replay.trigger[name][run]
	if !ok {
		return scan.branch
	}
	if t.subBranch {
		return deriveSubBranch(t.branch, name+"@"+strconv.Itoa(run))
	}
	return t.branch
}
