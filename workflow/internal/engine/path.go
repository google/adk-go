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

// Package engine implements the workflow orchestration loop. It is
// internal to the workflow package; the public API lives one directory
// up.
package engine

import "fmt"

// JoinPath composes a child node_path from a parent path, the child node
// name, and the per-name run counter.
//
//   - JoinPath("", "wf", 1)              => "wf@1"
//   - JoinPath("wf@1", "classify", 1)    => "wf@1/classify@1"
//   - JoinPath("wf@1/loop@1", "step", 3) => "wf@1/loop@1/step@3"
func JoinPath(parent, name string, runID int) string {
	seg := fmt.Sprintf("%s@%d", name, runID)
	if parent == "" {
		return seg
	}
	return parent + "/" + seg
}
