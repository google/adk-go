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

package session_test

import (
	"fmt"

	"google.golang.org/adk/v2/session"
)

// A node path joins one segment per workflow level with "/", and each segment
// may carry its run id after "@". The accessors read the last two segments.
func ExampleNodeInfo_ParentRunID() {
	child := &session.NodeInfo{Path: "workflow@10/child_node@20"}
	fmt.Println(child.Name(), child.RunID())
	fmt.Println(child.ParentRunID())

	root := &session.NodeInfo{Path: "root_node@5"}
	id, ok := root.ParentRunID()
	fmt.Printf("%q %v\n", id, ok)

	// A nil NodeInfo, as on the user's message event or an event built by
	// hand, is valid.
	var none *session.NodeInfo
	fmt.Printf("%q\n", none.Name())
	// Output:
	// child_node 20
	// 10 true
	// "" false
	// ""
}
