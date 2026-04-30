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

import "errors"

// startNode is the START sentinel — a Node that never executes. Edges
// outgoing from START seed the workflow's initial triggers with the
// workflow's input.
type startNode struct {
	Base
}

// RunImpl is never invoked by the engine; START is treated specially
// during graph traversal. We implement it for interface satisfaction.
func (s *startNode) RunImpl(ctx *NodeContext, _ any, _ EventEmitter) error {
	return errors.New("workflow: START sentinel must not run")
}

// START is the entry-point sentinel. Connect edges from START to the
// nodes that should fire on workflow entry. Equivalent to adk-python's
// workflow.START.
var START Node = newStart()

func newStart() *startNode {
	s := &startNode{}
	if err := s.SetMetadata("__START__", "Workflow start sentinel", NodeSpec{}); err != nil {
		// Should never fail: name is hardcoded valid.
		panic(err)
	}
	return s
}

// IsStart reports whether n is the START sentinel.
func IsStart(n Node) bool { return n == START }
