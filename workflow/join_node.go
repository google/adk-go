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

// JoinNode is a fan-in node that waits for all of its predecessors to
// complete before running. Its input is a map[string]any keyed by
// predecessor name, carrying each predecessor's last output.
//
// JoinNode passes the aggregated map through as its output, mirroring
// adk-python's _join_node.JoinNode. To produce a derived output instead,
// wrap a Func node downstream.
type JoinNode struct {
	Base
}

// Join constructs a JoinNode with RequiresAllPredecessors set.
func Join(name string, opts ...NodeOpt) *JoinNode {
	o := applyOpts(opts)
	o.requiresAllPredecessors = true
	n := &JoinNode{}
	if err := n.SetMetadata(name, o.description, o.toSpec()); err != nil {
		panic(err)
	}
	return n
}

// RunImpl returns the aggregated predecessor outputs as the node's output.
func (j *JoinNode) RunImpl(_ *NodeContext, input any, em EventEmitter) error {
	return em.Output(input)
}
