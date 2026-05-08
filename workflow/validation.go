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
	"errors"
	"fmt"
)

// ErrDuplicateNodeName is returned when an edge set contains two
// distinct Node instances that share the same Name.
var ErrDuplicateNodeName = errors.New("duplicate node name")

// ErrDuplicateEdge is returned when an edge set contains two identical edges.
// Two edges with the same (From, To) are rejected regardless of Route; use
// MultiRoute to express alternatives to the same target.
var ErrDuplicateEdge = errors.New("duplicate edge")

// validateUniqueNames checks that all nodes in the edge set have unique names.
// If duplicate node names are found, it returns an error. The equality between
// nodes is checked by comparing the nodes directly.
func validateUniqueNames(edges []Edge) error {
	names := make(map[string]Node)
	checkNode := func(node Node) error {
		if storedNode, ok := names[node.Name()]; ok {
			if storedNode != node {
				return fmt.Errorf("%w: %s", ErrDuplicateNodeName, node.Name())
			}
		} else {
			names[node.Name()] = node
		}
		return nil
	}
	for _, edge := range edges {
		if err := checkNode(edge.From); err != nil {
			return err
		}
		if err := checkNode(edge.To); err != nil {
			return err
		}
	}
	return nil
}

// validateWorkflow executes a set of workflow validation checks.
func validateWorkflow(workflow *Workflow) error {
	if err := validateUniqueEdges(workflow); err != nil {
		return err
	}
	return nil
}

// validateUniqueEdges checks that there are no duplicate edges in the workflow.
// Two edges with the same (From, To) are rejected regardless of Route; use
// MultiRoute to express alternatives to the same target.
func validateUniqueEdges(workflow *Workflow) error {
	for node, edges := range workflow.edges {
		uniqueEdges := make(map[Node]struct{})
		for _, edge := range edges {
			if _, ok := uniqueEdges[edge.To]; ok {
				return fmt.Errorf("%w: from %q to %q", ErrDuplicateEdge, node.Name(), edge.To.Name())
			}
			uniqueEdges[edge.To] = struct{}{}
		}
	}
	return nil
}
