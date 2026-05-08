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

// ErrUnconditionalCycle is returned when a cycle is detected that does not
// contain any conditional edges.
var ErrUnconditionalCycle = errors.New("unconditional cycle detected")

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
	if err := validateCycles(workflow); err != nil {
		return err
	}
	return nil
}

// validateCycles checks that there are no unconditional cycles in the workflow.
// It performs a depth first search for every node in the workflow and checks
// for cycles where all edges in the cycle have nil routes.
// Default routes (where Route == Default) are treated as conditional edges
// and are ignored during unconditional cycle detection.
func validateCycles(workflow *Workflow) error {
	visited := make(map[Node]struct{})

	var traverse func(n Node, inStack map[Node]struct{}) error
	traverse = func(n Node, inStack map[Node]struct{}) error {
		if _, ok := inStack[n]; ok {
			return fmt.Errorf("%w: %q", ErrUnconditionalCycle, n.Name())
		}

		if _, ok := visited[n]; ok {
			return nil
		}

		inStack[n] = struct{}{}
		visited[n] = struct{}{}

		for _, edge := range workflow.edges[n] {
			if edge.Route == nil {
				if err := traverse(edge.To, inStack); err != nil {
					return err
				}
			}
		}

		delete(inStack, n)
		return nil
	}

	for node := range workflow.edges {
		if _, ok := visited[node]; !ok {
			inStack := make(map[Node]struct{})
			if err := traverse(node, inStack); err != nil {
				return err
			}
		}
	}

	return nil
}
