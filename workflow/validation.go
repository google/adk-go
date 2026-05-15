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
	"slices"
	"strings"
)

// ErrDuplicateNodeName is returned when an edge set contains two
// distinct Node instances that share the same Name.
var ErrDuplicateNodeName = errors.New("duplicate node name")

// ErrNoStartNode is returned when no start node is found in the edge set.
var ErrNoStartNode = errors.New("no start node found")

// ErrNodePointsToStart is returned when a node points to the start node.
var ErrNodePointsToStart = errors.New("node points to start node")

// ErrDuplicateEdge is returned when an edge set contains two identical edges.
// Two edges with the same (From, To) are rejected regardless of Route; use
// MultiRoute to express alternatives to the same target.
var ErrDuplicateEdge = errors.New("duplicate edge")

// ErrMultipleDefaultRoutes is returned when a node has more than one default route.
var ErrMultipleDefaultRoutes = errors.New("node has more than one default route")

// ErrNodesNotReachable is returned when some nodes are not reachable from the start node.
var ErrNodesNotReachable = errors.New("nodes not reachable from start node")

// ErrUnconditionalCycle is returned when a cycle is detected that does not
// contain any conditional edges.
var ErrUnconditionalCycle = errors.New("unconditional cycle detected")

// validateNodes executes a set of edges validation checks.
func validateNodes(edges []Edge) error {
	if err := validateUniqueNames(edges); err != nil {
		return err
	}
	if err := validateStartNodePresent(edges); err != nil {
		return err
	}
	if err := validateStartNodeNoIncoming(edges); err != nil {
		return err
	}
	return nil
}

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

// validateStartNodePresent checks that there is at least one edge starting from the start node.
func validateStartNodePresent(edges []Edge) error {
	for _, edge := range edges {
		if edge.From == Start {
			return nil
		}
	}
	return ErrNoStartNode
}

// validateStartNodeNoIncoming checks that no node points to the start node.
func validateStartNodeNoIncoming(edges []Edge) error {
	for _, edge := range edges {
		if edge.To == Start {
			return fmt.Errorf("%w: %s", ErrNodePointsToStart, edge.From.Name())
		}
	}
	return nil
}

// validateWorkflow executes a set of workflow validation checks.
func validateWorkflow(workflow *graph) error {
	if err := validateUniqueEdges(workflow); err != nil {
		return err
	}
	if err := validateDefaultRoute(workflow); err != nil {
		return err
	}
	if err := validateConnectivity(workflow); err != nil {
		return err
	}
	if err := validateCycles(workflow); err != nil {
		return err
	}
	return nil
}

// validateUniqueEdges checks that there are no duplicate edges in the workflow.
// Two edges with the same (From, To) are rejected regardless of Route; use
// MultiRoute to express alternatives to the same target.
func validateUniqueEdges(workflow *graph) error {
	for node, edges := range workflow.successors {
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

// validateDefaultRoute checks that there are no multiple default routes for one node.
func validateDefaultRoute(workflow *graph) error {
	for node, edges := range workflow.successors {
		hasDefault := false
		for _, edge := range edges {
			if edge.Route == Default && !hasDefault {
				hasDefault = true
			} else if edge.Route == Default && hasDefault {
				return fmt.Errorf("%w: %q", ErrMultipleDefaultRoutes, node.Name())
			}
		}
	}
	return nil
}

// validateConnectivity checks that all nodes in the edge set are reachable from the start node.
func validateConnectivity(workflow *graph) error {
	if len(workflow.successors) == 0 {
		return nil
	}

	visited := make(map[Node]bool)
	var traverse func(n Node)
	traverse = func(n Node) {
		visited[n] = true
		for _, neighbor := range workflow.successors[n] {
			if !visited[neighbor.To] {
				traverse(neighbor.To)
			}
		}
	}

	traverse(Start)

	allNodes := make(map[Node]bool)
	for node, edges := range workflow.successors {
		allNodes[node] = true
		for _, edge := range edges {
			allNodes[edge.To] = true
		}
	}

	var unreachable []string
	for node := range allNodes {
		if !visited[node] {
			unreachable = append(unreachable, node.Name())
		}
	}
	slices.Sort(unreachable)

	if len(unreachable) > 0 {
		return fmt.Errorf("%w: %q", ErrNodesNotReachable, strings.Join(unreachable, ", "))
	}

	return nil
}

// validateCycles checks that there are no unconditional cycles in the workflow.
// It performs a depth first search for every node in the workflow and checks
// for cycles where all edges in the cycle have nil routes.
// Default routes (where Route == Default) are treated as conditional edges
// and are ignored during unconditional cycle detection.
func validateCycles(workflow *graph) error {
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

		for _, edge := range workflow.successors[n] {
			if edge.Route == nil {
				if err := traverse(edge.To, inStack); err != nil {
					return err
				}
			}
		}

		delete(inStack, n)
		return nil
	}

	for node := range workflow.successors {
		if _, ok := visited[node]; !ok {
			inStack := make(map[Node]struct{})
			if err := traverse(node, inStack); err != nil {
				return err
			}
		}
	}

	return nil
}
