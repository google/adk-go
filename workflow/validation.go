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

// ErrNoStartNode is returned when no start node is found in the edge set.
var ErrNoStartNode = errors.New("no start node found")

// ErrNodePointsToStart is returned when a node points to the start node.
var ErrNodePointsToStart = errors.New("node points to start node")

// ErrMultipleDefaultRoutes is returned when a node has more than one default route.
var ErrMultipleDefaultRoutes = errors.New("node has more than one default route")

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
func validateWorkflow(workflow *Workflow) error {
	if err := validateDefaultRoute(workflow); err != nil {
		return err
	}
	return nil
}

// validateDefaultRoute checks that there are no multiple default routes for one node.
func validateDefaultRoute(workflow *Workflow) error {
	for node, edges := range workflow.graph.successors {
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
