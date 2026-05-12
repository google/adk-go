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

// ErrJoinNodeNoIncoming is returned when a JoinNode appears in the
// edge set without any incoming edges. A JoinNode without
// predecessors would never have anything to aggregate; this is
// almost always a graph-construction mistake. adk-python catches
// the same condition at runtime; we catch it at build time so the
// error surfaces before the workflow runs.
var ErrJoinNodeNoIncoming = errors.New("join node has no incoming edges")

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
	if err := validateJoinNodesHaveIncoming(edges); err != nil {
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

// validateJoinNodesHaveIncoming checks that every JoinNode appearing
// in the edge set has at least one incoming edge. A JoinNode with
// no predecessors has nothing to aggregate and would either dead-end
// silently (if it never gets activated) or fail at runtime.
//
// JoinNodes that appear only as edge sources (never as targets)
// trip this check; JoinNodes that appear only as edge targets are
// fine — they have predecessors but no successors, which is a valid
// terminal-fan-in configuration.
func validateJoinNodesHaveIncoming(edges []Edge) error {
	joinSeen := map[Node]struct{}{}
	hasIncoming := map[Node]struct{}{}
	for _, edge := range edges {
		if _, ok := edge.From.(*JoinNode); ok {
			joinSeen[edge.From] = struct{}{}
		}
		if _, ok := edge.To.(*JoinNode); ok {
			joinSeen[edge.To] = struct{}{}
			hasIncoming[edge.To] = struct{}{}
		}
	}
	for jn := range joinSeen {
		if _, ok := hasIncoming[jn]; !ok {
			return fmt.Errorf("%w: %s", ErrJoinNodeNoIncoming, jn.Name())
		}
	}
	return nil
}
