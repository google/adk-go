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
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/session"
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

// ErrSubWorkflowNameCollision is returned when a sub-workflow has the same name as the parent workflow.
var ErrSubWorkflowNameCollision = errors.New("sub-workflow name collision")

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
	if err := validateNoTaskModeGraphNodes(edges); err != nil {
		return err
	}
	return nil
}

// validateSubWorkflowNames checks that no sub-workflow has the same name as the parent workflow.
func validateSubWorkflowNames(workflowName string, edges []Edge) error {
	for _, edge := range edges {
		if err := checkSubWorkflowName(edge.From, workflowName); err != nil {
			return err
		}
		if err := checkSubWorkflowName(edge.To, workflowName); err != nil {
			return err
		}
	}
	return nil
}

// checkSubWorkflowName checks if the node is a WorkflowNode and if its sub-workflow has the same name as the parent workflow.
func checkSubWorkflowName(node Node, workflowName string) error {
	if wfNode, ok := node.(*WorkflowNode); ok {
		if wfNode.subWorkflow.Name() == workflowName {
			return fmt.Errorf("%w: %q", ErrSubWorkflowNameCollision, workflowName)
		}
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
func validateWorkflow(workflow *graph, schema *jsonschema.Resolved) error {
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
	if err := validateStateSchemaConsistency(workflow, schema); err != nil {
		return err
	}
	return nil
}

// validateNoTaskModeGraphNodes rejects task-mode LlmAgents that appear
// as static workflow graph nodes.
//
// Task-mode agents are multi-turn — they pause for user replies and
// expect the original node_input (the task brief) to remain visible
// across re-dispatches. The workflow scheduler currently overwrites
// node_input with the latest user message on every re-entry, so the
// task brief is lost and the agent loses context. Until the scheduler
// preserves the originating node_input on resume, task agents may only
// be used:
//
//   - as chat sub-agents of an LlmAgent coordinator (FC delegation via
//     workflowinternal.TaskAgentTool / dispatchTaskFC), or
//   - dispatched dynamically via workflow.RunNode from a function/
//     dynamic node — never as static graph nodes.
func validateNoTaskModeGraphNodes(edges []Edge) error {
	allNodes := make(map[Node]bool)
	for _, e := range edges {
		allNodes[e.From] = true
		allNodes[e.To] = true
	}

	for node := range allNodes {
		agentNode, ok := node.(*AgentNode)
		if !ok {
			continue
		}
		llmA, ok := agentNode.agent.(llminternal.Agent)
		if !ok || llmA == nil {
			continue
		}

		if llminternal.Reveal(llmA).Mode == llminternal.ModeTask {
			return fmt.Errorf(
				"Agent %q has mode='task' and cannot be used as a workflow graph node. Use a chat coordinator with task sub-agents, or "+
					"dispatch dynamically via RunNode from a function node",
				node.Name(),
			)
		}
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

	var unreachable []string
	for _, node := range workflow.allNodes() {
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

// defaultValidateInput validates data against schema. When data is a
// string and the schema expects a non-string shape, it first tries to
// parse the string as JSON. If parsing or validation fails, it falls
// back to validating the raw string (useful for enum/literal schemas).
// If both attempts fail, it returns the error from standard schema
// validation.
func defaultValidateInput(data any, schema *jsonschema.Resolved) (any, error) {
	if schema == nil {
		return data, nil
	}
	if data == nil {
		return nil, nil
	}
	// Bypasses ConvertToWithJSONSchema for all string inputs since ConvertToWithJSONSchema
	// expects the value to marshal into map[string]any for validation, which fails for string types.
	if text, ok := data.(string); ok {
		if !schemaIsString(schema) {
			// Step 1: try JSON parse
			var parsed any
			if err := json.Unmarshal([]byte(text), &parsed); err == nil {
				if err := schema.Validate(parsed); err == nil {
					return parsed, nil
				}
			}
			// Step 2: raw string may match an enum/literal schema
			if err := schema.Validate(text); err == nil {
				return text, nil
			}
			// Step 3: fall through to standard validation (which will return an error)
		} else {
			// If schema expects string, validate the raw string directly.
			if err := schema.Validate(text); err != nil {
				return nil, err
			}
			return text, nil
		}
	}
	return typeutil.ConvertToWithJSONSchema[any, any](data, schema)
}

// schemaIsString reports whether the resolved schema expects a JSON
// string at the top level.
func schemaIsString(s *jsonschema.Resolved) bool {
	if s == nil {
		return false
	}
	schema := s.Schema()
	if schema == nil {
		return false
	}
	if schema.Type == "string" {
		return true
	}
	for _, t := range schema.Types {
		if t == "string" {
			return true
		}
	}
	return false
}

// validateStateSchemaConsistency checks that all nodes in the graph that reference state fields
// have those fields declared in the workflow's state schema.
func validateStateSchemaConsistency(g *graph, schema *jsonschema.Resolved) error {
	if schema == nil {
		return nil
	}
	schemaFields := extractFieldNames(schema)

	for _, n := range g.allNodes() {
		spa, ok := n.(StateParamsAware)
		if !ok {
			continue
		}
		for _, fieldName := range spa.StateFieldNames() {
			if strings.HasPrefix(fieldName, session.KeyPrefixApp) ||
				strings.HasPrefix(fieldName, session.KeyPrefixUser) ||
				strings.HasPrefix(fieldName, session.KeyPrefixTemp) {
				continue
			}
			if !slices.Contains(schemaFields, fieldName) {
				return fmt.Errorf("node %q references state field %q which is not declared in StateSchema (declared: %v)", n.Name(), fieldName, schemaFields)
			}
		}
	}
	return nil
}

func extractFieldNames(schema *jsonschema.Resolved) []string {
	var fields []string
	if schema != nil && schema.Schema() != nil && schema.Schema().Properties != nil {
		for k := range schema.Schema().Properties {
			fields = append(fields, k)
		}
	}
	slices.Sort(fields)
	return fields
}
