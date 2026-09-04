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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/typeutil"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
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

// ErrUnsupportedFanIn is returned when a non-JoinNode has two or more
// unconditional incoming edges. Such a fan-in target would be activated
// once per predecessor, which the scheduler does not yet serialize
// safely. Use a JoinNode to converge multiple branches.
var ErrUnsupportedFanIn = errors.New("non-JoinNode fan-in is not yet supported")

// validateNodes executes a set of edges validation checks. Every check
// runs and all violations are reported together, so a caller with
// several mistakes can fix them in one pass.
func validateNodes(edges []Edge) error {
	return joinViolations(
		validateUniqueNames(edges),
		validateStartNodePresent(edges),
		validateStartNodeNoIncoming(edges),
		validateNoTaskModeGraphNodes(edges),
		validateChatModeWiring(edges),
	)
}

// joinViolations reports errs as a single error, skipping nils and
// messages already reported (two edges violating one rule the same way
// read as one problem to the user). A lone violation is returned as-is,
// unjoined, so it keeps the exact identity and message it had before
// validation started aggregating.
func joinViolations(errs ...error) error {
	var kept []error
	seen := make(map[string]bool)
	for _, err := range errs {
		if err == nil || seen[err.Error()] {
			continue
		}
		seen[err.Error()] = true
		kept = append(kept, err)
	}
	if len(kept) == 1 {
		return kept[0]
	}
	return errors.Join(kept...)
}

// distinctNodes yields each node of edges once, in first-appearance
// order, so validation reports violations in the order the caller
// declared them.
func distinctNodes(edges []Edge) iter.Seq[Node] {
	return func(yield func(Node) bool) {
		seen := make(map[Node]bool)
		for _, edge := range edges {
			for _, node := range [2]Node{edge.From, edge.To} {
				if seen[node] {
					continue
				}
				seen[node] = true
				if !yield(node) {
					return
				}
			}
		}
	}
}

// validateSubWorkflowNames checks that no sub-workflow has the same name as the parent workflow.
func validateSubWorkflowNames(workflowName string, edges []Edge) error {
	var errs []error
	for node := range distinctNodes(edges) {
		if err := checkSubWorkflowName(node, workflowName); err != nil {
			errs = append(errs, err)
		}
	}
	return joinViolations(errs...)
}

// checkSubWorkflowName checks if the node is a WorkflowNode and if its sub-workflow has the same name as the parent workflow.
func checkSubWorkflowName(node Node, workflowName string) error {
	if wfNode, ok := node.(*WorkflowNode); ok {
		if wfNode.subWorkflow.Name() == workflowName {
			return fmt.Errorf("%w: %q (node %q)", ErrSubWorkflowNameCollision, workflowName, node.Name())
		}
	}
	return nil
}

// validateUniqueNames checks that all nodes in the edge set have unique names.
// It reports every duplicated name, once each. The equality between nodes is
// checked by comparing the nodes directly.
func validateUniqueNames(edges []Edge) error {
	names := make(map[string]Node)
	var errs []error
	for node := range distinctNodes(edges) {
		storedNode, ok := names[node.Name()]
		if !ok {
			names[node.Name()] = node
			continue
		}
		if storedNode != node {
			errs = append(errs, fmt.Errorf("%w: %s", ErrDuplicateNodeName, node.Name()))
		}
	}
	return joinViolations(errs...)
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
	var errs []error
	for _, edge := range edges {
		if edge.To == Start {
			errs = append(errs, fmt.Errorf("%w: %s", ErrNodePointsToStart, edge.From.Name()))
		}
	}
	return joinViolations(errs...)
}

// validateWorkflow executes a set of workflow validation checks. Every
// check runs and all violations are reported together.
func validateWorkflow(workflow *graph, schema *jsonschema.Resolved) error {
	return joinViolations(
		validateUniqueEdges(workflow),
		validateDefaultRoute(workflow),
		validateConnectivity(workflow),
		validateCycles(workflow),
		validateFanIn(workflow),
		validateStaticSchemas(workflow),
		validateStateSchemaConsistency(workflow, schema),
	)
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
	var errs []error
	for node := range distinctNodes(edges) {
		if mode, ok := agentNodeMode(node); ok && mode == llminternal.ModeTask {
			errs = append(errs, fmt.Errorf(
				"Agent %q has mode='task' and cannot be used as a workflow graph node. Use a chat coordinator with task sub-agents, or "+
					"dispatch dynamically via RunNode from a function node",
				node.Name(),
			))
		}
	}
	return joinViolations(errs...)
}

// validateChatModeWiring rejects a chat-mode LlmAgent that is reached from
// any node other than Start. A chat agent builds its prompt from the
// conversation history (session events) and ignores the node input handed
// down an edge, so feeding it from a predecessor silently drops that
// predecessor's output. Mirrors adk-python's _validate_chat_agent_wiring.
func validateChatModeWiring(edges []Edge) error {
	var errs []error
	for _, e := range edges {
		if e.From == Start {
			continue
		}
		if mode, ok := agentNodeMode(e.To); ok && mode == llminternal.ModeChat {
			errs = append(errs, fmt.Errorf(
				"Agent %q has mode='chat' and cannot follow node %q: chat agents rely on conversation history and cannot consume a predecessor's node input. Use mode='single_turn', or wire the agent directly from Start",
				e.To.Name(), e.From.Name(),
			))
		}
	}
	return joinViolations(errs...)
}

// agentNodeMode returns the LlmAgent mode of node, or ok=false when node
// is not an AgentNode wrapping an LlmAgent.
func agentNodeMode(node Node) (llminternal.Mode, bool) {
	agentNode, ok := node.(*AgentNode)
	if !ok {
		return llminternal.ModeUnset, false
	}
	llmA, ok := agentNode.agent.(llminternal.Agent)
	if !ok || llmA == nil {
		return llminternal.ModeUnset, false
	}
	return llminternal.Reveal(llmA).Mode, true
}

// validateUniqueEdges checks that there are no duplicate edges in the workflow.
// Two edges with the same (From, To) are rejected regardless of Route; use
// MultiRoute to express alternatives to the same target.
func validateUniqueEdges(workflow *graph) error {
	var errs []error
	for _, node := range workflow.sortedNodes() {
		seen := make(map[Node]bool)
		for _, edge := range workflow.successorsOf(node) {
			if seen[edge.To] {
				errs = append(errs, fmt.Errorf("%w: from %q to %q", ErrDuplicateEdge, node.Name(), edge.To.Name()))
				continue
			}
			seen[edge.To] = true
		}
	}
	return joinViolations(errs...)
}

// validateDefaultRoute checks that there are no multiple default routes for one node.
func validateDefaultRoute(workflow *graph) error {
	var errs []error
	for _, node := range workflow.sortedNodes() {
		defaults := 0
		for _, edge := range workflow.successorsOf(node) {
			if edge.Route == Default {
				defaults++
			}
		}
		if defaults > 1 {
			errs = append(errs, fmt.Errorf("%w: %q", ErrMultipleDefaultRoutes, node.Name()))
		}
	}
	return joinViolations(errs...)
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
	reported := make(map[Node]struct{})
	var errs []error

	var traverse func(n Node, inStack map[Node]struct{})
	traverse = func(n Node, inStack map[Node]struct{}) {
		if _, ok := inStack[n]; ok {
			// One error per node a cycle closes on: a dense graph has
			// O(edges) back-edges, and repeating them helps nobody.
			if _, done := reported[n]; !done {
				reported[n] = struct{}{}
				errs = append(errs, fmt.Errorf("%w: %q", ErrUnconditionalCycle, n.Name()))
			}
			return
		}

		if _, ok := visited[n]; ok {
			return
		}

		inStack[n] = struct{}{}
		visited[n] = struct{}{}

		for _, edge := range workflow.successorsOf(n) {
			if edge.Route == nil {
				traverse(edge.To, inStack)
			}
		}

		delete(inStack, n)
	}

	for _, node := range workflow.sortedNodes() {
		if _, ok := visited[node]; !ok {
			traverse(node, make(map[Node]struct{}))
		}
	}

	return joinViolations(errs...)
}

// validateFanIn rejects a non-JoinNode target that has two or more
// unconditional incoming edges. Such a node would be activated once per
// completed predecessor, colliding the scheduler's per-name bookkeeping
// (mixed outputs, lost cancel funcs). JoinNode handles fan-in via a
// barrier; everything else must converge through one. Only unconditional
// (Route == nil) edges are counted so conditional fan-in and loop-back
// back-edges — where the predecessors don't all fire together — are not
// rejected.
func validateFanIn(workflow *graph) error {
	var errs []error
	for _, node := range workflow.sortedNodes() {
		if _, isJoin := node.(*JoinNode); isJoin {
			continue
		}
		unconditional := 0
		for _, edge := range workflow.predecessorsOf(node) {
			if edge.Route == nil {
				unconditional++
			}
		}
		if unconditional > 1 {
			errs = append(errs, fmt.Errorf("%w: node %q has %d unconditional incoming edges; "+
				"use a JoinNode to converge branches", ErrUnsupportedFanIn, node.Name(), unconditional))
		}
	}
	return joinViolations(errs...)
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

	var errs []error
	for _, n := range g.sortedNodes() {
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
				errs = append(errs, fmt.Errorf("node %q references state field %q which is not declared in StateSchema (declared: %v)", n.Name(), fieldName, schemaFields))
			}
		}
	}
	return joinViolations(errs...)
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

func validateStaticSchemas(g *graph) error {
	var errs []error
	for _, edge := range g.allEdges() {
		outResolved := edge.From.OutputSchema()
		inResolved := edge.To.InputSchema()
		if outResolved == nil || inResolved == nil {
			continue // validate only when both ends declare schemas (Python parity)
		}
		eq, err := schemasEqualCanonical(outResolved.Schema(), inResolved.Schema())
		if err != nil {
			errs = append(errs, fmt.Errorf("comparing schemas on edge %s->%s: %w",
				edge.From.Name(), edge.To.Name(), err))
			continue
		}
		if !eq {
			errs = append(errs, fmt.Errorf("graph validation failed: schema mismatch on edge %s -> %s",
				edge.From.Name(), edge.To.Name()))
		}
	}
	return joinViolations(errs...)
}

func schemasEqualCanonical(a, b *jsonschema.Schema) (bool, error) {
	ac, err := utils.CanonicalSchemaJSON(a)
	if err != nil {
		return false, err
	}
	bc, err := utils.CanonicalSchemaJSON(b)
	if err != nil {
		return false, err
	}
	return bytes.Equal(ac, bc), nil
}
