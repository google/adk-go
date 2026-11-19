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
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/typeutil"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/internal/workflowroute"
)

// ErrGraphValidation is matched, with [errors.Is], by every error [New],
// [NewWorkflowNode] and [NewWorkflowNodeWithSchemas] return because the edges
// do not describe a valid graph. workflowagent.New returns the same errors.
// The message has the form "graph validation failed: <detail>", and where the
// failed rule has its own sentinel, such as [ErrDuplicateNodeName] or
// [ErrUnconditionalCycle], the error matches that sentinel as well.
//
// ErrGraphValidation matches when:
//
//   - an edge has a nil From or To node, including a nil pointer or func
//     held in a non-nil [Node];
//   - [Concat] was given an item that is neither an [Edge] nor a []Edge;
//   - two distinct nodes share a name;
//   - the edge list is not empty and no edge has [Start] at either end;
//   - an edge from [Start] has a route;
//   - a node cannot be reached from [Start];
//   - an edge points to [Start];
//   - two edges have the same From and To nodes;
//   - a [MultiRoute] contains [Default];
//   - the edges form a cycle in which no edge has a route;
//   - the output schema of an edge's From node does not match the input
//     schema of its To node, or the two schemas cannot be compared, in which
//     case the error also wraps the cause;
//   - a chat-mode agent node follows a node other than [Start];
//   - an agent node runs in task mode;
//   - a node has more than one [Default] edge;
//   - a node other than a [JoinNode] has two or more incoming edges without
//     a route;
//   - a [WorkflowNode] has the same name as the workflow that contains it.
//
// It does not match a mismatch with the state schema given by
// [WithStateSchema], an error returned by a node's own validator of its
// outgoing edges, the error for edges that do not suit [WithNodeRootWrapper],
// or an error that [EdgeBuilder.Err] reports.
//
// ErrGraphValidation is the counterpart of adk-python's GraphValidationError.
var ErrGraphValidation = errors.New("graph validation failed")

// ErrDuplicateNodeName is returned when an edge set contains two
// distinct Node instances that share the same Name.
var ErrDuplicateNodeName = errors.New("duplicate node name")

// ErrNoStartNode is returned when the edge list is not empty and no edge
// has [Start] at either end. An empty edge list is the empty workflow and
// does not return it.
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
// unconditional incoming edges. New refuses such a graph even though the
// scheduler would run the target once per predecessor completion, one run at
// a time, each with that predecessor's output. Use a JoinNode to converge
// multiple branches into a single run.
var ErrUnsupportedFanIn = errors.New("non-JoinNode fan-in is not yet supported")

// validateNodes executes a set of edges validation checks.
//
// They are the rules that read only the edge list, run before any graph is
// built from it. It returns the first failure as a graph validation error.
// The order of the table is the rule order; it continues in
// validateWorkflow.
func validateNodes(edges []Edge) error {
	rules := []func([]Edge) error{
		validateEdgeShapes,       // first, so no later rule meets a nil node
		validateUniqueNames,      // adk-python's _validate_duplicate_node_names
		validateStartNodePresent, // _validate_start_node
		validateStartEdges,       // _validate_start_edges
	}
	for _, rule := range rules {
		if err := rule(edges); err != nil {
			return invalidGraph(err)
		}
	}
	return nil
}

// invalidGraph marks err as a graph validation error: the result matches
// ErrGraphValidation and everything err matches.
func invalidGraph(err error) error {
	return fmt.Errorf("%w: %w", ErrGraphValidation, err)
}

// validateEdgeShapes checks that no edge is a placeholder that Concat or
// EdgeBuilder.AddRoutes put in the list (see refusedEdge), and that every
// edge has a From and a To node. It reports the first defect by edge index,
// and runs before any rule that calls a node's methods.
func validateEdgeShapes(edges []Edge) error {
	for i, e := range edges {
		if r, ok := e.Route.(refusedEdge); ok {
			return fmt.Errorf("edges[%d]: %w", i, r.err)
		}
		if isNilNode(e.From) {
			return fmt.Errorf("edges[%d] has a nil From node", i)
		}
		if isNilNode(e.To) {
			return fmt.Errorf("edges[%d] has a nil To node", i)
		}
	}
	return nil
}

// isNilNode reports whether n is nil or holds a nil pointer or a nil func.
// Only those two kinds are checked: a value of another nil-able kind (map,
// slice, chan) can have methods that work when it is nil, and nothing about
// the kind tells such a node apart from a broken one.
func isNilNode(n Node) bool {
	if n == nil {
		return true
	}
	v := reflect.ValueOf(n)
	switch v.Kind() {
	case reflect.Pointer, reflect.Func:
		return v.IsNil()
	default:
		return false
	}
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
// If duplicate node names are found, it returns an error that lists each
// such name once, sorted, as adk-python does. The equality between nodes is
// checked by comparing the nodes directly.
func validateUniqueNames(edges []Edge) error {
	names := make(map[string]Node)
	var duplicates []string
	for _, edge := range edges {
		for _, node := range [...]Node{edge.From, edge.To} {
			storedNode, ok := names[node.Name()]
			if !ok {
				names[node.Name()] = node
				continue
			}
			if storedNode != node && !slices.Contains(duplicates, node.Name()) {
				duplicates = append(duplicates, node.Name())
			}
		}
	}
	if len(duplicates) == 0 {
		return nil
	}
	slices.Sort(duplicates)
	return fmt.Errorf("%w: %s", ErrDuplicateNodeName, strings.Join(duplicates, ", "))
}

// validateStartNodePresent checks that some edge has the start node at
// either end, as adk-python counts Start among the graph's nodes when any
// edge names it. An empty edge list passes: it is the empty workflow.
func validateStartNodePresent(edges []Edge) error {
	if len(edges) == 0 {
		return nil
	}
	for _, edge := range edges {
		if edge.From == Start || edge.To == Start {
			return nil
		}
	}
	return ErrNoStartNode
}

// validateStartEdges checks that no edge from Start has a route, Default
// included, as adk-python does. It reports the first such edge.
func validateStartEdges(edges []Edge) error {
	for _, e := range edges {
		if e.From == Start && e.Route != nil {
			return fmt.Errorf("edge from %q to %q has a route; edges from START take none", e.From.Name(), e.To.Name())
		}
	}
	return nil
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

// validateWorkflow runs the rules that read the graph and returns the first
// failure. The order of the table continues validateNodes: adk-python's
// rules in the order of its validate_graph, then its state-schema check,
// then the rules only adk-go has, so a graph with several defects reports
// the defect adk-python reports. A graph rule's failure is a graph
// validation error; the state-schema error and a node's own outgoing-edge
// validator error are returned as they are.
func validateWorkflow(workflow *graph, schema *jsonschema.Resolved) error {
	rules := []func(*graph) error{
		graphRule(validateConnectivity),                 // _validate_connectivity
		graphRule(onEdges(validateStartNodeNoIncoming)), // _validate_connectivity, edge into START
		graphRule(validateUniqueEdges),                  // _validate_duplicate_edges
		graphRule(validateDefaultInList),                // _validate_default_routes
		graphRule(validateCycles),                       // _detect_unconditional_cycles
		graphRule(validateStaticSchemas),                // _validate_static_schemas
		graphRule(onEdges(validateChatModeWiring)),      // _validate_chat_agent_wiring
		func(g *graph) error { return validateStateSchemaConsistency(g, schema) },
		graphRule(onEdges(validateNoTaskModeGraphNodes)), // adk-go only, from here on
		graphRule(validateDefaultRoute),
		graphRule(validateFanIn),
		validateNodeOutgoingEdges,
	}
	for _, rule := range rules {
		if err := rule(workflow); err != nil {
			return err
		}
	}
	return nil
}

// graphRule returns rule with its failure marked as a graph validation
// error.
func graphRule(rule func(*graph) error) func(*graph) error {
	return func(g *graph) error {
		if err := rule(g); err != nil {
			return invalidGraph(err)
		}
		return nil
	}
}

// onEdges adapts a rule over the edge list to the graph's edges in
// declaration order.
func onEdges(rule func([]Edge) error) func(*graph) error {
	return func(g *graph) error {
		return rule(g.edges)
	}
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
//
// It reports the first such node in first-mention order.
func validateNoTaskModeGraphNodes(edges []Edge) error {
	for _, e := range edges {
		for _, node := range [...]Node{e.From, e.To} {
			if mode, ok := agentNodeMode(node); ok && mode == llminternal.ModeTask {
				return fmt.Errorf(
					"Agent %q has mode='task' and cannot be used as a workflow graph node. Use a chat coordinator with task sub-agents, or "+
						"dispatch dynamically via RunNode from a function node",
					node.Name(),
				)
			}
		}
	}
	return nil
}

// validateChatModeWiring rejects a chat-mode LlmAgent that is reached from
// any node other than Start. A chat agent builds its prompt from the
// conversation history (session events) and ignores the node input handed
// down an edge, so feeding it from a predecessor silently drops that
// predecessor's output. Mirrors adk-python's _validate_chat_agent_wiring.
func validateChatModeWiring(edges []Edge) error {
	for _, e := range edges {
		if e.From == Start {
			continue
		}
		if mode, ok := agentNodeMode(e.To); ok && mode == llminternal.ModeChat {
			return fmt.Errorf(
				"Agent %q has mode='chat' and cannot follow node %q: chat agents rely on conversation history and cannot consume a predecessor's node input. Use mode='single_turn', or wire the agent directly from Start",
				e.To.Name(), e.From.Name(),
			)
		}
	}
	return nil
}

// agentNodeMode returns the mode the LlmAgent wrapped by node runs under
// as a graph node, or ok=false when node is not an AgentNode wrapping an
// LlmAgent.
//
// Resolving rather than reading the declaration makes no difference to either
// caller today: both compare against task and chat, and resolving only ever
// turns unset into single_turn. It is written this way because the answer this
// returns is what the agent runs under at that placement, which is the whole
// point of the change around it, and because a caller added later that compares
// against single_turn would otherwise be silently wrong for an undeclared agent.
func agentNodeMode(node Node) (llminternal.Mode, bool) {
	agentNode, ok := node.(*AgentNode)
	if !ok {
		return llminternal.ModeUnset, false
	}
	llmA, ok := agentNode.agent.(llminternal.Agent)
	if !ok || llmA == nil {
		return llminternal.ModeUnset, false
	}
	return llminternal.ResolveMode(llminternal.Reveal(llmA).Mode, llminternal.ModeSingleTurn), true
}

// validateUniqueEdges checks that there are no duplicate edges in the workflow.
// Two edges with the same (From, To) are rejected regardless of Route; use
// MultiRoute to express alternatives to the same target. It reports the
// first edge, in declaration order, whose (From, To) an earlier edge already
// has, as adk-python does.
func validateUniqueEdges(workflow *graph) error {
	seen := make(map[[2]Node]struct{}, len(workflow.edges))
	for _, edge := range workflow.edges {
		key := [2]Node{edge.From, edge.To}
		if _, ok := seen[key]; ok {
			return fmt.Errorf("%w: from %q to %q", ErrDuplicateEdge, edge.From.Name(), edge.To.Name())
		}
		seen[key] = struct{}{}
	}
	return nil
}

// defaultHolder is implemented by a route that can contain Default.
type defaultHolder interface {
	hasDefault() bool
}

// validateDefaultInList checks that no edge's MultiRoute contains Default,
// as adk-python does; a Default edge is written as an edge of its own. It
// reports the first such edge in declaration order.
func validateDefaultInList(workflow *graph) error {
	for _, e := range workflow.edges {
		if r, ok := e.Route.(defaultHolder); ok && r.hasDefault() {
			return fmt.Errorf("edge from %q to %q: a MultiRoute must not contain Default", e.From.Name(), e.To.Name())
		}
	}
	return nil
}

// validateDefaultRoute checks that there are no multiple default routes for one node.
// It reports the first such node in first-mention order.
func validateDefaultRoute(workflow *graph) error {
	for _, node := range workflow.allNodes() {
		edges := workflow.successorsOf(node)
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
//
// The search starts from the nodes in first-mention order and follows edges
// in declaration order, so the same edges always report the same cycle. The
// error names the cycle's nodes from the first one the search reached back
// to, for example "A" -> "B" -> "A".
func validateCycles(workflow *graph) error {
	visited := make(map[Node]struct{})
	var path []Node
	onPath := make(map[Node]int) // index of a node in path

	var traverse func(n Node) error
	traverse = func(n Node) error {
		if i, ok := onPath[n]; ok {
			return fmt.Errorf("%w: %s", ErrUnconditionalCycle, formatCycle(path[i:], n))
		}

		if _, ok := visited[n]; ok {
			return nil
		}

		onPath[n] = len(path)
		path = append(path, n)
		visited[n] = struct{}{}

		for _, edge := range workflow.successorsOf(n) {
			if edge.Route == nil {
				if err := traverse(edge.To); err != nil {
					return err
				}
			}
		}

		path = path[:len(path)-1]
		delete(onPath, n)
		return nil
	}

	for _, node := range workflow.allNodes() {
		if err := traverse(node); err != nil {
			return err
		}
	}

	return nil
}

// formatCycle returns the quoted names of the cycle's nodes followed by
// the name of closing, the node the cycle returns to, joined by " -> ".
func formatCycle(cycle []Node, closing Node) string {
	names := make([]string, 0, len(cycle)+1)
	for _, n := range cycle {
		names = append(names, strconv.Quote(n.Name()))
	}
	names = append(names, strconv.Quote(closing.Name()))
	return strings.Join(names, " -> ")
}

// validateFanIn rejects a non-JoinNode target that has two or more
// unconditional incoming edges. Such a node would be activated once per
// completed predecessor, and is refused until fan-in to a plain node is
// supported (see ErrUnsupportedFanIn). JoinNode handles fan-in via a
// barrier; everything else must converge through one. Only unconditional
// (Route == nil) edges are counted so conditional fan-in and loop-back
// back-edges — where the predecessors don't all fire together — are not
// rejected. It reports the first such node in first-mention order.
func validateFanIn(workflow *graph) error {
	for _, node := range workflow.allNodes() {
		edges := workflow.predecessorsOf(node)
		if _, isJoin := node.(*JoinNode); isJoin {
			continue
		}
		unconditional := 0
		for _, edge := range edges {
			if edge.Route == nil {
				unconditional++
			}
		}
		if unconditional > 1 {
			return fmt.Errorf("%w: node %q has %d unconditional incoming edges; "+
				"use a JoinNode to converge branches", ErrUnsupportedFanIn, node.Name(), unconditional)
		}
	}
	return nil
}

// defaultValidateInput validates data against schema and returns the
// value the node receives. A nil schema returns data unchanged. Otherwise
// the user message of a startSeed, a *genai.Content and a genai.Content
// are validated as their payload, by the rules of validateContentOutput;
// when those rules keep the user message, the startSeed is returned, so
// each successor of Start still receives the form it takes (see
// graph.startSeedFor). A nil data returns nil. A string against a
// string schema is validated as it is. A string against any other schema
// is parsed as JSON and conformed; when parsing or validation fails it is
// tried as the raw string (for an enum or literal schema). Any other value
// is converted to its JSON form and conformed (see conform), so defaults
// are filled and optional nulls dropped; an absent value (JSON null)
// satisfies an object schema as {}. The result is never data itself
// unless data is a string. On failure it returns nil and the schema
// error.
func defaultValidateInput(data any, schema *jsonschema.Resolved) (any, error) {
	if schema == nil {
		return data, nil
	}
	if seed, ok := data.(startSeed); ok {
		validated, err := validateContentOutput(seed.content, schema)
		if err != nil || validated != any(seed.content) {
			return validated, err
		}
		return seed, nil
	}
	if c, ok := asContent(data); ok {
		return validateContentOutput(c, schema)
	}
	if data == nil {
		return nil, nil
	}
	if text, ok := data.(string); ok {
		if schemaIsString(schema) {
			if err := schema.Validate(text); err != nil {
				return nil, err
			}
			return text, nil
		}
		var parsed any
		if err := json.Unmarshal([]byte(text), &parsed); err == nil {
			if form, err := conformed(parsed, schema); err == nil {
				return form, nil
			}
		}
		if err := schema.Validate(text); err == nil {
			return text, nil
		}
		// Fall through to standard validation, which returns the error.
	}
	form, err := typeutil.ConvertToWithJSONSchema[any, any](data, nil)
	if err != nil {
		return nil, err
	}
	if form == nil && schemaExpectsObject(schema) {
		form = map[string]any{}
	}
	return conformed(form, schema)
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
	return slices.Contains(schema.Types, "string")
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
			if isOwnerPrefixed(fieldName) {
				continue
			}
			if !slices.Contains(schemaFields, fieldName) {
				return &StateSchemaError{
					Node:     n.Name(),
					Key:      fieldName,
					Schema:   schemaTitle(schema),
					Reason:   StateBindingUndeclared,
					Declared: slices.Clone(schemaFields),
				}
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

// validateStaticSchemas checks, for each edge in declaration order whose
// From node declares an output schema and whose To node declares an input
// schema, that the two schemas are equal. It reports the first edge that
// fails.
func validateStaticSchemas(g *graph) error {
	for _, edge := range g.edges {
		outResolved := edge.From.OutputSchema()
		inResolved := edge.To.InputSchema()
		if outResolved == nil || inResolved == nil {
			continue // validate only when both ends declare schemas (Python parity)
		}
		eq, err := schemasEqualCanonical(outResolved.Schema(), inResolved.Schema())
		if err != nil {
			return fmt.Errorf("comparing schemas on edge %s->%s: %w",
				edge.From.Name(), edge.To.Name(), err)
		}
		if !eq {
			return fmt.Errorf("schema mismatch on edge %s -> %s",
				edge.From.Name(), edge.To.Name())
		}
	}
	return nil
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

// validateNodeOutgoingEdges invokes registered node-specific outgoing-edge
// validators (such as exp/classifier.Node), mirroring adk-python's
// _validate_node_outgoing_edges.
func validateNodeOutgoingEdges(g *graph) error {
	nodes := g.allNodes()
	slices.SortFunc(nodes, func(a, b Node) int {
		return strings.Compare(a.Name(), b.Name())
	})
	for _, n := range nodes {
		summary := summarizeOutgoingEdges(g.successorsOf(n))
		if err := workflowroute.Validate(n, summary); err != nil {
			return err
		}
	}
	return nil
}

type multiRouteSummarizer interface {
	summarizeMultiRoute() []workflowroute.Value
}

// hasDefault reports whether r contains Default. An element of another
// dynamic type compares unequal to Default without being compared, so an
// uncomparable element cannot panic here.
func (r MultiRoute[T]) hasDefault() bool {
	for _, elem := range r {
		if any(elem) == Default {
			return true
		}
	}
	return false
}

func (r MultiRoute[T]) summarizeMultiRoute() []workflowroute.Value {
	if len(r) == 0 {
		return []workflowroute.Value{{Kind: workflowroute.KindOther, Label: "[]"}}
	}
	vals := make([]workflowroute.Value, 0, len(r))
	for _, elem := range r {
		a := any(elem)
		if a == nil {
			vals = append(vals, workflowroute.Value{Kind: workflowroute.KindOther, Label: "<nil>"})
			continue
		}
		if a == Default {
			vals = append(vals, workflowroute.Value{Kind: workflowroute.KindOther, Label: "Default"})
			continue
		}
		switch v := a.(type) {
		case string:
			vals = append(vals, workflowroute.Value{Kind: workflowroute.KindString, Label: strconv.Quote(v)})
		case StringRoute:
			vals = append(vals, workflowroute.Value{Kind: workflowroute.KindString, Label: strconv.Quote(string(v))})
		case bool:
			vals = append(vals, workflowroute.Value{Kind: workflowroute.KindBool, Label: strconv.FormatBool(v)})
		case BoolRoute:
			vals = append(vals, workflowroute.Value{Kind: workflowroute.KindBool, Label: strconv.FormatBool(bool(v))})
		case int:
			vals = append(vals, workflowroute.Value{Kind: workflowroute.KindOther, Label: strconv.Itoa(v)})
		case IntRoute:
			vals = append(vals, workflowroute.Value{Kind: workflowroute.KindOther, Label: strconv.Itoa(int(v))})
		default:
			vals = append(vals, workflowroute.Value{Kind: workflowroute.KindOther, Label: fmt.Sprintf("%T", a)})
		}
	}
	return vals
}

func summarizeOutgoingEdges(edges []Edge) workflowroute.OutgoingSummary {
	var out workflowroute.OutgoingSummary
	for _, e := range edges {
		switch r := e.Route.(type) {
		case nil:
			out.HasUnrouted = true
		case StringRoute:
			out.Routes = append(out.Routes, workflowroute.Value{Kind: workflowroute.KindString, Label: strconv.Quote(string(r))})
		case BoolRoute:
			out.Routes = append(out.Routes, workflowroute.Value{Kind: workflowroute.KindBool, Label: strconv.FormatBool(bool(r))})
		case IntRoute:
			out.Routes = append(out.Routes, workflowroute.Value{Kind: workflowroute.KindOther, Label: strconv.Itoa(int(r))})
		case multiRouteSummarizer:
			out.Routes = append(out.Routes, r.summarizeMultiRoute()...)
		default:
			if e.Route == Default {
				out.HasDefault = true
				continue
			}
			out.Routes = append(out.Routes, workflowroute.Value{Kind: workflowroute.KindOther, Label: fmt.Sprintf("%T", e.Route)})
		}
	}
	return out
}
