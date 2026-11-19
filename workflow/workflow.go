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

// Package workflow hosts workflow definitions and primitives for building
// multi-agent applications.
//
// # Input and output schemas
//
// A node validates what it receives and what it emits against the schemas
// its InputSchema and OutputSchema return. A workflow run as the root takes
// its own schemas through [WithInputSchema] and [WithOutputSchema]: the input
// schema validates the user message before any node runs, and the output
// schema validates the single terminal output when the run finishes. A
// workflow nested as a node carries its schemas on the node, see
// [NewWorkflowNodeWithSchemas]. Validation fills schema defaults at any depth
// reached through properties, patternProperties, additionalProperties,
// prefixItems, items, allOf and local $ref (not through anyOf or oneOf), and
// drops optional properties whose value is null. Keys a schema does not
// declare are kept when the schema leaves additionalProperties open and
// rejected under additionalProperties: false. A Go struct validated against a
// schema arrives downstream as its JSON form, a map[string]any.
package workflow

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// Node is the interface for all nodes in a workflow.
//
// Custom nodes typically embed BaseNode (constructed via NewBaseNode)
// to inherit Name, Description, and Config implementations, and
// supply only Run.
//
// A custom node that calls another node's Run directly does not forward
// its private recovery inventory. On failure, retries may conservatively
// invalidate the custom node's subtree and repeat completed descendants.
// Use RunNode within a DynamicNode to track child activations explicitly.
type Node interface {
	// Name identifies the node in its workflow: two distinct nodes of one
	// workflow may not share it (ErrDuplicateNodeName). It is the node's
	// segment of a node path, "<name>@<run id>", and the key of the node's
	// output in a JoinNode's map.
	Name() string
	// Description is a human-readable summary of the node. The scheduler
	// does not read it. An AgentNode or ToolNode takes it from its agent or
	// tool, and a node that is runner.Config.Node gives it to the runner's
	// root agent.
	Description() string
	// Config returns the node's settings, such as RetryConfig, Timeout and
	// StateSchema. The workflow calls it at several points of a run, so it
	// should return the same value on every call.
	Config() NodeConfig
	// InputSchema returns the JSON schema for the node's input.
	// If it returns nil, it indicates there is no schema and no input validation is performed.
	InputSchema() *jsonschema.Resolved
	// OutputSchema returns the JSON schema for the node's output.
	// If it returns nil, it indicates there is no schema and no output validation is performed.
	OutputSchema() *jsonschema.Resolved
	// ValidateInput validates and optionally coerces/transforms the input before the node runs.
	// It returns the validated input (which might be coerced/parsed/transformed) or an error.
	// It will be called by the scheduler before Run on every activation.
	ValidateInput(input any) (any, error)
	// ValidateOutput validates and optionally coerces/transforms an output emitted by the node.
	// It returns the validated output (which might be coerced/parsed/transformed) or an error.
	// The scheduler invokes it on every yielded event with a non-nil output, before forwarding it to the consumer.
	ValidateOutput(output any) (any, error)
	// Run runs one attempt of the node on its validated input. The node
	// yields events: Event.Output carries its output, and Event.Routes
	// selects its routed edges (at most one event of an activation may set
	// Routes; a second fails the node with ErrMultipleRoutingEvents). A
	// non-nil error ends the attempt as a failure, which RetryConfig may
	// retry. When yield returns false the node must return. An event the
	// node yields once ctx is done is dropped, and its yield returns false.
	Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error]
}

// StateParamsAware is implemented by nodes that bind their parameters
// from ctx.state. Used by Workflow's static state-schema validation
// to detect mismatches between declared state fields and node consumers.
type StateParamsAware interface {
	// StateFieldNames returns the state keys the node reads its parameters
	// from. New refuses a workflow with a state schema that does not declare
	// one of them, other than a key with a colon, with a *StateSchemaError.
	StateFieldNames() []string
}

// Route defines the interface for matching execution results to edges.
type Route interface {
	// Matches reports whether the edge fires for event, the one event of
	// the source node's activation whose Routes is non-nil. It is not
	// called when no event of the activation set Routes, and never for the
	// Default route, which fires when no other routed edge of the node
	// matched.
	Matches(event *session.Event) bool
}

func matchRoute(routeValue string, event *session.Event) bool {
	return slices.Contains(event.Routes, routeValue)
}

// StringRoute is a route defined by a string value.
type StringRoute string

// Matches reports whether event was routed to this string value.
func (r StringRoute) Matches(event *session.Event) bool {
	return matchRoute(string(r), event)
}

// IntRoute is a route defined by an integer value.
type IntRoute int

// Matches reports whether event was routed to this integer value, compared by
// its decimal string form.
func (r IntRoute) Matches(event *session.Event) bool {
	return matchRoute(fmt.Sprint(r), event)
}

// BoolRoute is a route defined by a boolean value.
type BoolRoute bool

// Matches reports whether event was routed to this boolean value, compared by
// its "true"/"false" string form.
func (r BoolRoute) Matches(event *session.Event) bool {
	return matchRoute(fmt.Sprint(r), event)
}

// MultiRoute matches any value within a specified list of allowed routes.
//
// A MultiRoute must not contain [Default]: [New] refuses an edge whose
// MultiRoute does. Write a separate edge with Route [Default] instead.
type MultiRoute[T comparable] []T

// Matches reports whether event was routed to any of the values in this list.
func (r MultiRoute[T]) Matches(event *session.Event) bool {
	for _, route := range r {
		if matchRoute(fmt.Sprint(route), event) {
			return true
		}
	}
	return false
}

// Default is a special route that matches when no other concrete routes match.
//
// It is the Route of an edge of its own: [New] refuses it inside a
// [MultiRoute] and on an edge from [Start].
var Default Route = &defaultRoute{}

type defaultRoute struct{}

func (r *defaultRoute) Matches(event *session.Event) bool {
	return false
}

// Edge defines a directed connection between nodes in the workflow graph.
type Edge struct {
	From  Node  // The source node
	To    Node  // The target node
	Route Route // Routing condition
}

// Start is a sentinel node used to indicate the entry point of the workflow.
//
// An edge from Start takes no route: [New] refuses one whose Route is not
// nil, [Default] included. An edge into Start is refused too.
var Start Node = &startNode{
	BaseNode: NewBaseNode("START", "Start node", NodeConfig{}),
}

type startNode struct {
	BaseNode
}

func (s *startNode) ValidateOutput(output any) (any, error) {
	return output, nil
}

func (s *startNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {}
}

// Workflow manages the workflow graph execution.
type Workflow struct {
	graph *graph

	// name labels this workflow's nodes on the paths Run records: Run
	// prefixes the node path with it, except for a root wrapper.
	// RunNode and Resume add no prefix. The prefix is a label, not a
	// namespace - history is attributed to a run by invocation ID, not
	// by this name, so the only thing keeping two workflows that
	// share a node name apart is that their runs carry different
	// invocation IDs.
	// Empty leaves the workflow unnamed. Set at construction by New.
	name string

	// maxConcurrency caps the number of graph-scheduled nodes that
	// may run concurrently within a single Run invocation. 0
	// (the default) means unlimited. Set via WithMaxConcurrency.
	maxConcurrency int

	// stateSchema holds the state schema for the workflow.
	stateSchema *jsonschema.Resolved
	// stateValidator checks node-context state writes and node event state
	// deltas against stateSchema. Nil when the workflow has no state schema,
	// so a nested workflow without one keeps its parent's validator.
	stateValidator agent.StateValidator

	// inputSchema validates the input of every RunNode call and the seed
	// ReconstructRunState rebuilds. Nil means no validation. Set via
	// WithInputSchema. The output schema lives on graph, which the
	// scheduler's finalize reads.
	inputSchema *jsonschema.Resolved

	// isRootWrapper marks this workflow as a synthetic single-node wrapper
	// created by Runner.runNode to drive a standalone agent.
	isRootWrapper bool
}

// Option configures a Workflow at construction time. Pass options
// as trailing variadic arguments to New.
type Option func(*workflowOptions)

// workflowOptions holds the resolved settings derived from the
// caller's Option list. Zero values mean "no override / use
// engine defaults".
type workflowOptions struct {
	maxConcurrency    int
	stateSchema       *jsonschema.Resolved
	inputSchema       *jsonschema.Resolved
	outputSchema      *jsonschema.Resolved
	isRootWrapper     bool
	isNodeRootWrapper bool
}

// WithRootWrapper marks this workflow as a synthetic single-node wrapper
// created by Runner.runNode to drive a standalone agent.
//
// Architectural note: Unlike Go, Python ADK does not wrap standalone agents
// in a synthetic workflow (Python's Runner.run drives agent.run_async directly).
// Go introduced this wrapper so all agent execution rides through a unified
// graph engine. When marked as a root wrapper, Workflow.Run does not stamp
// an extraneous parent namespace prefix ("app/agent@1") onto root contexts,
// maintaining strict path parity with Python reference recordings.
func WithRootWrapper() Option {
	return func(o *workflowOptions) {
		o.isRootWrapper = true
	}
}

// WithNodeRootWrapper marks this anonymous workflow as the wrapper the runner
// builds around a runner.Config.Node root. It is meant for the runner; other
// callers have no reason to set it.
//
// [New] returns an error unless the workflow holds exactly one edge, from
// [Start] to the node root. The node root is treated as follows:
//
//   - A node root that is a [*WorkflowNode] receives the START seed
//     unchanged, so its inner [Start] gives each of its own successors the
//     form that successor declares (see [NodeConfig.ContentInput]). Any
//     other node root receives the user message as a *genai.Content
//     whatever its ContentInput, or nil when there is no user content.
//   - [Workflow.Resume] re-runs an interrupted node root, as if its
//     [NodeConfig.RerunOnResume] were &true, and passes it the
//     invocation's first user message again.
//
// The wrapper is invisible in the events: Workflow.Run adds no path segment
// of its own to the node root's contexts, adds no adk.workflow.base_path key
// to events, and yields no completion or outcome event, so the events look
// as if the node root ran alone.
func WithNodeRootWrapper() Option {
	return func(o *workflowOptions) {
		o.isNodeRootWrapper = true
	}
}

// WithMaxConcurrency caps how many graph-scheduled nodes may run at once in a
// single Workflow invocation. A node waiting for a retry backoff keeps its
// slot. A trigger that finds the cap reached, or its node busy, waits in
// arrival order and starts when a slot and the node are free; it takes no
// slot while it waits. n <= 0 disables the cap (unlimited).
//
// Does NOT apply to dynamic sub-nodes invoked via workflow.RunNode
// from inside a DynamicNode body — they are awaited inline by the
// parent and gating them would deadlock.
//
// The cap counts the graph-scheduled nodes of this workflow in one run,
// wherever the workflow runs: as a root built with agent.New over [Workflow.Run],
// with workflowagent.Config.MaxConcurrency, as a [NewWorkflowNodeWithOptions]
// node that is runner.Config.Node, nested in another workflow, or run by
// [RunNode]. A nested workflow counts as one node of its parent, its own cap
// applies inside it, and the parent's cap does not reach its nodes. The cap
// holds after a resume.
func WithMaxConcurrency(n int) Option {
	return func(o *workflowOptions) {
		if n < 0 {
			n = 0
		}
		o.maxConcurrency = n
	}
}

// WithStateSchema sets the state schema for the workflow.
//
// s must already be resolved; nil, the default, means no state is checked.
// The properties of s's root name the state keys the workflow's nodes may
// write, and each property constrains its key's value. Every write is checked
// on its own against its key's property, so a write that leaves a required key
// unset is accepted, and no other root keyword takes part. A key that contains
// a colon (app:, user:, temp:, or a component's own "<owner>:<key>") is never
// checked. References in s must stay inside the schema document: [New] returns
// an error for a property that reaches a document s's own loader supplied.
//
// Two writes are checked: a call of Set on a node context's State, which
// returns the error and writes nothing, and the state delta of every event a
// node yields, which is checked before the event leaves the node; a refused
// event is not emitted and fails the node. A refused write is reported as a
// [*StateSchemaError]. Writes a node made before it failed are still stored,
// unless it was canceled or failed because a dynamic child failed. A
// nested workflow without a state schema checks its nodes against the
// enclosing workflow's. A node's own [NodeConfig.StateSchema] replaces this
// schema for that node.
//
// The schema applies to this workflow's nodes wherever the workflow runs and
// on every turn of a resumed run. A nested workflow's own schema, given to
// [NewWorkflowNodeWithOptions], replaces its parent's for its nodes; a
// workflowagent run as an [AgentNode] with workflowagent.Config.StateSchema
// behaves the same. A write that a resumed turn makes is refused in the
// placements that continue a paused run: workflowagent.New, and a
// [NewWorkflowNodeWithOptions] node that is runner.Config.Node. An agent built
// with agent.New over [Workflow.Run] does not continue a paused run: on a later
// turn the answer is not delivered and the node asks again.
//
// s is shared by every run of the workflow and is never modified.
func WithStateSchema(s *jsonschema.Resolved) Option {
	return func(o *workflowOptions) {
		o.stateSchema = s
	}
}

// WithInputSchema sets the schema the workflow's own input must satisfy.
//
// s must already be resolved; nil, the default, means no input
// validation. Every [Workflow.RunNode] call, and so every [Workflow.Run],
// validates its input before any node is scheduled. The user message that
// Run passes is validated as [BaseNode.ValidateOutput] validates a
// *genai.Content: the empty schema keeps the message, and so does a schema
// generated from genai.Content when the message's text fails it and the
// message satisfies it; each successor of [Start] then receives the
// message as it would without s. Otherwise the message's text is parsed
// as JSON unless s is a string schema, defaults are filled, and optional
// properties that are null are dropped; the conformed value replaces the
// input, so the successors of [Start] receive it whatever their
// [NodeConfig.ContentInput] says. A nil input, and a run without user
// content, pass unchanged.
//
// When validation fails the run yields no event and runs no node; it
// yields one error that wraps [ErrInputValidation] and the schema error.
// [Workflow.ReconstructRunState] validates the input it rebuilds from
// history in the same way and returns the same error.
//
// Given to [NewWorkflowNodeWithOptions], s is the node's input schema, as
// [NewWorkflowNodeWithSchemas] sets it.
//
// s is shared by every run of the workflow and is never modified.
func WithInputSchema(s *jsonschema.Resolved) Option {
	return func(o *workflowOptions) {
		o.inputSchema = s
	}
}

// WithOutputSchema sets the schema the workflow's single terminal output must satisfy.
//
// s must already be resolved; nil, the default, means no output
// validation. When a run or a resume ends with no node waiting and
// exactly one terminal node produced output, that output is validated
// against s, with defaults filled first as for a node's output. On
// failure the run's last item is an error that wraps [ErrNodeFailed] and
// the schema error and names the workflow:
// `workflow: dynamic child failed: output validation failed for node "<name>": ...`.
// The terminal node's own events have already been yielded by then.
//
// Nothing is validated while a node waits for input, nor when no terminal
// node produced output. More than one terminal output fails with
// [ErrMultipleTerminalOutputs] before the schema is consulted. The value
// validation produces is not emitted; the terminal node's event keeps the
// output it yielded.
//
// Given to [NewWorkflowNodeWithOptions], s is the node's output schema, as
// [NewWorkflowNodeWithSchemas] sets it, and the next node receives the
// coerced value.
//
// s is shared by every run of the workflow and is never modified.
func WithOutputSchema(s *jsonschema.Resolved) Option {
	return func(o *workflowOptions) {
		o.outputSchema = s
	}
}

// New creates a new Workflow engine with the given name and edges.
//
// The name labels this workflow's nodes on the paths Run records: Run
// prefixes the node path it records with it, except for a root
// wrapper. RunNode and Resume add no prefix. The same workflow may
// safely share a name across different sessions.
//
// The prefix is a label, not a namespace. ReconstructRunState scopes
// history to a run by invocation ID; attributing an event to a node
// goes through eventNodeName, which matches the first path segment
// naming one of this workflow's nodes and ignores the workflow-name
// segment. A name therefore does not keep two workflows' histories
// apart, and two workflows that share a node name resolve to that
// name regardless of what they are called. With an empty invocation
// ID they collide whatever their names are.
//
// An empty name adds no prefix, so this workflow's nodes are recorded
// under the bare node path.
//
// edges may be nil or empty. The result is then the empty workflow: its
// Run yields no event and produces no output.
//
// New checks that edges describe a valid graph and returns the first
// defect it finds as an error that matches [ErrGraphValidation], whose doc
// lists the defects, and the defect's own sentinel where it has one, such
// as [ErrNodesNotReachable]. The checks run in adk-python's order, followed
// by the checks only adk-go makes, so a graph with several defects reports
// the one adk-python reports. Within one check the first edge or node in
// declaration order is reported, so the same edges always give the same
// error. A node that reads a state field the [WithStateSchema] schema does
// not declare, and a node's own validator of its outgoing edges, return
// errors that do not match ErrGraphValidation.
//
// Optional Option values configure engine behavior
// (concurrency cap, etc.); see WithMaxConcurrency.
func New(name string, edges []Edge, opts ...Option) (*Workflow, error) {
	if err := validateNodes(edges); err != nil {
		return nil, err
	}
	// TODO(wolo): sanity-check name (reject whitespace-only,
	// reject characters that break the event node-path shape).
	// TODO(wolo): record a graph fingerprint (e.g. sorted node
	// names hash) on the Workflow and verify it against the
	// RunState reconstructed in Resume; today a graph evolution
	// between deploys silently corrupts the resume path.
	var o workflowOptions
	for _, opt := range opts {
		opt(&o)
	}
	if o.isNodeRootWrapper && len(edges) != 1 {
		return nil, fmt.Errorf("workflow: WithNodeRootWrapper needs exactly one edge, from Start to the node root; got %d edges", len(edges))
	}
	graph := newGraph(edges)
	graph.isRootWrapper = o.isRootWrapper
	graph.isNodeRootWrapper = o.isNodeRootWrapper
	graph.outputSchema = o.outputSchema
	if err := validateWorkflow(graph, o.stateSchema); err != nil {
		return nil, err
	}
	if err := validateSubWorkflowNames(name, edges); err != nil {
		return nil, invalidGraph(err)
	}
	// A node's own schema that cannot be checked at run time is reported
	// here rather than when the node starts. The validators are discarded:
	// each run memoizes its own (withStateValidator).
	var nodeSchemaErr error
	walkNodes(edges, func(n Node) {
		s := n.Config().StateSchema
		if s == nil || nodeSchemaErr != nil {
			return
		}
		if _, err := newStateValidator(s); err != nil {
			nodeSchemaErr = fmt.Errorf("node %q: %w", n.Name(), err)
		}
	})
	if nodeSchemaErr != nil {
		return nil, nodeSchemaErr
	}
	w := &Workflow{
		graph:          graph,
		name:           name,
		maxConcurrency: o.maxConcurrency,
		stateSchema:    o.stateSchema,
		inputSchema:    o.inputSchema,
		isRootWrapper:  o.isRootWrapper,
	}
	if o.stateSchema != nil {
		v, err := newStateValidator(o.stateSchema)
		if err != nil {
			return nil, err
		}
		w.stateValidator = v
	}
	return w, nil
}

// withStateValidator returns ctx carrying the workflow's state validator and
// a memo of the validators derived from node schemas. The validator is set
// only when the workflow has a state schema, so a nested workflow without one
// keeps its parent's. The memo is a fresh one when ctx carries none, so each
// run has its own and a nested workflow shares the outermost run's. When
// neither applies, it returns ctx itself.
func (w *Workflow) withStateValidator(ctx agent.Context) agent.Context {
	delta := &agent.CommonContextDelta{}
	if stateValidatorsFrom(ctx) == nil {
		base := context.WithValue(ctx, stateValidatorsKey{}, &stateValidators{})
		delta.InvocationContextDelta = &agent.InvocationContextDelta{Context: &base}
	}
	if w.stateValidator != nil {
		v := w.stateValidator
		delta.StateValidator = &v
	}
	if delta.InvocationContextDelta == nil && delta.StateValidator == nil {
		return ctx
	}
	return ctx.WithDelta(delta)
}

// Name returns the workflow's node-path namespacing name as set by
// New. Empty when the workflow is anonymous and adds no path
// prefix.
func (w *Workflow) Name() string {
	return w.name
}

// InputSchema returns the schema set by WithInputSchema, or nil.
// The returned schema is shared and must not be modified.
func (w *Workflow) InputSchema() *jsonschema.Resolved {
	return w.inputSchema
}

// OutputSchema returns the schema set by WithOutputSchema, or nil.
// The returned schema is shared and must not be modified.
func (w *Workflow) OutputSchema() *jsonschema.Resolved {
	return w.graph.outputSchema
}

// Run drives the workflow to completion or to a graceful pause
// when any node enters NodeWaiting. It returns an iter.Seq2 that
// yields events from per-node goroutines in arrival order; the
// caller may break from the range loop at any point and the
// engine will cancel all in-flight nodes before returning.
//
// The engine model: each scheduled node runs in its own goroutine
// pushing events into a buffered channel. A single consumer
// goroutine (this one) drains the channel, applies state-side
// effects, yields events to the caller, and schedules successors
// when nodes complete. The consumer is the only mutator of the
// per-node lifecycle map and of session state.
//
// Run has no resume detection: every call starts the graph from Start, as an
// agent built with agent.New(agent.Config{Run: w.Run}) does on each turn. To
// continue a paused or failed run on a later turn, run the workflow with
// workflowagent.New, or as a [NewWorkflowNodeWithOptions] node that is
// runner.Config.Node.
func (w *Workflow) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	var c agent.Context
	if !w.isRootWrapper && w.Name() != "" {
		wfPath := w.Name() + "@1"
		if ac, ok := ctx.(agent.Context); ok && ac.Path() != "" {
			wfPath = ac.Path() + "/" + w.Name() + "@1"
		}
		var ancestors []string
		if ac, ok := ctx.(agent.Context); ok {
			ancestors = ac.OutputForAncestors()
		}
		runID := "1"
		var subScheduler agent.DynamicSubScheduler = nil
		delta := &agent.CommonContextDelta{
			Path:               &wfPath,
			RunID:              &runID,
			OutputForAncestors: &ancestors,
			SubScheduler:       &subScheduler,
		}
		c = agent.PromoteWithDelta(ctx, delta)
	} else {
		c = agent.Promote(ctx)
	}
	return w.RunNode(c, newStartSeed(c.UserContent()))
}

// RunNode drives the workflow with the given input.
// This is used by WorkflowNode to run nested workflows.
//
// With an input schema (see WithInputSchema), input is validated first;
// on failure RunNode yields only the error and schedules nothing.
func (w *Workflow) RunNode(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		input, err := w.validateRootInput(input)
		if err != nil {
			yield(nil, err)
			return
		}
		ctx := w.withStateValidator(ctx)
		s := newScheduler(ctx, w.graph, w.maxConcurrency)
		s.workflowName = w.name
		// Seed: schedule START with the supplied input.
		startState := s.state.EnsureNode(Start.Name())
		startState.Input = input
		s.scheduleNode(Start, input, "", ctx.Branch())

		s.run(yield, false)

		// All goroutines have returned; ensure no leak.
		s.wg.Wait()
	}
}

// validateRootInput conforms input to the workflow's input schema (see
// WithInputSchema) and returns the conformed value. Without a schema it
// returns input untouched, a startSeed included. A failure wraps
// ErrInputValidation and names the workflow, as a node's input failure
// names the node.
func (w *Workflow) validateRootInput(input any) (any, error) {
	validated, err := defaultValidateInput(input, w.inputSchema)
	if err != nil {
		return nil, fmt.Errorf("%w for node %q: %w", ErrInputValidation, w.name, err)
	}
	return validated, nil
}

// startSeed is the input Workflow.Run gives Start: the user message in
// both forms a successor of Start can take. [graph.startSeedFor] picks
// one per successor, so no node other than a [*WorkflowNode] sees a
// startSeed.
type startSeed struct {
	content *genai.Content
	// text is the concatenated text of content's parts, or nil when
	// content is nil.
	text any
}

func newStartSeed(uc *genai.Content) startSeed {
	if uc == nil {
		return startSeed{}
	}
	var sb strings.Builder
	for _, part := range uc.Parts {
		if part != nil && part.Text != "" {
			sb.WriteString(part.Text)
		}
	}
	return startSeed{content: uc, text: sb.String()}
}

// startSeedFor returns the input target receives from Start when Start
// carries seed. A [*WorkflowNode] receives the seed unchanged, so its own
// Start resolves it for each inner successor. A node declaring
// [NodeConfig.ContentInput], and the node root of a workflow built with
// [WithNodeRootWrapper], receive the *genai.Content; every other node
// receives the text. An input that is not a startSeed, such as the
// input a caller passes to [Workflow.RunNode], is returned unchanged.
func (g *graph) startSeedFor(seed any, target Node) any {
	s, ok := seed.(startSeed)
	if !ok {
		return seed
	}
	if _, isWorkflow := target.(*WorkflowNode); isWorkflow {
		return s
	}
	if !target.Config().ContentInput && !g.isNodeRootWrapper {
		return s.text
	}
	if s.content == nil {
		return nil
	}
	return s.content
}
