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

package main

import (
	"bytes"
	"fmt"
	"maps"
	"math"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/google/cel-go/cel"
	"google.golang.org/genai"
	"gopkg.in/yaml.v3"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/workflow"
)

const (
	apiVersion = "adk/v1alpha1"
	kindSystem = "AgentSystem"

	classLLM      = "LlmAgent"
	classWorkflow = "Workflow"

	nodeAgent    = "agent"
	nodeLLM      = "llm"
	nodeCEL      = "cel"
	nodeJoin     = "join"
	nodeTool     = "tool"
	nodeWorkflow = "workflow"
	nodeInput    = "input"

	// startNode and defaultRoute name workflow.Start and workflow.Default in
	// edges.
	startNode    = "START"
	defaultRoute = "DEFAULT"
)

// registry holds what a document refers to by name but cannot declare: the
// models and tools of the Go host.
type registry struct {
	models map[string]model.LLM
	tools  map[string]tool.Tool
}

// systemSpec is an AgentSystem document. The first agent is the root.
type systemSpec struct {
	APIVersion string      `yaml:"apiVersion"`
	Kind       string      `yaml:"kind"`
	Agents     []agentSpec `yaml:"agents"`
}

// agentSpec declares an LlmAgent or a Workflow. The fields after AgentClass
// belong to one class or the other.
type agentSpec struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
	AgentClass  string `yaml:"agent_class"`

	// LlmAgent.
	Model       string   `yaml:"model"`
	Instruction string   `yaml:"instruction"`
	Mode        string   `yaml:"mode"`
	OutputKey   string   `yaml:"output_key"`
	Tools       []string `yaml:"tools"`
	SubAgents   []string `yaml:"sub_agents"`

	// Workflow.
	Nodes []nodeSpec `yaml:"nodes"`
	Edges []edgeSpec `yaml:"edges"`
}

// nodeSpec declares one workflow node. Which fields apply depends on Type:
//
//   - agent reads Agent
//   - llm reads Model, Instruction and Tools
//   - cel reads Vars, Routes, Output and StateDelta
//   - tool reads Tool
//   - workflow reads Nodes and Edges, a nested graph with its own START
//   - input reads Message
//   - join reads none
//
// RetryConfig and Timeout apply to agent, llm and tool nodes; ParallelWorker
// and MaxParallelWorkers to every type but join and input.
type nodeSpec struct {
	ID   string `yaml:"id"`
	Type string `yaml:"type"`

	Agent string `yaml:"agent"`

	Model       string   `yaml:"model"`
	Instruction string   `yaml:"instruction"`
	Tools       []string `yaml:"tools"`

	Vars       []varSpec         `yaml:"vars"`
	Routes     []routeSpec       `yaml:"routes"`
	Output     string            `yaml:"output"`
	StateDelta map[string]string `yaml:"state_delta"`

	Tool string `yaml:"tool"`

	Nodes []nodeSpec `yaml:"nodes"`
	Edges []edgeSpec `yaml:"edges"`

	// Message is a CEL expression: what an input node asks the user.
	Message string `yaml:"message"`

	// ParallelWorker runs the node once per item of a list input and
	// outputs the list of results. MaxParallelWorkers caps the runs in
	// flight; zero is no cap.
	ParallelWorker     bool `yaml:"parallel_worker"`
	MaxParallelWorkers int  `yaml:"max_parallel_workers"`

	// Timeout is in seconds. With ParallelWorker, RetryConfig applies to
	// each item and Timeout to all of them together.
	RetryConfig *retrySpec `yaml:"retry_config"`
	Timeout     float64    `yaml:"timeout"`
}

// retrySpec mirrors workflow.RetryConfig, with delays in seconds. An unset
// field keeps the value of workflow.DefaultRetryConfig. MaxAttempts counts
// the first attempt, so 0 and 1 both mean no retry.
type retrySpec struct {
	MaxAttempts   *int     `yaml:"max_attempts"`
	InitialDelay  *float64 `yaml:"initial_delay"`
	MaxDelay      *float64 `yaml:"max_delay"`
	BackoffFactor *float64 `yaml:"backoff_factor"`
	Jitter        *float64 `yaml:"jitter"`
}

type varSpec struct {
	Name string `yaml:"name"`
	Expr string `yaml:"expr"`
}

type routeSpec struct {
	Route string `yaml:"route"`
	When  string `yaml:"when"`
}

type edgeSpec struct {
	From  string    `yaml:"from_node"`
	To    string    `yaml:"to_node"`
	Route routeList `yaml:"route"`
}

// routeList is the route of an edge: none, one name, or a list of names any
// of which takes the edge.
type routeList []string

// UnmarshalYAML accepts a single route name as well as a list of them.
func (r *routeList) UnmarshalYAML(n *yaml.Node) error {
	if n.Kind == yaml.SequenceNode {
		return n.Decode((*[]string)(r))
	}
	var name string
	if err := n.Decode(&name); err != nil {
		return err
	}
	*r = routeList{name}
	return nil
}

// issue is one problem found in a document, at a field path such as
// agents[1].nodes[5].vars[2].expr.
type issue struct {
	Path    string
	Message string
}

func (i issue) String() string { return i.Path + ": " + i.Message }

// validationError lists every problem found in a document.
type validationError struct {
	Issues []issue
}

func (e *validationError) Error() string {
	lines := make([]string, len(e.Issues))
	for i, is := range e.Issues {
		lines[i] = is.String()
	}
	return "invalid agent system:\n  " + strings.Join(lines, "\n  ")
}

// loadSystem parses doc and builds its agents, root first and the rest in
// document order. reg maps the names the document uses in model, tool and
// tools fields to the models and tools they stand for.
//
// Every problem with the document, including CEL that does not compile, is
// reported at once in a *validationError before any agent is built. Problems
// that only ADK detects, such as an unconditional cycle in a workflow graph,
// are returned as they come from ADK.
func loadSystem(doc []byte, reg registry) ([]agent.Agent, error) {
	var spec systemSpec
	dec := yaml.NewDecoder(bytes.NewReader(doc))
	dec.KnownFields(true)
	if err := dec.Decode(&spec); err != nil {
		return nil, fmt.Errorf("decode agent system: %w", err)
	}
	v := validator{spec: &spec, reg: reg, compiled: compiled{
		cel:      map[*nodeSpec]*celNode{},
		messages: map[*nodeSpec]*expression{},
		configs:  map[*nodeSpec]workflow.NodeConfig{},
	}}
	v.validate()
	if len(v.issues) > 0 {
		return nil, &validationError{Issues: v.issues}
	}
	b := builder{spec: &spec, reg: reg, compiled: v.compiled, built: map[string]agent.Agent{}}
	agents := make([]agent.Agent, len(spec.Agents))
	for i := range spec.Agents {
		a, err := b.agent(spec.Agents[i].Name)
		if err != nil {
			return nil, err
		}
		agents[i] = a
	}
	return agents, nil
}

var namePattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// compiled is what validation works out for the builder, by node.
type compiled struct {
	cel      map[*nodeSpec]*celNode            // cel nodes
	messages map[*nodeSpec]*expression         // input nodes
	configs  map[*nodeSpec]workflow.NodeConfig // retry and timeout
}

type validator struct {
	spec   *systemSpec
	reg    registry
	issues []issue
	compiled
}

func (v *validator) add(path, format string, args ...any) {
	v.issues = append(v.issues, issue{Path: path, Message: fmt.Sprintf(format, args...)})
}

func (v *validator) validate() {
	if v.spec.APIVersion != apiVersion {
		v.add("apiVersion", "got %q, want %q", v.spec.APIVersion, apiVersion)
	}
	if v.spec.Kind != kindSystem {
		v.add("kind", "got %q, want %q", v.spec.Kind, kindSystem)
	}
	if len(v.spec.Agents) == 0 {
		v.add("agents", "at least one agent is required")
	}
	agents := map[string]bool{}
	for i, a := range v.spec.Agents {
		path := fmt.Sprintf("agents[%d]", i)
		v.checkName(path+".name", a.Name)
		if agents[a.Name] {
			v.add(path+".name", "agent %q is declared twice", a.Name)
		}
		agents[a.Name] = true
	}
	for i := range v.spec.Agents {
		path := fmt.Sprintf("agents[%d]", i)
		a := &v.spec.Agents[i]
		switch a.AgentClass {
		case classLLM:
			v.llmAgent(path, a, agents)
		case classWorkflow:
			v.workflow(path, a)
		default:
			v.add(path+".agent_class", "got %q, want %s or %s", a.AgentClass, classLLM, classWorkflow)
		}
	}
}

func (v *validator) checkName(path, name string) {
	if !namePattern.MatchString(name) {
		v.add(path, "%q is not a valid name: use letters, digits and underscores", name)
	}
}

func (v *validator) checkModel(path, name string) {
	if _, ok := v.reg.models[name]; !ok {
		v.add(path, "model %q is not registered", name)
	}
}

func (v *validator) checkTools(path string, names []string) {
	for i, name := range names {
		if _, ok := v.reg.tools[name]; !ok {
			v.add(fmt.Sprintf("%s[%d]", path, i), "tool %q is not registered", name)
		}
	}
}

func (v *validator) llmAgent(path string, a *agentSpec, agents map[string]bool) {
	v.checkModel(path+".model", a.Model)
	v.checkTools(path+".tools", a.Tools)
	switch llmagent.Mode(a.Mode) {
	case "", llmagent.ModeChat, llmagent.ModeTask, llmagent.ModeSingleTurn:
	default:
		v.add(path+".mode", "got %q, want chat, task or single_turn", a.Mode)
	}
	for i, sub := range a.SubAgents {
		if !agents[sub] {
			v.add(fmt.Sprintf("%s.sub_agents[%d]", path, i), "agent %q is not declared", sub)
		}
	}
	if len(a.Nodes) > 0 || len(a.Edges) > 0 {
		v.add(path, "nodes and edges apply only to a %s", classWorkflow)
	}
}

func (v *validator) workflow(path string, a *agentSpec) {
	if a.Model != "" || a.Instruction != "" || a.Mode != "" || a.OutputKey != "" || len(a.Tools) > 0 || len(a.SubAgents) > 0 {
		v.add(path, "model, instruction, mode, output_key, tools and sub_agents apply only to an %s", classLLM)
	}
	v.graph(path, a.Nodes, a.Edges, scope{})
}

// graph checks the nodes and edges of a Workflow agent, or of a workflow node
// nested in one. Edges name the nodes of their own graph only.
func (v *validator) graph(path string, specs []nodeSpec, edges []edgeSpec, in scope) {
	if len(specs) == 0 {
		v.add(path+".nodes", "a workflow needs at least one node")
	}
	nodes := map[string]*nodeSpec{}
	for i := range specs {
		n := &specs[i]
		npath := fmt.Sprintf("%s.nodes[%d]", path, i)
		v.checkName(npath+".id", n.ID)
		if n.ID == startNode {
			v.add(npath+".id", "%s is reserved for the workflow entry", startNode)
		}
		if nodes[n.ID] != nil {
			v.add(npath+".id", "node %q is declared twice", n.ID)
		}
		nodes[n.ID] = n
		v.node(npath, n, in)
	}
	// The workflow engine names an agent node after the agent it runs, not
	// after its id, so the agent's name must not also be a node id.
	for _, n := range specs {
		if n.Type == nodeAgent && nodes[n.Agent] != nil {
			v.add(path+".nodes", "agent %q collides with a node id", n.Agent)
		}
	}
	targets := map[string]bool{}
	for i, e := range edges {
		v.edge(fmt.Sprintf("%s.edges[%d]", path, i), e, nodes)
		targets[e.To] = true
	}
	// The workflow is built from its edges alone, so a node no edge leads to
	// would be dropped without a word.
	for i, n := range specs {
		if !targets[n.ID] {
			v.add(fmt.Sprintf("%s.nodes[%d].id", path, i), "node %q is not the target of any edge, so it never runs", n.ID)
		}
	}
}

// scope is where a graph sits: nested in a workflow node or not, and under a
// parallel worker or not.
type scope struct{ nested, parallel bool }

// nodeTypes are the values of a node's type field.
var nodeTypes = []string{nodeAgent, nodeLLM, nodeCEL, nodeJoin, nodeTool, nodeWorkflow, nodeInput}

// nodeFields says, for each field of a node but id and type, how to tell that
// a document set it and which node types read it.
var nodeFields = []struct {
	name  string
	set   func(*nodeSpec) bool
	types []string
}{
	{"agent", func(n *nodeSpec) bool { return n.Agent != "" }, []string{nodeAgent}},
	{"model", func(n *nodeSpec) bool { return n.Model != "" }, []string{nodeLLM}},
	{"instruction", func(n *nodeSpec) bool { return n.Instruction != "" }, []string{nodeLLM}},
	{"tools", func(n *nodeSpec) bool { return len(n.Tools) > 0 }, []string{nodeLLM}},
	{"vars", func(n *nodeSpec) bool { return len(n.Vars) > 0 }, []string{nodeCEL}},
	{"routes", func(n *nodeSpec) bool { return len(n.Routes) > 0 }, []string{nodeCEL}},
	{"output", func(n *nodeSpec) bool { return n.Output != "" }, []string{nodeCEL}},
	{"state_delta", func(n *nodeSpec) bool { return len(n.StateDelta) > 0 }, []string{nodeCEL}},
	{"tool", func(n *nodeSpec) bool { return n.Tool != "" }, []string{nodeTool}},
	{"nodes", func(n *nodeSpec) bool { return len(n.Nodes) > 0 }, []string{nodeWorkflow}},
	{"edges", func(n *nodeSpec) bool { return len(n.Edges) > 0 }, []string{nodeWorkflow}},
	{"message", func(n *nodeSpec) bool { return n.Message != "" }, []string{nodeInput}},
	// A join or a pause has no list of items to work through.
	{"parallel_worker", func(n *nodeSpec) bool { return n.ParallelWorker }, []string{nodeAgent, nodeLLM, nodeCEL, nodeTool, nodeWorkflow}},
	// Retrying or timing out an expression, a join, a pause or a whole graph
	// would change nothing, or too much.
	{"retry_config", func(n *nodeSpec) bool { return n.RetryConfig != nil }, []string{nodeAgent, nodeLLM, nodeTool}},
	{"timeout", func(n *nodeSpec) bool { return n.Timeout != 0 }, []string{nodeAgent, nodeLLM, nodeTool}},
}

// misapplied returns the fields set on n that a node of its type ignores.
func misapplied(n *nodeSpec) []string {
	var names []string
	for _, f := range nodeFields {
		if f.set(n) && !slices.Contains(f.types, n.Type) {
			names = append(names, f.name)
		}
	}
	return names
}

func (v *validator) node(path string, n *nodeSpec, in scope) {
	if !slices.Contains(nodeTypes, n.Type) {
		v.add(path+".type", "got %q, want one of %s", n.Type, strings.Join(nodeTypes, ", "))
		return
	}
	for _, field := range misapplied(n) {
		v.add(path+"."+field, "%s does not apply to %s nodes", field, n.Type)
	}
	// A parallel worker yields only the list of its results. Whatever else
	// its node, or a graph nested in it, emits is dropped.
	dropped := in.parallel || n.ParallelWorker
	const lost = "a parallel worker drops %s"

	switch n.Type {
	case nodeAgent:
		family := v.family(n.Agent)
		if family == nil {
			v.add(path+".agent", "agent %q is not declared", n.Agent)
		}
		// What the node cannot run, the agents it can transfer to cannot be
		// either.
		for _, a := range family {
			switch {
			case slices.ContainsFunc(a.Nodes, func(n nodeSpec) bool { return n.Type == nodeInput }):
				// The reply would become the output of this node, skipping
				// the rest of the agent's graph.
				v.add(path+".agent", "workflow %q pauses for input, which it cannot do under an agent node", a.Name)
			case dropped && a.AgentClass == classWorkflow:
				// Its graph is checked as its own agent, not as one under a
				// parallel worker.
				v.add(path+".agent", "a parallel worker cannot run workflow agent %q: nest its graph in a workflow node", a.Name)
			case dropped && (len(a.Tools) > 0 || a.OutputKey != ""):
				v.add(path+".agent", lost, fmt.Sprintf("the tool calls and output_key of agent %q", a.Name))
			}
		}
	case nodeLLM:
		v.checkModel(path+".model", n.Model)
		v.checkTools(path+".tools", n.Tools)
		if dropped && len(n.Tools) > 0 {
			v.add(path+".tools", lost, "tool calls, so the model never sees a result")
		}
	case nodeCEL:
		v.celNode(path, n)
		if n.ParallelWorker && len(n.Routes) > 0 {
			v.add(path+".routes", lost, "routes")
		}
		if dropped && len(n.StateDelta) > 0 {
			v.add(path+".state_delta", lost, "state changes")
		}
	case nodeTool:
		if _, ok := v.reg.tools[n.Tool]; !ok {
			v.add(path+".tool", "tool %q is not registered", n.Tool)
		}
	case nodeWorkflow:
		v.graph(path, n.Nodes, n.Edges, scope{nested: true, parallel: dropped})
	case nodeInput:
		if in.nested {
			// The reply would become the output of the workflow node,
			// skipping the rest of its graph.
			v.add(path+".type", "an input node cannot be in a nested workflow")
		}
		// An input node declares no vars, so its message can read none.
		v.messages[n] = v.compile(path+".message", n.Message, cel.StringType, nil)
	}

	if n.MaxParallelWorkers != 0 && !n.ParallelWorker {
		v.add(path+".max_parallel_workers", "max_parallel_workers applies only with parallel_worker")
	}
	if n.MaxParallelWorkers < 0 {
		v.add(path+".max_parallel_workers", "must not be negative")
	}
	cfg, issues := nodeConfig(n)
	for _, is := range issues {
		v.add(path+is.Path, "%s", is.Message)
	}
	v.configs[n] = cfg
}

// family returns the agent declared as name, followed by every agent it can
// transfer to through sub_agents at any depth. It is nil when no agent is
// declared as name.
func (v *validator) family(name string) []*agentSpec {
	var family []*agentSpec
	seen := map[string]bool{}
	var visit func(name string)
	visit = func(name string) {
		i := slices.IndexFunc(v.spec.Agents, func(a agentSpec) bool { return a.Name == name })
		if i < 0 || seen[name] {
			return
		}
		seen[name] = true
		a := &v.spec.Agents[i]
		family = append(family, a)
		for _, sub := range a.SubAgents {
			visit(sub)
		}
	}
	visit(name)
	return family
}

// nodeConfig translates the timeout and retry_config of n, which count in
// seconds. The paths of the issues it returns are relative to the node.
func nodeConfig(n *nodeSpec) (workflow.NodeConfig, []issue) {
	var issues []issue
	check := func(ok bool, path, message string) {
		if !ok {
			issues = append(issues, issue{Path: path, Message: message})
		}
	}
	// A time.Duration holds about 292 years.
	const maxSeconds = math.MaxInt64 / int64(time.Second)
	seconds := func(path string, s float64) time.Duration {
		check(s >= 0 && s <= float64(maxSeconds), path, fmt.Sprintf("must be between 0 and %d seconds", maxSeconds))
		return time.Duration(s * float64(time.Second))
	}
	cfg := workflow.NodeConfig{Timeout: seconds(".timeout", n.Timeout)}
	r := n.RetryConfig
	if r == nil {
		return cfg, issues
	}
	rc := workflow.DefaultRetryConfig()
	if r.MaxAttempts != nil {
		check(*r.MaxAttempts >= 0, ".retry_config.max_attempts", "must not be negative")
		rc.MaxAttempts = *r.MaxAttempts
	}
	if r.InitialDelay != nil {
		rc.InitialDelay = seconds(".retry_config.initial_delay", *r.InitialDelay)
	}
	if r.MaxDelay != nil {
		rc.MaxDelay = seconds(".retry_config.max_delay", *r.MaxDelay)
	}
	if r.BackoffFactor != nil {
		check(*r.BackoffFactor >= 0, ".retry_config.backoff_factor", "must not be negative")
		rc.BackoffFactor = *r.BackoffFactor
	}
	if r.Jitter != nil {
		check(*r.Jitter >= 0 && *r.Jitter <= 1, ".retry_config.jitter", "must be between 0 and 1")
		rc.Jitter = *r.Jitter
	}
	cfg.RetryConfig = rc
	return cfg, issues
}

// compile type-checks a CEL expression that may read the vars in declared,
// and reports what is wrong with it at path. A non-nil want is the type the
// expression must have.
func (v *validator) compile(path, source string, want *cel.Type, declared map[string]bool) *expression {
	e, err := compileCEL(source, want)
	if err != nil {
		v.add(path, "%v", err)
		return nil
	}
	for _, ref := range e.varRefs {
		if !declared[ref] {
			v.add(path, "vars.%s is not declared before this expression", ref)
		}
	}
	return e
}

// celNode compiles every expression of n. Each expression may read only the
// vars declared before it; routes, output and state_delta may read them all.
func (v *validator) celNode(path string, n *nodeSpec) {
	c := &celNode{id: n.ID}
	declared := map[string]bool{}
	compile := func(path, source string, want *cel.Type) *expression {
		return v.compile(path, source, want, declared)
	}
	for i, vs := range n.Vars {
		vpath := fmt.Sprintf("%s.vars[%d]", path, i)
		v.checkName(vpath+".name", vs.Name)
		if declared[vs.Name] {
			v.add(vpath+".name", "var %q is declared twice", vs.Name)
		}
		c.vars = append(c.vars, celVar{name: vs.Name, expr: compile(vpath+".expr", vs.Expr, nil)})
		declared[vs.Name] = true
	}
	for i, r := range n.Routes {
		rpath := fmt.Sprintf("%s.routes[%d]", path, i)
		if r.Route == "" || r.Route == defaultRoute {
			v.add(rpath+".route", "a route name is required and must not be %s", defaultRoute)
		}
		c.routes = append(c.routes, celRoute{route: r.Route, when: compile(rpath+".when", r.When, cel.BoolType)})
	}
	if n.Output != "" {
		c.output = compile(path+".output", n.Output, nil)
	}
	for _, key := range slices.Sorted(maps.Keys(n.StateDelta)) {
		c.stateDelta = append(c.stateDelta, celVar{name: key, expr: compile(path+".state_delta."+key, n.StateDelta[key], nil)})
	}
	v.cel[n] = c
}

func (v *validator) edge(path string, e edgeSpec, nodes map[string]*nodeSpec) {
	from := nodes[e.From]
	if from == nil && e.From != startNode {
		v.add(path+".from_node", "node %q is not declared", e.From)
	}
	if nodes[e.To] == nil {
		v.add(path+".to_node", "node %q is not declared", e.To)
	}
	if e.From == startNode && len(e.Route) > 0 {
		v.add(path+".route", "%s emits no routes", startNode)
	}
	if len(e.Route) > 1 && slices.Contains(e.Route, defaultRoute) {
		v.add(path+".route", "%s cannot be combined with other routes", defaultRoute)
	}
	if from == nil {
		return
	}
	for _, route := range e.Route {
		if route == defaultRoute {
			continue
		}
		if from.Type != nodeCEL {
			v.add(path+".route", "only a cel node emits routes, and %q is a %s node", e.From, from.Type)
			return
		}
		if !slices.ContainsFunc(from.Routes, func(r routeSpec) bool { return r.Route == route }) {
			v.add(path+".route", "node %q never emits route %q", e.From, route)
		}
	}
}

type builder struct {
	spec *systemSpec
	reg  registry
	compiled
	built map[string]agent.Agent
	// building holds the agents being built, to report a cycle of
	// references instead of recursing forever.
	building []string
}

// agent returns the agent declared as name, building it and the agents it
// refers to on first use. An agent is built once, so it has one parent.
func (b *builder) agent(name string) (agent.Agent, error) {
	if a, ok := b.built[name]; ok {
		return a, nil
	}
	if slices.Contains(b.building, name) {
		return nil, fmt.Errorf("agent %q refers to itself through %s", name, strings.Join(b.building, " -> "))
	}
	b.building = append(b.building, name)
	defer func() { b.building = b.building[:len(b.building)-1] }()

	i := slices.IndexFunc(b.spec.Agents, func(a agentSpec) bool { return a.Name == name })
	spec := &b.spec.Agents[i]
	var (
		a   agent.Agent
		err error
	)
	if spec.AgentClass == classWorkflow {
		a, err = b.workflowAgent(spec)
	} else {
		a, err = b.llmAgent(spec)
	}
	if err != nil {
		return nil, fmt.Errorf("build agent %q: %w", name, err)
	}
	b.built[name] = a
	return a, nil
}

func (b *builder) llmAgent(spec *agentSpec) (agent.Agent, error) {
	subAgents := make([]agent.Agent, len(spec.SubAgents))
	for i, name := range spec.SubAgents {
		sub, err := b.agent(name)
		if err != nil {
			return nil, err
		}
		subAgents[i] = sub
	}
	return llmagent.New(llmagent.Config{
		Name:        spec.Name,
		Description: spec.Description,
		Model:       b.reg.models[spec.Model],
		Instruction: spec.Instruction,
		Mode:        llmagent.Mode(spec.Mode),
		OutputKey:   spec.OutputKey,
		Tools:       b.tools(spec.Tools),
		SubAgents:   subAgents,
	})
}

func (b *builder) tools(names []string) []tool.Tool {
	tools := make([]tool.Tool, len(names))
	for i, name := range names {
		tools[i] = b.reg.tools[name]
	}
	return tools
}

func (b *builder) workflowAgent(spec *agentSpec) (agent.Agent, error) {
	edges, err := b.graph(spec.Nodes, spec.Edges, true)
	if err != nil {
		return nil, err
	}
	return workflowagent.New(workflowagent.Config{
		Name:        spec.Name,
		Description: spec.Description,
		Edges:       edges,
	})
}

// graph builds the nodes of one graph and connects them. shown says whether
// the output of the graph is the workflow's reply; see node.
func (b *builder) graph(specs []nodeSpec, edgeSpecs []edgeSpec, shown bool) ([]workflow.Edge, error) {
	hasSuccessor := map[string]bool{}
	for _, e := range edgeSpecs {
		hasSuccessor[e.From] = true
	}
	nodes := map[string]workflow.Node{startNode: workflow.Start}
	for i := range specs {
		n := &specs[i]
		node, err := b.node(n, shown && !hasSuccessor[n.ID])
		if err != nil {
			return nil, fmt.Errorf("node %q: %w", n.ID, err)
		}
		nodes[n.ID] = node
	}
	edges := make([]workflow.Edge, len(edgeSpecs))
	for i, e := range edgeSpecs {
		edges[i] = workflow.Edge{From: nodes[e.From], To: nodes[e.To]}
		switch {
		case len(e.Route) == 0:
		case e.Route[0] == defaultRoute:
			edges[i].Route = workflow.Default
		case len(e.Route) == 1:
			edges[i].Route = workflow.StringRoute(e.Route[0])
		default:
			edges[i].Route = workflow.MultiRoute[string](e.Route)
		}
	}
	return edges, nil
}

// node builds n. A terminal node has no outgoing edges and ends a graph whose
// output is the workflow's reply, so a cel node there also sets its output as
// the event's content.
func (b *builder) node(n *nodeSpec, terminal bool) (workflow.Node, error) {
	cfg := b.configs[n]
	if !n.ParallelWorker {
		return b.bareNode(n, terminal, cfg)
	}
	// The worker applies the retry to each item, and rejects a node that has
	// one of its own. It outputs the list of results and shows none of them.
	worker, err := b.bareNode(n, terminal, workflow.NodeConfig{})
	if err != nil {
		return nil, err
	}
	return workflow.NewParallelWorker(n.ID, worker, n.MaxParallelWorkers, cfg)
}

func (b *builder) bareNode(n *nodeSpec, terminal bool, cfg workflow.NodeConfig) (workflow.Node, error) {
	switch n.Type {
	case nodeAgent:
		a, err := b.agent(n.Agent)
		if err != nil {
			return nil, err
		}
		return workflow.NewAgentNode(a, cfg)
	case nodeLLM:
		a, err := llmagent.New(llmagent.Config{
			Name:        n.ID,
			Model:       b.reg.models[n.Model],
			Instruction: n.Instruction,
			Mode:        llmagent.ModeSingleTurn,
			Tools:       b.tools(n.Tools),
		})
		if err != nil {
			return nil, err
		}
		return workflow.NewAgentNode(a, cfg)
	case nodeCEL:
		c := b.cel[n]
		c.terminal = terminal
		return workflow.NewFunctionNode(n.ID, c.run, workflow.NodeConfig{}), nil
	case nodeJoin:
		return workflow.NewJoinNode(n.ID), nil
	case nodeTool:
		return workflow.NewNamedToolNode(n.ID, b.reg.tools[n.Tool], cfg)
	case nodeWorkflow:
		edges, err := b.graph(n.Nodes, n.Edges, terminal)
		if err != nil {
			return nil, err
		}
		return workflow.NewWorkflowNode(n.ID, edges)
	case nodeInput:
		in := &inputNode{id: n.ID, message: b.messages[n]}
		return workflow.NewFunctionNode(n.ID, in.run, workflow.NodeConfig{}), nil
	}
	return nil, fmt.Errorf("unknown node type %q", n.Type)
}

// inputNode is the compiled form of an input node.
type inputNode struct {
	id      string
	message *expression
}

// run returns the event that asks the user for input, which pauses the
// workflow. The user's reply resumes it on the next turn, as the input of the
// nodes that follow this one.
func (n *inputNode) run(ctx agent.Context, input any) (*session.Event, error) {
	activation := contextVars(ctx)
	activation["input"] = plainValue(input)
	msg, err := n.message.evalString(activation)
	if err != nil {
		return nil, fmt.Errorf("input node %s: message: %w", n.id, err)
	}
	return workflow.NewRequestInputEvent(ctx, session.RequestInput{Message: msg}), nil
}

// celNode is the compiled form of a cel node.
type celNode struct {
	id         string
	vars       []celVar
	routes     []celRoute
	output     *expression // nil forwards the input
	stateDelta []celVar    // sorted by name
	terminal   bool
}

type celVar struct {
	name string
	expr *expression
}

type celRoute struct {
	route string
	when  *expression
}

// run evaluates the vars in order, then takes the first route whose when is
// true, the output, and the state delta. Every expression sees the state as
// it was before the node ran.
func (c *celNode) run(ctx agent.Context, input any) (*session.Event, error) {
	activation := contextVars(ctx)
	activation["input"] = plainValue(input)
	vars := map[string]any{}
	activation["vars"] = vars
	for _, v := range c.vars {
		val, err := v.expr.eval(activation)
		if err != nil {
			return nil, fmt.Errorf("cel node %s: vars.%s: %w", c.id, v.name, err)
		}
		vars[v.name] = val
	}

	ev := session.NewEvent(ctx, ctx.InvocationID())
	for _, r := range c.routes {
		ok, err := r.when.evalBool(activation)
		if err != nil {
			return nil, fmt.Errorf("cel node %s: route %s: %w", c.id, r.route, err)
		}
		if ok {
			ev.Routes = []string{r.route}
			break
		}
	}

	ev.Output = activation["input"]
	if c.output != nil {
		out, err := c.output.eval(activation)
		if err != nil {
			return nil, fmt.Errorf("cel node %s: output: %w", c.id, err)
		}
		ev.Output = out
	}
	if c.terminal && ev.Output != nil {
		ev.Content = genai.NewContentFromText(fmt.Sprint(ev.Output), genai.RoleModel)
	}

	if len(c.stateDelta) > 0 {
		ev.Actions.StateDelta = make(map[string]any, len(c.stateDelta))
		for _, d := range c.stateDelta {
			val, err := d.expr.eval(activation)
			if err != nil {
				return nil, fmt.Errorf("cel node %s: state_delta.%s: %w", c.id, d.name, err)
			}
			ev.Actions.StateDelta[d.name] = val
		}
	}
	return ev, nil
}
