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
	"regexp"
	"slices"
	"strings"

	"google.golang.org/genai"
	"gopkg.in/yaml.v3"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

const (
	apiVersion = "adk/v1alpha1"
	kindSystem = "AgentSystem"

	classLLM      = "LlmAgent"
	classWorkflow = "Workflow"

	nodeAgent = "agent"
	nodeLLM   = "llm"
	nodeCEL   = "cel"
	nodeJoin  = "join"

	// startNode and defaultRoute name workflow.Start and workflow.Default in
	// edges.
	startNode    = "START"
	defaultRoute = "DEFAULT"
)

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
	SubAgents   []string `yaml:"sub_agents"`

	// Workflow.
	Nodes []nodeSpec `yaml:"nodes"`
	Edges []edgeSpec `yaml:"edges"`
}

// nodeSpec declares one workflow node. Which fields apply depends on Type:
// agent reads Agent; llm reads Model and Instruction; cel reads Vars,
// Routes, Output and StateDelta; join reads none.
type nodeSpec struct {
	ID   string `yaml:"id"`
	Type string `yaml:"type"`

	Agent string `yaml:"agent"`

	Model       string `yaml:"model"`
	Instruction string `yaml:"instruction"`

	Vars       []varSpec         `yaml:"vars"`
	Routes     []routeSpec       `yaml:"routes"`
	Output     string            `yaml:"output"`
	StateDelta map[string]string `yaml:"state_delta"`
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
	From  string `yaml:"from_node"`
	To    string `yaml:"to_node"`
	Route string `yaml:"route"`
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
// document order. models maps the names the document uses in model fields to
// the models they stand for.
//
// Every problem with the document, including CEL that does not compile, is
// reported at once in a *validationError before any agent is built. Problems
// that only ADK detects, such as an unconditional cycle in a workflow graph,
// are returned as they come from ADK.
func loadSystem(doc []byte, models map[string]model.LLM) ([]agent.Agent, error) {
	var spec systemSpec
	dec := yaml.NewDecoder(bytes.NewReader(doc))
	dec.KnownFields(true)
	if err := dec.Decode(&spec); err != nil {
		return nil, fmt.Errorf("decode agent system: %w", err)
	}
	v := validator{spec: &spec, models: models, cel: map[*nodeSpec]*celNode{}}
	v.validate()
	if len(v.issues) > 0 {
		return nil, &validationError{Issues: v.issues}
	}
	b := builder{spec: &spec, models: models, cel: v.cel, built: map[string]agent.Agent{}}
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

type validator struct {
	spec   *systemSpec
	models map[string]model.LLM
	issues []issue
	// cel holds the compiled programs of each cel node, for the builder.
	cel map[*nodeSpec]*celNode
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
			v.workflow(path, a, agents)
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
	if _, ok := v.models[name]; !ok {
		v.add(path, "model %q is not registered", name)
	}
}

func (v *validator) llmAgent(path string, a *agentSpec, agents map[string]bool) {
	v.checkModel(path+".model", a.Model)
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

func (v *validator) workflow(path string, a *agentSpec, agents map[string]bool) {
	if a.Model != "" || a.Instruction != "" || a.Mode != "" || a.OutputKey != "" || len(a.SubAgents) > 0 {
		v.add(path, "model, instruction, mode, output_key and sub_agents apply only to an %s", classLLM)
	}
	if len(a.Nodes) == 0 {
		v.add(path+".nodes", "a workflow needs at least one node")
	}
	nodes := map[string]*nodeSpec{}
	for i := range a.Nodes {
		n := &a.Nodes[i]
		npath := fmt.Sprintf("%s.nodes[%d]", path, i)
		v.checkName(npath+".id", n.ID)
		if n.ID == startNode {
			v.add(npath+".id", "%s is reserved for the workflow entry", startNode)
		}
		if nodes[n.ID] != nil {
			v.add(npath+".id", "node %q is declared twice", n.ID)
		}
		nodes[n.ID] = n
		v.node(npath, n, agents)
	}
	// The workflow engine names an agent node after the agent it runs, not
	// after its id, so the agent's name must not also be a node id.
	for _, n := range a.Nodes {
		if n.Type == nodeAgent && nodes[n.Agent] != nil {
			v.add(path+".nodes", "agent %q collides with a node id", n.Agent)
		}
	}
	targets := map[string]bool{}
	for i, e := range a.Edges {
		v.edge(fmt.Sprintf("%s.edges[%d]", path, i), e, nodes)
		targets[e.To] = true
	}
	// The workflow is built from its edges alone, so a node no edge leads to
	// would be dropped without a word.
	for i, n := range a.Nodes {
		if !targets[n.ID] {
			v.add(fmt.Sprintf("%s.nodes[%d].id", path, i), "node %q is not the target of any edge, so it never runs", n.ID)
		}
	}
}

func (v *validator) node(path string, n *nodeSpec, agents map[string]bool) {
	switch n.Type {
	case nodeAgent:
		if !agents[n.Agent] {
			v.add(path+".agent", "agent %q is not declared", n.Agent)
		}
	case nodeLLM:
		v.checkModel(path+".model", n.Model)
	case nodeCEL:
		v.celNode(path, n)
	case nodeJoin:
	default:
		v.add(path+".type", "got %q, want %s, %s, %s or %s", n.Type, nodeAgent, nodeLLM, nodeCEL, nodeJoin)
	}
}

// celNode compiles every expression of n. Each expression may read only the
// vars declared before it; routes, output and state_delta may read them all.
func (v *validator) celNode(path string, n *nodeSpec) {
	c := &celNode{id: n.ID}
	declared := map[string]bool{}
	compile := func(path, source string, wantBool bool) *expression {
		e, err := compileCEL(source, wantBool)
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
	for i, vs := range n.Vars {
		vpath := fmt.Sprintf("%s.vars[%d]", path, i)
		v.checkName(vpath+".name", vs.Name)
		if declared[vs.Name] {
			v.add(vpath+".name", "var %q is declared twice", vs.Name)
		}
		c.vars = append(c.vars, celVar{name: vs.Name, expr: compile(vpath+".expr", vs.Expr, false)})
		declared[vs.Name] = true
	}
	for i, r := range n.Routes {
		rpath := fmt.Sprintf("%s.routes[%d]", path, i)
		if r.Route == "" || r.Route == defaultRoute {
			v.add(rpath+".route", "a route name is required and must not be %s", defaultRoute)
		}
		c.routes = append(c.routes, celRoute{route: r.Route, when: compile(rpath+".when", r.When, true)})
	}
	if n.Output != "" {
		c.output = compile(path+".output", n.Output, false)
	}
	for _, key := range slices.Sorted(maps.Keys(n.StateDelta)) {
		c.stateDelta = append(c.stateDelta, celVar{name: key, expr: compile(path+".state_delta."+key, n.StateDelta[key], false)})
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
	if e.From == startNode && e.Route != "" {
		v.add(path+".route", "%s emits no routes", startNode)
	}
	if e.Route == "" || e.Route == defaultRoute || from == nil {
		return
	}
	if from.Type != nodeCEL {
		v.add(path+".route", "only a cel node emits routes, and %q is a %s node", e.From, from.Type)
		return
	}
	if !slices.ContainsFunc(from.Routes, func(r routeSpec) bool { return r.Route == e.Route }) {
		v.add(path+".route", "node %q never emits route %q", e.From, e.Route)
	}
}

type builder struct {
	spec   *systemSpec
	models map[string]model.LLM
	cel    map[*nodeSpec]*celNode
	built  map[string]agent.Agent
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
		Model:       b.models[spec.Model],
		Instruction: spec.Instruction,
		Mode:        llmagent.Mode(spec.Mode),
		OutputKey:   spec.OutputKey,
		SubAgents:   subAgents,
	})
}

func (b *builder) workflowAgent(spec *agentSpec) (agent.Agent, error) {
	hasSuccessor := map[string]bool{}
	for _, e := range spec.Edges {
		hasSuccessor[e.From] = true
	}
	nodes := map[string]workflow.Node{startNode: workflow.Start}
	for i := range spec.Nodes {
		n := &spec.Nodes[i]
		node, err := b.node(n, !hasSuccessor[n.ID])
		if err != nil {
			return nil, fmt.Errorf("node %q: %w", n.ID, err)
		}
		nodes[n.ID] = node
	}
	edges := make([]workflow.Edge, len(spec.Edges))
	for i, e := range spec.Edges {
		edges[i] = workflow.Edge{From: nodes[e.From], To: nodes[e.To]}
		switch e.Route {
		case "":
		case defaultRoute:
			edges[i].Route = workflow.Default
		default:
			edges[i].Route = workflow.StringRoute(e.Route)
		}
	}
	return workflowagent.New(workflowagent.Config{
		Name:        spec.Name,
		Description: spec.Description,
		Edges:       edges,
	})
}

// node builds n. A terminal node has no outgoing edges; its output is the
// workflow's reply, so a cel node also sets it as the event's content.
func (b *builder) node(n *nodeSpec, terminal bool) (workflow.Node, error) {
	switch n.Type {
	case nodeAgent:
		a, err := b.agent(n.Agent)
		if err != nil {
			return nil, err
		}
		return workflow.NewAgentNode(a, workflow.NodeConfig{})
	case nodeLLM:
		a, err := llmagent.New(llmagent.Config{
			Name:        n.ID,
			Model:       b.models[n.Model],
			Instruction: n.Instruction,
			Mode:        llmagent.ModeSingleTurn,
		})
		if err != nil {
			return nil, err
		}
		return workflow.NewAgentNode(a, workflow.NodeConfig{})
	case nodeCEL:
		c := b.cel[n]
		c.terminal = terminal
		return workflow.NewFunctionNode(n.ID, c.run, workflow.NodeConfig{}), nil
	case nodeJoin:
		return workflow.NewJoinNode(n.ID), nil
	}
	return nil, fmt.Errorf("unknown node type %q", n.Type)
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
