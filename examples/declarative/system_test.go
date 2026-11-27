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
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/workflow"
)

// editedSystem returns agent-system.yaml with the first old replaced by new.
func editedSystem(t *testing.T, old, new string) []byte {
	t.Helper()
	doc := strings.Replace(string(agentSystem), old, new, 1)
	if doc == string(agentSystem) {
		t.Fatalf("%q not found in agent-system.yaml", old)
	}
	return []byte(doc)
}

// TestLoadSystemReportsIssues checks that a broken document is rejected with
// the field path of each problem, before any agent is built.
func TestLoadSystemReportsIssues(t *testing.T) {
	tests := []struct {
		name     string
		old, new string // replacement applied to agent-system.yaml
		wantPath string
		wantMsg  string // substring of the issue's message
	}{
		{
			name:     "wrong apiVersion",
			old:      "apiVersion: adk/v1alpha1",
			new:      "apiVersion: adk/v2",
			wantPath: "apiVersion",
			wantMsg:  `got "adk/v2"`,
		},
		{
			name:     "wrong kind",
			old:      "kind: AgentSystem",
			new:      "kind: Agent",
			wantPath: "kind",
			wantMsg:  `got "Agent"`,
		},
		{
			name:     "unknown agent_class",
			old:      "agent_class: LlmAgent",
			new:      "agent_class: SequentialAgent",
			wantPath: "agents[0].agent_class",
			wantMsg:  `got "SequentialAgent"`,
		},
		{
			name:     "invalid agent name",
			old:      "- name: studio",
			new:      "- name: my-studio",
			wantPath: "agents[0].name",
			wantMsg:  "is not a valid name",
		},
		{
			name:     "agent declared twice",
			old:      "- name: writer\n",
			new:      "- name: writer_critics\n",
			wantPath: "agents[2].name",
			wantMsg:  `agent "writer_critics" is declared twice`,
		},
		{
			name:     "unknown mode",
			old:      "mode: single_turn",
			new:      "mode: once",
			wantPath: "agents[2].mode",
			wantMsg:  `got "once"`,
		},
		{
			name:     "LlmAgent field on a Workflow",
			old:      "    agent_class: Workflow\n",
			new:      "    agent_class: Workflow\n    output_key: draft\n",
			wantPath: "agents[1]",
			wantMsg:  "apply only to an LlmAgent",
		},
		{
			name:     "node declared twice",
			old:      "{ id: gather, type: join }",
			new:      "{ id: decide, type: join }",
			wantPath: "agents[1].nodes[5].id",
			wantMsg:  `node "decide" is declared twice`,
		},
		{
			name:     "node named START",
			old:      "{ id: gather, type: join }",
			new:      "{ id: START, type: join }",
			wantPath: "agents[1].nodes[4].id",
			wantMsg:  "START is reserved",
		},
		{
			name:     "agent node runs an undeclared agent",
			old:      "agent: writer }",
			new:      "agent: writr }",
			wantPath: "agents[1].nodes[0].agent",
			wantMsg:  `agent "writr" is not declared`,
		},
		{
			name:     "node no edge leads to",
			old:      "      - { from_node: gather,          to_node: decide }\n",
			new:      "",
			wantPath: "agents[1].nodes[5].id",
			wantMsg:  `node "decide" is not the target of any edge`,
		},
		{
			name:     "var declared twice",
			old:      "- name: done",
			new:      "- name: turn",
			wantPath: "agents[1].nodes[5].vars[2].name",
			wantMsg:  `var "turn" is declared twice`,
		},
		{
			name:     "CEL indexes an undeclared var",
			old:      `expr: "vars.approved == 3 || vars.turn >= 5"`,
			new:      `expr: "vars['approvd'] == 3 || vars.turn >= 5"`,
			wantPath: "agents[1].nodes[5].vars[2].expr",
			wantMsg:  "vars.approvd is not declared before this expression",
		},
		{
			name:     "CEL optionally selects an undeclared var",
			old:      `expr: "vars.approved == 3 || vars.turn >= 5"`,
			new:      `expr: "vars.?approvd.orValue(0) == 3 || vars.turn >= 5"`,
			wantPath: "agents[1].nodes[5].vars[2].expr",
			wantMsg:  "vars.approvd is not declared before this expression",
		},
		{
			name:     "CEL optionally indexes an undeclared var",
			old:      `expr: "vars.approved == 3 || vars.turn >= 5"`,
			new:      `expr: "vars[?'approvd'].orValue(0) == 3 || vars.turn >= 5"`,
			wantPath: "agents[1].nodes[5].vars[2].expr",
			wantMsg:  "vars.approvd is not declared before this expression",
		},
		{
			name:     "CEL reads a var by a computed name",
			old:      `expr: "vars.approved == 3 || vars.turn >= 5"`,
			new:      `expr: "vars[state.key] == 3 || vars.turn >= 5"`,
			wantPath: "agents[1].nodes[5].vars[2].expr",
			wantMsg:  "vars must be read by a constant name",
		},
		{
			name:     "cel route named DEFAULT",
			old:      "{ route: REVISE, when:",
			new:      "{ route: DEFAULT, when:",
			wantPath: "agents[1].nodes[5].routes[1].route",
			wantMsg:  "must not be DEFAULT",
		},
		{
			name:     "edge from an undeclared node",
			old:      "{ from_node: gather,          to_node: decide }",
			new:      "{ from_node: gathr,           to_node: decide }",
			wantPath: "agents[1].edges[7].from_node",
			wantMsg:  `node "gathr" is not declared`,
		},
		{
			name:     "edge route from START",
			old:      "{ from_node: START,    to_node: classify }",
			new:      "{ from_node: START,    to_node: classify, route: DEFAULT }",
			wantPath: "agents[3].edges[0].route",
			wantMsg:  "START emits no routes",
		},
		{
			name:     "CEL reads a var declared after it",
			old:      `? state.turn + 1 : 1"`,
			new:      `? state.turn + 1 : (vars.done ? 0 : 1)"`,
			wantPath: "agents[1].nodes[5].vars[0].expr",
			wantMsg:  "vars.done is not declared before this expression",
		},
		{
			name:     "CEL var reads itself",
			old:      `? state.turn + 1 : 1"`,
			new:      `? state.turn + 1 : (vars.turn > 0 ? 1 : 0)"`,
			wantPath: "agents[1].nodes[5].vars[0].expr",
			wantMsg:  "vars.turn is not declared before this expression",
		},
		{
			name:     "CEL reads an undeclared var",
			old:      `expr: "vars.approved == 3 || vars.turn >= 5"`,
			new:      `expr: "vars.approvd == 3 || vars.turn >= 5"`,
			wantPath: "agents[1].nodes[5].vars[2].expr",
			wantMsg:  "vars.approvd is not declared before this expression",
		},
		{
			name:     "CEL does not compile",
			old:      "output: user.content",
			new:      "output: user.content +",
			wantPath: "agents[3].nodes[1].output",
			wantMsg:  "Syntax error",
		},
		{
			name:     "route condition is not a bool",
			old:      `when: "vars.done"`,
			new:      `when: "'DONE'"`,
			wantPath: "agents[1].nodes[5].routes[0].when",
			wantMsg:  "want bool",
		},
		{
			name:     "edge route is never emitted",
			old:      "route: REVISE }",
			new:      "route: REVISED }",
			wantPath: "agents[1].edges[9].route",
			wantMsg:  `node "decide" never emits route "REVISED"`,
		},
		{
			name:     "edge route from a node that cannot emit routes",
			old:      "{ from_node: classify, to_node: route }",
			new:      "{ from_node: classify, to_node: route, route: QUESTION }",
			wantPath: "agents[3].edges[1].route",
			wantMsg:  `only a cel node emits routes`,
		},
		{
			name:     "edge to an undeclared node",
			old:      "{ from_node: gather,          to_node: decide }",
			new:      "{ from_node: gather,          to_node: decider }",
			wantPath: "agents[1].edges[7].to_node",
			wantMsg:  `node "decider" is not declared`,
		},
		{
			name:     "agent node shares an id with its agent",
			old:      "{ id: write, type: agent, agent: writer }",
			new:      "{ id: writer, type: agent, agent: writer }",
			wantPath: "agents[1].nodes",
			wantMsg:  `agent "writer" collides with a node id`,
		},
		{
			name:     "sub-agent is not declared",
			old:      "sub_agents: [writer_critics, llm_router]",
			new:      "sub_agents: [writer_critics, chat_router]",
			wantPath: "agents[0].sub_agents[1]",
			wantMsg:  `agent "chat_router" is not declared`,
		},
		{
			name:     "model is not registered",
			old:      "type: llm\n        model: default",
			new:      "type: llm\n        model: pro",
			wantPath: "agents[1].nodes[1].model",
			wantMsg:  `model "pro" is not registered`,
		},
		{
			name:     "unknown node type",
			old:      "{ id: gather, type: join }",
			new:      "{ id: gather, type: merge }",
			wantPath: "agents[1].nodes[4].type",
			wantMsg:  `got "merge"`,
		},
		{
			name:     "workflow field on an LlmAgent",
			old:      "    output_key: draft",
			new:      "    output_key: draft\n    edges: [{ from_node: START, to_node: x }]",
			wantPath: "agents[2]",
			wantMsg:  "nodes and edges apply only to a Workflow",
		},
		{
			name:     "tools on a Workflow",
			old:      "    agent_class: Workflow\n",
			new:      "    agent_class: Workflow\n    tools: [word_count]\n",
			wantPath: "agents[1]",
			wantMsg:  "apply only to an LlmAgent",
		},
		{
			name:     "LlmAgent lists an unregistered tool",
			old:      "    output_key: draft",
			new:      "    output_key: draft\n    tools: [sentiment]",
			wantPath: "agents[2].tools[0]",
			wantMsg:  `tool "sentiment" is not registered`,
		},
		{
			name:     "join as a parallel worker",
			old:      "{ id: gather, type: join }",
			new:      "{ id: gather, type: join, parallel_worker: true }",
			wantPath: "agents[1].nodes[4].parallel_worker",
			wantMsg:  "parallel_worker does not apply to join nodes",
		},
		{
			name:     "empty route",
			old:      "to_node: publish, route: DONE }",
			new:      `to_node: publish, route: "" }`,
			wantPath: "agents[1].edges[8].route",
			wantMsg:  `never emits route ""`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			doc := editedSystem(t, tc.old, tc.new)

			_, err := loadSystem(doc, testRegistry(t, &fakeModel{}))

			var verr *validationError
			if !errors.As(err, &verr) {
				t.Fatalf("loadSystem() error = %v, want a *validationError", err)
			}
			found := slices.ContainsFunc(verr.Issues, func(i issue) bool {
				return i.Path == tc.wantPath && strings.Contains(i.Message, tc.wantMsg)
			})
			if !found {
				t.Errorf("loadSystem() issues = %v, want one at %s containing %q", verr.Issues, tc.wantPath, tc.wantMsg)
			}
		})
	}
}

// TestLoadSystemReportsAllIssues checks that every problem is reported at
// once: with no models registered, each model reference is an issue.
func TestLoadSystemReportsAllIssues(t *testing.T) {
	_, err := loadSystem(agentSystem, registry{})

	var verr *validationError
	if !errors.As(err, &verr) {
		t.Fatalf("loadSystem() error = %v, want a *validationError", err)
	}
	// studio, writer, the three critics and the four llm_router llm nodes.
	if got, want := len(verr.Issues), 9; got != want {
		t.Errorf("loadSystem() reported %d issues, want %d:\n%v", got, want, err)
	}
}

func TestLoadSystemRejectsUnknownField(t *testing.T) {
	doc := editedSystem(t, "kind: AgentSystem", "kind: AgentSystem\nmax_concurrency: 3")

	_, err := loadSystem(doc, testRegistry(t, &fakeModel{}))

	if err == nil || !strings.Contains(err.Error(), "field max_concurrency not found") {
		t.Errorf("loadSystem() error = %v, want an unknown field error", err)
	}
}

// TestLoadSystemRejectsAgentCycle checks that an agent which reaches itself
// through its references is an error rather than endless recursion.
func TestLoadSystemRejectsAgentCycle(t *testing.T) {
	doc := editedSystem(t, "agent: writer }", "agent: studio }")

	_, err := loadSystem(doc, testRegistry(t, &fakeModel{}))

	if err == nil || !strings.Contains(err.Error(), `agent "studio" refers to itself`) {
		t.Errorf("loadSystem() error = %v, want a reference cycle error", err)
	}
}

// TestLoadSystemRejectsUnconditionalCycle checks a graph rule that only the
// workflow engine enforces: a loop-back edge must be routed.
func TestLoadSystemRejectsUnconditionalCycle(t *testing.T) {
	doc := editedSystem(t, "{ from_node: decide,          to_node: write,   route: REVISE }", "{ from_node: decide,          to_node: write }")

	_, err := loadSystem(doc, testRegistry(t, &fakeModel{}))

	if !errors.Is(err, workflow.ErrUnconditionalCycle) {
		t.Errorf("loadSystem() error = %v, want %v", err, workflow.ErrUnconditionalCycle)
	}
}

func TestLoadSystemRequiresAnAgent(t *testing.T) {
	_, err := loadSystem([]byte("apiVersion: adk/v1alpha1\nkind: AgentSystem\nagents: []\n"), registry{})

	var verr *validationError
	if !errors.As(err, &verr) {
		t.Fatalf("loadSystem() error = %v, want a *validationError", err)
	}
	want := []issue{{Path: "agents", Message: "at least one agent is required"}}
	if diff := cmp.Diff(want, verr.Issues); diff != "" {
		t.Errorf("loadSystem() issues mismatch (-want +got):\n%s", diff)
	}
}
