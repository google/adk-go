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
	"fmt"
	"sort"
)

// workflowGraph is the compiled, validated representation of a workflow's
// edge set. Internal to the package; the public WorkflowGraph alias is
// what users see.
type workflowGraph struct {
	// nodes maps node names to node values. START is always present.
	nodes map[string]Node

	// adjacency from-name -> list of edges. Order is preserved from the
	// caller's edge slice so behavior is deterministic.
	out map[string][]edgeRef

	// reverse adjacency for predecessor lookups.
	in map[string][]edgeRef

	// startNames is the set of node names directly reachable from START.
	startNames []string
}

// edgeRef is the indexed form of an Edge stored on the graph. Routes alias
// the original Edge.Routes slice — never mutate.
type edgeRef struct {
	to     string
	from   string
	routes []Route
}

// buildGraph validates and indexes the edge slice.
func buildGraph(edges []Edge) (*workflowGraph, error) {
	g := &workflowGraph{
		nodes: map[string]Node{"__START__": START},
		out:   map[string][]edgeRef{},
		in:    map[string][]edgeRef{},
	}

	if len(edges) == 0 {
		return nil, fmt.Errorf("workflow: at least one edge is required")
	}

	// First pass: register every node, ensuring names are unique per node
	// instance. (Two distinct node instances can't share a name.)
	register := func(n Node) error {
		if n == nil {
			return fmt.Errorf("workflow: edge has nil node")
		}
		if _, ok := g.nodes[n.Name()]; ok && g.nodes[n.Name()] != n {
			return fmt.Errorf("workflow: duplicate node name %q", n.Name())
		}
		g.nodes[n.Name()] = n
		return nil
	}
	for _, e := range edges {
		if err := register(e.From); err != nil {
			return nil, err
		}
		if err := register(e.To); err != nil {
			return nil, err
		}
	}

	// Second pass: index adjacency and start edges.
	for _, e := range edges {
		ref := edgeRef{from: e.From.Name(), to: e.To.Name(), routes: e.Routes}
		g.out[ref.from] = append(g.out[ref.from], ref)
		g.in[ref.to] = append(g.in[ref.to], ref)
		if e.From == START {
			g.startNames = append(g.startNames, e.To.Name())
		}
	}

	if len(g.startNames) == 0 {
		return nil, fmt.Errorf("workflow: at least one edge must originate from START")
	}

	// Reachability: every non-START node must be reachable from START
	// following directed edges.
	reachable := map[string]bool{"__START__": true}
	stack := append([]string(nil), g.startNames...)
	for _, n := range g.startNames {
		reachable[n] = true
	}
	for len(stack) > 0 {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, e := range g.out[cur] {
			if !reachable[e.to] {
				reachable[e.to] = true
				stack = append(stack, e.to)
			}
		}
	}
	for name := range g.nodes {
		if !reachable[name] {
			return nil, fmt.Errorf("workflow: node %q is unreachable from START", name)
		}
	}

	return g, nil
}

func (g *workflowGraph) has(name string) bool {
	_, ok := g.nodes[name]
	return ok
}

func (g *workflowGraph) successors(name string) []string {
	out := make([]string, 0, len(g.out[name]))
	for _, e := range g.out[name] {
		out = append(out, e.to)
	}
	return out
}

func (g *workflowGraph) predecessors(name string) []string {
	out := make([]string, 0, len(g.in[name]))
	for _, e := range g.in[name] {
		out = append(out, e.from)
	}
	return out
}

func (g *workflowGraph) nodeNames() []string {
	out := make([]string, 0, len(g.nodes))
	for k := range g.nodes {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
