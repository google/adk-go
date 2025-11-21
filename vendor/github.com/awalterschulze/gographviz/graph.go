//Copyright 2013 GoGraphviz Authors
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

package gographviz

import (
	"fmt"
	"strings"
)

// Graph is the analysed representation of the Graph parsed from the DOT format.
type Graph struct {
	Attrs     Attrs
	Name      string
	Directed  bool
	Strict    bool
	Nodes     *Nodes
	Edges     *Edges
	SubGraphs *SubGraphs
	Relations *Relations
}

// NewGraph creates a new empty graph, ready to be populated.
func NewGraph() *Graph {
	return &Graph{
		Attrs:     make(Attrs),
		Name:      "",
		Directed:  false,
		Strict:    false,
		Nodes:     NewNodes(),
		Edges:     NewEdges(),
		SubGraphs: NewSubGraphs(),
		Relations: NewRelations(),
	}
}

// SetStrict sets whether a graph is strict.
// If the graph is strict then multiple edges are not allowed between the same pairs of nodes,
// see dot man page.
func (g *Graph) SetStrict(strict bool) error {
	g.Strict = strict
	return nil
}

// SetDir sets whether the graph is directed (true) or undirected (false).
func (g *Graph) SetDir(dir bool) error {
	g.Directed = dir
	return nil
}

// SetName sets the graph name.
func (g *Graph) SetName(name string) error {
	g.Name = name
	return nil
}

// AddPortEdge adds an edge to the graph from node src to node dst.
// srcPort and dstPort are the port the node ports, leave as empty strings if it is not required.
// This does not imply the adding of missing nodes.
func (g *Graph) AddPortEdge(src, srcPort, dst, dstPort string, directed bool, attrs map[string]string) error {
	as, err := NewAttrs(attrs)
	if err != nil {
		return err
	}
	g.Edges.Add(&Edge{src, srcPort, dst, dstPort, directed, as})
	return nil
}

// AddEdge adds an edge to the graph from node src to node dst.
// This does not imply the adding of missing nodes.
// If directed is set to true then SetDir(true) must also be called or there will be a syntax error in the output.
func (g *Graph) AddEdge(src, dst string, directed bool, attrs map[string]string) error {
	return g.AddPortEdge(src, "", dst, "", directed, attrs)
}

// AddNode adds a node to a graph/subgraph.
// If not subgraph exists use the name of the main graph.
// This does not imply the adding of a missing subgraph.
func (g *Graph) AddNode(parentGraph string, name string, attrs map[string]string) error {
	as, err := NewAttrs(attrs)
	if err != nil {
		return err
	}
	g.Nodes.Add(&Node{name, as})
	g.Relations.Add(parentGraph, name)
	return nil
}

// RemoveNode removes a node from the graph
func (g *Graph) RemoveNode(parentGraph string, name string) error {
	err := g.Nodes.Remove(name)
	if err != nil {
		return err
	}

	g.Relations.Remove(parentGraph, name)

	edges := NewEdges()
	for _, e := range g.Edges.Edges {
		if e.Dst == name || e.Src == name {
			continue
		}

		edges.Add(e)
	}

	g.Edges = edges

	return nil
}

func (g *Graph) getAttrs(graphName string) (Attrs, error) {
	if g.Name == graphName {
		return g.Attrs, nil
	}
	sub, ok := g.SubGraphs.SubGraphs[graphName]
	if !ok {
		return nil, fmt.Errorf("graph or subgraph %s does not exist", graphName)
	}
	return sub.Attrs, nil
}

// AddAttr adds an attribute to a graph/subgraph.
func (g *Graph) AddAttr(parentGraph string, field string, value string) error {
	a, err := g.getAttrs(parentGraph)
	if err != nil {
		return err
	}
	return a.Add(field, value)
}

// AddSubGraph adds a subgraph to a graph/subgraph.
func (g *Graph) AddSubGraph(parentGraph string, name string, attrs map[string]string) error {
	g.Relations.Add(parentGraph, name)
	g.SubGraphs.Add(name)
	for key, value := range attrs {
		if err := g.AddAttr(name, key, value); err != nil {
			return err
		}
	}
	return nil
}

// RemoveSubGraph removes the subgraph including nodes
func (g *Graph) RemoveSubGraph(parentGraph string, name string) error {
	for child := range g.Relations.ParentToChildren[name] {
		err := g.RemoveNode(parentGraph, child)
		if err != nil {
			return err
		}
	}

	g.Relations.Remove(parentGraph, name)
	g.SubGraphs.Remove(name)

	edges := NewEdges()
	for _, e := range g.Edges.Edges {
		if e.Dst == name || e.DstPort == name || e.Src == name || e.SrcPort == name {
			continue
		}

		edges.Add(e)
	}

	g.Edges = edges

	return nil
}

// IsNode returns whether a given node name exists as a node in the graph.
func (g *Graph) IsNode(name string) bool {
	_, ok := g.Nodes.Lookup[name]
	return ok
}

// IsSubGraph returns whether a given subgraph name exists as a subgraph in the graph.
func (g *Graph) IsSubGraph(name string) bool {
	_, ok := g.SubGraphs.SubGraphs[name]
	return ok
}

func (g *Graph) isClusterSubGraph(name string) bool {
	isSubGraph := g.IsSubGraph(name)
	isCluster := strings.HasPrefix(name, "cluster")
	return isSubGraph && isCluster
}
