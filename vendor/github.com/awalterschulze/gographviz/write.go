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

	"github.com/awalterschulze/gographviz/ast"
)

type writer struct {
	*Graph
	writtenLocations map[string]bool
}

func newWriter(g *Graph) *writer {
	return &writer{g, make(map[string]bool)}
}

func appendAttrs(list ast.StmtList, attrs Attrs) ast.StmtList {
	for _, name := range attrs.sortedNames() {
		stmt := &ast.Attr{
			Field: ast.ID(name),
			Value: ast.ID(attrs[name]),
		}
		list = append(list, stmt)
	}
	return list
}

func (w *writer) newSubGraph(name string) (*ast.SubGraph, error) {
	sub := w.SubGraphs.SubGraphs[name]
	w.writtenLocations[sub.Name] = true
	s := &ast.SubGraph{}
	s.ID = ast.ID(sub.Name)
	s.StmtList = appendAttrs(s.StmtList, sub.Attrs)
	children := w.Relations.SortedChildren(name)
	for _, child := range children {
		if w.IsNode(child) {
			s.StmtList = append(s.StmtList, w.newNodeStmt(child))
		} else if w.IsSubGraph(child) {
			subgraph, err := w.newSubGraph(child)
			if err != nil {
				return nil, err
			}
			s.StmtList = append(s.StmtList, subgraph)
		} else {
			return nil, fmt.Errorf("%v is not a node or a subgraph", child)
		}
	}
	return s, nil
}

func (w *writer) newNodeID(name string, port string) *ast.NodeID {
	node := w.Nodes.Lookup[name]
	return ast.MakeNodeID(node.Name, port)
}

func (w *writer) newNodeStmt(name string) *ast.NodeStmt {
	node := w.Nodes.Lookup[name]
	id := ast.MakeNodeID(node.Name, "")
	w.writtenLocations[node.Name] = true
	return &ast.NodeStmt{
		NodeID: id,
		Attrs:  ast.PutMap(node.Attrs.toMap()),
	}
}

func (w *writer) newLocation(name string, port string) (ast.Location, error) {
	if w.IsNode(name) {
		return w.newNodeID(name, port), nil
	} else if w.isClusterSubGraph(name) {
		if len(port) != 0 {
			return nil, fmt.Errorf("subgraph cannot have a port: %v", port)
		}
		return ast.MakeNodeID(name, port), nil
	} else if w.IsSubGraph(name) {
		if len(port) != 0 {
			return nil, fmt.Errorf("subgraph cannot have a port: %v", port)
		}
		return w.newSubGraph(name)
	}
	return nil, fmt.Errorf("%v is not a node or a subgraph", name)
}

func (w *writer) newEdgeStmt(edge *Edge) (*ast.EdgeStmt, error) {
	src, err := w.newLocation(edge.Src, edge.SrcPort)
	if err != nil {
		return nil, err
	}
	dst, err := w.newLocation(edge.Dst, edge.DstPort)
	if err != nil {
		return nil, err
	}
	stmt := &ast.EdgeStmt{
		Source: src,
		EdgeRHS: ast.EdgeRHS{
			&ast.EdgeRH{
				Op:          ast.EdgeOp(edge.Dir),
				Destination: dst,
			},
		},
		Attrs: ast.PutMap(edge.Attrs.toMap()),
	}
	return stmt, nil
}

func (w *writer) Write() (*ast.Graph, error) {
	t := &ast.Graph{}
	t.Strict = w.Strict
	t.Type = ast.GraphType(w.Directed)
	t.ID = ast.ID(w.Name)

	t.StmtList = appendAttrs(t.StmtList, w.Attrs)

	for _, edge := range w.Edges.Edges {
		e, err := w.newEdgeStmt(edge)
		if err != nil {
			return nil, err
		}
		t.StmtList = append(t.StmtList, e)
	}

	subGraphs := w.SubGraphs.Sorted()
	for _, s := range subGraphs {
		if _, ok := w.writtenLocations[s.Name]; !ok {
			if _, ok := w.Relations.ParentToChildren[w.Name][s.Name]; ok {
				s, err := w.newSubGraph(s.Name)
				if err != nil {
					return nil, err
				}
				t.StmtList = append(t.StmtList, s)
			}
		}
	}

	nodes := w.Nodes.Sorted()
	for _, n := range nodes {
		if _, ok := w.writtenLocations[n.Name]; !ok {
			t.StmtList = append(t.StmtList, w.newNodeStmt(n.Name))
		}
	}

	return t, nil
}

// WriteAst creates an Abstract Syntrax Tree from the Graph.
func (g *Graph) WriteAst() (*ast.Graph, error) {
	w := newWriter(g)
	return w.Write()
}

// String returns a DOT string representing the Graph.
func (g *Graph) String() string {
	w, err := g.WriteAst()
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	return w.String()
}
