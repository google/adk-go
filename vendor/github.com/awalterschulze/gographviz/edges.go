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
	"sort"
)

// Edge represents an Edge.
type Edge struct {
	Src     string
	SrcPort string
	Dst     string
	DstPort string
	Dir     bool
	Attrs   Attrs
}

// Edges represents a set of Edges.
type Edges struct {
	SrcToDsts map[string]map[string][]*Edge
	DstToSrcs map[string]map[string][]*Edge
	Edges     []*Edge
}

// NewEdges creates a blank set of Edges.
func NewEdges() *Edges {
	return &Edges{make(map[string]map[string][]*Edge), make(map[string]map[string][]*Edge), make([]*Edge, 0)}
}

// Add adds an Edge to the set of Edges.
func (edges *Edges) Add(edge *Edge) {
	if _, ok := edges.SrcToDsts[edge.Src]; !ok {
		edges.SrcToDsts[edge.Src] = make(map[string][]*Edge)
	}
	if _, ok := edges.SrcToDsts[edge.Src][edge.Dst]; !ok {
		edges.SrcToDsts[edge.Src][edge.Dst] = make([]*Edge, 0)
	}
	edges.SrcToDsts[edge.Src][edge.Dst] = append(edges.SrcToDsts[edge.Src][edge.Dst], edge)

	if _, ok := edges.DstToSrcs[edge.Dst]; !ok {
		edges.DstToSrcs[edge.Dst] = make(map[string][]*Edge)
	}
	if _, ok := edges.DstToSrcs[edge.Dst][edge.Src]; !ok {
		edges.DstToSrcs[edge.Dst][edge.Src] = make([]*Edge, 0)
	}
	edges.DstToSrcs[edge.Dst][edge.Src] = append(edges.DstToSrcs[edge.Dst][edge.Src], edge)

	edges.Edges = append(edges.Edges, edge)
}

// Sorted returns a sorted list of Edges.
func (edges Edges) Sorted() []*Edge {
	es := make(edgeSorter, len(edges.Edges))
	copy(es, edges.Edges)
	sort.Sort(es)
	return es
}

type edgeSorter []*Edge

func (es edgeSorter) Len() int      { return len(es) }
func (es edgeSorter) Swap(i, j int) { es[i], es[j] = es[j], es[i] }
func (es edgeSorter) Less(i, j int) bool {
	if es[i].Src < es[j].Src {
		return true
	} else if es[i].Src > es[j].Src {
		return false
	}

	if es[i].Dst < es[j].Dst {
		return true
	} else if es[i].Dst > es[j].Dst {
		return false
	}

	if es[i].SrcPort < es[j].SrcPort {
		return true
	} else if es[i].SrcPort > es[j].SrcPort {
		return false
	}

	if es[i].DstPort < es[j].DstPort {
		return true
	} else if es[i].DstPort > es[j].DstPort {
		return false
	}

	if es[i].Dir != es[j].Dir {
		return es[i].Dir
	}

	attrs := es[i].Attrs.Copy()
	for k, v := range es[j].Attrs {
		attrs[k] = v
	}

	for _, k := range attrs.sortedNames() {
		if es[i].Attrs[k] < es[j].Attrs[k] {
			return true
		} else if es[i].Attrs[k] > es[j].Attrs[k] {
			return false
		}
	}

	return false
}
