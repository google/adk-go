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

// SubGraph represents a Subgraph.
type SubGraph struct {
	Attrs Attrs
	Name  string
}

// NewSubGraph creates a new Subgraph.
func NewSubGraph(name string) *SubGraph {
	return &SubGraph{
		Attrs: make(Attrs),
		Name:  name,
	}
}

// SubGraphs represents a set of SubGraphs.
type SubGraphs struct {
	SubGraphs map[string]*SubGraph
}

// NewSubGraphs creates a new blank set of SubGraphs.
func NewSubGraphs() *SubGraphs {
	return &SubGraphs{make(map[string]*SubGraph)}
}

// Add adds and creates a new Subgraph to the set of SubGraphs.
func (subgraphs *SubGraphs) Add(name string) {
	if _, ok := subgraphs.SubGraphs[name]; !ok {
		subgraphs.SubGraphs[name] = NewSubGraph(name)
	}
}

// Remove removes a subgraph
func (subgraphs *SubGraphs) Remove(name string) {
	delete(subgraphs.SubGraphs, name)
}

// Sorted returns a sorted list of SubGraphs.
func (subgraphs *SubGraphs) Sorted() []*SubGraph {
	keys := make([]string, 0)
	for key := range subgraphs.SubGraphs {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	s := make([]*SubGraph, len(keys))
	for i, key := range keys {
		s[i] = subgraphs.SubGraphs[key]
	}
	return s
}
