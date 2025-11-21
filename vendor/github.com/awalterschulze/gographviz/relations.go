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

// Relations represents the relations between graphs and nodes.
// Each node belongs the main graph or a subgraph.
type Relations struct {
	ParentToChildren map[string]map[string]bool
	ChildToParents   map[string]map[string]bool
}

// NewRelations creates an empty set of relations.
func NewRelations() *Relations {
	return &Relations{make(map[string]map[string]bool), make(map[string]map[string]bool)}
}

// Add adds a node to a parent graph.
func (relations *Relations) Add(parent string, child string) {
	if _, ok := relations.ParentToChildren[parent]; !ok {
		relations.ParentToChildren[parent] = make(map[string]bool)
	}
	relations.ParentToChildren[parent][child] = true
	if _, ok := relations.ChildToParents[child]; !ok {
		relations.ChildToParents[child] = make(map[string]bool)
	}
	relations.ChildToParents[child][parent] = true
}

// Remove removes relation
func (relations *Relations) Remove(parent string, child string) {
	if _, ok := relations.ParentToChildren[parent]; ok {
		delete(relations.ParentToChildren[parent], child)
	}

	if _, ok := relations.ChildToParents[child]; ok {
		delete(relations.ChildToParents[child], parent)
	}
}

// SortedChildren returns a list of sorted children of the given parent graph.
func (relations *Relations) SortedChildren(parent string) []string {
	keys := make([]string, 0)
	for key := range relations.ParentToChildren[parent] {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
