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
	"sort"
)

// Node represents a Node.
type Node struct {
	Name  string
	Attrs Attrs
}

// Nodes represents a set of Nodes.
type Nodes struct {
	Lookup map[string]*Node
	Nodes  []*Node
}

// NewNodes creates a new set of Nodes.
func NewNodes() *Nodes {
	return &Nodes{make(map[string]*Node), make([]*Node, 0)}
}

// Remove removes a node
func (nodes *Nodes) Remove(name string) error {
	for i := 0; i < len(nodes.Nodes); i++ {
		if nodes.Nodes[i].Name != name {
			continue
		}

		nodes.Nodes = append(nodes.Nodes[:i], nodes.Nodes[i+1:]...)
		delete(nodes.Lookup, name)

		return nil
	}

	return fmt.Errorf("node %s not found", name)
}

// Add adds a Node to the set of Nodes, extending the attributes of an already existing node.
func (nodes *Nodes) Add(node *Node) {
	n, ok := nodes.Lookup[node.Name]
	if ok {
		n.Attrs.Extend(node.Attrs)
		return
	}
	nodes.Lookup[node.Name] = node
	nodes.Nodes = append(nodes.Nodes, node)
}

// Sorted returns a sorted list of nodes.
func (nodes Nodes) Sorted() []*Node {
	keys := make([]string, 0, len(nodes.Lookup))
	for key := range nodes.Lookup {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	nodeList := make([]*Node, len(keys))
	for i := range keys {
		nodeList[i] = nodes.Lookup[keys[i]]
	}
	return nodeList
}
