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

//Package gographviz provides parsing for the DOT grammar into
//an abstract syntax tree representing a graph,
//analysis of the abstract syntax tree into a more usable structure,
//and writing back of this structure into the DOT format.
package gographviz

import (
	"github.com/awalterschulze/gographviz/ast"
	"github.com/awalterschulze/gographviz/internal/parser"
)

var _ Interface = NewGraph()

//Interface allows you to parse the graph into your own structure.
type Interface interface {
	SetStrict(strict bool) error
	SetDir(directed bool) error
	SetName(name string) error
	AddPortEdge(src, srcPort, dst, dstPort string, directed bool, attrs map[string]string) error
	AddEdge(src, dst string, directed bool, attrs map[string]string) error
	AddNode(parentGraph string, name string, attrs map[string]string) error
	AddAttr(parentGraph string, field, value string) error
	AddSubGraph(parentGraph string, name string, attrs map[string]string) error
	String() string
}

//Parse parses the buffer into a abstract syntax tree representing the graph.
func Parse(buf []byte) (*ast.Graph, error) {
	return parser.ParseBytes(buf)
}

//ParseString parses the buffer into a abstract syntax tree representing the graph.
func ParseString(buf string) (*ast.Graph, error) {
	return parser.ParseBytes([]byte(buf))
}

//Read parses and creates a new Graph from the data.
func Read(buf []byte) (*Graph, error) {
	st, err := Parse(buf)
	if err != nil {
		return nil, err
	}
	return NewAnalysedGraph(st)
}
