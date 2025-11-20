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

//A parser for the DOT grammar.
package parser

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"github.com/awalterschulze/gographviz/ast"
	"github.com/awalterschulze/gographviz/internal/lexer"
)

//Parses a DOT string and outputs the
//abstract syntax tree representing the graph.
func ParseString(dotString string) (*ast.Graph, error) {
	return ParseBytes([]byte(dotString))
}

//Parses the bytes representing a DOT string
//and outputs the abstract syntax tree representing the graph.
func ParseBytes(dotBytes []byte) (*ast.Graph, error) {
	lex := lexer.NewLexer(dotBytes)
	parser := NewParser()
	st, err := parser.Parse(lex)
	if err != nil {
		return nil, err
	}
	g, ok := st.(*ast.Graph)
	if !ok {
		panic(fmt.Sprintf("Parser did not return an *ast.Graph, but rather a %T", st))
	}
	return g, nil
}

//Parses a reader which contains a DOT string
//and outputs the abstract syntax tree representing the graph.
func Parse(r io.Reader) (*ast.Graph, error) {
	bytes, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return ParseBytes(bytes)
}

//Parses a file which contains a DOT string
//and outputs the abstract syntax tree representing the graph.
func ParseFile(filename string) (*ast.Graph, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	g, err := Parse(f)
	if err := f.Close(); err != nil {
		return nil, err
	}
	return g, err
}
