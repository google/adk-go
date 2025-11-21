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
	"text/template"
	"unicode"
)

// Escape is just a Graph that escapes some strings when required.
type Escape struct {
	*Graph
}

// NewEscape returns a graph which will try to escape some strings when required
func NewEscape() *Escape {
	return &Escape{NewGraph()}
}

func isHTML(s string) bool {
	if len(s) == 0 {
		return false
	}
	ss := strings.TrimSpace(s)
	if ss[0] != '<' {
		return false
	}
	count := 0
	for _, c := range ss {
		if c == '<' {
			count++
		}
		if c == '>' {
			count--
		}
	}
	if count == 0 {
		return true
	}
	return false
}

func isLetter(ch rune) bool {
	return 'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' || ch == '_' ||
		ch >= 0x80 && unicode.IsLetter(ch) && ch != 'ε'
}

func isID(s string) bool {
	i := 0
	pos := false
	for _, c := range s {
		if i == 0 {
			if !isLetter(c) {
				return false
			}
			pos = true
		}
		if unicode.IsSpace(c) {
			return false
		}
		if c == '-' {
			return false
		}
		if c == '/' {
			return false
		}
		if c == '.' {
			return false
		}
		i++
	}
	return pos
}

func isDigit(ch rune) bool {
	return '0' <= ch && ch <= '9' || ch >= 0x80 && unicode.IsDigit(ch)
}

func isNumber(s string) bool {
	state := 0
	for _, c := range s {
		if state == 0 {
			if isDigit(c) || c == '.' {
				state = 2
			} else if c == '-' {
				state = 1
			} else {
				return false
			}
		} else if state == 1 {
			if isDigit(c) || c == '.' {
				state = 2
			}
		} else if c != '.' && !isDigit(c) {
			return false
		}
	}
	return (state == 2)
}

func isStringLit(s string) bool {
	if !strings.HasPrefix(s, `"`) || !strings.HasSuffix(s, `"`) {
		return false
	}
	var prev rune
	for _, r := range s[1 : len(s)-1] {
		if r == '"' && prev != '\\' {
			return false
		}
		prev = r
	}
	return true
}

func esc(s string) string {
	if len(s) == 0 {
		return s
	}
	if isHTML(s) {
		return s
	}
	ss := strings.TrimSpace(s)
	if ss[0] == '<' {
		return fmt.Sprintf("\"%s\"", strings.Replace(s, "\"", "\\\"", -1))
	}
	if isID(s) {
		return s
	}
	if isNumber(s) {
		return s
	}
	if isStringLit(s) {
		return s
	}
	return fmt.Sprintf("\"%s\"", template.HTMLEscapeString(s))
}

func escAttrs(attrs map[string]string) map[string]string {
	newAttrs := make(map[string]string)
	for k, v := range attrs {
		newAttrs[esc(k)] = esc(v)
	}
	return newAttrs
}

// SetName sets the graph name and escapes it, if needed.
func (escape *Escape) SetName(name string) error {
	return escape.Graph.SetName(esc(name))
}

// AddPortEdge adds an edge with ports and escapes the src, dst and attrs, if needed.
func (escape *Escape) AddPortEdge(src, srcPort, dst, dstPort string, directed bool, attrs map[string]string) error {
	return escape.Graph.AddPortEdge(esc(src), srcPort, esc(dst), dstPort, directed, escAttrs(attrs))
}

// AddEdge adds an edge and escapes the src, dst and attrs, if needed.
func (escape *Escape) AddEdge(src, dst string, directed bool, attrs map[string]string) error {
	return escape.AddPortEdge(src, "", dst, "", directed, attrs)
}

// AddNode adds a node and escapes the parentGraph, name and attrs, if needed.
func (escape *Escape) AddNode(parentGraph string, name string, attrs map[string]string) error {
	return escape.Graph.AddNode(esc(parentGraph), esc(name), escAttrs(attrs))
}

// AddAttr adds an attribute and escapes the parentGraph, field and value, if needed.
func (escape *Escape) AddAttr(parentGraph string, field, value string) error {
	return escape.Graph.AddAttr(esc(parentGraph), esc(field), esc(value))
}

// AddSubGraph adds a subgraph and escapes the parentGraph, name and attrs, if needed.
func (escape *Escape) AddSubGraph(parentGraph string, name string, attrs map[string]string) error {
	return escape.Graph.AddSubGraph(esc(parentGraph), esc(name), escAttrs(attrs))
}

// IsNode returns whether the, escaped if needed, name is a node in the graph.
func (escape *Escape) IsNode(name string) bool {
	return escape.Graph.IsNode(esc(name))
}

// IsSubGraph returns whether the, escaped if needed, name is a subgraph in the grahp.
func (escape *Escape) IsSubGraph(name string) bool {
	return escape.Graph.IsSubGraph(esc(name))
}
