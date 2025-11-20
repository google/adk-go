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

// Attrs represents attributes for an Edge, Node or Graph.
type Attrs map[Attr]string

// NewAttrs creates an empty Attributes type.
func NewAttrs(m map[string]string) (Attrs, error) {
	as := make(Attrs)
	for k, v := range m {
		if err := as.Add(k, v); err != nil {
			return nil, err
		}
	}
	return as, nil
}

// Add adds an attribute name and value.
func (attrs Attrs) Add(field string, value string) error {
	a, err := NewAttr(field)
	if err != nil {
		return err
	}
	attrs.add(a, value)
	return nil
}

func (attrs Attrs) add(field Attr, value string) {
	attrs[field] = value
}

// Extend adds the attributes into attrs Attrs type overwriting duplicates.
func (attrs Attrs) Extend(more Attrs) {
	for key, value := range more {
		attrs.add(key, value)
	}
}

// Ammend only adds the missing attributes to attrs Attrs type.
func (attrs Attrs) Ammend(more Attrs) {
	for key, value := range more {
		if _, ok := attrs[key]; !ok {
			attrs.add(key, value)
		}
	}
}

func (attrs Attrs) toMap() map[string]string {
	m := make(map[string]string)
	for k, v := range attrs {
		m[string(k)] = v
	}
	return m
}

type attrList []Attr

func (attrs attrList) Len() int { return len(attrs) }
func (attrs attrList) Less(i, j int) bool {
	return attrs[i] < attrs[j]
}
func (attrs attrList) Swap(i, j int) {
	attrs[i], attrs[j] = attrs[j], attrs[i]
}

func (attrs Attrs) sortedNames() []Attr {
	keys := make(attrList, 0)
	for key := range attrs {
		keys = append(keys, key)
	}
	sort.Sort(keys)
	return []Attr(keys)
}

// Copy returns a copy of the attributes map
func (attrs Attrs) Copy() Attrs {
	mm := make(Attrs)
	for k, v := range attrs {
		mm[k] = v
	}
	return mm
}
