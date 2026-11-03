// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package openapitoolset

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"

	"gopkg.in/yaml.v3"
)

// A spec is held as a tree of JSON values: nil, bool, int, int64, float64,
// string, []any and *object. Objects keep their key order because adk-python
// works on dicts, whose order decides which tools come first, which
// parameter gets a dedupe suffix and which security scheme is used.

// object is a JSON object that keeps the order in which its keys were first
// set.
type object struct {
	keys []string
	vals map[string]any
}

func newObject() *object {
	return &object{vals: map[string]any{}}
}

// get returns the value under k, or nil when there is none.
func (o *object) get(k string) any {
	return o.vals[k]
}

func (o *object) lookup(k string) (any, bool) {
	v, ok := o.vals[k]
	return v, ok
}

// set sets k to v. A key that is already present keeps its position, as in a
// Python dict.
func (o *object) set(k string, v any) {
	if _, ok := o.vals[k]; !ok {
		o.keys = append(o.keys, k)
	}
	o.vals[k] = v
}

func (o *object) delete(k string) {
	if _, ok := o.vals[k]; !ok {
		return
	}
	delete(o.vals, k)
	o.keys = slices.DeleteFunc(o.keys, func(key string) bool { return key == k })
}

func (o *object) len() int {
	return len(o.keys)
}

// cloneValue returns a deep copy of v.
func cloneValue(v any) any {
	switch v := v.(type) {
	case *object:
		out := &object{keys: slices.Clone(v.keys), vals: make(map[string]any, len(v.vals))}
		for k, val := range v.vals {
			out.vals[k] = cloneValue(val)
		}
		return out
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = cloneValue(item)
		}
		return out
	}
	return v
}

// plainSchema returns v with every object turned into a map[string]any and
// every null-valued object key dropped, which is how adk-python writes a
// schema into a function declaration (jsonable_encoder with exclude_none).
func plainSchema(v any) any {
	switch v := v.(type) {
	case *object:
		out := make(map[string]any, len(v.vals))
		for k, val := range v.vals {
			if val != nil {
				out[k] = plainSchema(val)
			}
		}
		return out
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = plainSchema(item)
		}
		return out
	}
	return v
}

// objectFromMap converts decoded JSON to an object. A Go map has no order, so
// keys come out sorted, as encoding/json writes them.
func objectFromMap(m map[string]any) (*object, error) {
	b, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	v, err := decodeJSON(b)
	if err != nil {
		return nil, err
	}
	o, ok := v.(*object)
	if !ok {
		// Only a nil map encodes as something other than an object.
		return newObject(), nil
	}
	return o, nil
}

// decodeDocument decodes JSON or YAML text. Text that starts with "{" is read
// as JSON first, since YAML differs from JSON in places such as tabs; text
// that is not valid JSON is read as YAML.
func decodeDocument(data []byte) (any, error) {
	if bytes.HasPrefix(bytes.TrimSpace(data), []byte("{")) {
		v, jsonErr := decodeJSON(data)
		if jsonErr == nil {
			return v, nil
		}
		if v, err := decodeYAML(data); err == nil {
			return v, nil
		}
		return nil, jsonErr
	}
	return decodeYAML(data)
}

// decodeJSON decodes one JSON value, keeping object key order. An integer
// becomes an int64 and any other number a float64, as Python's json.loads
// reads them as int and float.
func decodeJSON(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	v, err := decodeJSONValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("unexpected data after the top-level JSON value")
	}
	return v, nil
}

func decodeJSONValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch tok := tok.(type) {
	case json.Delim:
		switch tok {
		case '{':
			o := newObject()
			for dec.More() {
				keyTok, err := dec.Token()
				if err != nil {
					return nil, err
				}
				val, err := decodeJSONValue(dec)
				if err != nil {
					return nil, err
				}
				// The decoder only yields string keys inside an object.
				o.set(keyTok.(string), val)
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return o, nil
		case '[':
			arr := []any{}
			for dec.More() {
				val, err := decodeJSONValue(dec)
				if err != nil {
					return nil, err
				}
				arr = append(arr, val)
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return arr, nil
		}
		return nil, fmt.Errorf("unexpected JSON delimiter %q", tok)
	case json.Number:
		if i, err := tok.Int64(); err == nil {
			return i, nil
		}
		return tok.Float64()
	}
	return tok, nil
}

// decodeYAML decodes one YAML document, keeping mapping key order. Every
// mapping key becomes a string, so a response code written as 200 is the key
// "200".
func decodeYAML(data []byte) (any, error) {
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	if doc.Kind == 0 {
		return nil, nil // An empty document.
	}
	return fromYAMLNode(&doc)
}

func fromYAMLNode(n *yaml.Node) (any, error) {
	switch n.Kind {
	case yaml.DocumentNode:
		if len(n.Content) == 0 {
			return nil, nil
		}
		return fromYAMLNode(n.Content[0])
	case yaml.AliasNode:
		return fromYAMLNode(n.Alias)
	case yaml.SequenceNode:
		arr := make([]any, 0, len(n.Content))
		for _, c := range n.Content {
			v, err := fromYAMLNode(c)
			if err != nil {
				return nil, err
			}
			arr = append(arr, v)
		}
		return arr, nil
	case yaml.MappingNode:
		return fromYAMLMapping(n)
	case yaml.ScalarNode:
		switch n.ShortTag() {
		case "!!str", "!!timestamp", "!!binary":
			// A timestamp stays the text it was written as: it can only be an
			// example or a default, and the model needs the text.
			return n.Value, nil
		}
		var v any
		if err := n.Decode(&v); err != nil {
			return nil, err
		}
		return v, nil
	}
	return nil, fmt.Errorf("unsupported YAML node kind %v", n.Kind)
}

// fromYAMLMapping converts a mapping. Merge keys (<<) are applied as PyYAML
// applies them: the mapping's own keys win over merged ones, and an earlier
// merged mapping wins over a later one.
func fromYAMLMapping(n *yaml.Node) (any, error) {
	o := newObject()
	var merges []*yaml.Node
	type entry struct {
		key string
		val *yaml.Node
	}
	var own []entry
	for i := 0; i+1 < len(n.Content); i += 2 {
		k, v := n.Content[i], n.Content[i+1]
		if k.Kind != yaml.ScalarNode {
			return nil, errors.New("YAML mapping keys must be scalars")
		}
		if k.ShortTag() == "!!merge" {
			if v.Kind == yaml.SequenceNode {
				merges = append(merges, v.Content...)
			} else {
				merges = append(merges, v)
			}
			continue
		}
		own = append(own, entry{k.Value, v})
	}
	for _, m := range slices.Backward(merges) {
		mv, err := fromYAMLNode(m)
		if err != nil {
			return nil, err
		}
		mo, ok := mv.(*object)
		if !ok {
			return nil, errors.New("a YAML merge key must refer to a mapping")
		}
		for _, k := range mo.keys {
			o.set(k, mo.vals[k])
		}
	}
	for _, e := range own {
		v, err := fromYAMLNode(e.val)
		if err != nil {
			return nil, err
		}
		o.set(e.key, v)
	}
	return o, nil
}
