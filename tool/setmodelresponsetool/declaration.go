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

package setmodelresponsetool

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/url"
	"reflect"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
)

// description is adk-python's docstring of set_model_response, dedented.
const description = "Set your final response using the required output schema.\n\n" +
	"Use this tool to provide your final structured answer instead\n" +
	"of outputting text directly."

func (t *smrTool) Description() string { return description }

// Declaration declares a record's own schema as the parameters, and wraps
// every other schema in a single required parameter: "items" for a record
// list, "response" otherwise. A schema that holds a titled object below its
// root is declared as a JSON schema in which each such object appears once,
// under "$defs" at the root of the parameters; any other schema is declared
// as the *genai.Schema itself.
func (t *smrTool) Declaration() *genai.FunctionDeclaration {
	decl := &genai.FunctionDeclaration{Name: Name, Description: description}
	if t.shape == shapeRaw {
		decl.ParametersJsonSchema = utils.WrapJSONSchema("response", t.raw)
		return decl
	}
	if js, ok := t.jsonSchemaWithDefs(); ok {
		decl.ParametersJsonSchema = js
		return decl
	}
	switch t.shape {
	case shapeRecord:
		decl.ParametersJsonSchema = t.schema
	case shapeRecordList:
		decl.ParametersJsonSchema = wrapGenAI("items", t.schema)
	case shapeValue:
		decl.ParametersJsonSchema = wrapGenAI("response", t.schema)
	}
	return decl
}

func wrapGenAI(param string, s *genai.Schema) *genai.Schema {
	return &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{param: s},
		Required:   []string{param},
	}
}

// jsonSchemaWithDefs returns the parameters as a JSON schema with "$defs", and
// false when the schema holds no titled object below its root, has a default
// or example that is not JSON, or has a cycle that passes through no titled
// object (a JSON schema can only express a cycle as a reference). It builds a new tree on every call and
// keeps nothing, so concurrent calls share no state.
func (t *smrTool) jsonSchemaWithDefs() (*jsonschema.Schema, bool) {
	b := &defsBuilder{defs: map[string]*jsonschema.Schema{}, open: map[*genai.Schema]string{}}
	var params *jsonschema.Schema
	switch t.shape {
	case shapeRecord:
		params = b.body(t.schema)
		params.Title = t.schema.Title
		params.Description = t.schema.Description
	case shapeRecordList:
		params = wrapJSON("items", b.inline(t.schema))
	default:
		params = wrapJSON("response", b.inline(t.schema))
	}
	if b.err != nil || len(b.defs) == 0 {
		return nil, false
	}
	params.Defs = b.defs
	return params, true
}

func wrapJSON(param string, s *jsonschema.Schema) *jsonschema.Schema {
	return &jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{param: s},
		Required:   []string{param},
	}
}

// defsBuilder converts a *genai.Schema into a JSON schema whose titled objects
// are collected in defs, keyed by title.
type defsBuilder struct {
	defs map[string]*jsonschema.Schema
	// open holds each titled object whose def is being built, with the key
	// that a use inside that def reserved for it, or "" while none has.
	open map[*genai.Schema]string
	// inlined holds the schemas being inlined since the innermost def began.
	// Meeting one of them again is a cycle with no titled object to refer to.
	inlined map[*genai.Schema]bool
	// err is the first reason the schema cannot be written in this form: a
	// default or example that could not be marshaled, or errUntitledCycle.
	err error
}

var errUntitledCycle = errors.New("cycle through no titled object")

// use converts s where a property, an item or an anyOf arm uses it. A titled
// object becomes a reference to its def.
func (b *defsBuilder) use(s *genai.Schema) *jsonschema.Schema {
	if s == nil {
		return nil
	}
	if s.Title != "" && upperType(s) == genai.TypeObject {
		return b.decorate(s, &jsonschema.Schema{Ref: defRef(b.define(s))})
	}
	return b.inline(s)
}

// pointerEscaper escapes a JSON Pointer reference token (RFC 6901).
var pointerEscaper = strings.NewReplacer("~", "~0", "/", "~1")

// defRef returns the reference to the def with the given key: the key escaped
// as a JSON Pointer token, then as a URI fragment, so that any title resolves.
func defRef(key string) string {
	return "#" + (&url.URL{Fragment: "/$defs/" + pointerEscaper.Replace(key)}).EscapedFragment()
}

// inline converts s in place of its use, without a reference.
func (b *defsBuilder) inline(s *genai.Schema) *jsonschema.Schema {
	if b.inlined[s] {
		b.fail(errUntitledCycle)
		return &jsonschema.Schema{}
	}
	if b.inlined == nil {
		b.inlined = map[*genai.Schema]bool{}
	}
	b.inlined[s] = true
	defer delete(b.inlined, s)
	return b.decorate(s, b.body(s))
}

// decorate adds the fields that belong to the use of s rather than to its
// definition: nullable becomes anyOf [inner, null], and the description and
// default go on the outermost object.
func (b *defsBuilder) decorate(s *genai.Schema, inner *jsonschema.Schema) *jsonschema.Schema {
	out := inner
	if s.Nullable != nil && *s.Nullable {
		out = &jsonschema.Schema{AnyOf: []*jsonschema.Schema{inner, {Type: "null"}}}
	}
	out.Description = s.Description
	if s.Default != nil {
		out.Default = b.marshal(s.Default)
	}
	return out
}

// define adds the def of the titled object s and returns its key: the title,
// or the title with the first suffix _1, _2, ... whose key is free or already
// holds an equal def. A use of s inside its own def reserves the first free
// key at that point, and the def is stored under it.
func (b *defsBuilder) define(s *genai.Schema) string {
	if key, open := b.open[s]; open {
		if key == "" {
			key = b.freeKey(s.Title)
			b.defs[key] = &jsonschema.Schema{} // holds the key until the def is built
			b.open[s] = key
		}
		return key
	}
	b.open[s] = ""
	outer := b.inlined
	b.inlined = nil
	def := b.body(s)
	b.inlined = outer
	def.Title = s.Title
	reserved := b.open[s]
	delete(b.open, s)
	if reserved != "" {
		b.defs[reserved] = def
		return reserved
	}
	for i := 0; ; i++ {
		key := defKey(s.Title, i)
		existing, taken := b.defs[key]
		if !taken {
			b.defs[key] = def
			return key
		}
		if reflect.DeepEqual(existing, def) {
			return key
		}
	}
}

// freeKey returns the first key for title that holds no def.
func (b *defsBuilder) freeKey(title string) string {
	for i := 0; ; i++ {
		if key := defKey(title, i); b.defs[key] == nil {
			return key
		}
	}
}

// defKey returns title for i == 0, and title with the suffix _i otherwise.
func defKey(title string, i int) string {
	if i == 0 {
		return title
	}
	return fmt.Sprintf("%s_%d", title, i)
}

// body converts the fields that define s, leaving out its title and the
// fields of its use. Properties are visited in sorted key order, so the
// suffixes of defs that share a title are the same on every call.
func (b *defsBuilder) body(s *genai.Schema) *jsonschema.Schema {
	out := &jsonschema.Schema{
		Type:          strings.ToLower(string(s.Type)),
		Format:        s.Format,
		Pattern:       s.Pattern,
		Minimum:       cloneFloat(s.Minimum),
		Maximum:       cloneFloat(s.Maximum),
		MinLength:     toInt(s.MinLength),
		MaxLength:     toInt(s.MaxLength),
		MinItems:      toInt(s.MinItems),
		MaxItems:      toInt(s.MaxItems),
		MinProperties: toInt(s.MinProperties),
		MaxProperties: toInt(s.MaxProperties),
		Required:      slices.Clone(s.Required),
		PropertyOrder: slices.Clone(s.PropertyOrdering),
		Items:         b.use(s.Items),
	}
	if s.Type == genai.TypeUnspecified {
		out.Type = ""
	}
	for _, e := range s.Enum {
		out.Enum = append(out.Enum, e)
	}
	if s.Example != nil {
		out.Examples = []any{b.marshal(s.Example)}
	}
	for _, k := range slices.Sorted(maps.Keys(s.Properties)) {
		if out.Properties == nil {
			out.Properties = make(map[string]*jsonschema.Schema, len(s.Properties))
		}
		out.Properties[k] = b.use(s.Properties[k])
	}
	for _, arm := range s.AnyOf {
		out.AnyOf = append(out.AnyOf, b.use(arm))
	}
	return out
}

// marshal returns v as JSON, and records the error if v is not JSON.
func (b *defsBuilder) marshal(v any) json.RawMessage {
	raw, err := json.Marshal(v)
	if err != nil {
		b.fail(err)
	}
	return raw
}

// fail records err unless an earlier error is recorded.
func (b *defsBuilder) fail(err error) {
	if b.err == nil {
		b.err = err
	}
}

func cloneFloat(f *float64) *float64 {
	if f == nil {
		return nil
	}
	v := *f
	return &v
}

func toInt(n *int64) *int {
	if n == nil {
		return nil
	}
	v := int(*n)
	return &v
}
