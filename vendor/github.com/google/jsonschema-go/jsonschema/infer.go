// Copyright 2025 The JSON Schema Go Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

// This file contains functions that infer a schema from a Go type.

package jsonschema

import (
	"fmt"
	"log/slog"
	"maps"
	"math/big"
	"reflect"
	"regexp"
	"time"
)

// ForOptions are options for the [For] and [ForType] functions.
type ForOptions struct {
	// If IgnoreInvalidTypes is true, fields that can't be represented as a JSON
	// Schema are ignored instead of causing an error.
	// This allows callers to adjust the resulting schema using custom knowledge.
	// For example, an interface type where all the possible implementations are
	// known can be described with "oneof".
	IgnoreInvalidTypes bool

	// TypeSchemas maps types to their schemas.
	// If [For] encounters a type that is a key in this map, the
	// corresponding value is used as the resulting schema (after cloning to
	// ensure uniqueness).
	// Types in this map override the default translations, as described
	// in [For]'s documentation.
	TypeSchemas map[reflect.Type]*Schema
}

// For constructs a JSON schema object for the given type argument.
// If non-nil, the provided options configure certain aspects of this contruction,
// described below.

// It translates Go types into compatible JSON schema types, as follows.
// These defaults can be overridden by [ForOptions.TypeSchemas].
//
//   - Strings have schema type "string".
//   - Bools have schema type "boolean".
//   - Signed and unsigned integer types have schema type "integer".
//   - Floating point types have schema type "number".
//   - Slices and arrays have schema type "array", and a corresponding schema
//     for items.
//   - Maps with string key have schema type "object", and corresponding
//     schema for additionalProperties.
//   - Structs have schema type "object", and disallow additionalProperties.
//     Their properties are derived from exported struct fields, using the
//     struct field JSON name. Fields that are marked "omitempty" are
//     considered optional; all other fields become required properties.
//   - Some types in the standard library that implement json.Marshaler
//     translate to schemas that match the values to which they marshal.
//     For example, [time.Time] translates to the schema for strings.
//
// For will return an error if there is a cycle in the types.
//
// By default, For returns an error if t contains (possibly recursively) any of the
// following Go types, as they are incompatible with the JSON schema spec.
// If [ForOptions.IgnoreInvalidTypes] is true, then these types are ignored instead.
//   - maps with key other than 'string'
//   - function types
//   - channel types
//   - complex numbers
//   - unsafe pointers
//
// This function recognizes struct field tags named "jsonschema".
// A jsonschema tag on a field is used as the description for the corresponding property.
// For future compatibility, descriptions must not start with "WORD=", where WORD is a
// sequence of non-whitespace characters.
func For[T any](opts *ForOptions) (*Schema, error) {
	if opts == nil {
		opts = &ForOptions{}
	}
	schemas := maps.Clone(initialSchemaMap)
	// Add types from the options. They override the default ones.
	maps.Copy(schemas, opts.TypeSchemas)
	s, err := forType(reflect.TypeFor[T](), map[reflect.Type]bool{}, opts.IgnoreInvalidTypes, schemas)
	if err != nil {
		var z T
		return nil, fmt.Errorf("For[%T](): %w", z, err)
	}
	return s, nil
}

// ForType is like [For], but takes a [reflect.Type]
func ForType(t reflect.Type, opts *ForOptions) (*Schema, error) {
	schemas := maps.Clone(initialSchemaMap)
	// Add types from the options. They override the default ones.
	maps.Copy(schemas, opts.TypeSchemas)
	s, err := forType(t, map[reflect.Type]bool{}, opts.IgnoreInvalidTypes, schemas)
	if err != nil {
		return nil, fmt.Errorf("ForType(%s): %w", t, err)
	}
	return s, nil
}

func forType(t reflect.Type, seen map[reflect.Type]bool, ignore bool, schemas map[reflect.Type]*Schema) (*Schema, error) {
	// Follow pointers: the schema for *T is almost the same as for T, except that
	// an explicit JSON "null" is allowed for the pointer.
	allowNull := false
	for t.Kind() == reflect.Pointer {
		allowNull = true
		t = t.Elem()
	}

	// Check for cycles
	// User defined types have a name, so we can skip those that are natively defined
	if t.Name() != "" {
		if seen[t] {
			return nil, fmt.Errorf("cycle detected for type %v", t)
		}
		seen[t] = true
		defer delete(seen, t)
	}

	if s := schemas[t]; s != nil {
		return s.CloneSchemas(), nil
	}

	var (
		s   = new(Schema)
		err error
	)

	switch t.Kind() {
	case reflect.Bool:
		s.Type = "boolean"

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Uintptr:
		s.Type = "integer"

	case reflect.Float32, reflect.Float64:
		s.Type = "number"

	case reflect.Interface:
		// Unrestricted

	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			if ignore {
				return nil, nil // ignore
			}
			return nil, fmt.Errorf("unsupported map key type %v", t.Key().Kind())
		}
		if t.Key().Kind() != reflect.String {
		}
		s.Type = "object"
		s.AdditionalProperties, err = forType(t.Elem(), seen, ignore, schemas)
		if err != nil {
			return nil, fmt.Errorf("computing map value schema: %v", err)
		}
		if ignore && s.AdditionalProperties == nil {
			// Ignore if the element type is invalid.
			return nil, nil
		}

	case reflect.Slice, reflect.Array:
		s.Type = "array"
		s.Items, err = forType(t.Elem(), seen, ignore, schemas)
		if err != nil {
			return nil, fmt.Errorf("computing element schema: %v", err)
		}
		if ignore && s.Items == nil {
			// Ignore if the element type is invalid.
			return nil, nil
		}
		if t.Kind() == reflect.Array {
			s.MinItems = Ptr(t.Len())
			s.MaxItems = Ptr(t.Len())
		}

	case reflect.String:
		s.Type = "string"

	case reflect.Struct:
		s.Type = "object"
		// no additional properties are allowed
		s.AdditionalProperties = falseSchema()
		for _, field := range reflect.VisibleFields(t) {
			if field.Anonymous {
				continue
			}

			info := fieldJSONInfo(field)
			if info.omit {
				continue
			}
			if s.Properties == nil {
				s.Properties = make(map[string]*Schema)
			}
			fs, err := forType(field.Type, seen, ignore, schemas)
			if err != nil {
				return nil, err
			}
			if ignore && fs == nil {
				// Skip fields of invalid type.
				continue
			}
			if tag, ok := field.Tag.Lookup("jsonschema"); ok {
				if tag == "" {
					return nil, fmt.Errorf("empty jsonschema tag on struct field %s.%s", t, field.Name)
				}
				if disallowedPrefixRegexp.MatchString(tag) {
					return nil, fmt.Errorf("tag must not begin with 'WORD=': %q", tag)
				}
				fs.Description = tag
			}
			s.Properties[info.name] = fs
			if !info.settings["omitempty"] && !info.settings["omitzero"] {
				s.Required = append(s.Required, info.name)
			}
		}

	default:
		if ignore {
			// Ignore.
			return nil, nil
		}
		return nil, fmt.Errorf("type %v is unsupported by jsonschema", t)
	}
	if allowNull && s.Type != "" {
		s.Types = []string{"null", s.Type}
		s.Type = ""
	}
	return s, nil
}

// initialSchemaMap holds types from the standard library that have MarshalJSON methods.
var initialSchemaMap = make(map[reflect.Type]*Schema)

func init() {
	ss := &Schema{Type: "string"}
	initialSchemaMap[reflect.TypeFor[time.Time]()] = ss
	initialSchemaMap[reflect.TypeFor[slog.Level]()] = ss
	initialSchemaMap[reflect.TypeFor[big.Int]()] = &Schema{Types: []string{"null", "string"}}
	initialSchemaMap[reflect.TypeFor[big.Rat]()] = ss
	initialSchemaMap[reflect.TypeFor[big.Float]()] = ss
}

// Disallow jsonschema tag values beginning "WORD=", for future expansion.
var disallowedPrefixRegexp = regexp.MustCompile("^[^ \t\n]*=")
