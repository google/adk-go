// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains functions that infer a schema from a Go type.

package jsonschema

import (
	"fmt"
	"reflect"

	"github.com/google/adk-go/internal/util"
)

// For constructs a JSON schema object for the given type argument.
//
// It is a convenience for ForType.
func For[T any]() (*Schema, error) {
	return ForType(reflect.TypeFor[T]())
}

// ForType constructs a JSON schema object for the given type.
// It translates Go types into compatible JSON schema types, as follows:
//   - strings have schema type "string"
//   - bools have schema type "boolean"
//   - signed and unsigned integer types have schema type "integer"
//   - floating point types have schema type "number"
//   - slices and arrays have schema type "array", and a corresponding schema
//     for items
//   - maps with string key have schema type "object", and corresponding
//     schema for additionalProperties
//   - structs have schema type "object", and disallow additionalProperties.
//     Their properties are derived from exported struct fields, using the
//     struct field json name. Fields that are marked "omitempty" are
//     considered optional; all other fields become required properties.
//
// It returns an error if t contains (possibly recursively) any of the following Go
// types, as they are incompatible with the JSON schema spec.
//   - maps with key other than 'string'
//   - function types
//   - complex numbers
//   - unsafe pointers
//
// The cannot be any cycles in the types.
// TODO(rfindley): we could perhaps just skip these incompatible fields.
func ForType(t reflect.Type) (*Schema, error) {
	return typeSchema(t)
}

func typeSchema(t reflect.Type) (*Schema, error) {
	// Follow pointers: the schema for *T is almost the same as for T, except that
	// an explicit JSON "null" is allowed for the pointer.
	allowNull := false
	for t.Kind() == reflect.Pointer {
		allowNull = true
		t = t.Elem()
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
			return nil, fmt.Errorf("unsupported map key type %v", t.Key().Kind())
		}
		s.Type = "object"
		s.AdditionalProperties, err = typeSchema(t.Elem())
		if err != nil {
			return nil, fmt.Errorf("computing map value schema: %v", err)
		}

	case reflect.Slice, reflect.Array:
		s.Type = "array"
		s.Items, err = typeSchema(t.Elem())
		if err != nil {
			return nil, fmt.Errorf("computing element schema: %v", err)
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

		for i := range t.NumField() {
			field := t.Field(i)
			info := util.FieldJSONInfo(field)
			if info.Omit {
				continue
			}
			if s.Properties == nil {
				s.Properties = make(map[string]*Schema)
			}
			s.Properties[info.Name], err = typeSchema(field.Type)
			if err != nil {
				return nil, err
			}
			if !info.Settings["omitempty"] && !info.Settings["omitzero"] {
				s.Required = append(s.Required, info.Name)
			}
		}

	default:
		return nil, fmt.Errorf("type %v is unsupported by jsonschema", t)
	}
	if allowNull && s.Type != "" {
		s.Types = []string{"null", s.Type}
		s.Type = ""
	}
	return s, nil
}
