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

package helper

import (
	"encoding"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// Bound recursion even for acyclic output. The path grows on each step.
const maxSnakeDepth = 256

var (
	errSnakeCycle = errors.New("cyclic value in snake-case conversion")
	errSnakeDepth = errors.New("value too deeply nested for snake-case conversion")
)

type snakeVisit struct {
	typ reflect.Type
	ptr uintptr
	len int
}

// convertSnake converts JSON field names to snake case and preserves known
// Agent Engine wire types such as Unix timestamps.
func convertSnake(path string, o any) (any, error) {
	return convertSnakeValue(path, reflect.ValueOf(o), 0, make(map[snakeVisit]struct{}))
}

// Work with reflect.Values so unexported embedded structs can expose their
// exported fields without calling Interface on the struct itself.
func convertSnakeValue(path string, v reflect.Value, depth int, active map[snakeVisit]struct{}) (any, error) {
	if depth > maxSnakeDepth {
		return nil, errSnakeDepth
	}
	if !v.IsValid() {
		return nil, nil
	}
	// An interface adds no JSON nesting, so it costs no depth step. The value
	// inside it is never addressable, so unwrapping it here loses nothing.
	if v.Kind() == reflect.Interface {
		if v.IsNil() {
			return nil, nil
		}
		v = v.Elem()
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if !v.IsNil() {
			visit := snakeVisit{typ: v.Type(), ptr: v.Pointer()}
			if v.Kind() == reflect.Slice {
				visit.len = v.Len()
			}
			if _, ok := active[visit]; ok {
				return nil, errSnakeCycle
			}
			// The same value can appear in sibling fields without forming a cycle.
			active[visit] = struct{}{}
			defer delete(active, visit)
		}
	}
	switch v.Kind() {
	case reflect.String:
		if v.CanInterface() {
			return v.Interface(), nil
		}
		return v.String(), nil
	case reflect.Struct:
		vt := v.Type()
		if vt == reflect.TypeFor[time.Time]() {
			if !v.CanInterface() {
				return nil, fmt.Errorf("cannot convert unexported time field")
			}
			t := v.Interface().(time.Time)
			return float64(t.UnixMicro()) / 1e6, nil // returns a number of seconds "Unix-way"
		}
		if !hasWalkableFields(vt) {
			if val, ok, err := marshalOpaque(v); ok {
				return val, err
			}
		}

		m := make(map[string]any)
		for i := range v.NumField() {
			fv := v.Field(i)
			fvt := vt.Field(i)
			if !fvt.IsExported() && (!fvt.Anonymous || reflect.Indirect(fv).Kind() != reflect.Struct) {
				// Unexported embedded structs can promote exported fields.
				continue
			}
			tag := fvt.Tag.Get("json")
			name, omitEmpty, _, skip, err := fieldName(fvt.Name, tag)
			if err != nil {
				return nil, fmt.Errorf("failed to parse tag (%v): %w", tag, err)
			}
			if skip {
				continue
			}
			newPath := path + "." + name
			if fvt.Anonymous {
				embed, err := convertSnakeValue(newPath, fv, depth+1, active)
				if err != nil {
					return nil, fmt.Errorf("failed to convert embedded field: %w", err)
				}
				if fields, ok := embed.(map[string]any); ok {
					maps.Copy(m, fields)
				} else if embed != nil {
					// Scalar embeddings such as time.Time cannot be flattened.
					m[convertName(newPath, name)] = embed
				}
				continue
			}

			// Pass the field as is: it stays addressable when its parent is, so
			// a marshaler with a pointer receiver (big.Int) is found at any
			// depth, as encoding/json does.
			val, err := convertSnakeValue(newPath, fv, depth+1, active)
			if err != nil {
				return nil, fmt.Errorf("failed to convert regular struct field with path: %v err: %w", newPath, err)
			}
			// encoding/json never omits a struct for omitempty, and omits a
			// pointer or interface only when nil, which already yields a nil
			// val. So a marshaled netip.Addr{}, or a non-nil pointer to one,
			// stays as "".
			if k := fvt.Type.Kind(); k == reflect.Pointer || k == reflect.Interface || isOpaque(fvt.Type) {
				omitEmpty = false
			}
			if val != nil && (!omitEmpty || notEmpty(val)) {
				m[convertName(newPath, name)] = val
			}
		}
		return m, nil
	case reflect.Array, reflect.Slice:
		if v.Kind() == reflect.Array {
			if val, ok, err := marshalOpaque(v); ok {
				return val, err
			}
		}
		if v.Type() == reflect.TypeFor[json.RawMessage]() {
			rm := json.RawMessage(v.Bytes())
			if len(rm) == 0 {
				return nil, nil
			}
			return rm, nil
		}
		if v.Kind() == reflect.Slice && v.Type().Elem().Kind() == reflect.Uint8 {
			return base64.StdEncoding.EncodeToString(v.Bytes()), nil
		}
		res := []any{}
		for i := range v.Len() {
			elem, err := convertSnakeValue(path+".[]", v.Index(i), depth+1, active)
			if err != nil {
				return nil, fmt.Errorf("failed to convert slice element with path: %v err: %w", path+".[]", err)
			}
			res = append(res, elem)
		}
		if len(res) == 0 {
			return []any{}, nil
		}
		return res, nil
	case reflect.Map:
		res := make(map[string]any)
		for _, k := range v.MapKeys() {
			key, err := snakeMapKey(k)
			if err != nil {
				return nil, err
			}
			elem, err := convertSnakeValue(path+"->", v.MapIndex(k), depth+1, active)
			if err != nil {
				return nil, fmt.Errorf("failed to convert map element with path: %v err: %w", path+"->", err)
			}
			res[key] = elem
		}
		if len(res) == 0 {
			return map[string]any{}, nil
		}
		return res, nil
	case reflect.Pointer:
		if v.IsNil() {
			return nil, nil
		}
		return convertSnakeValue(path+"*", v.Elem(), depth+1, active)
	case reflect.Bool:
		return v.Bool(), nil
	case reflect.Float32, reflect.Float64:
		return v.Float(), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int(), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint(), nil

	default:
		return nil, fmt.Errorf("unsupported type: %v", v.Kind())
	}
}

// hasWalkableFields reports whether the struct walk in convertSnakeValue
// can emit a key for t: an exported field not tagged "-", an embedded
// struct it emits as one value, or a field promoted from an embedded struct.
func hasWalkableFields(t reflect.Type) bool {
	return walkableFields(t, map[reflect.Type]bool{})
}

func walkableFields(t reflect.Type, seen map[reflect.Type]bool) bool {
	if seen[t] {
		return false
	}
	seen[t] = true
	for f := range t.Fields() {
		if f.Tag.Get("json") == "-" {
			continue
		}
		ft := f.Type
		if ft.Kind() == reflect.Pointer {
			ft = ft.Elem()
		}
		if f.Anonymous && ft.Kind() == reflect.Struct {
			// An embedded time.Time or exported marshaled struct is emitted as
			// one key. An unexported marshaled struct cannot be called through
			// the embedding, so its methods count only on the promoting struct.
			if walkableFields(ft, seen) || ft == reflect.TypeFor[time.Time]() ||
				f.IsExported() && (hasMarshaler(ft) || hasMarshaler(reflect.PointerTo(ft))) {
				return true
			}
			continue
		}
		if f.IsExported() {
			return true
		}
	}
	return false
}

// isOpaque reports whether convertSnakeValue hands values of t (or of the
// type t points to) to marshalOpaque instead of walking them.
func isOpaque(t reflect.Type) bool {
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.Array:
		return hasMarshaler(t) || hasMarshaler(reflect.PointerTo(t))
	case reflect.Struct:
		return t != reflect.TypeFor[time.Time]() && !hasWalkableFields(t) &&
			(hasMarshaler(t) || hasMarshaler(reflect.PointerTo(t)))
	}
	return false
}

// marshalOpaque encodes a value the field walk cannot represent (a fixed-size
// array, or a struct without walkable fields) the way encoding/json does, when
// its type defines json.Marshaler or encoding.TextMarshaler. Examples are
// uuid.UUID, netip.Addr and *big.Int. It is not applied to other structs:
// their own MarshalJSON would replace the snake_case keys. ok is false when
// the value has neither method.
func marshalOpaque(v reflect.Value) (val any, ok bool, err error) {
	if !v.CanInterface() {
		return nil, false, nil
	}
	m := v
	if !hasMarshaler(m.Type()) && v.CanAddr() && hasMarshaler(reflect.PointerTo(v.Type())) {
		m = v.Addr()
	}
	switch x := m.Interface().(type) {
	case json.Marshaler:
		b, err := x.MarshalJSON()
		if err != nil {
			return nil, true, fmt.Errorf("failed to marshal %v: %w", v.Type(), err)
		}
		return json.RawMessage(b), true, nil
	case encoding.TextMarshaler:
		b, err := x.MarshalText()
		if err != nil {
			return nil, true, fmt.Errorf("failed to marshal %v as text: %w", v.Type(), err)
		}
		return string(b), true, nil
	}
	return nil, false, nil
}

func hasMarshaler(t reflect.Type) bool {
	return t.Implements(reflect.TypeFor[json.Marshaler]()) || t.Implements(reflect.TypeFor[encoding.TextMarshaler]())
}

func snakeMapKey(k reflect.Value) (string, error) {
	if k.Kind() == reflect.String {
		return k.String(), nil
	}
	if !k.Type().Implements(reflect.TypeFor[encoding.TextMarshaler]()) {
		if k.CanInt() {
			return strconv.FormatInt(k.Int(), 10), nil
		}
		if k.CanUint() {
			return strconv.FormatUint(k.Uint(), 10), nil
		}
	}
	return "", fmt.Errorf("unsupported map key type in snake-case conversion: %v", k.Kind())
}

func notEmpty(val any) bool {
	switch v := val.(type) {
	case map[string]any:
		return len(v) != 0
	case []any:
		return len(v) != 0
	case bool:
		return v
	case string:
		return v != ""
	default:
		return true
	}
}

// pathToName allows to provide a list of exceptions for a known input structures.
// key is a path for a name to be converted. Value is its custom replacement.
var pathToName = map[string]string{
	".LongRunningToolIDs": "long_running_tool_ids", // long_running_tool_i_ds
}

// convertName converts a name to snake case.
func convertName(path, name string) string {
	// uncomment this to check how your data is processed
	// fmt.Printf("convert(%s, %s)\n", path, name)
	if res, ok := pathToName[path]; ok {
		return res
	}

	l := strings.ToLower(name)
	b := &strings.Builder{}
	afterUnderscore := true
	for i := range len(name) {
		// Ab  => _ab
		if !afterUnderscore && i > 0 && i+1 < len(name) && name[i] != l[i] && name[i+1] == l[i+1] {
			fmt.Fprintf(b, "_%c", l[i])
			afterUnderscore = true
			continue
		}
		// aB  => a_b
		if !afterUnderscore && i+1 < len(name) && name[i] == l[i] && name[i+1] != l[i+1] {
			fmt.Fprintf(b, "%c_", l[i])
			afterUnderscore = true
			continue
		}
		afterUnderscore = false
		fmt.Fprintf(b, "%c", l[i])
	}
	return b.String()
}

// parseTag handles json tags. Accepted format is comma-separated list of strings.
// "-", "omitempty" and "omitzero" are recognized. The remaining one is treated as a name
// returns an error if duplicates are found
func parseTag(tag string) (name string, omitEmpty, omitZero, skip bool, err error) {
	if tag == "" {
		return "", false, false, false, nil
	}
	if tag == "-" {
		return "", false, false, true, nil
	}
	vals := strings.Split(tag, ",")
	name = ""
	omitEmpty = false
	omitZero = false
	skip = false
	for _, val := range vals {
		// ignore empty values
		if val == "" {
			continue
		}
		switch val {
		case "omitempty":
			if omitEmpty {
				return "", false, false, false, fmt.Errorf("duplicate omitempty")
			}
			omitEmpty = true
		case "omitzero":
			if omitZero {
				return "", false, false, false, fmt.Errorf("duplicate omitzero")
			}
			omitZero = true
		default:
			if name != "" {
				return "", false, false, false, fmt.Errorf("duplicate name")
			}
			name = val
		}
	}
	// allow empty name
	return name, omitEmpty, omitZero, skip, nil
}

// fieldName returns a name for the field after json tag is taken into the consideration
func fieldName(name, tag string) (newName string, omitEmpty, omitZero, skip bool, err error) {
	newName, omitEmpty, omitZero, skip, err = parseTag(tag)
	if newName == "" {
		newName = name
	}
	return newName, omitEmpty, omitZero, skip, err
}
