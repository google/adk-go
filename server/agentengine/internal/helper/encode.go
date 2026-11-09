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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"maps"
	"reflect"
	"strings"
	"time"
)

// ConvertSnake returns an object corresponding the input object. It uses simple types, map[string]any and []any to reflect the input.
// All names are converted to snake case ('RequestedToolConfirmations' becomes 'requested_tool_confirmations') using simple method.
// There are some possible exceptions, which are handled in a custom way using pathToName.
// json tags are supported.
// struct embedding is supported.
func ConvertSnake(o any) any {
	res, err := convertSnake("", "", o)
	if err != nil {
		log.Printf("Failed to convert: %+v of type %T: %v", o, o, err) //nolint:forbidigo // pre-slog call site
		// better to return an original version than nothing
		return o
	}
	return res
}

func convertSnake(path, indent string, o any) (any, error) {
	return convertSnakeValue(path, indent, reflect.ValueOf(o))
}

// Work with reflect.Values so unexported embedded structs can expose their
// exported fields without calling Interface on the struct itself.
func convertSnakeValue(path, indent string, v reflect.Value) (any, error) {
	if !v.IsValid() {
		return nil, nil
	}
	switch v.Kind() {
	case reflect.String:
		if v.CanInterface() {
			return v.Interface(), nil
		}
		return v.String(), nil
	case reflect.Struct:
		vt := v.Type()
		if vt == reflect.TypeOf(time.Time{}) {
			if !v.CanInterface() {
				return nil, fmt.Errorf("cannot convert unexported time field")
			}
			t := v.Interface().(time.Time)
			return t.UnixMilli() / 1000.0, nil // returns a number of seconds "Unix-way"
		}

		m := make(map[string]any)
		for i := 0; i < v.NumField(); i++ {
			fv := v.Field(i)
			fvt := vt.Field(i)
			tag := fvt.Tag.Get("json")
			name, omitEmpty, omitZero, skip, err := fieldName(fvt.Name, tag)
			if err != nil {
				return nil, fmt.Errorf("failed to parse tag (%v): %w", tag, err)
			}
			if skip {
				continue
			}
			newPath := path + "." + name
			if fvt.Anonymous {
				embed, err := convertSnakeValue(newPath, indent+".   ", fv)
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

			newName := convertName(newPath, name)
			if fv.CanInterface() {
				val, err := convertSnake(newPath, indent+".   ", fv.Interface())
				if err != nil {
					return nil, fmt.Errorf("failed to convert regular struct field with path: %v err: %w", newPath, err)
				}
				if omitEmpty {
					// check for emptiness
					if val != nil {
						// empty map
						addIfNotEmpty(val, m, newName)
					}
				} else {
					if val != nil {
						m[newName] = val
					}
				}
			} else {
				// respect omitZero
				val := convertValue(fv)
				if val != 0 || !omitZero {
					m[newName] = val
				}
			}

		}
		return m, nil
	case reflect.Slice:
		if v.Type() == reflect.TypeFor[json.RawMessage]() {
			rm := json.RawMessage(v.Bytes())
			if len(rm) == 0 {
				return nil, nil
			}
			return rm, nil
		}
		if v.Type().Elem().Kind() == reflect.Uint8 {
			return base64.StdEncoding.EncodeToString(v.Bytes()), nil
		}
		res := []any{}
		for i := 0; i < v.Len(); i++ {
			elem, err := convertSnakeValue(path+".[]", indent+"    ", v.Index(i))
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
			elem, err := convertSnakeValue(path+"->", indent+"    ", v.MapIndex(k))
			if err != nil {
				return nil, fmt.Errorf("failed to convert map element with path: %v err: %w", path+"->", err)
			}
			res[k.String()] = elem
		}
		if len(res) == 0 {
			return map[string]any{}, nil
		}
		return res, nil
	case reflect.Pointer, reflect.Interface:
		if v.IsNil() {
			return nil, nil
		}
		return convertSnakeValue(path+"*", indent+"    ", v.Elem())
	case reflect.Bool:
		return v.Bool(), nil
	case reflect.Float32, reflect.Float64:
		return v.Float(), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int(), nil

	default:
		return nil, fmt.Errorf("unsupported type: %v", v.Kind())
	}
}

func addIfNotEmpty(val any, m map[string]any, newName string) {
	switch t := val.(type) {
	case map[string]any:
		if len(t) != 0 {
			m[newName] = val
		}
	case []any:
		if len(t) != 0 {
			m[newName] = val
		}
	case bool:
		if t {
			m[newName] = val
		}
	case string:
		if t != "" {
			m[newName] = val
		}
	default:
		m[newName] = val
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
	for i := 0; i < len(name); i++ {
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

// convertValue handles String, Int, Uint and Float
func convertValue(o reflect.Value) any {
	if o.CanInt() {
		return o.Int()
	}
	if o.CanUint() {
		return o.Uint()
	}
	if o.CanFloat() {
		return o.Float()
	}
	if o.Kind() == reflect.String {
		return o.String()
	}
	return o
}
