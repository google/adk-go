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
	"reflect"
	"strings"
	"sync"
	"time"
)

// ConvertSnake returns an object corresponding the input object. It uses simple types, map[string]any and []any to reflect the input.
// All names are converted to snake case ('RequestedToolConfirmations' becomes 'requested_tool_confirmations') using simple method.
// There are some possible exceptions, which are handled in a custom way using pathToName.
// json tags are supported.
// struct embedding is supported.
func ConvertSnake(o any) any {
	res, err := convertSnake("", o)
	if err != nil {
		log.Printf("Failed to convert: %+v of type %T: %v", o, o, err) //nolint:forbidigo // pre-slog call site
		// better to return an original version than nothing
		return o
	}
	return res
}

var (
	timeType   = reflect.TypeFor[time.Time]()
	stringType = reflect.TypeFor[string]()
)

// convertSnake traverses o with reflection and converts struct keys to snake_case.
func convertSnake(path string, o any) (any, error) {
	if o == nil {
		return nil, nil
	}
	v := reflect.ValueOf(o)
	switch v.Kind() {
	case reflect.String:
		s, ok := o.(string)
		if !ok {
			return o, nil
		}
		return s, nil
	case reflect.Struct:
		vt := v.Type()
		if vt == timeType {
			t := o.(time.Time)
			return t.UnixMilli() / 1000.0, nil
		}

		m := make(map[string]any)
		plan, err := planFor(vt)
		if err != nil {
			return nil, err
		}
		atRoot := path == ""
		for i := range plan {
			f := &plan[i]
			fv := v.Field(f.index)
			if f.anonymous {
				embed, err := convertSnake(".", fv.Interface())
				if err != nil {
					return nil, fmt.Errorf("failed to convert embedded struct with name:%v o: %+v %T err: %w", f.name, fv.Interface(), fv.Interface(), err)
				}
				for k, v := range embed.(map[string]any) {
					m[k] = v
				}
				continue
			}

			if f.exported {
				switch f.kind {
				case reflect.Pointer, reflect.Interface:
					if fv.IsNil() {
						continue
					}
				case reflect.Slice, reflect.Map:
					if f.omitEmpty && fv.Len() == 0 {
						continue
					}
				case reflect.String:
					if f.omitEmpty && f.plainString && fv.Len() == 0 {
						continue
					}
				case reflect.Bool:
					if f.omitEmpty && !fv.Bool() {
						continue
					}
				}
				newName := f.snake
				if atRoot {
					newName = f.rootSnake
				}
				val, err := convertSnake(".", fv.Interface())
				if err != nil {
					return nil, fmt.Errorf("failed to convert regular struct field with path: %v err: %w", path+"."+f.name, err)
				}
				if f.omitEmpty {
					if val != nil {
						addIfNotEmpty(val, m, newName)
					}
				} else if val != nil {
					m[newName] = val
				}
			} else {
				newName := f.snake
				if atRoot {
					newName = f.rootSnake
				}
				val := convertValue(fv)
				if val != 0 || !f.omitZero {
					m[newName] = val
				}
			}
		}
		return m, nil
	case reflect.Slice:
		if rm, ok := o.(json.RawMessage); ok {
			if len(rm) == 0 {
				return nil, nil
			}
			return rm, nil
		}
		if v.Type().Elem().Kind() == reflect.Uint8 {
			return base64.StdEncoding.EncodeToString(v.Bytes()), nil
		}
		n := v.Len()
		if n == 0 {
			return []any{}, nil
		}
		res := make([]any, n)
		for i := range n {
			elem, err := convertSnake(".", v.Index(i).Interface())
			if err != nil {
				return nil, fmt.Errorf("failed to convert slice element with path: %v err: %w", path+".[]", err)
			}
			res[i] = elem
		}
		return res, nil
	case reflect.Map:
		if v.Len() == 0 {
			return map[string]any{}, nil
		}
		res := make(map[string]any, v.Len())
		iter := v.MapRange()
		for iter.Next() {
			elem, err := convertSnake(".", iter.Value().Interface())
			if err != nil {
				return nil, fmt.Errorf("failed to convert map element with path: %v err: %w", path+"->", err)
			}
			res[iter.Key().String()] = elem
		}
		return res, nil
	case reflect.Pointer:
		if v.IsNil() {
			return nil, nil
		}
		return convertSnake(".", v.Elem().Interface())
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

// fieldPlan holds the per-field metadata convertSnake needs for a struct type.
type fieldPlan struct {
	index       int
	kind        reflect.Kind
	name        string // json tag name, else the Go field name
	snake       string // name in snake case
	rootSnake   string // name at the root path ("."+name), honoring pathToName
	anonymous   bool
	exported    bool
	plainString bool
	omitEmpty   bool
	omitZero    bool
}

// fieldPlans caches []fieldPlan by reflect.Type. Fields tagged "-" are left out.
var fieldPlans sync.Map

func planFor(t reflect.Type) ([]fieldPlan, error) {
	if cached, ok := fieldPlans.Load(t); ok {
		return cached.([]fieldPlan), nil
	}
	plan := make([]fieldPlan, 0, t.NumField())
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag := f.Tag.Get("json")
		name, omitEmpty, omitZero, skip, err := fieldName(f.Name, tag)
		if err != nil {
			return nil, fmt.Errorf("failed to parse tag (%v): %w", tag, err)
		}
		if skip {
			continue
		}
		snake := snakeName(name)
		rootSnake := snake
		if custom, ok := pathToName["."+name]; ok {
			rootSnake = custom
		}
		plan = append(plan, fieldPlan{
			index:       i,
			kind:        f.Type.Kind(),
			name:        name,
			snake:       snake,
			rootSnake:   rootSnake,
			anonymous:   f.Anonymous,
			exported:    f.IsExported(),
			plainString: f.Type == stringType,
			omitEmpty:   omitEmpty,
			omitZero:    omitZero,
		})
	}
	fieldPlans.Store(t, plan)
	return plan, nil
}

// convertName converts a name to snake case.
func convertName(path, name string) string {
	if res, ok := pathToName[path]; ok {
		return res
	}
	return snakeName(name)
}

func snakeName(name string) string {
	l := strings.ToLower(name)
	var b strings.Builder
	b.Grow(len(name) + 4)
	afterUnderscore := true
	for i := 0; i < len(name); i++ {
		// Ab  => _ab
		if !afterUnderscore && i > 0 && i+1 < len(name) && name[i] != l[i] && name[i+1] == l[i+1] {
			b.WriteByte('_')
			b.WriteByte(l[i])
			afterUnderscore = true
			continue
		}
		// aB  => a_b
		if !afterUnderscore && i+1 < len(name) && name[i] == l[i] && name[i+1] != l[i+1] {
			b.WriteByte(l[i])
			b.WriteByte('_')
			afterUnderscore = true
			continue
		}
		afterUnderscore = false
		b.WriteByte(l[i])
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
