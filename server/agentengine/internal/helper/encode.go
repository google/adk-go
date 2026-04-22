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
	"fmt"
	"log"
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
	return convertSnake("", "", o)
}

// converSnake does the job. It includes indent for debugging purposes
// uses reflect to traverse the object
func convertSnake(path string, indent string, o any) any {
	// handle nil
	if o == nil {
		return nil
	}
	v := reflect.ValueOf(o)
	switch v.Kind() {
	case reflect.String:
		// return string as-is
		return o
	case reflect.Struct:
		vt := v.Type()
		// handle time.Time in a special way
		if vt.String() == "time.Time" {
			var t time.Time
			t = o.(time.Time)
			return t.UnixMilli() / 1000.0 // returns a number of seconds "Unix-way"
		}

		// this map will hold all the fields
		m := make(map[string]any)
		// iterate over the fields handling all the cases
		for i := 0; i < v.NumField(); i++ {
			fv := v.Field(i)
			fvt := vt.Field(i)
			tag := fvt.Tag.Get("json")
			name, omitEmpty, omitZero, skip, err := fieldName(fvt.Name, tag)
			if err != nil {
				log.Fatalf("Failed to parse tag: %v", err)
			}
			// respect json "-"
			if skip {
				continue
			}
			// handle embedded structs
			if fvt.Anonymous {
				embed := convertSnake(path+"."+name, indent+".   ", fv.Interface())
				// merge them to m
				for k, v := range embed.(map[string]any) {
					m[k] = v
				}
			} else {
				// regular struct
				newPath := path + "." + name
				newName := convertName(newPath, name)
				if fv.CanInterface() {
					val := convertSnake(newPath, indent+".   ", fv.Interface())
					if omitEmpty {
						// check for emptiness
						if val != nil {
							// empty map
							if mapVal, ok := val.(map[string]any); ok {
								if mapVal != nil {
									m[newName] = val
								}
							} else {
								// empty array
								if arrVal, ok := val.([]any); ok {
									if len(arrVal) != 0 {
										m[newName] = val
									}
								} else {
									// bool with false
									if boolVal, ok := val.(bool); ok {
										if boolVal {
											m[newName] = val
										}
									} else {
										// empty string
										if strVal, ok := val.(string); ok {
											if strVal != "" {
												m[newName] = val
											}
										} else {
											m[newName] = val
										}
									}
								}
							}
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
		}
		return m
	case reflect.Slice:
		res := []any{}
		for i := 0; i < v.Len(); i++ {
			res = append(res, convertSnake(path+".[]", indent+"    ", v.Index(i).Interface()))
		}
		if len(res) == 0 {
			return []any{}
		}
		return res
	case reflect.Map:
		res := make(map[string]any)
		for _, k := range v.MapKeys() {
			res[k.String()] = convertSnake(path+"->", indent+"    ", v.MapIndex(k).Interface())
		}
		if len(res) == 0 {
			return map[string]any{}
		}
		return res
	case reflect.Ptr:
		if v.IsNil() {
			return nil
		}
		return convertSnake(path+"*", indent+"    ", v.Elem().Interface())
	case reflect.Bool:
		return v.Bool()
	case reflect.Float32, reflect.Float64:
		return v.Float()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int()

	default:
		log.Fatalf("Unsupported type: %v", v.Kind())
	}
	return nil
}

// pathToName allows to provide a list of exceptions for a known input structures.
// key is a path for a name to be converted. Value is its custom replacement.
var pathToName = map[string]string{
	".LongRunningToolIDs": "long_running_tool_ids", //long_running_tool_i_ds
}

// convertName converts a name to snake case.
func convertName(path string, name string) string {
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
func parseTag(tag string) (name string, omitEmpty bool, omitZero bool, skip bool, err error) {
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
func fieldName(name string, tag string) (newName string, omitEmpty bool, omitZero bool, skip bool, err error) {
	newName, omitEmpty, omitZero, skip, err = parseTag(tag)
	if newName == "" {
		newName = name
	}
	return newName, omitEmpty, omitZero, skip, err
}

// convertValue handles Int, Uint and Float
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
	return o
}
