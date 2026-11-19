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

package typeutil

import "reflect"

// DeepCopy deep-copies a tool payload, such as function call arguments or a
// function response. A shallow map clone protects the top level and leaves a
// nested map shared, which is the same hole one level down. The result has
// v's dynamic type; a nil map[string]any stays a nil map[string]any, and an
// untyped nil stays nil.
//
// The decoded-JSON shapes are handled directly because they are overwhelmingly
// the common case. Everything else goes through reflection, because a payload
// is not always decoded JSON: FunctionResponse.Response holds whatever a Go
// tool handler returned, stored as-is and never normalized. functiontool wraps
// a non-object result as map[string]any{"result": v} for any TResults, and
// loadmemorytool stores map[string]any{"memories": []memory.Entry}, where an
// Entry holds a *genai.Content and a map.
//
// Unexported reference fields of structs, channels and functions stay shared,
// and so does anything nested deeper than 64 levels, which bounds the copy of
// a cyclic value.
func DeepCopy(v any) any {
	switch t := v.(type) {
	case map[string]any:
		if t == nil {
			return map[string]any(nil)
		}
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = DeepCopy(val)
		}
		return out
	case []any:
		if t == nil {
			return []any(nil)
		}
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = DeepCopy(val)
		}
		return out
	}
	if v == nil {
		return nil
	}
	return deepCopyValue(reflect.ValueOf(v), 0).Interface()
}

// maxCopyDepth bounds deepCopyValue. A payload is a tree in practice, but it is
// third-party data and nothing stops it being cyclic, and a stack overflow here
// would take down a process that was only trying to shorten a prompt. Past the
// bound the value is returned as it stands: still aliased, but bounded, and far
// deeper than real payloads reach.
const maxCopyDepth = 64

// deepCopyValue returns a copy of v that shares no mutable memory with it.
func deepCopyValue(v reflect.Value, depth int) reflect.Value {
	if !v.IsValid() || depth > maxCopyDepth || !mayAlias(v.Type()) {
		return v
	}
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return v
		}
		out := reflect.New(v.Type().Elem())
		out.Elem().Set(deepCopyValue(v.Elem(), depth+1))
		return out
	case reflect.Interface:
		if v.IsNil() {
			return v
		}
		out := reflect.New(v.Type()).Elem()
		out.Set(deepCopyValue(v.Elem(), depth+1))
		return out
	case reflect.Map:
		if v.IsNil() {
			return v
		}
		out := reflect.MakeMapWithSize(v.Type(), v.Len())
		for it := v.MapRange(); it.Next(); {
			out.SetMapIndex(deepCopyValue(it.Key(), depth+1), deepCopyValue(it.Value(), depth+1))
		}
		return out
	case reflect.Slice:
		if v.IsNil() {
			return v
		}
		out := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		reflect.Copy(out, v)
		for i := range v.Len() {
			out.Index(i).Set(deepCopyValue(v.Index(i), depth+1))
		}
		return out
	case reflect.Array:
		out := reflect.New(v.Type()).Elem()
		reflect.Copy(out, v)
		for i := range v.Len() {
			out.Index(i).Set(deepCopyValue(v.Index(i), depth+1))
		}
		return out
	case reflect.Struct:
		out := reflect.New(v.Type()).Elem()
		// Whole-struct assignment first, so unexported fields survive: they
		// cannot be assigned individually, and a field-by-field rebuild would
		// zero them -- time.Time being the everyday example. What stays shared
		// is an unexported reference field, which a summarizer cannot reach
		// without unsafe.
		out.Set(v)
		for i := range v.NumField() {
			if v.Type().Field(i).IsExported() {
				out.Field(i).Set(deepCopyValue(v.Field(i), depth+1))
			}
		}
		return out
	default:
		// Chan, Func and UnsafePointer reach here. None of them is
		// conversation, and none can be meaningfully copied.
		return v
	}
}

// mayAlias reports whether a value of type t can hold a reference to memory
// somebody else may write through, so that plain data is returned as it stands
// rather than rebuilt field by field.
func mayAlias(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func, reflect.UnsafePointer:
		return true
	case reflect.Array:
		return mayAlias(t.Elem())
	case reflect.Struct:
		for field := range t.Fields() {
			if mayAlias(field.Type) {
				return true
			}
		}
		return false
	default:
		return false
	}
}
