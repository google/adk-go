// Copyright 2025 Google LLC
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

package artifact

import (
	"reflect"

	"google.golang.org/genai"
)

// maxCloneDepth bounds deepCopyValue. A part is a tree in practice, but
// PartMetadata and tool payloads are caller data and could be cyclic. Past the
// bound the value is returned as it stands.
const maxCloneDepth = 64

// clonePart returns a deep copy of p. No pointer, slice or map reachable from
// the copy is reachable from p, at any depth, so neither the saver nor a loader
// can change a stored version by editing the part it holds. Nil stays nil and
// an empty slice or map stays empty and non-nil.
//
// The copy walks the type with reflection instead of listing fields, so a
// member that genai adds to Part, or to any type reachable from it, is copied
// without a change here. TestClonePartCopiesEveryField fails if that stops
// being true. Unexported fields, a time.Time for example, are copied by value.
// Func and channel values are not copyable and stay shared.
//
// adk-python deep-copies on save and on load too (in_memory_artifact_service.py).
func clonePart(p *genai.Part) *genai.Part {
	if p == nil {
		return nil
	}
	return deepCopyValue(reflect.ValueOf(p), 0).Interface().(*genai.Part)
}

// deepCopyValue returns a copy of v that shares no mutable memory with it.
func deepCopyValue(v reflect.Value, depth int) reflect.Value {
	if !v.IsValid() || depth > maxCloneDepth || !mayAlias(v.Type()) {
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
		// Assign the whole struct first so unexported fields survive, then
		// replace each exported field with its deep copy.
		out.Set(v)
		for i := range v.NumField() {
			if v.Type().Field(i).IsExported() {
				out.Field(i).Set(deepCopyValue(v.Field(i), depth+1))
			}
		}
		return out
	default:
		return v
	}
}

// mayAlias reports whether a value of type t can hold a reference to memory
// that another holder may write through.
func mayAlias(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func, reflect.UnsafePointer:
		return true
	case reflect.Array:
		return mayAlias(t.Elem())
	case reflect.Struct:
		for i := range t.NumField() {
			if mayAlias(t.Field(i).Type) {
				return true
			}
		}
	}
	return false
}
