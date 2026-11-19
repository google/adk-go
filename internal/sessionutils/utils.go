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

package sessionutils

import (
	"maps"
	"reflect"
	"strings"
	"time"
)

const (
	appPrefix  = "app:"
	userPrefix = "user:"
	tempPrefix = "temp:"
)

// ExtractStateDeltas splits a single state delta map into three separate maps
// for app, user, and session states based on key prefixes.
// Temporary keys (starting with tempPrefix) are ignored.
func ExtractStateDeltas(delta map[string]any) (
	appStateDelta, userStateDelta, sessionStateDelta map[string]any,
) {
	// Initialize the maps to be returned.
	appStateDelta = make(map[string]any)
	userStateDelta = make(map[string]any)
	sessionStateDelta = make(map[string]any)

	if delta == nil {
		return appStateDelta, userStateDelta, sessionStateDelta
	}

	for key, value := range delta {
		if cleanKey, found := strings.CutPrefix(key, appPrefix); found {
			appStateDelta[cleanKey] = value
		} else if cleanKey, found := strings.CutPrefix(key, userPrefix); found {
			userStateDelta[cleanKey] = value
		} else if !strings.HasPrefix(key, tempPrefix) {
			// This key belongs to the session state, as long as it's not temporary.
			sessionStateDelta[key] = value
		}
	}
	return appStateDelta, userStateDelta, sessionStateDelta
}

// MergeStates combines app, user, and session state maps into a single map
// for client-side responses, adding the appropriate prefixes back.
func MergeStates(appState, userState, sessionState map[string]any) map[string]any {
	// Pre-allocate map capacity for efficiency.
	totalSize := len(appState) + len(userState) + len(sessionState)
	mergedState := make(map[string]any, totalSize)

	maps.Copy(mergedState, sessionState)

	for key, value := range appState {
		mergedState[appPrefix+key] = value
	}

	for key, value := range userState {
		mergedState[userPrefix+key] = value
	}

	return mergedState
}

// CloneStateMap returns a deep copy of state that shares no mutable memory
// with state at any depth.
//
// The result is never nil: a nil or empty state gives an empty map. Maps,
// slices, arrays, pointers and the exported fields of structs are copied at
// every depth, and values keep their dynamic types, so an int stays an int and
// a named slice type keeps its name. Every map, pointer and slice of nonzero
// capacity is copied once, so a value that refers to itself comes back as a
// copy that refers to the copy, and two references to one value come back as
// two references to one copy. A nil map, slice, pointer or interface stays
// nil, and a key whose value is nil stays present.
//
// The unexported fields of a struct are copied by value, so a struct such as
// time.Time survives intact, and func and channel values cannot be copied and
// stay shared. The caller must keep writers off state and everything reachable
// from it until CloneStateMap returns.
func CloneStateMap(state map[string]any) map[string]any {
	if state == nil {
		return make(map[string]any)
	}
	c := cloner{memo: make(map[visitKey]reflect.Value)}
	return c.copy(reflect.ValueOf(state)).Interface().(map[string]any)
}

// CloneValue returns a deep copy of v that shares no mutable memory with v at
// any depth, using the same reflection cloner as [CloneStateMap].
func CloneValue[T any](v T) T {
	c := cloner{memo: make(map[visitKey]reflect.Value)}
	// Go through pointers so that an interface-typed T holding nil stays a
	// valid reflect.Value instead of becoming the zero Value.
	var out T
	reflect.ValueOf(&out).Elem().Set(c.copy(reflect.ValueOf(&v).Elem()))
	return out
}

// MutableSession is implemented by service-created and cloned [session.Session]
// values (*session.session in package session and *database.localSession in
// package session/database) so that [session.Service.AppendEvent] can atomically
// apply state deltas, append the temp-trimmed event, and update the session's
// last update time after storage validation and persistence succeed.
type MutableSession interface {
	CommitAppend(trimmedEvent any, rawStateDelta map[string]any, updatedAt time.Time)
}

// visitKey identifies a map, pointer or slice already copied by a cloner. The
// length and capacity are zero for a map or pointer. A slice needs them because
// s and s[:1] share a data pointer but are different values.
type visitKey struct {
	typ      reflect.Type
	ptr      uintptr
	len, cap int
}

// cloner deep-copies values. memo maps each map, pointer and slice copied so
// far to its copy, so a value reached again through a cycle or a second
// reference resolves to the same copy instead of being copied again.
type cloner struct {
	memo map[visitKey]reflect.Value
}

// copy returns a copy of v that shares no mutable memory with it, except
// through unexported struct fields and func and channel values.
func (c *cloner) copy(v reflect.Value) reflect.Value {
	if !v.IsValid() || !mayAlias(v.Type()) {
		return v
	}
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return v
		}
		k := visitKey{typ: v.Type(), ptr: v.Pointer()}
		if out, ok := c.memo[k]; ok {
			return out
		}
		out := reflect.New(v.Type().Elem())
		c.memo[k] = out
		out.Elem().Set(c.copy(v.Elem()))
		return out
	case reflect.Interface:
		if v.IsNil() {
			return v
		}
		out := reflect.New(v.Type()).Elem()
		out.Set(c.copy(v.Elem()))
		return out
	case reflect.Map:
		if v.IsNil() {
			return v
		}
		k := visitKey{typ: v.Type(), ptr: v.Pointer()}
		if out, ok := c.memo[k]; ok {
			return out
		}
		out := reflect.MakeMapWithSize(v.Type(), v.Len())
		c.memo[k] = out
		for it := v.MapRange(); it.Next(); {
			out.SetMapIndex(c.copy(it.Key()), c.copy(it.Value()))
		}
		return out
	case reflect.Slice:
		if v.IsNil() {
			return v
		}
		out := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		// A slice of capacity zero has no backing array of its own, so its
		// data pointer can be shared by unrelated values and is no identity.
		if v.Cap() > 0 {
			k := visitKey{typ: v.Type(), ptr: v.Pointer(), len: v.Len(), cap: v.Cap()}
			if out, ok := c.memo[k]; ok {
				return out
			}
			c.memo[k] = out
		}
		if !mayAlias(v.Type().Elem()) {
			reflect.Copy(out, v)
			return out
		}
		for i := range v.Len() {
			out.Index(i).Set(c.copy(v.Index(i)))
		}
		return out
	case reflect.Array:
		out := reflect.New(v.Type()).Elem()
		for i := range v.Len() {
			out.Index(i).Set(c.copy(v.Index(i)))
		}
		return out
	case reflect.Struct:
		out := reflect.New(v.Type()).Elem()
		// Set the whole struct first so unexported fields survive, then
		// replace each exported field with its copy.
		out.Set(v)
		for i := range v.NumField() {
			if v.Type().Field(i).IsExported() {
				out.Field(i).Set(c.copy(v.Field(i)))
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
		for f := range t.Fields() {
			if mayAlias(f.Type) {
				return true
			}
		}
	}
	return false
}
