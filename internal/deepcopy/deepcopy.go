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

// Package deepcopy copies a value so that the copy shares no mutable memory
// with the original.
package deepcopy

import (
	"reflect"
	"unsafe"
)

// maxDepth bounds the recursion of a copy. Each step through a pointer,
// interface, map, slice, array or struct costs one unit, so a nested
// map[string]any costs two per level. A value deeper than the bound is not
// copied and stays shared with the original. Cycles do not need the bound,
// since the copy memoizes what it has already copied; the bound keeps very deep
// data from overflowing the stack.
const maxDepth = 64

// Copy returns a deep copy of v. No pointer, slice or map reachable from the
// copy within the depth bound is reachable from v, so writing through the copy
// cannot change v and the reverse. Nil stays nil, and an empty slice or map
// stays empty and non-nil.
//
// The copy walks the type with reflection instead of listing fields, so a
// field added to a type later is copied without a change here. A pointer, map
// or slice reached twice in v is copied once and reached twice in the copy, so
// shared structure and cycles are preserved rather than unrolled.
//
// Unexported fields, a time.Time for example, are copied by value: whatever
// they point to stays shared. Func and channel values cannot be copied and
// stay shared too. A copied slice has the same length as the original, and its
// capacity equals its length.
func Copy[T any](v T) T {
	var c copier
	out := c.copy(reflect.ValueOf(v), 0)
	if !out.IsValid() {
		return v
	}
	return out.Interface().(T)
}

// ref identifies memory already copied. The type is part of the key because a
// struct and its first field share an address, and the length because two
// slices of one array can start at the same element.
type ref struct {
	addr unsafe.Pointer
	typ  reflect.Type
	len  int
}

// copied is a copy and the depth it was made at.
type copied struct {
	val   reflect.Value
	depth int
}

type copier struct {
	seen map[ref]copied
}

// lookup returns the copy already made of k, if it was made at depth or
// shallower. A copy made deeper stopped at the bound sooner, so parts of it are
// still shared that a copy made here must not share; the caller copies again.
func (c *copier) lookup(k ref, depth int) (reflect.Value, bool) {
	e, ok := c.seen[k]
	if !ok || e.depth > depth {
		return reflect.Value{}, false
	}
	return e.val, true
}

func (c *copier) remember(k ref, depth int, out reflect.Value) {
	if c.seen == nil {
		c.seen = map[ref]copied{}
	}
	c.seen[k] = copied{out, depth}
}

func (c *copier) copy(v reflect.Value, depth int) reflect.Value {
	if !v.IsValid() || depth > maxDepth || !mayAlias(v.Type()) {
		return v
	}
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return v
		}
		key := ref{v.UnsafePointer(), v.Type(), 0}
		if out, ok := c.lookup(key, depth); ok {
			return out
		}
		out := reflect.New(v.Type().Elem())
		c.remember(key, depth, out)
		out.Elem().Set(c.copy(v.Elem(), depth+1))
		return out
	case reflect.Interface:
		if v.IsNil() {
			return v
		}
		out := reflect.New(v.Type()).Elem()
		out.Set(c.copy(v.Elem(), depth+1))
		return out
	case reflect.Map:
		if v.IsNil() {
			return v
		}
		key := ref{v.UnsafePointer(), v.Type(), 0}
		if out, ok := c.lookup(key, depth); ok {
			return out
		}
		out := reflect.MakeMapWithSize(v.Type(), v.Len())
		c.remember(key, depth, out)
		for it := v.MapRange(); it.Next(); {
			out.SetMapIndex(c.copy(it.Key(), depth+1), c.copy(it.Value(), depth+1))
		}
		return out
	case reflect.Slice:
		if v.IsNil() {
			return v
		}
		key := ref{v.UnsafePointer(), v.Type(), v.Len()}
		if out, ok := c.lookup(key, depth); ok {
			return out
		}
		out := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		reflect.Copy(out, v)
		if v.Len() > 0 {
			// An empty slice may point at memory shared by every zero-size
			// allocation, so it is not an identity worth keeping.
			c.remember(key, depth, out)
		}
		// reflect.Copy is the whole copy for elements that hold no reference,
		// such as the bytes of a genai.Blob.
		if !mayAlias(v.Type().Elem()) {
			return out
		}
		for i := range v.Len() {
			out.Index(i).Set(c.copy(v.Index(i), depth+1))
		}
		return out
	case reflect.Array:
		// Only arrays whose element can alias get here: the mayAlias check
		// above returns an array of plain values, such as [N]byte, as is.
		out := reflect.New(v.Type()).Elem()
		reflect.Copy(out, v)
		for i := range v.Len() {
			out.Index(i).Set(c.copy(v.Index(i), depth+1))
		}
		return out
	case reflect.Struct:
		out := reflect.New(v.Type()).Elem()
		// Assign the whole struct first so unexported fields survive, then
		// replace each exported field with its deep copy.
		out.Set(v)
		for i := range v.NumField() {
			if v.Type().Field(i).IsExported() {
				out.Field(i).Set(c.copy(v.Field(i), depth+1))
			}
		}
		return out
	default:
		// Chan, Func and UnsafePointer.
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
		for field := range t.Fields() {
			if mayAlias(field.Type) {
				return true
			}
		}
	}
	return false
}
