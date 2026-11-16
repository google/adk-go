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

// Package deepcopy makes deep copies of Go values.
package deepcopy

import (
	"reflect"
	"sync"
	"time"
	"unsafe"
)

// maxDepth bounds the recursion in copyInto, so a value nested deeper than any
// real one cannot exhaust the stack. Each pointer, interface, map, slice, array
// or struct on the way down costs one step, and whatever lies below the bound
// is kept as it is.
const maxDepth = 10000

// Of returns a deep copy of v. No pointer, map or slice reachable from the
// copy is reachable from v, except for these, which the copy keeps as they are
// in v:
//
//   - A struct with an unexported field, a struct or array that holds one by
//     value, and a pointer, slice or map whose elements are such values. Of
//     cannot copy the unexported field, and copying the rest of the struct
//     around it can break the type: a strings.Builder copied that way panics
//     on its next write, and a big.Int shares its digits with the original.
//     Keeping the value also keeps its identity, so errors.Is still matches a
//     sentinel error. time.Time is the exception: it is meant to be copied by
//     value, and a copy keeps its *time.Location, which never changes.
//   - Funcs, channels and unsafe pointers, which cannot be copied.
//   - Map keys, which the map compares by value.
//   - Anything more than maxDepth steps below v.
//
// A pointer, map or slice that v reaches more than once is copied once, so the
// copy has the same cycles and shared references as v. Nil stays nil, and an
// empty slice or map stays empty and non-nil.
func Of[T any](v T) T {
	out := v
	var c copier
	c.copyInto(reflect.ValueOf(&out).Elem(), 0)
	return out
}

// copier holds the state of one call to Of.
type copier struct {
	// copies maps each pointer, map and slice copied so far to its copy.
	copies map[ref]reflect.Value
}

// ref identifies a pointer, map or slice. A slice is identified by its
// backing array and length, since two slices of one array can differ.
type ref struct {
	typ reflect.Type
	ptr unsafe.Pointer
	len int
}

// copyInto replaces each reference that dst holds, or that a value inside dst
// holds, with a reference to a copy. dst must be settable. It starts as a
// shallow copy of the original, so structs, arrays and plain values need no
// further work.
func (c *copier) copyInto(dst reflect.Value, depth int) {
	info := infoOf(dst.Type())
	if depth > maxDepth || !info.needsCopy {
		return
	}
	switch dst.Kind() {
	case reflect.Pointer:
		if dst.IsNil() {
			return
		}
		key := ref{typ: dst.Type(), ptr: dst.UnsafePointer()}
		if p, ok := c.copies[key]; ok {
			dst.Set(p)
			return
		}
		p := reflect.New(dst.Type().Elem())
		c.remember(key, p)
		p.Elem().Set(dst.Elem())
		c.copyInto(p.Elem(), depth+1)
		dst.Set(p)
	case reflect.Interface:
		if dst.IsNil() {
			return
		}
		if v, ok := c.copyOf(dst.Elem(), depth+1); ok {
			dst.Set(v)
		}
	case reflect.Map:
		if dst.IsNil() {
			return
		}
		key := ref{typ: dst.Type(), ptr: dst.UnsafePointer()}
		if m, ok := c.copies[key]; ok {
			dst.Set(m)
			return
		}
		m := reflect.MakeMapWithSize(dst.Type(), dst.Len())
		c.remember(key, m)
		for it := dst.MapRange(); it.Next(); {
			v, _ := c.copyOf(it.Value(), depth+1)
			m.SetMapIndex(it.Key(), v)
		}
		dst.Set(m)
	case reflect.Slice:
		if dst.IsNil() {
			return
		}
		key := ref{typ: dst.Type(), ptr: dst.UnsafePointer(), len: dst.Len()}
		if s, ok := c.copies[key]; ok {
			dst.Set(s)
			return
		}
		s := reflect.MakeSlice(dst.Type(), dst.Len(), dst.Len())
		c.remember(key, s)
		// reflect.Copy is the whole copy for elements that hold no reference,
		// such as the bytes of a blob.
		reflect.Copy(s, dst)
		if infoOf(dst.Type().Elem()).needsCopy {
			for i := range s.Len() {
				c.copyInto(s.Index(i), depth+1)
			}
		}
		dst.Set(s)
	case reflect.Array:
		for i := range dst.Len() {
			c.copyInto(dst.Index(i), depth+1)
		}
	case reflect.Struct:
		// An opaque struct does not need a copy, so every field listed here
		// is exported and settable.
		for _, i := range info.fields {
			c.copyInto(dst.Field(i), depth+1)
		}
	}
}

// copyOf returns a copy of v, which need not be settable, and whether the copy
// differs from v.
func (c *copier) copyOf(v reflect.Value, depth int) (reflect.Value, bool) {
	if !infoOf(v.Type()).needsCopy {
		return v, false
	}
	// Look through an interface, so the plain values a map[string]any holds
	// are kept without allocating.
	if v.Kind() == reflect.Interface && (v.IsNil() || !infoOf(v.Elem().Type()).needsCopy) {
		return v, false
	}
	out := reflect.New(v.Type()).Elem()
	out.Set(v)
	c.copyInto(out, depth)
	return out, true
}

func (c *copier) remember(key ref, copied reflect.Value) {
	if c.copies == nil {
		c.copies = make(map[ref]reflect.Value)
	}
	c.copies[key] = copied
}

// typeInfo is what copyInto needs to know about a type. infoOf computes it
// once per type.
type typeInfo struct {
	// needsCopy reports whether a value of the type can hold a reference that
	// Of replaces with a copy.
	needsCopy bool
	// fields lists, for a struct, the fields whose type needs a copy.
	fields []int
}

var (
	typeInfos   sync.Map // reflect.Type -> *typeInfo
	opaqueTypes sync.Map // reflect.Type -> bool
	timeType    = reflect.TypeFor[time.Time]()
)

func infoOf(t reflect.Type) *typeInfo {
	if v, ok := typeInfos.Load(t); ok {
		return v.(*typeInfo)
	}
	info := &typeInfo{}
	// A time.Time is complete as a by-value copy.
	if t != timeType && !opaque(t) {
		switch t.Kind() {
		case reflect.Pointer, reflect.Slice, reflect.Map:
			info.needsCopy = !opaque(t.Elem())
		case reflect.Interface:
			info.needsCopy = true
		case reflect.Array:
			info.needsCopy = infoOf(t.Elem()).needsCopy
		case reflect.Struct:
			// infoOf recurses only into struct fields and array elements, which
			// a type holds by value, so it ends for recursive types too.
			for field := range t.Fields() {
				if infoOf(field.Type).needsCopy {
					info.fields = append(info.fields, field.Index[0])
				}
			}
			info.needsCopy = len(info.fields) > 0
		}
	}
	v, _ := typeInfos.LoadOrStore(t, info)
	return v.(*typeInfo)
}

// opaque reports whether a value of type t cannot be copied field by field: it
// is a struct with an unexported field, or a struct or array that holds one by
// value. time.Time is not opaque, because it is meant to be copied by value.
func opaque(t reflect.Type) bool {
	if v, ok := opaqueTypes.Load(t); ok {
		return v.(bool)
	}
	o := false
	switch t.Kind() {
	case reflect.Struct:
		if t == timeType {
			break
		}
		for field := range t.Fields() {
			if !field.IsExported() || opaque(field.Type) {
				o = true
				break
			}
		}
	case reflect.Array:
		o = opaque(t.Elem())
	}
	opaqueTypes.Store(t, o)
	return o
}
