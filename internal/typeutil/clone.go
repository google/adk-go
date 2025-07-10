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

package typeutil

import (
	"fmt"
	"reflect"

	"google.golang.org/genai"
)

// Clone returns a deep copy of the src.
// This is not generally usable. Only selected types typeutil.cloneable
// are allowed to use.
// If the type is a struct type with unexported fields, or contains
// unhandled types (function, array, channel, or embedded type),
// this function will panic.
func Clone[M cloneable](src M) M {
	return clone(src)
}

// clonable are the types explicitly allowed to use the Clone.
type cloneable interface {
	*genai.GenerateContentConfig | []*genai.Content | *genai.Content
}

// clone returns a deep copy of the src.
// NOTE: this does not work for types with unexported fields.
func clone[M any](src M) M {
	val := reflect.ValueOf(src)

	// Handle nil pointers
	if val.Kind() == reflect.Ptr && val.IsNil() {
		var zero M
		return zero
	}

	pointerLevel := 0
	// Dereference all pointers, counting the levels.
	// We loop until we find the underlying, non-pointer value.
	for val.Kind() == reflect.Ptr {
		// If we encounter a nil pointer at any level, we can't go deeper.
		// We return a zero value of the original type M.
		if val.IsNil() {
			var zero M
			return zero
		}
		val = val.Elem()
		pointerLevel++
	}

	// Create a new instance of the underlying, non-pointer type.
	newVal := reflect.New(val.Type()).Elem()

	// Recursively copy the base value.
	deepCopy(newVal, val)

	// Re-wrap the copied value with the original number of pointers.
	finalVal := newVal
	for i := 0; i < pointerLevel; i++ {
		// Create a new pointer to the current value.
		pv := reflect.New(finalVal.Type())
		// Set the new pointer's element to our current value.
		pv.Elem().Set(finalVal)
		// Update our value to be the new pointer.
		finalVal = pv
	}

	return finalVal.Interface().(M)
}

// deepCopy copies src to dst using reflect.
func deepCopy(dst, src reflect.Value) {
	switch src.Kind() {
	case reflect.Struct:
		t := src.Type()
		for i := range src.NumField() {
			if f := t.Field(i); !f.IsExported() {
				panic(fmt.Sprintf("deepCopy: unexported field %q in type %q", f.Name, t.Name()))
			}
			// Create a copy of the field and set it on the destination struct
			fieldCopy := reflect.New(src.Field(i).Type()).Elem()
			deepCopy(fieldCopy, src.Field(i))
			dst.Field(i).Set(fieldCopy)
		}
	case reflect.Slice:
		if src.IsNil() {
			return
		}
		dst.Set(reflect.MakeSlice(src.Type(), src.Len(), src.Cap()))
		for i := range src.Len() {
			// Create a copy of each element and set it in the new slice
			elemCopy := reflect.New(src.Index(i).Type()).Elem()
			deepCopy(elemCopy, src.Index(i))
			dst.Index(i).Set(elemCopy)
		}
	case reflect.Array:
		// Unlike a slice, the destination array is already allocated.
		// We just need to iterate and copy each element.
		for i := range src.Len() {
			elemCopy := reflect.New(src.Index(i).Type()).Elem()
			deepCopy(elemCopy, src.Index(i))
			dst.Index(i).Set(elemCopy)
		}
	case reflect.Map:
		if src.IsNil() {
			return
		}
		dst.Set(reflect.MakeMap(src.Type()))
		for _, key := range src.MapKeys() {
			// Create copies of the key and value and set them in the new map
			keyCopy := reflect.New(key.Type()).Elem()
			deepCopy(keyCopy, key)
			m := src.MapIndex(key)
			valCopy := reflect.New(m.Type()).Elem()
			deepCopy(valCopy, m)
			dst.SetMapIndex(keyCopy, valCopy)
		}
	case reflect.Interface:
		if src.IsNil() {
			return
		}
		elem := src.Elem()
		newVal := reflect.New(elem.Type()).Elem()
		deepCopy(newVal, elem)
		dst.Set(newVal)
	case reflect.Ptr:
		if src.IsNil() {
			return
		}
		// Create a new pointer and deep copy the underlying value
		newPtr := reflect.New(src.Elem().Type())
		deepCopy(newPtr.Elem(), src.Elem())
		dst.Set(newPtr)
	case reflect.Chan, reflect.Func:
		panic(fmt.Sprintf("unsupported type: %v", src.Type()))
	default:
		// For basic types, direct assignment is sufficient
		dst.Set(src)
	}
}
