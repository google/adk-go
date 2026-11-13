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

package memory

import (
	"reflect"
	"time"
)

type memoryCloneReference struct {
	typeOf  reflect.Type
	pointer uintptr
	length  int
}

// cloneMemoryData follows the copy policy documented on InMemoryService.
func cloneMemoryData[T any](src T) T {
	var dst T
	reflect.ValueOf(&dst).Elem().Set(cloneMemoryValue(reflect.ValueOf(&src).Elem(), make(map[memoryCloneReference]reflect.Value)))
	return dst
}

func cloneMemoryValue(src reflect.Value, copies map[memoryCloneReference]reflect.Value) reflect.Value {
	var reference memoryCloneReference
	switch src.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if src.IsNil() {
			return src
		}
		reference = memoryCloneReference{typeOf: src.Type(), pointer: src.Pointer()}
		if src.Kind() == reflect.Slice {
			reference.length = src.Len()
		}
		if copy, ok := copies[reference]; ok {
			return copy
		}
	}

	switch src.Kind() {
	case reflect.Pointer:
		if memoryTypeIsOpaque(src.Type().Elem()) {
			return src
		}
		dst := reflect.New(src.Type().Elem()).Convert(src.Type())
		copies[reference] = dst
		dst.Elem().Set(cloneMemoryValue(src.Elem(), copies))
		return dst
	case reflect.Interface:
		if src.IsNil() {
			return src
		}
		return cloneMemoryValue(src.Elem(), copies)
	case reflect.Map:
		dst := reflect.MakeMapWithSize(src.Type(), src.Len())
		copies[reference] = dst
		for it := src.MapRange(); it.Next(); {
			dst.SetMapIndex(cloneMemoryValue(it.Key(), copies), cloneMemoryValue(it.Value(), copies))
		}
		return dst
	case reflect.Slice:
		if memoryTypeIsOpaque(src.Type().Elem()) {
			return src
		}
		dst := reflect.MakeSlice(src.Type(), src.Len(), src.Len())
		copies[reference] = dst
		// Attachments can contain large byte slices.
		if src.Type().Elem().Kind() == reflect.Uint8 {
			reflect.Copy(dst, src)
			return dst
		}
		for i := range src.Len() {
			dst.Index(i).Set(cloneMemoryValue(src.Index(i), copies))
		}
		return dst
	case reflect.Array:
		dst := reflect.New(src.Type()).Elem()
		for i := range src.Len() {
			dst.Index(i).Set(cloneMemoryValue(src.Index(i), copies))
		}
		return dst
	case reflect.Struct:
		dst := reflect.New(src.Type()).Elem()
		// Preserve unexported fields in value types such as time.Time.
		dst.Set(src)
		for i := range src.NumField() {
			if src.Type().Field(i).IsExported() {
				dst.Field(i).Set(cloneMemoryValue(src.Field(i), copies))
			}
		}
		return dst
	default:
		return src
	}
}

func memoryTypeIsOpaque(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Array:
		return memoryTypeIsOpaque(t.Elem())
	case reflect.Struct:
		if t == reflect.TypeFor[time.Time]() {
			return false
		}
		// Private state may forbid copying, as in strings.Builder or sync.Mutex.
		// Inspect inline fields; referenced objects are checked when cloned.
		for field := range t.Fields() {
			if !field.IsExported() || memoryTypeIsOpaque(field.Type) {
				return true
			}
		}
	}
	return false
}
