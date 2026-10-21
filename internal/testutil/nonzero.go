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

package testutil

import (
	"reflect"
	"testing"
)

// NonZero builds a value of typ that reflect reports as non-zero, for tests
// that walk a struct's fields and set each in turn. It fails t only for a type
// it has no such value for: an interface with methods, an unsafe pointer, or a
// struct or array with nothing it can fill.
func NonZero(t testing.TB, typ reflect.Type) reflect.Value {
	t.Helper()
	v, ok := nonZero(typ)
	if !ok {
		t.Fatalf("testutil.NonZero cannot build a non-zero %s", typ)
	}
	return v
}

func nonZero(typ reflect.Type) (reflect.Value, bool) {
	switch typ.Kind() {
	case reflect.Pointer:
		return reflect.New(typ.Elem()), true
	case reflect.Map:
		return reflect.MakeMap(typ), true
	case reflect.Slice:
		return reflect.MakeSlice(typ, 1, 1), true
	case reflect.Chan:
		// MakeChan refuses a one-way type, which a two-way channel still fills.
		return reflect.MakeChan(reflect.ChanOf(reflect.BothDir, typ.Elem()), 0), true
	case reflect.Func:
		return reflect.MakeFunc(typ, func([]reflect.Value) []reflect.Value {
			out := make([]reflect.Value, typ.NumOut())
			for i := range out {
				out[i] = reflect.Zero(typ.Out(i))
			}
			return out
		}), true
	case reflect.String:
		return reflect.ValueOf("x").Convert(typ), true
	case reflect.Bool:
		return reflect.ValueOf(true).Convert(typ), true
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return reflect.ValueOf(1).Convert(typ), true
	case reflect.Float32, reflect.Float64:
		return reflect.ValueOf(1.0).Convert(typ), true
	case reflect.Complex64, reflect.Complex128:
		return reflect.ValueOf(complex(1, 0)).Convert(typ), true
	case reflect.Interface:
		if typ.NumMethod() == 0 {
			return reflect.ValueOf(true), true
		}
	case reflect.Array:
		if elem, ok := nonZero(typ.Elem()); ok && typ.Len() > 0 {
			arr := reflect.New(typ).Elem()
			arr.Index(0).Set(elem)
			return arr, true
		}
	case reflect.Struct:
		// One set exported field is enough to make the whole struct non-zero.
		for i := range typ.NumField() {
			if !typ.Field(i).IsExported() {
				continue
			}
			if field, ok := nonZero(typ.Field(i).Type); ok {
				s := reflect.New(typ).Elem()
				s.Field(i).Set(field)
				return s, true
			}
		}
	}
	return reflect.Value{}, false
}
