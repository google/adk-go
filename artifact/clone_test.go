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
	"fmt"
	"reflect"
	"testing"

	"google.golang.org/genai"
)

// TestClonePartCopiesEveryField walks genai.Part with reflection, fills every
// field reachable from it, and checks that the copy is equal and that no
// pointer, slice or map is shared with the original. A member added to a genai
// type fails here if clonePart stops copying it.
func TestClonePartCopiesEveryField(t *testing.T) {
	orig := &genai.Part{}
	fillValue(t, reflect.ValueOf(orig).Elem(), "Part", map[reflect.Type]bool{})

	got := clonePart(orig)
	if !reflect.DeepEqual(orig, got) {
		t.Fatalf("clonePart() changed the part's contents")
	}
	seen := map[uintptr]string{}
	collectRefs(reflect.ValueOf(orig), "Part", seen, nil)
	collectRefs(reflect.ValueOf(got), "Part", nil, func(path string, addr uintptr) {
		if origPath, ok := seen[addr]; ok {
			t.Errorf("clonePart() shares memory between the original at %s and the copy at %s", origPath, path)
		}
	})
	if len(seen) < 20 {
		t.Errorf("walk found only %d references, the fill helper is not reaching the nested fields", len(seen))
	}
}

func TestClonePartKeepsNilAndEmptyDistinct(t *testing.T) {
	for name, p := range map[string]*genai.Part{
		"nil members":   {Text: "x"},
		"empty members": {ThoughtSignature: []byte{}, PartMetadata: map[string]any{}, InlineData: &genai.Blob{Data: []byte{}}},
	} {
		if got := clonePart(p); !reflect.DeepEqual(p, got) {
			t.Errorf("%s: clonePart() = %#v, want %#v", name, got, p)
		}
	}
	if clonePart(nil) != nil {
		t.Error("clonePart(nil) != nil")
	}
}

// fillValue sets every exported field below v to a non-zero value, with fresh
// memory for each pointer, slice, map and interface. Types already on the stack
// are left nil so a recursive type terminates.
func fillValue(t *testing.T, v reflect.Value, path string, stack map[reflect.Type]bool) {
	t.Helper()
	switch v.Kind() {
	case reflect.String:
		v.SetString("s")
	case reflect.Bool:
		v.SetBool(true)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v.SetInt(3)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		v.SetUint(3)
	case reflect.Float32, reflect.Float64:
		v.SetFloat(1.5)
	case reflect.Pointer:
		if stack[v.Type().Elem()] {
			return
		}
		stack[v.Type().Elem()] = true
		defer delete(stack, v.Type().Elem())
		v.Set(reflect.New(v.Type().Elem()))
		fillValue(t, v.Elem(), path, stack)
	case reflect.Slice:
		v.Set(reflect.MakeSlice(v.Type(), 2, 2))
		for i := range 2 {
			fillValue(t, v.Index(i), fmt.Sprintf("%s[%d]", path, i), stack)
		}
	case reflect.Map:
		v.Set(reflect.MakeMap(v.Type()))
		k := reflect.New(v.Type().Key()).Elem()
		fillValue(t, k, path+"{key}", stack)
		e := reflect.New(v.Type().Elem()).Elem()
		fillValue(t, e, path+"{value}", stack)
		v.SetMapIndex(k, e)
	case reflect.Interface:
		if v.NumMethod() != 0 {
			t.Fatalf("%s: cannot fill non-empty interface %s", path, v.Type())
		}
		v.Set(reflect.ValueOf(map[string]any{"n": []any{map[string]any{"k": "v"}}, "s": "t"}))
	case reflect.Struct:
		for i := range v.NumField() {
			if v.Type().Field(i).IsExported() {
				fillValue(t, v.Field(i), path+"."+v.Type().Field(i).Name, stack)
			}
		}
	default:
		t.Fatalf("%s: fill does not handle kind %s", path, v.Kind())
	}
}

// collectRefs visits the address of every pointer, non-empty slice backing
// array and map below v. Zero-size pointees are skipped because Go may point
// them all at the same address.
func collectRefs(v reflect.Value, path string, record map[uintptr]string, check func(path string, addr uintptr)) {
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() || v.Type().Elem().Size() == 0 {
			return
		}
		mark(v.Pointer(), path, record, check)
		collectRefs(v.Elem(), path, record, check)
	case reflect.Interface:
		if !v.IsNil() {
			collectRefs(v.Elem(), path, record, check)
		}
	case reflect.Map:
		if v.IsNil() {
			return
		}
		mark(v.Pointer(), path, record, check)
		for it := v.MapRange(); it.Next(); {
			collectRefs(it.Value(), fmt.Sprintf("%s[%v]", path, it.Key()), record, check)
		}
	case reflect.Slice:
		if v.Len() == 0 {
			return
		}
		mark(v.Pointer(), path, record, check)
		for i := range v.Len() {
			collectRefs(v.Index(i), fmt.Sprintf("%s[%d]", path, i), record, check)
		}
	case reflect.Struct:
		for i := range v.NumField() {
			if v.Type().Field(i).IsExported() {
				collectRefs(v.Field(i), path+"."+v.Type().Field(i).Name, record, check)
			}
		}
	}
}

func mark(addr uintptr, path string, record map[uintptr]string, check func(path string, addr uintptr)) {
	if record != nil {
		record[addr] = path
	}
	if check != nil {
		check(path, addr)
	}
}
