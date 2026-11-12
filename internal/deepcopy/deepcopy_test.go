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

package deepcopy

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"google.golang.org/genai"
)

// TestCopyEveryFieldOfPart walks genai.Part with reflection, fills every field
// reachable from it, and checks that the copy is equal and that no pointer,
// slice or map is shared with the original. A member added to a genai type
// fails here if Copy stops copying it.
func TestCopyEveryFieldOfPart(t *testing.T) {
	orig := &genai.Part{}
	fillValue(t, reflect.ValueOf(orig).Elem(), "Part", map[reflect.Type]bool{})
	checkNoSharedRefs(t, "Part", orig, Copy(orig), 20)
}

// TestCopyEveryFieldOfGenerateContentConfig is TestCopyEveryFieldOfPart for
// the per-request model config, which a model backend writes into.
func TestCopyEveryFieldOfGenerateContentConfig(t *testing.T) {
	orig := &genai.GenerateContentConfig{}
	fillValue(t, reflect.ValueOf(orig).Elem(), "GenerateContentConfig", map[reflect.Type]bool{})
	checkNoSharedRefs(t, "GenerateContentConfig", orig, Copy(orig), 20)
}

func checkNoSharedRefs[T any](t *testing.T, name string, orig, got T, minRefs int) {
	t.Helper()
	if !reflect.DeepEqual(orig, got) {
		t.Fatalf("Copy() changed the %s's contents", name)
	}
	seen := map[uintptr]string{}
	collectRefs(reflect.ValueOf(orig), name, seen, nil)
	collectRefs(reflect.ValueOf(got), name, nil, func(path string, addr uintptr) {
		if origPath, ok := seen[addr]; ok {
			t.Errorf("Copy() shares memory between the original at %s and the copy at %s", origPath, path)
		}
	})
	if len(seen) < minRefs {
		t.Errorf("walk found only %d references, the fill helper is not reaching the nested fields", len(seen))
	}
}

func TestCopyKeepsNilAndEmptyDistinct(t *testing.T) {
	for name, p := range map[string]*genai.Part{
		"nil members":   {Text: "x"},
		"empty members": {ThoughtSignature: []byte{}, PartMetadata: map[string]any{}, InlineData: &genai.Blob{Data: []byte{}}},
	} {
		if got := Copy(p); !reflect.DeepEqual(p, got) {
			t.Errorf("%s: Copy() = %#v, want %#v", name, got, p)
		}
	}
	if Copy((*genai.Part)(nil)) != nil {
		t.Error("Copy((*genai.Part)(nil)) != nil")
	}
	if Copy(any(nil)) != nil {
		t.Error("Copy(any(nil)) != nil")
	}
}

// TestCopyThroughEveryShape checks that a copy made through a pointer, a value
// or an interface writes back to nothing in the original.
func TestCopyThroughEveryShape(t *testing.T) {
	type node struct {
		S  string
		Sl []string
		M  map[string]string
		P  *int
		N  *node
		A  any
	}
	data := func() *node {
		i := 456
		return &node{
			S:  "test",
			Sl: []string{"a"},
			M:  map[string]string{"k": "v"},
			P:  &i,
			N:  &node{S: "nested"},
			A:  map[string]any{"k": []any{"v"}},
		}
	}
	mutate := func(n *node) {
		n.Sl[0] = "c"
		n.M["k"] = "v2"
		*n.P = 789
		n.N.S = "nested2"
		n.A.(map[string]any)["k"].([]any)[0] = "v2"
	}
	for name, copyOf := range map[string]func(*node) *node{
		"pointer":   func(n *node) *node { return Copy(n) },
		"value":     func(n *node) *node { c := Copy(*n); return &c },
		"interface": func(n *node) *node { return Copy(any(n)).(*node) },
	} {
		t.Run(name, func(t *testing.T) {
			orig := data()
			got := copyOf(orig)
			if !reflect.DeepEqual(orig, got) {
				t.Fatalf("Copy() = %+v, want %+v", got, orig)
			}
			mutate(got)
			if want := data(); !reflect.DeepEqual(orig, want) {
				t.Errorf("writing through the copy changed the original to %+v", orig)
			}
		})
	}
}

func TestCopyKeepsUnexportedFields(t *testing.T) {
	type withUnexported struct {
		When time.Time
		s    string
	}
	orig := &withUnexported{When: time.Date(2026, 1, 2, 3, 4, 5, 6, time.UTC), s: "x"}
	got := Copy(orig)
	if got == orig || !got.When.Equal(orig.When) || got.s != "x" {
		t.Errorf("Copy() = %+v, want an equal copy of %+v", got, orig)
	}
}

// TestCopyStopsAtDepthBound pins where the copy stops: PartMetadata sits two
// steps below the part and each nested map[string]any costs two more, so the
// first maxDepth/2 levels are copied and the rest stay shared.
func TestCopyStopsAtDepthBound(t *testing.T) {
	maps := make([]map[string]any, maxDepth)
	for i := range maps {
		maps[i] = map[string]any{"level": i}
		if i > 0 {
			maps[i-1]["next"] = maps[i]
		}
	}
	got := Copy(&genai.Part{Text: "x", PartMetadata: maps[0]})

	copied := 0
	m := got.PartMetadata
	for i := range maps {
		if sameMap(m, maps[i]) {
			break
		}
		copied++
		next, _ := m["next"].(map[string]any)
		m = next
	}
	if want := maxDepth / 2; copied != want {
		t.Errorf("Copy() copied %d levels of nested PartMetadata, want %d", copied, want)
	}
}

// TestCopyPreservesCycles checks that a map containing itself is copied as a
// map containing its copy, sharing nothing with the original.
func TestCopyPreservesCycles(t *testing.T) {
	m := map[string]any{}
	m["self"] = m
	got := Copy(&genai.Part{Text: "x", PartMetadata: m})

	if sameMap(got.PartMetadata, m) {
		t.Error("Copy() shares the top-level PartMetadata map")
	}
	inner, ok := got.PartMetadata["self"].(map[string]any)
	if !ok {
		t.Fatalf("Copy() PartMetadata[\"self\"] = %T, want map[string]any", got.PartMetadata["self"])
	}
	if !sameMap(inner, got.PartMetadata) {
		t.Error("Copy() did not copy PartMetadata[\"self\"] as a reference to the copied map")
	}
}

// TestCopyBranchingCycle checks that a cycle reached through several branches
// at every level is copied once rather than once per path. Without the memo
// the copy visits 2^maxDepth paths and does not finish.
func TestCopyBranchingCycle(t *testing.T) {
	type node struct{ L, R *node }
	n := &node{}
	n.L, n.R = n, n

	done := make(chan *node)
	go func() { done <- Copy(n) }()
	select {
	case got := <-done:
		if got == n || got.L != got || got.R != got {
			t.Errorf("Copy() did not copy the cycle as a cycle of the copy")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Copy() of a branching cycle did not finish")
	}
}

// TestCopyBranchingSliceCycle is TestCopyBranchingCycle for a cycle that runs
// through a slice rather than a pointer.
func TestCopyBranchingSliceCycle(t *testing.T) {
	s := make([]any, 2)
	s[0], s[1] = s, s

	done := make(chan []any)
	go func() { done <- Copy(s) }()
	select {
	case got := <-done:
		inner, ok := got[0].([]any)
		if !ok || &inner[0] != &got[0] || &got[0] == &s[0] {
			t.Errorf("Copy() did not copy the cycle as a cycle of the copy")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Copy() of a branching slice cycle did not finish")
	}
}

// TestCopyDoesNotReuseACopyCutShortByTheBound reaches one pointer first near
// the depth bound, where its copy stops before its map, and then near the top.
// The second path must get a full copy rather than the truncated one.
func TestCopyDoesNotReuseACopyCutShortByTheBound(t *testing.T) {
	type leaf struct{ M map[string]int }
	type link struct {
		Next *link
		Leaf *leaf
	}
	type root struct {
		Deep    *link
		Shallow *leaf
	}
	l := &leaf{M: map[string]int{"a": 1}}
	// The first link pointer sits at depth 2 and each Next adds 2, so after 30
	// links the Leaf pointer sits at 4+2*30 = maxDepth. That pointer is copied
	// but the leaf struct below it is not, so the copy made on this path shares
	// l.M.
	head := &link{Leaf: l}
	for range 30 {
		head = &link{Next: head}
	}
	got := Copy(&root{Deep: head, Shallow: l})

	got.Shallow.M["a"] = 2
	if l.M["a"] != 1 {
		t.Error("writing through the copy's shallow path changed the original's map")
	}
}

// TestCopyTellsAStructFromItsFirstField checks that a pointer to a struct and a
// pointer to its first field, which share an address, get separate copies of
// the right types.
func TestCopyTellsAStructFromItsFirstField(t *testing.T) {
	type inner struct{ N *int }
	type outer struct{ In inner }
	type both struct {
		Outer *outer
		Inner *inner
	}
	n := 1
	o := &outer{In: inner{N: &n}}
	got := Copy(&both{Outer: o, Inner: &o.In})
	if got.Outer == o || got.Inner == &o.In || *got.Inner.N != 1 {
		t.Errorf("Copy() = %+v, want copies of both pointers", got)
	}
}

// TestShareInterfaces checks that the option shares what an interface holds
// and still copies everything else, including the map that holds it.
func TestShareInterfaces(t *testing.T) {
	nested := map[string]any{"n": 1}
	orig := &genai.Content{Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{Name: "f", Args: map[string]any{"k": nested}}},
		{InlineData: &genai.Blob{Data: []byte("x"), DisplayName: "d"}},
	}}
	got := Copy(orig, ShareInterfaces())

	if !reflect.DeepEqual(orig, got) {
		t.Fatalf("Copy() = %+v, want %+v", got, orig)
	}
	if got.Parts[0] == orig.Parts[0] || got.Parts[0].FunctionCall == orig.Parts[0].FunctionCall {
		t.Error("Copy() shares a part or its FunctionCall")
	}
	if sameMap(got.Parts[0].FunctionCall.Args, orig.Parts[0].FunctionCall.Args) {
		t.Error("Copy() shares the Args map")
	}
	if !sameMap(got.Parts[0].FunctionCall.Args["k"].(map[string]any), nested) {
		t.Error("Copy() copied the value held in Args, want it shared")
	}
	got.Parts[1].InlineData.DisplayName = ""
	got.Parts[1].InlineData.Data[0] = 'y'
	if b := orig.Parts[1].InlineData; b.DisplayName != "d" || b.Data[0] != 'x' {
		t.Error("writing through the copy's blob changed the original")
	}
}

// TestCopyPreservesSharing checks that two references to one value in the
// original are two references to one copy.
func TestCopyPreservesSharing(t *testing.T) {
	shared := &genai.Blob{Data: []byte("x")}
	orig := []*genai.Part{{InlineData: shared}, {InlineData: shared}}
	got := Copy(orig)
	if got[0].InlineData == shared {
		t.Fatal("Copy() shares the blob with the original")
	}
	if got[0].InlineData != got[1].InlineData {
		t.Error("Copy() made two copies of one blob")
	}
}

func BenchmarkCopyInlineData(b *testing.B) {
	p := &genai.Part{InlineData: &genai.Blob{MIMEType: "application/octet-stream", Data: make([]byte, 10<<20)}}
	b.SetBytes(int64(len(p.InlineData.Data)))
	for b.Loop() {
		Copy(p)
	}
}

func sameMap(a, b map[string]any) bool {
	return reflect.ValueOf(a).UnsafePointer() == reflect.ValueOf(b).UnsafePointer()
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
	case reflect.Func:
		// Copy shares funcs, so there is nothing to check; leave it nil.
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
