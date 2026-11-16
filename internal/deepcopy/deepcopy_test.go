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

package deepcopy

import (
	"errors"
	"math/big"
	"reflect"
	"strings"
	"testing"
	"time"
)

type leaf struct {
	Text string
	Tags []string
}

type tree struct {
	Name  string
	Ptr   *leaf
	Map   map[string]any
	Slice []*leaf
	Any   any
	Bytes []byte
	Array [2]*leaf
	When  time.Time
}

// withSecret has an unexported field, so Of keeps it as it is.
type withSecret struct {
	Name   string
	secret *leaf
}

func TestOf_CopySharesNoMemoryWithOriginal(t *testing.T) {
	newTree := func() *tree {
		return &tree{
			Name:  "n",
			Ptr:   &leaf{Text: "p", Tags: []string{"a"}},
			Map:   map[string]any{"nested": map[string]any{"k": "v"}, "list": []any{"x"}},
			Slice: []*leaf{{Text: "s"}},
			Any:   map[string]any{"k": 1},
			Bytes: []byte{1, 2},
			Array: [2]*leaf{{Text: "a0"}, nil},
			When:  time.Date(2024, time.January, 2, 3, 4, 5, 0, time.UTC),
		}
	}
	orig := newTree()

	got := Of(orig)
	orig.Ptr.Text = "changed"
	orig.Ptr.Tags[0] = "changed"
	orig.Map["nested"].(map[string]any)["k"] = "changed"
	orig.Map["list"].([]any)[0] = "changed"
	orig.Slice[0].Text = "changed"
	orig.Any.(map[string]any)["k"] = 2
	orig.Bytes[0] = 9
	orig.Array[0].Text = "changed"

	if want := newTree(); !reflect.DeepEqual(got, want) {
		t.Errorf("Of() = %+v after the original changed, want %+v", got, want)
	}
}

func TestOf_KeepsNilAndEmptyDistinct(t *testing.T) {
	tests := []struct {
		name string
		in   tree
	}{
		{name: "nil members", in: tree{Name: "x"}},
		{name: "empty members", in: tree{Map: map[string]any{}, Slice: []*leaf{}, Bytes: []byte{}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Of(tt.in); !reflect.DeepEqual(got, tt.in) {
				t.Errorf("Of() = %#v, want %#v", got, tt.in)
			}
		})
	}
}

var errSentinel = errors.New("sentinel")

// A struct with an unexported field cannot be copied field by field without
// breaking it, so Of keeps it, and a pointer to it, as they are.
func TestOf_KeepsValuesWithUnexportedFields(t *testing.T) {
	secret := &leaf{Text: "hidden"}
	byPointer := &withSecret{Name: "p", secret: secret}
	var builder strings.Builder
	builder.WriteString("x")
	number := big.NewInt(-5)
	loc := time.FixedZone("UTC+1", 3600)
	when := time.Date(2024, time.January, 2, 3, 4, 5, 6, loc)
	orig := map[string]any{
		"pointer": byPointer,
		"value":   withSecret{Name: "v", secret: secret},
		"builder": &builder,
		"number":  number,
		"err":     errSentinel,
		"when":    when,
	}

	got := Of(orig)

	if got["pointer"] != byPointer {
		t.Errorf("Of() [\"pointer\"] = %p, want the original %p", got["pointer"], byPointer)
	}
	if v, ok := got["value"].(withSecret); !ok || v.secret != secret {
		t.Errorf("Of() [\"value\"] = %+v, want the original withSecret value", got["value"])
	}
	if got["builder"] != &builder {
		t.Errorf("Of() [\"builder\"] = %p, want the original %p", got["builder"], &builder)
	}
	if got["number"] != number {
		t.Errorf("Of() [\"number\"] = %p, want the original %p", got["number"], number)
	}
	if err, ok := got["err"].(error); !ok || !errors.Is(err, errSentinel) {
		t.Errorf("Of() [\"err\"] = %v, want an error that is errSentinel", got["err"])
	}
	if w, ok := got["when"].(time.Time); !ok || !w.Equal(when) || w.Location() != loc {
		t.Errorf("Of() [\"when\"] = %v, want %v", got["when"], when)
	}
}

// holder has only exported fields, but its Builder holds unexported ones by
// value, so copying a holder copies the Builder too.
type holder struct {
	B strings.Builder
	M map[string]string
}

// A struct that holds a struct with unexported fields by value cannot be
// copied either, nor can a slice of them: Of keeps them as they are.
func TestOf_KeepsValuesThatHoldUnexportedFieldsByValue(t *testing.T) {
	byPointer := &holder{M: map[string]string{"k": "v"}}
	byPointer.B.WriteString("x")
	slice := make([]strings.Builder, 1)
	slice[0].WriteString("x")
	orig := map[string]any{"pointer": byPointer, "slice": slice}

	got := Of(orig)

	if got["pointer"] != byPointer {
		t.Errorf("Of() [\"pointer\"] = %p, want the original %p", got["pointer"], byPointer)
	}
	if s, ok := got["slice"].([]strings.Builder); !ok || &s[0] != &slice[0] {
		t.Errorf("Of() [\"slice\"] is a copy, want the original slice")
	}
	byPointer.B.WriteString("y")
	if s := byPointer.B.String(); s != "xy" {
		t.Errorf("holder.B.String() = %q, want %q", s, "xy")
	}
}

// time.Time has unexported fields but is a value type: Of copies a struct that
// holds one.
func TestOf_CopiesStructThatHoldsTime(t *testing.T) {
	type stamped struct {
		When time.Time
		Tags []string
	}
	when := time.Date(2024, time.January, 2, 3, 4, 5, 0, time.UTC)
	orig := &stamped{When: when, Tags: []string{"a"}}

	got := Of(orig)
	orig.Tags[0] = "changed"

	if got == orig || !got.When.Equal(when) || got.Tags[0] != "a" {
		t.Errorf("Of() = %+v, want a copy with When %v and Tags [a]", got, when)
	}
}

func TestOf_InterfaceTypedValue(t *testing.T) {
	orig := map[string]any{"k": []any{"v"}}

	got := Of[any](orig)
	orig["k"].([]any)[0] = "changed"

	if want := map[string]any{"k": []any{"v"}}; !reflect.DeepEqual(got, want) {
		t.Errorf("Of[any]() = %v after the original changed, want %v", got, want)
	}
	if got := Of[any](nil); got != nil {
		t.Errorf("Of[any](nil) = %v, want nil", got)
	}
	if got := Of[*tree](nil); got != nil {
		t.Errorf("Of[*tree](nil) = %v, want nil", got)
	}
}

// Funcs and channels cannot be copied. Of keeps them rather than failing on
// a value that holds one.
func TestOf_KeepsFuncAndChannel(t *testing.T) {
	ch := make(chan int)
	called := false
	orig := map[string]any{"f": func() { called = true }, "ch": ch}

	got := Of(orig)

	f, ok := got["f"].(func())
	if !ok {
		t.Fatalf("Of() [\"f\"] = %T, want func()", got["f"])
	}
	f()
	if !called {
		t.Error("Of() [\"f\"] is not the original func")
	}
	if got["ch"] != ch {
		t.Error("Of() [\"ch\"] is not the original channel")
	}
}

// Map keys are compared by value, so Of keeps them as they are: a pointer key
// still finds its entry in the copy.
func TestOf_KeepsMapKeys(t *testing.T) {
	key := &leaf{Text: "k"}
	orig := map[*leaf]int{key: 1}

	got := Of(orig)

	if v, ok := got[key]; !ok || v != 1 {
		t.Errorf("Of() [key] = %v, %v; want 1, true", v, ok)
	}
}

type node struct {
	Name     string
	Parent   *node
	Children []*node
}

// A pointer, map or slice reached more than once is copied once, so the copy
// has the original's cycles and shared references, and copying a value with
// a cycle that branches takes time in proportion to its size.
func TestOf_CopiesSharedReferencesOnce(t *testing.T) {
	t.Run("map that holds itself twice", func(t *testing.T) {
		m := map[string]any{}
		m["a"], m["b"] = m, m

		got := Of(m)

		if reflect.ValueOf(got).UnsafePointer() == reflect.ValueOf(m).UnsafePointer() {
			t.Fatal("Of() returned the original map")
		}
		for _, k := range []string{"a", "b"} {
			inner, ok := got[k].(map[string]any)
			if !ok || reflect.ValueOf(inner).UnsafePointer() != reflect.ValueOf(got).UnsafePointer() {
				t.Errorf("Of() [%q] is not the copy itself", k)
			}
		}
	})
	t.Run("slice that holds itself", func(t *testing.T) {
		s := []any{nil}
		s[0] = s

		got := Of(s)

		inner, ok := got[0].([]any)
		if !ok || reflect.ValueOf(inner).UnsafePointer() != reflect.ValueOf(got).UnsafePointer() {
			t.Error("Of() [0] is not the copy itself")
		}
		if reflect.ValueOf(got).UnsafePointer() == reflect.ValueOf(s).UnsafePointer() {
			t.Error("Of() returned the original slice")
		}
	})
	t.Run("children that point to their parent", func(t *testing.T) {
		root := &node{Name: "root"}
		for range 12 {
			root.Children = append(root.Children, &node{Name: "child", Parent: root})
		}

		got := Of(root)

		if got == root {
			t.Fatal("Of() returned the original root")
		}
		for i, child := range got.Children {
			if child == root.Children[i] || child.Parent != got {
				t.Errorf("Of() child %d = %p with Parent %p, want a copy whose Parent is the copied root %p", i, child, child.Parent, got)
			}
		}
	})
	t.Run("slices of one array with different lengths", func(t *testing.T) {
		type twoSlices struct {
			Short, Long []string
		}
		arr := []string{"a", "b", "c"}

		got := Of(twoSlices{Short: arr[:1], Long: arr[:3]})

		want := twoSlices{Short: []string{"a"}, Long: []string{"a", "b", "c"}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("Of() = %v, want %v", got, want)
		}
	})
	t.Run("two fields that point to one value", func(t *testing.T) {
		shared := &leaf{Text: "s"}
		orig := tree{Ptr: shared, Array: [2]*leaf{shared, shared}}

		got := Of(orig)

		if got.Ptr == shared || got.Array[0] != got.Ptr || got.Array[1] != got.Ptr {
			t.Errorf("Of() = %p, %p, %p; want one new pointer three times", got.Ptr, got.Array[0], got.Array[1])
		}
	})
}

type link struct {
	Next *link
}

// Of stops at a depth bound, so a value nested deeper than any real one keeps
// its deepest part as it is instead of exhausting the stack.
func TestOf_KeepsValuesBelowDepthBound(t *testing.T) {
	head := &link{}
	cur := head
	for range maxDepth {
		cur.Next = &link{}
		cur = cur.Next
	}

	got := Of(head)

	if got == head {
		t.Fatal("Of() returned the original head")
	}
	o, c := head, got
	for o != nil && o != c {
		o, c = o.Next, c.Next
	}
	if o == nil {
		t.Error("Of() copied every link, want the links below the depth bound kept")
	}
}
