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

package llminternal

import (
	"encoding/json"
	"errors"
	"math/big"
	"reflect"
	"strings"
	"testing"
)

func mustClone[M any](t *testing.T, value M) M {
	t.Helper()
	result, err := clone(value)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestClone(t *testing.T) {
	type testStruct struct {
		S  string
		I  int
		Sl []string
		M  map[string]string
		P  *int
		N  *testStruct
	}

	testData := func() *testStruct {
		return &testStruct{
			S:  "test",
			I:  123,
			Sl: []string{"a", "b"},
			M:  map[string]string{"k": "v"},
			P:  func() *int { i := 456; return &i }(),
			N: &testStruct{
				S: "nested",
			},
		}
	}

	check := func(t *testing.T, original, cloned *testStruct) {
		if !reflect.DeepEqual(original, cloned) {
			t.Errorf("clone() = %+v, want %+v", cloned, original)
		}

		// Modify cloned and check if original is affected
		cloned.Sl[0] = "c"
		cloned.M["k"] = "v2"
		*cloned.P = 789
		cloned.N.S = "nested2"

		if reflect.DeepEqual(original, cloned) {
			t.Errorf("clone() should not be affected by modifications to original")
		}
		if original.Sl[0] != "a" {
			t.Errorf("original slice was modified")
		}
		if original.M["k"] != "v" {
			t.Errorf("original map was modified")
		}
		if *original.P != 456 {
			t.Errorf("original pointer value was modified")
		}
		if original.N.S != "nested" {
			t.Errorf("original nested struct was modified")
		}
	}

	t.Run("pointer", func(t *testing.T) {
		original := testData()
		cloned := mustClone(t, original)
		check(t, original, cloned)
	})
	t.Run("value", func(t *testing.T) {
		original := testData()
		cloned := mustClone(t, *original)
		check(t, original, &cloned)
	})
	t.Run("interface", func(t *testing.T) {
		original := testData()
		cloned := mustClone(t, any(original))
		typed, ok := cloned.(*testStruct)
		if !ok {
			t.Fatalf("clone failed with interface: %v", cloned)
		}
		check(t, original, typed)
	})
}

func TestCloneNil(t *testing.T) {
	var original *int
	cloned := mustClone(t, original)
	if cloned != nil {
		t.Errorf("clone(nil) = %v, want nil", cloned)
	}
}

func TestCloneInterfaceFields(t *testing.T) {
	type record struct {
		Values []string
	}
	type payload struct {
		Map      any
		Slice    any
		Pointer  any
		Struct   any
		Array    any
		Nested   map[string]any
		MapArray [1]map[string]string
	}
	newPayload := func() payload {
		return payload{
			Map:      map[string]string{"key": "original"},
			Slice:    []string{"original"},
			Pointer:  &record{Values: []string{"original"}},
			Struct:   record{Values: []string{"original"}},
			Array:    [1][]string{{"original"}},
			Nested:   map[string]any{"items": []any{map[string]any{"key": "original"}}},
			MapArray: [1]map[string]string{{"key": "original"}},
		}
	}
	original := newPayload()
	copied := mustClone(t, original)
	if !reflect.DeepEqual(copied, original) {
		t.Fatalf("clone() = %#v, want %#v", copied, original)
	}
	copied.Map.(map[string]string)["key"] = "changed"
	copied.Slice.([]string)[0] = "changed"
	copied.Pointer.(*record).Values[0] = "changed"
	copied.Struct.(record).Values[0] = "changed"
	copied.Array.([1][]string)[0][0] = "changed"
	copied.Nested["items"].([]any)[0].(map[string]any)["key"] = "changed"
	copied.MapArray[0]["key"] = "changed"
	if want := newPayload(); !reflect.DeepEqual(original, want) {
		t.Errorf("mutating clone changed original: got %#v, want %#v", original, want)
	}
}

func TestCloneNilInterfaces(t *testing.T) {
	tests := []struct {
		name  string
		value any
	}{
		{name: "nil"},
		{name: "nil pointer", value: (*int)(nil)},
		{name: "nil map", value: map[string]any(nil)},
		{name: "nil slice", value: []any(nil)},
		{name: "empty map", value: map[string]any{}},
		{name: "empty slice", value: []any{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := mustClone(t, tt.value); !reflect.DeepEqual(got, tt.value) {
				t.Errorf("clone() = %#v (%T), want %#v (%T)", got, got, tt.value, tt.value)
			}
			original := struct{ Value any }{Value: tt.value}
			if got := mustClone(t, original); !reflect.DeepEqual(got, original) {
				t.Errorf("clone() = %#v, want %#v", got, original)
			}
		})
	}
}

func TestCloneUnexported(t *testing.T) {
	type testStructUnexported struct {
		s string
	}
	original := &testStructUnexported{s: "test"}
	got, err := clone(original)
	if !errors.Is(err, errCloneUnexported) {
		t.Errorf("clone() error = %v, want unexported field rejection", err)
	}
	if got != nil {
		t.Errorf("clone() returned partial result: %#v", got)
	}
}

type cloneTestError struct {
	Messages []string
}

func (e *cloneTestError) Error() string { return e.Messages[0] }

func TestCloneNonEmptyInterface(t *testing.T) {
	original := struct{ Err error }{Err: &cloneTestError{Messages: []string{"original"}}}
	copied := mustClone(t, original)
	if !reflect.DeepEqual(copied, original) {
		t.Fatalf("clone() = %#v, want %#v", copied, original)
	}
	copied.Err.(*cloneTestError).Messages[0] = "changed"
	if got := original.Err.Error(); got != "original" {
		t.Errorf("original error = %q, want original", got)
	}
}

func TestCloneRecursionLimit(t *testing.T) {
	cyclicMap := map[string]any{}
	cyclicMap["self"] = cyclicMap
	cyclicSlice := make([]any, 1)
	cyclicSlice[0] = cyclicSlice
	type node struct{ Next *node }
	cyclicPointer := &node{}
	cyclicPointer.Next = cyclicPointer
	var cyclicInterface any
	cyclicInterface = &cyclicInterface
	var nested any = "leaf"
	for i := 0; i <= maxCloneDepth; i++ {
		nested = []any{nested}
	}
	for name, value := range map[string]any{
		"map cycle":       cyclicMap,
		"slice cycle":     cyclicSlice,
		"pointer cycle":   cyclicPointer,
		"interface cycle": cyclicInterface,
		"excessive depth": nested,
	} {
		t.Run(name, func(t *testing.T) {
			got, err := clone(value)
			if !errors.Is(err, errCloneDepth) {
				t.Errorf("clone() error = %v, want depth limit", err)
			}
			if got != nil {
				t.Fatal("clone() returned partial result")
			}
		})
	}
}

func TestCloneNestedSharedValue(t *testing.T) {
	shared := map[string]any{"value": "original"}
	original := map[string]any{"left": shared, "right": shared}
	for range 130 {
		original = map[string]any{"nested": original}
	}
	copied := mustClone(t, original)
	if !reflect.DeepEqual(copied, original) {
		t.Fatal("clone changed a deeply nested value")
	}
	for range 130 {
		copied = copied["nested"].(map[string]any)
	}
	copied["left"].(map[string]any)["value"] = "changed"
	copied["right"].(map[string]any)["value"] = "changed"
	if shared["value"] != "original" {
		t.Errorf("original shared map changed: %v", shared)
	}
}

type cloneTestJSON struct {
	raw []byte
	err error
}

func (v *cloneTestJSON) MarshalJSON() ([]byte, error) { return v.raw, v.err }

func TestCloneOpaqueJSON(t *testing.T) {
	integer := big.NewInt(9007199254740993)
	schema := &struct {
		Type  string `json:"type"`
		cache []string
	}{Type: "string", cache: []string{"private"}}
	custom := &cloneTestJSON{raw: []byte(`{"minimum":9007199254740993}`)}
	for _, tt := range []struct {
		name   string
		value  any
		mutate func()
	}{
		{name: "big integer", value: integer, mutate: func() { integer.SetInt64(0) }},
		{name: "private cache", value: schema, mutate: func() { schema.Type = "number"; schema.cache[0] = "changed" }},
		{name: "custom marshaler", value: custom, mutate: func() { custom.raw[0] = ' ' }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			want, err := json.Marshal(tt.value)
			if err != nil {
				t.Fatal(err)
			}
			original := struct{ Value any }{Value: tt.value}
			first := mustClone(t, original)
			second := mustClone(t, original)
			raw, ok := first.Value.(json.RawMessage)
			if !ok {
				t.Fatalf("opaque value type = %T, want json.RawMessage", first.Value)
			}
			tt.mutate()
			if string(raw) != string(want) {
				t.Errorf("snapshot = %s, want %s", raw, want)
			}
			raw[0] = ' '
			got, err := json.Marshal(second.Value)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != string(want) {
				t.Errorf("other snapshot changed: got %s, want %s", got, want)
			}
		})
	}
}

func TestCloneOpaqueJSONErrors(t *testing.T) {
	t.Run("marshal failure", func(t *testing.T) {
		wantErr := errors.New("marshal failed")
		original := struct{ Value any }{Value: &cloneTestJSON{err: wantErr}}
		got, err := clone(original)
		if !errors.Is(err, wantErr) {
			t.Errorf("clone() error = %v, want %v", err, wantErr)
		}
		if got.Value != nil {
			t.Fatal("clone() returned a partial value")
		}
	})
	t.Run("nonempty interface", func(t *testing.T) {
		original := struct{ Value json.Marshaler }{Value: &cloneTestJSON{raw: []byte(`{}`)}}
		if _, err := clone(original); !errors.Is(err, errCloneUnexported) {
			t.Errorf("clone() error = %v, want unexported field rejection", err)
		}
	})
	t.Run("interface map key", func(t *testing.T) {
		key := struct{ value int }{value: 1}
		if _, err := clone(map[any]string{key: "value"}); err == nil || !strings.Contains(err.Error(), "not comparable") {
			t.Errorf("clone() error = %v, want non-comparable key error", err)
		}
	})
}
