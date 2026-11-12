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
	"bytes"
	"reflect"
	"testing"

	"google.golang.org/genai"
)

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
		cloned := clone(original)
		check(t, original, cloned)
	})
	t.Run("value", func(t *testing.T) {
		original := testData()
		cloned := clone(*original)
		check(t, original, &cloned)
	})
	t.Run("interface", func(t *testing.T) {
		original := testData()
		cloned := clone(any(original))
		typed, ok := cloned.(*testStruct)
		if !ok {
			t.Fatalf("clone failed with interface: %v", cloned)
		}
		check(t, original, typed)
	})
}

func TestCloneNil(t *testing.T) {
	var original *int
	cloned := clone(original)
	if cloned != nil {
		t.Errorf("clone(nil) = %v, want nil", cloned)
	}
}

func TestCloneUnexported(t *testing.T) {
	type testStructUnexported struct {
		s string
	}
	original := &testStructUnexported{s: "test"}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("clone() did not panic on unexported field")
		}
	}()
	clone(original)
}

// TestCloneInlineDataInOneBlock pins the cost of copying inline data. Every
// model call copies the history, so copying a 1 MiB image one element at a
// time cost a million allocations per call.
func TestCloneInlineDataInOneBlock(t *testing.T) {
	data := bytes.Repeat([]byte{1, 2, 3, 4}, 1<<18)
	original := &genai.Content{Parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "image/png", Data: data}}}}

	cloned := clone(original)
	if !reflect.DeepEqual(original, cloned) {
		t.Fatal("clone() changed the inline data")
	}
	cloned.Parts[0].InlineData.Data[0] = 9
	if data[0] != 1 {
		t.Error("clone() shares the inline data bytes with the original")
	}

	if allocs := testing.AllocsPerRun(5, func() { clone(original) }); allocs > 100 {
		t.Errorf("clone() of 1 MiB of inline data made %v allocations, want at most 100", allocs)
	}
}

func BenchmarkCloneInlineData(b *testing.B) {
	c := &genai.Content{Parts: []*genai.Part{{InlineData: &genai.Blob{Data: make([]byte, 1<<20)}}}}
	b.SetBytes(1 << 20)
	b.ReportAllocs()
	for b.Loop() {
		clone(c)
	}
}
