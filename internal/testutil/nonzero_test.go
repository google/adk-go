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
	"fmt"
	"reflect"
	"testing"
)

func TestNonZero(t *testing.T) {
	type inner struct {
		_     int // unexported, so skipped in favor of Shown
		Shown uint16
	}
	type named string
	type firstUnfillable struct {
		S fmt.Stringer
		N int
	}
	for _, typ := range []reflect.Type{
		reflect.TypeFor[*int](),
		reflect.TypeFor[map[string]any](),
		reflect.TypeFor[[]byte](),
		reflect.TypeFor[chan int](),
		reflect.TypeFor[<-chan int](),
		reflect.TypeFor[func(int) (string, error)](),
		reflect.TypeFor[named](),
		reflect.TypeFor[bool](),
		reflect.TypeFor[int32](),
		reflect.TypeFor[uint64](),
		reflect.TypeFor[float32](),
		reflect.TypeFor[complex64](),
		reflect.TypeFor[any](),
		reflect.TypeFor[[2]int8](),
		reflect.TypeFor[inner](),
		reflect.TypeFor[firstUnfillable](),
	} {
		t.Run(typ.String(), func(t *testing.T) {
			v := NonZero(t, typ)
			if v.Type() != typ && !v.Type().AssignableTo(typ) {
				t.Fatalf("NonZero(%s) built a %s", typ, v.Type())
			}
			if v.IsZero() {
				t.Errorf("NonZero(%s) = %v, which reflect reports as zero", typ, v)
			}
		})
	}
}

// A type with no buildable non-zero value is reported, not returned as zero.
func TestNonZeroReportsWhatItCannotBuild(t *testing.T) {
	for _, typ := range []reflect.Type{
		reflect.TypeFor[fmt.Stringer](),
		reflect.TypeFor[[0]int](),
		reflect.TypeFor[struct{ hidden int }](),
		reflect.TypeFor[struct{ S fmt.Stringer }](),
	} {
		if v, ok := nonZero(typ); ok {
			t.Errorf("nonZero(%s) = %v, true; want false", typ, v)
		}
	}
}
