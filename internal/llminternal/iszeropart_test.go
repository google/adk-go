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
	"reflect"
	"testing"

	"google.golang.org/genai"
)

// TestIsZeroPartCoversEveryField sets each field of genai.Part to a non-zero
// value in turn and checks isZeroPart notices, so a field added to genai.Part
// by a later release cannot make a part look empty.
func TestIsZeroPartCoversEveryField(t *testing.T) {
	if !isZeroPart(genai.Part{}) {
		t.Fatal("isZeroPart(genai.Part{}) = false, want true")
	}
	typ := reflect.TypeFor[genai.Part]()
	for i := range typ.NumField() {
		f := typ.Field(i)
		var p genai.Part
		fv := reflect.ValueOf(&p).Elem().Field(i)
		switch f.Type.Kind() {
		case reflect.Pointer:
			fv.Set(reflect.New(f.Type.Elem()))
		case reflect.Slice:
			fv.Set(reflect.MakeSlice(f.Type, 1, 1))
		case reflect.Map:
			fv.Set(reflect.MakeMap(f.Type))
		case reflect.String:
			fv.SetString("x")
		case reflect.Bool:
			fv.SetBool(true)
		default:
			t.Fatalf("field %s has kind %s, which this test does not know how to populate", f.Name, f.Type.Kind())
		}
		if isZeroPart(p) {
			t.Errorf("isZeroPart ignores field %s", f.Name)
		}
	}
}
