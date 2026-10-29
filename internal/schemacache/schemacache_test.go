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

package schemacache

import (
	"encoding/json"
	"runtime"
	"testing"
	"time"
	"weak"

	"github.com/google/jsonschema-go/jsonschema"
)

func TestEncoded(t *testing.T) {
	s := &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{"q": {Type: "string"}}}
	Remember(s)
	want, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := Encoded(s); !ok || string(got) != string(want) {
		t.Errorf("Encoded(remembered) = %s, %v, want %s, true", got, ok, want)
	}
	for _, v := range []any{
		&jsonschema.Schema{Type: "object"}, // never remembered
		(*jsonschema.Schema)(nil),
		nil,
		map[string]any{"type": "object"},
		json.RawMessage(want),
	} {
		if got, ok := Encoded(v); ok {
			t.Errorf("Encoded(%#v) = %s, true, want false", v, got)
		}
	}
}

// TestEntryGoesWithItsSchema checks that a remembered schema's entry is
// deleted once the schema is collected, so tools built per request do not
// grow the map without bound.
func TestEntryGoesWithItsSchema(t *testing.T) {
	var key weak.Pointer[jsonschema.Schema]
	func() {
		s := &jsonschema.Schema{Type: "string", Description: "collected"}
		key = weak.Make(s)
		Remember(s)
	}()
	if _, ok := encoded.Load(key); !ok {
		t.Fatal("Remember did not record the schema")
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, ok := encoded.Load(key); !ok {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("entry still present 5s after its schema became unreachable")
		}
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
}
