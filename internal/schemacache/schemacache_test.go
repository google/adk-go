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

// TestRemember_UnencodableSchema checks that a schema json.Marshal rejects is
// not remembered, so the adapter's own encoding reports the error.
func TestRemember_UnencodableSchema(t *testing.T) {
	s := &jsonschema.Schema{Type: "object", Types: []string{"string"}}
	if _, err := json.Marshal(s); err == nil {
		t.Fatal("json.Marshal accepted a schema with both Type and Types, the test needs another unencodable schema")
	}
	Remember(s)
	if _, ok := Encoded(s); ok {
		t.Error("Encoded(unencodable) = true, want false")
	}
}

// TestRemember_Twice checks that remembering a schema again records its
// current encoding: a schema changed between two tools built from it is sent
// as the second tool declares it.
func TestRemember_Twice(t *testing.T) {
	s := &jsonschema.Schema{Type: "string", Description: "first"}
	Remember(s)
	s.Description = "second"
	Remember(s)
	got, _ := Encoded(s)
	if want := `{"type":"string","description":"second"}`; string(got) != want {
		t.Errorf("Encoded after a second Remember = %s, want %s", got, want)
	}
}

// TestEncoded_CannotCorruptTheCache checks that appending to a returned
// encoding leaves the remembered bytes alone.
func TestEncoded_CannotCorruptTheCache(t *testing.T) {
	s := &jsonschema.Schema{Type: "string"}
	Remember(s)
	got, _ := Encoded(s)
	want := string(got)
	_ = append(got, "garbage"...)
	if again, _ := Encoded(s); string(again[:cap(again)]) != want {
		t.Errorf("remembered bytes became %q after an append, want %q", again[:cap(again)], want)
	}
}

// TestEntriesGoWithTheirSchemas checks that every remembered schema's entry is
// deleted once the schema is collected, so tools built per request do not
// grow the map without bound.
func TestEntriesGoWithTheirSchemas(t *testing.T) {
	const n = 1000
	// Built in a closure so no stack slot keeps a schema reachable, even
	// without optimizations.
	keys := func() []weak.Pointer[jsonschema.Schema] {
		keys := make([]weak.Pointer[jsonschema.Schema], n)
		schemas := make([]*jsonschema.Schema, n)
		for i := range schemas {
			schemas[i] = &jsonschema.Schema{Type: "string", Description: "collected"}
			keys[i] = weak.Make(schemas[i])
			Remember(schemas[i])
		}
		runtime.GC()
		for i, s := range schemas {
			if _, ok := Encoded(s); !ok {
				t.Fatalf("schema %d, still reachable, lost its entry", i)
			}
		}
		return keys
	}()
	left := func() int {
		n := 0
		for _, k := range keys {
			if _, ok := encoded.Load(k); ok {
				n++
			}
		}
		return n
	}
	deadline := time.Now().Add(10 * time.Second)
	for left() > 0 {
		if time.Now().After(deadline) {
			t.Fatalf("%d of %d entries still present 10s after their schemas became unreachable", left(), n)
		}
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
}
