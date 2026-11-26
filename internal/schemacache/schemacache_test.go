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

// TestRemember_Twice checks that remembering a schema again keeps the first
// encoding, the one a resolved schema was frozen with.
func TestRemember_Twice(t *testing.T) {
	s := &jsonschema.Schema{Type: "string", Description: "first"}
	Remember(s)
	s.Description = "second"
	Remember(s)
	got, _ := Encoded(s)
	if want := `{"type":"string","description":"first"}`; string(got) != want {
		t.Errorf("Encoded after a second Remember = %s, want %s", got, want)
	}
}

func entries() int {
	n := 0
	encoded.Range(func(any, any) bool { n++; return true })
	return n
}

// TestEntriesGoWithTheirSchemas checks that remembered schemas' entries are
// deleted once the schemas are collected, so tools built per request do not
// grow the map without bound.
func TestEntriesGoWithTheirSchemas(t *testing.T) {
	const n = 1000
	before := entries()
	schemas := make([]*jsonschema.Schema, n)
	for i := range schemas {
		schemas[i] = &jsonschema.Schema{Type: "string", Description: "collected"}
		Remember(schemas[i])
	}
	runtime.GC()
	for i, s := range schemas {
		if _, ok := Encoded(s); !ok {
			t.Fatalf("schema %d, still reachable, lost its entry", i)
		}
	}
	runtime.KeepAlive(schemas)
	deadline := time.Now().Add(10 * time.Second)
	for entries() >= before+n/10 {
		if time.Now().After(deadline) {
			t.Fatalf("map still holds %d entries 10s after %d schemas became unreachable, started with %d", entries(), n, before)
		}
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
}
