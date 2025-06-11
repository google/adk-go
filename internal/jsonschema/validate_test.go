// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonschema

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// The test for validation uses the official test suite, expressed as a set of JSON files.
// Each file is an array of group objects.

// A testGroup consists of a schema and some tests on it.
type testGroup struct {
	Description string
	Schema      *Schema
	Tests       []test
}

// A test consists of a JSON instance to be validated and the expected result.
type test struct {
	Description string
	Data        any
	Valid       bool
}

func TestValidate(t *testing.T) {
	files, err := filepath.Glob(filepath.FromSlash("testdata/draft2020-12/*.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatal("no files")
	}
	for _, file := range files {
		base := filepath.Base(file)
		t.Run(base, func(t *testing.T) {
			data, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			var groups []testGroup
			if err := json.Unmarshal(data, &groups); err != nil {
				t.Fatal(err)
			}
			for _, g := range groups {
				t.Run(g.Description, func(t *testing.T) {
					rs, err := g.Schema.Resolve(&ResolveOptions{Loader: loadRemote})
					if err != nil {
						t.Fatal(err)
					}
					for _, test := range g.Tests {
						t.Run(test.Description, func(t *testing.T) {
							err = rs.Validate(test.Data)
							if err != nil && test.Valid {
								t.Errorf("wanted success, but failed with: %v", err)
							}
							if err == nil && !test.Valid {
								t.Error("succeeded but wanted failure")
							}
							if t.Failed() {
								t.Errorf("schema: %s", g.Schema.json())
								t.Fatalf("instance: %v (%[1]T)", test.Data)
							}
						})
					}
				})
			}
		})
	}
}

func TestValidateErrors(t *testing.T) {
	schema := &Schema{
		PrefixItems: []*Schema{{Contains: &Schema{Type: "integer"}}},
	}
	rs, err := schema.Resolve(nil)
	if err != nil {
		t.Fatal(err)
	}
	err = rs.Validate([]any{[]any{"1"}})
	want := "prefixItems/0"
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("error:\n%s\ndoes not contain %q", err, want)
	}
}

func TestValidateDefaults(t *testing.T) {
	s := &Schema{
		Properties: map[string]*Schema{
			"a": {Type: "integer", Default: mustMarshal(1)},
			"b": {Type: "string", Default: mustMarshal("s")},
		},
		Default: mustMarshal(map[string]any{"a": 1, "b": "two"}),
	}
	if _, err := s.Resolve(&ResolveOptions{ValidateDefaults: true}); err != nil {
		t.Fatal(err)
	}

	s = &Schema{
		Properties: map[string]*Schema{
			"a": {Type: "integer", Default: mustMarshal(3)},
			"b": {Type: "string", Default: mustMarshal("s")},
		},
		Default: mustMarshal(map[string]any{"a": 1, "b": 2}),
	}
	_, err := s.Resolve(&ResolveOptions{ValidateDefaults: true})
	want := `has type "integer", want "string"`
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("Resolve returned error %q, want %q", err, want)
	}
}

func TestApplyDefaults(t *testing.T) {
	schema := &Schema{
		Properties: map[string]*Schema{
			"A": {Default: mustMarshal(1)},
			"B": {Default: mustMarshal(2)},
			"C": {Default: mustMarshal(3)},
		},
		Required: []string{"C"},
	}
	rs, err := schema.Resolve(&ResolveOptions{ValidateDefaults: true})
	if err != nil {
		t.Fatal(err)
	}

	type S struct{ A, B, C int }
	for _, tt := range []struct {
		instancep any // pointer to instance value
		want      any // desired value (not a pointer)
	}{
		{
			&map[string]any{"B": 0},
			map[string]any{
				"A": float64(1), // filled from default
				"B": 0,          // untouched: it was already there
				// "C" not added: it is required (Validate will catch that)
			},
		},
		{
			&S{B: 1},
			S{
				A: 1, // filled from default
				B: 1, // untouched: non-zero
				C: 0, // untouched: required
			},
		},
	} {
		if err := rs.ApplyDefaults(tt.instancep); err != nil {
			t.Fatal(err)
		}
		got := reflect.ValueOf(tt.instancep).Elem().Interface() // dereference the pointer
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("\ngot  %#v\nwant %#v", got, tt.want)
		}
	}
}

func TestStructInstance(t *testing.T) {
	instance := struct {
		I int
		B bool `json:"b"`
		P *int // either missing or nil
		u int  // unexported: not a property
	}{1, true, nil, 0}

	for _, tt := range []struct {
		s    Schema
		want bool
	}{
		{
			Schema{MinProperties: Ptr(4)},
			false,
		},
		{
			Schema{MinProperties: Ptr(3)},
			true, // P interpreted as present
		},
		{
			Schema{MaxProperties: Ptr(1)},
			false,
		},
		{
			Schema{MaxProperties: Ptr(2)},
			true, // P interpreted as absent
		},
		{
			Schema{Required: []string{"i"}}, // the name is "I"
			false,
		},
		{
			Schema{Required: []string{"B"}}, // the name is "b"
			false,
		},
		{
			Schema{PropertyNames: &Schema{MinLength: Ptr(2)}},
			false,
		},
		{
			Schema{Properties: map[string]*Schema{"b": {Type: "boolean"}}},
			true,
		},
		{
			Schema{Properties: map[string]*Schema{"b": {Type: "number"}}},
			false,
		},
		{
			Schema{Required: []string{"I"}},
			true,
		},
		{
			Schema{Required: []string{"I", "P"}},
			true, // P interpreted as present
		},
		{
			Schema{Required: []string{"I", "P"}, Properties: map[string]*Schema{"P": {Type: "number"}}},
			false, // P interpreted as present, but not a number
		},
		{
			Schema{Required: []string{"I"}, Properties: map[string]*Schema{"P": {Type: "number"}}},
			true, // P not required, so interpreted as absent
		},
		{
			Schema{Required: []string{"I"}, AdditionalProperties: falseSchema()},
			false,
		},
		{
			Schema{DependentRequired: map[string][]string{"b": {"u"}}},
			false,
		},
		{
			Schema{DependentSchemas: map[string]*Schema{"b": falseSchema()}},
			false,
		},
		{
			Schema{UnevaluatedProperties: falseSchema()},
			false,
		},
	} {
		res, err := tt.s.Resolve(nil)
		if err != nil {
			t.Fatal(err)
		}
		err = res.Validate(instance)
		if err == nil && !tt.want {
			t.Errorf("succeeded unexpectedly\nschema = %s", tt.s.json())
		} else if err != nil && tt.want {
			t.Errorf("Validate: %v\nschema = %s", err, tt.s.json())
		}
	}
}

func mustMarshal(x any) json.RawMessage {
	data, err := json.Marshal(x)
	if err != nil {
		panic(err)
	}
	return json.RawMessage(data)
}

// loadRemote loads a remote reference used in the test suite.
func loadRemote(uri *url.URL) (*Schema, error) {
	// Anything with localhost:1234 refers to the remotes directory in the test suite repo.
	if uri.Host == "localhost:1234" {
		return loadSchemaFromFile(filepath.FromSlash(filepath.Join("testdata/remotes", uri.Path)))
	}
	// One test needs the meta-schema files.
	const metaPrefix = "https://json-schema.org/draft/2020-12/"
	if after, ok := strings.CutPrefix(uri.String(), metaPrefix); ok {
		return loadSchemaFromFile(filepath.FromSlash("meta-schemas/draft2020-12/" + after + ".json"))
	}
	return nil, fmt.Errorf("don't know how to load %s", uri)
}

func loadSchemaFromFile(filename string) (*Schema, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	var s Schema
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("unmarshaling JSON at %s: %w", filename, err)
	}
	return &s, nil
}
