// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonschema

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"testing"
)

func TestGoRoundTrip(t *testing.T) {
	// Verify that Go representations round-trip.
	for _, s := range []*Schema{
		{Type: "null"},
		{Types: []string{"null", "number"}},
		{Type: "string", MinLength: Ptr(20)},
		{Minimum: Ptr(20.0)},
		{Items: &Schema{Type: "integer"}},
		{Const: Ptr(any(0))},
		{Const: Ptr(any(nil))},
		{Const: Ptr(any([]int{}))},
		{Const: Ptr(any(map[string]any{}))},
		{Default: mustMarshal(1)},
		{Default: mustMarshal(nil)},
	} {
		data, err := json.Marshal(s)
		if err != nil {
			t.Fatal(err)
		}
		var got *Schema
		mustUnmarshal(t, data, &got)
		if !Equal(got, s) {
			t.Errorf("got %s, want %s", got.json(), s.json())
			if got.Const != nil && s.Const != nil {
				t.Logf("Consts: got %#v (%[1]T), want %#v (%[2]T)", *got.Const, *s.Const)
			}
		}
	}
}

func TestJSONRoundTrip(t *testing.T) {
	// Verify that JSON texts for schemas marshal into equivalent forms.
	// We don't expect everything to round-trip perfectly. For example, "true" and "false"
	// will turn into their object equivalents.
	// But most things should.
	// Some of these cases test Schema.{UnM,M}arshalJSON.
	// Most of others follow from the behavior of encoding/json, but they are still
	// valuable as regression tests of this package's behavior.
	for _, tt := range []struct {
		in, want string
	}{
		{`true`, `{}`}, // boolean schemas become object schemas
		{`false`, `{"not":{}}`},
		{`{"type":"", "enum":null}`, `{}`}, // empty fields are omitted
		{`{"minimum":1}`, `{"minimum":1}`},
		{`{"minimum":1.0}`, `{"minimum":1}`},     // floating-point integers lose their fractional part
		{`{"minLength":1.0}`, `{"minLength":1}`}, // some floats are unmarshaled into ints, but you can't tell
		{
			// map keys are sorted
			`{"$vocabulary":{"b":true, "a":false}}`,
			`{"$vocabulary":{"a":false,"b":true}}`,
		},
		{`{"unk":0}`, `{}`}, // unknown fields are dropped, unfortunately
	} {
		var s Schema
		mustUnmarshal(t, []byte(tt.in), &s)
		data, err := json.Marshal(s)
		if err != nil {
			t.Fatal(err)
		}
		if got := string(data); got != tt.want {
			t.Errorf("%s:\ngot  %s\nwant %s", tt.in, got, tt.want)
		}
	}
}

func TestUnmarshalErrors(t *testing.T) {
	for _, tt := range []struct {
		in   string
		want string // error must match this regexp
	}{
		{`1`, "cannot unmarshal number"},
		{`{"type":1}`, `invalid value for "type"`},
		{`{"minLength":1.5}`, `not an integer value`},
		{`{"maxLength":1.5}`, `not an integer value`},
		{`{"minItems":1.5}`, `not an integer value`},
		{`{"maxItems":1.5}`, `not an integer value`},
		{`{"minProperties":1.5}`, `not an integer value`},
		{`{"maxProperties":1.5}`, `not an integer value`},
		{`{"minContains":1.5}`, `not an integer value`},
		{`{"maxContains":1.5}`, `not an integer value`},
		{fmt.Sprintf(`{"maxContains":%d}`, int64(math.MaxInt32+1)), `out of range`},
		{`{"minLength":9e99}`, `cannot be unmarshaled`},
		{`{"minLength":"1.5"}`, `not a number`},
	} {
		var s Schema
		err := json.Unmarshal([]byte(tt.in), &s)
		if err == nil {
			t.Fatalf("%s: no error but expected one", tt.in)
		}
		if !regexp.MustCompile(tt.want).MatchString(err.Error()) {
			t.Errorf("%s: error %q does not match %q", tt.in, err, tt.want)
		}

	}
}

func mustUnmarshal(t *testing.T, data []byte, ptr any) {
	t.Helper()
	if err := json.Unmarshal(data, ptr); err != nil {
		t.Fatal(err)
	}
}

// json returns the schema in json format.
func (s *Schema) json() string {
	data, err := json.Marshal(s)
	if err != nil {
		return fmt.Sprintf("<jsonschema.Schema:%v>", err)
	}
	return string(data)
}

// json returns the schema in json format, indented.
func (s *Schema) jsonIndent() string {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Sprintf("<jsonschema.Schema:%v>", err)
	}
	return string(data)
}
