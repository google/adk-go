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

package pythonjson_test

import (
	"math"
	"testing"

	"google.golang.org/adk/v2/internal/pythonjson"
)

// The want values are what adk-python's json.dumps(v, ensure_ascii=False)
// writes for the same value, keys aside: Go writes them in sorted order.
func TestEncode(t *testing.T) {
	t.Parallel()
	type named struct {
		Name string `json:"name"`
		Skip string `json:"-"`
	}
	testCases := []struct {
		name string
		v    any
		want string
	}{
		{name: "separators and sorted keys", v: map[string]any{"b": 1, "a": "x"}, want: `{"a": "x", "b": 1}`},
		{name: "HTML characters stay literal", v: "a < b & c > d", want: `"a < b & c > d"`},
		{name: "non-ASCII stays literal", v: "שלום 北京", want: `"שלום 北京"`},
		{name: "line and paragraph separators stay literal", v: "a\u2028b\u2029c", want: "\"a\u2028b\u2029c\""},
		{
			name: "quote, backslash and control characters are escaped",
			v:    "line\nbreak\ttab\"quote\\back\x01\x1f\x7f\b\f\r",
			want: `"line\nbreak\ttab\"quote\\back\u0001\u001f` + "\x7f" + `\b\f\r"`,
		},
		{name: "small float in exponent form", v: 0.00001, want: `1e-05`},
		{name: "negative small float", v: -0.00001, want: `-1e-05`},
		{name: "float with seven fractional digits", v: 1.5e-7, want: `1.5e-07`},
		{name: "float just above the exponent threshold", v: 0.0001, want: `0.0001`},
		{name: "float at 1e16 in exponent form", v: 1e16, want: `1e+16`},
		{name: "negative float at 1e16 in exponent form", v: -1e16, want: `-1e+16`},
		{name: "integer just below 1e16", v: int64(1234567890123456), want: `1234567890123456`},
		{name: "fractional float just below 1e16", v: 1234567890123456.5, want: `1234567890123456.5`},
		{name: "float at 1e21", v: 1e21, want: `1e+21`},
		{name: "large float with fraction digits", v: 1.2345678901234567e+20, want: `1.2345678901234567e+20`},
		{name: "plain float", v: 123456789.125, want: `123456789.125`},
		{name: "integral float as an integer", v: float64(3), want: `3`},
		{name: "integer", v: 42, want: `42`},
		{name: "integer float64 cannot hold exactly", v: int64(12345678901234567), want: `12345678901234567`},
		{name: "empty list", v: []any{}, want: `[]`},
		{name: "empty map", v: map[string]any{}, want: `{}`},
		{name: "null", v: nil, want: `null`},
		{name: "booleans", v: []any{true, false}, want: `[true, false]`},
		{name: "nested", v: []any{map[string]any{"k": []any{1, "v"}}, []any{}}, want: `[{"k": [1, "v"]}, []]`},
		{name: "typed slice", v: []string{"a < b", "c"}, want: `["a < b", "c"]`},
		{name: "typed map", v: map[string]string{"k": "v"}, want: `{"k": "v"}`},
		{name: "struct with JSON tags", v: named{Name: "x", Skip: "y"}, want: `{"name": "x"}`},
		{name: "NaN in a map", v: map[string]any{"k": "v", "x": math.NaN()}, want: `{"k": "v", "x": NaN}`},
		{name: "infinities in a list", v: []any{math.Inf(1), math.Inf(-1), 1.5}, want: `[Infinity, -Infinity, 1.5]`},
		{name: "float32 infinity", v: map[string]any{"f": float32(math.Inf(1))}, want: `{"f": Infinity}`},
		{name: "negative zero", v: math.Copysign(0, -1), want: `-0.0`},
		{name: "positive zero as an integer", v: 0.0, want: `0`},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := pythonjson.Encode(tc.v)
			if err != nil {
				t.Fatalf("Encode() error = %v", err)
			}
			if got != tc.want {
				t.Errorf("Encode() = %s, want %s", got, tc.want)
			}
		})
	}
}

func TestEncode_UnencodableValue(t *testing.T) {
	t.Parallel()
	for _, v := range []any{complex(1, 2), map[string]any{"ch": make(chan int)}} {
		if got, err := pythonjson.Encode(v); err == nil {
			t.Errorf("Encode(%T) = %q, want an error", v, got)
		}
	}
}
