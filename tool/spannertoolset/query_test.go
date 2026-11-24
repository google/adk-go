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

package spannertoolset

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func readStream(t *testing.T, body string, limit int) (*resultSet, error) {
	t.Helper()
	return readResultSet(json.NewDecoder(strings.NewReader(body)), limit)
}

const twoStringColumns = `"metadata":{"rowType":{"fields":[{"name":"a","type":{"code":"STRING"}},{"name":"b","type":{"code":"STRING"}}]}}`

func TestReadResultSet(t *testing.T) {
	tests := []struct {
		name          string
		body          string
		limit         int
		wantRows      [][]any
		wantTruncated bool
	}{
		{
			name:     "rows in one part",
			body:     `[{` + twoStringColumns + `,"values":["1","2","3","4"]}]`,
			limit:    -1,
			wantRows: [][]any{{"1", "2"}, {"3", "4"}},
		},
		{
			name:     "a row across parts",
			body:     `[{` + twoStringColumns + `,"values":["1"]},{"values":["2","3"]},{"values":["4"]}]`,
			limit:    -1,
			wantRows: [][]any{{"1", "2"}, {"3", "4"}},
		},
		{
			name:     "a chunked string",
			body:     `[{` + twoStringColumns + `,"values":["1","ab"],"chunkedValue":true},{"values":["cd"]}]`,
			limit:    -1,
			wantRows: [][]any{{"1", "abcd"}},
		},
		{
			name:     "a value chunked over three parts",
			body:     `[{` + twoStringColumns + `,"values":["1","a"],"chunkedValue":true},{"values":["b"],"chunkedValue":true},{"values":["c","x","y"]}]`,
			limit:    -1,
			wantRows: [][]any{{"1", "abc"}, {"x", "y"}},
		},
		{
			name:     "no rows",
			body:     `[{` + twoStringColumns + `}]`,
			limit:    -1,
			wantRows: nil,
		},
		{
			name:          "more rows than the limit",
			body:          `[{` + twoStringColumns + `,"values":["1","2","3","4","5","6"]}]`,
			limit:         2,
			wantRows:      [][]any{{"1", "2"}, {"3", "4"}},
			wantTruncated: true,
		},
		{
			name:     "exactly the limit",
			body:     `[{` + twoStringColumns + `,"values":["1","2","3","4"]}]`,
			limit:    2,
			wantRows: [][]any{{"1", "2"}, {"3", "4"}},
		},
		{
			name:          "a limit of zero",
			body:          `[{` + twoStringColumns + `,"values":["1","2"]}]`,
			limit:         0,
			wantRows:      nil,
			wantTruncated: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rs, err := readStream(t, tc.body, tc.limit)
			if err != nil {
				t.Fatalf("readResultSet() error = %v", err)
			}
			if diff := cmp.Diff(tc.wantRows, rs.rows); diff != "" {
				t.Errorf("rows mismatch (-want +got):\n%s", diff)
			}
			if rs.truncated != tc.wantTruncated {
				t.Errorf("truncated = %v, want %v", rs.truncated, tc.wantTruncated)
			}
		})
	}
}

func TestReadResultSet_TransactionID(t *testing.T) {
	rs, err := readStream(t, `[{"metadata":{"rowType":{"fields":[]},"transaction":{"id":"dHgx"}}}]`, -1)
	if err != nil {
		t.Fatalf("readResultSet() error = %v", err)
	}
	if rs.transactionID != "dHgx" {
		t.Errorf("transactionID = %q, want %q", rs.transactionID, "dHgx")
	}
}

func TestReadResultSet_Errors(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{"not an array", `{"values":[]}`, "malformed Spanner query response"},
		{"empty body", ``, "malformed Spanner query response"},
		{"undecodable part", `[{"values":}]`, "decoding the Spanner query response"},
		{"error after rows", `[{` + twoStringColumns + `,"values":["1","2"]},{"error":{"code":3,"message":"Division by zero"}}]`, "Division by zero"},
		{"ends inside a row", `[{` + twoStringColumns + `,"values":["1"]}]`, "ended in the middle of a row"},
		{"ends inside a chunked value", `[{` + twoStringColumns + `,"values":["1","a"],"chunkedValue":true}]`, "ended in the middle of a row"},
		{"chunks that cannot merge", `[{` + twoStringColumns + `,"values":["1","a"],"chunkedValue":true},{"values":[2]}]`, "cannot merge chunked Spanner values"},
		{"value of the wrong type", `[{"metadata":{"rowType":{"fields":[{"name":"n","type":{"code":"INT64"}}]}},"values":[true]}]`, "column 0: invalid Spanner INT64 value"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := readStream(t, tc.body, -1)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("readResultSet() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestMergeChunks(t *testing.T) {
	tests := []struct {
		name string
		a, b any
		want any
	}{
		{"strings", "ab", "cd", "abcd"},
		{"lists of strings merge the edge", []any{"a", "b"}, []any{"c", "d"}, []any{"a", "bc", "d"}},
		{"lists of numbers concatenate", []any{1.0, 2.0}, []any{3.0}, []any{1.0, 2.0, 3.0}},
		{"nested lists", []any{[]any{"a", "b"}}, []any{[]any{"c"}, []any{"d"}}, []any{[]any{"a", "bc"}, []any{"d"}}},
		{"empty first list", []any{}, []any{"a"}, []any{"a"}},
		{"empty second list", []any{"a"}, []any{}, []any{"a"}},
		{"a null edge is not merged", []any{"a", nil}, []any{"b"}, []any{"a", nil, "b"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := mergeChunks(tc.a, tc.b)
			if err != nil {
				t.Fatalf("mergeChunks() error = %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("mergeChunks() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMergeChunks_DoesNotChangeFirstList(t *testing.T) {
	a := make([]any, 2, 4)
	a[0], a[1] = "a", "b"
	if _, err := mergeChunks(a, []any{"c", "d"}); err != nil {
		t.Fatalf("mergeChunks() error = %v", err)
	}
	if diff := cmp.Diff([]any{"a", "b"}, a); diff != "" {
		t.Errorf("first list changed (-want +got):\n%s", diff)
	}
}

func TestMergeChunks_Errors(t *testing.T) {
	tests := []struct {
		name string
		a, b any
	}{
		{"string and number", "a", 1.0},
		{"list and string", []any{"a"}, "b"},
		{"number", 1.0, 2.0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got, err := mergeChunks(tc.a, tc.b); err == nil {
				t.Errorf("mergeChunks() = %v, want an error", got)
			}
		})
	}
}

func typeOf(code string) spannerType { return spannerType{Code: code} }

func arrayOf(elem spannerType) spannerType {
	return spannerType{Code: "ARRAY", ArrayElementType: &elem}
}

func structOf(fields ...field) spannerType {
	t := spannerType{Code: "STRUCT"}
	t.StructType = &struct {
		Fields []field `json:"fields"`
	}{Fields: fields}
	return t
}

func TestConvertValue(t *testing.T) {
	tests := []struct {
		name  string
		typ   spannerType
		value any
		want  any
	}{
		{"null", typeOf("INT64"), nil, nil},
		{"bool", typeOf("BOOL"), true, true},
		{"int64", typeOf("INT64"), "42", int64(42)},
		{"negative int64", typeOf("INT64"), "-7", int64(-7)},
		{"int64 at 2^53", typeOf("INT64"), "9007199254740992", int64(9007199254740992)},
		{"int64 beyond 2^53", typeOf("INT64"), "9007199254740993", "9007199254740993"},
		{"int64 below -2^53", typeOf("INT64"), "-9007199254740993", "-9007199254740993"},
		{"enum", typeOf("ENUM"), "2", int64(2)},
		{"float64", typeOf("FLOAT64"), 1.5, 1.5},
		{"float32", typeOf("FLOAT32"), 0.25, 0.25},
		{"float64 NaN", typeOf("FLOAT64"), "NaN", "NaN"},
		{"float64 infinity", typeOf("FLOAT64"), "-Infinity", "-Infinity"},
		{"string", typeOf("STRING"), "hi", "hi"},
		{"numeric", typeOf("NUMERIC"), "1.10", "1.10"},
		{"timestamp", typeOf("TIMESTAMP"), "2024-01-02T03:04:05.123Z", "2024-01-02T03:04:05.123Z"},
		{"date", typeOf("DATE"), "2024-01-02", "2024-01-02"},
		{"bytes", typeOf("BYTES"), "aGk=", "aGk="},
		{"json object", typeOf("JSON"), `{"a":1,"b":[true,null],"c":1.5}`, map[string]any{"a": int64(1), "b": []any{true, nil}, "c": json.Number("1.5")}},
		{"json large integer", typeOf("JSON"), `{"n":9007199254740993}`, map[string]any{"n": "9007199254740993"}},
		{"json integer beyond int64", typeOf("JSON"), `99999999999999999999`, "99999999999999999999"},
		{"array of int64", arrayOf(typeOf("INT64")), []any{"1", nil, "3"}, []any{int64(1), nil, int64(3)}},
		{"empty array", arrayOf(typeOf("STRING")), []any{}, []any{}},
		{"struct", structOf(field{Name: "x", Type: typeOf("INT64")}, field{Name: "y", Type: typeOf("STRING")}), []any{"5", "z"}, []any{int64(5), "z"}},
		{"array of structs", arrayOf(structOf(field{Name: "x", Type: typeOf("BOOL")})), []any{[]any{true}, []any{false}}, []any{[]any{true}, []any{false}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := convertValue(tc.typ, tc.value)
			if err != nil {
				t.Fatalf("convertValue() error = %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("convertValue() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvertValue_Errors(t *testing.T) {
	tests := []struct {
		name  string
		typ   spannerType
		value any
	}{
		{"bool as string", typeOf("BOOL"), "true"},
		{"int64 as number", typeOf("INT64"), 1.0},
		{"int64 not a number", typeOf("INT64"), "one"},
		{"float64 as bool", typeOf("FLOAT64"), true},
		{"json as object", typeOf("JSON"), map[string]any{}},
		{"invalid json", typeOf("JSON"), "{"},
		{"array as string", arrayOf(typeOf("INT64")), "[1]"},
		{"array without element type", spannerType{Code: "ARRAY"}, []any{}},
		{"bad array element", arrayOf(typeOf("INT64")), []any{"x"}},
		{"struct without type", spannerType{Code: "STRUCT"}, []any{}},
		{"struct of the wrong length", structOf(field{Name: "x", Type: typeOf("INT64")}), []any{"1", "2"}},
		{"bad struct field", structOf(field{Name: "x", Type: typeOf("INT64")}), []any{"x"}},
		{"string as number", typeOf("STRING"), 1.0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := convertValue(tc.typ, tc.value)
			if err == nil {
				t.Fatalf("convertValue() = %v, want an error", got)
			}
			if s, ok := tc.value.(string); ok && s != "" && strings.Contains(err.Error(), s) {
				t.Errorf("error %q quotes the value", err)
			}
		})
	}
}
