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

package vertexai

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestToStructPBEquivalence(t *testing.T) {
	type sampleStruct struct {
		Name  string `json:"name"`
		Count int    `json:"count,omitempty"`
	}

	cases := []struct {
		name  string
		input any
	}{
		{name: "untyped nil", input: nil},
		{name: "typed nil map", input: map[string]any(nil)},
		{name: "empty map", input: map[string]any{}},
		{
			name: "json native primitives and containers",
			input: map[string]any{
				"null":       nil,
				"bool_true":  true,
				"bool_false": false,
				"str":        "hello world 🌍 <script>&",
				"empty_str":  "",
				"float_zero": 0.0,
				"neg_zero":   math.Copysign(0, -1),
				"float_pos":  12345.6789,
				"float_tiny": 1e-300,
				"float_huge": math.MaxFloat64,
				"int":        42,
				"int_neg":    -999999,
				"int64_max":  int64(1 << 53),
				"int64_min":  -int64(1 << 53),
				"uint64_max": uint64(1 << 53),
				"empty_list": []any{},
				"nil_list":   []any(nil),
				"nil_map":    map[string]any(nil),
				"list":       []any{nil, true, "abc", 3.14, int64(7), map[string]any{"k": "v"}, map[string]any(nil), []any(nil), []any{1, 2}},
				"nested_map": map[string]any{"empty": map[string]any{}, "inner": map[string]any{"x": 1.5}},
			},
		},
		{
			name: "int64 beyond 2^53 falls back to JSON",
			input: map[string]any{
				"big_int":  int64(1<<53 + 7),
				"big_uint": uint64(1<<62 + 3),
			},
		},
		{
			name: "float32 falls back to JSON",
			input: map[string]any{
				"f32": float32(0.1),
			},
		},
		{
			name: "invalid utf8 in value falls back to JSON",
			input: map[string]any{
				"bad": string([]byte{0xff, 0xfe, 'a'}),
			},
		},
		{
			name: "invalid utf8 in key falls back to JSON",
			input: map[string]any{
				string([]byte{0xff, 'k'}): "v",
			},
		},
		{
			name: "nested struct and typed slice fall back to JSON",
			input: map[string]any{
				"struct":      sampleStruct{Name: "test", Count: 5},
				"typed_slice": []string{"a", "b"},
				"typed_map":   map[string]int{"x": 1},
			},
		},
		{
			name:  "top level struct",
			input: sampleStruct{Name: "top"},
		},
		{
			name:  "NaN returns identical error",
			input: map[string]any{"nan": math.NaN()},
		},
		{
			name:  "PosInf returns identical error",
			input: map[string]any{"inf": math.Inf(1)},
		},
		{
			name:  "NegInf returns identical error",
			input: map[string]any{"ninf": math.Inf(-1)},
		},
		{
			name:  "top level string returns identical error",
			input: "not a struct",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, gotErr := toStructPB(tc.input)
			want, wantErr := toStructPBOriginal(tc.input)
			if (gotErr != nil) != (wantErr != nil) {
				t.Fatalf("toStructPB() err = %v, want %v", gotErr, wantErr)
			}
			if gotErr != nil {
				if gotErr.Error() != wantErr.Error() {
					t.Fatalf("toStructPB() err = %q, want %q", gotErr.Error(), wantErr.Error())
				}
				return
			}
			if !proto.Equal(got, want) {
				t.Fatalf("toStructPB() proto mismatch:\ngot:  %v\nwant: %v", got, want)
			}
		})
	}
}

func BenchmarkToStructPB(b *testing.B) {
	stateDelta := make(map[string]any, 20)
	for i := 0; i < 20; i++ {
		switch i % 4 {
		case 0:
			stateDelta[fmt.Sprintf("str_%d", i)] = fmt.Sprintf("value_%d", i)
		case 1:
			stateDelta[fmt.Sprintf("num_%d", i)] = float64(i) * 1.5
		case 2:
			stateDelta[fmt.Sprintf("bool_%d", i)] = i%2 == 0
		case 3:
			stateDelta[fmt.Sprintf("nested_%d", i)] = map[string]any{
				"id":   i,
				"tags": []any{"a", "b", "c"},
			}
		}
	}

	resp1KB := makeBenchResponseMap(4)
	resp10KB := makeBenchResponseMap(40)

	b.Run("StateDelta/keys=20", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s, err := toStructPB(stateDelta)
			if err != nil || len(s.Fields) != 20 {
				b.Fatalf("toStructPB() = %v, %v", s, err)
			}
		}
	})

	b.Run("FunctionResponse/bytes=1KB", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s, err := toStructPB(resp1KB)
			if err != nil || len(s.Fields) == 0 {
				b.Fatalf("toStructPB() = %v, %v", s, err)
			}
		}
	})

	b.Run("FunctionResponse/bytes=10KB", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s, err := toStructPB(resp10KB)
			if err != nil || len(s.Fields) == 0 {
				b.Fatalf("toStructPB() = %v, %v", s, err)
			}
		}
	})
}

func toStructPBOriginal(value any) (*structpb.Struct, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal value: %w", err)
	}
	if string(data) == "null" {
		return &structpb.Struct{}, nil
	}
	res := &structpb.Struct{}
	if err := res.UnmarshalJSON(data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON data to structpb: %w", err)
	}
	return res, nil
}

func makeBenchResponseMap(nItems int) map[string]any {
	items := make([]any, nItems)
	summary := strings.Repeat("Result snippet from external knowledge base lookup. ", 4)
	for i := 0; i < nItems; i++ {
		items[i] = map[string]any{
			"id":      fmt.Sprintf("doc-%04d", i),
			"score":   0.95 - float64(i)*0.01,
			"matched": true,
			"snippet": summary,
			"tokens":  128 + i,
		}
	}
	return map[string]any{
		"status": "ok",
		"count":  nItems,
		"items":  items,
	}
}
