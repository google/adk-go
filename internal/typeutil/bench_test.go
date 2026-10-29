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

package typeutil

import (
	"encoding/json"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
)

type benchArgs struct {
	City string `json:"city" jsonschema:"the city to report on"`
	Unit string `json:"unit,omitempty" jsonschema:"celsius or fahrenheit"`
}

type benchResult struct {
	TempC      float64 `json:"temp_c"`
	Conditions string  `json:"conditions"`
	Humidity   int     `json:"humidity"`
}

func benchResolve[T any](b *testing.B) *jsonschema.Resolved {
	b.Helper()
	s, err := jsonschema.For[T](nil)
	if err != nil {
		b.Fatal(err)
	}
	r, err := s.Resolve(nil)
	if err != nil {
		b.Fatal(err)
	}
	return r
}

// BenchmarkConvertToWithJSONSchema splits a function tool call's two
// conversions, model arguments to the Go struct and the Go result back to a
// map, into the steps ConvertToWithJSONSchema takes, so the share of each is
// visible.
func BenchmarkConvertToWithJSONSchema(b *testing.B) {
	args := map[string]any{"city": "Paris", "unit": "celsius"}
	result := benchResult{TempC: 22.5, Conditions: "sunny", Humidity: 40}
	argsSchema, resultSchema := benchResolve[benchArgs](b), benchResolve[benchResult](b)

	b.Run("args/total", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := ConvertToWithJSONSchema[map[string]any, benchArgs](args, argsSchema); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("result/total", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := ConvertToWithJSONSchema[benchResult, map[string]any](result, resultSchema); err != nil {
				b.Fatal(err)
			}
		}
	})
	raw, _ := json.Marshal(result)
	var decoded any
	_ = json.Unmarshal(raw, &decoded)
	b.Run("result/marshal", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := json.Marshal(result); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("result/unmarshal", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			var v any
			if err := json.Unmarshal(raw, &v); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("result/validate", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if err := resultSchema.Validate(decoded); err != nil {
				b.Fatal(err)
			}
		}
	})
}
