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
	"fmt"
	"strings"
	"testing"
)

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
