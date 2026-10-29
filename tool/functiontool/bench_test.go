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

package functiontool_test

import (
	"fmt"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/functiontool"
)

type benchArgs struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

type benchItem struct {
	ID      string `json:"id"`
	Title   string `json:"title"`
	Snippet string `json:"snippet"`
	Score   int    `json:"score"`
}

type benchStructResult struct {
	Items []benchItem `json:"items"`
	Total int         `json:"total"`
}

// BenchmarkFunctionToolRun measures the per-call JSON conversion and schema
// validation overhead inside functiontool.Run for a ~5 KB struct result and a
// 100 KB string result.
func BenchmarkFunctionToolRun(b *testing.B) {
	items := make([]benchItem, 20)
	snippet := strings.Repeat("result snippet text ", 10) // 200 B
	for i := range items {
		items[i] = benchItem{
			ID:      fmt.Sprintf("item-%02d", i),
			Title:   fmt.Sprintf("Document title %02d", i),
			Snippet: snippet,
			Score:   100 - i,
		}
	}
	structOut := benchStructResult{Items: items, Total: len(items)}
	stringOut := strings.Repeat("x", 100<<10)
	args := map[string]any{"query": "search query", "limit": float64(20)}

	ic := icontext.NewInvocationContext(b.Context(), icontext.InvocationContextParams{})
	tctx := agent.NewToolContext(ic, "call-1", &session.EventActions{}, nil)

	b.Run("result=struct-5KB", func(b *testing.B) {
		tl, err := functiontool.New(functiontool.Config{
			Name:        "search_struct",
			Description: "returns a ~5 KB struct result",
		}, func(_ agent.Context, _ benchArgs) (benchStructResult, error) {
			return structOut, nil
		})
		if err != nil {
			b.Fatal(err)
		}
		ft := tl.(toolinternal.FunctionTool)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			if _, err := ft.Run(tctx, args); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("result=string-100KB", func(b *testing.B) {
		tl, err := functiontool.New(functiontool.Config{
			Name:        "fetch_blob",
			Description: "returns a 100 KB string result",
		}, func(_ agent.Context, _ benchArgs) (string, error) {
			return stringOut, nil
		})
		if err != nil {
			b.Fatal(err)
		}
		ft := tl.(toolinternal.FunctionTool)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			if _, err := ft.Run(tctx, args); err != nil {
				b.Fatal(err)
			}
		}
	})
}
