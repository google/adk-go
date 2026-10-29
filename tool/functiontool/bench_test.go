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
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/benchutil"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// BenchmarkFunctionToolRun is one call of a struct-typed function tool with
// the arguments a model sends: the per-tool-call framework cost, since the
// handler itself does almost nothing.
func BenchmarkFunctionToolRun(b *testing.B) {
	tools, err := benchutil.Tools()
	if err != nil {
		b.Fatal(err)
	}
	cases := []struct {
		name string
		tool int
		args map[string]any
	}{
		{"get_weather", 0, map[string]any{"city": "Paris", "unit": "celsius"}},
		{"lookup_order", 1, map[string]any{"order_id": "A-1234", "include_items": true}},
		{"convert_currency", 2, map[string]any{"amount": 125.5, "from": "EUR", "to": "USD"}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			ft := tools[tc.tool].(toolinternal.FunctionTool)
			ctx := agent.NewToolContext(icontext.NewInvocationContext(b.Context(), icontext.InvocationContextParams{}), "call-1", &session.EventActions{}, nil)
			b.ReportAllocs()
			for b.Loop() {
				if _, err := ft.Run(ctx, tc.args); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkFunctionToolProcessRequest packs the three tools' declarations
// into a request, which the flow does before every model call.
func BenchmarkFunctionToolProcessRequest(b *testing.B) {
	tools, err := benchutil.Tools()
	if err != nil {
		b.Fatal(err)
	}
	ctx := agent.NewToolContext(icontext.NewInvocationContext(b.Context(), icontext.InvocationContextParams{}), "", &session.EventActions{}, nil)
	b.ReportAllocs()
	for b.Loop() {
		req := &model.LLMRequest{}
		for _, t := range tools {
			if err := t.(toolinternal.RequestProcessor).ProcessRequest(ctx, req); err != nil {
				b.Fatal(err)
			}
		}
	}
}
