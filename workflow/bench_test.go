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

package workflow

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool/functiontool"
)

// benchLookupArgs is the argument shape of the benchmark tool: a few fields
// whose JSON encoding is about 1 KB, the size of a typical structured tool call.
type benchLookupArgs struct {
	Query   string            `json:"query"`
	Tags    []string          `json:"tags"`
	Limit   int               `json:"limit"`
	Filters map[string]string `json:"filters"`
}

// benchToolNode returns a Start→ToolNode workflow around a function tool with a
// typed input schema, plus the tool's arguments as a map and as JSON text.
func benchToolNode(b *testing.B) (*Workflow, *ToolNode, map[string]any, []byte) {
	b.Helper()
	lookup, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "looks things up"},
		func(ctx agent.Context, in benchLookupArgs) (map[string]any, error) {
			return map[string]any{"hits": len(in.Tags)}, nil
		})
	if err != nil {
		b.Fatal(err)
	}
	node, err := NewToolNodeTyped[benchLookupArgs, map[string]any](lookup, NodeConfig{})
	if err != nil {
		b.Fatal(err)
	}
	wf, err := New("", Chain(Start, node))
	if err != nil {
		b.Fatal(err)
	}
	tags := make([]any, 0, 20)
	filters := map[string]any{}
	for i := range 20 {
		tags = append(tags, fmt.Sprintf("tag-%02d", i))
		filters[fmt.Sprintf("field_%02d", i)] = fmt.Sprintf("value-%02d", i)
	}
	args := map[string]any{
		"query":   strings.Repeat("quick brown fox ", 32),
		"tags":    tags,
		"limit":   10.0,
		"filters": filters,
	}
	raw, err := json.Marshal(args)
	if err != nil {
		b.Fatal(err)
	}
	return wf, node, args, raw
}

// BenchmarkToolNodeActivation runs a Start→ToolNode workflow once per
// iteration, with the arguments as a map (the shape a structured upstream node
// produces) and as JSON text (the shape an agent node produces). What it
// measures is the engine's per-activation work around one tool call: input
// validation, scheduling, the tool's own argument conversion, and output
// validation. The model is not involved.
func BenchmarkToolNodeActivation(b *testing.B) {
	wf, _, args, raw := benchToolNode(b)
	for _, tc := range []struct {
		name  string
		input any
	}{
		{"map", args},
		{"json-string", string(raw)},
	} {
		b.Run(tc.name, func(b *testing.B) {
			ctx := agent.NewContext(&MockInvocationContext{Context: b.Context(), sess: &mockSession{id: "bench"}})
			b.ReportAllocs()
			for b.Loop() {
				for _, err := range wf.RunNode(ctx, tc.input) {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
			b.ReportMetric(float64(len(raw)), "args-bytes")
		})
	}
}

// BenchmarkToolNodeValidateInput is one ValidateInput pass over the same
// arguments: the unit of work an activation pays per validation.
func BenchmarkToolNodeValidateInput(b *testing.B) {
	_, node, args, raw := benchToolNode(b)
	for _, tc := range []struct {
		name  string
		input any
	}{
		{"map", args},
		{"json-string", string(raw)},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := node.ValidateInput(tc.input); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
