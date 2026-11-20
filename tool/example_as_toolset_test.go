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

package tool_test

import (
	"fmt"
	"log"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// AsToolset wraps one tool as a toolset named after the tool, so that it can
// sit between other toolsets in llmagent.Config.Toolsets, where the agent
// offers the toolsets' tools in declared order.
func ExampleAsToolset() {
	summarize, err := functiontool.New(functiontool.Config{Name: "summarize", Description: "Summarizes a text."},
		func(agent.Context, struct{ Text string }) (map[string]any, error) { return map[string]any{}, nil })
	if err != nil {
		log.Fatal(err)
	}
	ts := tool.AsToolset(summarize)
	tools, err := ts.Tools(nil)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(ts.Name(), len(tools), tools[0] == summarize)
	// Output:
	// summarize 1 true
}
