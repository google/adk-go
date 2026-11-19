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
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// docsToolset is a toolset with a fixed list of tools.
type docsToolset struct{ tools []tool.Tool }

func (docsToolset) Name() string { return "docs" }

func (ts docsToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return ts.tools, nil }

// WithToolNamePrefix offers the tools of a toolset as "<prefix>_<name>", so
// that two toolsets with tools of the same name can serve one agent. Wrapped
// around FilterToolset, the filter sees the original names, as adk-python's
// tool_filter does.
func ExampleWithToolNamePrefix() {
	var tools []tool.Tool
	for _, name := range []string{"search", "fetch"} {
		t, err := functiontool.New(functiontool.Config{Name: name, Description: "Works on the docs."},
			func(agent.Context, struct{}) (map[string]any, error) { return nil, nil })
		if err != nil {
			log.Fatal(err)
		}
		tools = append(tools, t)
	}
	ts := docsToolset{tools: tools}

	printNames := func(ts tool.Toolset) {
		listed, err := ts.Tools(nil)
		if err != nil {
			log.Fatal(err)
		}
		var names []string
		for _, t := range listed {
			names = append(names, t.Name())
		}
		fmt.Println(strings.Join(names, " "))
	}
	printNames(tool.WithToolNamePrefix(ts, "docs"))
	printNames(tool.WithToolNamePrefix(tool.FilterToolset(ts, tool.AllowedToolsPredicate([]string{"search"})), "docs"))
	printNames(ts) // the original tools keep their names

	// Output:
	// docs_search docs_fetch
	// docs_search
	// search fetch
}
