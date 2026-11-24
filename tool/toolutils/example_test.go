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

package toolutils_test

import (
	"fmt"
	"log"
	"maps"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/toolutils"
)

// declaredTool is the least a tool needs for PackTool: a name and a
// declaration, which may be nil.
type declaredTool struct {
	name string
	decl *genai.FunctionDeclaration
}

func (t declaredTool) Name() string                            { return t.name }
func (t declaredTool) Declaration() *genai.FunctionDeclaration { return t.decl }

// PackTool adds each declaration to the first genai.Tool that already holds
// function declarations, or to the first bare genai.Tool placeholder when none
// does, skipping non-function tools such as Google Search, and appends a new
// genai.Tool otherwise. A tool without a declaration is registered in req.Tools
// and adds nothing to req.Config.Tools; the LLM flow deletes that entry before
// the model sees the request.
func ExamplePackTool() {
	req := &model.LLMRequest{Config: &genai.GenerateContentConfig{Tools: []*genai.Tool{
		{GoogleSearch: &genai.GoogleSearch{}},
		{FunctionDeclarations: []*genai.FunctionDeclaration{}},
	}}}
	for _, t := range []declaredTool{
		{name: "get_weather", decl: &genai.FunctionDeclaration{Name: "get_weather"}},
		{name: "audit_log"},
		{name: "get_time", decl: &genai.FunctionDeclaration{Name: "get_time"}},
	} {
		if err := toolutils.PackTool(req, t); err != nil {
			log.Fatal(err)
		}
	}
	for i, gt := range req.Config.Tools {
		var names []string
		for _, d := range gt.FunctionDeclarations {
			names = append(names, d.Name)
		}
		fmt.Printf("Config.Tools[%d]: google search %v, declarations %v\n", i, gt.GoogleSearch != nil, names)
	}
	fmt.Println("registered:", slices.Sorted(maps.Keys(req.Tools)))
	// Output:
	// Config.Tools[0]: google search true, declarations []
	// Config.Tools[1]: google search false, declarations [get_weather get_time]
	// registered: [audit_log get_time get_weather]
}
