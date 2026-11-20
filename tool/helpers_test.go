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
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/toolutils"
)

// packingFunctionTool is a function tool whose request hook packs itself into
// the request, as functiontool's does.
type packingFunctionTool struct {
	name string
	decl *genai.FunctionDeclaration
	long bool
	run  func(ctx agent.Context, args any) (map[string]any, error)
}

func (f *packingFunctionTool) Name() string        { return f.name }
func (f *packingFunctionTool) Description() string { return "" }
func (f *packingFunctionTool) IsLongRunning() bool { return f.long }
func (f *packingFunctionTool) Declaration() *genai.FunctionDeclaration {
	return f.decl
}

func (f *packingFunctionTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	return f.run(ctx, args)
}

func (f *packingFunctionTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, f)
}

// newFunctionTool returns a packingFunctionTool named name with a declaration
// of the same name.
func newFunctionTool(name string) *packingFunctionTool {
	return &packingFunctionTool{name: name, decl: &genai.FunctionDeclaration{Name: name}}
}
