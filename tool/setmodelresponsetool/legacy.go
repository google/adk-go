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

package setmodelresponsetool

import (
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/sessionutils"
	"google.golang.org/adk/v2/internal/typeutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// AsLegacy returns a tool that behaves like t, but whose Declaration sets the
// typed Parameters (a *genai.Schema) and leaves ParametersJsonSchema nil. The
// caller's schema is not modified. If t is not a tool of this package, AsLegacy
// returns t unchanged. The returned tool implements the interface the flow runs
// (Name, Description, IsLongRunning, Declaration, Run) and ProcessRequest, which
// packs the typed declaration, so it can be listed in llmagent.Config.Tools as
// t can. It is safe for concurrent use.
//
// EXPERIMENTAL: AsLegacy is experimental; its API and behavior may change or be removed in a later version.
func AsLegacy(t tool.Tool) tool.Tool {
	st, ok := t.(*smrTool)
	if !ok || st == nil {
		return t
	}
	return &legacyTool{t: st}
}

// legacyTool declares the parameters of t in the typed form and forwards
// everything else to t. Like t, it holds nothing that a call changes.
type legacyTool struct {
	t *smrTool
}

func (l *legacyTool) Name() string { return l.t.Name() }

func (l *legacyTool) Description() string { return l.t.Description() }

func (l *legacyTool) IsLongRunning() bool { return l.t.IsLongRunning() }

func (l *legacyTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	return l.t.Run(ctx, args)
}

// ProcessRequest packs the typed declaration into the request, with the error
// of toolutils.PackTool for a duplicate name.
func (l *legacyTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, l)
}

// Declaration declares a record as a copy of its schema, and every other
// schema as a copy under the single required parameter "items" or "response",
// with every title cleared. A raw JSON schema cannot be written as a
// *genai.Schema, so its "response" parameter is an object of any shape, as in
// adk-python's typed form of a dict schema.
func (l *legacyTool) Declaration() *genai.FunctionDeclaration {
	decl := &genai.FunctionDeclaration{Name: Name, Description: description}
	switch l.t.shape {
	case shapeRecord:
		decl.Parameters = untitledCopy(l.t.schema)
	case shapeRecordList:
		decl.Parameters = wrapGenAI("items", untitledCopy(l.t.schema))
	case shapeValue:
		decl.Parameters = wrapGenAI("response", untitledCopy(l.t.schema))
	case shapeRaw:
		decl.Parameters = wrapGenAI("response", &genai.Schema{Type: genai.TypeObject})
	}
	return decl
}

// untitledCopy returns a deep copy of s with the Title cleared at every depth.
// The copy is made first, so s keeps its titles.
func untitledCopy(s *genai.Schema) *genai.Schema {
	c := sessionutils.CloneValue(s)
	typeutil.ClearTitles(c)
	return c
}
