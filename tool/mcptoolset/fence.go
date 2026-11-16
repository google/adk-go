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

package mcptoolset

import (
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/fencing"
	"google.golang.org/adk/v2/internal/utils"
)

// fencedTool hands toolutils.PackTool the request's fenced declaration.
// ProcessRequest replaces it in req.Tools by the tool itself right after
// packing, so it never outlives the call.
type fencedTool struct {
	*mcpTool
	decl *genai.FunctionDeclaration
}

// Declaration returns the fenced declaration built for one request.
func (f *fencedTool) Declaration() *genai.FunctionDeclaration { return f.decl }

// fencedDeclaration returns a copy of d for the model, with the description,
// ParametersJsonSchema and ResponseJsonSchema fenced. d must be non-nil and is
// not modified. An absent schema stays absent, as in adk-python
// (tools/mcp_tool/mcp_tool.py:384, 393). The fields this package never sets
// (Parameters, Response, Behavior) are shared with d.
func fencedDeclaration(d *genai.FunctionDeclaration) *genai.FunctionDeclaration {
	c := *d
	c.Description = fencing.ToolDescription(d.Description)
	if d.ParametersJsonSchema != nil {
		c.ParametersJsonSchema = fencing.Schema(d.ParametersJsonSchema)
	}
	if d.ResponseJsonSchema != nil {
		c.ResponseJsonSchema = fencing.Schema(d.ResponseJsonSchema)
	}
	return &c
}

// plainTextInstruction reports whether si is what adk-python calls a str or
// None system instruction (tools/mcp_tool/mcp_tool.py:418-428): nil, or a
// Content whose parts are all text. An empty text part, and a Content with no
// parts, count as text, as Python's "" is a str. A part with any other payload,
// a thought, or a nil part makes it false: the preamble cannot describe such an
// instruction, and utils.AppendInstructions cannot append to a nil last part.
func plainTextInstruction(si *genai.Content) bool {
	if si == nil {
		return true
	}
	for _, p := range si.Parts {
		if p == nil {
			return false
		}
		// IsProsePart rejects empty text, which is still text here.
		q := *p
		q.Text = "x"
		if !utils.IsProsePart(&q) {
			return false
		}
	}
	return true
}

// hasPreamble reports whether si already carries fencing.ToolDescriptionPreamble.
// utils.AppendInstructions writes the preamble into one part, so it is never
// split across parts. Nil parts are skipped.
func hasPreamble(si *genai.Content) bool {
	if si == nil {
		return false
	}
	for _, p := range si.Parts {
		if p != nil && strings.Contains(p.Text, fencing.ToolDescriptionPreamble) {
			return true
		}
	}
	return false
}
