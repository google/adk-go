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
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"
)

// description is adk-python's docstring of set_model_response, dedented.
const description = "Set your final response using the required output schema.\n\n" +
	"Use this tool to provide your final structured answer instead\n" +
	"of outputting text directly."

func (t *smrTool) Description() string { return description }

// Declaration declares a record's own schema as the parameters, unchanged, and
// wraps every other schema in a single required parameter: "items" for a
// record list, "response" otherwise.
func (t *smrTool) Declaration() *genai.FunctionDeclaration {
	decl := &genai.FunctionDeclaration{Name: Name, Description: description}
	switch t.shape {
	case shapeRecord:
		decl.ParametersJsonSchema = t.schema
	case shapeRecordList:
		decl.ParametersJsonSchema = wrapGenAI("items", t.schema)
	case shapeValue:
		decl.ParametersJsonSchema = wrapGenAI("response", t.schema)
	case shapeRaw:
		decl.ParametersJsonSchema = wrapRaw(t.raw)
	}
	return decl
}

func wrapGenAI(param string, s *genai.Schema) *genai.Schema {
	return &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{param: s},
		Required:   []string{param},
	}
}

// wrapRaw moves the root definitions of s to the wrapper, so that references
// such as "#/$defs/Item" still resolve from the root of the parameters. s
// itself is not modified.
func wrapRaw(s *jsonschema.Schema) *jsonschema.Schema {
	inner := *s
	inner.Defs = nil
	inner.Definitions = nil
	return &jsonschema.Schema{
		Type:        "object",
		Properties:  map[string]*jsonschema.Schema{"response": &inner},
		Required:    []string{"response"},
		Defs:        s.Defs,
		Definitions: s.Definitions,
	}
}
