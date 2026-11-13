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

// Package setmodelresponsetool provides the tool a model calls to return its
// structured final answer.
//
// An LLM agent adds this tool to its requests by itself when four things
// hold: it has an output schema (llmagent.Config.OutputSchema or
// OutputJsonSchema), it has tools or toolsets, it is not in task mode, and
// its model does not declare model.Capabilities.OutputSchemaAndTools. The
// model can then call the other tools and gives its final answer by calling
// set_model_response. Most programs never import this package. Use it to
// list the tool on an agent without an output schema, or to read the
// declaration the model receives.
//
// New builds the tool for a *genai.Schema, and NewFromJSONSchema for a raw
// JSON schema. The schema decides the parameters and how an answer is
// checked:
//
//   - An object schema with properties declares one parameter per property.
//     The answer is validated and normalized as adk-python's pydantic
//     validation does in lax mode: a missing required property is an issue,
//     an absent property with a Default gets the default, a property the
//     schema does not list is dropped, a null for an optional nullable
//     property is dropped, a string that holds a number is converted for an
//     INTEGER or NUMBER property, and a schema with AnyOf arms is checked as
//     a union.
//   - An array whose items are such objects declares one parameter, "items",
//     and each element is validated as above.
//   - Any other *genai.Schema, and every raw JSON schema, declares one
//     parameter, "response". Its value is accepted as the model sent it.
//
// The declaration carries the parameters as a JSON schema
// (FunctionDeclaration.ParametersJsonSchema); a titled object below the
// root appears once, under "$defs". AsLegacy wraps a tool so that it
// declares the typed form (FunctionDeclaration.Parameters) instead, and
// feature.WithLegacyFunctionDeclarationSchema in package
// google.golang.org/adk/v2/exp/feature makes the tool an agent adds do the
// same.
//
// A valid answer is stored in session.EventActions.SetModelResponse of the
// function response event, and the agent then ends its turn with a model
// text event that holds the answer as JSON. An invalid answer stores
// nothing: its function response is {"error": feedback}, where the feedback
// names each issue in the form of adk-python's validation errors, and the
// model is called again.
//
// List the tool in llmagent.Config.Tools only for an agent without an
// OutputSchema: an agent with one adds its own set_model_response tool when it
// needs one, and a second tool of that name fails the run as a duplicate.
package setmodelresponsetool

import (
	"errors"
	"fmt"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/exp/feature"
	"google.golang.org/adk/v2/internal/sessionutils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// Name is the name of the tool, "set_model_response".
const Name = "set_model_response"

// New returns the set_model_response tool for an output schema given as a
// *genai.Schema, or an error if schema is nil. The tool keeps schema and
// does not modify it; the caller must not modify it while the tool is in use.
// The tool is safe for concurrent use.
func New(schema *genai.Schema) (tool.Tool, error) {
	if schema == nil {
		return nil, errors.New("set_model_response: schema is nil")
	}
	return &smrTool{shape: shapeOf(schema), schema: schema}, nil
}

// NewFromJSONSchema returns the set_model_response tool for an output schema
// given as a raw JSON schema, or an error if schema is nil. The tool declares
// schema as its single parameter, "response". It does not validate an answer
// against schema; it accepts the model's response argument as it is, as
// adk-python does. The tool keeps schema and does not modify it; the caller
// must not modify it while the tool is in use. The tool is safe for
// concurrent use.
func NewFromJSONSchema(schema *jsonschema.Schema) (tool.Tool, error) {
	if schema == nil {
		return nil, errors.New("set_model_response: schema is nil")
	}
	return &smrTool{shape: shapeRaw, raw: schema}, nil
}

// shape is how the tool declares and checks an answer for its schema.
type shape int

const (
	shapeRecord     shape = iota // OBJECT with properties: one parameter per property, validated
	shapeRecordList              // ARRAY whose items are records: parameter "items", each item validated
	shapeValue                   // any other genai schema: parameter "response", not validated
	shapeRaw                     // a raw JSON schema: parameter "response", not validated
)

func isRecord(s *genai.Schema) bool {
	return s != nil && upperType(s) == genai.TypeObject && len(s.Properties) > 0
}

func shapeOf(s *genai.Schema) shape {
	switch {
	case isRecord(s):
		return shapeRecord
	case upperType(s) == genai.TypeArray && isRecord(s.Items):
		return shapeRecordList
	}
	return shapeValue
}

// upperType reads the schema's type in any case, as the model and the
// developer may write "object" or "OBJECT".
func upperType(s *genai.Schema) genai.Type {
	return genai.Type(strings.ToUpper(string(s.Type)))
}

// smrTool is immutable after construction, so one value serves concurrent
// runs; Run writes only to the agent.Context it is given.
type smrTool struct {
	shape  shape
	schema *genai.Schema      // set unless shape is shapeRaw
	raw    *jsonschema.Schema // set only for shapeRaw
}

func (t *smrTool) Name() string { return Name }

func (t *smrTool) IsLongRunning() bool { return false }

// ProcessRequest packs the tool's declaration into the request: the typed form
// of AsLegacy when feature.LegacyFunctionDeclarationSchema(ctx) is true, and
// Declaration otherwise. It returns the error of toolutils.PackTool when req
// already holds a tool named set_model_response.
func (t *smrTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	var declared toolutils.Tool = t
	if feature.LegacyFunctionDeclarationSchema(ctx) {
		declared = &legacyTool{t: t}
	}
	return toolutils.PackTool(req, declared)
}

// Run checks the model's answer. A valid answer is stored, as a clone, in
// ctx.Actions().SetModelResponse and returned: a map as it is, anything else
// as {"result": answer}. An invalid answer returns the feedback for the model
// as the error and stores nothing.
func (t *smrTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type for set_model_response: %T", args)
	}
	answer, err := t.answer(m)
	if err != nil {
		return nil, err
	}
	// The clone keeps the stored answer and the function response apart:
	// parallel calls' actions are merged in place later.
	ctx.Actions().SetModelResponse = sessionutils.CloneValue(answer)
	if am, ok := answer.(map[string]any); ok {
		return am, nil
	}
	return map[string]any{"result": answer}, nil
}

// answer returns the normalized answer in args, never nil, or a
// *validationError.
func (t *smrTool) answer(args map[string]any) (any, error) {
	switch t.shape {
	case shapeRecord:
		return checked(titleOf(t.schema), args, t.schema)
	case shapeRecordList:
		items, present := args["items"]
		if !present {
			return nil, missingParameter("items")
		}
		title := "list[" + titleOf(t.schema.Items) + "]"
		// A null items is never an answer, even when the root is nullable:
		// accepting it would store nothing and leave the turn unfinished.
		if items == nil {
			return nil, &validationError{title: title, issues: []issue{{msg: message("list_type"), kind: "list_type"}}}
		}
		return checked(title, items, t.schema)
	default:
		// Python validates neither form (set_model_response_tool.py:315-318).
		v := args["response"]
		if v == nil {
			return nil, missingParameter("response")
		}
		return v, nil
	}
}

// checked normalizes v against s and reports every issue under title.
func checked(title string, v any, s *genai.Schema) (any, error) {
	var issues []issue
	out := normalize(v, s, nil, &issues)
	if len(issues) > 0 {
		return nil, &validationError{title: title, issues: issues}
	}
	return out, nil
}

// titleOf names a record in feedback, as pydantic names the model class.
func titleOf(s *genai.Schema) string {
	if s.Title == "" {
		return "object"
	}
	return s.Title
}
