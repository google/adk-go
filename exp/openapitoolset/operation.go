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

package openapitoolset

import (
	"errors"
	"fmt"
)

// apiParameter is one argument of a REST API tool: adk-python's ApiParameter
// (tools/openapi_tool/common/common.py).
type apiParameter struct {
	// originalName is the name in the spec, and location is where the value
	// goes: path, query, header, cookie or body.
	originalName string
	location     string
	// pyName is the argument name the model sees, adk-python's py_name.
	pyName string
	// schema is the argument's JSON schema as it appears in the declaration.
	schema   map[string]any
	required bool
	// style and explode are a query parameter's OpenAPI serialization, left
	// at "" and nil for the location's default.
	style   string
	explode *bool
}

// defaultParamNames name an argument whose spec name yields no usable name,
// such as the body of a scalar request.
var defaultParamNames = map[string]string{
	"body":   "body",
	"query":  "query_param",
	"path":   "path_param",
	"header": "header_param",
	"cookie": "cookie_param",
}

// newAPIParameter names the argument the way adk-python's ApiParameter does:
// the spec name with Python keywords renamed when preserveName is set, else
// (or when that is empty) the spec name in snake_case, else a default for the
// location.
func newAPIParameter(originalName, location string, schema map[string]any, required, preserveName bool) *apiParameter {
	var pyName string
	if preserveName {
		pyName = renameKeyword(originalName)
	}
	if pyName == "" {
		pyName = renameKeyword(snakeCase(originalName))
	}
	if pyName == "" {
		pyName = defaultParamNames[location]
	}
	if pyName == "" {
		pyName = "value"
	}
	return &apiParameter{
		originalName: originalName,
		location:     location,
		pyName:       pyName,
		schema:       schema,
		required:     required,
	}
}

// requestBodyInfo describes how an operation's request body is sent: the media
// type the body is encoded as, and whether its schema is an object (whose
// properties are spread across several arguments) rather than a single value.
type requestBodyInfo struct {
	mediaType string
	object    bool
}

// parseOperation returns the arguments of an OpenAPI operation object: its
// parameters, then its request body, with clashing names made unique, plus how
// the request body is sent. It is adk-python's OperationParser
// (openapi_spec_parser/operation_parser.py). The body info is nil when the
// operation has no request body.
func parseOperation(op *object, preserveNames bool) ([]*apiParameter, *requestBodyInfo, error) {
	params, err := operationParameters(op, preserveNames)
	if err != nil {
		return nil, nil, err
	}
	body, info, err := requestBodyParameters(op, preserveNames)
	if err != nil {
		return nil, nil, err
	}
	params = append(params, body...)
	dedupeParamNames(params)
	return params, info, nil
}

var parameterLocations = map[string]bool{"query": true, "header": true, "path": true, "cookie": true}

func operationParameters(op *object, preserveNames bool) ([]*apiParameter, error) {
	raw := op.get("parameters")
	if raw == nil {
		return nil, nil
	}
	list, ok := raw.([]any)
	if !ok {
		return nil, errors.New("parameters must be a list")
	}
	var params []*apiParameter
	for i, item := range list {
		p, ok := item.(*object)
		if !ok {
			return nil, fmt.Errorf("parameter %d must be an object", i)
		}
		// A reference left unresolved is skipped rather than rejected, as in
		// adk-python: one dangling pointer must not take down every tool.
		if _, isRef := p.get("$ref").(string); isRef {
			continue
		}
		name, ok := p.get("name").(string)
		if !ok {
			return nil, fmt.Errorf("parameter %d has no name", i)
		}
		in, _ := p.get("in").(string)
		if !parameterLocations[in] {
			return nil, fmt.Errorf("parameter %q has an unsupported location", name)
		}
		s, err := schemaObject(p.get("schema"))
		if err != nil {
			return nil, fmt.Errorf("parameter %q: %w", name, err)
		}
		schema := plainSchema(s).(map[string]any)
		if d, _ := schema["description"].(string); d == "" {
			// adk-python writes the parameter's description into its schema,
			// as "" when there is none.
			description, _ := p.get("description").(string)
			schema["description"] = description
		}
		// OpenAPI requires every path parameter, and the URL cannot be built
		// without one, so it is required even when the spec leaves the flag out.
		required := in == "path" || p.get("required") == true
		param := newAPIParameter(name, in, schema, required, preserveNames)
		if style, ok := p.get("style").(string); ok {
			param.style = style
		}
		if explode, ok := p.get("explode").(bool); ok {
			param.explode = &explode
		}
		params = append(params, param)
	}
	return params, nil
}

// requestBodyParameters returns the arguments for the first media type of
// the request body, plus how it is sent: an object's properties, one argument
// "array" for an array, and one argument for anything else.
func requestBodyParameters(op *object, preserveNames bool) ([]*apiParameter, *requestBodyInfo, error) {
	raw := op.get("requestBody")
	if raw == nil {
		return nil, nil, nil
	}
	body, ok := raw.(*object)
	if !ok {
		return nil, nil, errors.New("requestBody must be an object")
	}
	if ref, ok := body.get("$ref").(string); ok {
		return nil, nil, fmt.Errorf("request body contains unresolved reference %q", ref)
	}
	content, _ := body.get("content").(*object)
	if content == nil || content.len() == 0 {
		return nil, nil, nil
	}
	mediaType := content.keys[0]
	media, ok := content.get(mediaType).(*object)
	if !ok {
		return nil, nil, fmt.Errorf("request body media type %q must be an object", mediaType)
	}
	schema, err := schemaObject(media.get("schema"))
	if err != nil {
		return nil, nil, fmt.Errorf("request body media type %q: %w", mediaType, err)
	}

	switch typ, _ := schema.get("type").(string); typ {
	case "object":
		info := &requestBodyInfo{mediaType: mediaType, object: true}
		props, _ := schema.get("properties").(*object)
		if props == nil {
			return nil, info, nil
		}
		required := map[string]bool{}
		if list, ok := schema.get("required").([]any); ok {
			for _, r := range list {
				if s, ok := r.(string); ok {
					required[s] = true
				}
			}
		}
		params := make([]*apiParameter, 0, props.len())
		for _, name := range props.keys {
			ps, err := schemaObject(props.get(name))
			if err != nil {
				return nil, nil, fmt.Errorf("request body property %q: %w", name, err)
			}
			params = append(params, newAPIParameter(name, "body", plainSchema(ps).(map[string]any), required[name], preserveNames))
		}
		return params, info, nil
	case "array":
		info := &requestBodyInfo{mediaType: mediaType, object: false}
		return []*apiParameter{newAPIParameter("array", "body", plainSchema(schema).(map[string]any), false, false)}, info, nil
	}
	// A body without a type, such as a oneOf, gets the name "body" rather
	// than an empty one; a scalar keeps adk-python's empty name, which also
	// ends up as "body".
	name := ""
	if truthy(schema.get("oneOf")) || truthy(schema.get("anyOf")) || truthy(schema.get("allOf")) || !truthy(schema.get("type")) {
		name = "body"
	}
	info := &requestBodyInfo{mediaType: mediaType, object: false}
	return []*apiParameter{newAPIParameter(name, "body", plainSchema(schema).(map[string]any), false, false)}, info, nil
}

// schemaObject returns the schema an OpenAPI field holds. No schema and the
// boolean schema true accept anything, so both are the empty schema; false
// accepts nothing, which no argument can satisfy.
func schemaObject(v any) (*object, error) {
	switch v := v.(type) {
	case nil:
		return newObject(), nil
	case bool:
		if v {
			return newObject(), nil
		}
		return nil, errors.New("uses an unsatisfiable false schema")
	case *object:
		return v, nil
	}
	return nil, fmt.Errorf("must be an OpenAPI schema, got %T", v)
}

// truthy reports whether v is true in Python's sense.
func truthy(v any) bool {
	switch v := v.(type) {
	case nil:
		return false
	case bool:
		return v
	case string:
		return v != ""
	case []any:
		return len(v) > 0
	case *object:
		return v.len() > 0
	case int:
		return v != 0
	case int64:
		return v != 0
	case float64:
		return v != 0
	}
	return true
}

// dedupeParamNames renames arguments whose names clash to name_0, name_1 and
// so on, in order. It is adk-python's _dedupe_param_names, which counts only
// the names it started with, so a renamed argument can still clash with one
// that was given that name in the spec.
func dedupeParamNames(params []*apiParameter) {
	counts := map[string]int{}
	for _, p := range params {
		n, seen := counts[p.pyName]
		if !seen {
			counts[p.pyName] = 0
			continue
		}
		counts[p.pyName] = n + 1
		p.pyName = fmt.Sprintf("%s_%d", p.pyName, n)
	}
}

// parametersSchema returns the JSON schema of a tool's arguments, as
// adk-python's get_json_schema builds it. required is always present, even
// when it is empty.
func parametersSchema(operationID string, params []*apiParameter) map[string]any {
	props := make(map[string]any, len(params))
	required := []string{}
	for _, p := range params {
		props[p.pyName] = p.schema
		if p.required {
			required = append(required, p.pyName)
		}
	}
	title := operationID
	if title == "" {
		title = "unnamed"
	}
	return map[string]any{
		"properties": props,
		"required":   required,
		"title":      title + "_Arguments",
		"type":       "object",
	}
}

// requiredSchemeName returns the name of the security scheme a security
// requirement list asks for: the first scheme of the first requirement. It
// returns "" when the list is empty or has an empty requirement, which is the
// OpenAPI way of making authentication optional; adk-python's
// _required_scheme_name then attaches no scheme, so the tool does not stop to
// ask for a credential.
func requiredSchemeName(security any) string {
	list, ok := security.([]any)
	if !ok || len(list) == 0 {
		return ""
	}
	for _, req := range list {
		if o, ok := req.(*object); !ok || o.len() == 0 {
			return ""
		}
	}
	return list[0].(*object).keys[0]
}
