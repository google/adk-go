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
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"google.golang.org/adk/v2/tool/toolauth"
)

// parsedOperation is one operation of a spec, ready to become a tool:
// adk-python's ParsedOperation (openapi_spec_parser/openapi_spec_parser.py).
type parsedOperation struct {
	// name is the snake_case operationId, at most maxToolNameLength long.
	name        string
	description string
	endpoint    Endpoint
	operationID string
	params      []*apiParameter
	// body describes how the request body is sent, or nil when there is none.
	body *requestBodyInfo
	// authScheme is the scheme the operation's security requirement names,
	// or nil when it needs none.
	authScheme *toolauth.Scheme
}

// httpMethods are the operation keys of a path item, in the order
// adk-python visits them.
var httpMethods = []string{"get", "post", "put", "delete", "patch", "head", "options", "trace"}

// parseSpec turns an OpenAPI spec into its operations. It is adk-python's
// OpenApiSpecParser.parse: references are resolved, invalid schema types are
// dropped, and then each method of each path is parsed. spec is not modified.
func parseSpec(spec *object, preserveNames bool) ([]*parsedOperation, error) {
	resolved, err := resolveRefs(spec)
	if err != nil {
		return nil, err
	}
	sanitizeSchemaTypes(resolved, false)
	return collectOperations(resolved, preserveNames)
}

func collectOperations(spec *object, preserveNames bool) ([]*parsedOperation, error) {
	// The first server's URL, as adk-python takes it.
	var baseURL string
	if servers, ok := spec.get("servers").([]any); ok && len(servers) > 0 {
		if s, ok := servers[0].(*object); ok {
			baseURL, _ = s.get("url").(string)
		}
	}
	globalScheme := requiredSchemeName(spec.get("security"))
	var schemes *object
	if components, ok := spec.get("components").(*object); ok {
		schemes, _ = components.get("securitySchemes").(*object)
	}
	paths, _ := spec.get("paths").(*object)
	if paths == nil {
		return nil, nil
	}

	var ops []*parsedOperation
	for _, path := range paths.keys {
		raw := paths.get(path)
		if raw == nil {
			continue
		}
		pathItem, ok := raw.(*object)
		if !ok {
			return nil, fmt.Errorf("path %q must be an object", path)
		}
		for _, method := range httpMethods {
			raw := pathItem.get(method)
			if raw == nil {
				continue
			}
			op, ok := raw.(*object)
			if !ok {
				return nil, fmt.Errorf("%s %s: operation must be an object", strings.ToUpper(method), path)
			}
			parsed, err := parseSpecOperation(op, pathItem, path, method, baseURL, globalScheme, schemes, preserveNames)
			if err != nil {
				return nil, fmt.Errorf("%s %s: %w", strings.ToUpper(method), path, err)
			}
			ops = append(ops, parsed)
		}
	}
	return ops, nil
}

func parseSpecOperation(op, pathItem *object, path, method, baseURL, globalScheme string, schemes *object, preserveNames bool) (*parsedOperation, error) {
	// Each operation gets its own copy: the path item's parameters are added
	// to it, and the path item is shared by its methods.
	op = cloneValue(op).(*object)

	// Path-level parameters are appended, except those an operation-level
	// parameter of the same name and location overrides, so the model is not
	// asked for the same value twice.
	var opParams []any
	if raw := op.get("parameters"); raw != nil {
		var ok bool
		if opParams, ok = raw.([]any); !ok {
			return nil, errors.New("parameters must be a list")
		}
	}
	type paramKey struct{ name, in string }
	keyOf := func(p any) (paramKey, bool) {
		po, ok := p.(*object)
		if !ok {
			return paramKey{}, false
		}
		name, ok := po.get("name").(string)
		in, _ := po.get("in").(string)
		return paramKey{name, in}, ok
	}
	overridden := map[paramKey]bool{}
	for _, p := range opParams {
		if k, ok := keyOf(p); ok {
			overridden[k] = true
		}
	}
	params := append([]any{}, opParams...)
	pathParams, _ := pathItem.get("parameters").([]any)
	for _, p := range pathParams {
		if k, ok := keyOf(p); ok && overridden[k] {
			continue
		}
		params = append(params, cloneValue(p))
	}
	op.set("parameters", params)

	if _, ok := op.lookup("operationId"); !ok {
		op.set("operationId", snakeCase(path+"_"+method))
	}
	operationID, _ := op.get("operationId").(string)
	if operationID == "" {
		return nil, errors.New("operation ID is missing")
	}

	parameters, body, err := parseOperation(op, preserveNames)
	if err != nil {
		return nil, err
	}

	// An operation that declares security of its own overrides the global
	// requirement, including when it declares that none is needed.
	schemeName := requiredSchemeName(op.get("security"))
	if schemeName == "" && op.get("security") == nil {
		schemeName = globalScheme
	}
	var scheme *toolauth.Scheme
	if schemeName != "" && schemes != nil {
		if raw := schemes.get(schemeName); raw != nil {
			if scheme, err = decodeScheme(raw); err != nil {
				return nil, fmt.Errorf("security scheme %q: %w", schemeName, err)
			}
		}
	}

	description, _ := op.get("description").(string)
	if description == "" {
		description, _ = op.get("summary").(string)
	}
	return &parsedOperation{
		name:        truncate(snakeCase(operationID), maxToolNameLength),
		description: description,
		endpoint:    Endpoint{BaseURL: baseURL, Path: path, Method: method},
		operationID: operationID,
		params:      parameters,
		body:        body,
		authScheme:  scheme,
	}, nil
}

func decodeScheme(v any) (*toolauth.Scheme, error) {
	b, err := json.Marshal(plainSchema(v))
	if err != nil {
		return nil, err
	}
	var s toolauth.Scheme
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// resolveRefs returns a copy of spec with its local $ref references replaced
// by what they point at. It is adk-python's _resolve_references, quirks
// included: references are tracked across the whole document rather than
// along one path, so the second time a reference that is still being
// resolved (a cycle) or that pointed at nothing is met, it is replaced by its
// sibling keys alone. A reference to another document is an error.
func resolveRefs(spec *object) (*object, error) {
	r := &refResolver{doc: spec, seen: map[string]bool{}, cache: map[string]any{}}
	v, err := r.resolve(spec)
	if err != nil {
		return nil, err
	}
	return v.(*object), nil
}

type refResolver struct {
	doc   *object
	seen  map[string]bool
	cache map[string]any
}

// resolve returns a resolved copy of v; it never returns part of r.doc.
func (r *refResolver) resolve(v any) (any, error) {
	switch v := v.(type) {
	case *object:
		ref, isRef := v.get("$ref").(string)
		if !isRef {
			out := newObject()
			for _, k := range v.keys {
				rv, err := r.resolve(v.vals[k])
				if err != nil {
					return nil, err
				}
				out.set(k, rv)
			}
			return out, nil
		}
		cached, isCached := r.cache[ref]
		if r.seen[ref] && !isCached {
			out := cloneValue(v).(*object)
			out.delete("$ref")
			return out, nil
		}
		r.seen[ref] = true
		if isCached {
			return cloneValue(cached), nil
		}
		target, err := r.lookup(ref)
		if err != nil {
			return nil, err
		}
		if target == nil {
			return cloneValue(v), nil
		}
		resolved, err := r.resolve(target)
		if err != nil {
			return nil, err
		}
		r.cache[ref] = resolved
		return cloneValue(resolved), nil
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			rv, err := r.resolve(item)
			if err != nil {
				return nil, err
			}
			out[i] = rv
		}
		return out, nil
	}
	return v, nil
}

// lookup returns what a local reference points at, or nil when it points at
// nothing. Like adk-python, it splits the pointer on "/" and does not undo
// JSON Pointer escaping.
func (r *refResolver) lookup(ref string) (any, error) {
	parts := strings.Split(ref, "/")
	if parts[0] != "#" {
		return nil, fmt.Errorf("external references not supported: %q", ref)
	}
	var cur any = r.doc
	for _, part := range parts[1:] {
		o, ok := cur.(*object)
		if !ok {
			return nil, nil
		}
		if cur, ok = o.lookup(part); !ok {
			return nil, nil
		}
	}
	return cur, nil
}

// validSchemaTypes are the JSON Schema types OpenAPI 3.0 and 3.1 allow.
var validSchemaTypes = map[string]bool{
	"array": true, "boolean": true, "integer": true, "null": true,
	"number": true, "object": true, "string": true,
}

// sanitizeSchemaTypes fixes, in place, the type keyword of every object
// below a "schema" or "schemas" key: a valid type is lowercased, and an
// invalid one, such as the "Any" some generated specs use, is dropped. It is
// adk-python's _sanitize_schema_types. A type that is neither a string nor a
// list is left alone, so a property named "type" survives.
func sanitizeSchemaTypes(v any, inSchema bool) {
	switch v := v.(type) {
	case *object:
		if inSchema {
			sanitizeTypeField(v)
		}
		for _, k := range v.keys {
			sanitizeSchemaTypes(v.vals[k], inSchema || k == "schema" || k == "schemas")
		}
	case []any:
		for _, item := range v {
			sanitizeSchemaTypes(item, inSchema)
		}
	}
}

func sanitizeTypeField(o *object) {
	switch typ := o.get("type").(type) {
	case string:
		if t := strings.ToLower(typ); validSchemaTypes[t] {
			o.set("type", t)
		} else {
			o.delete("type")
		}
	case []any:
		var valid []any
		seen := map[string]bool{}
		for _, entry := range typ {
			s, ok := entry.(string)
			if !ok {
				continue
			}
			if t := strings.ToLower(s); validSchemaTypes[t] && !seen[t] {
				seen[t] = true
				valid = append(valid, t)
			}
		}
		if len(valid) > 0 {
			o.set("type", valid)
		} else {
			o.delete("type")
		}
	}
}
