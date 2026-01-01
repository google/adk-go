// Copyright 2025 Google LLC
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
	"fmt"
	"strings"
)

// ParsedOperation represents a parsed OpenAPI operation.
type ParsedOperation struct {
	// Name is the operation ID or generated name.
	Name string
	// Description is the operation description.
	Description string
	// Method is the HTTP method (GET, POST, etc.).
	Method string
	// Path is the URL path with parameter placeholders.
	Path string
	// BaseURL is the server base URL.
	BaseURL string
	// Parameters are the operation parameters.
	Parameters []Parameter
	// RequestBody describes the request body if present.
	RequestBody *RequestBody
	// Responses describes the expected responses.
	Responses map[string]*Response
}

// Parameter represents an API parameter.
type Parameter struct {
	Name        string
	In          string // "path", "query", "header", "cookie"
	Description string
	Required    bool
	Schema      map[string]any
}

// RequestBody represents a request body specification.
type RequestBody struct {
	Description string
	Required    bool
	Content     map[string]MediaType
}

// MediaType represents a media type specification.
type MediaType struct {
	Schema map[string]any
}

// Response represents an API response.
type Response struct {
	Description string
	Content     map[string]MediaType
}

// parseOpenAPISpec parses an OpenAPI specification into RestApiTools.
func parseOpenAPISpec(spec map[string]any) ([]*RestApiTool, error) {
	var tools []*RestApiTool

	// Extract base URL from servers
	baseURL := ""
	if servers, ok := spec["servers"].([]any); ok && len(servers) > 0 {
		if server, ok := servers[0].(map[string]any); ok {
			if url, ok := server["url"].(string); ok {
				baseURL = strings.TrimSuffix(url, "/")
			}
		}
	}

	// Parse paths
	paths, ok := spec["paths"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("no paths found in OpenAPI spec")
	}

	for path, pathItem := range paths {
		pathItemMap, ok := pathItem.(map[string]any)
		if !ok {
			continue
		}

		// Parse each HTTP method
		for method, operation := range pathItemMap {
			// Skip non-operation fields
			if method == "parameters" || method == "servers" || method == "$ref" {
				continue
			}

			op, ok := operation.(map[string]any)
			if !ok {
				continue
			}

			parsed, err := parseOperation(path, method, op, baseURL, pathItemMap)
			if err != nil {
				continue // Skip invalid operations
			}

			tool := newRestApiToolFromParsed(parsed)
			tools = append(tools, tool)
		}
	}

	return tools, nil
}

// parseOperation parses a single OpenAPI operation.
func parseOperation(path, method string, op map[string]any, baseURL string, pathItem map[string]any) (*ParsedOperation, error) {
	parsed := &ParsedOperation{
		Path:    path,
		Method:  strings.ToUpper(method),
		BaseURL: baseURL,
	}

	// Get operation ID or generate name
	if opID, ok := op["operationId"].(string); ok {
		parsed.Name = opID
	} else {
		// Generate name from method and path
		parsed.Name = generateOperationName(method, path)
	}

	// Get description
	if desc, ok := op["description"].(string); ok {
		parsed.Description = desc
	} else if summary, ok := op["summary"].(string); ok {
		parsed.Description = summary
	}

	// Parse parameters
	parsed.Parameters = parseParameters(op, pathItem)

	// Parse request body
	if reqBody, ok := op["requestBody"].(map[string]any); ok {
		parsed.RequestBody = parseRequestBody(reqBody)
	}

	// Parse responses
	if responses, ok := op["responses"].(map[string]any); ok {
		parsed.Responses = parseResponses(responses)
	}

	return parsed, nil
}

// generateOperationName generates an operation name from method and path.
func generateOperationName(method, path string) string {
	// Convert path to snake_case name
	name := strings.ReplaceAll(path, "/", "_")
	name = strings.ReplaceAll(name, "{", "")
	name = strings.ReplaceAll(name, "}", "")
	name = strings.ReplaceAll(name, "-", "_")
	name = strings.Trim(name, "_")
	return strings.ToLower(method) + "_" + name
}

// parseParameters parses operation and path-level parameters.
func parseParameters(op map[string]any, pathItem map[string]any) []Parameter {
	var params []Parameter

	// Parse path-level parameters
	if pathParams, ok := pathItem["parameters"].([]any); ok {
		params = append(params, parseParameterList(pathParams)...)
	}

	// Parse operation-level parameters (override path-level)
	if opParams, ok := op["parameters"].([]any); ok {
		params = append(params, parseParameterList(opParams)...)
	}

	return params
}

// parseParameterList parses a list of parameters.
func parseParameterList(paramList []any) []Parameter {
	var params []Parameter
	for _, p := range paramList {
		pm, ok := p.(map[string]any)
		if !ok {
			continue
		}

		param := Parameter{
			Name: getString(pm, "name"),
			In:   getString(pm, "in"),
		}
		if desc, ok := pm["description"].(string); ok {
			param.Description = desc
		}
		if required, ok := pm["required"].(bool); ok {
			param.Required = required
		}
		if schema, ok := pm["schema"].(map[string]any); ok {
			param.Schema = schema
		}
		params = append(params, param)
	}
	return params
}

// parseRequestBody parses a request body specification.
func parseRequestBody(reqBody map[string]any) *RequestBody {
	rb := &RequestBody{}
	if desc, ok := reqBody["description"].(string); ok {
		rb.Description = desc
	}
	if required, ok := reqBody["required"].(bool); ok {
		rb.Required = required
	}
	if content, ok := reqBody["content"].(map[string]any); ok {
		rb.Content = make(map[string]MediaType)
		for mediaType, mtSpec := range content {
			mt := MediaType{}
			if mtMap, ok := mtSpec.(map[string]any); ok {
				if schema, ok := mtMap["schema"].(map[string]any); ok {
					mt.Schema = schema
				}
			}
			rb.Content[mediaType] = mt
		}
	}
	return rb
}

// parseResponses parses response specifications.
func parseResponses(responses map[string]any) map[string]*Response {
	result := make(map[string]*Response)
	for code, resp := range responses {
		respMap, ok := resp.(map[string]any)
		if !ok {
			continue
		}
		r := &Response{}
		if desc, ok := respMap["description"].(string); ok {
			r.Description = desc
		}
		if content, ok := respMap["content"].(map[string]any); ok {
			r.Content = make(map[string]MediaType)
			for mediaType, mtSpec := range content {
				mt := MediaType{}
				if mtMap, ok := mtSpec.(map[string]any); ok {
					if schema, ok := mtMap["schema"].(map[string]any); ok {
						mt.Schema = schema
					}
				}
				r.Content[mediaType] = mt
			}
		}
		result[code] = r
	}
	return result
}

// getString safely gets a string from a map.
func getString(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}
