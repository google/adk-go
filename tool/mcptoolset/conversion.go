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
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/schemautil"
	"google.golang.org/adk/v2/tool"
)

// ToMCPTool converts t into an [mcp.Tool] definition suitable for registration
// on an [mcp.Server] with [mcp.Server.AddTool].
//
// If t implements Declaration() *genai.FunctionDeclaration, ToMCPTool sets
// InputSchema from ParametersJsonSchema when present, or converts Parameters
// into a JSON Schema map. When t has no declaration or declares neither
// parameter field, InputSchema defaults to {"type": "object"}.
//
// MCP tool arguments are always a JSON object, and [mcp.Server.AddTool] panics
// unless InputSchema has type "object". ToMCPTool therefore sets the top-level
// type to "object" when the schema leaves it out (for example an empty schema
// or a typed-nil one) or allows it among other types (for example
// ["null", "object"], which functiontool infers for a pointer argument). It
// returns an error when the schema is not a JSON object or its top-level type
// excludes "object", so a returned tool is always accepted by
// [mcp.Server.AddTool].
//
// ToMCPTool builds only the tool definition. The caller supplies the
// [mcp.ToolHandler] that runs the tool's logic when the server receives a
// tools/call request:
//
//	mcpTool, err := mcptoolset.ToMCPTool(adkTool)
//	if err != nil {
//		return err
//	}
//	server.AddTool(mcpTool, func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
//		// Decode req.Params.Arguments and run the tool's implementation.
//	})
func ToMCPTool(t tool.Tool) (*mcp.Tool, error) {
	if t == nil {
		return nil, errors.New("mcptoolset: tool is nil")
	}

	// Diverges from adk-python's adk_to_mcp_tool_type, which returns {} when a
	// tool has no declaration or parameters: go-sdk's Server.AddTool panics
	// unless InputSchema has type "object".
	inputSchema := any(map[string]any{"type": "object"})
	if d, ok := t.(interface {
		Declaration() *genai.FunctionDeclaration
	}); ok {
		if decl := d.Declaration(); decl != nil {
			switch {
			case decl.ParametersJsonSchema != nil:
				inputSchema = decl.ParametersJsonSchema
			case decl.Parameters != nil:
				params := decl.Parameters
				if params.Type == "" || params.Type == genai.TypeUnspecified {
					// MCP tool arguments are always an object. Treat an untyped
					// top level as OBJECT so its properties survive conversion,
					// where schemautil.GeminiToJSONSchema would drop them.
					objectParams := *params
					objectParams.Type = genai.TypeObject
					params = &objectParams
				}
				schema, err := schemautil.GeminiToJSONSchema(params)
				if err != nil {
					return nil, fmt.Errorf("mcptoolset: convert parameters schema for tool %q: %w", t.Name(), err)
				}
				inputSchema = schema
			}
		}
	}

	inputSchema, err := objectInputSchema(inputSchema)
	if err != nil {
		return nil, fmt.Errorf("mcptoolset: input schema for tool %q: %w", t.Name(), err)
	}

	return &mcp.Tool{
		Name:        t.Name(),
		Description: t.Description(),
		InputSchema: inputSchema,
	}, nil
}

// objectInputSchema returns schema in a form that [mcp.Server.AddTool] accepts:
// a value that marshals to a JSON object whose "type" is "object". A schema
// that already satisfies this is returned unchanged. A schema that omits the
// top-level type, or lists "object" among several types, is returned as a copy
// with type "object". Any other schema is an error. The caller's schema is
// never modified.
func objectInputSchema(schema any) (any, error) {
	if s, ok := schema.(*jsonschema.Schema); ok && s != nil && s.Type == "object" {
		return s, nil
	}
	data, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("marshal schema: %w", err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("schema is not a JSON object: %w", err)
	}
	if _, isJSONSchema := schema.(*jsonschema.Schema); !isJSONSchema && m["type"] == "object" {
		return schema, nil
	}
	switch typ := m["type"].(type) {
	case nil:
		// No top-level type constraint: arguments are an object anyway.
	case string:
		if typ != "object" {
			return nil, fmt.Errorf("schema type %q is not \"object\"", typ)
		}
	case []any:
		if !slices.Contains(typ, any("object")) {
			return nil, fmt.Errorf("schema types %v do not include \"object\"", typ)
		}
	default:
		return nil, fmt.Errorf("schema type %v is not \"object\"", typ)
	}
	if m == nil {
		m = map[string]any{}
	}
	m["type"] = "object"
	return m, nil
}
