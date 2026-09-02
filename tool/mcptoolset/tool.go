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

package mcptoolset

import (
	"errors"
	"fmt"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

func convertTool(t *mcp.Tool, client MCPClient, requireConfirmation bool, requireConfirmationProvider tool.ConfirmationProvider) (tool.Tool, error) {
	mcp := &mcpTool{
		name:        t.Name,
		description: t.Description,
		funcDeclaration: &genai.FunctionDeclaration{
			Name:        t.Name,
			Description: t.Description,
		},
		mcpClient:                   client,
		requireConfirmation:         requireConfirmation,
		requireConfirmationProvider: requireConfirmationProvider,
	}

	// Since t.InputSchema and t.OutputSchema are pointers (*jsonschema.Schema) and the destination ResponseJsonSchema
	// is an interface (any), we have encountered the type nil problem.
	// This will make the omitempty not work since ResponseJsonSchema becomes an interface wrapper
	// to a nil pointer and genai converter includes "responseJsonSchema": null in the json sent to the llm which causes it to crash.
	// we need the following "if" check to keep ResponseJsonSchema (nil,nil) instead of (*jsonschema.Schema, nil)
	if t.InputSchema != nil {
		mcp.funcDeclaration.ParametersJsonSchema = t.InputSchema
	}
	if t.OutputSchema != nil {
		mcp.funcDeclaration.ResponseJsonSchema = t.OutputSchema
	}
	return mcp, nil
}

type mcpTool struct {
	name            string
	description     string
	funcDeclaration *genai.FunctionDeclaration

	mcpClient MCPClient

	requireConfirmation bool

	requireConfirmationProvider tool.ConfirmationProvider
}

// Name implements the tool.Tool.
func (t *mcpTool) Name() string {
	return t.name
}

// Description implements the tool.Tool.
func (t *mcpTool) Description() string {
	return t.description
}

// IsLongRunning implements the tool.Tool.
func (t *mcpTool) IsLongRunning() bool {
	return false
}

func (t *mcpTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

func (t *mcpTool) Declaration() *genai.FunctionDeclaration {
	return t.funcDeclaration
}

func (t *mcpTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	if confirmation := ctx.ToolConfirmation(); confirmation != nil {
		if !confirmation.Confirmed {
			return nil, fmt.Errorf("error tool %q %w", t.Name(), tool.ErrConfirmationRejected)
		}
	} else {
		requireConfirmation := t.requireConfirmation

		// Only run the potentially expensive provider if the static flag didn't already trigger it
		// Provider takes precedence/overrides:
		if t.requireConfirmationProvider != nil {
			requireConfirmation = t.requireConfirmationProvider(t.Name(), args)
		}

		if requireConfirmation {
			err := ctx.RequestConfirmation(
				fmt.Sprintf("Please approve or reject the tool call %s() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					t.Name()), nil)
			if err != nil {
				return nil, err
			}
			ctx.Actions().SkipSummarization = true
			return nil, fmt.Errorf("error tool %q %w", t.Name(), tool.ErrConfirmationRequired)
		}
	}

	res, err := t.mcpClient.CallTool(ctx, &mcp.CallToolParams{
		Name:      t.name,
		Arguments: args,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to call MCP tool %q with err: %w", t.name, err)
	}

	if res.IsError {
		details := strings.Builder{}
		for _, c := range res.Content {
			textContent, ok := c.(*mcp.TextContent)
			if !ok {
				continue
			}
			if _, err := details.WriteString(textContent.Text); err != nil {
				return nil, fmt.Errorf("failed to write error details: %w", err)
			}
		}

		return nil, &ToolError{Details: details.String(), Meta: serverMeta(res.Meta)}
	}

	if res.StructuredContent != nil {
		return functionResponse(res, res.StructuredContent), nil
	}

	textResponse := strings.Builder{}

	for _, c := range res.Content {
		textContent, ok := c.(*mcp.TextContent)
		if !ok {
			continue
		}

		if _, err := textResponse.WriteString(textContent.Text); err != nil {
			return nil, fmt.Errorf("failed to write text response: %w", err)
		}
	}

	if textResponse.Len() == 0 {
		// A result that carries only metadata is a valid answer, for instance an
		// auth challenge from an MCP gateway, so it keeps the _meta passthrough
		// instead of becoming an error.
		if len(serverMeta(res.Meta)) == 0 {
			return nil, errors.New("no text content in tool response")
		}
		return functionResponse(res, ""), nil
	}

	return functionResponse(res, textResponse.String()), nil
}

// ToolError reports a tool result that the MCP server marked as an error.
// Callers reach it with errors.As to read the metadata the server attached to
// the failed call, which a plain error message cannot carry. The reachable
// caller is an entry of llmagent.Config.OnToolErrorCallbacks: the flow renders
// the error for the model as its message alone, so Meta, unlike the _meta of a
// successful result, never reaches the model.
type ToolError struct {
	// Details is the text content of the error result, empty when the server
	// sent none.
	Details string

	// Meta holds the metadata the server attached to the result, without keys
	// in prefixes the MCP protocol reserves for itself. It is nil when the
	// server attached no metadata of its own.
	Meta map[string]any
}

// Error implements error.
func (e *ToolError) Error() string {
	if e.Details == "" {
		return "Tool execution failed."
	}
	return "Tool execution failed. Details: " + e.Details
}

// functionResponse builds the function response map for a tool result.
// The map is the function response returned to the model, so anything placed
// in it reaches the LLM and is persisted to session and traces, in addition to
// being available to callbacks and the embedding application.
//
// Server metadata from the result's _meta field is preserved under the "_meta"
// key, minus keys in prefixes the MCP protocol reserves for itself. The key is
// absent when the server attached no metadata of its own.
func functionResponse(res *mcp.CallToolResult, output any) map[string]any {
	response := map[string]any{
		"output": output,
	}
	if meta := serverMeta(res.Meta); len(meta) > 0 {
		response["_meta"] = meta
	}
	return response
}

// serverMeta returns the entries of meta that the server itself attached,
// dropping keys reserved by the protocol. It returns nil when no entry
// qualifies.
func serverMeta(meta mcp.Meta) map[string]any {
	var serverKeys map[string]any
	for key, value := range meta {
		if isReservedMetaKey(key) {
			continue
		}
		if serverKeys == nil {
			serverKeys = make(map[string]any, len(meta))
		}
		serverKeys[key] = value
	}
	return serverKeys
}

// unprefixedReservedMetaKeys are the _meta keys the MCP protocol reserves
// without a prefix, as an exception to the prefix rule: the progress token of a
// request and the W3C trace context that carries it.
var unprefixedReservedMetaKeys = map[string]bool{
	"progressToken": true,
	"traceparent":   true,
	"tracestate":    true,
	"baggage":       true,
}

// isReservedMetaKey reports whether an MCP _meta key belongs to the protocol
// rather than to the server. A key is reserved when it is one of the
// unprefixed protocol keys, or when the second label of its prefix (the part
// before the first slash) is "modelcontextprotocol" or "mcp".
func isReservedMetaKey(key string) bool {
	if unprefixedReservedMetaKeys[key] {
		return true
	}
	// The prefix ends at the first slash, and a key name holds no slash, so a
	// later slash makes the key malformed rather than prefixed.
	slash := strings.Index(key, "/")
	if slash < 0 {
		return false
	}
	labels := strings.Split(key[:slash], ".")
	if len(labels) < 2 {
		return false
	}
	switch labels[1] {
	case "modelcontextprotocol", "mcp":
		return true
	}
	return false
}

var (
	_ toolinternal.FunctionTool     = (*mcpTool)(nil)
	_ toolinternal.RequestProcessor = (*mcpTool)(nil)
)
