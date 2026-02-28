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

	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

func convertTool(t *mcp.Tool, client MCPClient, requireConfirmation bool, requireConfirmationProvider ConfirmationProvider, metadataProvider MetadataProvider) (tool.Tool, error) {
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
		metadataProvider:            metadataProvider,
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

	requireConfirmationProvider ConfirmationProvider

	metadataProvider MetadataProvider
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

func (t *mcpTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

func (t *mcpTool) Declaration() *genai.FunctionDeclaration {
	return t.funcDeclaration
}

func (t *mcpTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	if confirmation := ctx.ToolConfirmation(); confirmation != nil {
		if !confirmation.Confirmed {
			return nil, fmt.Errorf("error tool %q call is rejected", t.Name())
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
			return nil, fmt.Errorf("error tool %q requires confirmation, please approve or reject", t.Name())
		}
	}
	params := &mcp.CallToolParams{
		Name:      t.name,
		Arguments: args,
	}

	if t.metadataProvider != nil {
		if meta := t.metadataProvider(ctx); meta != nil {
			params.Meta = meta
		}
	}
	// TODO: add auth
	res, err := t.mcpClient.CallTool(ctx, params)
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

		errMsg := "Tool execution failed."
		if details.Len() > 0 {
			errMsg += " Details: " + details.String()
		}

		return nil, errors.New(errMsg)
	}

	if res.StructuredContent != nil {
		return map[string]any{
			"output": res.StructuredContent,
		}, nil
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
		return nil, errors.New("no text content in tool response")
	}

	return map[string]any{
		"output": textResponse.String(),
	}, nil
}

var (
	_ toolinternal.FunctionTool     = (*mcpTool)(nil)
	_ toolinternal.RequestProcessor = (*mcpTool)(nil)
)
