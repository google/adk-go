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
	"google.golang.org/adk/v2/internal/toolinternal/toolutils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
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

	// InputSchema/OutputSchema are `any`; from the client they hold the server's
	// schema as map[string]any, or nil when absent. Assign only when present: a
	// nil value makes the genai converter emit a null "...JsonSchema", which it
	// rejects.
	if t.InputSchema != nil {
		mcp.funcDeclaration.ParametersJsonSchema = toolutils.SanitizeJSONSchemaForVertex(t.InputSchema)
	}
	if t.OutputSchema != nil {
		mcp.funcDeclaration.ResponseJsonSchema = toolutils.SanitizeJSONSchemaForVertex(t.OutputSchema)
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

	// TODO: add auth
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
