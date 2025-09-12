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

package mcptool

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

type getSessionFunc func(ctx context.Context) (*mcp.ClientSession, error)

func convertTool(t *mcp.Tool, getSessionFunc getSessionFunc) (tool.Tool, error) {
	return &mcpTool{
		name:        t.Name,
		description: t.Description,
		funcDeclaration: &genai.FunctionDeclaration{
			Name:                 t.Name,
			Description:          t.Description,
			ParametersJsonSchema: t.InputSchema,
			ResponseJsonSchema:   t.OutputSchema,
		},
		getSessionFunc: getSessionFunc,
	}, nil
}

type mcpTool struct {
	name            string
	description     string
	funcDeclaration *genai.FunctionDeclaration

	getSessionFunc getSessionFunc
}

func (t *mcpTool) Name() string {
	return t.name
}

func (t *mcpTool) Description() string {
	return t.description
}

func (t *mcpTool) IsLongRunning() bool {
	return false
}

func (t *mcpTool) ProcessRequest(ctx tool.Context, req *llm.Request) error {
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}

	name := t.Name()
	if _, ok := req.Tools[name]; ok {
		return fmt.Errorf("duplicate tool: %q", name)
	}
	req.Tools[name] = t

	if req.GenerateConfig == nil {
		req.GenerateConfig = &genai.GenerateContentConfig{}
	}
	req.GenerateConfig.Tools = append(req.GenerateConfig.Tools, &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			t.funcDeclaration,
		},
	})

	return nil
}

func (t *mcpTool) Declaration() *genai.FunctionDeclaration {
	return t.funcDeclaration
}

func (t *mcpTool) Run(ctx tool.Context, args any) (any, error) {
	session, err := t.getSessionFunc(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get session: %w", err)
	}

	// TODO: add auth
	res, err := session.CallTool(ctx, &mcp.CallToolParams{
		Name:      t.name,
		Arguments: args,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to call MCP tool %q with err: %w", t.name, err)
	}

	return t.parseCallResult(res), nil
}

func (t *mcpTool) parseCallResult(res *mcp.CallToolResult) map[string]any {
	if res == nil {
		return map[string]any{"error": "MCP framework error: CallToolResult was null"}
	}

	if res.IsError {
		details := strings.Builder{}
		for _, c := range res.Content {
			textContent, ok := c.(*mcp.TextContent)
			if !ok {
				continue
			}
			if _, err := details.WriteString(textContent.Text); err != nil {
				return map[string]any{"error": fmt.Sprintf("failed to parse error details: %q", err.Error())}
			}
		}

		errMsg := "Tool execution failed."
		if details.Len() > 0 {
			errMsg += " Details: " + details.String()
		}

		return map[string]any{
			"error": errMsg,
		}
	}

	mapRes := make(map[string]any)

	for _, c := range res.Content {
		b, err := c.MarshalJSON()
		if err != nil {
			return map[string]any{"error": fmt.Sprintf("failed to marshal response: %q", err.Error())}
		}

		var m map[string]any
		if err := json.Unmarshal(b, &m); err != nil {
			return map[string]any{"error": fmt.Sprintf("failed to unmarshal response: %q", err.Error())}
		}

		maps.Copy(mapRes, m)
	}

	return mapRes
}

var (
	_ toolinternal.FunctionTool     = (*mcpTool)(nil)
	_ toolinternal.RequestProcessor = (*mcpTool)(nil)
)
