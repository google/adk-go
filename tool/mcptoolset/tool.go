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
	"mime"
	"strings"
	"unicode/utf8"

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
		details := formatMCPContent(res.Content)

		errMsg := "Tool execution failed."
		if details != "" {
			errMsg += " Details: " + details
		}

		return nil, errors.New(errMsg)
	}

	if res.StructuredContent != nil {
		return map[string]any{
			"output": res.StructuredContent,
		}, nil
	}

	content := formatMCPContent(res.Content)

	return map[string]any{
		"output": content,
	}, nil
}

type formattedMCPContent struct {
	text    string
	isPlain bool
}

// formatMCPContent renders MCP's ordered content blocks into the text-only
// response shape supported by FunctionTool.Run.
func formatMCPContent(contents []mcp.Content) string {
	formatted := make([]formattedMCPContent, 0, len(contents))
	for _, content := range contents {
		block := formattedMCPContent{isPlain: true}
		switch content := content.(type) {
		case nil:
			block.text = "[MCP content: unavailable]"
			block.isPlain = false
		case *mcp.TextContent:
			if content == nil {
				block.text = "[MCP text content: unavailable]"
				block.isPlain = false
			} else {
				block.text = content.Text
			}
		case *mcp.EmbeddedResource:
			block.text = formatEmbeddedResource(content)
			block.isPlain = false
		case *mcp.ResourceLink:
			block.text = formatResourceLink(content)
			block.isPlain = false
		case *mcp.ImageContent:
			if content == nil {
				block.text = "[MCP image: unavailable]"
			} else {
				block.text = formatMediaContent("image", content.MIMEType, len(content.Data))
			}
			block.isPlain = false
		case *mcp.AudioContent:
			if content == nil {
				block.text = "[MCP audio: unavailable]"
			} else {
				block.text = formatMediaContent("audio", content.MIMEType, len(content.Data))
			}
			block.isPlain = false
		default:
			block.text = "[MCP content: unsupported]"
			block.isPlain = false
		}
		formatted = append(formatted, block)
	}

	var result strings.Builder
	var previous *formattedMCPContent
	for i := range formatted {
		block := &formatted[i]
		if block.text == "" {
			continue
		}
		if previous != nil && (!previous.isPlain || !block.isPlain) &&
			!strings.HasSuffix(previous.text, "\n") && !strings.HasPrefix(block.text, "\n") {
			result.WriteByte('\n')
		}
		result.WriteString(block.text)
		previous = block
	}
	return result.String()
}

func formatEmbeddedResource(content *mcp.EmbeddedResource) string {
	if content == nil || content.Resource == nil {
		return "[MCP embedded resource: unavailable]"
	}

	resource := content.Resource
	attributes := resourceAttributes(resource.URI, resource.MIMEType)
	if resource.Text != "" {
		return formatContentWithBody("embedded resource", attributes, resource.Text)
	}
	if text, ok := decodeTextBlob(resource.Blob, resource.MIMEType); ok {
		return formatContentWithBody("embedded resource", attributes, text)
	}
	if len(resource.Blob) > 0 {
		attributes = append(attributes, fmt.Sprintf("size=%d bytes", len(resource.Blob)))
	}
	return formatContentLabel("embedded resource", attributes)
}

func formatResourceLink(content *mcp.ResourceLink) string {
	if content == nil {
		return "[MCP resource link: unavailable]"
	}

	attributes := resourceAttributes(content.URI, content.MIMEType)
	if content.Name != "" {
		attributes = append(attributes, fmt.Sprintf("name=%q", content.Name))
	}
	if content.Title != "" {
		attributes = append(attributes, fmt.Sprintf("title=%q", content.Title))
	}
	if content.Description != "" {
		attributes = append(attributes, fmt.Sprintf("description=%q", content.Description))
	}
	if content.Size != nil {
		attributes = append(attributes, fmt.Sprintf("size=%d bytes", *content.Size))
	}
	return formatContentLabel("resource link", attributes)
}

func formatMediaContent(kind, mimeType string, size int) string {
	attributes := make([]string, 0, 2)
	if mimeType != "" {
		attributes = append(attributes, fmt.Sprintf("mimeType=%q", mimeType))
	}
	attributes = append(attributes, fmt.Sprintf("size=%d bytes", size))
	return formatContentLabel(kind, attributes)
}

func resourceAttributes(uri, mimeType string) []string {
	attributes := make([]string, 0, 2)
	if uri != "" {
		attributes = append(attributes, fmt.Sprintf("uri=%q", uri))
	}
	if mimeType != "" {
		attributes = append(attributes, fmt.Sprintf("mimeType=%q", mimeType))
	}
	return attributes
}

func formatContentWithBody(kind string, attributes []string, body string) string {
	return formatContentLabel(kind, attributes) + "\n" + body
}

func formatContentLabel(kind string, attributes []string) string {
	if len(attributes) == 0 {
		return "[MCP " + kind + "]"
	}
	return "[MCP " + kind + ": " + strings.Join(attributes, ", ") + "]"
}

func decodeTextBlob(blob []byte, mimeType string) (string, bool) {
	if len(blob) == 0 {
		return "", false
	}

	mediaType, params, err := mime.ParseMediaType(mimeType)
	if err != nil && (!errors.Is(err, mime.ErrInvalidMediaParameter) || hasMIMECharsetParameter(mimeType)) {
		return "", false
	}
	if !isTextMediaType(mediaType) {
		return "", false
	}

	charset := strings.ToLower(params["charset"])
	switch charset {
	case "", "utf-8", "utf8":
		if !utf8.Valid(blob) {
			return "", false
		}
	case "us-ascii":
		for _, b := range blob {
			if b >= utf8.RuneSelf {
				return "", false
			}
		}
	default:
		return "", false
	}
	return string(blob), true
}

func hasMIMECharsetParameter(value string) bool {
	_, params, ok := strings.Cut(value, ";")
	for ok {
		var parameter string
		parameter, params, ok = strings.Cut(params, ";")
		name, _, _ := strings.Cut(parameter, "=")
		name = strings.ToLower(strings.Join(strings.Fields(name), ""))
		if name == "charset" || strings.HasPrefix(name, "charset*") {
			return true
		}
	}
	return false
}

func isTextMediaType(mediaType string) bool {
	if strings.HasPrefix(mediaType, "text/") || strings.HasSuffix(mediaType, "+json") || strings.HasSuffix(mediaType, "+xml") {
		return true
	}
	switch mediaType {
	case "application/json", "application/javascript", "application/toml", "application/xml",
		"application/x-yaml", "application/yaml":
		return true
	default:
		return false
	}
}

var (
	_ toolinternal.FunctionTool     = (*mcpTool)(nil)
	_ toolinternal.RequestProcessor = (*mcpTool)(nil)
)
