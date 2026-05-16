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

// Package minimax implements the [model.LLM] interface for MiniMax models
// using the OpenAI-compatible API.
package minimax

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

const (
	// DefaultBaseURL is the default MiniMax API base URL.
	DefaultBaseURL = "https://api.minimax.io"
	// apiKeyEnvVar is the environment variable for the MiniMax API key.
	apiKeyEnvVar = "MINIMAX_API_KEY"

	// DefaultModel is the default MiniMax model.
	DefaultModel = "MiniMax-M2.7"
	// HighSpeedModel is a faster variant of the default model.
	HighSpeedModel = "MiniMax-M2.7-highspeed"
)

// minimaxModel implements the model.LLM interface for MiniMax.
type minimaxModel struct {
	name       string
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// Config holds configuration for the MiniMax model.
type Config struct {
	// APIKey is the MiniMax API key. If empty, reads from MINIMAX_API_KEY env var.
	APIKey string
	// BaseURL is the MiniMax API base URL. Defaults to DefaultBaseURL.
	BaseURL string
	// HTTPClient is the HTTP client to use. Defaults to http.DefaultClient.
	// For testing only.
	HTTPClient *http.Client
}

// Option is a function that configures the MiniMax model.
type Option func(*Config)

// WithAPIKey sets the API key.
func WithAPIKey(apiKey string) Option {
	return func(c *Config) {
		c.APIKey = apiKey
	}
}

// WithBaseURL sets the base URL.
func WithBaseURL(baseURL string) Option {
	return func(c *Config) {
		c.BaseURL = baseURL
	}
}

// WithHTTPClient sets the HTTP client. For testing only.
func WithHTTPClient(client *http.Client) Option {
	return func(c *Config) {
		c.HTTPClient = client
	}
}

// NewModel creates and initializes a new MiniMax model that satisfies the
// model.LLM interface using the OpenAI-compatible MiniMax API.
//
// modelName should be one of the supported MiniMax model IDs, e.g.
// "MiniMax-M2.7" or "MiniMax-M2.7-highspeed".
//
// The API key is read from the MINIMAX_API_KEY environment variable if not
// provided via WithAPIKey.
func NewModel(modelName string, opts ...Option) (model.LLM, error) {
	cfg := &Config{}
	for _, opt := range opts {
		opt(cfg)
	}

	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(apiKeyEnvVar)
	}
	if apiKey == "" {
		return nil, fmt.Errorf("%s environment variable is not set", apiKeyEnvVar)
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	return &minimaxModel{
		name:       modelName,
		apiKey:     apiKey,
		baseURL:    baseURL,
		httpClient: httpClient,
	}, nil
}

func (m *minimaxModel) Name() string {
	return m.name
}

// GenerateContent calls the MiniMax OpenAI-compatible API.
func (m *minimaxModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req, stream)
		yield(resp, err)
	}
}

// openAIRequest is the request body for the OpenAI-compatible API.
type openAIRequest struct {
	Model       string          `json:"model"`
	Messages    []openAIMessage `json:"messages"`
	Temperature *float32        `json:"temperature,omitempty"`
	MaxTokens   int32           `json:"max_tokens,omitempty"`
	Tools       []openAITool    `json:"tools,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	TopP        *float32        `json:"top_p,omitempty"`
	Stop        []string        `json:"stop,omitempty"`
}

// openAIMessage represents a message in the OpenAI API format.
type openAIMessage struct {
	Role       string             `json:"role"`
	Content    any                `json:"content"`
	ToolCalls  []openAIToolCall   `json:"tool_calls,omitempty"`
	ToolCallID string             `json:"tool_call_id,omitempty"`
	Name       string             `json:"name,omitempty"`
}

// openAIToolCall represents a tool call in the OpenAI API format.
type openAIToolCall struct {
	ID       string              `json:"id"`
	Type     string              `json:"type"`
	Function openAIFunctionCall  `json:"function"`
}

// openAIFunctionCall holds the function call details.
type openAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// openAITool represents a tool definition in the OpenAI API format.
type openAITool struct {
	Type     string              `json:"type"`
	Function openAIFunctionDef   `json:"function"`
}

// openAIFunctionDef is the function definition in a tool.
type openAIFunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// openAIResponse is the response from the OpenAI-compatible API.
type openAIResponse struct {
	Choices []openAIChoice `json:"choices"`
	Usage   *openAIUsage   `json:"usage,omitempty"`
	Model   string         `json:"model,omitempty"`
	Error   *openAIError   `json:"error,omitempty"`
}

// openAIChoice is a single choice in the response.
type openAIChoice struct {
	Message      openAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason,omitempty"`
}

// openAIUsage holds token usage information.
type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// openAIError represents an API error.
type openAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    any    `json:"code"`
}

// generate calls the MiniMax API and returns a single LLMResponse.
func (m *minimaxModel) generate(ctx context.Context, req *model.LLMRequest, stream bool) (*model.LLMResponse, error) {
	apiReq, err := m.buildRequest(req, stream)
	if err != nil {
		return nil, fmt.Errorf("failed to build request: %w", err)
	}

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := m.baseURL + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)

	httpResp, err := m.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to call MiniMax API: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("MiniMax API returned status %d: %s", httpResp.StatusCode, string(respBody))
	}

	var apiResp openAIResponse
	if err := json.Unmarshal(respBody, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if apiResp.Error != nil {
		return nil, fmt.Errorf("MiniMax API error: %s", apiResp.Error.Message)
	}

	return convertResponse(&apiResp), nil
}

// buildRequest converts the model.LLMRequest to an OpenAI-compatible request.
func (m *minimaxModel) buildRequest(req *model.LLMRequest, stream bool) (*openAIRequest, error) {
	messages, err := convertContents(req.Contents, req.Config)
	if err != nil {
		return nil, err
	}

	modelName := m.name
	if req.Model != "" {
		modelName = req.Model
	}

	apiReq := &openAIRequest{
		Model:    modelName,
		Messages: messages,
		Stream:   stream,
	}

	if req.Config != nil {
		if req.Config.Temperature != nil {
			temp := *req.Config.Temperature
			// MiniMax temperature must be in (0.0, 1.0]. Clamp to valid range.
			if temp <= 0 {
				temp = 1.0
			} else if temp > 1.0 {
				temp = 1.0
			}
			apiReq.Temperature = &temp
		}
		if req.Config.MaxOutputTokens > 0 {
			apiReq.MaxTokens = req.Config.MaxOutputTokens
		}
		if req.Config.TopP != nil {
			apiReq.TopP = req.Config.TopP
		}
		if len(req.Config.StopSequences) > 0 {
			apiReq.Stop = req.Config.StopSequences
		}
		if len(req.Config.Tools) > 0 {
			tools, err := convertTools(req.Config.Tools)
			if err != nil {
				return nil, err
			}
			apiReq.Tools = tools
		}
	}

	return apiReq, nil
}

// convertContents converts genai contents to OpenAI messages, injecting the
// system instruction from config as the first message if present.
func convertContents(contents []*genai.Content, cfg *genai.GenerateContentConfig) ([]openAIMessage, error) {
	var messages []openAIMessage

	if cfg != nil && cfg.SystemInstruction != nil {
		text := extractText(cfg.SystemInstruction)
		if text != "" {
			messages = append(messages, openAIMessage{
				Role:    "system",
				Content: text,
			})
		}
	}

	for _, c := range contents {
		if c == nil {
			continue
		}
		msgs, err := convertContent(c)
		if err != nil {
			return nil, err
		}
		messages = append(messages, msgs...)
	}

	return messages, nil
}

// convertContent converts a single genai.Content into one or more OpenAI messages.
// Function responses must be separate tool messages in the OpenAI format.
func convertContent(c *genai.Content) ([]openAIMessage, error) {
	role := convertRole(c.Role)

	// Separate function responses from other parts.
	var toolCalls []openAIToolCall
	var textParts []string
	var toolMessages []openAIMessage

	for _, part := range c.Parts {
		if part == nil {
			continue
		}
		switch {
		case part.FunctionCall != nil:
			args, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function call args: %w", err)
			}
			id := part.FunctionCall.ID
			if id == "" {
				id = "call_" + part.FunctionCall.Name
			}
			toolCalls = append(toolCalls, openAIToolCall{
				ID:   id,
				Type: "function",
				Function: openAIFunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: string(args),
				},
			})
		case part.FunctionResponse != nil:
			respJSON, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}
			id := part.FunctionResponse.ID
			if id == "" {
				id = "call_" + part.FunctionResponse.Name
			}
			toolMessages = append(toolMessages, openAIMessage{
				Role:       "tool",
				Content:    string(respJSON),
				ToolCallID: id,
				Name:       part.FunctionResponse.Name,
			})
		case part.Text != "":
			textParts = append(textParts, part.Text)
		}
	}

	// If the content has function responses (tool results), return those as
	// separate tool messages.
	if len(toolMessages) > 0 {
		return toolMessages, nil
	}

	msg := openAIMessage{Role: role}
	if len(textParts) > 0 {
		msg.Content = strings.Join(textParts, "")
	} else {
		msg.Content = ""
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	return []openAIMessage{msg}, nil
}

// convertRole maps genai roles to OpenAI roles.
func convertRole(role string) string {
	switch role {
	case "model":
		return "assistant"
	case "user":
		return "user"
	default:
		return role
	}
}

// extractText returns all text parts from a genai.Content joined together.
func extractText(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var parts []string
	for _, p := range c.Parts {
		if p != nil && p.Text != "" {
			parts = append(parts, p.Text)
		}
	}
	return strings.Join(parts, "")
}

// convertTools converts genai tools to OpenAI tools.
func convertTools(tools []*genai.Tool) ([]openAITool, error) {
	var result []openAITool
	for _, t := range tools {
		if t == nil {
			continue
		}
		for _, fd := range t.FunctionDeclarations {
			if fd == nil {
				continue
			}
			oaiTool := openAITool{
				Type: "function",
				Function: openAIFunctionDef{
					Name:        fd.Name,
					Description: fd.Description,
					Parameters:  convertSchema(fd.Parameters),
				},
			}
			result = append(result, oaiTool)
		}
	}
	return result, nil
}

// openAISchema is a JSON Schema compatible with OpenAI's function parameter format.
type openAISchema struct {
	Type                 string                   `json:"type,omitempty"`
	Description          string                   `json:"description,omitempty"`
	Enum                 []string                 `json:"enum,omitempty"`
	Properties           map[string]*openAISchema `json:"properties,omitempty"`
	Required             []string                 `json:"required,omitempty"`
	Items                *openAISchema            `json:"items,omitempty"`
	AnyOf                []*openAISchema          `json:"anyOf,omitempty"`
	Format               string                   `json:"format,omitempty"`
	Minimum              *float64                 `json:"minimum,omitempty"`
	Maximum              *float64                 `json:"maximum,omitempty"`
	AdditionalProperties any                      `json:"additionalProperties,omitempty"`
}

// convertSchema converts a genai.Schema to OpenAI schema format.
func convertSchema(s *genai.Schema) *openAISchema {
	if s == nil {
		return nil
	}

	out := &openAISchema{
		Type:        strings.ToLower(string(s.Type)),
		Description: s.Description,
		Enum:        s.Enum,
		Required:    s.Required,
		Format:      s.Format,
		Minimum:     s.Minimum,
		Maximum:     s.Maximum,
	}

	if s.Items != nil {
		out.Items = convertSchema(s.Items)
	}

	if len(s.Properties) > 0 {
		out.Properties = make(map[string]*openAISchema, len(s.Properties))
		for k, v := range s.Properties {
			out.Properties[k] = convertSchema(v)
		}
	}

	if len(s.AnyOf) > 0 {
		out.AnyOf = make([]*openAISchema, len(s.AnyOf))
		for i, a := range s.AnyOf {
			out.AnyOf[i] = convertSchema(a)
		}
	}

	return out
}

// convertResponse converts an OpenAI response to a model.LLMResponse.
func convertResponse(resp *openAIResponse) *model.LLMResponse {
	if len(resp.Choices) == 0 {
		return &model.LLMResponse{
			ErrorCode:    "UNKNOWN_ERROR",
			ErrorMessage: "MiniMax API returned no choices",
		}
	}

	choice := resp.Choices[0]
	msg := choice.Message

	var parts []*genai.Part

	// Handle text content.
	if text, ok := msg.Content.(string); ok && text != "" {
		parts = append(parts, &genai.Part{Text: text})
	}

	// Handle tool calls.
	for _, tc := range msg.ToolCalls {
		var args map[string]any
		if tc.Function.Arguments != "" {
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		}
		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.ID,
				Name: tc.Function.Name,
				Args: args,
			},
		})
	}

	finishReason := genai.FinishReason(strings.ToUpper(choice.FinishReason))
	if finishReason == "" {
		finishReason = genai.FinishReasonStop
	}

	if len(parts) == 0 && finishReason != genai.FinishReasonStop {
		return &model.LLMResponse{
			ErrorCode:    string(finishReason),
			ErrorMessage: "model stopped without content",
			FinishReason: finishReason,
		}
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Parts: parts,
			Role:  "model",
		},
		FinishReason: finishReason,
	}
}
