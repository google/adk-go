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

package openai

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// convertToOpenAIMessages converts genai.Content to OpenAI message format.
// This function manages conversation history by:
// 1. Retrieving existing history for the session
// 2. Converting new genai.Content messages to OpenAI format
// 3. Appending them to history
// 4. Returning the complete message list for the API call
func (m *openaiModel) convertToOpenAIMessages(ctx context.Context, req *model.LLMRequest) ([]OpenAIMessage, error) {
	// Extract session ID from context with logging
	sessionID := extractSessionIDWithLogging(ctx, m.logger)

	// Initialize result slice
	var allMessages []OpenAIMessage

	// Add SystemInstruction if present (must be first message with role "system")
	var systemText string
	if req.Config != nil && req.Config.SystemInstruction != nil {
		for _, part := range req.Config.SystemInstruction.Parts {
			if part.Text != "" {
				if systemText != "" {
					systemText += "\n"
				}
				systemText += part.Text
			}
		}
	}

	// JSON Mode Safety: OpenAI requires "JSON" keyword in prompt when using json_object mode
	jsonModeEnabled := req.Config != nil && req.Config.ResponseMIMEType == "application/json"
	if jsonModeEnabled {
		// Check if "JSON" keyword exists in system instruction
		hasJSONKeyword := strings.Contains(strings.ToUpper(systemText), "JSON")

		if !hasJSONKeyword {
			// Add JSON instruction to system prompt
			jsonInstruction := "You must respond with valid JSON."
			if systemText != "" {
				systemText = systemText + "\n\n" + jsonInstruction
			} else {
				systemText = jsonInstruction
			}

			if m.logger != nil {
				m.logger.Printf("INFO: JSON mode enabled - added JSON instruction to system prompt")
			}
		}
	}

	// Add system message if we have any system text
	if systemText != "" {
		allMessages = append(allMessages, OpenAIMessage{
			Role:    "system",
			Content: systemText,
		})
	}

	// Get existing history
	history := m.getConversationHistory(sessionID)
	if history == nil {
		history = make([]*OpenAIMessage, 0)
	}

	// Convert new contents
	newMessages := make([]*OpenAIMessage, 0, len(req.Contents))
	for _, content := range req.Contents {
		msgs, err := m.convertContent(content)
		if err != nil {
			return nil, fmt.Errorf("failed to convert content: %w", err)
		}
		newMessages = append(newMessages, msgs...)
	}

	// Add new messages to history
	m.addToHistory(sessionID, newMessages...)

	// Combine: System + History + New messages
	for _, msg := range history {
		allMessages = append(allMessages, *msg)
	}
	for _, msg := range newMessages {
		allMessages = append(allMessages, *msg)
	}

	return allMessages, nil
}

// convertContent converts a single genai.Content to one or more OpenAI messages.
func (m *openaiModel) convertContent(content *genai.Content) ([]*OpenAIMessage, error) {
	if content == nil {
		return nil, nil
	}

	// Determine role
	role := content.Role
	if role == "" {
		role = "user"
	}
	if role == "model" {
		role = "assistant"
	}

	messages := make([]*OpenAIMessage, 0)

	// Handle different part types
	var contentParts []any // Use interface slice for multimodal content (text + images)
	var toolCalls []ToolCall
	var functionResponses []*OpenAIMessage

	for _, part := range content.Parts {
		switch {
		case part.Text != "":
			// Add as text content part
			contentParts = append(contentParts, ContentPartText{
				Type: "text",
				Text: part.Text,
			})

		case part.FunctionCall != nil:
			// Convert function call to tool call
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function args: %w", err)
			}

			// Sanitize arguments
			sanitized, err := sanitizeJSONArgs(string(argsJSON))
			if err != nil {
				// Log error but continue with safe fallback
				sanitized = "{}"
				if m.logger != nil {
					m.logger.Printf("WARNING: Invalid function args sanitized to {}: %v", err)
				}
			}

			// Use ID from FunctionCall - MUST be present for OpenAI API
			toolCallID := part.FunctionCall.ID
			if toolCallID == "" {
				// CRITICAL: OpenAI API requires tool_call_id to match exactly
				// If we generate a fake ID, the response will be rejected with 400 error
				if m.logger != nil {
					m.logger.Printf("WARNING: FunctionCall missing ID for '%s' - generating fallback ID. This may cause API errors!", part.FunctionCall.Name)
				}
				toolCallID = generateToolCallID(part.FunctionCall.Name)
			}

			toolCalls = append(toolCalls, ToolCall{
				ID:   toolCallID,
				Type: "function",
				Function: FunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: sanitized,
				},
			})

		case part.FunctionResponse != nil:
			// Convert function response to tool message
			responseJSON, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}

			// Use ID from FunctionResponse - MUST match the original tool_call_id
			toolCallID := part.FunctionResponse.ID
			if toolCallID == "" {
				// CRITICAL: This is a serious problem!
				// OpenAI will reject the response with 400 if tool_call_id doesn't match
				return nil, fmt.Errorf("FunctionResponse for '%s' missing required ID field - cannot match with tool call", part.FunctionResponse.Name)
			}

			functionResponses = append(functionResponses, &OpenAIMessage{
				Role:       "tool",
				Content:    string(responseJSON),
				ToolCallID: toolCallID,
			})

		case part.ExecutableCode != nil:
			// Represent executable code as text
			codeText := fmt.Sprintf("```%s\n%s\n```", part.ExecutableCode.Language, part.ExecutableCode.Code)
			contentParts = append(contentParts, ContentPartText{
				Type: "text",
				Text: codeText,
			})

		case part.CodeExecutionResult != nil:
			// Represent code execution result as text
			resultText := fmt.Sprintf("Execution result (%s): %s", part.CodeExecutionResult.Outcome, part.CodeExecutionResult.Output)
			contentParts = append(contentParts, ContentPartText{
				Type: "text",
				Text: resultText,
			})

		case part.InlineData != nil:
			// OpenAI supports vision for images via multimodal content
			if part.InlineData.MIMEType != "" && len(part.InlineData.Data) > 0 {
				// Encode image as base64 data URL
				imageURL := fmt.Sprintf("data:%s;base64,%s",
					part.InlineData.MIMEType,
					base64.StdEncoding.EncodeToString(part.InlineData.Data))

				// Add as image content part (NOT as text!)
				contentParts = append(contentParts, ContentPartImage{
					Type: "image_url",
					ImageURL: struct {
						URL string `json:"url"`
					}{URL: imageURL},
				})
			}

		case part.FileData != nil:
			// File URIs (e.g., gs://, https://, file://)
			if part.FileData.FileURI != "" {
				// For HTTP(S) image URLs, add as image content part
				// For other URIs, add as text
				if strings.HasPrefix(part.FileData.FileURI, "http://") ||
					strings.HasPrefix(part.FileData.FileURI, "https://") {
					contentParts = append(contentParts, ContentPartImage{
						Type: "image_url",
						ImageURL: struct {
							URL string `json:"url"`
						}{URL: part.FileData.FileURI},
					})
				} else {
					contentParts = append(contentParts, ContentPartText{
						Type: "text",
						Text: part.FileData.FileURI,
					})
				}
			}
		}
	}

	// Create message(s) based on what we found
	if len(toolCalls) > 0 {
		// Assistant message with tool calls
		msg := &OpenAIMessage{
			Role:      "assistant",
			ToolCalls: toolCalls,
		}
		if len(contentParts) > 0 {
			// Use array format for multimodal or single text part
			msg.Content = convertContentToMessage(contentParts)
		}
		messages = append(messages, msg)
	} else if len(contentParts) > 0 {
		// Regular message (text or multimodal)
		messages = append(messages, &OpenAIMessage{
			Role:    role,
			Content: convertContentToMessage(contentParts),
		})
	}

	// Add function response messages
	messages = append(messages, functionResponses...)

	return messages, nil
}

// convertToLLMResponse converts an OpenAI message back to genai format.
func (m *openaiModel) convertToLLMResponse(msg *OpenAIMessage, usage *Usage) (*model.LLMResponse, error) {
	parts := make([]*genai.Part, 0)

	// Handle text content
	if msg.Content != nil {
		if text, ok := msg.Content.(string); ok && text != "" {
			parts = append(parts, genai.NewPartFromText(text))
		}
	}

	// Handle tool calls (function calls in genai format)
	for _, toolCall := range msg.ToolCalls {
		if toolCall.Type == "function" {
			// Handle empty arguments (some models send "" or "{}")
			argsStr := toolCall.Function.Arguments
			if argsStr == "" {
				argsStr = "{}" // Default to empty object
			}

			var args map[string]any
			if err := json.Unmarshal([]byte(argsStr), &args); err != nil {
				return nil, fmt.Errorf("failed to unmarshal tool call args: %w", err)
			}

			// Create FunctionCall part with ID preserved
			part := genai.NewPartFromFunctionCall(toolCall.Function.Name, args)
			if part.FunctionCall != nil {
				part.FunctionCall.ID = toolCall.ID
			}
			parts = append(parts, part)
		}
	}

	content := &genai.Content{
		Role:  "model",
		Parts: parts,
	}

	response := &model.LLMResponse{
		Content:      content,
		TurnComplete: true,
	}

	// Add usage metadata if available
	if usage != nil {
		response.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(usage.PromptTokens),
			CandidatesTokenCount: int32(usage.CompletionTokens),
			TotalTokenCount:      int32(usage.TotalTokens),
		}
	}

	return response, nil
}

// convertTools converts ADK tools to OpenAI tool format.
func (m *openaiModel) convertTools(adkTools map[string]any) []Tool {
	tools := make([]Tool, 0)

	for name, toolDef := range adkTools {
		// Try to extract tool information
		// ADK tools are typically in a specific format
		tool := Tool{
			Type: "function",
			Function: Function{
				Name: name,
			},
		}

		// Try to extract description and parameters
		if toolMap, ok := toolDef.(map[string]any); ok {
			if desc, ok := toolMap["description"].(string); ok {
				tool.Function.Description = desc
			}
			if params, ok := toolMap["parameters"].(map[string]any); ok {
				tool.Function.Parameters = params
			} else if params, ok := toolMap["input_schema"].(map[string]any); ok {
				tool.Function.Parameters = params
			}
		}

		tools = append(tools, tool)
	}

	return tools
}

// Helper functions
// Note: Session ID extraction moved to session.go

func generateToolCallID(functionName string) string {
	// Generate a unique ID to avoid collisions if the same function is called multiple times.
	// Using crypto/rand ensures uniqueness even for parallel calls.
	randomBytes := make([]byte, 4)
	// crypto/rand is safe for concurrent use - ignore error as it's extremely rare
	rand.Read(randomBytes)
	return fmt.Sprintf("call_%s_%s", functionName, hex.EncodeToString(randomBytes))
}

// convertContentToMessage converts content parts to the appropriate format.
// Returns string if single text part (optimization), array otherwise (multimodal).
func convertContentToMessage(parts []any) interface{} {
	if len(parts) == 0 {
		return ""
	}

	// Optimization: if only one text part, return as string
	if len(parts) == 1 {
		if textPart, ok := parts[0].(ContentPartText); ok {
			return textPart.Text
		}
	}

	// Otherwise return array for multimodal content
	return parts
}
