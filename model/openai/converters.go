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
	"encoding/json"
	"fmt"

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

	// Return complete history as slice of values (not pointers)
	result := make([]OpenAIMessage, len(history)+len(newMessages))
	for i, msg := range history {
		result[i] = *msg
	}
	for i, msg := range newMessages {
		result[len(history)+i] = *msg
	}

	return result, nil
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
	var textParts []string
	var toolCalls []ToolCall
	var functionResponses []*OpenAIMessage

	for _, part := range content.Parts {
		switch {
		case part.Text != "":
			textParts = append(textParts, part.Text)

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

			// Use ID from FunctionCall if available, otherwise generate one
			toolCallID := part.FunctionCall.ID
			if toolCallID == "" {
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

			// Use ID from FunctionResponse if available, otherwise generate one
			toolCallID := part.FunctionResponse.ID
			if toolCallID == "" {
				toolCallID = generateToolCallID(part.FunctionResponse.Name)
			}

			functionResponses = append(functionResponses, &OpenAIMessage{
				Role:       "tool",
				Content:    string(responseJSON),
				ToolCallID: toolCallID,
			})

		case part.ExecutableCode != nil:
			// Represent executable code as text
			codeText := fmt.Sprintf("```%s\n%s\n```", part.ExecutableCode.Language, part.ExecutableCode.Code)
			textParts = append(textParts, codeText)

		case part.CodeExecutionResult != nil:
			// Represent code execution result as text
			resultText := fmt.Sprintf("Execution result (%s): %s", part.CodeExecutionResult.Outcome, part.CodeExecutionResult.Output)
			textParts = append(textParts, resultText)

		case part.InlineData != nil:
			// OpenAI supports vision for images
			if part.InlineData.MIMEType != "" && len(part.InlineData.Data) > 0 {
				// For simplicity, we'll skip inline data for now
				// TODO: Add support for vision API with base64 encoded images
				continue
			}

		case part.FileData != nil:
			// Skip file references for now
			// TODO: Add support for file URIs if needed
			continue
		}
	}

	// Create message(s) based on what we found
	if len(toolCalls) > 0 {
		// Assistant message with tool calls
		msg := &OpenAIMessage{
			Role:      "assistant",
			ToolCalls: toolCalls,
		}
		if len(textParts) > 0 {
			msg.Content = joinTextParts(textParts)
		}
		messages = append(messages, msg)
	} else if len(textParts) > 0 {
		// Regular text message
		messages = append(messages, &OpenAIMessage{
			Role:    role,
			Content: joinTextParts(textParts),
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
			var args map[string]any
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
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
	// Generate a deterministic ID based on function name
	// In production, this should be unique per call
	return fmt.Sprintf("call_%s", functionName)
}

func joinTextParts(parts []string) string {
	if len(parts) == 0 {
		return ""
	}
	if len(parts) == 1 {
		return parts[0]
	}

	result := ""
	for i, part := range parts {
		if i > 0 {
			result += "\n"
		}
		result += part
	}
	return result
}
