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
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// StreamChunk represents a Server-Sent Event chunk from OpenAI streaming.
type StreamChunk struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
}

// generateStream implements streaming for OpenAI API.
func (m *openaiModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		// Convert genai.Content to OpenAI messages
		messages, err := m.convertToOpenAIMessages(ctx, req)
		if err != nil {
			yield(nil, fmt.Errorf("failed to convert messages: %w", err))
			return
		}

		// Build OpenAI request
		chatReq := ChatCompletionRequest{
			Model:    m.name,
			Messages: messages,
			Stream:   true,
		}

		// Add configuration from req.Config
		if req.Config != nil {
			if req.Config.Temperature != nil {
				chatReq.Temperature = req.Config.Temperature
			}
			if req.Config.MaxOutputTokens > 0 {
				tokens := req.Config.MaxOutputTokens
				chatReq.MaxTokens = &tokens
			}
		}

		// Add tools if present
		if len(req.Tools) > 0 {
			chatReq.Tools = m.convertTools(req.Tools)
			chatReq.ToolChoice = "auto"
		}

		// Make streaming API call
		if err := m.streamRequest(ctx, chatReq, yield); err != nil {
			yield(nil, err)
			return
		}
	}
}

// streamRequest makes a streaming HTTP request and processes SSE events.
func (m *openaiModel) streamRequest(ctx context.Context, req ChatCompletionRequest, yield func(*model.LLMResponse, error) bool) error {
	buf := m.jsonPool.Get().(*bytes.Buffer)
	defer func() {
		buf.Reset()
		m.jsonPool.Put(buf)
	}()

	if err := json.NewEncoder(buf).Encode(req); err != nil {
		return fmt.Errorf("failed to encode request: %w", err)
	}

	url := m.baseURL + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(buf.Bytes()))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if m.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)
	}

	resp, err := m.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Process SSE stream
	return m.processSSEStream(resp.Body, yield)
}

// processSSEStream reads and processes Server-Sent Events.
func (m *openaiModel) processSSEStream(reader io.Reader, yield func(*model.LLMResponse, error) bool) error {
	scanner := bufio.NewScanner(reader)

	// Aggregator for combining streaming chunks
	var aggregatedText strings.Builder
	var aggregatedToolCalls []ToolCall
	var lastChunk *StreamChunk

	for scanner.Scan() {
		line := scanner.Text()

		// SSE format: "data: {json}" or "data: [DONE]"
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		data = strings.TrimSpace(data)

		// Check for end of stream
		if data == "[DONE]" {
			// Send final aggregated response
			if aggregatedText.Len() > 0 || len(aggregatedToolCalls) > 0 {
				finalResp := m.createFinalResponse(aggregatedText.String(), aggregatedToolCalls)
				if !yield(finalResp, nil) {
					return nil
				}
			}
			return nil
		}

		// Parse chunk
		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			// Skip malformed chunks
			continue
		}

		lastChunk = &chunk

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]

		// Aggregate text content
		if choice.Delta.Content != nil {
			if text, ok := choice.Delta.Content.(string); ok && text != "" {
				aggregatedText.WriteString(text)

				// Yield partial response
				partialResp := &model.LLMResponse{
					Content: &genai.Content{
						Role:  "model",
						Parts: []*genai.Part{genai.NewPartFromText(text)},
					},
					Partial:      true,
					TurnComplete: false,
				}

				if !yield(partialResp, nil) {
					return nil
				}
			}
		}

		// Aggregate tool calls
		if len(choice.Delta.ToolCalls) > 0 {
			for _, toolCall := range choice.Delta.ToolCalls {
				// OpenAI streams tool calls incrementally
				// We need to merge them by index
				if toolCall.Type == "function" {
					aggregatedToolCalls = m.mergeToolCall(aggregatedToolCalls, toolCall)
				}
			}
		}

		// Check for finish
		if choice.FinishReason != "" && choice.FinishReason != "null" {
			// Send final response
			finalResp := m.createFinalResponse(aggregatedText.String(), aggregatedToolCalls)
			finalResp.TurnComplete = true

			if lastChunk != nil {
				finalResp.FinishReason = mapFinishReason(choice.FinishReason)
			}

			if !yield(finalResp, nil) {
				return nil
			}
			return nil
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading stream: %w", err)
	}

	return nil
}

// mergeToolCall merges incremental tool call updates.
func (m *openaiModel) mergeToolCall(existing []ToolCall, delta ToolCall) []ToolCall {
	// OpenAI uses index to identify which tool call to update
	// For simplicity, we'll append new ones
	// TODO: Implement proper merging by index

	// Check if this is an update to an existing tool call
	for i := range existing {
		if existing[i].ID == delta.ID || (delta.ID == "" && i == len(existing)-1) {
			// Merge the arguments
			existing[i].Function.Arguments += delta.Function.Arguments
			if delta.Function.Name != "" {
				existing[i].Function.Name = delta.Function.Name
			}
			if delta.ID != "" {
				existing[i].ID = delta.ID
			}
			return existing
		}
	}

	// New tool call
	return append(existing, delta)
}

// createFinalResponse creates a final LLMResponse from aggregated data.
func (m *openaiModel) createFinalResponse(text string, toolCalls []ToolCall) *model.LLMResponse {
	parts := make([]*genai.Part, 0)

	if text != "" {
		parts = append(parts, genai.NewPartFromText(text))
	}

	// Convert tool calls to function calls
	for _, toolCall := range toolCalls {
		if toolCall.Type == "function" {
			var args map[string]any
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
				// Log error but continue
				continue
			}
			parts = append(parts, genai.NewPartFromFunctionCall(toolCall.Function.Name, args))
		}
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: parts,
		},
		TurnComplete: true,
	}
}

// mapFinishReason maps OpenAI finish reasons to genai.FinishReason.
func mapFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "tool_calls":
		return genai.FinishReasonStop
	case "content_filter":
		return genai.FinishReasonSafety
	default:
		return genai.FinishReasonOther
	}
}
