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

// Package converters provides conversion functions between genai types and Anthropic SDK types.
package converters

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

// ContentsToMessages converts genai Contents to Anthropic MessageParams.
// It handles role mapping and content part conversion.
func ContentsToMessages(contents []*genai.Content) ([]anthropic.MessageParam, error) {
	if len(contents) == 0 {
		return nil, nil
	}

	var messages []anthropic.MessageParam
	for _, content := range contents {
		if content == nil {
			continue
		}

		msg, err := contentToMessage(content)
		if err != nil {
			return nil, fmt.Errorf("failed to convert content: %w", err)
		}
		if msg != nil {
			messages = append(messages, *msg)
		}
	}

	// Merge consecutive messages with the same role (Anthropic requires alternating roles)
	messages = mergeConsecutiveMessages(messages)

	return messages, nil
}

// contentToMessage converts a single genai.Content to an Anthropic MessageParam.
func contentToMessage(content *genai.Content) (*anthropic.MessageParam, error) {
	if content == nil || len(content.Parts) == 0 {
		return nil, nil
	}

	// Check if this content contains tool results (FunctionResponse).
	// Anthropic requires tool results to be in user messages.
	hasFunctionResponse := false
	hasFunctionCall := false
	for _, part := range content.Parts {
		if part != nil {
			if part.FunctionResponse != nil {
				hasFunctionResponse = true
			}
			if part.FunctionCall != nil {
				hasFunctionCall = true
			}
		}
	}

	// Determine the role - tool results must be user, tool calls must be assistant
	var role anthropic.MessageParamRole
	if hasFunctionResponse {
		// Tool results MUST be in user messages per Anthropic API requirements
		role = anthropic.MessageParamRoleUser
	} else if hasFunctionCall {
		// Tool calls (from model) MUST be in assistant messages
		role = anthropic.MessageParamRoleAssistant
	} else {
		var err error
		role, err = mapRole(content.Role)
		if err != nil {
			return nil, err
		}
	}

	var blocks []anthropic.ContentBlockParamUnion
	for _, part := range content.Parts {
		if part == nil {
			continue
		}
		block, err := PartToContentBlock(part)
		if err != nil {
			return nil, fmt.Errorf("failed to convert part: %w", err)
		}
		if block != nil {
			blocks = append(blocks, *block)
		}
	}

	if len(blocks) == 0 {
		return nil, nil
	}

	msg := anthropic.MessageParam{
		Role:    role,
		Content: blocks,
	}
	return &msg, nil
}

// mapRole maps genai role to Anthropic MessageParamRole.
func mapRole(role string) (anthropic.MessageParamRole, error) {
	switch strings.ToLower(role) {
	case "user":
		return anthropic.MessageParamRoleUser, nil
	case "model", "assistant":
		return anthropic.MessageParamRoleAssistant, nil
	default:
		return "", fmt.Errorf("unsupported role: %s", role)
	}
}

// PartToContentBlock converts a genai Part to an Anthropic ContentBlockParamUnion.
func PartToContentBlock(part *genai.Part) (*anthropic.ContentBlockParamUnion, error) {
	if part == nil {
		return nil, nil
	}

	// Text content
	if part.Text != "" {
		// Check if this is a thought block
		if part.Thought {
			// Thoughts from model responses need to be passed back with signature
			if len(part.ThoughtSignature) > 0 {
				block := anthropic.ContentBlockParamUnion{
					OfThinking: &anthropic.ThinkingBlockParam{
						Thinking:  part.Text,
						Signature: base64.StdEncoding.EncodeToString(part.ThoughtSignature),
					},
				}
				return &block, nil
			}
			// If no signature, treat as regular text (shouldn't happen in valid flow)
		}
		block := anthropic.NewTextBlock(part.Text)
		return &block, nil
	}

	// Inline binary data (images, PDFs)
	if part.InlineData != nil {
		return inlineDataToBlock(part.InlineData)
	}

	// File data (URI-based)
	if part.FileData != nil {
		return fileDataToBlock(part.FileData)
	}

	// Function response (tool result)
	if part.FunctionResponse != nil {
		return functionResponseToBlock(part.FunctionResponse)
	}

	// Function call - these should only appear in model responses, not requests
	// We return nil for these as they shouldn't be in user messages
	if part.FunctionCall != nil {
		return functionCallToBlock(part.FunctionCall)
	}

	// Executable code and CodeExecutionResult are Gemini-specific features
	// that don't have direct Anthropic equivalents
	if part.ExecutableCode != nil || part.CodeExecutionResult != nil {
		return nil, fmt.Errorf("ExecutableCode and CodeExecutionResult are not supported by Anthropic")
	}

	return nil, nil
}

// inlineDataToBlock converts inline binary data to an Anthropic content block.
func inlineDataToBlock(blob *genai.Blob) (*anthropic.ContentBlockParamUnion, error) {
	if blob == nil {
		return nil, nil
	}

	mimeType := strings.ToLower(blob.MIMEType)

	// Handle images
	if strings.HasPrefix(mimeType, "image/") {
		mediaType, err := mapImageMediaType(mimeType)
		if err != nil {
			return nil, err
		}
		block := anthropic.ContentBlockParamUnion{
			OfImage: &anthropic.ImageBlockParam{
				Source: anthropic.ImageBlockParamSourceUnion{
					OfBase64: &anthropic.Base64ImageSourceParam{
						Data:      base64.StdEncoding.EncodeToString(blob.Data),
						MediaType: mediaType,
					},
				},
			},
		}
		return &block, nil
	}

	// Handle PDFs (beta feature)
	if mimeType == "application/pdf" {
		block := anthropic.ContentBlockParamUnion{
			OfDocument: &anthropic.DocumentBlockParam{
				Source: anthropic.DocumentBlockParamSourceUnion{
					OfBase64: &anthropic.Base64PDFSourceParam{
						Data: base64.StdEncoding.EncodeToString(blob.Data),
					},
				},
			},
		}
		return &block, nil
	}

	return nil, fmt.Errorf("unsupported MIME type for inline data: %s", mimeType)
}

// mapImageMediaType maps MIME types to Anthropic Base64ImageSourceMediaType.
func mapImageMediaType(mimeType string) (anthropic.Base64ImageSourceMediaType, error) {
	switch mimeType {
	case "image/jpeg":
		return anthropic.Base64ImageSourceMediaTypeImageJPEG, nil
	case "image/png":
		return anthropic.Base64ImageSourceMediaTypeImagePNG, nil
	case "image/gif":
		return anthropic.Base64ImageSourceMediaTypeImageGIF, nil
	case "image/webp":
		return anthropic.Base64ImageSourceMediaTypeImageWebP, nil
	default:
		return "", fmt.Errorf("unsupported image media type: %s", mimeType)
	}
}

// fileDataToBlock converts URI-based file data to an Anthropic content block.
func fileDataToBlock(fileData *genai.FileData) (*anthropic.ContentBlockParamUnion, error) {
	if fileData == nil {
		return nil, nil
	}

	mimeType := strings.ToLower(fileData.MIMEType)

	// Handle images via URL
	if strings.HasPrefix(mimeType, "image/") {
		block := anthropic.ContentBlockParamUnion{
			OfImage: &anthropic.ImageBlockParam{
				Source: anthropic.ImageBlockParamSourceUnion{
					OfURL: &anthropic.URLImageSourceParam{
						URL: fileData.FileURI,
					},
				},
			},
		}
		return &block, nil
	}

	// Handle PDFs via URL (beta feature)
	if mimeType == "application/pdf" {
		block := anthropic.ContentBlockParamUnion{
			OfDocument: &anthropic.DocumentBlockParam{
				Source: anthropic.DocumentBlockParamSourceUnion{
					OfURL: &anthropic.URLPDFSourceParam{
						URL: fileData.FileURI,
					},
				},
			},
		}
		return &block, nil
	}

	return nil, fmt.Errorf("unsupported MIME type for file data: %s", mimeType)
}

// functionResponseToBlock converts a FunctionResponse to an Anthropic tool result block.
func functionResponseToBlock(resp *genai.FunctionResponse) (*anthropic.ContentBlockParamUnion, error) {
	if resp == nil {
		return nil, nil
	}

	// The function ID is required for proper tool call correlation.
	// Without it, Anthropic cannot match tool results to their originating tool calls.
	if resp.ID == "" {
		return nil, fmt.Errorf("FunctionResponse.ID is required for tool call correlation (function: %s)", resp.Name)
	}

	// Convert the response to JSON string
	var content string
	if resp.Response != nil {
		jsonBytes, err := json.Marshal(resp.Response)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal function response: %w", err)
		}
		content = string(jsonBytes)
	}

	block := anthropic.NewToolResultBlock(resp.ID, content, false)
	return &block, nil
}

// functionCallToBlock converts a FunctionCall to an Anthropic tool use block.
// This is used when passing model responses back (e.g., in conversation history).
func functionCallToBlock(call *genai.FunctionCall) (*anthropic.ContentBlockParamUnion, error) {
	if call == nil {
		return nil, nil
	}

	// Convert args to any type for the input
	var input any = call.Args
	if input == nil {
		input = map[string]any{}
	}

	block := anthropic.NewToolUseBlock(call.ID, input, call.Name)
	return &block, nil
}

// SystemInstructionToSystem converts a genai SystemInstruction to Anthropic system text blocks.
func SystemInstructionToSystem(instruction *genai.Content) []anthropic.TextBlockParam {
	if instruction == nil || len(instruction.Parts) == 0 {
		return nil
	}

	var blocks []anthropic.TextBlockParam
	for _, part := range instruction.Parts {
		if part != nil && part.Text != "" {
			blocks = append(blocks, anthropic.TextBlockParam{
				Text: part.Text,
			})
		}
	}
	return blocks
}

// mergeConsecutiveMessages merges consecutive messages with the same role.
// Anthropic requires strictly alternating user/assistant messages.
func mergeConsecutiveMessages(messages []anthropic.MessageParam) []anthropic.MessageParam {
	if len(messages) <= 1 {
		return messages
	}

	var merged []anthropic.MessageParam
	for i, msg := range messages {
		if i == 0 {
			merged = append(merged, msg)
			continue
		}

		last := &merged[len(merged)-1]
		if last.Role == msg.Role {
			// Merge content blocks
			last.Content = append(last.Content, msg.Content...)
		} else {
			merged = append(merged, msg)
		}
	}
	return merged
}
