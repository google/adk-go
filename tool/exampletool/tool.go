// Copyright 2026 Google LLC
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

// Package exampletool provides a tool that allows an agent to add (few-shot) examples to the LLM request.
package exampletool

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
)

// Example is a single few-shot example, pairing an input with its expected output.
type Example struct {
	Input  *genai.Content   `json:"input"`
	Output []*genai.Content `json:"output"`
}

// Provider fetches few-shot examples for a user query.
//
// Examples takes a context.Context and returns an error because providers may
// perform I/O.
type Provider interface {
	Examples(ctx context.Context, query string) ([]*Example, error)
}

// ExampleToolConfig configures an example tool with the few-shot examples to add.
// Set either Examples or Provider, not both.
type ExampleToolConfig struct {
	Examples []*Example
	Provider Provider
}

// exampleTool is a tool that adds (few-shot) examples to the LLM request.
type exampleTool struct {
	examples []*Example
	provider Provider
}

// New creates an example tool that adds the configured few-shot examples to the
// LLM request.
func New(config ExampleToolConfig) (*exampleTool, error) {
	if config.Examples != nil && config.Provider != nil {
		return nil, errors.New("exampletool: cannot specify both Examples and Provider")
	}
	return &exampleTool{
		examples: config.Examples,
		provider: config.Provider,
	}, nil
}

// Name implements tool.Tool.
func (s exampleTool) Name() string {
	return "example_tool"
}

// Description implements tool.Tool.
func (s exampleTool) Description() string {
	return "example tool"
}

// ProcessRequest adds the exampleTool examples to the LLM request.
func (s exampleTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	userContent := ctx.UserContent()
	if userContent == nil {
		return nil
	}
	parts := userContent.Parts
	if len(parts) == 0 {
		return nil
	}

	var examples []*Example
	if s.provider != nil {
		query := userQuery(userContent)
		if query == "" {
			return nil
		}
		var err error
		// Unlike adk-python's get_examples, the provider takes a context and
		// returns an error because it may perform I/O.
		examples, err = s.provider.Examples(ctx, query)
		if err != nil {
			return fmt.Errorf("exampletool: fetch examples: %w", err)
		}
	} else {
		// Unlike adk-python, the static list path checks only the first part for
		// non-empty text to preserve existing behavior.
		if parts[0] == nil || parts[0].Text == "" {
			return nil
		}
		examples = s.examples
	}

	instruction := buildExamplesSystemInstruction(examples, req.Model)
	utils.AppendInstructions(req, instruction)
	return nil
}

// userQuery joins the non-empty, non-thought text parts of c with spaces.
func userQuery(c *genai.Content) string {
	var texts []string
	for _, p := range c.Parts {
		if p != nil && !p.Thought && p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	return strings.Join(texts, " ")
}

// IsLongRunning implements tool.Tool.
func (t exampleTool) IsLongRunning() bool {
	return false
}

const (
	examplesIntro          = "<EXAMPLES>\nBegin few-shot\nThe following are examples of user queries and model responses using the available tools.\n\n"
	examplesEnd            = "End few-shot\n<EXAMPLES>"
	exampleStart           = "EXAMPLE %d:\nBegin example\n"
	exampleEnd             = "End example\n\n"
	userPrefix             = "[user]\n"
	modelPrefix            = "[model]\n"
	functionPrefix         = "```\n"
	functionCallPrefix     = "```tool_code\n"
	functionCallSuffix     = "\n```\n"
	functionResponsePrefix = "```tool_outputs\n"
	functionResponseSuffix = "\n```\n"
)

// Converts a list of examples to a string that can be used in a system instruction.
func buildExamplesSystemInstruction(examples []*Example, model string) string {
	var sb strings.Builder
	sb.WriteString(examplesIntro)
	// As in adk-python's convert_examples_to_text, Gemini models, and a request
	// that names no model, take plain fences; the tool_code and tool_outputs
	// fences are for other models.
	isGemini := model == "" || strings.Contains(model, "gemini-")
	for exampleNum, example := range examples {
		fmt.Fprintf(&sb, exampleStart, exampleNum+1)
		sb.WriteString(userPrefix)
		if example.Input != nil && len(example.Input.Parts) > 0 {
			for _, part := range example.Input.Parts {
				if part.Text != "" {
					safeText := strings.ReplaceAll(part.Text, "End few-shot", "[PROTECTED]")
					sb.WriteString(safeText)
					sb.WriteString("\n")
				}
			}
		}
		previousRole := ""
		for _, content := range example.Output {
			var role string
			if content.Role == "model" {
				role = modelPrefix
			} else {
				role = userPrefix
			}
			if role != previousRole {
				sb.WriteString(role)
			}
			previousRole = role
			for _, part := range content.Parts {
				if part.FunctionCall != nil {
					args := []string{}
					// Sorted, because map order would make the instruction differ
					// from one request to the next.
					for _, k := range slices.Sorted(maps.Keys(part.FunctionCall.Args)) {
						v := part.FunctionCall.Args[k]
						if _, ok := v.(string); ok {
							args = append(args, fmt.Sprintf("%s='%s'", k, v))
						} else {
							args = append(args, fmt.Sprintf("%s=%v", k, v))
						}
					}
					prefix := functionCallPrefix
					if isGemini {
						prefix = functionPrefix
					}
					fmt.Fprintf(&sb, "%s%s(%s)%s", prefix, part.FunctionCall.Name, strings.Join(args, ", "), functionCallSuffix)
				} else if part.FunctionResponse != nil {
					prefix := functionResponsePrefix
					if isGemini {
						prefix = functionPrefix
					}
					fmt.Fprintf(&sb, "%s%v%s", prefix, part.FunctionResponse, functionResponseSuffix)
				} else if part.Text != "" {
					// SANITIZATION: Again, protect the boundary tags
					safeText := strings.ReplaceAll(part.Text, "End few-shot", "[PROTECTED]")
					sb.WriteString(safeText)
					sb.WriteString("\n")
				}
			}
		}
		sb.WriteString(exampleEnd)
	}
	sb.WriteString(examplesEnd)
	return sb.String()
}
