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

// Package loadmemorytool provides a tool that loads memory for the current user.
// This tool allows the model to search and retrieve relevant memory entries
// based on a query.
package loadmemorytool

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

const memoryInstructions = `You have memory. You can use it to answer questions. If any questions need
you to look up the memory, you should call load_memory function with a query.`

type loadMemoryTool struct {
	name        string
	description string
}

// New creates a new loadMemoryTool.
func New() toolinternal.FunctionTool {
	return &loadMemoryTool{
		name:        "load_memory",
		description: "Loads the memory for the current user.",
	}
}

// Name implements tool.Tool.
func (t *loadMemoryTool) Name() string {
	return t.name
}

// Description implements tool.Tool.
func (t *loadMemoryTool) Description() string {
	return t.description
}

// IsLongRunning implements tool.Tool.
func (t *loadMemoryTool) IsLongRunning() bool {
	return false
}

// Declaration returns the GenAI FunctionDeclaration for the load_memory tool.
func (t *loadMemoryTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        t.name,
		Description: t.description,
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"query": {
					Type:        "STRING",
					Description: "The query to search memory for.",
				},
			},
			Required: []string{"query"},
		},
	}
}

// Run executes the tool with the provided context and arguments.
func (t *loadMemoryTool) Run(toolCtx tool.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type, got: %T", args)
	}

	queryRaw, exists := m["query"]
	if !exists {
		return nil, fmt.Errorf("missing required parameter: query")
	}

	query, ok := queryRaw.(string)
	if !ok {
		return nil, fmt.Errorf("query must be a string, got: %T", queryRaw)
	}

	searchResponse, err := toolCtx.SearchMemory(toolCtx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to search memory: %w", err)
	}

	if searchResponse == nil || searchResponse.Memories == nil {
		return map[string]any{"memories": []memory.Entry{}}, nil
	}
	return map[string]any{"memories": searchResponse.Memories}, nil
}

// ProcessRequest processes the LLM request by packing the tool and appending
// memory-related instructions.
func (t *loadMemoryTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	if err := toolutils.PackTool(req, t); err != nil {
		return err
	}
	utils.AppendInstructions(req, memoryInstructions)
	return nil
}
