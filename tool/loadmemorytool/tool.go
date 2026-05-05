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
	"encoding/json"
	"fmt"
	"strings"
	"time"

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
		return map[string]any{"memories": []any{}}, nil
	}
	return map[string]any{"memories": formatMemoryEntries(searchResponse.Memories)}, nil
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

// formatMemoryEntries converts ADK memory entries into the plain data shape
// returned by the load_memory tool.
//
// Tool responses are written back into the session as function responses. Some
// session backends, including Vertex AI sessions, persist those responses as a
// protobuf Struct. A protobuf Struct can only contain JSON-like values such as
// strings, numbers, booleans, maps, and slices. Returning memory.Entry directly
// would leak Go structs like genai.Content into the response and fail when the
// session service tries to store the event.
func formatMemoryEntries(memories []memory.Entry) []any {
	formatted := make([]any, 0, len(memories))
	for _, mem := range memories {
		entry := map[string]any{}
		if mem.ID != "" {
			entry["id"] = mem.ID
		}
		if mem.Author != "" {
			entry["author"] = mem.Author
		}
		if !mem.Timestamp.IsZero() {
			entry["timestamp"] = mem.Timestamp.Format(time.RFC3339)
		}
		if content := extractText(mem); content != "" {
			entry["content"] = content
		}
		if metadata := jsonCompatibleMap(mem.CustomMetadata); metadata != nil {
			entry["custom_metadata"] = metadata
		}
		formatted = append(formatted, entry)
	}
	return formatted
}

// extractText joins the text parts from a memory entry into one string that can
// be safely returned to the model.
//
// memory.Entry.Content can contain richer GenAI parts, but the memory-loading
// tool only needs readable text. Non-text parts are skipped because they cannot
// be represented cleanly in the simple function response shape.
func extractText(mem memory.Entry) string {
	if mem.Content == nil || len(mem.Content.Parts) == 0 {
		return ""
	}

	var b strings.Builder
	for _, part := range mem.Content.Parts {
		if part == nil || part.Text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(part.Text)
	}
	return b.String()
}

// jsonCompatibleMap returns metadata only when it can be represented as plain
// JSON-like values.
//
// Custom metadata is user supplied and may contain Go values that protobuf
// Struct cannot encode. The JSON round trip keeps compatible metadata and drops
// incompatible metadata instead of failing the whole memory load.
func jsonCompatibleMap(value map[string]any) map[string]any {
	if len(value) == 0 {
		return nil
	}

	b, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		return nil
	}
	return out
}
