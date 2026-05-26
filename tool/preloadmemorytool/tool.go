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

// Package preloadmemorytool provides a tool that automatically preloads memory
// for the current user at the start of each LLM request.
//
// Unlike loadmemorytool which is called explicitly by the model, this tool
// runs automatically for each LLM request and injects relevant memory context
// into the system instructions.

package preloadmemorytool

import (
	"fmt"
	"strings"
	"time"

	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

const preloadInstructions = `The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
%s
</PAST_CONVERSATIONS>`

// preloadMemoryTool is a tool that preloads the memory for the current user.
// It is automatically executed for each LLM request and will not be called
// directly by the model.
type preloadMemoryTool struct {
	name        string
	description string
}

// New creates a new preloadMemoryTool.
func New() *preloadMemoryTool {
	return &preloadMemoryTool{
		name:        "preload_memory",
		description: "Preloads relevant memory for the current user.",
	}
}

// Name implements tool.Tool.
func (t *preloadMemoryTool) Name() string {
	return t.name
}

// Description implements tool.Tool.
func (t *preloadMemoryTool) Description() string {
	return t.description
}

// IsLongRunning implements tool.Tool.
func (t *preloadMemoryTool) IsLongRunning() bool {
	return false
}

// ProcessRequest processes the LLM request by searching memory using the user's
// current query and injecting relevant past conversations into system instructions.
func (t *preloadMemoryTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	userContent := ctx.UserContent()
	if userContent == nil || len(userContent.Parts) == 0 ||
		userContent.Parts[0] == nil || userContent.Parts[0].Text == "" {
		return nil
	}
	userQuery := userContent.Parts[0].Text

	searchResponse, err := ctx.SearchMemory(ctx, userQuery)
	if err != nil {
		return fmt.Errorf("preload memory search failed: %v", err)
	}

	if searchResponse == nil || len(searchResponse.Memories) == 0 {
		return nil
	}

	memoryText := formatMemories(searchResponse.Memories)
	if memoryText == "" {
		return nil
	}

	utils.AppendInstructions(req, fmt.Sprintf(preloadInstructions, memoryText))
	return nil
}

func formatMemories(memories []memory.Entry) string {
	var lines []string
	for _, mem := range memories {
		memText := extractText(mem)
		if memText == "" {
			continue
		}

		if !mem.Timestamp.IsZero() {
			lines = append(lines, fmt.Sprintf("Time: %s", mem.Timestamp.Format(time.RFC3339)))
		}
		if mem.Author != "" {
			memText = fmt.Sprintf("%s: %s", mem.Author, memText)
		}
		lines = append(lines, memText)
	}
	return strings.Join(lines, "\n")
}

func extractText(mem memory.Entry) string {
	if mem.Content == nil || len(mem.Content.Parts) == 0 {
		return ""
	}

	var b strings.Builder
	for _, part := range mem.Content.Parts {
		if part.Text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(part.Text)
	}
	return b.String()
}
