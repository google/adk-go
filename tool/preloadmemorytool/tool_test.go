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

package preloadmemorytool_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/preloadmemorytool"
)

type mockMemory struct {
	memories []memory.Entry
	err      error
}

func (m *mockMemory) AddSession(ctx context.Context, s session.Session) error {
	return nil
}

func (m *mockMemory) Search(ctx context.Context, query string) (*memory.SearchResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &memory.SearchResponse{Memories: m.memories}, nil
}

func TestPreloadMemoryTool_BasicProperties(t *testing.T) {
	tool := preloadmemorytool.New()

	if got := tool.Name(); got != "preload_memory" {
		t.Errorf("Name() = %v, want preload_memory", got)
	}
	if got := tool.Description(); got != "Preloads relevant memory for the current user." {
		t.Errorf("Description() = %v, want 'Preloads relevant memory for the current user.'", got)
	}
	if got := tool.IsLongRunning(); got != false {
		t.Errorf("IsLongRunning() = %v, want false", got)
	}
}

func TestPreloadMemoryTool_ProcessRequest(t *testing.T) {
	tests := []struct {
		name             string
		userContent      *genai.Content
		memories         []memory.Entry
		searchErr        error
		wantErr          bool
		wantInstruction  bool
		wantTextContains []string
	}{
		{
			name:            "nil user content",
			userContent:     nil,
			wantInstruction: false,
		},
		{
			name:            "empty user content parts",
			userContent:     &genai.Content{Parts: []*genai.Part{}},
			wantInstruction: false,
		},
		{
			name:            "user content with no text",
			userContent:     &genai.Content{Parts: []*genai.Part{{InlineData: &genai.Blob{}}}},
			wantInstruction: false,
		},
		{
			name:            "text in later part is ignored",
			userContent:     &genai.Content{Parts: []*genai.Part{{InlineData: &genai.Blob{}}, genai.NewPartFromText("later text")}},
			wantInstruction: false,
		},
		{
			name:            "no memories found",
			userContent:     genai.NewContentFromText("test query", genai.RoleUser),
			memories:        []memory.Entry{},
			wantInstruction: false,
		},
		{
			name:            "memory search error",
			userContent:     genai.NewContentFromText("test query", genai.RoleUser),
			searchErr:       errors.New("search failed"),
			wantErr:         true,
			wantInstruction: false,
		},
		{
			name:        "single memory entry",
			userContent: genai.NewContentFromText("test query", genai.RoleUser),
			memories: []memory.Entry{
				{
					Author:    "user",
					Timestamp: time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC),
					Content:   genai.NewContentFromText("Hello world", genai.RoleUser),
				},
			},
			wantInstruction:  true,
			wantTextContains: []string{"PAST_CONVERSATIONS", "user: Hello world", "Time: " + time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC).Format(time.RFC3339)},
		},
		{
			name:        "multiple memory entries",
			userContent: genai.NewContentFromText("search term", genai.RoleUser),
			memories: []memory.Entry{
				{
					Author:    "user",
					Timestamp: time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC),
					Content:   genai.NewContentFromText("First memory", genai.RoleUser),
				},
				{
					Author:    "assistant",
					Timestamp: time.Date(2025, 1, 2, 12, 0, 0, 0, time.UTC),
					Content:   genai.NewContentFromText("Second memory", genai.RoleModel),
				},
			},
			wantInstruction:  true,
			wantTextContains: []string{"First memory", "Second memory", "user:", "assistant:"},
		},
		{
			name:        "memory entry without author",
			userContent: genai.NewContentFromText("test", genai.RoleUser),
			memories: []memory.Entry{
				{
					Timestamp: time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC),
					Content:   genai.NewContentFromText("Anonymous message", genai.RoleUser),
				},
			},
			wantInstruction:  true,
			wantTextContains: []string{"Anonymous message"},
		},
		{
			name:        "memory entry without timestamp",
			userContent: genai.NewContentFromText("test", genai.RoleUser),
			memories: []memory.Entry{
				{
					Author:  "user",
					Content: genai.NewContentFromText("No timestamp", genai.RoleUser),
				},
			},
			wantInstruction:  true,
			wantTextContains: []string{"user: No timestamp"},
		},
		{
			name:        "memory entry with empty content",
			userContent: genai.NewContentFromText("test", genai.RoleUser),
			memories: []memory.Entry{
				{
					Author:    "user",
					Timestamp: time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC),
					Content:   nil,
				},
			},
			wantInstruction: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc := createToolContext(t, &mockMemory{memories: tt.memories, err: tt.searchErr}, tt.userContent)
			llmRequest := &model.LLMRequest{}

			pmt := preloadmemorytool.New()

			err := pmt.ProcessRequest(tc, llmRequest)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("ProcessRequest expected error, got nil")
				}
			} else if err != nil {
				t.Fatalf("ProcessRequest failed: %v", err)
			}

			hasInstruction := llmRequest.Config != nil && llmRequest.Config.SystemInstruction != nil
			if hasInstruction != tt.wantInstruction {
				t.Errorf("hasInstruction = %v, want %v", hasInstruction, tt.wantInstruction)
			}

			if tt.wantInstruction && hasInstruction {
				instruction := llmRequest.Config.SystemInstruction.Parts[0].Text
				for _, want := range tt.wantTextContains {
					if !strings.Contains(instruction, want) {
						t.Errorf("Instruction should contain %q, got: %v", want, instruction)
					}
				}
			}
		})
	}
}

func createToolContext(t *testing.T, mem *mockMemory, userContent *genai.Content) tool.Context {
	t.Helper()

	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Memory:      mem,
		UserContent: userContent,
	})

	return toolinternal.NewToolContext(ctx, "", nil, nil)
}
