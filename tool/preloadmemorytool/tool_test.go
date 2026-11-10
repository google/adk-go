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

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/preloadmemorytool"
)

type mockMemory struct {
	memories []memory.Entry
	err      error

	searchCalls int
	lastQuery   string
}

func (m *mockMemory) AddSessionToMemory(ctx context.Context, s session.Session) error {
	return nil
}

func (m *mockMemory) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	m.searchCalls++
	m.lastQuery = query
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
		wantNoSearch     bool
		wantQuery        string
		wantInstruction  bool
		wantTextContains []string
		wantTextOmits    []string
	}{
		{
			name:            "nil user content",
			userContent:     nil,
			wantNoSearch:    true,
			wantInstruction: false,
		},
		{
			name:            "empty user content parts",
			userContent:     &genai.Content{Parts: []*genai.Part{}},
			wantNoSearch:    true,
			wantInstruction: false,
		},
		{
			name:            "user content with no text",
			userContent:     &genai.Content{Parts: []*genai.Part{{InlineData: &genai.Blob{}}}},
			wantNoSearch:    true,
			wantInstruction: false,
		},
		{
			name:        "text after an image is searched",
			userContent: &genai.Content{Parts: []*genai.Part{{InlineData: &genai.Blob{}}, genai.NewPartFromText("later text")}},
			memories: []memory.Entry{
				{Author: "user", Content: genai.NewContentFromText("a memory", genai.RoleUser)},
			},
			wantQuery:        "later text",
			wantInstruction:  true,
			wantTextContains: []string{"a memory"},
		},
		{
			name:        "multiple text parts are joined",
			userContent: &genai.Content{Parts: []*genai.Part{genai.NewPartFromText("my"), genai.NewPartFromText("preferences")}},
			memories: []memory.Entry{
				{Author: "user", Content: genai.NewContentFromText("a memory", genai.RoleUser)},
			},
			wantQuery:        "my preferences",
			wantInstruction:  true,
			wantTextContains: []string{"a memory"},
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
			name:        "memories with nil and empty parts",
			userContent: genai.NewContentFromText("test query", genai.RoleUser),
			memories: []memory.Entry{
				{
					Author:    "user",
					Timestamp: time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC),
					Content: &genai.Content{
						Parts: []*genai.Part{
							nil,
							{Text: "valid memory"},
							nil,
							{Text: ""},
						},
					},
				},
			},
			wantInstruction: true,
			wantTextContains: []string{
				"user: valid memory\n", // The newline catches a trailing separator.
			},
		},
		{
			name:        "empty text between text parts adds no extra separator",
			userContent: genai.NewContentFromText("test query", genai.RoleUser),
			memories: []memory.Entry{
				{
					Author: "user",
					Content: &genai.Content{
						Parts: []*genai.Part{
							{Text: "valid"},
							{Text: ""},
							{Text: "memory"},
						},
					},
				},
			},
			wantInstruction:  true,
			wantTextContains: []string{"user: valid memory"},
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
			name:        "thought parts are omitted",
			userContent: genai.NewContentFromText("test", genai.RoleUser),
			memories: []memory.Entry{
				{
					Author: "model",
					Content: &genai.Content{
						Parts: []*genai.Part{
							{Text: "private reasoning", Thought: true},
							{Text: "visible answer"},
						},
					},
				},
			},
			wantInstruction:  true,
			wantTextContains: []string{"model: visible answer"},
			wantTextOmits:    []string{"private reasoning"},
		},
		{
			name:        "thought between text parts adds no separator",
			userContent: genai.NewContentFromText("test", genai.RoleUser),
			memories: []memory.Entry{
				{
					Author: "model",
					Content: &genai.Content{
						Parts: []*genai.Part{
							{Text: "visible"},
							{Text: "private reasoning", Thought: true},
							{Text: "answer"},
						},
					},
				},
			},
			wantInstruction:  true,
			wantTextContains: []string{"\nmodel: visible answer\n"},
			wantTextOmits:    []string{"private reasoning"},
		},
		{
			name:        "thought after text adds no trailing separator",
			userContent: genai.NewContentFromText("test", genai.RoleUser),
			memories: []memory.Entry{
				{
					Author: "model",
					Content: &genai.Content{
						Parts: []*genai.Part{
							{Text: "visible answer"},
							{Text: "private reasoning", Thought: true},
						},
					},
				},
			},
			wantInstruction:  true,
			wantTextContains: []string{"\nmodel: visible answer\n"},
			wantTextOmits:    []string{"private reasoning"},
		},
		{
			name:        "thought-only memory entry",
			userContent: genai.NewContentFromText("test", genai.RoleUser),
			memories: []memory.Entry{
				{
					Author:    "model",
					Timestamp: time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC),
					Content: &genai.Content{
						Parts: []*genai.Part{{Text: "private reasoning", Thought: true}},
					},
				},
			},
			wantInstruction: false,
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
			mem := &mockMemory{memories: tt.memories, err: tt.searchErr}
			tc := createToolContext(t, mem, tt.userContent)
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

			if tt.wantNoSearch {
				if mem.searchCalls != 0 {
					t.Errorf("SearchMemory() called %d times, want 0", mem.searchCalls)
				}
			} else if tt.wantQuery != "" {
				if mem.searchCalls != 1 {
					t.Errorf("SearchMemory() called %d times, want 1", mem.searchCalls)
				} else if mem.lastQuery != tt.wantQuery {
					t.Errorf("SearchMemory() query = %q, want %q", mem.lastQuery, tt.wantQuery)
				}
			}

			hasInstruction := llmRequest.Config != nil && llmRequest.Config.SystemInstruction != nil
			if hasInstruction != tt.wantInstruction {
				t.Errorf("hasInstruction = %v, want %v", hasInstruction, tt.wantInstruction)
			}

			if tt.wantInstruction && hasInstruction {
				instruction := llmRequest.Config.SystemInstruction.Parts[0].Text
				for i, want := range tt.wantTextContains {
					if !strings.Contains(instruction, want) {
						t.Errorf("instruction is missing expected text at index %d", i)
					}
				}
				for i, omit := range tt.wantTextOmits {
					if strings.Contains(instruction, omit) {
						t.Errorf("instruction contains omitted text at index %d", i)
					}
				}
			}
		})
	}
}

func createToolContext(t *testing.T, mem *mockMemory, userContent *genai.Content) agent.Context {
	t.Helper()

	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Memory:      mem,
		UserContent: userContent,
	})

	return agent.NewToolContext(ctx, "", nil, nil)
}
