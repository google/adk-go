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

package loadmemorytool_test

import (
	"context"
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
	"google.golang.org/adk/tool/loadmemorytool"
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

func TestLoadMemoryTool_BasicProperties(t *testing.T) {
	tool := loadmemorytool.New()

	if got := tool.Name(); got != "load_memory" {
		t.Errorf("Name() = %v, want load_memory", got)
	}
	if got := tool.Description(); got != "Loads the memory for the current user." {
		t.Errorf("Description() = %v, want 'Loads the memory for the current user.'", got)
	}
	if got := tool.IsLongRunning(); got != false {
		t.Errorf("IsLongRunning() = %v, want false", got)
	}
}

func TestLoadMemoryTool_Run(t *testing.T) {
	tool := loadmemorytool.New()

	toolImpl, ok := tool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("loadMemoryTool does not implement FunctionTool")
	}

	tests := []struct {
		name     string
		args     map[string]any
		memories []memory.Entry
		wantLen  int
		wantErr  bool
	}{
		{
			name:     "empty memories",
			args:     map[string]any{"query": "test query"},
			memories: []memory.Entry{},
			wantLen:  0,
		},
		{
			name: "single memory entry",
			args: map[string]any{"query": "test query"},
			memories: []memory.Entry{
				{
					Author:    "user",
					Timestamp: time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC),
					Content:   genai.NewContentFromText("Hello world", genai.RoleUser),
				},
			},
			wantLen: 1,
		},
		{
			name: "multiple memory entries",
			args: map[string]any{"query": "search term"},
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
			wantLen: 2,
		},
		{
			name:    "missing query parameter",
			args:    map[string]any{},
			wantErr: true,
		},
		{
			name:    "invalid query type",
			args:    map[string]any{"query": 123},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc := createToolContext(t, &mockMemory{memories: tt.memories})

			result, err := toolImpl.Run(tc, tt.args)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Run() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}

			memories, ok := result["memories"].([]memory.Entry)
			if !ok {
				t.Fatalf("result['memories'] is not []memory.Entry, got %T", result["memories"])
			}
			if len(memories) != tt.wantLen {
				t.Errorf("Run() returned %d memories, want %d", len(memories), tt.wantLen)
			}
		})
	}
}

func TestLoadMemoryTool_ProcessRequest(t *testing.T) {
	tool := loadmemorytool.New()

	tc := createToolContext(t, &mockMemory{})
	llmRequest := &model.LLMRequest{}

	requestProcessor, ok := tool.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatal("loadMemoryTool does not implement RequestProcessor")
	}

	err := requestProcessor.ProcessRequest(tc, llmRequest)
	if err != nil {
		t.Fatalf("ProcessRequest failed: %v", err)
	}

	if llmRequest.Config == nil || llmRequest.Config.SystemInstruction == nil {
		t.Fatal("ProcessRequest did not set SystemInstruction")
	}

	instruction := llmRequest.Config.SystemInstruction.Parts[0].Text
	if !strings.Contains(instruction, "You have memory") {
		t.Errorf("Instruction should contain 'You have memory', got: %v", instruction)
	}
	if !strings.Contains(instruction, "load_memory") {
		t.Errorf("Instruction should contain 'load_memory', got: %v", instruction)
	}
}

func createToolContext(t *testing.T, mem *mockMemory) tool.Context {
	t.Helper()

	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Memory: mem,
	})

	return toolinternal.NewToolContext(ctx, "", nil)
}
