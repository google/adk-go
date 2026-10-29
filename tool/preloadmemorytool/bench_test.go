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
	"fmt"
	"sync/atomic"
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

type countingMemory struct {
	entries     []memory.Entry
	searchCalls atomic.Int64
}

func (m *countingMemory) AddSessionToMemory(ctx context.Context, s session.Session) error {
	return nil
}

func (m *countingMemory) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	m.searchCalls.Add(1)
	return &memory.SearchResponse{Memories: m.entries}, nil
}

// BenchmarkPreloadMemoryToolLoopTurn measures preloadmemorytool.ProcessRequest
// across a 4-model-call tool loop within a single invocation turn.
func BenchmarkPreloadMemoryToolLoopTurn(b *testing.B) {
	ctx := b.Context()
	entries := make([]memory.Entry, 5)
	for i := range entries {
		entries[i] = memory.Entry{
			Author:    "user",
			Timestamp: time.Date(2026, 1, i+1, 12, 0, 0, 0, time.UTC),
			Content:   genai.NewContentFromText(fmt.Sprintf("past memory entry %d", i), genai.RoleUser),
		}
	}
	mem := &countingMemory{entries: entries}
	userContent := genai.NewContentFromText("what did we discuss?", genai.RoleUser)
	pmt := preloadmemorytool.New()

	mem.searchCalls.Store(0)
	b.ReportAllocs()
	b.ResetTimer()

	for i := range b.N {
		invCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
			Memory:       mem,
			UserContent:  userContent,
			InvocationID: fmt.Sprintf("inv-%d", i),
		})
		tc := agent.NewToolContext(invCtx, "", nil, nil)
		for range 4 {
			req := &model.LLMRequest{}
			if err := pmt.ProcessRequest(tc, req); err != nil {
				b.Fatalf("ProcessRequest failed: %v", err)
			}
		}
	}

	b.StopTimer()
	b.ReportMetric(float64(mem.searchCalls.Load())/float64(b.N), "memory-searches/turn")
}
