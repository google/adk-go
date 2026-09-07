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

package memory_test

import (
	"iter"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// maxSearchResultsForTest mirrors the unexported cap in the memory package.
const maxSearchResultsForTest = 10

func Test_inMemoryService_SearchMemory(t *testing.T) {
	tests := []struct {
		name         string
		initSessions []session.Session
		req          *memory.SearchRequest
		wantResp     *memory.SearchResponse
		wantErr      bool
	}{
		{
			name: "find events",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess1", []*session.Event{
					{
						ID:     "event1",
						Author: "user1",
						LLMResponse: model.LLMResponse{
							Content:        genai.NewContentFromText("The Quick brown fox", genai.RoleUser),
							CustomMetadata: map[string]any{"key": "value"},
						},
						Timestamp: must(time.Parse(time.RFC3339, "2023-10-01T10:00:00Z")),
					},
					{
						LLMResponse: model.LLMResponse{
							Content: genai.NewContentFromText("jumps over the lazy dog", genai.RoleModel),
						},
					},
				}),
				makeSession(t, "app1", "user1", "sess2", []*session.Event{
					{
						Author:      "test-bot",
						LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hello world", genai.RoleModel)},
						Timestamp:   must(time.Parse(time.RFC3339, "2023-10-02T10:00:00Z")),
					},
				}),
				makeSession(t, "app1", "user1", "sess3", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("test text", genai.RoleUser)}},
				}),
			},
			req: &memory.SearchRequest{
				AppName: "app1",
				UserID:  "user1",
				Query:   "quick hello",
			},
			wantResp: &memory.SearchResponse{
				Memories: []memory.Entry{
					{
						ID:             "event1",
						Content:        genai.NewContentFromText("The Quick brown fox", genai.RoleUser),
						Author:         "user1",
						Timestamp:      must(time.Parse(time.RFC3339, "2023-10-01T10:00:00Z")),
						CustomMetadata: map[string]any{"key": "value"},
					},
					{
						Content:   genai.NewContentFromText("hello world", genai.RoleModel),
						Author:    "test-bot",
						Timestamp: must(time.Parse(time.RFC3339, "2023-10-02T10:00:00Z")),
					},
				},
			},
		},
		{
			name: "no leakage for different appName",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess3", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("test text", genai.RoleUser)}},
				}),
			},
			req: &memory.SearchRequest{
				AppName: "other_app",
				UserID:  "user1",
				Query:   "test text",
			},
			wantResp: &memory.SearchResponse{},
		},
		{
			name: "no leakage for different user",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess3", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("test text", genai.RoleUser)}},
				}),
			},
			req: &memory.SearchRequest{
				AppName: "app1",
				UserID:  "test_user",
				Query:   "test text",
			},
			wantResp: &memory.SearchResponse{},
		},
		{
			name: "no matches",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess3", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("test text", genai.RoleUser)}},
				}),
			},
			req: &memory.SearchRequest{
				AppName: "app1",
				UserID:  "test_user",
				Query:   "something different",
			},
			wantResp: &memory.SearchResponse{},
		},
		{
			name: "lookup on empty store",
			req: &memory.SearchRequest{
				AppName: "app1",
				UserID:  "test_user",
				Query:   "something different",
			},
			wantResp: &memory.SearchResponse{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := memory.InMemoryService()

			for _, session := range tt.initSessions {
				if err := s.AddSessionToMemory(t.Context(), session); err != nil {
					t.Fatalf("inMemoryService.AddSessionToMemory() error = %v", err)
				}
			}

			got, err := s.SearchMemory(t.Context(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Fatalf("inMemoryService.SearchMemory() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.wantResp, got, sortMemories); diff != "" {
				t.Errorf("inMemoryiService.SearchMemory() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func makeSession(t *testing.T, appName, userID, sessionID string, events []*session.Event) session.Session {
	t.Helper()

	return &testSession{
		appName:   appName,
		userID:    userID,
		sessionID: sessionID,
		events:    events,
	}
}

var sortMemories = cmp.Transformer("Sort", func(in *memory.SearchResponse) *memory.SearchResponse {
	slices.SortFunc(in.Memories, func(m1, m2 memory.Entry) int {
		return m1.Timestamp.Compare(m2.Timestamp)
	})
	return in
})

type testSession struct {
	appName, userID, sessionID string
	events                     []*session.Event
}

func (s *testSession) ID() string {
	return s.sessionID
}

func (s *testSession) AppName() string {
	return s.appName
}

func (s *testSession) UserID() string {
	return s.userID
}

func (s *testSession) Events() session.Events {
	return s
}

func (s *testSession) All() iter.Seq[*session.Event] {
	return slices.Values(s.events)
}

func (s *testSession) Len() int {
	return len(s.events)
}

func (s *testSession) At(i int) *session.Event {
	return s.events[i]
}

func (s *testSession) State() session.State {
	panic("not implemented")
}

func (s *testSession) LastUpdateTime() time.Time {
	panic("not implemented")
}

func must[V any](v V, err error) V {
	if err != nil {
		panic(err)
	}
	return v
}

// Test_inMemoryService_SearchMemory_Concurrent runs SearchMemory and
// AddSessionToMemory on the same app/user in parallel. The service is
// documented as thread-safe, so under -race (as CI runs it) the two must not
// touch the per-user session map without synchronization.
func Test_inMemoryService_SearchMemory_Concurrent(t *testing.T) {
	s := memory.InMemoryService()
	ctx := t.Context()

	// Seed one session so the per-user map is non-empty while searchers iterate.
	if err := s.AddSessionToMemory(ctx, makeSession(t, "app1", "user1", "seed", nil)); err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}

	const workers = 8
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(2)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				id := "s" + strconv.Itoa(i) + "-" + strconv.Itoa(j)
				if err := s.AddSessionToMemory(ctx, makeSession(t, "app1", "user1", id, nil)); err != nil {
					t.Errorf("AddSessionToMemory() error = %v", err)
					return
				}
			}
		}(i)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				if _, err := s.SearchMemory(ctx, &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "x"}); err != nil {
					t.Errorf("SearchMemory() error = %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

// memoryIDs returns the IDs of the returned memories in order. Ordering is
// meaningful once results are ranked, so these tests compare the sequence
// directly rather than through the order-insensitive sortMemories transformer.
func memoryIDs(resp *memory.SearchResponse) []string {
	ids := make([]string, 0, len(resp.Memories))
	for _, m := range resp.Memories {
		ids = append(ids, m.ID)
	}

	return ids
}

func Test_inMemoryService_SearchMemory_RanksByMatchCount(t *testing.T) {
	s := memory.InMemoryService()
	events := []*session.Event{
		{ID: "one", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("quick dog", genai.RoleUser)}},
		{ID: "three", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("quick brown fox", genai.RoleUser)}},
		{ID: "two", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("quick brown cat", genai.RoleUser)}},
	}
	if err := s.AddSessionToMemory(t.Context(), makeSession(t, "app1", "user1", "sess1", events)); err != nil {
		t.Fatalf("inMemoryService.AddSessionToMemory() error = %v", err)
	}

	got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{
		AppName: "app1",
		UserID:  "user1",
		Query:   "quick brown fox",
	})
	if err != nil {
		t.Fatalf("inMemoryService.SearchMemory() error = %v", err)
	}

	want := []string{"three", "two", "one"}
	if diff := cmp.Diff(want, memoryIDs(got)); diff != "" {
		t.Errorf("inMemoryService.SearchMemory() order mismatch (-want +got):\n%s", diff)
	}
}

func Test_inMemoryService_SearchMemory_CapsResults(t *testing.T) {
	s := memory.InMemoryService()

	var events []*session.Event
	for i := range maxSearchResultsForTest + 5 {
		events = append(events, &session.Event{
			ID: strconv.Itoa(i),
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("shared "+strconv.Itoa(i), genai.RoleUser),
			},
		})
	}

	if err := s.AddSessionToMemory(t.Context(), makeSession(t, "app1", "user1", "sess1", events)); err != nil {
		t.Fatalf("inMemoryService.AddSessionToMemory() error = %v", err)
	}

	got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{
		AppName: "app1",
		UserID:  "user1",
		Query:   "shared",
	})
	if err != nil {
		t.Fatalf("inMemoryService.SearchMemory() error = %v", err)
	}

	if len(got.Memories) != maxSearchResultsForTest {
		t.Errorf("inMemoryService.SearchMemory() returned %d memories, want %d",
			len(got.Memories), maxSearchResultsForTest)
	}
}

func Test_inMemoryService_SearchMemory_OrderIsStableAcrossRuns(t *testing.T) {
	// Sessions are stored in a map, so without a fixed iteration order equally
	// scored events come back in a different sequence on each call.
	s := memory.InMemoryService()
	for i := range 8 {
		id := strconv.Itoa(i)
		events := []*session.Event{
			{ID: id, LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("shared text", genai.RoleUser)}},
		}
		if err := s.AddSessionToMemory(t.Context(), makeSession(t, "app1", "user1", "sess"+id, events)); err != nil {
			t.Fatalf("inMemoryService.AddSessionToMemory() error = %v", err)
		}
	}

	req := &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "shared"}

	first, err := s.SearchMemory(t.Context(), req)
	if err != nil {
		t.Fatalf("inMemoryService.SearchMemory() error = %v", err)
	}

	want := memoryIDs(first)
	if len(want) != 8 {
		t.Fatalf("inMemoryService.SearchMemory() returned %d memories, want 8", len(want))
	}

	for i := range 25 {
		got, err := s.SearchMemory(t.Context(), req)
		if err != nil {
			t.Fatalf("inMemoryService.SearchMemory() error = %v", err)
		}
		if diff := cmp.Diff(want, memoryIDs(got)); diff != "" {
			t.Fatalf("inMemoryService.SearchMemory() order changed on run %d (-first +got):\n%s", i, diff)
		}
	}
}
