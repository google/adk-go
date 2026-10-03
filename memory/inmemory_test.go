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

func Test_inMemoryService_AddEventsToMemory(t *testing.T) {
	newEvent := func(id, text string) *session.Event {
		return &session.Event{
			ID:     id,
			Author: "user1",
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText(text, genai.RoleUser),
			},
		}
	}

	t.Run("events become searchable", func(t *testing.T) {
		s := memory.InMemoryService()
		adder := s.(memory.AddEventsToMemoryer)
		err := adder.AddEventsToMemory(t.Context(), &memory.AddEventsToMemoryRequest{
			AppName:   "app1",
			UserID:    "user1",
			SessionID: "sess1",
			Events:    []*session.Event{newEvent("event1", "The quick brown fox")},
		})
		if err != nil {
			t.Fatalf("AddEventsToMemory() error = %v", err)
		}

		got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "fox"})
		if err != nil {
			t.Fatalf("SearchMemory() error = %v", err)
		}
		if len(got.Memories) != 1 || got.Memories[0].ID != "event1" {
			t.Errorf("SearchMemory() = %+v, want a single match for event1", got.Memories)
		}
	})

	t.Run("repeated calls with overlapping events are deduped by ID", func(t *testing.T) {
		s := memory.InMemoryService()
		adder := s.(memory.AddEventsToMemoryer)
		req := &memory.AddEventsToMemoryRequest{
			AppName:   "app1",
			UserID:    "user1",
			SessionID: "sess1",
			Events:    []*session.Event{newEvent("event1", "The quick brown fox")},
		}
		if err := adder.AddEventsToMemory(t.Context(), req); err != nil {
			t.Fatalf("AddEventsToMemory() [1] error = %v", err)
		}
		req.Events = []*session.Event{newEvent("event1", "The quick brown fox"), newEvent("event2", "jumps over the lazy dog")}
		if err := adder.AddEventsToMemory(t.Context(), req); err != nil {
			t.Fatalf("AddEventsToMemory() [2] error = %v", err)
		}

		got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "fox dog"})
		if err != nil {
			t.Fatalf("SearchMemory() error = %v", err)
		}
		if len(got.Memories) != 2 {
			t.Errorf("SearchMemory() = %+v, want 2 deduped memories", got.Memories)
		}
	})

	t.Run("does not affect other sessions or users", func(t *testing.T) {
		s := memory.InMemoryService()
		adder := s.(memory.AddEventsToMemoryer)
		if err := adder.AddEventsToMemory(t.Context(), &memory.AddEventsToMemoryRequest{
			AppName:   "app1",
			UserID:    "user1",
			SessionID: "sess1",
			Events:    []*session.Event{newEvent("event1", "unique-marker-word")},
		}); err != nil {
			t.Fatalf("AddEventsToMemory() error = %v", err)
		}

		got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app1", UserID: "user2", Query: "unique-marker-word"})
		if err != nil {
			t.Fatalf("SearchMemory() error = %v", err)
		}
		if len(got.Memories) != 0 {
			t.Errorf("SearchMemory() leaked across users, got %+v", got.Memories)
		}
	})

	t.Run("events without content are ignored", func(t *testing.T) {
		s := memory.InMemoryService()
		adder := s.(memory.AddEventsToMemoryer)
		err := adder.AddEventsToMemory(t.Context(), &memory.AddEventsToMemoryRequest{
			AppName:   "app1",
			UserID:    "user1",
			SessionID: "sess1",
			Events:    []*session.Event{{ID: "event1", Author: "user1"}},
		})
		if err != nil {
			t.Fatalf("AddEventsToMemory() error = %v", err)
		}

		got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "anything"})
		if err != nil {
			t.Fatalf("SearchMemory() error = %v", err)
		}
		if len(got.Memories) != 0 {
			t.Errorf("SearchMemory() = %+v, want no memories for a contentless event", got.Memories)
		}
	})
}

// Test_inMemoryService_AddEventsToMemory_DisjointCallsAccumulate pins the
// incremental behaviour this PR adds: two calls carrying disjoint events must
// both survive. A wholesale-overwrite implementation that keeps none of a prior
// call's events would leave only the second call's event searchable and fail.
func Test_inMemoryService_AddEventsToMemory_DisjointCallsAccumulate(t *testing.T) {
	s := memory.InMemoryService()
	adder := s.(memory.AddEventsToMemoryer)
	add := func(id, text string) {
		t.Helper()
		if err := adder.AddEventsToMemory(t.Context(), &memory.AddEventsToMemoryRequest{
			AppName: "app1", UserID: "user1", SessionID: "sess1",
			Events: []*session.Event{memoryTextEvent(id, text)},
		}); err != nil {
			t.Fatalf("AddEventsToMemory(%s) error = %v", id, err)
		}
	}
	add("event1", "The quick brown fox")
	add("event2", "jumps over the lazy dog")

	got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "fox dog"})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}
	if len(got.Memories) != 2 {
		t.Errorf("SearchMemory() = %+v, want 2 memories from two disjoint calls", got.Memories)
	}
}

// Test_inMemoryService_AddEventsToMemory_SessionScopedDedup checks that dedup is
// per (app, user, session), not per (app, user): the same ID stored under two
// sessions of one user must both remain searchable. A bare "different user"
// case cannot catch this because SearchMemory scans every session bucket of a
// user and never observes which bucket an event landed in.
func Test_inMemoryService_AddEventsToMemory_SessionScopedDedup(t *testing.T) {
	s := memory.InMemoryService()
	adder := s.(memory.AddEventsToMemoryer)
	for _, sid := range []string{"sess1", "sess2"} {
		if err := adder.AddEventsToMemory(t.Context(), &memory.AddEventsToMemoryRequest{
			AppName: "app1", UserID: "user1", SessionID: sid,
			Events: []*session.Event{memoryTextEvent("event1", "shared marker word")},
		}); err != nil {
			t.Fatalf("AddEventsToMemory() for %s error = %v", sid, err)
		}
	}

	got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "marker"})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}
	if len(got.Memories) != 2 {
		t.Errorf("SearchMemory() = %+v, want one memory per session", got.Memories)
	}
}

// Test_inMemoryService_AddEventsToMemory_ContentlessIdDoesNotBlockRealEvent
// checks that a contentless event does not register its ID and knock out a real
// event that reuses the same ID in the same batch. Dropping the content guard
// for within-batch dedup would make the real event the one discarded, and
// nothing in the other subtests would notice.
func Test_inMemoryService_AddEventsToMemory_ContentlessIdDoesNotBlockRealEvent(t *testing.T) {
	s := memory.InMemoryService()
	adder := s.(memory.AddEventsToMemoryer)
	if err := adder.AddEventsToMemory(t.Context(), &memory.AddEventsToMemoryRequest{
		AppName: "app1", UserID: "user1", SessionID: "sess1",
		Events: []*session.Event{
			{ID: "event1", Author: "user1"},
			memoryTextEvent("event1", "real marker content"),
		},
	}); err != nil {
		t.Fatalf("AddEventsToMemory() error = %v", err)
	}

	got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "marker"})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}
	if len(got.Memories) != 1 || got.Memories[0].ID != "event1" {
		t.Errorf("SearchMemory() = %+v, want the single real event1 stored", got.Memories)
	}
}

// Test_inMemoryService_AddEventsToMemory_DuplicateWithinBatch checks that a
// duplicate ID within a single batch is stored once, pinning the within-batch
// guard that no separate-call test reaches.
func Test_inMemoryService_AddEventsToMemory_DuplicateWithinBatch(t *testing.T) {
	s := memory.InMemoryService()
	adder := s.(memory.AddEventsToMemoryer)
	if err := adder.AddEventsToMemory(t.Context(), &memory.AddEventsToMemoryRequest{
		AppName: "app1", UserID: "user1", SessionID: "sess1",
		Events: []*session.Event{
			memoryTextEvent("event1", "dup marker word"),
			memoryTextEvent("event1", "dup marker word"),
		},
	}); err != nil {
		t.Fatalf("AddEventsToMemory() error = %v", err)
	}

	got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "marker"})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}
	if len(got.Memories) != 1 {
		t.Errorf("SearchMemory() = %+v, want the duplicate stored once", got.Memories)
	}
}

// Test_inMemoryService_AddEventsToMemory_EmptyIDNotDropped guards against the
// dedup key collapsing on an empty event ID. Events built by hand carry no ID
// (session.Event.ID is "Set by storage"), and an ID-less event must not claim
// the dedup slot and knock out every later ID-less event on the same scope —
// neither within one batch nor across AddSessionToMemory and a later
// AddEventsToMemory for that scope.
func Test_inMemoryService_AddEventsToMemory_EmptyIDNotDropped(t *testing.T) {
	evt := func(text string, role genai.Role) *session.Event {
		return &session.Event{
			Author:      string(role),
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, role)},
		}
	}

	s := memory.InMemoryService()
	adder := s.(memory.AddEventsToMemoryer)
	if err := adder.AddEventsToMemory(t.Context(), &memory.AddEventsToMemoryRequest{
		AppName: "app1", UserID: "user1", SessionID: "sess1",
		Events: []*session.Event{
			evt("hello world", genai.RoleUser),
			evt("how can I help you", genai.RoleModel),
		},
	}); err != nil {
		t.Fatalf("AddEventsToMemory() error = %v", err)
	}
	got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "help"})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}
	if len(got.Memories) != 1 {
		t.Errorf("SearchMemory() = %+v, want the ID-less model turn, not dropped", got.Memories)
	}

	// The same across the two ingestion paths: an ID-less event ingested via
	// AddSessionToMemory must not block a later ID-less AddEventsToMemory.
	s = memory.InMemoryService()
	adder = s.(memory.AddEventsToMemoryer)
	if err := s.AddSessionToMemory(t.Context(), makeSession(t, "app1", "user1", "sess1", []*session.Event{
		evt("hello world", genai.RoleUser),
	})); err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}
	if err := adder.AddEventsToMemory(t.Context(), &memory.AddEventsToMemoryRequest{
		AppName: "app1", UserID: "user1", SessionID: "sess1",
		Events: []*session.Event{evt("how can I help you", genai.RoleModel)},
	}); err != nil {
		t.Fatalf("AddEventsToMemory() error = %v", err)
	}
	got, err = s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "help"})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}
	if len(got.Memories) != 1 {
		t.Errorf("SearchMemory() = %+v, want the ID-less event added after AddSessionToMemory", got.Memories)
	}
}

// Test_inMemoryService_AddEventsToMemory_Concurrent drives AddEventsToMemory
// concurrently with the other two writers and with SearchMemory, all on one
// (app, user, session) so the writers genuinely contend for the write lock and
// searchers read the same bucket they are being written to. The service is
// documented as thread-safe, so this must stay clean under -race, as CI runs it.
func Test_inMemoryService_AddEventsToMemory_Concurrent(t *testing.T) {
	s := memory.InMemoryService()
	adder := s.(memory.AddEventsToMemoryer)
	ctx := t.Context()

	const workers = 8
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(3)
		// Incremental ingester: distinct IDs so nothing silently drops on dedup.
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				id := "e-" + strconv.Itoa(i) + "-" + strconv.Itoa(j)
				if err := adder.AddEventsToMemory(ctx, &memory.AddEventsToMemoryRequest{
					AppName: "app1", UserID: "user1", SessionID: "sess1",
					Events: []*session.Event{memoryTextEvent(id, "marker")},
				}); err != nil {
					t.Errorf("AddEventsToMemory() error = %v", err)
					return
				}
			}
		}(i)
		// Full-session ingester on the same scope: wholesale-replaces the slot.
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				id := "seed-" + strconv.Itoa(i) + "-" + strconv.Itoa(j)
				if err := s.AddSessionToMemory(ctx, makeSession(t, "app1", "user1", "sess1", []*session.Event{
					memoryTextEvent(id, "marker"),
				})); err != nil {
					t.Errorf("AddSessionToMemory() error = %v", err)
					return
				}
			}
		}(i)
		// Concurrency checker reads the bucket the writers are mutating.
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				if _, err := s.SearchMemory(ctx, &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "marker"}); err != nil {
					t.Errorf("SearchMemory() error = %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
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

	// Seed a matching event so searchers also exercise scoring and sorting.
	if err := s.AddSessionToMemory(ctx, makeSession(t, "app1", "user1", "seed", []*session.Event{memoryTextEvent("seed", "x")})); err != nil {
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
				if err := s.AddSessionToMemory(ctx, makeSession(t, "app1", "user1", id, []*session.Event{memoryTextEvent(id, "x")})); err != nil {
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
