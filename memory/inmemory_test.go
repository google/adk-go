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
	"context"
	"errors"
	"iter"
	"slices"
	"strconv"
	"strings"
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
			name: "matches punctuation and non-space whitespace",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess-punctuation", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Error: connection\ntimeout! Please\tretry.", genai.RoleModel)}},
				}),
			},
			req: &memory.SearchRequest{
				AppName: "app1",
				UserID:  "user1",
				Query:   "error timeout retry",
			},
			wantResp: &memory.SearchResponse{
				Memories: []memory.Entry{
					{Content: genai.NewContentFromText("Error: connection\ntimeout! Please\tretry.", genai.RoleModel)},
				},
			},
		},
		{
			name: "splits newline independently",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess-newline-only", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("prefix\nzebra", genai.RoleModel)}},
				}),
			},
			req: &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "zebra"},
			wantResp: &memory.SearchResponse{Memories: []memory.Entry{
				{Content: genai.NewContentFromText("prefix\nzebra", genai.RoleModel)},
			}},
		},
		{
			name: "splits tab independently",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess-tab-only", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("prefix\twalrus", genai.RoleModel)}},
				}),
			},
			req: &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "walrus"},
			wantResp: &memory.SearchResponse{Memories: []memory.Entry{
				{Content: genai.NewContentFromText("prefix\twalrus", genai.RoleModel)},
			}},
		},
		{
			name: "splits internal punctuation",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess-internal-punctuation", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("error,timeout Use the built-in tool.", genai.RoleModel)}},
				}),
			},
			req: &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "timeout built"},
			wantResp: &memory.SearchResponse{Memories: []memory.Entry{
				{Content: genai.NewContentFromText("error,timeout Use the built-in tool.", genai.RoleModel)},
			}},
		},
		{
			name: "keeps digits as words",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess-digits", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Error 404.", genai.RoleModel)}},
				}),
			},
			req: &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "404"},
			wantResp: &memory.SearchResponse{Memories: []memory.Entry{
				{Content: genai.NewContentFromText("Error 404.", genai.RoleModel)},
			}},
		},
		{
			name: "preserves combining marks and underscores",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess-marks", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("CAFE\u0301TERIA snake_case", genai.RoleModel)}},
				}),
			},
			req: &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "cafe\u0301 snake_case"},
			wantResp: &memory.SearchResponse{Memories: []memory.Entry{
				{Content: genai.NewContentFromText("CAFE\u0301TERIA snake_case", genai.RoleModel)},
			}},
		},
		{
			name: "does not treat leading Unicode marks as words",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess-leading-mark", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("⚠️ disk full", genai.RoleModel)}},
				}),
			},
			req:      &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "thanks ❤️"},
			wantResp: &memory.SearchResponse{},
		},
		{
			name: "preserves combining mark boundary",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess-combining-negative", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Je bois un cafe\u0301.", genai.RoleModel)}},
				}),
			},
			req:      &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "cafe"},
			wantResp: &memory.SearchResponse{},
		},
		{
			name: "preserves underscore boundary",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess-underscore-negative", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("call snake_case now", genai.RoleModel)}},
				}),
			},
			req:      &memory.SearchRequest{AppName: "app1", UserID: "user1", Query: "snake"},
			wantResp: &memory.SearchResponse{},
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

func TestAddEventsToMemory_BasicAndEntryFields(t *testing.T) {
	svc := memory.InMemoryService()
	ev := &session.Event{
		ID:        "ev-1",
		Author:    "user",
		Timestamp: time.Date(2025, 1, 2, 3, 4, 5, 0, time.UTC),
		LLMResponse: model.LLMResponse{
			Content:        genai.NewContentFromText("alpha beta", genai.RoleUser),
			CustomMetadata: map[string]any{"event_key": "event_val"},
		},
	}
	addEvents(t, svc, "s1", nil, ev)

	got := search(t, svc, "alpha")

	want := []memory.Entry{{
		ID:             "ev-1",
		Author:         "user",
		Timestamp:      time.Date(2025, 1, 2, 3, 4, 5, 0, time.UTC),
		Content:        genai.NewContentFromText("alpha beta", genai.RoleUser),
		CustomMetadata: map[string]any{"event_key": "event_val"},
	}}
	if diff := cmp.Diff(want, got.Memories); diff != "" {
		t.Errorf("SearchMemory() mismatch (-want +got):\n%s", diff)
	}
}

func TestAddEventsToMemory_CustomMetadataIgnored(t *testing.T) {
	svc := memory.InMemoryService()
	ev := &session.Event{
		ID: "ev-1",
		LLMResponse: model.LLMResponse{
			Content:        genai.NewContentFromText("alpha beta", genai.RoleUser),
			CustomMetadata: map[string]any{"event_key": "event_val"},
		},
	}
	if err := memory.AddEventsToMemory(t.Context(), svc, &memory.AddEventsRequest{
		AppName:        "app",
		UserID:         "user",
		SessionID:      "s1",
		Events:         []*session.Event{ev},
		CustomMetadata: map[string]any{"call_key": "ignored"},
	}); err != nil {
		t.Fatalf("AddEventsToMemory() error = %v", err)
	}

	got := search(t, svc, "alpha")

	if len(got.Memories) != 1 {
		t.Fatalf("SearchMemory() returned %d entries, want 1", len(got.Memories))
	}
	if diff := cmp.Diff(map[string]any{"event_key": "event_val"}, got.Memories[0].CustomMetadata); diff != "" {
		t.Errorf("Entry.CustomMetadata mismatch (-want +got):\n%s", diff)
	}
}

func TestAddEventsToMemory_DefaultBucketDoesNotCollideWithUnknownSessionID(t *testing.T) {
	svc := memory.InMemoryService()
	addEvents(t, svc, "", memoryTextEvent("shared-id", "default copy"))
	addEvents(t, svc, "__unknown_session_id__", memoryTextEvent("shared-id", "sentinel copy"))
	addSession(t, svc, "", memoryTextEvent("shared-id", "empty session copy"))

	got := search(t, svc, "copy")

	want := []string{"default copy", "sentinel copy", "empty session copy"}
	if diff := cmp.Diff(want, memoryTexts(got)); diff != "" {
		t.Errorf("SearchMemory() texts mismatch (-want +got):\n%s", diff)
	}
}

func TestAddEventsToMemory_DeduplicationScopeAndWordlessReservations(t *testing.T) {
	blobOnly := func() *session.Event {
		return &session.Event{ID: "dup", LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts(
			[]*genai.Part{genai.NewPartFromBytes([]byte("abc"), "text/plain")}, genai.RoleUser)}}
	}
	thoughtOnly := func() *session.Event {
		return &session.Event{ID: "dup", LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts(
			[]*genai.Part{{Text: "hidden", Thought: true}}, genai.RoleModel)}}
	}
	emptyContent := func() *session.Event {
		return &session.Event{ID: "dup", LLMResponse: model.LLMResponse{Content: &genai.Content{}}}
	}
	nilPart := func() *session.Event {
		return &session.Event{ID: "dup", LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{nil}}}}
	}
	viaSession := func(t *testing.T, svc memory.Service, ev *session.Event) { addSession(t, svc, "s1", ev) }
	viaEvents := func(t *testing.T, svc memory.Service, ev *session.Event) { addEvents(t, svc, "s1", ev) }

	for _, tt := range []struct {
		name  string
		setup func(t *testing.T, svc memory.Service)
		query string
		want  []string
	}{
		{
			name: "duplicates in one call",
			setup: func(t *testing.T, svc memory.Service) {
				addEvents(t, svc, "s1", memoryTextEvent("dup", "in-call first"), memoryTextEvent("dup", "in-call second"))
			},
			query: "call",
			want:  []string{"in-call first"},
		},
		{
			name: "duplicate of an added session event",
			setup: func(t *testing.T, svc memory.Service) {
				addSession(t, svc, "s1", memoryTextEvent("dup", "from session"))
				addEvents(t, svc, "s1", memoryTextEvent("dup", "from events"))
			},
			query: "from",
			want:  []string{"from session"},
		},
		{
			name: "same id in other bucket",
			setup: func(t *testing.T, svc memory.Service) {
				addEvents(t, svc, "s1", memoryTextEvent("dup", "s1 text"))
				addEvents(t, svc, "s2", memoryTextEvent("dup", "s2 text"))
			},
			query: "text",
			want:  []string{"s1 text", "s2 text"},
		},
		{
			name:  "blob-only session event reserves id",
			setup: reuseAfter(viaSession, blobOnly, "reused text"),
			query: "reused",
		},
		{
			name:  "blob-only added event reserves id",
			setup: reuseAfter(viaEvents, blobOnly, "reused text"),
			query: "reused",
		},
		{
			name:  "thought-only session event reserves id",
			setup: reuseAfter(viaSession, thoughtOnly, "reused text"),
			query: "reused",
		},
		{
			name:  "thought-only added event reserves id",
			setup: reuseAfter(viaEvents, thoughtOnly, "reused text"),
			query: "reused",
		},
		{
			name:  "session event without parts does not reserve id",
			setup: reuseAfter(viaSession, emptyContent, "reused after nil part"),
			query: "reused",
			want:  []string{"reused after nil part"},
		},
		{
			name:  "added event without parts does not reserve id",
			setup: reuseAfter(viaEvents, emptyContent, "reused after nil part"),
			query: "reused",
			want:  []string{"reused after nil part"},
		},
		{
			name:  "session event with only nil parts does not reserve id",
			setup: reuseAfter(viaSession, nilPart, "reused after nil part"),
			query: "reused",
			want:  []string{"reused after nil part"},
		},
		{
			name:  "added event with only nil parts does not reserve id",
			setup: reuseAfter(viaEvents, nilPart, "reused after nil part"),
			query: "reused",
			want:  []string{"reused after nil part"},
		},
		{
			name: "empty ids are not duplicates",
			setup: func(t *testing.T, svc memory.Service) {
				addEvents(t, svc, "s1", memoryTextEvent("", "empty id 1"), memoryTextEvent("", "empty id 2"))
			},
			query: "empty",
			want:  []string{"empty id 1", "empty id 2"},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			svc := memory.InMemoryService()
			tt.setup(t, svc)

			got := search(t, svc, tt.query)

			if diff := cmp.Diff(tt.want, memoryTexts(got)); diff != "" {
				t.Errorf("SearchMemory(%q) texts mismatch (-want +got):\n%s", tt.query, diff)
			}
		})
	}
}

// reuseAfter returns a setup that stores first() in bucket "s1" with store,
// then adds an event to "s1" with the same ID "dup" and the given text.
func reuseAfter(store func(*testing.T, memory.Service, *session.Event), first func() *session.Event, text string) func(*testing.T, memory.Service) {
	return func(t *testing.T, svc memory.Service) {
		store(t, svc, first())
		addEvents(t, svc, "s1", memoryTextEvent("dup", text))
	}
}

func TestAddEventsToMemory_AppendReplaceAndEmptyBucketOrder(t *testing.T) {
	t.Run("append after session then replace", func(t *testing.T) {
		svc := memory.InMemoryService()
		addSession(t, svc, "s1", memoryTextEvent("1", "note one"))
		addEvents(t, svc, "s1", memoryTextEvent("2", "note two"))

		if diff := cmp.Diff([]string{"1", "2"}, memoryIDs(search(t, svc, "note"))); diff != "" {
			t.Errorf("SearchMemory() after append IDs mismatch (-want +got):\n%s", diff)
		}

		addSession(t, svc, "s1", memoryTextEvent("dup", "note a"), memoryTextEvent("dup", "note b"))

		if diff := cmp.Diff([]string{"note a", "note b"}, memoryTexts(search(t, svc, "note"))); diff != "" {
			t.Errorf("SearchMemory() after replace texts mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("zero-event call fixes named bucket position", func(t *testing.T) {
		svc := memory.InMemoryService()
		addEvents(t, svc, "early")
		addEvents(t, svc, "", memoryTextEvent("def", "tie"))
		addSession(t, svc, "late", memoryTextEvent("late", "tie"))
		addEvents(t, svc, "early", memoryTextEvent("early", "tie"))

		if diff := cmp.Diff([]string{"early", "def", "late"}, memoryIDs(search(t, svc, "tie"))); diff != "" {
			t.Errorf("SearchMemory() IDs mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("zero-event call fixes default bucket position", func(t *testing.T) {
		svc := memory.InMemoryService()
		addEvents(t, svc, "")
		addSession(t, svc, "late", memoryTextEvent("late", "tie"))
		addEvents(t, svc, "", memoryTextEvent("def", "tie"))

		if diff := cmp.Diff([]string{"def", "late"}, memoryIDs(search(t, svc, "tie"))); diff != "" {
			t.Errorf("SearchMemory() IDs mismatch (-want +got):\n%s", diff)
		}
	})
}

func Test_inMemoryService_SearchMemory_ConcurrentAddEvents(t *testing.T) {
	svc := memory.InMemoryService()
	ctx := t.Context()

	// Each worker adds 5 events to its own session and 5 to the shared default
	// bucket, all carrying "x" and the worker's own word "w<i>". Together the
	// workers put 20 events in the default bucket.
	const workers = 4
	var wg sync.WaitGroup
	for i := range workers {
		wg.Add(3)
		go func() {
			defer wg.Done()
			for j := range 5 {
				for _, sid := range []string{"s" + strconv.Itoa(i), ""} {
					id := sid + "-" + strconv.Itoa(i) + "-" + strconv.Itoa(j)
					if err := memory.AddEventsToMemory(ctx, svc, &memory.AddEventsRequest{
						AppName: "app", UserID: "user", SessionID: sid,
						Events: []*session.Event{memoryTextEvent(id, "x w"+strconv.Itoa(i))},
					}); err != nil {
						t.Errorf("AddEventsToMemory() error = %v", err)
						return
					}
				}
			}
		}()
		go func() {
			defer wg.Done()
			for range 25 {
				id := "sess" + strconv.Itoa(i)
				if err := svc.AddSessionToMemory(ctx, makeSession(t, "app", "user", id, []*session.Event{memoryTextEvent(id, "x")})); err != nil {
					t.Errorf("AddSessionToMemory() error = %v", err)
					return
				}
			}
		}()
		go func() {
			defer wg.Done()
			for range 25 {
				if _, err := svc.SearchMemory(ctx, &memory.SearchRequest{AppName: "app", UserID: "user", Query: "x"}); err != nil {
					t.Errorf("SearchMemory() error = %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()

	// The sessions alone hold 4 matching events; the rest come from the
	// concurrent AddEventsToMemory calls.
	if got := search(t, svc, "x"); len(got.Memories) != 10 {
		t.Errorf("SearchMemory() returned %d entries, want 10", len(got.Memories))
	}
	for _, tt := range []struct {
		query string
		want  []string
	}{
		{query: "w0", want: []string{"-0-0", "-0-1", "-0-2", "-0-3", "-0-4", "s0-0-0", "s0-0-1", "s0-0-2", "s0-0-3", "s0-0-4"}},
		{query: "w1", want: []string{"-1-0", "-1-1", "-1-2", "-1-3", "-1-4", "s1-1-0", "s1-1-1", "s1-1-2", "s1-1-3", "s1-1-4"}},
		{query: "w2", want: []string{"-2-0", "-2-1", "-2-2", "-2-3", "-2-4", "s2-2-0", "s2-2-1", "s2-2-2", "s2-2-3", "s2-2-4"}},
		{query: "w3", want: []string{"-3-0", "-3-1", "-3-2", "-3-3", "-3-4", "s3-3-0", "s3-3-1", "s3-3-2", "s3-3-3", "s3-3-4"}},
	} {
		got := memoryIDs(search(t, svc, tt.query))
		slices.Sort(got)
		if diff := cmp.Diff(tt.want, got); diff != "" {
			t.Errorf("SearchMemory(%q) IDs mismatch (-want +got):\n%s", tt.query, diff)
		}
	}
}

// eventlessService implements memory.Service but not memory.EventAdder.
type eventlessService struct{}

func (eventlessService) AddSessionToMemory(context.Context, session.Session) error { return nil }

func (eventlessService) SearchMemory(context.Context, *memory.SearchRequest) (*memory.SearchResponse, error) {
	return &memory.SearchResponse{}, nil
}

// acceptingEventAdder implements memory.EventAdder and accepts any request,
// nil included.
type acceptingEventAdder struct{ eventlessService }

func (acceptingEventAdder) AddEventsToMemory(context.Context, *memory.AddEventsRequest) error {
	return nil
}

func TestAddEventsToMemory_UnsupportedService(t *testing.T) {
	err := memory.AddEventsToMemory(t.Context(), eventlessService{}, &memory.AddEventsRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "s1",
		Events:    []*session.Event{memoryTextEvent("e1", "hello")},
	})

	if !errors.Is(err, memory.ErrAddEventsUnsupported) {
		t.Errorf("AddEventsToMemory() error = %v, want errors.Is(err, memory.ErrAddEventsUnsupported)", err)
	}
	if !errors.Is(err, errors.ErrUnsupported) {
		t.Errorf("AddEventsToMemory() error = %v, want errors.Is(err, errors.ErrUnsupported)", err)
	}
}

// The unsupported error must point a Go caller at the Go fallback method, in
// the package's error style, and not at adk-python's add_session_to_memory.
func TestAddEventsToMemory_UnsupportedServiceNamesGoFallback(t *testing.T) {
	err := memory.AddEventsToMemory(t.Context(), eventlessService{}, &memory.AddEventsRequest{
		AppName: "app",
		UserID:  "user",
		Events:  []*session.Event{memoryTextEvent("e1", "hello")},
	})
	if err == nil {
		t.Fatal("AddEventsToMemory() error = nil, want an unsupported error")
	}

	msg := err.Error()
	if !strings.HasPrefix(msg, "memory: ") {
		t.Errorf("AddEventsToMemory() error = %q, want prefix %q", msg, "memory: ")
	}
	if !strings.Contains(msg, "AddSessionToMemory") {
		t.Errorf("AddEventsToMemory() error = %q, want it to name %q", msg, "AddSessionToMemory")
	}
	if strings.Contains(msg, "add_session_to_memory") {
		t.Errorf("AddEventsToMemory() error = %q, want no Python name %q", msg, "add_session_to_memory")
	}
	if strings.HasSuffix(msg, ".") {
		t.Errorf("AddEventsToMemory() error = %q, want no trailing full stop", msg)
	}
}

func TestAddEventsToMemory_NilRequests(t *testing.T) {
	for _, tt := range []struct {
		name    string
		call    func(ctx context.Context) error
		wantErr string
	}{
		{
			name: "AddEventsToMemory nil service",
			call: func(ctx context.Context) error {
				return memory.AddEventsToMemory(ctx, nil, &memory.AddEventsRequest{})
			},
			wantErr: "memory: service must not be nil",
		},
		{
			name: "AddEventsToMemory nil request",
			call: func(ctx context.Context) error {
				return memory.AddEventsToMemory(ctx, memory.InMemoryService(), nil)
			},
			wantErr: "memory: AddEventsRequest must not be nil",
		},
		{
			name: "AddEventsToMemory nil request to an EventAdder that accepts it",
			call: func(ctx context.Context) error {
				return memory.AddEventsToMemory(ctx, acceptingEventAdder{}, nil)
			},
			wantErr: "memory: AddEventsRequest must not be nil",
		},
		{
			name: "EventAdder nil request",
			call: func(ctx context.Context) error {
				return memory.InMemoryService().(memory.EventAdder).AddEventsToMemory(ctx, nil)
			},
			wantErr: "memory: AddEventsRequest must not be nil",
		},
		{
			name: "AddSessionToMemory nil session",
			call: func(ctx context.Context) error {
				return memory.InMemoryService().AddSessionToMemory(ctx, nil)
			},
			wantErr: "memory: session must not be nil",
		},
		{
			name: "SearchMemory nil request",
			call: func(ctx context.Context) error {
				resp, err := memory.InMemoryService().SearchMemory(ctx, nil)
				if resp != nil {
					t.Errorf("SearchMemory(nil) response = %v, want nil", resp)
				}
				return err
			},
			wantErr: "memory: SearchRequest must not be nil",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.call(t.Context())
			if err == nil || err.Error() != tt.wantErr {
				t.Errorf("error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

// addEvents adds events to the bucket sessionID of ("app", "user").
func addEvents(t *testing.T, svc memory.Service, sessionID string, events ...*session.Event) {
	t.Helper()
	if err := memory.AddEventsToMemory(t.Context(), svc, &memory.AddEventsRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: sessionID,
		Events:    events,
	}); err != nil {
		t.Fatalf("AddEventsToMemory(%q) error = %v", sessionID, err)
	}
}

// addSession adds a session of ("app", "user") holding events.
func addSession(t *testing.T, svc memory.Service, sessionID string, events ...*session.Event) {
	t.Helper()
	if err := svc.AddSessionToMemory(t.Context(), makeSession(t, "app", "user", sessionID, events)); err != nil {
		t.Fatalf("AddSessionToMemory(%q) error = %v", sessionID, err)
	}
}

// search searches the memories of ("app", "user").
func search(t *testing.T, svc memory.Service, query string) *memory.SearchResponse {
	t.Helper()
	got, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app", UserID: "user", Query: query})
	if err != nil {
		t.Fatalf("SearchMemory(%q) error = %v", query, err)
	}
	return got
}

func memoryIDs(resp *memory.SearchResponse) []string {
	var ids []string
	for _, e := range resp.Memories {
		ids = append(ids, e.ID)
	}
	return ids
}

// memoryTexts returns the text of the last part of each memory.
func memoryTexts(resp *memory.SearchResponse) []string {
	var texts []string
	for _, e := range resp.Memories {
		texts = append(texts, e.Content.Parts[len(e.Content.Parts)-1].Text)
	}
	return texts
}
