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

package memoryservice_test

import (
	"iter"
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/memoryservice"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func Test_inMemoryService_SearchMemory(t *testing.T) {
	tests := []struct {
		name         string
		initSessions []session.Session
		req          *memoryservice.SearchRequest
		wantResp     *memoryservice.SearchResponse
		wantErr      bool
	}{
		{
			name: "find events",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess1", []*session.Event{
					{
						Author: "user1",
						LLMResponse: &llm.Response{
							Content: genai.NewContentFromText("The Quick brown fox", genai.RoleUser),
						},
						Time: must(time.Parse(time.RFC3339, "2023-10-01T10:00:00Z")),
					},
					{
						LLMResponse: &llm.Response{
							Content: genai.NewContentFromText("jumps over the lazy dog", genai.RoleModel),
						},
					},
				}),
				makeSession(t, "app1", "user1", "sess2", []*session.Event{
					{
						Author:      "test-bot",
						LLMResponse: &llm.Response{Content: genai.NewContentFromText("hello world", genai.RoleModel)},
						Time:        must(time.Parse(time.RFC3339, "2023-10-02T10:00:00Z")),
					},
				}),
				makeSession(t, "app1", "user1", "sess3", []*session.Event{
					{LLMResponse: &llm.Response{Content: genai.NewContentFromText("test text", genai.RoleUser)}},
				}),
			},
			req: &memoryservice.SearchRequest{
				AppName: "app1",
				UserID:  "user1",
				Query:   "quick hello",
			},
			wantResp: &memoryservice.SearchResponse{
				Memories: []memoryservice.MemoryEntry{
					{
						Content:   genai.NewContentFromText("The Quick brown fox", genai.RoleUser),
						Author:    "user1",
						Timestamp: must(time.Parse(time.RFC3339, "2023-10-01T10:00:00Z")),
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
					{LLMResponse: &llm.Response{Content: genai.NewContentFromText("test text", genai.RoleUser)}},
				}),
			},
			req: &memoryservice.SearchRequest{
				AppName: "other_app",
				UserID:  "user1",
				Query:   "test text",
			},
			wantResp: &memoryservice.SearchResponse{},
		},
		{
			name: "no leakage for different user",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess3", []*session.Event{
					{LLMResponse: &llm.Response{Content: genai.NewContentFromText("test text", genai.RoleUser)}},
				}),
			},
			req: &memoryservice.SearchRequest{
				AppName: "app1",
				UserID:  "test_user",
				Query:   "test text",
			},
			wantResp: &memoryservice.SearchResponse{},
		},
		{
			name: "no matches",
			initSessions: []session.Session{
				makeSession(t, "app1", "user1", "sess3", []*session.Event{
					{LLMResponse: &llm.Response{Content: genai.NewContentFromText("test text", genai.RoleUser)}},
				}),
			},
			req: &memoryservice.SearchRequest{
				AppName: "app1",
				UserID:  "test_user",
				Query:   "something different",
			},
			wantResp: &memoryservice.SearchResponse{},
		},
		{
			name: "lookup on empty store",
			req: &memoryservice.SearchRequest{
				AppName: "app1",
				UserID:  "test_user",
				Query:   "something different",
			},
			wantResp: &memoryservice.SearchResponse{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := memoryservice.Mem()

			for _, session := range tt.initSessions {
				if err := s.AddSession(t.Context(), session); err != nil {
					t.Fatalf("inMemoryService.AddSession() error = %v", err)
				}
			}

			got, err := s.Search(t.Context(), tt.req)
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

var sortMemories = cmp.Transformer("Sort", func(in *memoryservice.SearchResponse) *memoryservice.SearchResponse {
	slices.SortFunc(in.Memories, func(m1, m2 memoryservice.MemoryEntry) int {
		return m1.Timestamp.Compare(m2.Timestamp)
	})
	return in
})

type testSession struct {
	appName, userID, sessionID string
	events                     []*session.Event
}

func (s *testSession) ID() session.ID {
	return session.ID{
		AppName:   s.appName,
		UserID:    s.userID,
		SessionID: s.sessionID,
	}
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

func (s *testSession) Updated() time.Time {
	panic("not implemented")
}

func must[V any](v V, err error) V {
	if err != nil {
		panic(err)
	}
	return v
}
