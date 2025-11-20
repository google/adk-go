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
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
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
						Author: "user1",
						LLMResponse: model.LLMResponse{
							Content: genai.NewContentFromText("The Quick brown fox", genai.RoleUser),
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
