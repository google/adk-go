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
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	imemory "google.golang.org/adk/internal/memory"
	"google.golang.org/adk/internal/sessioninternal"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func TestMemory_AddAndSearch(t *testing.T) {
	appName, userID, sessionID := "testApp", "testUser", "sess1"
	memoryService := imemory.Memory{
		Service:   memory.InMemoryService(),
		UserID:    userID,
		AppName:   appName,
		SessionID: sessionID,
	}

	content1 := genai.NewContentFromText("The quick brown fox", genai.RoleUser)
	content2 := genai.NewContentFromText("jumps over the lazy dog", genai.RoleUser)

	events := []*session.Event{
		{
			Timestamp: time.Date(2025, 1, 1, 10, 0, 0, 0, time.UTC),
			Author:    "user1",
			LLMResponse: model.LLMResponse{
				Content: content1,
			},
		},
		{
			Timestamp: time.Date(2025, 1, 1, 10, 5, 0, 0, time.UTC),
			Author:    "user1",
			LLMResponse: model.LLMResponse{
				Content: content2,
			},
		},
	}
	sessionService := session.InMemoryService()
	createResponse, err := sessionService.Create(t.Context(), &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	session := createResponse.Session
	for _, event := range events {
		if err := sessionService.AppendEvent(t.Context(), session, event); err != nil {
			t.Fatalf("Failed to append event: %v", err)
		}
	}

	if err := memoryService.AddSession(t.Context(), sessioninternal.NewMutableSession(sessionService, session)); err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	// Expected MemoryEntry items
	entry1 := memory.Entry{
		Content:   content1,
		Author:    "user1",
		Timestamp: time.Date(2025, 1, 1, 10, 0, 0, 0, time.UTC),
	}
	entry2 := memory.Entry{
		Content:   content2,
		Author:    "user1",
		Timestamp: time.Date(2025, 1, 1, 10, 5, 0, 0, time.UTC),
	}

	tests := []struct {
		name  string
		query string
		want  *memory.SearchResponse
	}{
		{
			name:  "match first entry",
			query: "fox",
			want: &memory.SearchResponse{
				Memories: []memory.Entry{entry1},
			},
		},
		{
			name:  "match second entry",
			query: "DOG", // Search should be case-insensitive
			want: &memory.SearchResponse{
				Memories: []memory.Entry{entry2},
			},
		},
		{
			name:  "match both entries (any word)",
			query: "quick dog",
			want: &memory.SearchResponse{
				Memories: []memory.Entry{entry1, entry2},
			},
		},
		{
			name:  "match word in both",
			query: "the",
			want: &memory.SearchResponse{
				Memories: []memory.Entry{entry1, entry2},
			},
		},
		{
			name:  "no match",
			query: "unrelated",
			want:  &memory.SearchResponse{},
		},
		{
			name:  "empty query",
			query: "",
			want:  &memory.SearchResponse{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := memoryService.Search(t.Context(), tc.query)
			if err != nil {
				t.Fatalf("Search(%q) failed: %v", tc.query, err)
			}

			if diff := cmp.Diff(tc.want, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Search(%q) returned diff (-want +got):\n%s", tc.query, diff)
			}
		})
	}
}

func TestMemory_Search_NoData(t *testing.T) {
	memory := imemory.Memory{
		Service:   memory.InMemoryService(),
		UserID:    "testUser",
		AppName:   "testApp",
		SessionID: "sess2",
	}

	got, err := memory.Search(t.Context(), "any query")
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}
	if len(got.Memories) != 0 {
		t.Errorf("Search() on empty memory returned %d items, want 0", len(got.Memories))
	}
}

func TestMemory_Search_Isolation(t *testing.T) {
	memService := memory.InMemoryService()
	appName := "testApp"

	userID1, sessionID1 := "user1", "sess1"

	memory1 := imemory.Memory{
		Service:   memService,
		UserID:    userID1,
		AppName:   appName,
		SessionID: sessionID1,
	}
	content1 := genai.NewContentFromText("Content for user1", genai.RoleUser)
	sessionService := session.InMemoryService()
	createResponse, err := sessionService.Create(t.Context(), &session.CreateRequest{
		AppName:   appName,
		UserID:    userID1,
		SessionID: sessionID1,
	})
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	storedSession := createResponse.Session
	if err := sessionService.AppendEvent(t.Context(), storedSession, &session.Event{
		Timestamp:   time.Now(),
		Author:      "user1",
		LLMResponse: model.LLMResponse{Content: content1},
	}); err != nil {
		t.Fatalf("Failed to append event: %v", err)
	}

	if err := memory1.AddSession(t.Context(), sessioninternal.NewMutableSession(sessionService, storedSession)); err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	// Add data for User2
	userID2, sessionID2 := "user2", "sess2"
	memory2 := imemory.Memory{
		Service:   memService,
		UserID:    userID2,
		AppName:   "testApp",
		SessionID: sessionID2,
	}
	content2 := genai.NewContentFromText("Content for user2", genai.RoleUser)
	createResponse2, err := sessionService.Create(t.Context(), &session.CreateRequest{
		AppName:   appName,
		UserID:    userID2,
		SessionID: sessionID2,
	})
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	storedSession2 := createResponse2.Session
	if err := sessionService.AppendEvent(t.Context(), storedSession2, &session.Event{
		Timestamp:   time.Now(),
		Author:      "user2",
		LLMResponse: model.LLMResponse{Content: content2},
	}); err != nil {
		t.Fatalf("Failed to append event: %v", err)
	}

	if err := memory2.AddSession(t.Context(), sessioninternal.NewMutableSession(sessionService, storedSession2)); err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	// User1 search should only find user1's content
	got1, err := memory1.Search(t.Context(), "Content")
	if err != nil {
		t.Fatalf("memory1.Search failed: %v", err)
	}
	if len(got1.Memories) != 1 {
		t.Errorf("memory1.Search returned %d items, want 1", len(got1.Memories))
	} else if diff := cmp.Diff(content1, got1.Memories[0].Content); diff != "" {
		t.Errorf("memory1.Search returned diff (-want +got):\n%s", diff)
	}

	// User2 search should only find user2's content
	got2, err := memory2.Search(t.Context(), "Content")
	if err != nil {
		t.Fatalf("memory2.Search failed: %v", err)
	}
	if len(got2.Memories) != 1 {
		t.Errorf("memory2.Search returned %d items, want 1", len(got2.Memories))
	} else if diff := cmp.Diff(content2, got2.Memories[0].Content); diff != "" {
		t.Errorf("memory2.Search returned diff (-want +got):\n%s", diff)
	}
}
