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

package memoryinternal_test

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"google.golang.org/adk/internal/memoryinternal"
	"google.golang.org/adk/internal/sessioninternal"
	"google.golang.org/adk/memoryservice"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
	"google.golang.org/genai"
)

func TestMemory_AddAndSearch(t *testing.T) {
	memService := memoryservice.Mem()
	testID := session.ID{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "sess1",
	}
	memory := memoryinternal.NewMemory(memService, testID)

	content1 := genai.NewContentFromText("The quick brown fox", genai.RoleUser)
	content2 := genai.NewContentFromText("jumps over the lazy dog", genai.RoleUser)

	events := []*session.Event{
		{
			Time:   time.Date(2025, 1, 1, 10, 0, 0, 0, time.UTC),
			Author: "user1",
			LLMResponse: &model.LLMResponse{
				Content: content1,
			},
		},
		{
			Time:   time.Date(2025, 1, 1, 10, 5, 0, 0, time.UTC),
			Author: "user1",
			LLMResponse: &model.LLMResponse{
				Content: content2,
			},
		},
	}
	sessionService := sessionservice.Mem()
	createResponse, err := sessionService.Create(t.Context(), &sessionservice.CreateRequest{
		AppName:   testID.AppName,
		UserID:    testID.UserID,
		SessionID: testID.SessionID,
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

	if err := memory.AddSession(sessioninternal.NewMutableSession(sessionService, session)); err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	// Expected MemoryEntry items
	entry1 := memoryservice.MemoryEntry{
		Content:   content1,
		Author:    "user1",
		Timestamp: time.Date(2025, 1, 1, 10, 0, 0, 0, time.UTC),
	}
	entry2 := memoryservice.MemoryEntry{
		Content:   content2,
		Author:    "user1",
		Timestamp: time.Date(2025, 1, 1, 10, 5, 0, 0, time.UTC),
	}

	tests := []struct {
		name  string
		query string
		want  []memoryservice.MemoryEntry
	}{
		{
			name:  "match first entry",
			query: "fox",
			want:  []memoryservice.MemoryEntry{entry1},
		},
		{
			name:  "match second entry",
			query: "DOG", // Search should be case-insensitive
			want:  []memoryservice.MemoryEntry{entry2},
		},
		{
			name:  "match both entries (any word)",
			query: "quick dog",
			want:  []memoryservice.MemoryEntry{entry1, entry2},
		},
		{
			name:  "match word in both",
			query: "the",
			want:  []memoryservice.MemoryEntry{entry1, entry2},
		},
		{
			name:  "no match",
			query: "unrelated",
			want:  nil,
		},
		{
			name:  "empty query",
			query: "",
			want:  nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := memory.Search(tc.query)
			if err != nil {
				t.Fatalf("Search(%q) failed: %v", tc.query, err)
			}

			sortOpt := cmpopts.SortSlices(func(a, b memoryservice.MemoryEntry) bool {
				return a.Timestamp.Before(b.Timestamp)
			})

			if diff := cmp.Diff(tc.want, got, sortOpt, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Search(%q) returned diff (-want +got):\n%s", tc.query, diff)
			}
		})
	}
}

func TestMemory_Search_NoData(t *testing.T) {
	memService := memoryservice.Mem()
	testID := session.ID{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "sess2",
	}
	memory := memoryinternal.NewMemory(memService, testID)

	got, err := memory.Search("any query")
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("Search() on empty memory returned %d items, want 0", len(got))
	}
}

func TestMemory_Search_Isolation(t *testing.T) {
	memService := memoryservice.Mem()

	// Add data for User1
	testID1 := session.ID{
		AppName:   "testApp",
		UserID:    "user1",
		SessionID: "sess1",
	}
	memory1 := memoryinternal.NewMemory(memService, testID1)
	content1 := genai.NewContentFromText("Content for user1", genai.RoleUser)
	sessionService := sessionservice.Mem()
	createResponse, err := sessionService.Create(t.Context(), &sessionservice.CreateRequest{
		AppName:   testID1.AppName,
		UserID:    testID1.UserID,
		SessionID: testID1.SessionID,
	})
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	storedSession := createResponse.Session
	if err := sessionService.AppendEvent(t.Context(), storedSession, &session.Event{
		Time:        time.Now(),
		Author:      "user1",
		LLMResponse: &model.LLMResponse{Content: content1},
	}); err != nil {
		t.Fatalf("Failed to append event: %v", err)
	}

	if err := memory1.AddSession(sessioninternal.NewMutableSession(sessionService, storedSession)); err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	// Add data for User2
	testID2 := session.ID{
		AppName:   "testApp",
		UserID:    "user2",
		SessionID: "sess2",
	}
	memory2 := memoryinternal.NewMemory(memService, testID2)
	content2 := genai.NewContentFromText("Content for user2", genai.RoleUser)
	createResponse2, err := sessionService.Create(t.Context(), &sessionservice.CreateRequest{
		AppName:   testID2.AppName,
		UserID:    testID2.UserID,
		SessionID: testID2.SessionID,
	})
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	storedSession2 := createResponse2.Session
	if err := sessionService.AppendEvent(t.Context(), storedSession2, &session.Event{
		Time:        time.Now(),
		Author:      "user2",
		LLMResponse: &model.LLMResponse{Content: content2},
	}); err != nil {
		t.Fatalf("Failed to append event: %v", err)
	}

	if err := memory2.AddSession(sessioninternal.NewMutableSession(sessionService, storedSession2)); err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	// User1 search should only find user1's content
	got1, err := memory1.Search("Content")
	if err != nil {
		t.Fatalf("memory1.Search failed: %v", err)
	}
	if len(got1) != 1 {
		t.Errorf("memory1.Search returned %d items, want 1", len(got1))
	} else if diff := cmp.Diff(content1, got1[0].Content); diff != "" {
		t.Errorf("memory1.Search returned diff (-want +got):\n%s", diff)
	}

	// User2 search should only find user2's content
	got2, err := memory2.Search("Content")
	if err != nil {
		t.Fatalf("memory2.Search failed: %v", err)
	}
	if len(got2) != 1 {
		t.Errorf("memory2.Search returned %d items, want 1", len(got2))
	} else if diff := cmp.Diff(content2, got2[0].Content); diff != "" {
		t.Errorf("memory2.Search returned diff (-want +got):\n%s", diff)
	}
}
