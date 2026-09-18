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

package memory_test

import (
	"strconv"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func Test_inMemoryService_SearchMemory_Ranking(t *testing.T) {
	for _, tt := range []struct {
		name  string
		texts []string
		query string
		want  []string
	}{
		{
			name:  "most query words first",
			texts: []string{"deploy ready", "ready", "deploy status ready", "unrelated"},
			query: "deploy status ready",
			want:  []string{"2", "0", "1"},
		},
		{
			name:  "repeated words count once",
			texts: []string{"ready ready ready", "deploy status", "ready deploy"},
			query: "READY ready ready deploy status",
			want:  []string{"1", "2", "0"},
		},
		{
			name:  "token and substring matches count once",
			texts: []string{"caf\u00e9", "deploy ready", "CAF\u00c9 deploy"},
			query: "caf\u00e9 deploy ready",
			want:  []string{"1", "2", "0"},
		},
		{
			name:  "combine token and non ASCII substring scores",
			texts: []string{"deploy", "\u6211\u559c\u6b22\u673a\u5668\u5b66\u4e60 deploy", "\u6211\u559c\u6b22\u673a\u5668\u5b66\u4e60"},
			query: "deploy \u673a\u5668\u5b66\u4e60",
			want:  []string{"1", "0", "2"},
		},
		{
			name:  "empty query",
			texts: []string{"ready"},
		},
		{
			name:  "no matching words",
			texts: []string{"ready"},
			query: "missing",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := memory.InMemoryService()
			var events []*session.Event
			for i, text := range tt.texts {
				events = append(events, memoryTextEvent(strconv.Itoa(i), text))
			}
			if err := s.AddSessionToMemory(t.Context(), makeSession(t, "app", "user", "session", events)); err != nil {
				t.Fatal(err)
			}
			got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app", UserID: "user", Query: tt.query})
			if err != nil {
				t.Fatal(err)
			}
			var ids []string
			for _, entry := range got.Memories {
				ids = append(ids, entry.ID)
			}
			if diff := cmp.Diff(tt.want, ids); diff != "" {
				t.Errorf("SearchMemory() order mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func Test_inMemoryService_SearchMemory_LimitAndSessionOrder(t *testing.T) {
	s := memory.InMemoryService()
	// Session IDs and timestamps run backwards so neither can define tie order.
	for i := 0; i < 12; i++ {
		var events []*session.Event
		for j := 0; j < 2; j++ {
			e := memoryTextEvent(strconv.Itoa(2*i+j), "note about work")
			e.Timestamp = time.Unix(int64(100-2*i-j), 0)
			events = append(events, e)
		}
		if err := s.AddSessionToMemory(t.Context(), makeSession(t, "app", "user", strconv.Itoa(12-i), events)); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.AddSessionToMemory(t.Context(), makeSession(t, "app", "user", "best", []*session.Event{
		memoryTextEvent("best", "backlog note about work"),
	})); err != nil {
		t.Fatal(err)
	}

	want := []string{"best", "0", "1", "2", "3", "4", "5", "6", "7", "8"}
	for i := 0; i < 20; i++ {
		got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app", UserID: "user", Query: "work backlog note"})
		if err != nil {
			t.Fatal(err)
		}
		var ids []string
		for _, entry := range got.Memories {
			ids = append(ids, entry.ID)
		}
		if diff := cmp.Diff(want, ids); diff != "" {
			t.Fatalf("SearchMemory() iteration %d mismatch (-want +got):\n%s", i, diff)
		}
	}
}

func Test_inMemoryService_SearchMemory_ReplacedSessionOrder(t *testing.T) {
	s := memory.InMemoryService()
	for _, sess := range []session.Session{
		makeSession(t, "app", "user", "z", []*session.Event{memoryTextEvent("old", "note")}),
		makeSession(t, "app", "user", "empty", nil),
		makeSession(t, "app", "user", "a", []*session.Event{memoryTextEvent("last", "note")}),
		makeSession(t, "app", "user", "z", []*session.Event{memoryTextEvent("replacement", "note")}),
		makeSession(t, "app", "user", "empty", []*session.Event{memoryTextEvent("middle", "note")}),
	} {
		if err := s.AddSessionToMemory(t.Context(), sess); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 20; i++ {
		got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app", UserID: "user", Query: "note"})
		if err != nil {
			t.Fatal(err)
		}
		var ids []string
		for _, entry := range got.Memories {
			ids = append(ids, entry.ID)
		}
		if diff := cmp.Diff([]string{"replacement", "middle", "last"}, ids); diff != "" {
			t.Fatalf("SearchMemory() iteration %d mismatch (-want +got):\n%s", i, diff)
		}
	}
}

func Test_inMemoryService_SearchMemory_NonASCII(t *testing.T) {
	for _, tt := range []struct {
		name  string
		parts []string
		query string
		want  int
	}{
		{name: "Chinese substring", parts: []string{"\u6211\u559c\u6b22\u673a\u5668\u5b66\u4e60"}, query: "\u673a\u5668\u5b66\u4e60", want: 1},
		{name: "Chinese no match", parts: []string{"\u6211\u559c\u6b22\u673a\u5668\u5b66\u4e60"}, query: "\u5929\u6c14\u9884\u62a5"},
		{name: "Japanese substring", parts: []string{"\u79c1\u306e\u540d\u524d\u306f\u592a\u90ce\u3067\u3059"}, query: "\u592a\u90ce", want: 1},
		{name: "non ASCII case folding", parts: []string{"CAF\u00c9TERIA"}, query: "caf\u00e9", want: 1},
		{name: "substring in later part", parts: []string{"hello", "\u6211\u559c\u6b22\u673a\u5668\u5b66\u4e60"}, query: "\u673a\u5668\u5b66\u4e60", want: 1},
		{name: "do not join words across parts", parts: []string{"\u673a\u5668", "\u5b66\u4e60"}, query: "\u673a\u5668\u5b66\u4e60"},
		{name: "ASCII partial word does not match", parts: []string{"Python"}, query: "thon"},
		{name: "ASCII token matches", parts: []string{"Python"}, query: "python", want: 1},
		{name: "empty text", parts: []string{""}, query: "\u673a\u5668"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := memory.InMemoryService()
			e := memoryTextEvent("event", tt.parts...)
			if err := s.AddSessionToMemory(t.Context(), makeSession(t, "app", "user", "session", []*session.Event{e})); err != nil {
				t.Fatal(err)
			}
			got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app", UserID: "user", Query: tt.query})
			if err != nil {
				t.Fatal(err)
			}
			if len(got.Memories) != tt.want {
				t.Fatalf("SearchMemory() returned %d entries, want %d", len(got.Memories), tt.want)
			}
		})
	}
}

func memoryTextEvent(id string, texts ...string) *session.Event {
	var parts []*genai.Part
	for _, text := range texts {
		parts = append(parts, genai.NewPartFromText(text))
	}
	return &session.Event{
		ID: id,
		LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromParts(parts, genai.RoleUser),
		},
	}
}
