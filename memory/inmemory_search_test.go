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
	"golang.org/x/text/unicode/norm"
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

func Test_inMemoryService_SearchMemory_SkipsThoughtParts(t *testing.T) {
	thought := &genai.Part{Text: "secret plan", Thought: true}
	visible := genai.NewPartFromText("deploy ready")
	s := memory.InMemoryService()
	if err := s.AddSessionToMemory(t.Context(), makeSession(t, "app", "user", "session", []*session.Event{
		{ID: "thought-only", LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{thought}, genai.RoleModel)}},
		{ID: "mixed", LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{thought, visible}, genai.RoleModel)}},
	})); err != nil {
		t.Fatal(err)
	}

	got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app", UserID: "user", Query: "secret plan"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Memories) != 0 {
		t.Errorf("SearchMemory(%q) matched thought text, got %d entries", "secret plan", len(got.Memories))
	}

	got, err = s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app", UserID: "user", Query: "deploy"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Memories) != 1 {
		t.Fatalf("SearchMemory(%q) returned %d entries, want 1", "deploy", len(got.Memories))
	}
	want := genai.NewContentFromParts([]*genai.Part{visible}, genai.RoleModel)
	if diff := cmp.Diff(want, got.Memories[0].Content); diff != "" {
		t.Errorf("SearchMemory() content mismatch (-want +got):\n%s", diff)
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

func Test_inMemoryService_SearchMemory_PreservesBlobPartsAndSkipsThoughtsInAddedEvents(t *testing.T) {
	svc := memory.InMemoryService()
	addEvents(t, svc, "s1", &session.Event{
		ID: "mixed",
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{
			genai.NewPartFromBytes([]byte("abc"), "text/plain"),
			{Text: "secretplan", Thought: true},
			genai.NewPartFromText("hello world"),
		}, genai.RoleModel)},
	})

	if got := search(t, svc, "secretplan"); len(got.Memories) != 0 {
		t.Errorf("SearchMemory(%q) matched thought text, got %d entries", "secretplan", len(got.Memories))
	}

	got := search(t, svc, "hello")
	if len(got.Memories) != 1 {
		t.Fatalf("SearchMemory(%q) returned %d entries, want 1", "hello", len(got.Memories))
	}
	want := []*genai.Part{
		genai.NewPartFromBytes([]byte("abc"), "text/plain"),
		genai.NewPartFromText("hello world"),
	}
	if diff := cmp.Diff(want, got.Memories[0].Content.Parts); diff != "" {
		t.Errorf("SearchMemory() parts mismatch (-want +got):\n%s", diff)
	}
}

func Test_inMemoryService_SearchMemory_TokenizationNFCAndScriptSplitting(t *testing.T) {
	for _, tt := range []struct {
		name  string
		text  string
		query string
		want  int
	}{
		{name: "embedded Latin word", text: "私はPythonでADKを使っています", query: "Python", want: 1},
		{name: "embedded Latin word lowercase query", text: "私はPythonでADKを使っています", query: "adk", want: 1},
		{name: "embedded underscore word", text: "私はmy_var_nameとcafé_1を使う", query: "my_var_name", want: 1},
		{name: "embedded accented word", text: "私はmy_var_nameとcafé_1を使う", query: "café_1", want: 1},
		{name: "query is not split", text: "Python", query: "私はPython", want: 0},
		{name: "ordinal indicators are not Latin", text: "私はªPythonºを使う", query: "Python", want: 1},
		{name: "NFD event NFC query", text: "私はCafe\u0301でADKを使っています", query: "Café", want: 1},
		{name: "NFD event NFC lowercase query", text: "私はCafe\u0301でADKを使っています", query: "café", want: 1},
		{name: "NFD event other word", text: "私はCafe\u0301でADKを使っています", query: "Ruby", want: 0},
		{name: "NFC event NFD query", text: "Meet at the Café", query: "Cafe\u0301", want: 1},
		{name: "NFD Japanese event", text: norm.NFD.String("プログラミングを学ぶ"), query: "プログラミング", want: 1},
		{
			name:  "NFD Korean event",
			text:  "\u110c\u1166 \u110b\u1175\u1105\u1173\u11b7\u110b\u1173\u11ab \u1106\u1175\u11ab\u1109\u116e\u110b\u1175\u11b8\u1102\u1175\u1103\u1161",
			query: "민수입니다",
			want:  1,
		},
		{name: "Devanagari prefix", text: "\u0939\u093f\u0928\u094d\u0926\u0940", query: "\u0939", want: 1},
		{name: "uncomposed combining mark separates words", text: "q\u0301", query: "q", want: 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			svc := memory.InMemoryService()
			addEvents(t, svc, "s1", memoryTextEvent("event", tt.text))

			if got := search(t, svc, tt.query); len(got.Memories) != tt.want {
				t.Errorf("SearchMemory() returned %d entries, want %d", len(got.Memories), tt.want)
			}
		})
	}
}

func Test_inMemoryService_SearchMemory_FullUnicodeLowercasing(t *testing.T) {
	for _, tt := range []struct {
		name  string
		text  string
		query string
		want  int
	}{
		{name: "final sigma", text: "οδος", query: "ΟΔΟΣ", want: 1},
		{name: "final sigma in event text matched as substring", text: "私はΟΔΟΣ", query: "οδος", want: 1},
		{name: "dotted capital I", text: "İstanbul", query: "istanbul", want: 0},
		{name: "dotted capital I splits a run in CJK text", text: "私はİstanbulに", query: "stanbul", want: 1},
		{name: "Kelvin sign lowercases to ASCII", text: "kiwi", query: "\u212a", want: 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			svc := memory.InMemoryService()
			addEvents(t, svc, "s1", memoryTextEvent("event", tt.text))

			if got := search(t, svc, tt.query); len(got.Memories) != tt.want {
				t.Errorf("SearchMemory() returned %d entries, want %d", len(got.Memories), tt.want)
			}
		})
	}
}
