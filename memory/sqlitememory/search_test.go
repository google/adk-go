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

package sqlitememory

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
)

func TestSearchMemoryAfterReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "memory.db")
	db := openDB(t, path)
	s := newTestStore(t, Config{DB: db, FTS: "on"})
	const visible = "oolong 乌龙茶"
	event := textEvent("first", visible)
	event.Timestamp = time.Unix(-1, 900_000_000).UTC()
	event.Content.Parts = append(event.Content.Parts, nil, &genai.Part{Text: "reasoning", Thought: true})
	if err := s.AddSessionToMemory(t.Context(), newSession(t, event)); err != nil {
		t.Fatalf("AddSessionToMemory: %v", err)
	}
	for _, scope := range []memory.AddEventsRequest{
		{AppName: "other-app", UserID: "user", SessionID: "session"},
		{AppName: "app", UserID: "other-user", SessionID: "session"},
	} {
		scope.Events = []*session.Event{textEvent("other", visible)}
		addEvents(t, s, &scope)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}
	s = newTestStore(t, Config{DB: openDB(t, path)})

	for _, query := range []string{"OOLONG", "乌龙"} {
		entries := searchEntries(t, s, query)
		if len(entries) != 1 {
			t.Fatalf("search returned %d results, want 1", len(entries))
		}
		got := entries[0]
		if got.ID != "1" || got.Author != event.Author || !got.Timestamp.Equal(event.Timestamp) {
			t.Error("search result did not retain the row ID, author, and timestamp")
		}
		if len(got.CustomMetadata) != 2 || got.CustomMetadata["session_id"] != "session" || got.CustomMetadata["event_id"] != "first" {
			t.Error("search result did not identify the source event and session")
		}
		if got.Content == nil || got.Content.Role != event.Content.Role || len(got.Content.Parts) != 1 ||
			got.Content.Parts[0] == nil || got.Content.Parts[0].Text != visible {
			t.Error("search result did not return only the visible content")
		}
	}
	if n := len(searchEntries(t, s, "reasoning")); n != 0 {
		t.Errorf("thought query returned %d results, want 0", n)
	}
}

func TestSearchMemoryUnspacedScripts(t *testing.T) {
	for _, tc := range []struct{ name, text, query string }{
		{name: "Chinese", text: "我想买这辆车，价格是二十万。", query: "价格"},
		{name: "mixed Japanese and Latin", text: "私はPythonを使う", query: "Python 使う"},
		{name: "halfwidth katakana", text: "私はﾃｽﾄを書く", query: "ﾃｽﾄ"},
		{name: "Korean", text: "저는 파이썬을 좋아해요", query: "파이썬"},
		{name: "Thai", text: "ฉันชอบกินข้าว", query: "ข้าว"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newTestStore(t, Config{DB: openDB(t, filepath.Join(t.TempDir(), "memory.db")), FTS: "on"})
			addEvents(t, s, &memory.AddEventsRequest{
				AppName: "app", UserID: "user", Events: []*session.Event{textEvent("first", tc.text)},
			})
			entries := searchEntries(t, s, tc.query)
			if len(entries) != 1 || entries[0].CustomMetadata["event_id"] != "first" {
				t.Error("search did not match a word inside unspaced text")
			}
		})
	}
}

func TestSearchMemoryRanksAndLimits(t *testing.T) {
	for _, tc := range []struct {
		name, fts, bestText, tieText string
	}{
		{name: "FTS relevance", fts: "on", bestText: "oolong oolong oolong green", tieText: "oolong"},
		{name: "LIKE token count", fts: "off", bestText: "oolong green", tieText: "oolong tea"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newTestStore(t, Config{DB: openDB(t, filepath.Join(t.TempDir(), "memory.db")), FTS: tc.fts, MaxResults: 2})
			best := textEvent("best", tc.bestText)
			older := textEvent("older", tc.tieText)
			newer := textEvent("newer", tc.tieText)
			newer.Timestamp = older.Timestamp.Add(time.Second)
			best.Timestamp = older.Timestamp.Add(-time.Second)
			addEvents(t, s, &memory.AddEventsRequest{
				AppName: "app", UserID: "user", Events: []*session.Event{best, older, newer},
			})
			entries := searchEntries(t, s, "oolong green")
			if len(entries) != 2 {
				t.Fatalf("search returned %d results, want 2", len(entries))
			}
			if entries[0].CustomMetadata["event_id"] != "best" || entries[1].CustomMetadata["event_id"] != "newer" {
				t.Error("search did not sort by relevance, then by newest timestamp")
			}
		})
	}
}

func TestSearchMemoryLikeWildcardsAndDefaultLimit(t *testing.T) {
	db := openDB(t, filepath.Join(t.TempDir(), "memory.db"))
	s := newTestStore(t, Config{DB: db, FTS: "on"})
	events := []*session.Event{textEvent("literal", "oolong 100% user_id"), textEvent("decoy", "oolong 1000 userXid")}
	for i := range 49 {
		events = append(events, textEvent(fmt.Sprintf("extra-%d", i), "oolong"))
	}
	addEvents(t, s, &memory.AddEventsRequest{AppName: "app", UserID: "user", Events: events})

	if n := len(searchEntries(t, s, "olong")); n != 0 {
		t.Errorf("FTS substring query returned %d results, want 0", n)
	}
	if entries := searchEntries(t, s, "100%"); len(entries) != 1 || entries[0].CustomMetadata["event_id"] != "literal" {
		t.Error("FTS query with a LIKE wildcard did not match the literal event")
	}
	// Turning FTS off must use LIKE even if the database already has an index.
	s = newTestStore(t, Config{DB: db, FTS: "off"})
	for _, query := range []string{"100%", "user_id"} {
		entries := searchEntries(t, s, query)
		if len(entries) != 1 || entries[0].CustomMetadata["event_id"] != "literal" {
			t.Error("LIKE query did not match a wildcard literally")
		}
	}
	if n := len(searchEntries(t, s, "olong")); n != 50 {
		t.Errorf("substring query returned %d results, want 50", n)
	}
}

func TestSearchMemoryAutoFallsBackWithoutFTS5(t *testing.T) {
	path := filepath.Join(t.TempDir(), "memory.db")
	db := sql.OpenDB(noFTS5Connector{sqlite: openDB(t, path).Driver(), path: path})
	t.Cleanup(func() { _ = db.Close() })
	s := newTestStore(t, Config{DB: db, FTS: "auto"})
	addEvents(t, s, &memory.AddEventsRequest{
		AppName: "app", UserID: "user", Events: []*session.Event{textEvent("first", "oolong tea")},
	})
	// LIKE can match a substring where FTS5 would require the whole word.
	entries := searchEntries(t, s, "olong")
	if len(entries) != 1 || entries[0].CustomMetadata["event_id"] != "first" {
		t.Error("auto mode did not find the stored event with substring search")
	}
}

// noFTS5Connector simulates an SQLite build without FTS5 while leaving writes
// and searches backed by SQLite.
type noFTS5Connector struct {
	sqlite driver.Driver
	path   string
}

func (c noFTS5Connector) Connect(context.Context) (driver.Conn, error) {
	conn, err := c.sqlite.Open(c.path)
	if err != nil {
		return nil, err
	}
	return noFTS5Conn{Conn: conn}, nil
}

func (c noFTS5Connector) Driver() driver.Driver { return c.sqlite }

type noFTS5Conn struct{ driver.Conn }

func (c noFTS5Conn) ExecContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Result, error) {
	if strings.Contains(strings.ToLower(query), "using fts5(") {
		return nil, errors.New("no such module: fts5")
	}
	return c.Conn.(driver.ExecerContext).ExecContext(ctx, query, args)
}

func TestSearchMemoryEmptyAndErrors(t *testing.T) {
	db := openDB(t, filepath.Join(t.TempDir(), "memory.db"))
	s := newTestStore(t, Config{DB: db})
	addEvents(t, s, &memory.AddEventsRequest{
		AppName: "app", UserID: "user", Events: []*session.Event{textEvent("first", "oolong")},
	})
	if _, err := db.ExecContext(t.Context(), `UPDATE memory_events SET content_json = '{'`); err != nil {
		t.Fatalf("write malformed event content: %v", err)
	}
	request := &memory.SearchRequest{AppName: "app", UserID: "user", Query: "oolong"}
	if got, err := s.SearchMemory(t.Context(), request); err == nil || got != nil {
		t.Error("search accepted malformed event content")
	}
	const invalidTimestamp = "invalid timestamp value"
	if _, err := db.ExecContext(t.Context(), `UPDATE memory_events SET timestamp = ?`, invalidTimestamp); err != nil {
		t.Fatalf("write malformed timestamp: %v", err)
	}
	if got, err := s.SearchMemory(t.Context(), request); got != nil || err == nil || strings.Contains(err.Error(), invalidTimestamp) {
		t.Error("search should reject the malformed timestamp without disclosing its value")
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}
	if _, err := s.SearchMemory(t.Context(), request); err == nil {
		t.Error("search did not report a database error")
	}
	if _, err := s.SearchMemory(t.Context(), nil); err == nil {
		t.Error("search accepted a nil request")
	}
	for _, query := range []string{"", " \n ", "?!", "\u2764\ufe0f"} {
		if got := searchEntries(t, s, query); got == nil || len(got) != 0 {
			t.Error("query without tokens did not return an empty slice")
		}
	}
}

func TestExtractTokens(t *testing.T) {
	for _, tc := range []struct {
		name  string
		query string
		want  []string
	}{
		{name: "normalize and deduplicate", query: "Cafe\u0301 CAFÉ", want: []string{"café"}},
		{name: "split mixed script runs", query: "Oolong乌龙", want: []string{"oolong", "乌龙"}},
		{name: "keep internal marks", query: "ข้าว", want: []string{"ข้าว"}},
		{name: "drop leading marks", query: "\u0301\u0e02 \u2764\ufe0f", want: []string{"\u0e02"}},
		{name: "keep LIKE wildcards in tokens", query: "user_id 100%", want: []string{"user_id", "100%"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !slices.Equal(extractTokens(tc.query), tc.want) {
				t.Error("query did not produce the expected tokens")
			}
		})
	}
}

func TestHasUnspacedScript(t *testing.T) {
	for _, token := range []string{"乌", "\U00020000", "\uf900", "\u3042", "\uff71", "\uac00", "\u0e02"} {
		if !hasUnspacedScript(token) {
			t.Error("character did not select substring matching")
		}
	}
	if hasUnspacedScript("café") || hasUnspacedScript("\u03b2") || hasUnspacedScript("\u3005") {
		t.Error("unsupported character selected substring matching")
	}
}

func searchEntries(t *testing.T, s memory.Service, query string) []memory.Entry {
	t.Helper()
	response, err := s.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app", UserID: "user", Query: query})
	if err != nil {
		t.Fatalf("SearchMemory: %v", err)
	}
	return response.Memories
}
