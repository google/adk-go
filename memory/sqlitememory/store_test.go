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
	"database/sql"
	"encoding/json"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	_ "github.com/glebarez/sqlite" // Register the repository's SQLite driver for these tests.
	"google.golang.org/genai"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

func TestUpsertAndReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "memory#%.db")
	db := openDB(t, path)
	s := newTestStore(t, storeConfig{db: db})
	first := textEvent("first", "oolong tea")
	stateOnly := &session.Event{ID: "state-only"}
	sess := newSession(t, first, textEvent("second", "green tea"), stateOnly)
	if err := s.AddSessionToMemory(t.Context(), sess); err != nil {
		t.Fatalf("AddSessionToMemory: %v", err)
	}
	const firstEventRow = `FROM memory_events WHERE app_name = 'app' AND user_id = 'user' AND session_id = 'session' AND event_id = 'first'`
	var originalID int64
	if err := db.QueryRowContext(t.Context(), `SELECT id `+firstEventRow).Scan(&originalID); err != nil {
		t.Fatalf("read first event: %v", err)
	}
	updated := textEvent("first", "black tea")
	if err := s.AddSessionToMemory(t.Context(), newSession(t, updated)); err != nil {
		t.Fatalf("reimport shorter session: %v", err)
	}
	for _, scope := range []memory.AddEventsRequest{
		{AppName: "other-app", UserID: "user", SessionID: "session"},
		{AppName: "app", UserID: "other-user", SessionID: "session"},
		{AppName: "app", UserID: "user", SessionID: "other-session"},
	} {
		scope.Events = []*session.Event{first}
		addEvents(t, s, &scope)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close database: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal("database file does not exist at the supplied path")
	}
	reopened := openDB(t, path)
	newTestStore(t, storeConfig{db: reopened})
	checkCount(t, reopened, 5, `SELECT count(*) FROM memory_events`)
	checkCount(t, reopened, 1, `SELECT count(*) FROM memory_events WHERE event_id = 'second'`)
	checkCount(t, reopened, 0, `SELECT count(*) FROM memory_events WHERE event_id = 'state-only'`)
	var gotID int64
	var contentJSON string
	if err := reopened.QueryRowContext(t.Context(), `SELECT id, content_json `+firstEventRow).Scan(&gotID, &contentJSON); err != nil {
		t.Fatalf("read updated event: %v", err)
	}
	if gotID != originalID {
		t.Error("upsert changed the original row ID")
	}
	var content genai.Content
	if err := json.Unmarshal([]byte(contentJSON), &content); err != nil {
		t.Fatal("could not decode stored event content")
	}
	if len(content.Parts) != 1 || content.Parts[0].Text != updated.Content.Parts[0].Text {
		t.Error("upsert did not store the new content")
	}
}

func TestMissingEventID(t *testing.T) {
	db := openDB(t, filepath.Join(t.TempDir(), "memory.db"))
	s := newTestStore(t, storeConfig{db: db, maxEventBytes: 3})
	event := textEvent("", "oolong tea")
	stamp := event.Timestamp
	event.Timestamp = time.Time{}
	req := &memory.AddEventsRequest{
		AppName: "app", UserID: "user", Events: []*session.Event{nil, textEvent("blank", " \n "), event},
	}
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return stamp })
	for range 2 {
		if err := s.AddEventsToMemory(ctx, req); err != nil {
			t.Fatalf("AddEventsToMemory: %v", err)
		}
	}
	checkCount(t, db, 1, `SELECT count(*) FROM memory_events`)
	var sessionID, eventID, searchText string
	var contentJSON []byte
	if err := db.QueryRowContext(t.Context(), `SELECT session_id, event_id, search_text, content_json FROM memory_events`).Scan(&sessionID, &eventID, &searchText, &contentJSON); err != nil {
		t.Fatalf("read fallback identifiers: %v", err)
	}
	// This ID is the Python SHA-256 prefix for the same text event and timestamp.
	if sessionID != "__unknown_session_id__" || eventID != "15b2c8fffd99de1c278e4a51aab14b25" {
		t.Error("missing session or event ID did not use the expected fallback")
	}
	var content genai.Content
	if err := json.Unmarshal(contentJSON, &content); err != nil {
		t.Fatal("could not decode stored event content")
	}
	if searchText != "ool" || len(content.Parts) != 1 || content.Parts[0].Text != event.Content.Parts[0].Text {
		t.Error("byte limit should truncate only the search text")
	}
}

func TestTruncatedSearchTextCanBeEmpty(t *testing.T) {
	db := openDB(t, filepath.Join(t.TempDir(), "memory.db"))
	s := newTestStore(t, storeConfig{db: db, maxEventBytes: 1})
	addEvents(t, s, &memory.AddEventsRequest{
		AppName: "app", UserID: "user", Events: []*session.Event{textEvent("first", "茶")},
	})
	checkCount(t, db, 1, `SELECT count(*) FROM memory_events WHERE event_id = 'first' AND search_text = ''`)
}

func TestThoughtStoredButNotIndexed(t *testing.T) {
	db := openDB(t, filepath.Join(t.TempDir(), "memory.db"))
	s := newTestStore(t, storeConfig{db: db, fts: "on"})
	event := textEvent("first", "oolong tea")
	thought := &genai.Part{Text: "reasoning", Thought: true}
	event.Content.Parts = append(event.Content.Parts, thought)
	addEvents(t, s, &memory.AddEventsRequest{
		AppName: "app", UserID: "user", SessionID: "session", Events: []*session.Event{event},
	})

	var contentJSON []byte
	if err := db.QueryRowContext(t.Context(), `SELECT content_json FROM memory_events WHERE event_id = 'first'`).Scan(&contentJSON); err != nil {
		t.Fatalf("read stored event content: %v", err)
	}
	var content genai.Content
	if err := json.Unmarshal(contentJSON, &content); err != nil {
		t.Fatal("could not decode stored event content")
	}
	if len(content.Parts) != 2 || content.Parts[1] == nil || !content.Parts[1].Thought || content.Parts[1].Text != thought.Text {
		t.Error("stored event content did not retain the thought part")
	}

	const match = `SELECT count(*) FROM memory_events_fts WHERE search_text MATCH ?`
	checkCount(t, db, 1, match, "oolong")
	checkCount(t, db, 0, match, thought.Text)
}

func TestFTSBackfillAndTriggers(t *testing.T) {
	db := openDB(t, filepath.Join(t.TempDir(), "memory.db"))
	s := newTestStore(t, storeConfig{db: db, fts: "off"})
	checkCount(t, db, 0, `SELECT count(*) FROM sqlite_master WHERE name = 'memory_events_fts'`)
	req := &memory.AddEventsRequest{AppName: "app", UserID: "user", Events: []*session.Event{textEvent("first", "oolong tea")}}
	addEvents(t, s, req)

	s = newTestStore(t, storeConfig{db: db, fts: "on"})
	const match = `SELECT count(*) FROM memory_events_fts WHERE search_text MATCH ?`
	checkCount(t, db, 1, match, "oolong")
	req.Events = []*session.Event{textEvent("first", "black tea"), textEvent("second", "green tea")}
	addEvents(t, s, req)
	checkCount(t, db, 0, match, "oolong")
	checkCount(t, db, 1, match, "black")
	checkCount(t, db, 1, match, "green")
	if _, err := db.ExecContext(t.Context(), `DELETE FROM memory_events WHERE event_id = 'first'`); err != nil {
		t.Fatalf("delete event row: %v", err)
	}
	checkCount(t, db, 0, match, "black")
	checkCount(t, db, 1, match, "green")
}

func TestSchemaVersion(t *testing.T) {
	db := openDB(t, filepath.Join(t.TempDir(), "memory.db"))
	newTestStore(t, storeConfig{db: db})
	checkCount(t, db, 1, `SELECT count(*) FROM adk_memory_metadata WHERE key = 'schema_version' AND value = '1'`)
	checkCount(t, db, 1, `SELECT count(*) FROM sqlite_master WHERE name = 'memory_events_fts'`)
	if _, err := db.ExecContext(t.Context(), `UPDATE adk_memory_metadata SET value = '2' WHERE key = 'schema_version'`); err != nil {
		t.Fatalf("set incompatible schema version: %v", err)
	}
	if _, err := newStore(t.Context(), storeConfig{db: db}); err == nil || !strings.Contains(err.Error(), "unsupported schema version") {
		t.Fatal("newStore did not reject an unsupported schema version")
	}
	checkCount(t, db, 1, `SELECT count(*) FROM adk_memory_metadata WHERE key = 'schema_version' AND value = '2'`)
}

func TestWriteRollback(t *testing.T) {
	db := openDB(t, filepath.Join(t.TempDir(), "memory.db"))
	s := newTestStore(t, storeConfig{db: db})
	invalid := textEvent("invalid", "invalid memory")
	invalid.Content.Parts = append(invalid.Content.Parts, &genai.Part{
		FunctionCall: &genai.FunctionCall{Args: map[string]any{"unsupported": make(chan struct{})}},
	})
	req := &memory.AddEventsRequest{AppName: "app", UserID: "user", Events: []*session.Event{textEvent("first", "green tea"), invalid}}
	if err := s.AddEventsToMemory(t.Context(), req); err == nil || !strings.Contains(err.Error(), "encode event content") {
		t.Fatal("AddEventsToMemory did not report an encoding error")
	}
	checkCount(t, db, 0, `SELECT count(*) FROM memory_events`)
	checkCount(t, db, 0, `SELECT count(*) FROM memory_events_fts WHERE search_text MATCH 'green'`)
}

func TestCommitBusyRollsBack(t *testing.T) {
	ctx := t.Context()
	path := filepath.Join(t.TempDir(), "memory.db")
	db := openDB(t, path)
	db.SetMaxOpenConns(1)
	s := newTestStore(t, storeConfig{db: db})
	reader := openDB(t, path)
	tx, err := reader.BeginTx(ctx, nil)
	if err != nil {
		t.Fatalf("begin reader transaction: %v", err)
	}
	defer func() { _ = tx.Rollback() }()
	var count int
	if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM memory_events`).Scan(&count); err != nil {
		t.Fatalf("read inside transaction: %v", err)
	}

	req := &memory.AddEventsRequest{AppName: "app", UserID: "user", Events: []*session.Event{textEvent("first", "green tea")}}
	if err := s.AddEventsToMemory(ctx, req); err == nil || !strings.Contains(err.Error(), "commit transaction") {
		t.Fatal("write did not fail at COMMIT while the read lock was held")
	}
	checkCount(t, db, 0, `SELECT count(*) FROM memory_events`)
	if err := tx.Rollback(); err != nil {
		t.Fatalf("release read lock: %v", err)
	}
	addEvents(t, s, req)
	checkCount(t, db, 1, `SELECT count(*) FROM memory_events`)
}

func TestInvalidConfig(t *testing.T) {
	db := openDB(t, filepath.Join(t.TempDir(), "memory.db"))
	for _, tc := range []struct {
		name string
		cfg  storeConfig
	}{
		{name: "missing database", cfg: storeConfig{}},
		{name: "invalid FTS mode", cfg: storeConfig{db: db, fts: "unknown"}},
		{name: "negative byte limit", cfg: storeConfig{db: db, maxEventBytes: -1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := newStore(t.Context(), tc.cfg); err == nil {
				t.Error("newStore accepted invalid configuration")
			}
		})
	}
}

func openDB(t *testing.T, path string) *sql.DB {
	t.Helper()
	uri := url.URL{Scheme: "file", Path: filepath.ToSlash(path), RawQuery: "_pragma=busy_timeout(0)"}
	db, err := sql.Open("sqlite", uri.String())
	if err != nil {
		t.Fatalf("open SQLite: %v", err)
	}
	t.Cleanup(func() {
		if err := db.Close(); err != nil {
			t.Errorf("close database: %v", err)
		}
	})
	return db
}

func newTestStore(t *testing.T, cfg storeConfig) *store {
	t.Helper()
	s, err := newStore(t.Context(), cfg)
	if err != nil {
		t.Fatalf("newStore: %v", err)
	}
	return s
}

func addEvents(t *testing.T, s *store, req *memory.AddEventsRequest) {
	t.Helper()
	if err := s.AddEventsToMemory(t.Context(), req); err != nil {
		t.Fatalf("AddEventsToMemory: %v", err)
	}
}

func checkCount(t *testing.T, db *sql.DB, want int, query string, args ...any) {
	t.Helper()
	var got int
	if err := db.QueryRowContext(t.Context(), query, args...).Scan(&got); err != nil {
		t.Fatalf("count rows: %v", err)
	}
	if got != want {
		t.Errorf("query returned %d rows, want %d", got, want)
	}
}

func textEvent(id, text string) *session.Event {
	return &session.Event{
		ID: id, Author: "user", Timestamp: time.Date(2026, time.January, 1, 12, 0, 0, 0, time.UTC),
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleUser)},
	}
}

func newSession(t *testing.T, events ...*session.Event) session.Session {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "session"})
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	for _, event := range events {
		if err := svc.AppendEvent(t.Context(), resp.Session, event); err != nil {
			t.Fatalf("append event: %v", err)
		}
	}
	return resp.Session
}
