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

// Package sqlitememory provides local, persistent agent memory backed by SQLite.
package sqlitememory

import (
	"cmp"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"slices"

	"github.com/glebarez/sqlite"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/memory/internal/keyword"
	"google.golang.org/adk/v2/session"
)

const schema = `
CREATE TABLE IF NOT EXISTS adk_memory_events (
 id INTEGER PRIMARY KEY,
 app_name TEXT NOT NULL,
 user_id TEXT NOT NULL,
 session_id TEXT NOT NULL,
 search_text TEXT NOT NULL,
 entry BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS adk_memory_events_session
 ON adk_memory_events(app_name, user_id, session_id);
`

// Config configures a local SQLite memory service.
type Config struct {
	// Path is the database file to open or create. Its parent directory must exist.
	Path string
}

// Service stores searchable session events in a local SQLite file. It is safe
// for concurrent use. The caller owns the service and must close it after use.
type Service struct {
	db *sql.DB
}

var _ memory.Service = (*Service)(nil)

// New opens a SQLite memory service and initializes its tables if needed.
func New(ctx context.Context, cfg Config) (*Service, error) {
	if cfg.Path == "" {
		return nil, errors.New("sqlitememory: database path must not be empty")
	}
	path, err := filepath.Abs(cfg.Path)
	if err != nil {
		return nil, errors.New("sqlitememory: resolve database path")
	}
	uri := url.URL{Path: filepath.ToSlash(path)}
	params := url.Values{"_pragma": {"busy_timeout(5000)"}}
	db, err := sql.Open(sqlite.DriverName, "file:"+uri.EscapedPath()+"?"+params.Encode())
	if err != nil {
		return nil, fmt.Errorf("sqlitememory: open database: %w", err)
	}
	// Serialize operations on this service. A timeout also lets this connection
	// wait for a writer on a separate connection to the same file.
	db.SetMaxOpenConns(1)
	if _, err := db.ExecContext(ctx, schema); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("sqlitememory: initialize database: %w", err)
	}
	return &Service{db: db}, nil
}

// Close releases the database connection. It does not remove the database file.
func (s *Service) Close() error {
	return s.db.Close()
}

// AddSessionToMemory replaces the stored events of sess in a single transaction.
// Pass the complete session. Previously stored events omitted from sess are
// removed from memory. Events without visible text are skipped. Event content
// and custom metadata must be JSON-serializable.
func (s *Service) AddSessionToMemory(ctx context.Context, sess session.Session) error {
	if sess == nil {
		return errors.New("sqlitememory: session must not be nil")
	}
	appName, userID, sessionID := sess.AppName(), sess.UserID(), sess.ID()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlitememory: begin session import: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if _, err := tx.ExecContext(ctx, `DELETE FROM adk_memory_events
  WHERE app_name = ? AND user_id = ? AND session_id = ?`, appName, userID, sessionID); err != nil {
		return fmt.Errorf("sqlitememory: remove previous session events: %w", err)
	}
	if events := sess.Events(); events != nil {
		for event := range events.All() {
			if event == nil || event.Content == nil {
				continue
			}
			content, text := keyword.VisibleContent(event.Content)
			if text == "" {
				continue
			}
			encoded, err := json.Marshal(memory.Entry{
				ID: event.ID, Content: content, Author: event.Author,
				Timestamp: event.Timestamp, CustomMetadata: event.CustomMetadata,
			})
			if err != nil {
				return errors.New("sqlitememory: encode memory entry")
			}
			if _, err := tx.ExecContext(ctx, `INSERT INTO adk_memory_events
    (app_name, user_id, session_id, search_text, entry) VALUES (?, ?, ?, ?, ?)`,
				appName, userID, sessionID, text, encoded); err != nil {
				return fmt.Errorf("sqlitememory: insert memory entry: %w", err)
			}
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlitememory: commit session import: %w", err)
	}
	return nil
}

// SearchMemory searches imported events for req.AppName and req.UserID using
// [memory.InMemoryService]'s keyword matching rules. It returns at most ten
// entries ranked by the number of distinct query words matched. Ties follow
// the order of each session's most recent import, earliest first, then event
// order within the session.
func (s *Service) SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
	if req == nil {
		return nil, errors.New("sqlitememory: search request must not be nil")
	}
	query := keyword.NewQuery(req.Query)
	response := &memory.SearchResponse{}
	if len(query) == 0 {
		return response, nil
	}

	rows, err := s.db.QueryContext(ctx, `SELECT search_text, entry FROM adk_memory_events
  WHERE app_name = ? AND user_id = ? ORDER BY id`, req.AppName, req.UserID)
	if err != nil {
		return nil, fmt.Errorf("sqlitememory: search memory: %w", err)
	}
	defer func() { _ = rows.Close() }()

	type match struct {
		score   int
		encoded []byte
	}
	matches := make([]match, 0, keyword.MaxResults+1)
	for rows.Next() {
		var text string
		var encoded []byte
		if err := rows.Scan(&text, &encoded); err != nil {
			return nil, fmt.Errorf("sqlitememory: read search row: %w", err)
		}
		if score := query.Score(keyword.NewDocument(text)); score > 0 {
			matches = append(matches, match{score: score, encoded: encoded})
			slices.SortStableFunc(matches, func(a, b match) int { return cmp.Compare(b.score, a.score) })
			if len(matches) > keyword.MaxResults {
				matches = matches[:keyword.MaxResults]
			}
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlitememory: read search results: %w", err)
	}
	for _, match := range matches {
		var entry memory.Entry
		if err := json.Unmarshal(match.encoded, &entry); err != nil {
			return nil, errors.New("sqlitememory: decode memory entry")
		}
		response.Memories = append(response.Memories, entry)
	}
	return response, nil
}
