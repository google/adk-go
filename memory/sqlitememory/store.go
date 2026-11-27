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

// Package sqlitememory provides a SQLite-backed memory service.
package sqlitememory

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"database/sql/driver"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strconv"
	"strings"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

const (
	defaultMaxEventBytes = 262144
	defaultMaxResults    = 50
	unknownSessionID     = "__unknown_session_id__"
)

// Config configures a SQLite memory service.
type Config struct {
	// DB is an open SQLite database owned by the caller. Go takes a handle
	// in place of Python's db_path so callers can choose their SQLite driver
	// and avoid linking two drivers that both register the name "sqlite".
	//
	// A plain ":memory:" database belongs to one connection. Call
	// SetMaxOpenConns(1) and SetMaxIdleConns(1), and leave connection
	// lifetime and idle time limits disabled so the pool keeps it open.
	DB *sql.DB
	// FTS controls full-text search: "auto" (the default) uses FTS5 when
	// available, "on" requires it, and "off" uses substring matching.
	// Reopening a database with an existing FTS index requires an FTS5-capable driver.
	FTS string
	// MaxEventBytes limits the UTF-8 bytes used to make each event searchable. Zero
	// uses 256 KiB. The stored event content is kept in full.
	MaxEventBytes int
	// MaxResults limits the number of entries per search. Zero uses 50.
	MaxResults int
}

type store struct {
	db            *sql.DB
	ftsEnabled    bool
	maxEventBytes int
	maxResults    int
}

var (
	_ memory.Service    = (*store)(nil)
	_ memory.EventAdder = (*store)(nil)
)

// New initializes the schema and returns a SQLite memory service. The service
// supports incremental writes through [memory.AddEventsToMemory]. The caller
// remains responsible for closing cfg.DB.
func New(ctx context.Context, cfg Config) (memory.Service, error) {
	if cfg.DB == nil {
		return nil, errors.New("sqlitememory: database must not be nil")
	}
	switch cfg.FTS {
	case "", "auto", "on", "off":
	default:
		return nil, errors.New("sqlitememory: FTS must be auto, on, or off")
	}
	if cfg.MaxEventBytes < 0 {
		return nil, errors.New("sqlitememory: maximum event bytes must not be negative")
	}
	if cfg.MaxEventBytes == 0 {
		cfg.MaxEventBytes = defaultMaxEventBytes
	}
	if cfg.MaxResults < 0 {
		return nil, errors.New("sqlitememory: maximum results must not be negative")
	}
	if cfg.MaxResults == 0 {
		cfg.MaxResults = defaultMaxResults
	}
	var ftsEnabled bool
	if err := withTransaction(ctx, cfg.DB, func(conn *sql.Conn) error {
		var err error
		ftsEnabled, err = initSchema(ctx, conn, cfg.FTS)
		return err
	}); err != nil {
		return nil, fmt.Errorf("sqlitememory: initialize schema: %w", err)
	}
	return &store{
		db: cfg.DB, ftsEnabled: ftsEnabled, maxEventBytes: cfg.MaxEventBytes, maxResults: cfg.MaxResults,
	}, nil
}

// AddSessionToMemory upserts the events in sess. Go does not emit the session
// rewind events that the Python backend uses to remove stored events.
func (s *store) AddSessionToMemory(ctx context.Context, sess session.Session) error {
	if sess == nil {
		return errors.New("sqlitememory: session must not be nil")
	}
	events := sess.Events()
	if events == nil || events.Len() == 0 {
		return nil
	}
	return s.writeEvents(ctx, sess.AppName(), sess.UserID(), sess.ID(), events.All())
}

// AddEventsToMemory upserts a batch without removing previously stored events.
// Repeated event IDs update their row within the same app, user, and session.
func (s *store) AddEventsToMemory(ctx context.Context, req *memory.AddEventsRequest) error {
	if req == nil {
		return errors.New("sqlitememory: add events request must not be nil")
	}
	if len(req.Events) == 0 {
		return nil
	}
	return s.writeEvents(ctx, req.AppName, req.UserID, req.SessionID, slices.Values(req.Events))
}

const upsertEventSQL = `
INSERT INTO memory_events (app_name, user_id, session_id, event_id, author, timestamp, content_json, search_text)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(app_name, user_id, session_id, event_id) DO UPDATE SET
  author = excluded.author,
  timestamp = excluded.timestamp,
  content_json = excluded.content_json,
  search_text = excluded.search_text`

func (s *store) writeEvents(ctx context.Context, app, user, sessionID string, events iter.Seq[*session.Event]) error {
	if sessionID == "" {
		sessionID = unknownSessionID
	}
	return withTransaction(ctx, s.db, func(conn *sql.Conn) error {
		stmt, err := conn.PrepareContext(ctx, upsertEventSQL)
		if err != nil {
			return fmt.Errorf("sqlitememory: prepare event write: %w", err)
		}
		defer func() { _ = stmt.Close() }()
		for event := range events {
			if event == nil || event.Content == nil {
				continue
			}
			searchText, visible := eventSearchText(event.Content, s.maxEventBytes)
			if !visible {
				continue
			}
			var buf bytes.Buffer
			enc := json.NewEncoder(&buf)
			enc.SetEscapeHTML(false)
			if err := enc.Encode(event.Content); err != nil {
				return errors.New("sqlitememory: encode event content")
			}
			content := strings.TrimSuffix(buf.String(), "\n")
			timestamp := event.Timestamp
			if timestamp.IsZero() {
				timestamp = platform.Now(ctx)
			}
			seconds := float64(timestamp.Unix()) + float64(timestamp.Nanosecond())/1e9
			id := event.ID
			if id == "" {
				id = fallbackEventID(event.Author, seconds, content)
			}
			if _, err := stmt.ExecContext(ctx, app, user, sessionID, id, event.Author, seconds, content, searchText); err != nil {
				return fmt.Errorf("sqlitememory: upsert event: %w", err)
			}
		}
		return nil
	})
}

func fallbackEventID(author string, timestamp float64, content string) string {
	// Python renders Unix timestamps as floats, including .0 on whole seconds.
	t := strconv.FormatFloat(timestamp, 'f', -1, 64)
	if timestamp != 0 && timestamp > -1e-4 && timestamp < 1e-4 {
		t = strconv.FormatFloat(timestamp, 'e', -1, 64)
	} else if !strings.Contains(t, ".") {
		t += ".0"
	}
	sum := sha256.Sum256([]byte(author + ":" + t + ":" + content))
	return hex.EncodeToString(sum[:16])
}

func withTransaction(ctx context.Context, db *sql.DB, f func(*sql.Conn) error) error {
	// Keep the connection until cleanup. SQLite can leave a transaction open
	// after a failed COMMIT, which some drivers return to the pool unchanged.
	conn, err := db.Conn(ctx)
	if err != nil {
		return fmt.Errorf("sqlitememory: get database connection: %w", err)
	}
	defer func() { _ = conn.Close() }()
	// Take the write lock up front so a competing writer can wait for the
	// busy timeout instead of failing immediately when upgrading a read lock.
	if _, err := conn.ExecContext(ctx, "BEGIN IMMEDIATE"); err != nil {
		return fmt.Errorf("sqlitememory: begin transaction: %w", err)
	}
	committed := false
	defer func() {
		if !committed {
			if _, err := conn.ExecContext(context.WithoutCancel(ctx), "ROLLBACK"); err != nil {
				_ = conn.Raw(func(any) error { return driver.ErrBadConn })
			}
		}
	}()
	if err := f(conn); err != nil {
		return err
	}
	if _, err := conn.ExecContext(ctx, "COMMIT"); err != nil {
		return fmt.Errorf("sqlitememory: commit transaction: %w", err)
	}
	committed = true
	return nil
}
