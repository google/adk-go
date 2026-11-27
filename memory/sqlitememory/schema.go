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
	"errors"
	"fmt"
	"strings"
)

const schemaVersion = "1"

const metadataSchema = `
CREATE TABLE IF NOT EXISTS adk_memory_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
)`

const eventSchema = `
CREATE TABLE IF NOT EXISTS memory_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  app_name TEXT NOT NULL,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  author TEXT NOT NULL,
  timestamp REAL NOT NULL,
  content_json TEXT NOT NULL,
  search_text TEXT NOT NULL,
  UNIQUE(app_name, user_id, session_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_events_scope
ON memory_events(app_name, user_id, timestamp DESC);`

const ftsTable = `
CREATE VIRTUAL TABLE IF NOT EXISTS memory_events_fts
USING fts5(
  search_text,
  content='memory_events',
  content_rowid='id',
  tokenize='unicode61'
)`

const ftsTriggers = `
CREATE TRIGGER IF NOT EXISTS memory_events_ai
AFTER INSERT ON memory_events BEGIN
  INSERT INTO memory_events_fts(rowid, search_text)
  VALUES (new.id, new.search_text);
END;
CREATE TRIGGER IF NOT EXISTS memory_events_ad
AFTER DELETE ON memory_events BEGIN
  INSERT INTO memory_events_fts(memory_events_fts, rowid, search_text)
  VALUES ('delete', old.id, old.search_text);
END;
CREATE TRIGGER IF NOT EXISTS memory_events_au
AFTER UPDATE OF search_text ON memory_events BEGIN
  INSERT INTO memory_events_fts(memory_events_fts, rowid, search_text)
  VALUES ('delete', old.id, old.search_text);
  INSERT INTO memory_events_fts(rowid, search_text)
  VALUES (new.id, new.search_text);
END;`

func initSchema(ctx context.Context, conn *sql.Conn, fts string) error {
	if _, err := conn.ExecContext(ctx, metadataSchema); err != nil {
		return fmt.Errorf("create metadata table: %w", err)
	}
	if _, err := conn.ExecContext(ctx, `INSERT OR IGNORE INTO adk_memory_metadata (key, value) VALUES ('schema_version', ?)`, schemaVersion); err != nil {
		return fmt.Errorf("record schema version: %w", err)
	}
	var version string
	if err := conn.QueryRowContext(ctx, `SELECT value FROM adk_memory_metadata WHERE key = 'schema_version'`).Scan(&version); err != nil {
		return fmt.Errorf("read schema version: %w", err)
	}
	if version != schemaVersion {
		return errors.New("unsupported schema version")
	}
	if _, err := conn.ExecContext(ctx, eventSchema); err != nil {
		return fmt.Errorf("create event table and scope index: %w", err)
	}
	if fts == "off" {
		return nil
	}
	return initFTS(ctx, conn, fts == "on")
}

func initFTS(ctx context.Context, conn *sql.Conn, required bool) error {
	var exists bool
	if err := conn.QueryRowContext(ctx, `SELECT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_events_fts')`).Scan(&exists); err != nil {
		return fmt.Errorf("check FTS index: %w", err)
	}
	if _, err := conn.ExecContext(ctx, ftsTable); err != nil {
		// SQLite exposes a missing FTS5 module as a generic SQL error.
		if !required && strings.Contains(err.Error(), "no such module: fts5") {
			return nil
		}
		return fmt.Errorf("create FTS index: %w", err)
	}
	if _, err := conn.ExecContext(ctx, ftsTriggers); err != nil {
		return fmt.Errorf("create FTS triggers: %w", err)
	}
	if !exists {
		if _, err := conn.ExecContext(ctx, `INSERT INTO memory_events_fts(memory_events_fts) VALUES ('rebuild')`); err != nil {
			return fmt.Errorf("index existing memory events: %w", err)
		}
	}
	return nil
}
