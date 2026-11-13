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

// Package sqliteschema tells whether an existing SQLite database can be
// migrated by the session/database service without damaging it.
//
// adk-python writes its own session tables under the same names (sessions,
// events). Running GORM's AutoMigrate over them adds columns but keeps
// adk-python's NOT NULL columns, so every later write fails. The check runs
// before AutoMigrate so that such a file is left as it was.
package sqliteschema

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
)

// ErrIncompatible reports that a SQLite database holds tables of another
// session schema, such as adk-python's.
var ErrIncompatible = errors.New("database holds another session schema")

// CheckCompatible reports whether db is empty or uses the session/database
// schema. It returns an error wrapping [ErrIncompatible] when db holds
// adk-python's metadata table (DatabaseSessionService), an events table with
// an event_data column or without the actions and author columns, or a
// sessions table whose update_time column is REAL (SqliteSessionService).
// Any other error comes from querying db, and is ctx's error when ctx is
// done before a query starts. CheckCompatible only reads db.
func CheckCompatible(ctx context.Context, db *sql.DB) error {
	meta, err := columnTypes(ctx, db, "adk_internal_metadata")
	if err != nil {
		return err
	}
	if len(meta) > 0 {
		return fmt.Errorf("%w: table adk_internal_metadata is adk-python's", ErrIncompatible)
	}
	events, err := columnTypes(ctx, db, "events")
	if err != nil {
		return err
	}
	if len(events) > 0 {
		if _, ok := events["event_data"]; ok {
			return fmt.Errorf("%w: table events has adk-python's event_data column", ErrIncompatible)
		}
		for _, name := range []string{"actions", "author"} {
			if _, ok := events[name]; !ok {
				return fmt.Errorf("%w: table events has no %s column", ErrIncompatible, name)
			}
		}
	}
	sessions, err := columnTypes(ctx, db, "sessions")
	if err != nil {
		return err
	}
	if strings.Contains(strings.ToUpper(sessions["update_time"]), "REAL") {
		return fmt.Errorf("%w: column sessions.update_time is REAL, as adk-python stores it", ErrIncompatible)
	}
	return nil
}

// columnTypes maps each column of table to its declared type. The map is
// empty when db has no such table.
func columnTypes(ctx context.Context, db *sql.DB, table string) (map[string]string, error) {
	rows, err := db.QueryContext(ctx, "SELECT name, type FROM pragma_table_info(?)", table)
	if err != nil {
		return nil, fmt.Errorf("read the columns of table %s: %w", table, err)
	}
	defer func() { _ = rows.Close() }()
	cols := map[string]string{}
	for rows.Next() {
		var name, typ string
		if err := rows.Scan(&name, &typ); err != nil {
			return nil, fmt.Errorf("read the columns of table %s: %w", table, err)
		}
		cols[name] = typ
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read the columns of table %s: %w", table, err)
	}
	return cols, nil
}
