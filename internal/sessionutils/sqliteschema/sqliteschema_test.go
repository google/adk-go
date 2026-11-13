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

package sqliteschema_test

import (
	"database/sql"
	"errors"
	"path/filepath"
	"testing"

	"github.com/glebarez/sqlite"

	"google.golang.org/adk/v2/internal/sessionutils/sqliteschema"
	"google.golang.org/adk/v2/session/database"
)

// adk-python's SqliteSessionService schema
// (sessions/sqlite_session_service.py, SESSIONS_TABLE_SCHEMA and
// EVENTS_TABLE_SCHEMA).
const pythonSQLiteSchema = `
CREATE TABLE sessions (app_name TEXT NOT NULL, user_id TEXT NOT NULL, id TEXT NOT NULL,
  state TEXT NOT NULL, create_time REAL NOT NULL, update_time REAL NOT NULL,
  PRIMARY KEY (app_name, user_id, id));
CREATE TABLE events (id TEXT NOT NULL, app_name TEXT NOT NULL, user_id TEXT NOT NULL,
  session_id TEXT NOT NULL, invocation_id TEXT NOT NULL, timestamp REAL NOT NULL,
  event_data TEXT NOT NULL, PRIMARY KEY (app_name, user_id, session_id, id));`

func openFile(t *testing.T) (*sql.DB, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "session.db")
	db, err := sql.Open(sqlite.DriverName, path)
	if err != nil {
		t.Fatalf("sql.Open(%q): %v", path, err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db, path
}

func TestCheckCompatible(t *testing.T) {
	for _, tc := range []struct {
		name             string
		ddl              string
		wantIncompatible bool
	}{
		{name: "empty database", ddl: ""},
		{name: "adk-python SqliteSessionService", ddl: pythonSQLiteSchema, wantIncompatible: true},
		{
			name:             "adk-python metadata table only",
			ddl:              `CREATE TABLE adk_internal_metadata (key VARCHAR(128) PRIMARY KEY, value VARCHAR(128));`,
			wantIncompatible: true,
		},
		{
			name:             "events with event_data beside actions and author",
			ddl:              `CREATE TABLE events (id TEXT, actions BLOB, author TEXT, event_data TEXT);`,
			wantIncompatible: true,
		},
		{
			name:             "events without author",
			ddl:              `CREATE TABLE events (id TEXT, actions BLOB);`,
			wantIncompatible: true,
		},
		{
			name:             "events without actions",
			ddl:              `CREATE TABLE events (id TEXT, author TEXT);`,
			wantIncompatible: true,
		},
		{
			name:             "sessions with a REAL update_time only",
			ddl:              `CREATE TABLE sessions (id TEXT, update_time REAL NOT NULL);`,
			wantIncompatible: true,
		},
		{
			name: "sessions with a datetime update_time and events with actions and author",
			ddl: `CREATE TABLE sessions (id TEXT, update_time datetime);
CREATE TABLE events (id TEXT, actions BLOB, author TEXT);`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			db, _ := openFile(t)
			if tc.ddl != "" {
				if _, err := db.Exec(tc.ddl); err != nil {
					t.Fatalf("creating the schema: %v", err)
				}
			}
			err := sqliteschema.CheckCompatible(t.Context(), db)
			if got := errors.Is(err, sqliteschema.ErrIncompatible); got != tc.wantIncompatible {
				t.Errorf("CheckCompatible() = %v; errors.Is(err, ErrIncompatible) = %t, want %t", err, got, tc.wantIncompatible)
			}
			if !tc.wantIncompatible && err != nil {
				t.Errorf("CheckCompatible() = %v, want nil", err)
			}
		})
	}
}

// A file the database session service has migrated passes the check.
func TestCheckCompatible_DatabaseServiceSchema(t *testing.T) {
	db, path := openFile(t)
	svc, err := database.NewSessionService(sqlite.Open(path))
	if err != nil {
		t.Fatalf("database.NewSessionService: %v", err)
	}
	if err := database.AutoMigrate(svc); err != nil {
		t.Fatalf("database.AutoMigrate: %v", err)
	}
	if err := sqliteschema.CheckCompatible(t.Context(), db); err != nil {
		t.Errorf("CheckCompatible() on a migrated file = %v, want nil", err)
	}
}

// A failing query is reported, and is not mistaken for an incompatible
// schema.
func TestCheckCompatible_QueryError(t *testing.T) {
	db, _ := openFile(t)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	err := sqliteschema.CheckCompatible(t.Context(), db)
	if err == nil || errors.Is(err, sqliteschema.ErrIncompatible) {
		t.Errorf("CheckCompatible() on a closed database = %v, want an error that is not ErrIncompatible", err)
	}
}
