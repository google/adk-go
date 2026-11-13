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

package database

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/glebarez/sqlite"

	"google.golang.org/adk/v2/internal/sessionutils/sqliteschema"
	"google.golang.org/adk/v2/session"
)

// newFromURL calls NewSessionServiceFromURL and closes the service when t
// ends.
func newFromURL(t *testing.T, dbURL string) session.Service {
	t.Helper()
	svc, err := NewSessionServiceFromURL(dbURL)
	if err != nil {
		t.Fatalf("NewSessionServiceFromURL(%q) error = %v", dbURL, err)
	}
	t.Cleanup(func() { _ = svc.(io.Closer).Close() })
	return svc
}

// fileURL is the four-slash SQLite URL of the absolute path p.
func fileURL(p string) string {
	return "sqlite:////" + strings.TrimPrefix(filepath.ToSlash(p), "/")
}

func TestNewSessionServiceFromURL_InMemoryIsolation(t *testing.T) {
	ctx := t.Context()
	getReq := &session.GetRequest{AppName: "app", UserID: "u1", SessionID: "s1"}
	for _, u := range []string{
		"sqlite://", "sqlite:///", "sqlite:///:memory:",
		"sqlite+aiosqlite://", "sqlite+aiosqlite:///", "sqlite+aiosqlite:///:memory:",
		"sqlite3://", "sqlite3:///", "sqlite3:///:memory:",
	} {
		t.Run(u, func(t *testing.T) {
			svc1 := newFromURL(t, u)
			svc2 := newFromURL(t, u)
			if _, err := svc1.Create(ctx, &session.CreateRequest{
				AppName: "app", UserID: "u1", SessionID: "s1",
				State: map[string]any{"user:k": "v1"},
			}); err != nil {
				t.Fatalf("svc1.Create() error = %v", err)
			}

			got, err := svc1.Get(ctx, getReq)
			if err != nil {
				t.Fatalf("svc1.Get() error = %v", err)
			}
			if v, _ := got.Session.State().Get("user:k"); got.Session.ID() != "s1" || v != "v1" {
				t.Errorf("svc1.Get() = session %q with user:k %v, want %q with %q", got.Session.ID(), v, "s1", "v1")
			}
			if _, err := svc2.Get(ctx, getReq); !errors.Is(err, session.ErrNotFound) {
				t.Errorf("svc2.Get() error = %v, want session.ErrNotFound", err)
			}
			list, err := svc2.List(ctx, &session.ListRequest{AppName: "app"})
			if err != nil {
				t.Fatalf("svc2.List() error = %v", err)
			}
			if len(list.Sessions) != 0 {
				t.Errorf("svc2.List() returned %d sessions, want 0", len(list.Sessions))
			}
		})
	}
}

// The service keeps one connection, which serializes concurrent calls
// instead of failing them with "database table is locked".
func TestNewSessionServiceFromURL_InMemoryConcurrentUse(t *testing.T) {
	svc := newFromURL(t, "sqlite:///:memory:")
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	const goroutines, events = 16, 5
	base := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

	start := make(chan struct{})
	errs := make(chan error, goroutines*(events+1))
	var wg sync.WaitGroup
	for i := range goroutines {
		wg.Go(func() {
			<-start
			resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: fmt.Sprintf("s%d", i)})
			if err != nil {
				errs <- fmt.Errorf("Create(s%d): %w", i, err)
				return
			}
			for j := range events {
				ev := &session.Event{ID: fmt.Sprintf("e%d", j), Author: "user", Timestamp: base.Add(time.Duration(j) * time.Second)}
				if err := svc.AppendEvent(ctx, resp.Session, ev); err != nil {
					errs <- fmt.Errorf("AppendEvent(s%d, e%d): %w", i, j, err)
				}
			}
		})
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	list, err := svc.List(ctx, &session.ListRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(list.Sessions) != goroutines {
		t.Errorf("List() returned %d sessions, want %d", len(list.Sessions), goroutines)
	}
	for i := range goroutines {
		got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: fmt.Sprintf("s%d", i)})
		if err != nil {
			t.Errorf("Get(s%d) error = %v", i, err)
			continue
		}
		if n := got.Session.Events().Len(); n != events {
			t.Errorf("Get(s%d) returned %d events, want %d", i, n, events)
		}
	}
}

// A file URL opens the file at its path, three slashes for a relative path
// and four for an absolute one, and the tables exist without an AutoMigrate
// call.
func TestNewSessionServiceFromURL_SQLiteFile(t *testing.T) {
	ctx := t.Context()
	create := &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"}

	absPath := filepath.Join(t.TempDir(), "abs.db")
	abs := newFromURL(t, fileURL(absPath))
	if _, err := abs.Create(ctx, create); err != nil {
		t.Errorf("Create() on %q error = %v", fileURL(absPath), err)
	}
	if _, err := os.Stat(absPath); err != nil {
		t.Errorf("os.Stat(%q) error = %v, want the database file", absPath, err)
	}

	// The path is percent-decoded, as adk-python's SqliteSessionService
	// decodes it, and the decoded "?" and "#" stay part of the file name.
	escParent := t.TempDir()
	escDir := filepath.Join(escParent, "a?b")
	if err := os.Mkdir(escDir, 0o755); err != nil {
		t.Fatal(err)
	}
	escURL := fileURL(escParent) + "/a%3Fb/c%23d%253F.db"
	esc := newFromURL(t, escURL)
	if _, err := esc.Create(ctx, create); err != nil {
		t.Errorf("Create() on %q error = %v", escURL, err)
	}
	if _, err := os.Stat(filepath.Join(escDir, "c#d%3F.db")); err != nil {
		t.Errorf("os.Stat(a?b/c#d%%3F.db) error = %v, want the database file", err)
	}
	if _, err := os.Stat(filepath.Join(escParent, "a")); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("os.Stat(a) error = %v, want fs.ErrNotExist", err)
	}

	// Five slashes leave the absolute path "//<path>", which names the same
	// file as "/<path>".
	fivePath := filepath.Join(t.TempDir(), "five.db")
	five := newFromURL(t, "sqlite:///"+fileURL(fivePath)[len("sqlite://"):])
	if _, err := five.Create(ctx, create); err != nil {
		t.Errorf("Create() on a five-slash URL error = %v", err)
	}
	if _, err := os.Stat(fivePath); err != nil {
		t.Errorf("os.Stat(%q) error = %v, want the database file", fivePath, err)
	}

	t.Chdir(t.TempDir())
	if err := os.Mkdir("sub", 0o755); err != nil {
		t.Fatal(err)
	}
	rel := newFromURL(t, "sqlite:///sub/rel.db")
	if _, err := rel.Create(ctx, create); err != nil {
		t.Errorf("Create() on sqlite:///sub/rel.db error = %v", err)
	}
	if _, err := os.Stat(filepath.Join("sub", "rel.db")); err != nil {
		t.Errorf("os.Stat(sub/rel.db) error = %v, want the database file", err)
	}
}

func TestNewSessionServiceFromURL_InvalidURL(t *testing.T) {
	// A URL wrongly accepted opens its file here, not in the package.
	t.Chdir(t.TempDir())
	for _, tc := range []struct {
		name, dbURL, wantMsg string
		secrets              []string
	}{
		{
			name:    "no scheme",
			dbURL:   "definitely not a url sup3r-s3cret",
			wantMsg: "Invalid database URL format or argument '<unparseable database URL>'.",
			secrets: []string{"sup3r-s3cret"},
		},
		{
			name:    "sqlite URL with credentials and a host",
			dbURL:   "sqlite://admin:sup3r-s3cret@remotehost:5432/path.db?api_key=tok123&mode=ro",
			wantMsg: "Invalid database URL format or argument 'sqlite://admin:***@remotehost:5432/path.db?api_key=REDACTED&mode=REDACTED'.",
			secrets: []string{"sup3r-s3cret", "tok123"},
		},
		{
			name:    "empty",
			dbURL:   "",
			wantMsg: "Invalid database URL format or argument '<unparseable database URL>'.",
		},
		{
			name:    "bare in-memory name",
			dbURL:   ":memory:",
			wantMsg: "Invalid database URL format or argument '<unparseable database URL>'.",
		},
		{
			name:    "query that does not decode",
			dbURL:   "sqlite:///x.db?token=%zz",
			wantMsg: "Invalid database URL format or argument '<unparseable database URL>'.",
		},
		{
			name:    "in-memory URL that sets the cache",
			dbURL:   "sqlite:///:memory:?cache=private",
			wantMsg: "Invalid database URL format or argument 'sqlite:///:memory:?cache=REDACTED'.",
		},
		{
			name:    "in-memory URL that sets the mode",
			dbURL:   "sqlite://?mode=rw",
			wantMsg: "Invalid database URL format or argument 'sqlite://?mode=REDACTED'.",
		},
		{
			name:    "in-memory URL with an unknown _txlock",
			dbURL:   "sqlite:///:memory:?_txlock=sup3r-s3cret",
			wantMsg: "Invalid database URL format or argument 'sqlite:///:memory:?_txlock=REDACTED'.",
			secrets: []string{"sup3r-s3cret"},
		},
		{
			name:    "file URL with an unknown _txlock",
			dbURL:   "sqlite:///lock.db?_txlock=sup3r-s3cret",
			wantMsg: "Invalid database URL format or argument 'sqlite:///lock.db?_txlock=REDACTED'.",
			secrets: []string{"sup3r-s3cret"},
		},
		{
			name:    "file URL with an unknown _time_format",
			dbURL:   "sqlite:///time.db?_time_format=sup3r-s3cret",
			wantMsg: "Invalid database URL format or argument 'sqlite:///time.db?_time_format=REDACTED'.",
			secrets: []string{"sup3r-s3cret"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc, err := NewSessionServiceFromURL(tc.dbURL)

			if svc != nil {
				t.Errorf("NewSessionServiceFromURL(%q) returned a service, want nil", tc.dbURL)
			}
			if !errors.Is(err, ErrInvalidDatabaseURL) {
				t.Fatalf("NewSessionServiceFromURL(%q) error = %v, want ErrInvalidDatabaseURL", tc.dbURL, err)
			}
			if got := err.Error(); got != tc.wantMsg {
				t.Errorf("error message = %q, want %q", got, tc.wantMsg)
			}
			cause := errors.Unwrap(err)
			if _, ok := errors.AsType[*url.Error](cause); !ok {
				t.Errorf("errors.Unwrap(err) = %v (%T), want a *url.Error", cause, cause)
			}
			for _, secret := range tc.secrets {
				if strings.Contains(err.Error(), secret) {
					t.Errorf("error message %q contains %q", err.Error(), secret)
				}
				if cause != nil && strings.Contains(cause.Error(), secret) {
					t.Errorf("cause message %q contains %q", cause.Error(), secret)
				}
			}
			if entries, err := os.ReadDir("."); err != nil || len(entries) != 0 {
				t.Errorf("working directory after NewSessionServiceFromURL(%q) = %v, %v; want it empty, with no database file opened", tc.dbURL, entries, err)
			}
		})
	}
}

func TestNewSessionServiceFromURL_AcceptsDriverArguments(t *testing.T) {
	for _, dbURL := range []string{
		"sqlite:///:memory:?_txlock=deferred",
		"sqlite:///:memory:?_txlock=Immediate",
		"sqlite:///:memory:?_txlock=EXCLUSIVE",
		"sqlite:///:memory:?_time_format=sqlite",
	} {
		t.Run(dbURL, func(t *testing.T) {
			svc := newFromURL(t, dbURL)

			if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"}); err != nil {
				t.Errorf("Create() on %q error = %v", dbURL, err)
			}
		})
	}
}

func TestNewSessionServiceFromURL_UnsupportedScheme(t *testing.T) {
	dbURL := "postgresql+asyncpg://user:secret@localhost:5432/db?sslmode=require&token=priv"

	svc, err := NewSessionServiceFromURL(dbURL)

	if svc != nil {
		t.Error("NewSessionServiceFromURL() returned a service, want nil")
	}
	if !errors.Is(err, ErrInvalidDatabaseURL) || !errors.Is(err, errors.ErrUnsupported) {
		t.Fatalf("NewSessionServiceFromURL() error = %v, want ErrInvalidDatabaseURL and errors.ErrUnsupported", err)
	}
	if cause := errors.Unwrap(err); cause != errors.ErrUnsupported {
		t.Errorf("errors.Unwrap(err) = %v, want errors.ErrUnsupported", cause)
	}
	msg := err.Error()
	for _, want := range []string{`"postgresql+asyncpg"`, "'postgresql+asyncpg://user:***@localhost:5432/db?sslmode=REDACTED&token=REDACTED'", "database.New"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error message %q does not contain %q", msg, want)
		}
	}
	for _, secret := range []string{"secret", "priv", "require"} {
		if strings.Contains(msg, secret) {
			t.Errorf("error message %q contains %q", msg, secret)
		}
	}
}

// Close disposes of the connection the URL service opened; the file and its
// sessions stay.
func TestNewSessionServiceFromURL_Close_DisposesOwnedConnection(t *testing.T) {
	ctx := t.Context()
	u := fileURL(filepath.Join(t.TempDir(), "owned.db"))
	getReq := &session.GetRequest{AppName: "app", UserID: "u1", SessionID: "s1"}
	svc, err := NewSessionServiceFromURL(u)
	if err != nil {
		t.Fatalf("NewSessionServiceFromURL() error = %v", err)
	}
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	if err := svc.(io.Closer).Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	if _, err := svc.Get(ctx, getReq); err == nil {
		t.Error("Get() after Close() succeeded, want an error from the closed connection")
	}
	reopened := newFromURL(t, u)
	got, err := reopened.Get(ctx, getReq)
	if err != nil {
		t.Fatalf("Get() on a service reopened over the file error = %v", err)
	}
	if got.Session.ID() != "s1" {
		t.Errorf("Get() on the reopened service = session %q, want %q", got.Session.ID(), "s1")
	}
}

// A pragma the URL sets is kept: the service adds its own busy timeout only
// when the URL has none.
func TestNewSessionServiceFromURL_UserPragmaWins(t *testing.T) {
	svc := newFromURL(t, fileURL(filepath.Join(t.TempDir(), "x.db"))+"?_pragma=busy_timeout(10000)")
	var timeout int
	if err := svc.(*databaseService).db.Raw("PRAGMA busy_timeout").Scan(&timeout).Error; err != nil {
		t.Fatalf("PRAGMA busy_timeout error = %v", err)
	}
	if timeout != 10000 {
		t.Errorf("PRAGMA busy_timeout = %d, want 10000", timeout)
	}
}

// An existing file of adk-python's SqliteSessionService schema is rejected
// before AutoMigrate, and is left as it was.
func TestNewSessionServiceFromURL_IncompatibleSchemaLeavesFileUntouched(t *testing.T) {
	path := filepath.Join(t.TempDir(), "py.db")
	raw, err := sql.Open(sqlite.DriverName, path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := raw.Exec(`CREATE TABLE sessions (app_name TEXT NOT NULL, user_id TEXT NOT NULL, id TEXT NOT NULL,
  state TEXT NOT NULL, create_time REAL NOT NULL, update_time REAL NOT NULL, PRIMARY KEY (app_name, user_id, id));
CREATE TABLE events (id TEXT NOT NULL, app_name TEXT NOT NULL, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
  invocation_id TEXT NOT NULL, timestamp REAL NOT NULL, event_data TEXT NOT NULL,
  PRIMARY KEY (app_name, user_id, session_id, id));`); err != nil {
		t.Fatal(err)
	}
	if err := raw.Close(); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	svc, err := NewSessionServiceFromURL(fileURL(path))

	if svc != nil {
		t.Error("NewSessionServiceFromURL() returned a service, want nil")
	}
	if !errors.Is(err, sqliteschema.ErrIncompatible) || errors.Is(err, ErrInvalidDatabaseURL) {
		t.Errorf("NewSessionServiceFromURL() error = %v, want sqliteschema.ErrIncompatible and not ErrInvalidDatabaseURL", err)
	}
	if want := "Failed to create database engine for URL '" + fileURL(path) + "'"; err != nil && err.Error() != want {
		t.Errorf("error message = %q, want %q", err.Error(), want)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Error("the adk-python database file changed")
	}
}
