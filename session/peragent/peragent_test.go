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

package peragent_test

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/glebarez/sqlite"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
	"google.golang.org/adk/v2/session/peragent"
	"google.golang.org/adk/v2/session/sessiontestsuite"
	"google.golang.org/adk/v2/tool/toolauth"
)

// newService returns a per-agent service over cfg and closes it when the test
// ends.
func newService(t *testing.T, cfg peragent.Config) session.Service {
	t.Helper()
	svc, err := peragent.New(cfg)
	if err != nil {
		t.Fatalf("New(%+v) error = %v", cfg, err)
	}
	t.Cleanup(func() {
		if err := svc.(io.Closer).Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
	return svc
}

// mustNotExist fails the test unless path is absent.
func mustNotExist(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("os.Stat(%q) error = %v, want fs.ErrNotExist", path, err)
	}
}

// mustExist fails the test unless path exists.
func mustExist(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Errorf("os.Stat(%q) error = %v, want the file to exist", path, err)
	}
}

// openFDs counts the file descriptors of this process whose target is path,
// or lies under path when prefix is true. It skips the test where
// /proc/self/fd is absent.
func openFDs(t *testing.T, path string, prefix bool) int {
	t.Helper()
	entries, err := os.ReadDir("/proc/self/fd")
	if err != nil {
		t.Skipf("cannot count open files: %v", err)
	}
	n := 0
	for _, e := range entries {
		target, err := os.Readlink(filepath.Join("/proc/self/fd", e.Name()))
		if err != nil {
			continue
		}
		if target == path || prefix && strings.HasPrefix(target, path+string(filepath.Separator)) {
			n++
		}
	}
	return n
}

// realDir returns t.TempDir() with its symlinks evaluated, so it can be
// compared with the targets in /proc/self/fd.
func realDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestService_PassesSharedSessionContract(t *testing.T) {
	opts := sessiontestsuite.SuiteOptions{SupportsUserProvidedSessionID: true, ChecksRequestedAuthConfigs: true}
	sessiontestsuite.RunServiceTests(t, opts, func(t *testing.T) session.Service {
		return newService(t, peragent.Config{AgentsRoot: t.TempDir()})
	})
}

func TestNew_EmptyAgentsRootFails(t *testing.T) {
	for _, root := range []string{"", "  "} {
		svc, err := peragent.New(peragent.Config{AgentsRoot: root})
		if err == nil || svc != nil {
			t.Errorf("New(Config{AgentsRoot: %q}) = %v, %v; want nil and an error", root, svc, err)
		}
	}
}

func TestNew_CopiesAppNameToDir(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	dirs := map[string]string{"app": "dir_a"}
	svc := newService(t, peragent.Config{AgentsRoot: root, AppNameToDir: dirs})
	dirs["app"] = "dir_b"

	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	mustExist(t, filepath.Join(root, "dir_a", ".adk", "session.db"))
	mustNotExist(t, filepath.Join(root, "dir_b"))
}

func TestPerAgentService_DotToSeparatorAndFirstUseCreate(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	svc := newService(t, peragent.Config{AgentsRoot: root})
	dbPath := filepath.Join(root, "multi_agent", "hello_world_ma", ".adk", "session.db")
	mustNotExist(t, dbPath)

	_, err := svc.Create(ctx, &session.CreateRequest{AppName: "multi_agent.hello_world_ma", UserID: "user_a", SessionID: "s1", State: map[string]any{"k": "v"}})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	mustExist(t, dbPath)
	mustNotExist(t, filepath.Join(root, "multi_agent.hello_world_ma"))
	resp, err := svc.Get(ctx, &session.GetRequest{AppName: "multi_agent.hello_world_ma", UserID: "user_a", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got := resp.Session.ID(); got != "s1" {
		t.Errorf("Session.ID() = %q, want %q", got, "s1")
	}
	if got := resp.Session.AppName(); got != "multi_agent.hello_world_ma" {
		t.Errorf("Session.AppName() = %q, want %q", got, "multi_agent.hello_world_ma")
	}
	if got, err := resp.Session.State().Get("k"); err != nil || got != "v" {
		t.Errorf("State().Get(%q) = %v, %v; want %q", "k", got, err, "v")
	}
}

func TestPerAgentService_AppNameToDirAlias(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	svc := newService(t, peragent.Config{AgentsRoot: root, AppNameToDir: map[string]string{"alias_one": "shared_dir", "alias_two": "shared_dir"}})

	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "alias_one", UserID: "u1", SessionID: "s1"}); err != nil {
		t.Fatalf("Create(alias_one) error = %v", err)
	}
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "alias_two", UserID: "u1", SessionID: "s2"}); err != nil {
		t.Fatalf("Create(alias_two) error = %v", err)
	}

	mustNotExist(t, filepath.Join(root, "alias_one"))
	mustNotExist(t, filepath.Join(root, "alias_two"))
	file, err := database.New(database.Config{Dialector: sqlite.Open(filepath.Join(root, "shared_dir", ".adk", "session.db"))})
	if err != nil {
		t.Fatalf("database.New() error = %v", err)
	}
	t.Cleanup(func() { _ = file.(io.Closer).Close() })
	for app, want := range map[string]string{"alias_one": "s1", "alias_two": "s2"} {
		resp, err := file.List(ctx, &session.ListRequest{AppName: app})
		if err != nil {
			t.Fatalf("List(%q) on shared_dir/.adk/session.db error = %v", app, err)
		}
		if len(resp.Sessions) != 1 || resp.Sessions[0].ID() != want {
			t.Errorf("List(%q) on shared_dir/.adk/session.db = %d sessions, want only %q", app, len(resp.Sessions), want)
		}
	}
}

func TestPerAgentService_AliasSharesPool(t *testing.T) {
	ctx := t.Context()
	root := realDir(t)
	svc := newService(t, peragent.Config{AgentsRoot: root, AppNameToDir: map[string]string{"alias_one": "shared_dir", "alias_two": "shared_dir"}})

	for _, app := range []string{"alias_one", "alias_two"} {
		if _, err := svc.Create(ctx, &session.CreateRequest{AppName: app, UserID: "u1"}); err != nil {
			t.Fatalf("Create(%q) error = %v", app, err)
		}
	}

	if got := openFDs(t, filepath.Join(root, "shared_dir", ".adk", "session.db"), false); got != 1 {
		t.Errorf("open file descriptors on shared_dir/.adk/session.db = %d, want 1", got)
	}
}

func TestPerAgentService_BuiltInApps(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	svc := newService(t, peragent.Config{AgentsRoot: root, AppNameToDir: map[string]string{"__builtin": "ignored_dir"}})

	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "__builtin", UserID: "u1", SessionID: "s1"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	mustExist(t, filepath.Join(root, ".adk", "session.db"))
	mustNotExist(t, filepath.Join(root, "__builtin"))
	mustNotExist(t, filepath.Join(root, "ignored_dir"))
}

func TestPerAgentService_PathValidation(t *testing.T) {
	ctx := t.Context()
	root := filepath.Join(t.TempDir(), "agents")
	svc := newService(t, peragent.Config{AgentsRoot: root, AppNameToDir: map[string]string{"mapped": "../outside"}})

	tests := []struct {
		appName string
		target  string
	}{
		{appName: "../escape_attempt", target: "../escape_attempt"},
		{appName: "../agents_evil", target: "../agents_evil"},
		{appName: "..", target: ".."},
		{appName: "/etc/passwd", target: "/etc/passwd"},
		{appName: "mapped", target: "../outside"},
	}
	for _, tc := range tests {
		want := "Invalid app_name '" + tc.target + "': resolves outside base directory"
		_, err := svc.Create(ctx, &session.CreateRequest{AppName: tc.appName, UserID: "u1"})
		if !errors.Is(err, peragent.ErrInvalidAppName) || !strings.Contains(fmt.Sprint(err), want) {
			t.Errorf("Create(AppName: %q) error = %v, want ErrInvalidAppName with %q", tc.appName, err, want)
		}
		_, err = session.GetUserState(ctx, svc, &session.GetUserStateRequest{AppName: tc.appName, UserID: "u1"})
		if !errors.Is(err, peragent.ErrInvalidAppName) || !strings.Contains(fmt.Sprint(err), want) {
			t.Errorf("GetUserState(AppName: %q) error = %v, want ErrInvalidAppName with %q", tc.appName, err, want)
		}
	}
	mustNotExist(t, filepath.Join(filepath.Dir(root), "escape_attempt"))
	mustNotExist(t, filepath.Join(filepath.Dir(root), "agents_evil"))
	mustNotExist(t, filepath.Join(filepath.Dir(root), "outside"))

	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "a..b", UserID: "u1", SessionID: "s1"}); err != nil {
		t.Errorf("Create(AppName: %q) error = %v", "a..b", err)
	}
	mustExist(t, filepath.Join(root, "a", "b", ".adk", "session.db"))
	resp, err := session.GetUserState(ctx, svc, &session.GetUserStateRequest{AppName: "", UserID: "u1"})
	if err != nil || resp.State == nil || len(resp.State) != 0 {
		t.Errorf("GetUserState(AppName: \"\") = %v, %v; want an empty, non-nil state", resp, err)
	}
}

func TestService_SymlinkEscapeRejected(t *testing.T) {
	ctx := t.Context()
	root := filepath.Join(t.TempDir(), "agents")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	// A sibling whose path starts with the root's path: a containment check
	// by string prefix lets it through.
	other := root + "_evil"
	if err := os.MkdirAll(other, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(other, filepath.Join(root, "agent")); err != nil {
		t.Skipf("cannot create a symlink: %v", err)
	}
	svc := newService(t, peragent.Config{AgentsRoot: root})

	_, err := svc.Create(ctx, &session.CreateRequest{AppName: "agent", UserID: "u", SessionID: "s1"})

	want := "Invalid app_name 'agent': resolves outside base directory"
	if !errors.Is(err, peragent.ErrInvalidAppName) || !strings.Contains(fmt.Sprint(err), want) {
		t.Errorf("Create(AppName: %q) error = %v, want ErrInvalidAppName with %q", "agent", err, want)
	}
	mustNotExist(t, filepath.Join(other, ".adk"))
}

func TestService_SymlinkedAgentsRoot(t *testing.T) {
	ctx := t.Context()
	real := t.TempDir()
	root := filepath.Join(t.TempDir(), "agents")
	if err := os.Symlink(real, root); err != nil {
		t.Skipf("cannot create a symlink: %v", err)
	}
	svc := newService(t, peragent.Config{AgentsRoot: root})

	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s1"}); err != nil {
		t.Fatalf("Create() under a symlinked agents root error = %v", err)
	}
	mustExist(t, filepath.Join(real, "app", ".adk", "session.db"))
}

// An agents root whose path holds characters that a SQLite DSN or URI gives
// a meaning ("?", "#", "%") still gets its database at
// <root>/<app>/.adk/session.db, and no file is written outside it.
func TestPerAgentService_AgentsRootWithURICharacters(t *testing.T) {
	ctx := t.Context()
	parent := t.TempDir()
	root := filepath.Join(parent, "a?b#c%3Fd")
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	svc := newService(t, peragent.Config{AgentsRoot: root})

	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s1"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	mustExist(t, filepath.Join(root, "app", ".adk", "session.db"))
	mustNotExist(t, filepath.Join(parent, "a"))
	mustNotExist(t, filepath.Join(parent, "a?b"))
}

func TestPerAgentService_CloseAndReopen(t *testing.T) {
	ctx := t.Context()
	svc := newService(t, peragent.Config{AgentsRoot: t.TempDir()})
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app_a", UserID: "u1", SessionID: "s1"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	if err := svc.(io.Closer).Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	resp, err := svc.Get(ctx, &session.GetRequest{AppName: "app_a", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get() after Close() error = %v", err)
	}
	if got := resp.Session.ID(); got != "s1" {
		t.Errorf("Get() after Close(): Session.ID() = %q, want %q", got, "s1")
	}
}

func TestService_Close_ReleasesDatabaseFiles(t *testing.T) {
	ctx := t.Context()
	root := realDir(t)
	svc, err := peragent.New(peragent.Config{AgentsRoot: root})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	for app, id := range map[string]string{"a": "s1", "b": "s2"} {
		if _, err := svc.Create(ctx, &session.CreateRequest{AppName: app, UserID: "u1", SessionID: id}); err != nil {
			t.Fatalf("Create(%q) error = %v", app, err)
		}
	}
	if got := openFDs(t, root, true); got < 2 {
		t.Fatalf("open file descriptors under the agents root before Close() = %d, want at least 2", got)
	}

	if err := svc.(io.Closer).Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	if got := openFDs(t, root, true); got != 0 {
		t.Errorf("open file descriptors under the agents root after Close() = %d, want 0", got)
	}
}

// pythonSchema is the DDL of adk-python's SqliteSessionService
// (sessions/sqlite_session_service.py), with one session row.
const pythonSchema = `
CREATE TABLE sessions (app_name TEXT NOT NULL, user_id TEXT NOT NULL, id TEXT NOT NULL, state TEXT NOT NULL, create_time REAL NOT NULL, update_time REAL NOT NULL, PRIMARY KEY (app_name, user_id, id));
CREATE TABLE events (id TEXT NOT NULL, app_name TEXT NOT NULL, user_id TEXT NOT NULL, session_id TEXT NOT NULL, invocation_id TEXT NOT NULL, timestamp REAL NOT NULL, event_data TEXT NOT NULL, PRIMARY KEY (app_name, user_id, session_id, id));
INSERT INTO sessions VALUES ('py_app', 'u1', 's1', '{"k":"v"}', 1000.0, 1000.0);
`

func TestPerAgentService_IncompatiblePythonSchema(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	dbPath := filepath.Join(root, "py_app", ".adk", "session.db")
	if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
		t.Fatal(err)
	}
	py, err := sql.Open(sqlite.DriverName, dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := py.Exec(pythonSchema); err != nil {
		t.Fatal(err)
	}
	if err := py.Close(); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	svc := newService(t, peragent.Config{AgentsRoot: root})

	_, createErr := svc.Create(ctx, &session.CreateRequest{AppName: "py_app", UserID: "u1", SessionID: "s2"})
	_, getErr := svc.Get(ctx, &session.GetRequest{AppName: "py_app", UserID: "u1", SessionID: "s1"})

	if !errors.Is(createErr, peragent.ErrIncompatibleSchema) {
		t.Errorf("Create() error = %v, want ErrIncompatibleSchema", createErr)
	}
	if !errors.Is(getErr, peragent.ErrIncompatibleSchema) {
		t.Errorf("Get() error = %v, want ErrIncompatibleSchema", getErr)
	}
	after, err := os.ReadFile(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Errorf("py_app/.adk/session.db changed: %d bytes before, %d after", len(before), len(after))
	}
}

func TestService_ConcurrentFirstUseOpensOneDatabase(t *testing.T) {
	ctx := t.Context()
	root := realDir(t)
	svc := newService(t, peragent.Config{AgentsRoot: root})

	const goroutines = 16
	start := make(chan struct{})
	errs := make([]error, goroutines)
	var wg sync.WaitGroup
	for i := range goroutines {
		wg.Go(func() {
			<-start
			_, errs[i] = svc.Create(ctx, &session.CreateRequest{AppName: "shared_app", UserID: "u1", SessionID: fmt.Sprintf("s-%d", i)})
		})
	}
	close(start)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("Create(s-%d) error = %v", i, err)
		}
	}
	resp, err := svc.List(ctx, &session.ListRequest{AppName: "shared_app", UserID: "u1"})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if got := len(resp.Sessions); got != goroutines {
		t.Errorf("List() returned %d sessions, want %d", got, goroutines)
	}
	entries, err := os.ReadDir(filepath.Join(root, "shared_app", ".adk"))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "session.db" {
		t.Errorf("shared_app/.adk holds %v, want only session.db", entries)
	}
	if got := openFDs(t, filepath.Join(root, "shared_app", ".adk", "session.db"), false); got != 1 {
		t.Errorf("open file descriptors on shared_app/.adk/session.db = %d, want 1", got)
	}
}

// createAndAppend creates session id in app for user u and appends the events
// e0 to e4, one second apart.
func createAndAppend(ctx context.Context, svc session.Service, app, id string) error {
	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: app, UserID: "u", SessionID: id})
	if err != nil {
		return err
	}
	base := time.Date(2024, time.January, 1, 0, 0, 0, 0, time.UTC)
	for j := range 5 {
		ev := &session.Event{ID: fmt.Sprintf("e%d", j), Timestamp: base.Add(time.Duration(j) * time.Second)}
		if err := svc.AppendEvent(ctx, resp.Session, ev); err != nil {
			return fmt.Errorf("AppendEvent(%s, e%d): %w", id, j, err)
		}
	}
	return nil
}

func TestService_ConcurrentUseOfOneApp(t *testing.T) {
	ctx := t.Context()
	svc := newService(t, peragent.Config{AgentsRoot: t.TempDir()})

	const goroutines = 16
	start := make(chan struct{})
	errs := make([]error, goroutines)
	var wg sync.WaitGroup
	for i := range goroutines {
		wg.Go(func() {
			<-start
			errs[i] = createAndAppend(ctx, svc, "app", fmt.Sprintf("s%d", i))
		})
	}
	close(start)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("goroutine %d: %v", i, err)
		}
	}
	resp, err := svc.List(ctx, &session.ListRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if got := len(resp.Sessions); got != goroutines {
		t.Errorf("List() returned %d sessions, want %d", got, goroutines)
	}
	for i := range goroutines {
		id := fmt.Sprintf("s%d", i)
		got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: id})
		if err != nil {
			t.Errorf("Get(%s) error = %v", id, err)
			continue
		}
		if n := got.Session.Events().Len(); n != 5 {
			t.Errorf("Get(%s) returned %d events, want 5", id, n)
		}
	}
}

func TestPerAgentService_BusyTimeout(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	services := []session.Service{
		newService(t, peragent.Config{AgentsRoot: root}),
		newService(t, peragent.Config{AgentsRoot: root}),
	}

	const perService = 8
	start := make(chan struct{})
	errs := make([]error, 2*perService)
	var wg sync.WaitGroup
	for n, svc := range services {
		for i := range perService {
			wg.Go(func() {
				<-start
				errs[n*perService+i] = createAndAppend(ctx, svc, "app", fmt.Sprintf("svc%d-s%d", n, i))
			})
		}
	}
	close(start)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("goroutine %d: %v", i, err)
		}
	}
}

func TestPerAgentService_OpenWaitsForAWriteLockHeldPastTheBusyTimeout(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	dir := filepath.Join(root, "app", ".adk")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Another process holds the write lock of the new database for longer
	// than the five seconds an ordinary call waits.
	holder, err := sql.Open(sqlite.DriverName, filepath.Join(dir, "session.db")+"?_txlock=immediate")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = holder.Close() })
	tx, err := holder.Begin()
	if err != nil {
		t.Fatalf("BEGIN IMMEDIATE error = %v", err)
	}
	const hold = 7 * time.Second
	committed := make(chan error, 1)
	time.AfterFunc(hold, func() { committed <- tx.Commit() })
	svc := newService(t, peragent.Config{AgentsRoot: root})

	start := time.Now()
	_, err = svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"})
	elapsed := time.Since(start)

	if err := <-committed; err != nil {
		t.Fatalf("COMMIT error = %v", err)
	}
	if err != nil {
		t.Fatalf("Create() while another connection held the write lock for %v: error = %v", hold, err)
	}
	if elapsed < 5*time.Second {
		t.Errorf("Create() returned after %v, want it to wait for the lock held for %v", elapsed, hold)
	}
}

func TestPerAgentService_OpenWithCanceledContextDoesNotWaitForTheLock(t *testing.T) {
	// A write lock (IMMEDIATE) blocks the migration; an exclusive lock
	// blocks the schema check's reads as well.
	for _, txlock := range []string{"immediate", "exclusive"} {
		t.Run(txlock, func(t *testing.T) {
			root := t.TempDir()
			dir := filepath.Join(root, "app", ".adk")
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			holder, err := sql.Open(sqlite.DriverName, filepath.Join(dir, "session.db")+"?_txlock="+txlock)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = holder.Close() })
			tx, err := holder.Begin()
			if err != nil {
				t.Fatalf("BEGIN %s error = %v", txlock, err)
			}
			t.Cleanup(func() { _ = tx.Rollback() })
			svc := newService(t, peragent.Config{AgentsRoot: root})
			ctx, cancel := context.WithCancel(t.Context())
			cancel()

			start := time.Now()
			_, err = svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"})
			elapsed := time.Since(start)

			if !errors.Is(err, context.Canceled) {
				t.Errorf("Create() with a canceled context error = %v, want context.Canceled", err)
			}
			if elapsed > time.Second {
				t.Errorf("Create() with a canceled context returned after %v, want under a second", elapsed)
			}
		})
	}
}

func TestPerAgentService_EmptyAppNameRejectedBeforeRouting(t *testing.T) {
	ctx := t.Context()
	root := realDir(t)
	rootDB := filepath.Join(root, ".adk", "session.db")
	seed, err := peragent.New(peragent.Config{AgentsRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := seed.Create(ctx, &session.CreateRequest{AppName: "__builtin", UserID: "u1", SessionID: "s1"}); err != nil {
		t.Fatalf("Create() of a built-in app session error = %v", err)
	}
	if err := seed.(io.Closer).Close(); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		call func(svc session.Service) error
	}{
		{
			name: "Get",
			call: func(svc session.Service) error {
				_, err := svc.Get(ctx, &session.GetRequest{UserID: "u1", SessionID: "s1"})
				return err
			},
		},
		{
			name: "List",
			call: func(svc session.Service) error {
				_, err := svc.List(ctx, &session.ListRequest{AppName: ""})
				return err
			},
		},
		{
			name: "Delete",
			call: func(svc session.Service) error {
				return svc.Delete(ctx, &session.DeleteRequest{UserID: "u1", SessionID: "s1"})
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := newService(t, peragent.Config{AgentsRoot: root})

			err := tc.call(svc)

			if err == nil {
				t.Errorf("%s() with an empty app name succeeded, want an error", tc.name)
			}
			if got := openFDs(t, rootDB, false); got != 0 {
				t.Errorf("open descriptors of %s after %s() = %d, want 0: the request must fail before the database is opened", rootDB, tc.name, got)
			}
		})
	}
}

func TestPerAgentService_AppendEventWithoutDatabaseChecksRequestedAuthConfigs(t *testing.T) {
	ctx := t.Context()
	svc := newService(t, peragent.Config{AgentsRoot: t.TempDir()})
	other, err := session.InMemoryService().Create(ctx, &session.CreateRequest{AppName: "missing_app", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatal(err)
	}
	event := &session.Event{
		ID:     "e1",
		Author: "agent",
		Actions: session.EventActions{
			RequestedAuthConfigs: map[string]toolauth.Config{"c1": {CredentialKey: "k"}},
		},
	}

	err = svc.AppendEvent(ctx, other.Session, event)

	if err == nil || errors.Is(err, session.ErrNotFound) {
		t.Errorf("AppendEvent() of a requested auth config that does not decode: error = %v, want an error that is not session.ErrNotFound", err)
	}
}

func TestPerAgentService_AppendEventWithoutDatabaseAssignsEventID(t *testing.T) {
	ctx := t.Context()
	svc := newService(t, peragent.Config{AgentsRoot: t.TempDir()})
	other, err := session.InMemoryService().Create(ctx, &session.CreateRequest{AppName: "missing_app", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatal(err)
	}
	event := &session.Event{Author: "agent"}

	err = svc.AppendEvent(ctx, other.Session, event)

	if !errors.Is(err, session.ErrNotFound) {
		t.Errorf("AppendEvent() error = %v, want session.ErrNotFound", err)
	}
	if event.ID == "" {
		t.Error("AppendEvent() left the event's ID empty, want an assigned ID")
	}
}

func TestPerAgent_NonCreateOperationsDoNotCreateDotAdkFiles(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	svc := newService(t, peragent.Config{AgentsRoot: root})
	other, err := session.InMemoryService().Create(ctx, &session.CreateRequest{AppName: "missing_app", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := svc.Get(ctx, &session.GetRequest{AppName: "missing_app", UserID: "u1", SessionID: "s1"}); !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get() error = %v, want session.ErrNotFound", err)
	}
	if resp, err := svc.List(ctx, &session.ListRequest{AppName: "missing_app"}); err != nil || resp.Sessions == nil || len(resp.Sessions) != 0 {
		t.Errorf("List() = %v, %v; want an empty, non-nil list", resp, err)
	}
	if err := svc.Delete(ctx, &session.DeleteRequest{AppName: "missing_app", UserID: "u1", SessionID: "s1"}); err != nil {
		t.Errorf("Delete() error = %v", err)
	}
	if resp, err := session.GetUserState(ctx, svc, &session.GetUserStateRequest{AppName: "missing_app", UserID: "u1"}); err != nil || resp.State == nil || len(resp.State) != 0 {
		t.Errorf("GetUserState() = %v, %v; want an empty, non-nil state", resp, err)
	}
	if err := svc.AppendEvent(ctx, other.Session, &session.Event{ID: "e1", Timestamp: time.Now()}); !errors.Is(err, session.ErrNotFound) {
		t.Errorf("AppendEvent() error = %v, want session.ErrNotFound", err)
	}
	if err := svc.AppendEvent(ctx, other.Session, &session.Event{ID: "e2", Timestamp: time.Now(), LLMResponse: model.LLMResponse{Partial: true}}); err != nil {
		t.Errorf("AppendEvent() of a partial event error = %v, want nil", err)
	}

	mustNotExist(t, filepath.Join(root, "missing_app"))
	mustNotExist(t, filepath.Join(root, ".adk"))
}

func TestPerAgentService_InvalidRequestsFailWithoutDatabase(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	svc := newService(t, peragent.Config{AgentsRoot: root})
	other, err := session.InMemoryService().Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app"}); err == nil {
		t.Error("Create() without a user ID succeeded, want an error")
	}
	if _, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u1"}); err == nil || errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get() without a session ID error = %v, want an error that is not session.ErrNotFound", err)
	}
	if _, err := svc.List(ctx, &session.ListRequest{}); err == nil {
		t.Error("List() without an app name succeeded, want an error")
	}
	if err := svc.Delete(ctx, &session.DeleteRequest{AppName: "app", SessionID: "s1"}); err == nil {
		t.Error("Delete() without a user ID succeeded, want an error")
	}
	if _, err := svc.Create(ctx, nil); err == nil {
		t.Error("Create(nil) succeeded, want an error")
	}
	if _, err := svc.Get(ctx, nil); err == nil {
		t.Error("Get(nil) succeeded, want an error")
	}
	if _, err := svc.List(ctx, nil); err == nil {
		t.Error("List(nil) succeeded, want an error")
	}
	if err := svc.Delete(ctx, nil); err == nil {
		t.Error("Delete(nil) succeeded, want an error")
	}
	if _, err := svc.(session.UserStateReader).GetUserState(ctx, nil); err == nil {
		t.Error("GetUserState(nil) succeeded, want an error")
	}
	if err := svc.AppendEvent(ctx, nil, &session.Event{ID: "e1"}); err == nil {
		t.Error("AppendEvent() with a nil session succeeded, want an error")
	}
	if err := svc.AppendEvent(ctx, other.Session, nil); err == nil {
		t.Error("AppendEvent() with a nil event succeeded, want an error")
	}

	mustNotExist(t, filepath.Join(root, "app"))
	mustNotExist(t, filepath.Join(root, ".adk"))
}

func TestPerAgentService_EqualTimestampsKeepAppendOrder(t *testing.T) {
	ctx := t.Context()
	svc := newService(t, peragent.Config{AgentsRoot: t.TempDir()})
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ts := time.Date(2024, time.January, 1, 0, 0, 0, 0, time.UTC)
	for _, id := range []string{"b", "a"} {
		if err := svc.AppendEvent(ctx, created.Session, &session.Event{ID: id, Timestamp: ts}); err != nil {
			t.Fatalf("AppendEvent(%s) error = %v", id, err)
		}
	}

	resp, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var got []string
	for ev := range resp.Session.Events().All() {
		got = append(got, ev.ID)
	}
	if want := []string{"b", "a"}; !slices.Equal(got, want) {
		t.Errorf("event IDs = %v, want %v (append order)", got, want)
	}
}
