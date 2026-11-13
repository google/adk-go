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

// Package peragent provides a [session.Service] that keeps each agent's
// sessions in a SQLite database inside that agent's folder, at
// <AgentsRoot>/<agent folder>/.adk/session.db.
//
// It is the Go counterpart of the per-agent session storage that adk-python's
// adk web and adk run use for a directory of agents
// (cli/utils/local_storage.py). Each database is a session/database service,
// so the files hold adk-go's schema and not adk-python's.
package peragent

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/internal/sessionutils"
	"google.golang.org/adk/v2/internal/sessionutils/sqlitedsn"
	"google.golang.org/adk/v2/internal/sessionutils/sqliteschema"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
)

// ErrInvalidAppName reports that an app name resolves outside
// [Config.AgentsRoot].
var ErrInvalidAppName = errors.New("peragent: invalid app name")

// ErrIncompatibleSchema reports that an existing .adk/session.db file uses a
// foreign or incompatible database schema (such as adk-python's session.db
// schema) that cannot be opened or migrated by adk-go without damaging the file.
var ErrIncompatibleSchema = errors.New("peragent: incompatible database schema")

// Config configures a per-agent database [session.Service] created with [New].
//
// The zero Config is not usable: AgentsRoot must be set.
type Config struct {
	// AgentsRoot is the directory that holds the agents' folders. It is
	// required. An app's sessions are stored in
	// <AgentsRoot>/<folder>/.adk/session.db, where folder is the app's
	// entry in AppNameToDir, or the app name itself, with every "." replaced
	// by a path separator: app "multi_agent.hello" uses
	// <AgentsRoot>/multi_agent/hello/.adk/session.db.
	//
	// Apps whose names start with "__", such as ADK's built-in apps, share
	// <AgentsRoot>/.adk/session.db and ignore AppNameToDir.
	AgentsRoot string

	// AppNameToDir maps an app name to its folder under AgentsRoot, for apps
	// whose folder is not named after the app. It is optional. Apps that map
	// to the same folder share one database. New copies the map.
	AppNameToDir map[string]string
}

// builtInPrefix starts the names of apps that share the agents root's own
// database (local_storage.py's _BUILT_IN_SESSION_SERVICE_KEY).
const builtInPrefix = "__"

// New creates a [session.Service] that routes session operations by app name to
// per-agent SQLite database files under cfg.AgentsRoot.
//
// New touches no file. An app's folder, .adk directory and session.db are
// created by the first Create for that app. Get, List, Delete, AppendEvent and
// GetUserState open an existing session.db but never create one: for an app
// with no database they report what an empty database would (Get and
// AppendEvent an error wrapping [session.ErrNotFound], List no sessions,
// Delete nothing to do, GetUserState an empty state).
//
// Every method first resolves the app's folder, following symbolic links, and
// returns an error wrapping [ErrInvalidAppName] when the folder lies outside
// cfg.AgentsRoot. Opening an existing session.db that holds another session
// schema, such as one adk-python wrote, fails with an error wrapping
// [ErrIncompatibleSchema] and leaves the file as it was. Otherwise the
// database is migrated to the current schema when it is opened. A request
// that lacks its app name, or a Get with a negative NumRecentEvents, fails
// before any database is opened, and a call whose context is done before its
// database is open returns the context's error. An empty user ID is the user
// "", as in session/database.
//
// The returned service is safe for concurrent use. Each database is opened
// with a single connection, so calls on one app run one at a time; another
// service or process using the same file waits up to five seconds for its
// lock, and up to thirty seconds when it opens the database. Events with
// equal timestamps are returned in the order they were appended, as in
// adk-python's SqliteSessionService.
//
// The returned service also implements [session.UserStateReader] and
// [io.Closer]. Close closes every database opened so far, even when closing
// one of them fails, and returns the joined errors. Later calls open the
// databases again.
//
// New returns an error when cfg.AgentsRoot is empty or cannot be made
// absolute.
func New(cfg Config) (session.Service, error) {
	if strings.TrimSpace(cfg.AgentsRoot) == "" {
		return nil, errors.New("peragent: Config.AgentsRoot must not be empty")
	}
	root, err := filepath.Abs(cfg.AgentsRoot)
	if err != nil {
		return nil, fmt.Errorf("peragent: resolve Config.AgentsRoot: %w", err)
	}
	return &service{
		agentsRoot:   evalExistingSymlinks(root),
		appNameToDir: maps.Clone(cfg.AppNameToDir),
		routes:       map[string]route{},
	}, nil
}

// service routes each app to the database in its folder.
type service struct {
	// agentsRoot is absolute, with its symbolic links evaluated.
	agentsRoot   string
	appNameToDir map[string]string

	// mu guards routes, and is held while a database is opened so that
	// concurrent first calls for one app open it once.
	mu sync.Mutex
	// routes holds the databases opened so far, keyed by file path, so that
	// apps sharing a folder share one connection.
	routes map[string]route
}

// route is one opened database: the session service and the connection pool
// it runs on. The service does not own the pool (database.New over a
// sqlite.Dialector that carries a connection), so the route closes both.
type route struct {
	svc  session.Service
	pool io.Closer
}

func (r route) close() error {
	var errs []error
	if c, ok := r.svc.(io.Closer); ok {
		errs = append(errs, c.Close())
	}
	if r.pool != nil {
		errs = append(errs, r.pool.Close())
	}
	return errors.Join(errs...)
}

var (
	_ session.UserStateReader = (*service)(nil)
	_ io.Closer               = (*service)(nil)
)

// dbPath returns the database file of appName, mirroring adk-python's
// _resolve_agent_dir (cli/utils/dot_adk_folder.py:22-33).
func (s *service) dbPath(appName string) (string, error) {
	if strings.HasPrefix(appName, builtInPrefix) {
		return filepath.Join(s.agentsRoot, ".adk", "session.db"), nil
	}
	target := appName
	if dir, ok := s.appNameToDir[appName]; ok {
		target = dir
	}
	// Replacing every "." also removes every ".." element, so an escape can
	// only come from an absolute path or a symbolic link.
	sub := strings.ReplaceAll(target, ".", string(filepath.Separator))
	agentDir := sub
	if !filepath.IsAbs(sub) {
		agentDir = filepath.Join(s.agentsRoot, sub)
	}
	agentDir = evalExistingSymlinks(filepath.Clean(agentDir))
	rel, err := filepath.Rel(s.agentsRoot, agentDir)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("%w: Invalid app_name '%s': resolves outside base directory", ErrInvalidAppName, target)
	}
	return filepath.Join(agentDir, ".adk", "session.db"), nil
}

// evalExistingSymlinks evaluates the symbolic links of the longest prefix of
// the absolute path p that can be resolved, and keeps the rest of p as it is,
// as Python's Path.resolve(strict=False) does.
func evalExistingSymlinks(p string) string {
	var rest []string
	for {
		if resolved, err := filepath.EvalSymlinks(p); err == nil {
			return filepath.Join(append([]string{resolved}, rest...)...)
		}
		parent := filepath.Dir(p)
		if parent == p {
			return filepath.Join(append([]string{p}, rest...)...)
		}
		rest = append([]string{filepath.Base(p)}, rest...)
		p = parent
	}
}

// service returns the session service of appName's database. When the
// database file does not exist, it creates the file if create is true, and
// otherwise returns a nil service and no error. ctx bounds the opening of a
// database that is not open yet.
func (s *service) service(ctx context.Context, appName string, create bool) (session.Service, error) {
	path, err := s.dbPath(appName)
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if r, ok := s.routes[path]; ok {
		return r.svc, nil
	}
	if !create {
		if _, err := os.Stat(path); errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		} else if err != nil {
			return nil, fmt.Errorf("peragent: %w", err)
		}
	}
	r, err := open(ctx, path)
	if err != nil {
		return nil, err
	}
	s.routes[path] = r
	return r.svc, nil
}

// open opens the database file at path, creating it and its directory when
// they do not exist, and migrates it after checking that it holds no other
// session schema. The check and the migration stop when ctx is done.
func open(ctx context.Context, path string) (route, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return route{}, fmt.Errorf("peragent: create the .adk directory: %w", err)
	}
	// The migration waits longer for the lock than later calls do: a second
	// service or process that opens the same new file waits while the first
	// one migrates and then writes, and the first one's calls can hold the
	// lock back to back for longer than five seconds.
	if err := migrate(ctx, path); err != nil {
		return route{}, err
	}
	pool, err := openPool(path, 5000)
	if err != nil {
		return route{}, err
	}
	svc, err := database.New(database.Config{
		Dialector:           &sqlite.Dialector{Conn: pool},
		AppendOrderTiebreak: true,
		GORMOptions:         []gorm.Option{&gorm.Config{Logger: logger.Discard}},
	})
	if err != nil {
		return route{}, errors.Join(fmt.Errorf("peragent: open %s: %w", path, err), pool.Close())
	}
	return route{svc: svc, pool: pool}, nil
}

// openPool opens the database file at path with a single connection that
// waits up to busyTimeoutMS milliseconds for a lock.
func openPool(path string, busyTimeoutMS int) (*sql.DB, error) {
	// Transactions begin with BEGIN IMMEDIATE: two connections that both
	// read in a deferred transaction and then write fail with SQLITE_BUSY at
	// once, without waiting out the busy timeout. This matters when another
	// service or process uses the same file.
	dsn := sqlitedsn.File(path, fmt.Sprintf("_pragma=foreign_keys(1)&_pragma=busy_timeout(%d)&_txlock=immediate", busyTimeoutMS))
	pool, err := sql.Open(sqlite.DriverName, dsn)
	if err != nil {
		return nil, fmt.Errorf("peragent: open %s: %w", path, err)
	}
	// SQLite allows one writer per file. One connection makes concurrent
	// calls on the app wait for each other instead of failing with
	// SQLITE_BUSY.
	pool.SetMaxOpenConns(1)
	return pool, nil
}

// migrate checks that the database file at path holds no other session
// schema and migrates it, on a connection of its own that waits up to 30
// seconds for the lock.
func migrate(ctx context.Context, path string) (err error) {
	pool, err := openPool(path, 30000)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, pool.Close()) }()
	if err := sqliteschema.CheckCompatible(ctx, pool); err != nil {
		if errors.Is(err, sqliteschema.ErrIncompatible) {
			return fmt.Errorf("%w: %s: %w", ErrIncompatibleSchema, path, err)
		}
		return fmt.Errorf("peragent: open %s: %w", path, err)
	}
	db, err := gorm.Open(&sqlite.Dialector{Conn: pool}, &gorm.Config{Logger: logger.Discard})
	if err != nil {
		return fmt.Errorf("peragent: open %s: %w", path, err)
	}
	// AutoMigrate checks for each table before it creates it. In one
	// transaction, a second service or process opening the same new file
	// waits instead of failing with "table already exists"; adk-python's
	// SqliteSessionService uses CREATE TABLE IF NOT EXISTS.
	err = db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txSvc, err := database.NewSessionServiceFromDB(tx)
		if err != nil {
			return err
		}
		return database.AutoMigrate(txSvc)
	})
	if err != nil {
		return fmt.Errorf("peragent: migrate %s: %w", path, err)
	}
	return nil
}

func (s *service) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if req == nil {
		return nil, errors.New("create request is nil")
	}
	if _, err := s.dbPath(req.AppName); err != nil {
		return nil, err
	}
	// Checked here as well as in session/database, so that a request that
	// cannot succeed creates no database.
	if req.AppName == "" {
		return nil, errors.New("app_name is required")
	}
	svc, err := s.service(ctx, req.AppName, true)
	if err != nil {
		return nil, err
	}
	return svc.Create(ctx, req)
}

func (s *service) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	if req == nil {
		return nil, errors.New("get request is nil")
	}
	if _, err := s.dbPath(req.AppName); err != nil {
		return nil, err
	}
	// Checked here as well as in session/database, so that a request that
	// cannot succeed opens no database.
	if req.AppName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}
	if err := sessionutils.CheckNumRecentEvents(req.NumRecentEvents); err != nil {
		return nil, err
	}
	svc, err := s.service(ctx, req.AppName, false)
	if err != nil {
		return nil, err
	}
	if svc == nil {
		return nil, fmt.Errorf("%w: %q", session.ErrNotFound, req.SessionID)
	}
	return svc.Get(ctx, req)
}

func (s *service) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	if req == nil {
		return nil, errors.New("list request is nil")
	}
	if _, err := s.dbPath(req.AppName); err != nil {
		return nil, err
	}
	// Checked here as well as in session/database, so that a request that
	// cannot succeed opens no database.
	if req.AppName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}
	svc, err := s.service(ctx, req.AppName, false)
	if err != nil {
		return nil, err
	}
	if svc == nil {
		return &session.ListResponse{Sessions: []session.Session{}}, nil
	}
	return svc.List(ctx, req)
}

func (s *service) Delete(ctx context.Context, req *session.DeleteRequest) error {
	if req == nil {
		return errors.New("delete request is nil")
	}
	if _, err := s.dbPath(req.AppName); err != nil {
		return err
	}
	// Checked here as well as in session/database, so that a request that
	// cannot succeed opens no database.
	if req.AppName == "" {
		return fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}
	svc, err := s.service(ctx, req.AppName, false)
	if err != nil {
		return err
	}
	if svc == nil {
		return nil
	}
	return svc.Delete(ctx, req)
}

func (s *service) AppendEvent(ctx context.Context, curSession session.Session, event *session.Event) error {
	if curSession == nil {
		return errors.New("session is nil")
	}
	if event == nil {
		return errors.New("event is nil")
	}
	svc, err := s.service(ctx, curSession.AppName(), false)
	if err != nil {
		return err
	}
	if svc != nil {
		return svc.AppendEvent(ctx, curSession, event)
	}
	// Partial events are never stored, so the session's absence does not
	// matter for them.
	if event.Partial {
		return nil
	}
	// session/database checks the request and gives the event an ID before
	// it looks the session up; an app without a database does the same.
	if err := sessionutils.CheckRequestedAuthConfigs(event.Actions.RequestedAuthConfigs); err != nil {
		return fmt.Errorf("cannot append event: %w", err)
	}
	if event.ID == "" {
		event.ID = platform.NewUUID(ctx)
	}
	return fmt.Errorf("%w: %q, cannot apply event", session.ErrNotFound, curSession.ID())
}

func (s *service) GetUserState(ctx context.Context, req *session.GetUserStateRequest) (*session.GetUserStateResponse, error) {
	if req == nil {
		return nil, errors.New("get user state request is nil")
	}
	svc, err := s.service(ctx, req.AppName, false)
	if err != nil {
		return nil, err
	}
	if svc == nil {
		return &session.GetUserStateResponse{State: map[string]any{}}, nil
	}
	return session.GetUserState(ctx, svc, req)
}

// Close closes every database opened so far and empties the routing table,
// so that later calls open the databases again. It closes them all even when
// one fails, and returns the joined errors. adk-python stops at the first
// failing close and keeps the table (cli/utils/local_storage.py:240-245).
func (s *service) Close() error {
	s.mu.Lock()
	routes := s.routes
	s.routes = map[string]route{}
	s.mu.Unlock()
	var errs []error
	for _, r := range routes {
		errs = append(errs, r.close())
	}
	return errors.Join(errs...)
}
