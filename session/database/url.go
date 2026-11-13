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
	"context"
	"errors"
	"fmt"
	"net/url"
	"slices"
	"strings"

	"github.com/glebarez/sqlite"
	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/internal/sessionutils/sqlitedsn"
	"google.golang.org/adk/v2/internal/sessionutils/sqliteschema"
	"google.golang.org/adk/v2/session"
)

// ErrInvalidDatabaseURL reports that a database URL passed to
// [NewSessionServiceFromURL] is malformed, has invalid driver arguments, or
// specifies an unsupported URL scheme.
var ErrInvalidDatabaseURL = errors.New("invalid database URL")

// NewSessionServiceFromURL creates a [session.Service] backed by the database
// named by dbURL, passing opts to [gorm.Open].
//
// GORM's log output is discarded, as adk-python creates its engine with
// echo=False, unless opts set a logger: a [*gorm.Config] whose Logger is
// non-nil, or an option that sets it. GORM's default logger would write
// statements, with the state and event content they carry, to standard
// output.
//
// The supported URLs use the SQLite driver this module already includes
// (github.com/glebarez/sqlite). The scheme is "sqlite", "sqlite+aiosqlite"
// or "sqlite3"; Go's driver has no separate synchronous and asynchronous
// forms, so all three open the same database:
//
//   - "sqlite://", "sqlite:///" and "sqlite:///:memory:" open a private
//     in-memory database that only the returned service sees; closing the
//     service discards it.
//   - "sqlite:///relative/path.db" opens a file relative to the current
//     working directory, and "sqlite:////absolute/path.db" (four slashes) the
//     file /absolute/path.db.
//
// Query parameters are passed to the driver. The service enables foreign
// keys (PRAGMA foreign_keys = ON) and gives a file a busy timeout of five
// seconds, each unless the URL sets that pragma itself with _pragma. An
// in-memory URL must not set mode or cache, which keep the database private.
//
// The service uses a single database connection, so concurrent calls run
// one at a time. Before it returns, NewSessionServiceFromURL checks that an
// existing file does not hold another session schema, such as adk-python's,
// and runs [AutoMigrate], so the service is ready for use; calling
// AutoMigrate again is a no-op.
//
// For a malformed URL, including "" and ":memory:", for a SQLite URL with a
// host or credentials, for an in-memory URL that sets mode or cache, and for a
// _txlock or _time_format value the driver does not accept, the error matches [ErrInvalidDatabaseURL] and its [errors.Unwrap] cause is a
// [*url.Error]. For any other scheme (such as "postgresql", "mysql" or
// "spanner") the error matches ErrInvalidDatabaseURL and
// [errors.ErrUnsupported], and says to build the service with [New] or
// [NewSessionService] from a [gorm.Dialector] for that database. When the
// database cannot be opened or prepared, the error wraps the cause. No error
// message contains the URL's password or query values: the password is shown
// as "***", each query value as "REDACTED", and a URL that cannot be parsed
// as "<unparseable database URL>".
//
// The returned service also implements [session.UserStateReader] and
// [io.Closer]. Close closes the connection and so drops an in-memory
// database.
func NewSessionServiceFromURL(dbURL string, opts ...gorm.Option) (session.Service, error) {
	dsn, redacted, err := sqliteDSN(dbURL)
	if err != nil {
		return nil, err
	}
	svc, err := New(Config{Dialector: sqlite.Open(dsn), GORMOptions: opts})
	if err != nil {
		return nil, &databaseURLError{msg: engineFailure(redacted), cause: err}
	}
	ds := svc.(*databaseService)
	if err := prepareURLService(ds); err != nil {
		_ = ds.Close()
		return nil, &databaseURLError{msg: engineFailure(redacted), cause: err}
	}
	return ds, nil
}

// discardLogs is a [gorm.Option] that sets [logger.Discard] as the logger
// when no other option has set one. gorm.Open applies every *gorm.Config
// before any other option, so discardLogs sees a logger a caller's
// *gorm.Config sets.
type discardLogs struct{}

func (discardLogs) Apply(c *gorm.Config) error {
	if c.Logger == nil {
		c.Logger = logger.Discard
	}
	return nil
}

func (discardLogs) AfterInitialize(*gorm.DB) error { return nil }

// prepareURLService limits s to one connection, checks that the database
// does not hold another session schema, and creates the tables.
func prepareURLService(s *databaseService) error {
	sqlDB, err := s.db.DB()
	if err != nil {
		return err
	}
	// One connection keeps an in-memory database alive and private to s,
	// as adk-python's StaticPool does, and serializes writers, which SQLite
	// would otherwise fail with "database is locked".
	sqlDB.SetMaxOpenConns(1)
	if err := sqliteschema.CheckCompatible(context.Background(), sqlDB); err != nil {
		return err
	}
	return AutoMigrate(s)
}

// unparseableURL stands for a database URL that cannot be parsed, in error
// messages, as adk-python writes it (sessions/migration/_schema_check_utils.py).
const unparseableURL = "<unparseable database URL>"

// databaseURLError is an error of NewSessionServiceFromURL. Its message is
// msg alone, so that it never shows a cause that may echo the raw URL, such
// as a *url.Error.
type databaseURLError struct {
	msg   string
	cause error
	// invalidURL makes the error match ErrInvalidDatabaseURL.
	invalidURL bool
}

func (e *databaseURLError) Error() string { return e.msg }

func (e *databaseURLError) Unwrap() error { return e.cause }

func (e *databaseURLError) Is(target error) bool {
	return e.invalidURL && target == ErrInvalidDatabaseURL
}

// invalidURL is the error for a URL that is malformed or has an argument the
// service rejects; cause says why.
func invalidURL(redacted string, cause error) error {
	return &databaseURLError{
		msg:        fmt.Sprintf("Invalid database URL format or argument '%s'.", redacted),
		cause:      &url.Error{Op: "parse", URL: redacted, Err: cause},
		invalidURL: true,
	}
}

func engineFailure(redacted string) string {
	return fmt.Sprintf("Failed to create database engine for URL '%s'", redacted)
}

// sqliteSchemes are the URL schemes NewSessionServiceFromURL opens with the
// SQLite driver.
var sqliteSchemes = map[string]bool{"sqlite": true, "sqlite+aiosqlite": true, "sqlite3": true}

// sqliteDSN translates dbURL into the SQLite driver's DSN, and returns the
// redacted URL for error messages.
func sqliteDSN(dbURL string) (dsn, redacted string, err error) {
	u, err := parseStrictURL(dbURL)
	if err != nil {
		return "", "", invalidURL(unparseableURL, err)
	}
	redacted = redactURL(u)
	if !sqliteSchemes[u.Scheme] {
		return "", "", &databaseURLError{
			msg: fmt.Sprintf("unsupported database URL scheme %q in '%s': open a gorm.Dialector for that driver and pass it to database.New or database.NewSessionService",
				u.Scheme, redacted),
			cause:      errors.ErrUnsupported,
			invalidURL: true,
		}
	}
	// SQLite opens a local file: SQLAlchemy's SQLite dialect rejects a host,
	// port or credentials as an invalid argument, and so does the service.
	if u.User != nil || u.Host != "" {
		return "", "", invalidURL(redacted, errors.New("a sqlite URL takes no host or credentials"))
	}
	query := u.Query()
	if err := checkDriverArguments(query); err != nil {
		return "", "", invalidURL(redacted, err)
	}
	if u.Path == "" || u.Path == "/" || u.Path == "/:memory:" {
		for _, key := range []string{"mode", "cache"} {
			if query.Has(key) {
				return "", "", invalidURL(redacted, fmt.Errorf("an in-memory sqlite URL cannot set %s", key))
			}
		}
		addPragma(query, "foreign_keys(1)")
		// A name of its own keeps the database private to the service;
		// cache=shared lets the pool's connections reach it.
		return "file:adk_mem_" + uuid.NewString() + "?mode=memory&cache=shared&" + query.Encode(), redacted, nil
	}
	addPragma(query, "foreign_keys(1)")
	addPragma(query, "busy_timeout(5000)")
	return sqlitedsn.File(strings.TrimPrefix(u.Path, "/"), query.Encode()), redacted, nil
}

// checkDriverArguments rejects a _txlock or _time_format value that the
// SQLite driver refuses when it opens a connection
// (github.com/glebarez/go-sqlite's applyQueryParams), so that such a URL
// fails before a database is opened. The error does not repeat the value.
func checkDriverArguments(query url.Values) error {
	if v := query.Get("_txlock"); v != "" && !slices.Contains([]string{"deferred", "immediate", "exclusive"}, strings.ToLower(v)) {
		return errors.New("_txlock must be deferred, immediate or exclusive")
	}
	if v := query.Get("_time_format"); v != "" && v != "sqlite" {
		return errors.New(`_time_format must be "sqlite"`)
	}
	return nil
}

// addPragma adds the _pragma parameter pragma to query unless query already
// sets a pragma of that name.
func addPragma(query url.Values, pragma string) {
	name, _, _ := strings.Cut(pragma, "(")
	for _, v := range query["_pragma"] {
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(v)), name) {
			return
		}
	}
	query.Add("_pragma", pragma)
}

// parseStrictURL parses a database URL as SQLAlchemy's make_url does: unlike
// url.Parse, it requires "<scheme>://", so a string such as "not a url" or
// ":memory:" is malformed rather than a relative path. A query that does not
// decode is malformed too, rather than losing its bad pairs.
func parseStrictURL(raw string) (*url.URL, error) {
	scheme, _, ok := strings.Cut(raw, "://")
	if !ok || !validScheme(scheme) {
		return nil, errors.New("missing or invalid URL scheme")
	}
	u, err := url.Parse(raw)
	if err != nil {
		// The *url.Error repeats raw, password included; keep only its cause.
		return nil, errors.Unwrap(err)
	}
	if _, err := url.ParseQuery(u.RawQuery); err != nil {
		return nil, err
	}
	return u, nil
}

// validScheme reports whether s is an RFC 3986 scheme.
func validScheme(s string) bool {
	for i, r := range s {
		switch {
		case 'a' <= r && r <= 'z', 'A' <= r && r <= 'Z':
		case i > 0 && ('0' <= r && r <= '9' || r == '+' || r == '-' || r == '.'):
		default:
			return false
		}
	}
	return s != ""
}

// redactURL renders u with its password as "***" and each query value and
// the fragment as "REDACTED", as adk-python's _redact_db_url does.
func redactURL(u *url.URL) string {
	r := *u
	userinfo := ""
	if u.User != nil {
		// url.Userinfo would escape the asterisks.
		userinfo = url.User(u.User.Username()).String()
		if _, ok := u.User.Password(); ok {
			userinfo += ":***"
		}
		userinfo += "@"
		r.User = nil
	}
	query := u.Query()
	for key := range query {
		query[key] = []string{"REDACTED"}
	}
	r.RawQuery = query.Encode()
	if r.Fragment != "" {
		r.Fragment = "REDACTED"
		r.RawFragment = ""
	}
	// url.URL.String drops the "//" of an empty authority and path
	// ("sqlite://?x=1"); the database URL keeps it.
	return r.Scheme + "://" + userinfo + strings.TrimPrefix(strings.TrimPrefix(r.String(), r.Scheme+":"), "//")
}
