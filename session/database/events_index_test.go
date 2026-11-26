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
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"github.com/google/go-cmp/cmp"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/session"
)

const eventsIndex = "idx_events_app_user_session_ts_id"

// sqlRecorder is a GORM logger that keeps every statement it traces, with the
// arguments interpolated, so a test can inspect the SQL a method really ran.
type sqlRecorder struct {
	logger.Interface
	mu   sync.Mutex
	sqls []string
}

func (r *sqlRecorder) Trace(_ context.Context, _ time.Time, fc func() (string, int64), _ error) {
	sql, _ := fc()
	r.mu.Lock()
	defer r.mu.Unlock()
	r.sqls = append(r.sqls, sql)
}

func (r *sqlRecorder) take() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	s := r.sqls
	r.sqls = nil
	return s
}

// privateService opens a migrated service on its own database file, unlike
// emptyService, whose shared in-memory database other tests also write to.
func privateService(t *testing.T, rec *sqlRecorder) *databaseService {
	t.Helper()
	cfg := &gorm.Config{Logger: logger.Discard}
	if rec != nil {
		rec.Interface = logger.Discard
		cfg.Logger = rec
	}
	db, err := gorm.Open(sqlite.Open(t.TempDir()+"/sessions.db"), cfg)
	if err != nil {
		t.Fatalf("gorm.Open() error = %v", err)
	}
	t.Cleanup(func() {
		sqlDB, err := db.DB()
		if err != nil {
			t.Errorf("db.DB() error = %v", err)
			return
		}
		if err := sqlDB.Close(); err != nil {
			t.Errorf("sqlDB.Close() error = %v", err)
		}
	})
	svc, err := NewSessionServiceFromDB(db)
	if err != nil {
		t.Fatalf("NewSessionServiceFromDB() error = %v", err)
	}
	if err := AutoMigrate(svc); err != nil {
		t.Fatalf("AutoMigrate() error = %v", err)
	}
	return svc.(*databaseService)
}

// AutoMigrate must create the events index with the name adk-python uses and
// with its columns in Get's filter-then-order sequence. GORM orders index
// columns by struct field position unless told otherwise, which would put id
// first and leave the index useless for a per-session read.
func TestAutoMigrateCreatesEventsIndex(t *testing.T) {
	s := privateService(t, nil)

	type column struct {
		Name string
		Desc bool
	}
	var got []column
	if err := s.db.Raw("SELECT name, \"desc\" FROM pragma_index_xinfo(?) WHERE key = 1 ORDER BY seqno", eventsIndex).Scan(&got).Error; err != nil {
		t.Fatalf("reading index columns: %v", err)
	}

	want := []column{
		{Name: "app_name"},
		{Name: "user_id"},
		{Name: "session_id"},
		{Name: "timestamp", Desc: true},
		{Name: "id", Desc: true},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("columns of %s (-want +got):\n%s", eventsIndex, diff)
	}
}

func eventsIndexColumns(t *testing.T, db *gorm.DB) []string {
	t.Helper()
	var cols []string
	if err := db.Raw("SELECT name FROM pragma_index_info(?) ORDER BY seqno", eventsIndex).Scan(&cols).Error; err != nil {
		t.Fatalf("reading index columns: %v", err)
	}
	return cols
}

// A deployment upgrading from a release without the index already has the
// events table, so AutoMigrate takes its existing-table path. It must add the
// missing index there, and must leave alone an index of that name the
// operator created beforehand, which is what the package doc tells operators
// of large PostgreSQL tables to do.
func TestAutoMigrateAddsEventsIndexToExistingTable(t *testing.T) {
	s := privateService(t, nil)
	if err := s.db.Exec("DROP INDEX " + eventsIndex).Error; err != nil {
		t.Fatalf("DROP INDEX: %v", err)
	}
	if err := AutoMigrate(s); err != nil {
		t.Fatalf("AutoMigrate() on an existing table error = %v", err)
	}
	want := []string{"app_name", "user_id", "session_id", "timestamp", "id"}
	if diff := cmp.Diff(want, eventsIndexColumns(t, s.db)); diff != "" {
		t.Errorf("index added to an existing table (-want +got):\n%s", diff)
	}

	// A pre-created index is recognized by name alone. A deliberately
	// different definition shows AutoMigrate did not drop and rebuild it.
	if err := s.db.Exec("DROP INDEX " + eventsIndex).Error; err != nil {
		t.Fatalf("DROP INDEX: %v", err)
	}
	if err := s.db.Exec("CREATE INDEX " + eventsIndex + " ON events (app_name)").Error; err != nil {
		t.Fatalf("CREATE INDEX: %v", err)
	}
	if err := AutoMigrate(s); err != nil {
		t.Fatalf("AutoMigrate() with a pre-created index error = %v", err)
	}
	if diff := cmp.Diff([]string{"app_name"}, eventsIndexColumns(t, s.db)); diff != "" {
		t.Errorf("AutoMigrate replaced a pre-created index (-want +got):\n%s", diff)
	}
}

// Get's events query must be answered from the events index in index order:
// a SEARCH on the index and no temporary B-tree to sort the rows. The SQL is
// the statement Get actually ran, captured from GORM, so a filter or ORDER BY
// in Get that the index cannot serve fails here.
//
// Only SQLite's plan is checked. PostgreSQL and MySQL plans were not checked
// locally, since neither server is available to the test suite.
func TestGetEventsQueryUsesEventsIndex(t *testing.T) {
	rec := &sqlRecorder{}
	s := privateService(t, rec)
	ctx := t.Context()
	const appName, userID = "app", "user"
	created, err := s.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: "s"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ts := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for range 3 {
		ev := session.NewEvent(ctx, "inv")
		ev.Timestamp = ts
		ts = ts.Add(time.Second)
		if err := s.AppendEvent(ctx, created.Session, ev); err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}
	}

	for _, tc := range []struct {
		name string
		req  session.GetRequest
	}{
		{name: "all events"},
		{name: "recent events", req: session.GetRequest{NumRecentEvents: 2}},
		{name: "events after", req: session.GetRequest{After: ts.Add(-2 * time.Second)}},
		{name: "recent events after", req: session.GetRequest{After: ts.Add(-2 * time.Second), NumRecentEvents: 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := tc.req
			req.AppName, req.UserID, req.SessionID = appName, userID, "s"
			rec.take()
			if _, err := s.Get(ctx, &req); err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			var query string
			for _, sql := range rec.take() {
				if strings.Contains(sql, "FROM `events`") {
					query = sql
				}
			}
			if query == "" {
				t.Fatal("Get() ran no query on the events table")
			}

			var steps []struct{ Detail string }
			if err := s.db.Raw("EXPLAIN QUERY PLAN " + query).Scan(&steps).Error; err != nil {
				t.Fatalf("EXPLAIN QUERY PLAN %s: %v", query, err)
			}
			var details []string
			for _, st := range steps {
				details = append(details, st.Detail)
			}

			plan := strings.Join(details, "\n")
			t.Logf("query: %s\nplan:\n%s", query, plan)
			if !strings.Contains(plan, "SEARCH events USING INDEX "+eventsIndex) {
				t.Errorf("events query does not search %s.\nquery: %s\nplan:\n%s", eventsIndex, query, plan)
			}
			if strings.Contains(plan, "TEMP B-TREE") {
				t.Errorf("events query sorts in a temporary B-tree.\nquery: %s\nplan:\n%s", query, plan)
			}
		})
	}
}
