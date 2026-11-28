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
	"database/sql"
	"database/sql/driver"
	"errors"
	"slices"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

var errTestActiveBatch = errors.New("DDL batch is already active")

type migrationBatchState struct {
	active, dml, failed  bool
	inspectedActiveBatch bool
	attempts, aborts     int
	err, abortErr        error
	events               []string
}

type migrationBatchConnector struct {
	driver     driver.Driver
	state      *migrationBatchState
	introspect bool
}

func (c migrationBatchConnector) Connect(context.Context) (driver.Conn, error) {
	conn, err := c.driver.Open(":memory:")
	if err != nil || !c.introspect {
		return conn, err
	}
	return migrationBatchConn{Conn: conn, state: c.state}, nil
}

func (c migrationBatchConnector) Driver() driver.Driver {
	return c.driver
}

type migrationBatchConn struct {
	driver.Conn
	state *migrationBatchState
}

func (c migrationBatchConn) InDDLBatch() bool { return c.state.active && !c.state.dml }
func (c migrationBatchConn) InDMLBatch() bool { return c.state.active && c.state.dml }

type batchDialector struct {
	gorm.Dialector
	state  *migrationBatchState
	before migrationHook
	opaque bool
}

func (d batchDialector) Migrator(db *gorm.DB) gorm.Migrator {
	m := batchMigrator{
		interleavingMigrator: interleavingMigrator{Migrator: d.Dialector.Migrator(db), before: d.before},
		state:                d.state,
	}
	if d.opaque {
		return opaqueBatchMigrator{m}
	}
	return m
}

type opaqueBatchMigrator struct{ gorm.Migrator }

func (m opaqueBatchMigrator) AbortBatch() error {
	return m.Migrator.(interface{ AbortBatch() error }).AbortBatch()
}

type batchMigrator struct {
	interleavingMigrator
	state *migrationBatchState
}

// Model Spanner's START BATCH -> migrate -> RUN BATCH lifecycle: an error
// during migration leaves the batch active until AbortBatch is called.
func (m batchMigrator) AutoMigrate(values ...any) error {
	m.state.attempts++
	m.state.events = append(m.state.events, "start")
	if m.state.active {
		return errTestActiveBatch
	}
	m.state.active, m.state.failed = true, false
	err := m.state.err
	if err == nil {
		err = m.interleavingMigrator.AutoMigrate(values...)
	}
	if err != nil {
		m.state.failed = true
		m.state.events = append(m.state.events, "failure")
		return err
	}
	m.state.active = false
	m.state.events = append(m.state.events, "run")
	return nil
}

func (m batchMigrator) AbortBatch() error {
	m.state.aborts++
	m.state.events = append(m.state.events, "abort")
	if m.state.abortErr != nil {
		return m.state.abortErr
	}
	m.state.active = false
	return nil
}

func (m batchMigrator) HasIndex(value any, name string) bool {
	if m.state.failed {
		m.state.events = append(m.state.events, "inspect")
		m.state.inspectedActiveBatch = m.state.inspectedActiveBatch || m.state.active
	}
	return m.interleavingMigrator.HasIndex(value, name)
}

func TestAutoMigrateBatchCleanup(t *testing.T) {
	ddlErr := errors.New("synthetic DDL failure")
	abortErr := errors.New("synthetic abort failure")
	for _, tc := range []struct {
		name                  string
		missingIndex, compete bool
		opaque                bool
		migrationErr, hookErr error
		abortErr, wantErr     error
		wantAttempts, aborts  int
		wantActive            bool
	}{
		{name: "successful batch", wantAttempts: 1},
		{name: "retry after cleanup", missingIndex: true, compete: true, wantAttempts: 2, aborts: 1},
		{name: "unresolved DDL", missingIndex: true, migrationErr: ddlErr, wantErr: ddlErr, wantAttempts: 1, aborts: 1},
		{name: "no pending objects", migrationErr: ddlErr, wantErr: ddlErr, wantAttempts: 1, aborts: 1},
		{name: "abort failure preserves DDL error", missingIndex: true, compete: true, hookErr: ddlErr, abortErr: abortErr, wantErr: ddlErr, wantAttempts: 1, aborts: 1, wantActive: true},
		{name: "opaque migrator", opaque: true, migrationErr: ddlErr, wantErr: ddlErr, wantAttempts: 1, wantActive: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := privateService(t, nil)
			peer := migrationPeer(t, s.db)
			if tc.missingIndex {
				if err := s.db.Migrator().DropIndex(&storageEvent{}, eventsIndex); err != nil {
					t.Fatalf("preparing missing index: %v", err)
				}
			}
			state := &migrationBatchState{err: tc.migrationErr, abortErr: tc.abortErr}
			db := s.db.Session(&gorm.Session{})
			db.Dialector = batchDialector{Dialector: db.Dialector, state: state, opaque: tc.opaque, before: func(kind string, value any, name string) error {
				if kind != "index" || !tc.compete {
					return nil
				}
				if peer.Migrator().HasIndex(value, name) {
					t.Fatal("competing index already exists")
				}
				if err := peer.Migrator().CreateIndex(value, name); err != nil {
					t.Fatalf("competing index creation: %v", err)
				}
				return tc.hookErr
			}}
			err := AutoMigrate(&databaseService{db: db})
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("AutoMigrate() error = %v, want original migration error %v", err, tc.wantErr)
			}
			if state.attempts != tc.wantAttempts || state.aborts != tc.aborts {
				t.Errorf("attempts/aborts = %d/%d, want %d/%d", state.attempts, state.aborts, tc.wantAttempts, tc.aborts)
			}
			if state.active != tc.wantActive {
				t.Errorf("active batch = %v, want %v", state.active, tc.wantActive)
			}
			if state.inspectedActiveBatch {
				t.Error("inspected schema before aborting the failed batch")
			}
			if tc.name == "retry after cleanup" {
				want := []string{"start", "failure", "abort", "inspect", "start", "run"}
				if !slices.Equal(state.events, want) {
					t.Errorf("batch lifecycle = %v, want %v", state.events, want)
				}
			}
		})
	}
}

func TestAutoMigratePinnedBatch(t *testing.T) {
	ddlErr := errors.New("synthetic pinned DDL failure")
	for _, tc := range []struct {
		name                        string
		active, dml, opaque, closed bool
		wantErr                     error
		wantAttempts, wantAborts    int
	}{
		{name: "caller DDL batch", active: true, wantErr: errTestActiveBatch, wantAttempts: 1},
		{name: "caller DML batch", active: true, dml: true, wantErr: errTestActiveBatch, wantAttempts: 1},
		{name: "owned batch", wantErr: ddlErr, wantAttempts: 1, wantAborts: 1},
		{name: "driver without batch introspection", opaque: true, wantErr: ddlErr, wantAttempts: 1, wantAborts: 1},
		{name: "closed connection", closed: true, wantErr: sql.ErrConnDone},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := privateService(t, nil)
			native, err := base.db.DB()
			if err != nil {
				t.Fatal(err)
			}
			state := &migrationBatchState{}
			pool := sql.OpenDB(migrationBatchConnector{driver: native.Driver(), state: state, introspect: !tc.opaque})
			t.Cleanup(func() { _ = pool.Close() })
			conn, err := pool.Conn(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = conn.Close() })
			db, err := gorm.Open(sqlite.Dialector{Conn: conn}, &gorm.Config{Logger: logger.Discard})
			if err != nil {
				t.Fatal(err)
			}
			service, err := NewSessionServiceFromDB(db)
			if err != nil {
				t.Fatal(err)
			}
			if err := AutoMigrate(service); err != nil {
				t.Fatal(err)
			}
			db.Dialector = batchDialector{Dialector: db.Dialector, state: state, before: func(string, any, string) error { return nil }}
			state.active, state.dml, state.err = tc.active, tc.dml, ddlErr
			if tc.closed {
				if err := conn.Close(); err != nil {
					t.Fatal(err)
				}
			}
			err = AutoMigrate(service)
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("AutoMigrate() error = %v, want %v", err, tc.wantErr)
			}
			if state.attempts != tc.wantAttempts || state.aborts != tc.wantAborts {
				t.Errorf("attempts/aborts = %d/%d, want %d/%d", state.attempts, state.aborts, tc.wantAttempts, tc.wantAborts)
			}
			if state.active != tc.active {
				t.Errorf("caller batch active = %v, want %v", state.active, tc.active)
			}
		})
	}
}
