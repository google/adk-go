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
	"slices"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"gorm.io/gorm/schema"
)

// migrationHook runs after GORM's existence check, immediately before DDL.
// A separate connection can then win the race without timing-dependent sleeps.
type migrationHook func(kind string, value any, name string) error

type interleavingDialector struct {
	gorm.Dialector
	before migrationHook
}

func (d interleavingDialector) Migrator(db *gorm.DB) gorm.Migrator {
	return interleavingMigrator{Migrator: d.Dialector.Migrator(db), before: d.before}
}

type interleavingMigrator struct {
	gorm.Migrator
	before migrationHook
}

func (m interleavingMigrator) GetQueryAndExecTx() (*gorm.DB, *gorm.DB) {
	return m.Migrator.(migrationQueryProvider).GetQueryAndExecTx()
}

type opaqueDialector struct {
	gorm.Dialector
}

func (d opaqueDialector) Migrator(db *gorm.DB) gorm.Migrator {
	return struct{ gorm.Migrator }{d.Dialector.Migrator(db)}
}

func (m interleavingMigrator) CreateTable(values ...any) error {
	for _, value := range values {
		if err := m.before("table", value, ""); err != nil {
			return err
		}
	}
	return m.Migrator.CreateTable(values...)
}

func (m interleavingMigrator) AddColumn(value any, name string) error {
	if err := m.before("column", value, name); err != nil {
		return err
	}
	return m.Migrator.AddColumn(value, name)
}

func (m interleavingMigrator) CreateIndex(value any, name string) error {
	if err := m.before("index", value, name); err != nil {
		return err
	}
	return m.Migrator.CreateIndex(value, name)
}

func (m interleavingMigrator) CreateConstraint(value any, name string) error {
	if err := m.before("constraint", value, name); err != nil {
		return err
	}
	return m.Migrator.CreateConstraint(value, name)
}

func (m interleavingMigrator) HasColumn(value any, name string) bool {
	_ = m.before("column check", value, name)
	return m.Migrator.HasColumn(value, name)
}

func (m interleavingMigrator) HasConstraint(value any, name string) bool {
	_ = m.before("constraint check", value, name)
	return m.Migrator.HasConstraint(value, name)
}

func migrationPeer(t *testing.T, db *gorm.DB) *gorm.DB {
	t.Helper()
	var file string
	var seq int
	var name string
	if err := db.Raw("PRAGMA database_list").Row().Scan(&seq, &name, &file); err != nil {
		t.Fatalf("reading database file: %v", err)
	}
	peer, err := gorm.Open(sqlite.Open(file), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("opening independent connection: %v", err)
	}
	t.Cleanup(func() {
		sqlDB, err := peer.DB()
		if err != nil {
			t.Errorf("peer.DB() error = %v", err)
			return
		}
		if err := sqlDB.Close(); err != nil {
			t.Errorf("closing independent connection: %v", err)
		}
	})
	return peer
}

func withMigrationHook(s *databaseService, before migrationHook) *databaseService {
	db := s.db.Session(&gorm.Session{})
	db.Dialector = interleavingDialector{Dialector: db.Dialector, before: before}
	return &databaseService{db: db}
}

func TestAutoMigrateConcurrentCreation(t *testing.T) {
	for _, kind := range []string{"table", "column", "index"} {
		t.Run(kind, func(t *testing.T) {
			s := privateService(t, nil)
			peer := migrationPeer(t, s.db)
			var prepareErr error
			switch kind {
			case "table":
				prepareErr = s.db.Migrator().DropTable(&storageEvent{})
			case "column":
				prepareErr = s.db.Migrator().DropColumn(&storageEvent{}, "NodeInfoJSON")
			case "index":
				prepareErr = s.db.Migrator().DropIndex(&storageEvent{}, eventsIndex)
			}
			if prepareErr != nil {
				t.Fatalf("preparing missing %s: %v", kind, prepareErr)
			}
			// This table follows events in the migration. Recovering from the
			// collision must still finish the rest of the schema.
			if err := s.db.Migrator().DropTable(&storageUserState{}); err != nil {
				t.Fatalf("preparing later table: %v", err)
			}
			calls := 0
			racing := withMigrationHook(s, func(gotKind string, value any, name string) error {
				if _, ok := value.(*storageEvent); !ok || gotKind != kind {
					return nil
				}
				calls++
				if calls > 1 {
					t.Fatal("retried creation instead of refreshing schema metadata")
				}
				var err error
				switch kind {
				case "table":
					if peer.Migrator().HasTable(value) {
						t.Fatal("competing table already exists")
					}
					err = peer.Migrator().CreateTable(value)
				case "column":
					if peer.Migrator().HasColumn(value, name) {
						t.Fatal("competing column already exists")
					}
					err = peer.Migrator().AddColumn(value, name)
				case "index":
					if peer.Migrator().HasIndex(value, name) {
						t.Fatal("competing index already exists")
					}
					err = peer.Migrator().CreateIndex(value, name)
				}
				if err != nil {
					t.Fatalf("competing migration: %v", err)
				}
				return nil
			})
			if err := AutoMigrate(racing); err != nil {
				t.Fatalf("AutoMigrate() after concurrent %s creation: %v", kind, err)
			}
			if calls != 1 {
				t.Errorf("competing creations = %d, want 1", calls)
			}
			if !peer.Migrator().HasTable(&storageUserState{}) ||
				!peer.Migrator().HasColumn(&storageEvent{}, "NodeInfoJSON") ||
				!peer.Migrator().HasIndex(&storageEvent{}, eventsIndex) {
				t.Error("AutoMigrate() returned success with an incomplete schema")
			}
		})
	}
}

func TestAutoMigrateReturnsUnresolvedError(t *testing.T) {
	s := privateService(t, nil)
	if err := s.db.Migrator().DropIndex(&storageEvent{}, eventsIndex); err != nil {
		t.Fatalf("preparing missing index: %v", err)
	}
	want := errors.New("DDL denied")
	calls := 0
	failing := withMigrationHook(s, func(kind string, _ any, _ string) error {
		if kind == "index" {
			calls++
			return want
		}
		return nil
	})
	if err := AutoMigrate(failing); !errors.Is(err, want) {
		t.Errorf("AutoMigrate() error = %v, want original DDL error", err)
	}
	if calls != 1 {
		t.Errorf("failed creation calls = %d, want 1 without schema progress", calls)
	}
}

func TestAutoMigrateReturnsLaterErrorAfterConcurrentCreation(t *testing.T) {
	s := privateService(t, nil)
	peer := migrationPeer(t, s.db)
	if err := s.db.Migrator().DropIndex(&storageEvent{}, eventsIndex); err != nil {
		t.Fatalf("preparing missing index: %v", err)
	}
	if err := s.db.Migrator().DropTable(&storageUserState{}); err != nil {
		t.Fatalf("preparing later table: %v", err)
	}
	want := errors.New("later DDL denied")
	indexCalls, tableCalls := 0, 0
	failing := withMigrationHook(s, func(kind string, value any, name string) error {
		if kind == "index" {
			indexCalls++
			if peer.Migrator().HasIndex(value, name) {
				t.Fatal("competing index already exists")
			}
			if err := peer.Migrator().CreateIndex(value, name); err != nil {
				t.Fatalf("competing index creation: %v", err)
			}
		}
		if _, ok := value.(*storageUserState); ok && kind == "table" {
			tableCalls++
			return want
		}
		return nil
	})
	if err := AutoMigrate(failing); !errors.Is(err, want) {
		t.Errorf("AutoMigrate() error = %v, want later DDL error", err)
	}
	if indexCalls != 1 || tableCalls != 1 {
		t.Errorf("creation calls = (%d, %d), want (1, 1)", indexCalls, tableCalls)
	}
}

func TestAutoMigrateCanceledContext(t *testing.T) {
	s := privateService(t, nil)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	s.db = s.db.WithContext(ctx)
	if err := AutoMigrate(s); !errors.Is(err, context.Canceled) {
		t.Errorf("AutoMigrate() error = %v, want context.Canceled", err)
	}
}

func TestAutoMigrateDryRun(t *testing.T) {
	makeDry := func(db *gorm.DB) *gorm.DB {
		return db.Session(&gorm.Session{DryRun: true})
	}
	for _, tc := range []struct {
		name      string
		configure func(*gorm.DB) *gorm.DB
	}{
		{name: "direct", configure: makeDry},
		{name: "scope", configure: func(db *gorm.DB) *gorm.DB { return db.Scopes(makeDry) }},
		{name: "scope setting config", configure: func(db *gorm.DB) *gorm.DB {
			return db.Scopes(func(tx *gorm.DB) *gorm.DB {
				tx.DryRun = true
				return tx
			})
		}},
		{name: "nested scope", configure: func(db *gorm.DB) *gorm.DB {
			return db.Scopes(func(tx *gorm.DB) *gorm.DB { return tx.Scopes(makeDry) })
		}},
		{name: "scope replacing DB", configure: func(db *gorm.DB) *gorm.DB {
			return db.Scopes(func(*gorm.DB) *gorm.DB { return makeDry(db) })
		}},
		{name: "custom migrator", configure: func(db *gorm.DB) *gorm.DB {
			db = makeDry(db)
			db.Dialector = opaqueDialector{Dialector: db.Dialector}
			return db
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := privateService(t, nil)
			if err := s.db.Migrator().DropIndex(&storageEvent{}, eventsIndex); err != nil {
				t.Fatalf("preparing missing index: %v", err)
			}
			db := tc.configure(s.db)
			wantDryRun := db.DryRun
			dry, err := NewSessionServiceFromDB(db)
			if err != nil {
				t.Fatalf("NewSessionServiceFromDB() error = %v", err)
			}
			if err := AutoMigrate(dry); err != nil {
				t.Fatalf("AutoMigrate() in DryRun mode: %v", err)
			}
			if db.DryRun != wantDryRun {
				t.Error("AutoMigrate() changed the caller's DryRun setting")
			}
			if s.db.Migrator().HasIndex(&storageEvent{}, eventsIndex) {
				t.Error("dry-run migration created an index")
			}
		})
	}
}

func TestAutoMigrateScopedNamingStrategy(t *testing.T) {
	db, err := gorm.Open(sqlite.Open(t.TempDir()+"/sessions.db"), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("gorm.Open() error = %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("db.DB() error = %v", err)
	}
	t.Cleanup(func() {
		if err := sqlDB.Close(); err != nil {
			t.Errorf("sqlDB.Close() error = %v", err)
		}
	})
	scoped := db.Scopes(func(tx *gorm.DB) *gorm.DB {
		tx = tx.Session(&gorm.Session{})
		tx.NamingStrategy = schema.NamingStrategy{NoLowerCase: true}
		return tx
	})
	svc, err := NewSessionServiceFromDB(scoped)
	if err != nil {
		t.Fatalf("NewSessionServiceFromDB() error = %v", err)
	}
	if err := AutoMigrate(svc); err != nil {
		t.Fatalf("AutoMigrate() with scoped naming: %v", err)
	}
	var columns []string
	if err := db.Raw("SELECT name FROM pragma_table_info('events')").Scan(&columns).Error; err != nil {
		t.Fatalf("reading column names: %v", err)
	}
	if !slices.Contains(columns, "AppName") || slices.Contains(columns, "app_name") {
		t.Error("metadata inspection replaced the scope's column naming")
	}
}

func TestAutoMigrateResolvesScopesOnce(t *testing.T) {
	s := privateService(t, nil)
	calls := 0
	s.db = s.db.Scopes(func(db *gorm.DB) *gorm.DB {
		calls++
		return db
	})
	if err := AutoMigrate(s); err != nil {
		t.Fatalf("AutoMigrate() with a scope: %v", err)
	}
	if calls != 1 {
		t.Errorf("scope calls = %d, want 1", calls)
	}
}

func TestAutoMigrateSuccessiveConcurrentAdditions(t *testing.T) {
	s := privateService(t, nil)
	peer := migrationPeer(t, s.db)
	for _, name := range []string{"NodeInfoJSON", "RequestedInputJSON", "CacheMetadata"} {
		if err := s.db.Migrator().DropColumn(&storageEvent{}, name); err != nil {
			t.Fatalf("preparing missing column: %v", err)
		}
	}
	calls := 0
	racing := withMigrationHook(s, func(kind string, value any, name string) error {
		if kind != "column" {
			return nil
		}
		calls++
		if peer.Migrator().HasColumn(value, name) {
			t.Fatal("competing column already exists")
		}
		if err := peer.Migrator().AddColumn(value, name); err != nil {
			t.Fatalf("competing column creation: %v", err)
		}
		return nil
	})
	if err := AutoMigrate(racing); err != nil {
		t.Fatalf("AutoMigrate() after successive concurrent additions: %v", err)
	}
	if calls != 3 {
		t.Errorf("competing column creations = %d, want 3", calls)
	}
	for _, name := range []string{"NodeInfoJSON", "RequestedInputJSON", "CacheMetadata"} {
		if !peer.Migrator().HasColumn(&storageEvent{}, name) {
			t.Error("AutoMigrate() returned success with a missing column")
		}
	}
}

func TestAutoMigrateConcurrentForeignKey(t *testing.T) {
	s := privateService(t, nil)
	peer := migrationPeer(t, s.db)
	const name = "fk_sessions_events"
	if err := s.db.Migrator().DropConstraint(&storageEvent{}, name); err != nil {
		t.Fatalf("preparing missing foreign key: %v", err)
	}
	calls := 0
	racing := withMigrationHook(s, func(kind string, value any, name string) error {
		if kind != "constraint" {
			return nil
		}
		calls++
		if peer.Migrator().HasConstraint(value, name) {
			t.Fatal("competing constraint already exists")
		}
		if err := peer.Migrator().CreateConstraint(value, name); err != nil {
			t.Fatalf("competing constraint creation: %v", err)
		}
		// SQLite rebuilds the table instead of rejecting a duplicate foreign
		// key. Model the DDL failure other databases return after a race.
		return errors.New("concurrent constraint creation")
	})
	if err := AutoMigrate(racing); err != nil {
		t.Fatalf("AutoMigrate() after concurrent foreign key creation: %v", err)
	}
	if calls != 1 {
		t.Errorf("competing constraint creations = %d, want 1", calls)
	}
}

func TestAutoMigrateDisabledConstraints(t *testing.T) {
	for _, option := range []string{"foreign keys", "relationships"} {
		t.Run(option, func(t *testing.T) {
			s := privateService(t, nil)
			s.db = s.db.Session(&gorm.Session{})
			s.db.DisableForeignKeyConstraintWhenMigrating = option == "foreign keys"
			s.db.IgnoreRelationshipsWhenMigrating = option == "relationships"
			s = withMigrationHook(s, func(kind string, _ any, _ string) error {
				if kind == "constraint check" {
					t.Error("inspected a constraint disabled by the caller")
				}
				return nil
			})
			if err := AutoMigrate(s); err != nil {
				t.Fatalf("AutoMigrate() with disabled constraints: %v", err)
			}
		})
	}
}

func TestAutoMigrateConstraintOwner(t *testing.T) {
	s := withMigrationHook(privateService(t, nil), func(kind string, value any, _ string) error {
		if _, ok := value.(*storageSession); ok && kind == "constraint check" {
			t.Error("inspected an events foreign key on the sessions table")
		}
		return nil
	})
	if err := AutoMigrate(s); err != nil {
		t.Fatalf("AutoMigrate() with related models: %v", err)
	}
}

type migrationParent struct {
	ID int
}

type migrationIgnoredFields struct {
	ID       int
	Ignored  string `gorm:"-:migration"`
	ParentID int
	Parent   migrationParent `gorm:"-:migration"`
}

type migrationNoConstraint struct {
	ID       int
	ParentID int
	Parent   migrationParent `gorm:"constraint:-"`
}

func TestAutoMigrateIgnoredFields(t *testing.T) {
	s := withMigrationHook(privateService(t, nil), func(kind string, _ any, name string) error {
		if kind == "column check" && name == "ignored" {
			t.Error("inspected a column excluded from migration")
		}
		if kind == "constraint check" {
			t.Error("inspected an ignored relationship constraint")
		}
		return nil
	})
	if err := autoMigrate(s.db, &migrationIgnoredFields{}, &migrationNoConstraint{}); err != nil {
		t.Fatalf("autoMigrate() with ignored fields: %v", err)
	}
}

type migrationCheckedValue struct {
	ID    int
	Value int `gorm:"check:positive_value,value > 0"`
}

func TestAutoMigrateConcurrentCheckConstraint(t *testing.T) {
	s := privateService(t, nil)
	peer := migrationPeer(t, s.db)
	model := &migrationCheckedValue{}
	if err := s.db.AutoMigrate(model); err != nil {
		t.Fatalf("preparing table: %v", err)
	}
	if err := s.db.Migrator().DropConstraint(model, "positive_value"); err != nil {
		t.Fatalf("preparing missing check constraint: %v", err)
	}
	s = withMigrationHook(s, func(kind string, value any, name string) error {
		if kind == "constraint" {
			if peer.Migrator().HasConstraint(value, name) {
				t.Fatal("competing constraint already exists")
			}
			if err := peer.Migrator().CreateConstraint(value, name); err != nil {
				t.Fatalf("competing check constraint creation: %v", err)
			}
			return errors.New("concurrent check constraint creation")
		}
		return nil
	})
	if err := autoMigrate(s.db, model); err != nil {
		t.Fatalf("autoMigrate() after concurrent check constraint creation: %v", err)
	}
}

func TestAutoMigrateInvalidModel(t *testing.T) {
	s := privateService(t, nil)
	if err := autoMigrate(s.db, 123); !errors.Is(err, schema.ErrUnsupportedDataType) {
		t.Errorf("autoMigrate() error = %v, want schema.ErrUnsupportedDataType", err)
	}
}
