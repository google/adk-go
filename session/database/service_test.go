// Copyright 2025 Google LLC
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
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"

	"google.golang.org/adk/session"
	"google.golang.org/adk/session/session_test"
)

func Test_databaseService(t *testing.T) {
	opts := session_test.SuiteOptions{SupportsUserProvidedSessionID: true}
	session_test.RunServiceTests(t, opts, func(t *testing.T) session.Service {
		return emptyService(t)
	})
}

func emptyService(t *testing.T) *databaseService {
	t.Helper()
	gormConfig := &gorm.Config{
		PrepareStmt: true,
	}

	service, err := NewSessionService(sqlite.Open("file::memory:?cache=shared"), gormConfig)
	if err != nil {
		t.Fatalf("Failed to create session service: %v", err)
	}
	dbservice, ok := service.(*databaseService)
	if !ok {
		t.Fatalf("invalid session service type")
	}

	err = AutoMigrate(service)
	if err != nil {
		t.Fatalf("Failed to AutoMigrate db: %v", err)
	}

	t.Cleanup(func() {
		t.Log("CLEANUP: Deleting all rows from tables...")

		// Define models in Child-to-Parent order
		modelsToDelete := []any{
			&storageEvent{}, // Child-most
			&storageSession{},
			&storageUserState{},
			&storageAppState{}, // Parent-most
		}

		for _, model := range modelsToDelete {
			// GORM statement parser to get table names
			stmt := &gorm.Statement{DB: dbservice.db}
			// Parse the model to get its table name
			if err := stmt.Parse(model); err != nil {
				t.Errorf("Failed to parse model schema for cleanup: %v", err)
				continue
			}
			tableName := stmt.Table

			// Exec with "WHERE true" instead of gorm.Delete()
			// satisfies Spanner's requirement for a WHERE clause.
			if err := dbservice.db.Exec(`DELETE FROM ` + tableName + ` WHERE true`).Error; err != nil {
				t.Errorf("Failed to delete from table %s: %v", tableName, err)
			}
		}
		sqlDB, err := dbservice.db.DB()
		if err != nil {
			t.Errorf("Failed to get underlying *sql.DB: %v", err)
			return
		}
		if err := sqlDB.Close(); err != nil {
			t.Errorf("Failed to close database connection: %v", err)
		}
	})

	return dbservice
}
