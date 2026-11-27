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
	"slices"

	"gorm.io/gorm"
)

type migrationQueryProvider interface {
	GetQueryAndExecTx() (queryTx, execTx *gorm.DB)
}

func autoMigrate(db *gorm.DB, models ...any) error {
	migrator := db.Session(&gorm.Session{}).Migrator()
	checks, err := migrationChecks(migrator, models...)
	if err != nil {
		return err
	}
	missing := slices.DeleteFunc(checks, func(exists func() bool) bool { return exists() })
	for {
		err := migrator.AutoMigrate(models...)
		if err == nil {
			return nil
		}
		before := len(missing)
		missing = slices.DeleteFunc(missing, func(exists func() bool) bool { return exists() })
		if len(missing) == before {
			return err
		}
		// Like adk-python's index creation, inspect the schema after a DDL
		// failure. GORM also races when creating tables, columns and foreign
		// keys, so retry the whole migration with fresh metadata. Each retry
		// requires another missing object to appear, bounding the loop without
		// matching dialect-specific error messages or hiding persistent errors.
	}
}

// migrationChecks describes objects GORM may create, using the caller's naming
// strategy. An existing index is accepted by name, including an operator's
// pre-created index with a different definition (see the package documentation).
func migrationChecks(migrator gorm.Migrator, models ...any) ([]func() bool, error) {
	// Resolve scopes before using GORM's own metadata query session, which
	// reads real metadata even in DryRun mode without changing the caller.
	provider, ok := migrator.(migrationQueryProvider)
	if !ok {
		// Custom migrators need not expose GORM's query session. Preserve their
		// original migration path rather than inspect through the wrong DB.
		return nil, nil
	}
	queryDB, _ := provider.GetQueryAndExecTx()
	m := queryDB.Migrator()
	var checks []func() bool
	for _, model := range models {
		stmt := &gorm.Statement{DB: queryDB}
		if err := stmt.Parse(model); err != nil {
			return nil, err
		}
		checks = append(checks, func() bool { return m.HasTable(model) })
		for _, name := range stmt.Schema.DBNames {
			if !stmt.Schema.FieldsByDBName[name].IgnoreMigration {
				checks = append(checks, func() bool { return m.HasColumn(model, name) })
			}
		}
		for _, index := range stmt.Schema.ParseIndexes() {
			checks = append(checks, func() bool { return m.HasIndex(model, index.Name) })
		}
		if !queryDB.DisableForeignKeyConstraintWhenMigrating && !queryDB.IgnoreRelationshipsWhenMigrating {
			for _, rel := range stmt.Schema.Relationships.Relations {
				if rel.Field.IgnoreMigration {
					continue
				}
				if c := rel.ParseConstraint(); c != nil && c.Schema == stmt.Schema {
					checks = append(checks, func() bool { return m.HasConstraint(model, c.Name) })
				}
			}
		}
		for _, c := range stmt.Schema.ParseCheckConstraints() {
			checks = append(checks, func() bool { return m.HasConstraint(model, c.Name) })
		}
	}
	return checks, nil
}
