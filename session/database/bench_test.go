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
	"fmt"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/session"
)

// BenchmarkSQLiteGet loads a whole session of n events from a file-backed
// SQLite database, which every turn on the database backend starts with.
func BenchmarkSQLiteGet(b *testing.B) {
	for _, n := range []int{100, 1000} {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			db, err := gorm.Open(sqlite.Open(b.TempDir()+"/bench.db"), &gorm.Config{Logger: logger.Discard})
			if err != nil {
				b.Fatal(err)
			}
			svc, err := NewSessionServiceFromDB(db)
			if err != nil {
				b.Fatal(err)
			}
			if err := AutoMigrate(svc); err != nil {
				b.Fatal(err)
			}
			if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", n); err != nil {
				b.Fatal(err)
			}
			req := &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s"}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := svc.Get(b.Context(), req); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
