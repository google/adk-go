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
	"fmt"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/session"
)

var benchSizes = []int{10, 100, 1000}

func benchService(b *testing.B) session.Service {
	b.Helper()
	// A file-backed database, not :memory:, so the cost includes the fsync and
	// page I/O a deployment pays. Logger off so the benchmark measures the
	// service rather than the log sink.
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
	return svc
}

func BenchmarkSQLiteGet(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			svc := benchService(b)
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

// BenchmarkSQLiteGetRecent is the same read limited to the last 20 events,
// the shape an existence check or a tail read could use.
func BenchmarkSQLiteGetRecent(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			svc := benchService(b)
			if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", n); err != nil {
				b.Fatal(err)
			}
			req := &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s", NumRecentEvents: 20}
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

func BenchmarkSQLiteAppendEvent(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			svc := benchService(b)
			sess, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", n)
			if err != nil {
				b.Fatal(err)
			}
			evs := benchutil.BenchEvents(2, "agent", time.Unix(1_800_000_000, 0))
			b.ReportAllocs()
			b.ResetTimer()
			i := 0
			for b.Loop() {
				ev := *evs[i%2]
				ev.ID = ""
				// Monotonic timestamps: the service rejects a stale session.
				ev.Timestamp = time.Unix(1_800_000_000+int64(i), 0)
				if err := svc.AppendEvent(b.Context(), sess, &ev); err != nil {
					b.Fatal(err)
				}
				i++
			}
		})
	}
}

// BenchmarkSQLiteEventDecode isolates the per-event JSON decode Get does for
// each row, so its share of a full Get can be read off directly.
func BenchmarkSQLiteEventDecode(b *testing.B) {
	svc := benchService(b)
	sess, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", 100)
	if err != nil {
		b.Fatal(err)
	}
	var rows []storageEvent
	if err := svc.(*databaseService).db.Find(&rows).Error; err != nil {
		b.Fatal(err)
	}
	if len(rows) != sess.Events().Len() {
		b.Fatalf("got %d rows, want %d", len(rows), sess.Events().Len())
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		for i := range rows {
			if _, err := createEventFromStorageEvent(&rows[i]); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkSQLiteGetCrowded reads one 100-event session while other sessions
// hold the rest of the events table. The events primary key leads with the
// event id, so a lookup by session cannot use it and the cost tracks the size
// of the whole table rather than the session.
func BenchmarkSQLiteGetCrowded(b *testing.B) {
	for _, others := range []int{0, 100, 1000} {
		b.Run(fmt.Sprintf("other-sessions=%d", others), func(b *testing.B) {
			svc := benchService(b)
			for i := range others {
				if _, err := benchutil.SeedSession(b.Context(), svc, "app", fmt.Sprintf("u%d", i), fmt.Sprintf("s%d", i), "agent", 100); err != nil {
					b.Fatal(err)
				}
			}
			if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", 100); err != nil {
				b.Fatal(err)
			}
			for _, recent := range []int{0, 20} {
				b.Run(fmt.Sprintf("recent=%d", recent), func(b *testing.B) {
					req := &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s", NumRecentEvents: recent}
					b.ReportAllocs()
					for b.Loop() {
						if _, err := svc.Get(b.Context(), req); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		})
	}
}
