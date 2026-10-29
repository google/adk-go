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
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/session"
)

var benchSizes = []int{10, 100, 1000}

// countingLogger counts the SQL statements GORM runs (one Trace call per
// statement; BEGIN and COMMIT are not statements to GORM), so a benchmark can
// report statements per call. It logs nothing.
type countingLogger struct {
	statements atomic.Int64
}

func (l *countingLogger) LogMode(logger.LogLevel) logger.Interface { return l }
func (*countingLogger) Info(context.Context, string, ...any)       {}
func (*countingLogger) Warn(context.Context, string, ...any)       {}
func (*countingLogger) Error(context.Context, string, ...any)      {}
func (l *countingLogger) Trace(context.Context, time.Time, func() (string, int64), error) {
	l.statements.Add(1)
}

// reset zeroes the counter; call it right before the measured loop so seeding
// is not counted.
func (l *countingLogger) reset() { l.statements.Store(0) }

// report adds statements/op for a loop that ran b.N times.
func (l *countingLogger) report(b *testing.B) {
	b.Helper()
	b.ReportMetric(float64(l.statements.Load())/float64(b.N), "statements/op")
}

type benchOptions struct {
	// prepared turns on GORM's prepared-statement cache, a caller-side
	// setting of the *gorm.DB handed to the service.
	prepared bool
	// rtt, when non-zero, is slept before every statement to stand in for
	// the round trip to a networked database. BEGIN and COMMIT are not
	// delayed, so a transaction of n statements costs n round trips here
	// where a real network pays n+2.
	rtt time.Duration
}

func benchService(b *testing.B) (session.Service, *countingLogger) {
	b.Helper()
	return benchServiceWith(b, benchOptions{})
}

func benchServiceWith(b *testing.B, opts benchOptions) (session.Service, *countingLogger) {
	b.Helper()
	// A file-backed database, not :memory:, so the cost includes the fsync and
	// page I/O a deployment pays. The logger only counts statements, so the
	// benchmark measures the service rather than a log sink.
	l := &countingLogger{}
	db, err := gorm.Open(sqlite.Open(b.TempDir()+"/bench.db"), &gorm.Config{Logger: l, PrepareStmt: opts.prepared})
	if err != nil {
		b.Fatal(err)
	}
	if opts.rtt > 0 {
		sleep := func(*gorm.DB) { time.Sleep(opts.rtt) }
		for _, c := range []interface {
			Register(string, func(*gorm.DB)) error
		}{
			db.Callback().Create().Before("gorm:create"),
			db.Callback().Query().Before("gorm:query"),
			db.Callback().Update().Before("gorm:update"),
			db.Callback().Delete().Before("gorm:delete"),
			db.Callback().Row().Before("gorm:row"),
			db.Callback().Raw().Before("gorm:raw"),
		} {
			if err := c.Register("bench:rtt", sleep); err != nil {
				b.Fatal(err)
			}
		}
	}
	svc, err := NewSessionServiceFromDB(db)
	if err != nil {
		b.Fatal(err)
	}
	if err := AutoMigrate(svc); err != nil {
		b.Fatal(err)
	}
	return svc, l
}

func benchGet(b *testing.B, svc session.Service, l *countingLogger, req *session.GetRequest) {
	b.Helper()
	// One untimed read so the page cache is warm before the loop.
	if _, err := svc.Get(b.Context(), req); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	l.reset()
	for b.Loop() {
		if _, err := svc.Get(b.Context(), req); err != nil {
			b.Fatal(err)
		}
	}
	l.report(b)
}

func BenchmarkSQLiteGet(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			svc, l := benchService(b)
			if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", n); err != nil {
				b.Fatal(err)
			}
			benchGet(b, svc, l, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s"})
		})
	}
}

// BenchmarkSQLiteGetRecent is the same read limited to the last 20 events,
// the shape an existence check or a tail read could use.
func BenchmarkSQLiteGetRecent(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			svc, l := benchService(b)
			if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", n); err != nil {
				b.Fatal(err)
			}
			benchGet(b, svc, l, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s", NumRecentEvents: 20})
		})
	}
}

// BenchmarkSQLiteGetPrepared is the 1000-event read on a *gorm.DB opened with
// PrepareStmt, the setting that skips re-parsing the SQL on every call. The
// service does not set it; the caller that opens the database does.
func BenchmarkSQLiteGetPrepared(b *testing.B) {
	svc, l := benchServiceWith(b, benchOptions{prepared: true})
	if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", 1000); err != nil {
		b.Fatal(err)
	}
	benchGet(b, svc, l, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s"})
}

func benchAppendEvent(b *testing.B, svc session.Service, l *countingLogger, sess session.Session, evs []*session.Event) {
	b.Helper()
	b.ReportAllocs()
	l.reset()
	i := 0
	for b.Loop() {
		ev := *evs[i%len(evs)]
		ev.ID = ""
		// Monotonic timestamps: the service rejects a stale session.
		ev.Timestamp = time.Unix(1_800_000_000+int64(i), 0)
		if err := svc.AppendEvent(b.Context(), sess, &ev); err != nil {
			b.Fatal(err)
		}
		i++
	}
	l.report(b)
}

func BenchmarkSQLiteAppendEvent(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			svc, l := benchService(b)
			sess, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", n)
			if err != nil {
				b.Fatal(err)
			}
			// Two plain conversation events, no state delta: the common case.
			benchAppendEvent(b, svc, l, sess, benchutil.BenchEvents(2, "agent", time.Unix(1_800_000_000, 0)))
		})
	}
}

// BenchmarkSQLiteAppendEventStateDelta appends events that each change one
// session-state key, the path that has to merge the stored state.
func BenchmarkSQLiteAppendEventStateDelta(b *testing.B) {
	svc, l := benchService(b)
	sess, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", 100)
	if err != nil {
		b.Fatal(err)
	}
	evs := benchutil.BenchEvents(2, "agent", time.Unix(1_800_000_000, 0))
	for _, ev := range evs {
		ev.Actions.StateDelta = map[string]any{"turn": 1}
	}
	benchAppendEvent(b, svc, l, sess, evs)
}

// BenchmarkSQLiteRTT runs the 1000-event Get, the 20-event tail read and
// AppendEvent with a simulated 5 ms round trip before every statement, so
// that a statement saved or run concurrently shows up in wall-clock time
// the way it would on a networked database. 5 ms rather than 1 ms because
// time.Sleep overshoots a 1 ms request by up to half on a loaded machine.
func BenchmarkSQLiteRTT(b *testing.B) {
	const rtt = 5 * time.Millisecond
	b.Run("Get", func(b *testing.B) {
		svc, l := benchServiceWith(b, benchOptions{rtt: rtt})
		if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", 1000); err != nil {
			b.Fatal(err)
		}
		benchGet(b, svc, l, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s"})
	})
	b.Run("GetRecent", func(b *testing.B) {
		svc, l := benchServiceWith(b, benchOptions{rtt: rtt})
		if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", 1000); err != nil {
			b.Fatal(err)
		}
		benchGet(b, svc, l, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s", NumRecentEvents: 20})
	})
	b.Run("AppendEvent", func(b *testing.B) {
		svc, l := benchServiceWith(b, benchOptions{rtt: rtt})
		sess, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", 100)
		if err != nil {
			b.Fatal(err)
		}
		benchAppendEvent(b, svc, l, sess, benchutil.BenchEvents(2, "agent", time.Unix(1_800_000_000, 0)))
	})
}

// BenchmarkSQLiteEventDecode isolates the per-event JSON decode Get does for
// each row, so its share of a full Get can be read off directly.
func BenchmarkSQLiteEventDecode(b *testing.B) {
	svc, _ := benchService(b)
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
			svc, l := benchService(b)
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
					benchGet(b, svc, l, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s", NumRecentEvents: recent})
				})
			}
		})
	}
}
