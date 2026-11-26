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
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/session"
)

// BenchmarkGetSessionInCrowdedTable reads one 100-event session while other
// sessions fill the rest of the events table, so the cost of finding the
// session's rows shows up next to the cost of decoding them.
func BenchmarkGetSessionInCrowdedTable(b *testing.B) {
	const eventsPerSession = 100
	for _, totalEvents := range []int{1_000, 100_000} {
		b.Run(fmt.Sprintf("events=%d", totalEvents), func(b *testing.B) {
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

			// Rows are inserted directly: one AppendEvent transaction per
			// event would make seeding the large table take minutes.
			base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
			content := []byte(`{"parts":[{"text":"hello"}],"role":"model"}`)
			sessions := totalEvents / eventsPerSession
			for i := range sessions {
				sessionID := fmt.Sprintf("s%d", i)
				if err := db.Create(&storageSession{AppName: "app", UserID: "user", ID: sessionID, State: map[string]any{}, CreateTime: base, UpdateTime: base}).Error; err != nil {
					b.Fatal(err)
				}
				events := make([]storageEvent, eventsPerSession)
				for j := range events {
					events[j] = storageEvent{
						ID:        fmt.Sprintf("e%d-%d", i, j),
						AppName:   "app",
						UserID:    "user",
						SessionID: sessionID,
						Author:    "agent",
						Actions:   []byte(`{}`),
						Content:   content,
						Timestamp: base.Add(time.Duration(j) * time.Second),
					}
				}
				if err := db.CreateInBatches(events, 500).Error; err != nil {
					b.Fatal(err)
				}
			}

			req := &session.GetRequest{AppName: "app", UserID: "user", SessionID: fmt.Sprintf("s%d", sessions/2)}
			b.ReportAllocs()
			for b.Loop() {
				resp, err := svc.Get(b.Context(), req)
				if err != nil {
					b.Fatal(err)
				}
				if n := resp.Session.Events().Len(); n != eventsPerSession {
					b.Fatalf("Get() returned %d events, want %d", n, eventsPerSession)
				}
			}
		})
	}
}
