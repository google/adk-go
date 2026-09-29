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
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/compactioninternal"
	"google.golang.org/adk/v2/session"
)

// TestACompactionCorrectionSortsAfterTheRecordItCorrects pins that a corrected
// compaction record is read back after the one it supersedes.
//
// The correction carries the same range and one more hole, and the repair path
// relies on being able to tell which of the two is the later record. Giving it
// the same timestamp made that a coin flip here: ties break on id, which is a
// comparison of two freshly generated UUIDs, so the correction came back first
// about half the time. Prompt assembly tolerates either order because holes are
// unioned across records over one range, but the write side does not:
// LatestCompactionEvent picks exactly one, and picking the hole-less original
// leaves the straggler out of the next record's exclusions.
//
// In-memory storage cannot see this, because append order there deterministically
// puts the correction last. It needs a backend that reads by timestamp.
func TestACompactionCorrectionSortsAfterTheRecordItCorrects(t *testing.T) {
	// Not parallel: emptyService opens file::memory:?cache=shared, so every
	// test in this package works against one database.

	ctx := context.Background()
	const appName, userID = "app", "user"

	for trial := range 20 {
		svc := emptyService(t)
		created, err := svc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID})
		if err != nil {
			t.Fatalf("Create() error = %v", err)
		}
		sess := created.Session

		base := time.Date(2026, 5, 6, 7, 8, 9, 0, time.UTC)
		record := session.NewEvent(ctx, "inv-sum")
		record.Author = "user"
		record.Timestamp = base
		record.Actions.Compaction = &session.EventCompaction{
			StartTimestamp:   base.Add(-time.Hour),
			EndTimestamp:     base,
			CompactedContent: genai.NewContentFromText("summary", genai.RoleModel),
		}
		if err := svc.AppendEvent(ctx, sess, record); err != nil {
			t.Fatalf("AppendEvent(record) error = %v", err)
		}

		// The straggler: inside the record's range, and not part of what the
		// summary was built from.
		known := compactioninternal.KnownEventIDs(sess)
		late := session.NewEvent(ctx, "inv-late")
		late.Author = "user"
		late.Timestamp = base.Add(-time.Minute)
		late.LLMResponse.Content = genai.NewContentFromText("late", genai.RoleUser)
		if err := svc.AppendEvent(ctx, sess, late); err != nil {
			t.Fatalf("AppendEvent(late) error = %v", err)
		}
		reread, err := svc.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sess.ID()})
		if err != nil {
			t.Fatalf("Get() error = %v", err)
		}
		correction := compactioninternal.RepairAfterAppend(record, known, reread.Session)
		if correction == nil {
			t.Fatalf("trial %d: RepairAfterAppend() = nil, want a correction naming the straggler", trial)
		}
		if err := svc.AppendEvent(ctx, sess, correction); err != nil {
			t.Fatalf("AppendEvent(correction) error = %v", err)
		}

		got, err := svc.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sess.ID()})
		if err != nil {
			t.Fatalf("Get() error = %v", err)
		}
		var recs []*session.Event
		for ev := range got.Session.Events().All() {
			if ev.Actions.Compaction != nil {
				recs = append(recs, ev)
			}
		}
		if len(recs) != 2 {
			t.Fatalf("trial %d: read back %d records, want 2", trial, len(recs))
		}
		if n := len(recs[1].Actions.Compaction.ExcludedEvents); n != 1 {
			t.Fatalf("trial %d: the record read back last names %d holes, want 1: the correction must sort after the record it corrects",
				trial, n)
		}
	}
}
