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

package session

import (
	"slices"
	"testing"
	"time"
)

// TestCompareByLastUpdate tests the comparator directly. List scans a map
// ordered by (app, user, session), so ties already arrive in tie-break order
// there and a test through List cannot tell whether the comparator breaks them.
func TestCompareByLastUpdate(t *testing.T) {
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	mk := func(user, sessionID string, d time.Duration) Session {
		return &session{id: id{appName: "app", userID: user, sessionID: sessionID}, updatedAt: base.Add(d)}
	}
	// At time 0, u2/a ties with u1/a and u1/b. Its session ID sorts before
	// u1/b's but its user ID after, so only user ID then session ID gives the
	// want.
	sessions := []Session{
		mk("u2", "x", time.Minute),
		mk("u2", "a", 0),
		mk("u1", "b", 0),
		mk("u1", "c", time.Hour),
		mk("u1", "a", 0),
	}
	slices.SortStableFunc(sessions, compareByLastUpdate)
	var got []string
	for _, s := range sessions {
		got = append(got, s.UserID()+"/"+s.ID())
	}
	if want := []string{"u1/a", "u1/b", "u2/a", "u2/x", "u1/c"}; !slices.Equal(got, want) {
		t.Errorf("sorted with compareByLastUpdate = %v, want %v", got, want)
	}
}
