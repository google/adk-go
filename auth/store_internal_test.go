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

package auth

import (
	"strconv"
	"testing"
	"time"

	"google.golang.org/adk/v2/platform"
)

// Expired entries for principals that never come back must not accumulate:
// nothing but the sweep evicts them.
func TestInMemoryCredentialStoreSweepsExpired(t *testing.T) {
	clock := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return clock })
	s := NewInMemoryCredentialStore()
	cred := BearerCredential{Token: "t"}

	fill := func(prefix string, n int) {
		t.Helper()
		for i := range n {
			key := CredentialKey{UserID: prefix + strconv.Itoa(i), Key: "res"}
			if err := s.Set(ctx, key, cred, clock.Add(time.Minute)); err != nil {
				t.Fatalf("Set() error = %v", err)
			}
		}
	}
	fill("early-", 300)
	clock = clock.Add(time.Hour) // every entry above is now expired
	fill("late-", 300)

	s.mu.Lock()
	got := len(s.entries)
	s.mu.Unlock()
	if got > 300 {
		t.Errorf("store holds %d entries, want the 300 expired ones swept", got)
	}
}
