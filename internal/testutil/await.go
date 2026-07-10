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

package testutil

import (
	"testing"
	"time"
)

// AwaitN receives n values from ch, or fails the test via t.Fatalf if they do
// not all arrive within a generous, contention-tolerant deadline. A closed
// channel counts as a receive, so AwaitN also joins a goroutine that closed ch
// without sending.
func AwaitN[T any](t *testing.T, ch <-chan T, n int, what string) {
	t.Helper()
	const deadline = 30 * time.Second
	timer := time.NewTimer(deadline)
	defer timer.Stop()
	for i := range n {
		select {
		case <-ch:
		case <-timer.C:
			t.Fatalf("%s: got %d of %d within %v", what, i, n, deadline)
		}
	}
}
