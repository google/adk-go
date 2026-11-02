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

package llminternal

import "testing"

func TestLiveModeToolInjection_RetriesAfterPanicAndLatchesAfterSuccess(t *testing.T) {
	var injection LiveModeToolInjection
	calls := 0
	for range 2 {
		func() {
			defer func() {
				if got := recover(); got != "injection failed" {
					t.Error("Do did not propagate the injection's panic")
				}
			}()
			injection.Do(func() {
				calls++
				panic("injection failed")
			})
		}()
	}
	if calls != 2 {
		t.Fatalf("panicking injection ran %d times, want 2", calls)
	}

	injection.Do(func() { calls++ })
	if calls != 3 {
		t.Fatalf("successful retry did not run: calls = %d, want 3", calls)
	}
	injection.Do(func() { calls++ })
	if calls != 3 {
		t.Fatalf("successful injection ran again: calls = %d, want 3", calls)
	}
}
