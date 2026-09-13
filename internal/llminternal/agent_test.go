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

package llminternal

import "testing"

func TestLiveModeToolInjectionRetriesAfterPanic(t *testing.T) {
	var calls int

	for attempt := 0; attempt < 2; attempt++ {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("Do() did not panic on attempt %d", attempt+1)
				}
			}()
			injection := LiveModeToolInjection{}
			injection.Do(func() {
				calls++
				panic("injection failed")
			})
		}()
	}

	if calls != 2 {
		t.Fatalf("injection calls = %d, want 2", calls)
	}
}

func TestLiveModeToolInjectionRunsOnceAfterSuccess(t *testing.T) {
	var calls int
	injection := LiveModeToolInjection{}

	for range 3 {
		injection.Do(func() { calls++ })
	}

	if calls != 1 {
		t.Fatalf("injection calls = %d, want 1", calls)
	}
}