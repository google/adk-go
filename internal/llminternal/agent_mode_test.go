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

import (
	"sync"
	"testing"
)

func TestResolveMode_ConcurrentReadersAndWriters(t *testing.T) {
	t.Parallel()

	s := &State{} // ModeUnset
	const goroutines = 32

	var wg sync.WaitGroup
	wg.Add(goroutines * 2)
	for range goroutines {
		go func() {
			defer wg.Done()
			if got := s.ResolveMode(ModeSingleTurn); got != ModeSingleTurn {
				t.Errorf("ResolveMode() = %q, want %q", got, ModeSingleTurn)
			}
		}()
		go func() {
			defer wg.Done()
			_ = s.CurrentMode()
		}()
	}
	wg.Wait()

	if got := s.CurrentMode(); got != ModeSingleTurn {
		t.Fatalf("CurrentMode() = %q, want %q", got, ModeSingleTurn)
	}
}

func TestResolveMode_PreservesExplicitMode(t *testing.T) {
	t.Parallel()

	s := &State{Mode: ModeTask}
	if got := s.ResolveMode(ModeChat); got != ModeTask {
		t.Fatalf("ResolveMode() = %q, want %q", got, ModeTask)
	}
}
