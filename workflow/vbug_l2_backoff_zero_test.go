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

// FINDING L2 — CalculateDelay collapses to zero when BackoffFactor is unset.
//
// Bug: CalculateDelay multiplies the running delay by cfg.BackoffFactor once
// per prior attempt. RetryConfig allows leaving BackoffFactor at its zero
// value (the struct doc documents defaults for the other fields and treats the
// factor as a plain optional value), but a zero factor multiplies the delay to
// 0 for any failedAttempts >= 2, defeating the configured InitialDelay.
//
// Expected: with InitialDelay > 0, the computed delay for attempt 2 is still
// positive (a zero/unset BackoffFactor should behave as a constant delay,
// i.e. effectively a factor of 1.0, never zeroing out the configured delay).
//
// This test currently FAILS, demonstrating the bug.

package workflow

import (
	"testing"
	"time"
)

func TestVbugL2_CalculateDelay_ZeroBackoffFactor(t *testing.T) {
	cfg := &RetryConfig{
		MaxAttempts:  3,
		InitialDelay: time.Second,
		// BackoffFactor intentionally left at its zero value.
	}
	d := CalculateDelay(cfg, 2)
	if d <= 0 {
		t.Errorf("CalculateDelay with unset BackoffFactor = %v, want > 0", d)
	}
}
