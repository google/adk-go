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

package workflow

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

// TestDefaultRetryConfig verifies the values returned by
// DefaultRetryConfig (5 attempts, 1s initial delay, 60s cap, 2x
// backoff, full jitter, retry-every-error predicate).
func TestDefaultRetryConfig(t *testing.T) {
	want := &RetryConfig{
		MaxAttempts:   5,
		InitialDelay:  time.Second,
		MaxDelay:      60 * time.Second,
		BackoffFactor: 2.0,
		Jitter:        1.0,
	}
	got := DefaultRetryConfig()

	if diff := cmp.Diff(want, got, cmpopts.IgnoreFields(RetryConfig{}, "ShouldRetry")); diff != "" {
		t.Errorf("DefaultRetryConfig() mismatch (-want +got):\n%s", diff)
	}

	if got.ShouldRetry == nil || !got.ShouldRetry(nil) {
		t.Errorf("ShouldRetry is nil or returns false")
	}
}
