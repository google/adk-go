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
	"errors"
	"testing"
	"time"
)

func TestCalculateDelay(t *testing.T) {
	cfg := &RetryConfig{
		InitialDelay:  time.Second,
		BackoffFactor: 2.0,
		MaxDelay:      10 * time.Second,
		Jitter:        0.0, // Deterministic for base testing
	}

	tests := []struct {
		failedAttempts int
		want           time.Duration
	}{
		{failedAttempts: 0, want: 0},
		{failedAttempts: 1, want: time.Second},
		{failedAttempts: 2, want: 2 * time.Second},
		{failedAttempts: 3, want: 4 * time.Second},
		{failedAttempts: 4, want: 8 * time.Second},
		{failedAttempts: 5, want: 10 * time.Second}, // Capped at MaxDelay
		{failedAttempts: 6, want: 10 * time.Second},
	}

	for _, tt := range tests {
		got := CalculateDelay(cfg, tt.failedAttempts)
		if got != tt.want {
			t.Errorf("CalculateDelay(..., %d) = %v, want %v", tt.failedAttempts, got, tt.want)
		}
	}
}

// TestCalculateDelayUnsetBackoffFactor guards that an unset BackoffFactor
// gives a constant delay instead of collapsing to 0.
func TestCalculateDelayUnsetBackoffFactor(t *testing.T) {
	cfg := &RetryConfig{
		MaxAttempts:  3,
		InitialDelay: time.Second,
		// BackoffFactor intentionally unset (zero).
	}

	for attempt := 1; attempt <= 3; attempt++ {
		got := CalculateDelay(cfg, attempt)
		if got != time.Second {
			t.Errorf("CalculateDelay(unset backoff, %d) = %v, want %v", attempt, got, time.Second)
		}
	}
}

func TestCalculateDelayWithJitter(t *testing.T) {
	cfg := &RetryConfig{
		InitialDelay:  time.Second,
		BackoffFactor: 2.0,
		MaxDelay:      10 * time.Second,
		Jitter:        0.5,
	}

	// With jitter 0.5, delay for 1st failed attempt (base 1s) should be in [0.5s, 1.5s]
	got := CalculateDelay(cfg, 1)
	if got < 500*time.Millisecond || got > 1500*time.Millisecond {
		t.Errorf("CalculateDelay with jitter returned %v, expected in range [0.5s, 1.5s]", got)
	}
}

func TestShouldRetry(t *testing.T) {
	errTest := errors.New("test error")

	tests := []struct {
		name           string
		cfg            *RetryConfig
		err            error
		failedAttempts int
		want           bool
	}{
		{
			name:           "Nil config",
			cfg:            nil,
			err:            errTest,
			failedAttempts: 1,
			want:           false,
		},
		{
			name: "Under max attempts",
			cfg: &RetryConfig{
				MaxAttempts: 3,
				ShouldRetry: func(e error) bool { return true },
			},
			err:            errTest,
			failedAttempts: 1,
			want:           true,
		},
		{
			name:           "Default to true when ShouldRetry is nil",
			cfg:            &RetryConfig{MaxAttempts: 3},
			err:            errTest,
			failedAttempts: 1,
			want:           true,
		},
		{
			name:           "At max attempts",
			cfg:            &RetryConfig{MaxAttempts: 3},
			err:            errTest,
			failedAttempts: 3,
			want:           false,
		},
		{
			name:           "Above max attempts",
			cfg:            &RetryConfig{MaxAttempts: 3},
			err:            errTest,
			failedAttempts: 4,
			want:           false,
		},
		{
			name:           "Zero max attempts (no retry)",
			cfg:            &RetryConfig{MaxAttempts: 0},
			err:            errTest,
			failedAttempts: 1,
			want:           false,
		},
		{
			name: "Predicate allows",
			cfg: &RetryConfig{
				MaxAttempts: 3,
				ShouldRetry: func(e error) bool { return true },
			},
			err:            errTest,
			failedAttempts: 1,
			want:           true,
		},
		{
			name: "Predicate denies",
			cfg: &RetryConfig{
				MaxAttempts: 3,
				ShouldRetry: func(e error) bool { return false },
			},
			err:            errTest,
			failedAttempts: 1,
			want:           false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ShouldRetry(tt.cfg, tt.err, tt.failedAttempts)
			if got != tt.want {
				t.Errorf("ShouldRetry() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestCalculateDelay_JitterNeverExceedsMaxDelay checks that with a MaxDelay
// the delay is capped at MaxDelay / (1 + Jitter) before the jitter is drawn
// and at MaxDelay after it, as adk-python's _get_retry_delay does.
//
// Fails if either cap is removed: the draws then reach above MaxDelay, or
// never go below the jittered uncapped range.
func TestCalculateDelay_JitterNeverExceedsMaxDelay(t *testing.T) {
	tests := []struct {
		name string
		cfg  *RetryConfig
		// Every draw lies in [lo, hi]; some draw is below below and some
		// above above.
		lo, hi, below, above time.Duration
	}{
		{
			// Capped at 10s / 2 = 5s before the jitter, so draws spread
			// over [0s, 10s] around 5s.
			name:  "full_jitter",
			cfg:   &RetryConfig{InitialDelay: 20 * time.Second, MaxDelay: 10 * time.Second, BackoffFactor: 2, Jitter: 1.0},
			lo:    0,
			hi:    10 * time.Second,
			below: 5 * time.Second,
			above: 5 * time.Second,
		},
		{
			// Capped at 10s / 1.5 before the jitter, so draws spread over
			// [3.33s, 10s]; an 8s delay jittered without the cap would stay
			// in [4s, 12s].
			name:  "half_jitter",
			cfg:   &RetryConfig{InitialDelay: 8 * time.Second, MaxDelay: 10 * time.Second, BackoffFactor: 2, Jitter: 0.5},
			lo:    3333333333 * time.Nanosecond,
			hi:    10 * time.Second,
			below: 4 * time.Second,
			above: 6 * time.Second,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var atMax int
			sawBelow, sawAbove := false, false
			for range 2000 {
				d := CalculateDelay(tt.cfg, 1)
				if d < tt.lo || d > tt.hi {
					t.Fatalf("CalculateDelay() = %v, want within [%v, %v]", d, tt.lo, tt.hi)
				}
				if d == tt.hi {
					atMax++
				}
				sawBelow = sawBelow || d < tt.below
				sawAbove = sawAbove || d > tt.above
			}
			if atMax >= 50 {
				t.Errorf("%d of 2000 draws equal %v, want fewer than 50", atMax, tt.hi)
			}
			if !sawBelow || !sawAbove {
				t.Errorf("draws below %v: %v, above %v: %v; want both", tt.below, sawBelow, tt.above, sawAbove)
			}
		})
	}
}
