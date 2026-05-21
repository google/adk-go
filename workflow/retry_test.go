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
