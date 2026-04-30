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
	"math"
	"math/rand/v2"
	"time"
)

// RetryConfig controls retry behavior around a node's RunImpl. Defaults
// (used when a field is left at the zero value) match adk-python:
// max_attempts=5, initial_delay=1s, max_delay=60s, backoff_factor=2.0,
// jitter=1.0.
type RetryConfig struct {
	// MaxAttempts is the total attempt count, including the original call.
	// 0 → default 5; 1 → no retries.
	MaxAttempts int

	// InitialDelay is the delay before the first retry. 0 → default 1s.
	InitialDelay time.Duration

	// MaxDelay caps the delay between retries. 0 → default 60s.
	MaxDelay time.Duration

	// BackoffFactor multiplies the delay after each attempt. <=1 → default 2.0.
	BackoffFactor float64

	// Jitter is the multiplicative jitter window applied to each delay,
	// uniform in [1-jitter, 1+jitter]. 0 → default 1.0; <0 → no jitter.
	Jitter float64

	// Retryable, when non-nil, restricts retries to errors matching one of
	// the listed sentinels (via errors.Is) or types (via errors.As). nil
	// means retry on any error.
	Retryable []error
}

// withDefaults returns r with zero fields filled in.
func (r RetryConfig) withDefaults() RetryConfig {
	if r.MaxAttempts <= 0 {
		r.MaxAttempts = 5
	}
	if r.InitialDelay <= 0 {
		r.InitialDelay = time.Second
	}
	if r.MaxDelay <= 0 {
		r.MaxDelay = 60 * time.Second
	}
	if r.BackoffFactor <= 1 {
		r.BackoffFactor = 2.0
	}
	if r.Jitter == 0 {
		r.Jitter = 1.0
	} else if r.Jitter < 0 {
		r.Jitter = 0
	}
	return r
}

// DelayFor computes the delay before retry attempt N (1-based: 1 is the
// first retry, 2 the second, …). Returns a duration in [0, MaxDelay].
func (r RetryConfig) DelayFor(attempt int) time.Duration {
	c := r.withDefaults()
	exp := math.Pow(c.BackoffFactor, float64(attempt-1))
	d := time.Duration(float64(c.InitialDelay) * exp)
	if d > c.MaxDelay {
		d = c.MaxDelay
	}
	if c.Jitter > 0 {
		factor := 1 + (rand.Float64()*2-1)*c.Jitter
		d = time.Duration(float64(d) * factor)
	}
	if d < 0 {
		d = 0
	}
	return d
}
