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
	"context"
	"math/rand"
	"time"
)

// CalculateDelay calculates the delay before the next retry attempt.
// failedAttempts is the number of times the node has already failed; zero or
// less, or a nil cfg, gives 0.
//
// The delay is InitialDelay times BackoffFactor to the power
// failedAttempts-1 (an unset or non-positive factor counts as 1). With Jitter
// above 0 it is offset by a uniform draw from [-Jitter*delay, +Jitter*delay]
// and floored at 0. With MaxDelay above 0 it never exceeds MaxDelay: the delay
// is capped at MaxDelay / (1 + Jitter) before the jitter is drawn and at
// MaxDelay after it. A MaxDelay of 0 means no cap.
func CalculateDelay(cfg *RetryConfig, failedAttempts int) time.Duration {
	if cfg == nil || failedAttempts <= 0 {
		return 0
	}

	// Unset BackoffFactor would collapse the delay to 0; use constant delay.
	backoffFactor := cfg.BackoffFactor
	if backoffFactor <= 0 {
		backoffFactor = 1.0
	}

	delay := float64(cfg.InitialDelay)
	for i := 1; i < failedAttempts; i++ {
		delay *= backoffFactor
	}

	maxDelay := float64(cfg.MaxDelay)
	if maxDelay > 0 {
		// Leave room for the jitter, so the draw stays within MaxDelay
		// around the delay, as adk-python's _get_retry_delay does.
		delay = min(delay, maxDelay/(1.0+max(cfg.Jitter, 0)))
	}

	if cfg.Jitter > 0 {
		randVal := rand.Float64()
		randomOffset := (randVal*2.0 - 1.0) * cfg.Jitter * delay
		delay = max(delay+randomOffset, 0)
	}

	if maxDelay > 0 {
		delay = min(delay, maxDelay)
	}

	return time.Duration(delay)
}

// ShouldRetry decides whether a node should be retried after a failure.
// failedAttempts is the number of times the node has already failed.
func ShouldRetry(cfg *RetryConfig, err error, failedAttempts int) bool {
	if cfg == nil {
		return false
	}
	if cfg.MaxAttempts <= 1 {
		return false
	}
	if failedAttempts >= cfg.MaxAttempts {
		return false
	}
	if cfg.ShouldRetry != nil {
		// If a custom retry predicate is explicitly provided, we honor it.
		// This allows users to override the default behavior and retry validation errors if desired.
		return cfg.ShouldRetry(err)
	}
	// By default, we do not retry input validation errors because the input is deterministic
	// per activation. Retrying would just loop on the same failure.
	return defaultShouldRetry(err)
}

// waitRetryDelay waits d or until ctx is done, and returns ctx.Err(). A done
// ctx wins even when d has elapsed, so a canceled caller never starts another
// attempt. d <= 0 returns ctx.Err() without waiting.
func waitRetryDelay(ctx context.Context, d time.Duration) error {
	if d > 0 {
		t := time.NewTimer(d)
		defer t.Stop()
		select {
		case <-t.C:
		case <-ctx.Done():
		}
	}
	// A two-case select picks at random when the delay has elapsed and ctx
	// is done, so decide on ctx itself.
	return ctx.Err()
}
