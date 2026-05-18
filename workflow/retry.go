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
	"math/rand"
	"time"
)

// CalculateDelay calculates the delay before the next retry attempt.
// failedAttempts is the number of times the node has already failed.
func CalculateDelay(cfg *RetryConfig, failedAttempts int) time.Duration {
	if cfg == nil || failedAttempts <= 0 {
		return 0
	}

	delay := float64(cfg.InitialDelay)
	for i := 1; i < failedAttempts; i++ {
		delay *= cfg.BackoffFactor
	}

	maxDelay := float64(cfg.MaxDelay)
	if maxDelay > 0 && delay > maxDelay {
		delay = maxDelay
	}

	if cfg.Jitter > 0 {
		randVal := rand.Float64()
		randomOffset := (randVal*2.0 - 1.0) * cfg.Jitter * delay
		delay += randomOffset
		if delay < 0 {
			delay = 0
		}
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
		return cfg.ShouldRetry(err)
	}
	return false // Default to false if not specified
}
