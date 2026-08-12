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
	"math"
	"math/rand/v2"
	"time"
)

// Defaults for RunLive's reconnect policy. Not user-configurable: adk-python
// exposes no knobs here either.
const (
	defaultReconnectInitialDelay = 250 * time.Millisecond
	defaultReconnectBackoff      = 2.0
	defaultReconnectJitter       = 0.2
	defaultReconnectMaxDelay     = 5 * time.Second
	defaultReconnectMaxRetries   = 8
	defaultReconnectResetAfter   = 30 * time.Second
)

// maxUncappedDelay bounds growth when a policy sets no maxDelay, purely so the
// doubling cannot overflow time.Duration. 1<<62 ns is exactly representable as
// a float64, so the clamp is lossless.
const maxUncappedDelay = time.Duration(1) << 62

// reconnectPolicy bounds how hard RunLive retries a connection that keeps
// dropping. Nil means defaultReconnectPolicy; only tests pass one.
//
// The delay maths duplicates workflow.CalculateDelay. Degenerate fields are read
// the way it reads them, so the two cannot drift: jitter <= 0 means no jitter,
// backoff <= 0 a constant delay, maxDelay <= 0 no cap, and maxRetries counts
// retries, not total attempts.
type reconnectPolicy struct {
	initialDelay time.Duration
	// maxDelay caps the delay returned, jitter included. Non-positive means no cap.
	maxDelay   time.Duration
	backoff    float64
	jitter     float64
	maxRetries int

	// resetAfter is how long a connection must stay up to count as progress and
	// restore the budget. Lifetime alone, deliberately: also requiring model
	// output would never reset an idle session, so a few ordinary GoAways would
	// kill a healthy call. Non-positive means never.
	resetAfter time.Duration
}

func defaultReconnectPolicy() *reconnectPolicy {
	return &reconnectPolicy{
		initialDelay: defaultReconnectInitialDelay,
		maxDelay:     defaultReconnectMaxDelay,
		backoff:      defaultReconnectBackoff,
		jitter:       defaultReconnectJitter,
		maxRetries:   defaultReconnectMaxRetries,
		resetAfter:   defaultReconnectResetAfter,
	}
}

// madeProgress reports whether a connection that stayed up for d counts as
// working, restoring the retry budget.
func (p *reconnectPolicy) madeProgress(d time.Duration) bool {
	return p.resetAfter > 0 && d >= p.resetAfter
}

// delay returns the wait before retry n (1-based): initialDelay *
// backoff^(n-1), spread by ±jitter, never exceeding maxDelay.
//
// Growth is capped at maxDelay/(1+jitter) rather than at maxDelay so the
// jitter still spreads once saturated; clamping jittered values to the cap
// instead would pile half of them onto maxDelay exactly.
func (p *reconnectPolicy) delay(attempt int) time.Duration {
	if attempt <= 0 {
		return 0
	}
	// An unset backoff would collapse every delay after the first to zero.
	backoff := p.backoff
	if backoff <= 0 {
		backoff = 1.0
	}
	// An unset maxDelay means no cap; capping at zero would remove all pacing.
	ceiling := float64(maxUncappedDelay)
	if p.maxDelay > 0 {
		ceiling = float64(p.maxDelay)
		if p.jitter > 0 {
			ceiling /= 1 + p.jitter
		}
	}
	d := float64(p.initialDelay)
	for range attempt - 1 {
		d *= backoff
		// Cap inside the loop so a long outage cannot overflow to +Inf.
		if d >= ceiling {
			d = ceiling
			break
		}
	}
	d = min(d, ceiling) // also covers attempt == 1, which skips the loop
	if p.jitter > 0 {
		d += (rand.Float64()*2.0 - 1.0) * p.jitter * d
	}
	// A negative initialDelay or a NaN backoff would otherwise fire a timer
	// immediately; NaN fails every comparison, so screen it explicitly.
	if math.IsNaN(d) || d < 0 {
		d = 0
	}
	return time.Duration(min(d, float64(maxUncappedDelay)))
}
