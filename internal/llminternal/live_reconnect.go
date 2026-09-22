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
	"errors"
	"fmt"
	"math/rand/v2"
	"time"
)

// Defaults for RunLive's reconnect policy.
//
// The attempt limit matches adk-python's DEFAULT_MAX_RECONNECT_ATTEMPTS. The
// rest have no Python counterpart: adk-python neither paces its reconnects nor
// bounds a backend that keeps accepting connections and dropping them.
const (
	defaultLiveReconnectInitialBackoff = 250 * time.Millisecond
	defaultLiveReconnectMaxBackoff     = 2 * time.Second
	defaultLiveReconnectJitter         = 0.2
	defaultLiveReconnectMaxAttempts    = 5
	defaultLiveReconnectMaxTotal       = 20
	defaultLiveReconnectHealthyUptime  = 30 * time.Second
)

// errLiveReconnectExhausted reports that RunLive stopped reconnecting a live
// session because it hit a retry bound.
//
// The package is internal, so a caller outside the module reaches this only as
// message text. Exposing a public sentinel is a separate API decision.
var errLiveReconnectExhausted = errors.New("live session: reconnect budget exhausted")

// liveReconnectGaveUpError builds the error handed to the caller when a bound
// fires. bound names which one, so a flapping backend is distinguishable from a
// dead one, and cause is the failure that triggered the last attempt.
func liveReconnectGaveUpError(bound string, cause error) error {
	if cause == nil {
		return fmt.Errorf("%w: %s", errLiveReconnectExhausted, bound)
	}
	return fmt.Errorf("%w: %s: %w", errLiveReconnectExhausted, bound, cause)
}

// liveConnError is a failure on one live connection, carrying when it happened.
// The time is taken where the failure is produced rather than where it is
// consumed: the consumer loop also runs tools and blocks on the caller, so
// reading the clock there would credit that delay to the connection as uptime.
type liveConnError struct {
	err error
	at  time.Time
}

// liveReconnectPolicy bounds how hard RunLive retries a live connection that
// keeps dropping. Nil means the defaults; only tests supply one, to shrink the
// delays. Every duration field must be positive.
//
// Two counters bound the loop, because either alone leaves a hole:
//
//   - maxAttempts bounds consecutive reconnects and drives the backoff. It
//     resets whenever a connection works — the model delivers content, or the
//     connection outlives healthyUptime — so a call that rides out an
//     occasional dropped connection is never killed for it. On its own it never
//     fires against a backend that serves one frame per connection and then
//     hangs up, because that backend keeps resetting it.
//   - maxTotal bounds reconnects across the whole invocation and resets
//     nowhere. Only a connection that died inside healthyUptime counts towards
//     it, so the routine cycling the Live API does to a long session is free
//     while a backend that flaps is not.
//
// What this does not bound is a backend that serves for longer than
// healthyUptime and then drops, over and over. That is indistinguishable from a
// healthy session the server keeps cycling, so it is left to run. A caller slow
// enough to stretch every connection past healthyUptime buys the same
// treatment, which is safe for the failure this bounds: the loop can only run
// as fast as the caller consumes it, so it cannot become a redial storm.
type liveReconnectPolicy struct {
	initialBackoff time.Duration
	// maxBackoff caps the delay before jitter is applied.
	maxBackoff time.Duration
	// jitter spreads each delay over ±jitter, so clients that dropped together
	// do not redial together. Non-positive means no jitter.
	jitter float64
	// maxAttempts bounds consecutive reconnects since a connection last
	// delivered content or outlived healthyUptime.
	maxAttempts int
	// maxTotal bounds short-lived connections over the invocation's life.
	maxTotal int
	// healthyUptime is how long a connection must last to count as ordinary
	// service rather than a symptom of a failing endpoint.
	healthyUptime time.Duration
}

func defaultLiveReconnectPolicy() *liveReconnectPolicy {
	return &liveReconnectPolicy{
		initialBackoff: defaultLiveReconnectInitialBackoff,
		maxBackoff:     defaultLiveReconnectMaxBackoff,
		jitter:         defaultLiveReconnectJitter,
		maxAttempts:    defaultLiveReconnectMaxAttempts,
		maxTotal:       defaultLiveReconnectMaxTotal,
		healthyUptime:  defaultLiveReconnectHealthyUptime,
	}
}

// nextBackoff returns the delay to grow into after waiting for d, doubling it
// up to maxBackoff. Saturating at the cap is also what stops the doubling
// overflowing to a negative duration, which would fire the next timer at once
// and restore the loop this policy exists to bound.
func (p *liveReconnectPolicy) nextBackoff(d time.Duration) time.Duration {
	if d >= p.maxBackoff/2 {
		return p.maxBackoff
	}
	return d * 2
}

// jittered spreads d over ±jitter.
func (p *liveReconnectPolicy) jittered(d time.Duration) time.Duration {
	if d <= 0 || p.jitter <= 0 {
		return max(d, 0)
	}
	spread := time.Duration(float64(d) * (1 + (rand.Float64()*2-1)*p.jitter))
	return max(spread, 0)
}
