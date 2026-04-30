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

// Package retry provides a provider-agnostic wrapper around model.LLM that
// retries transient errors (HTTP 429, 5xx, common network resets) with
// exponential backoff and jitter.
//
// Retries apply to the initial model response. If the stream begins and
// then errors mid-flight, the wrapper does NOT restart — restarting
// mid-stream would produce duplicate tokens that confuse the agent loop.
package retry

import (
	"context"
	"errors"
	"iter"
	"math"
	"math/rand/v2"
	"time"

	"google.golang.org/adk/model"
)

// Config tunes the retry behavior. The zero value is valid and uses the
// defaults documented on each field.
type Config struct {
	// MaxAttempts is the total number of attempts (1 = no retries).
	// Default: 3.
	MaxAttempts int

	// InitialDelay is the delay before the first retry. Default: 1s.
	InitialDelay time.Duration

	// MaxDelay caps the delay between retries. Default: 60s.
	MaxDelay time.Duration

	// BackoffFactor multiplies the delay after each attempt. Default: 2.0.
	BackoffFactor float64

	// Jitter is a multiplicative jitter window applied to each delay,
	// uniformly random in [1-jitter, 1+jitter]. Default: 0.5.
	// Set to 0 for no jitter.
	Jitter float64

	// RetryOn classifies an error as retriable. If nil, IsTransient is used.
	// Returns false for ctx errors so parent-context cancellation is honored.
	RetryOn func(error) bool

	// OnRetry is an optional hook invoked before each sleep, useful for
	// telemetry. attempt is 1-based and indicates the attempt that just
	// failed (i.e. the next attempt will be attempt+1).
	OnRetry func(attempt int, err error, delay time.Duration)
}

func (c Config) withDefaults() Config {
	if c.MaxAttempts <= 0 {
		c.MaxAttempts = 3
	}
	if c.InitialDelay <= 0 {
		c.InitialDelay = time.Second
	}
	if c.MaxDelay <= 0 {
		c.MaxDelay = 60 * time.Second
	}
	if c.BackoffFactor <= 1 {
		c.BackoffFactor = 2.0
	}
	if c.Jitter < 0 {
		c.Jitter = 0
	} else if c.Jitter == 0 {
		c.Jitter = 0.5
	}
	if c.RetryOn == nil {
		c.RetryOn = IsTransient
	}
	return c
}

// Wrap returns a model.LLM that retries on transient errors before yielding
// the first response. Once the first response has been emitted, subsequent
// stream errors propagate verbatim — mid-stream retries are not safe.
//
// The returned LLM forwards Name() unchanged.
func Wrap(llm model.LLM, cfg Config) model.LLM {
	return &retrier{inner: llm, cfg: cfg.withDefaults()}
}

type retrier struct {
	inner model.LLM
	cfg   Config
}

func (r *retrier) Name() string { return r.inner.Name() }

func (r *retrier) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		var lastErr error
		for attempt := 1; attempt <= r.cfg.MaxAttempts; attempt++ {
			// Fresh seq per attempt; never reuse a partially-consumed iterator.
			next, stop := iter.Pull2(r.inner.GenerateContent(ctx, req, stream))

			firstResp, firstErr, ok := next()
			if !ok {
				// Stream ended with no items at all — treat as success (empty stream).
				stop()
				return
			}

			if firstErr != nil {
				stop()
				lastErr = firstErr
				if attempt == r.cfg.MaxAttempts || !r.shouldRetry(ctx, firstErr) {
					yield(firstResp, firstErr)
					return
				}
				delay := r.delayFor(attempt)
				if r.cfg.OnRetry != nil {
					r.cfg.OnRetry(attempt, firstErr, delay)
				}
				if !sleep(ctx, delay) {
					yield(nil, ctx.Err())
					return
				}
				continue
			}

			// Success: forward first item, then drain remaining stream verbatim.
			if !yield(firstResp, nil) {
				stop()
				return
			}
			for {
				resp, err, ok := next()
				if !ok {
					stop()
					return
				}
				if !yield(resp, err) {
					stop()
					return
				}
			}
		}
		// Exhausted attempts with errors only.
		yield(nil, lastErr)
	}
}

func (r *retrier) shouldRetry(ctx context.Context, err error) bool {
	if err == nil {
		return false
	}
	// Honor parent context cancellation.
	if ctx.Err() != nil && (errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)) {
		return false
	}
	return r.cfg.RetryOn(err)
}

func (r *retrier) delayFor(attempt int) time.Duration {
	// attempt is 1-based; first retry uses InitialDelay.
	exp := math.Pow(r.cfg.BackoffFactor, float64(attempt-1))
	d := time.Duration(float64(r.cfg.InitialDelay) * exp)
	if d > r.cfg.MaxDelay {
		d = r.cfg.MaxDelay
	}
	if r.cfg.Jitter > 0 {
		// Uniform multiplicative jitter in [1-j, 1+j].
		factor := 1 + (rand.Float64()*2-1)*r.cfg.Jitter
		d = time.Duration(float64(d) * factor)
	}
	if d < 0 {
		d = 0
	}
	return d
}

func sleep(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}
