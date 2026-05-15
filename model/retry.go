// Copyright 2025 Google LLC
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

package model

import (
	"context"
	"iter"
	"log/slog"
	"math"
	"math/rand/v2"
	"strings"
	"time"
)

// RetryConfig controls retry behavior for transient LLM errors.
type RetryConfig struct {
	// MaxRetries is the maximum number of retry attempts. Defaults to 5.
	MaxRetries int
	// InitialDelay is the delay before the first retry. Defaults to 1s.
	InitialDelay time.Duration
	// MaxDelay caps the backoff delay. Defaults to 60s.
	MaxDelay time.Duration
	// Multiplier is the exponential backoff multiplier. Defaults to 2.0.
	Multiplier float64
	// Jitter is the maximum absolute random delay added on top of the
	// exponential backoff. Defaults to 1s.
	Jitter time.Duration
	// IsRetryable determines whether an error should be retried.
	// When nil, the default policy retries 408, 429, 500, 502, 503, 504,
	// UNAVAILABLE, RESOURCE_EXHAUSTED, and network-related errors.
	IsRetryable func(error) bool
}

func (c *RetryConfig) maxRetries() int {
	if c != nil && c.MaxRetries > 0 {
		return c.MaxRetries
	}
	return 5
}

func (c *RetryConfig) initialDelay() time.Duration {
	if c != nil && c.InitialDelay > 0 {
		return c.InitialDelay
	}
	return time.Second
}

func (c *RetryConfig) maxDelay() time.Duration {
	if c != nil && c.MaxDelay > 0 {
		return c.MaxDelay
	}
	return 60 * time.Second
}

func (c *RetryConfig) multiplier() float64 {
	if c != nil && c.Multiplier > 0 {
		return c.Multiplier
	}
	return 2.0
}

func (c *RetryConfig) jitter() time.Duration {
	if c != nil && c.Jitter > 0 {
		return c.Jitter
	}
	return time.Second
}

func (c *RetryConfig) isRetryable(err error) bool {
	if c != nil && c.IsRetryable != nil {
		return c.IsRetryable(err)
	}
	return defaultIsRetryable(err)
}

// backoff computes the delay for the given zero-based attempt number.
// Formula matches tenacity.wait_exponential_jitter:
//
//	min(initial * multiplier^attempt, maxDelay) + random(0, jitter)
func (c *RetryConfig) backoff(attempt int) time.Duration {
	base := float64(c.initialDelay()) * math.Pow(c.multiplier(), float64(attempt))
	delay := math.Min(base, float64(c.maxDelay()))
	j := rand.Float64() * float64(c.jitter())
	return time.Duration(delay + j)
}

var retryableSubstrings = []string{
	// HTTP status codes (aligned with adk-python default_status_codes).
	"408",
	"429",
	"500",
	"502",
	"503",
	"504",
	// gRPC status names.
	"UNAVAILABLE",
	"RESOURCE_EXHAUSTED",
	"ResourceExhausted",
	"ServiceUnavailable",
	// Network errors (aligned with adk-python httpx.NetworkError).
	"connection refused",
	"connection reset",
	"no such host",
	"i/o timeout",
	"network is unreachable",
}

func defaultIsRetryable(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	for _, sub := range retryableSubstrings {
		if strings.Contains(msg, sub) {
			return true
		}
	}
	return false
}

// WithRetry wraps an LLM with automatic retry logic using exponential backoff
// with jitter. Pass nil for cfg to use sensible defaults.
func WithRetry(llm LLM, cfg *RetryConfig) LLM {
	return &retryLLM{inner: llm, cfg: cfg}
}

type retryLLM struct {
	inner LLM
	cfg   *RetryConfig
}

func (r *retryLLM) Name() string {
	return r.inner.Name()
}

func (r *retryLLM) GenerateContent(ctx context.Context, req *LLMRequest, stream bool) iter.Seq2[*LLMResponse, error] {
	if stream {
		return r.generateContentStream(ctx, req)
	}
	return r.generateContentUnary(ctx, req)
}

// generateContentUnary retries the full unary call on retryable errors.
func (r *retryLLM) generateContentUnary(ctx context.Context, req *LLMRequest) iter.Seq2[*LLMResponse, error] {
	return func(yield func(*LLMResponse, error) bool) {
		maxRetries := r.cfg.maxRetries()
		var lastErr error

		for attempt := range maxRetries + 1 {
			if attempt > 0 {
				delay := r.cfg.backoff(attempt - 1)
				slog.WarnContext(ctx, "retrying LLM call",
					"model", r.inner.Name(),
					"attempt", attempt,
					"max_retries", maxRetries,
					"delay", delay,
					"error", lastErr,
				)
				if err := sleepCtx(ctx, delay); err != nil {
					yield(nil, lastErr)
					return
				}
			}

			var resp *LLMResponse
			var callErr error
			for r, e := range r.inner.GenerateContent(ctx, req, false) {
				resp, callErr = r, e
			}

			if callErr == nil {
				yield(resp, nil)
				return
			}

			lastErr = callErr
			if !r.cfg.isRetryable(callErr) {
				yield(nil, callErr)
				return
			}
		}

		yield(nil, lastErr)
	}
}

// generateContentStream retries only when the stream fails before yielding
// any successful response, preventing duplicate partial content.
func (r *retryLLM) generateContentStream(ctx context.Context, req *LLMRequest) iter.Seq2[*LLMResponse, error] {
	return func(yield func(*LLMResponse, error) bool) {
		maxRetries := r.cfg.maxRetries()
		var lastErr error

		for attempt := range maxRetries + 1 {
			if attempt > 0 {
				delay := r.cfg.backoff(attempt - 1)
				slog.WarnContext(ctx, "retrying LLM stream",
					"model", r.inner.Name(),
					"attempt", attempt,
					"max_retries", maxRetries,
					"delay", delay,
					"error", lastErr,
				)
				if err := sleepCtx(ctx, delay); err != nil {
					yield(nil, lastErr)
					return
				}
			}

			yieldedData := false
			shouldRetry := false

			for resp, err := range r.inner.GenerateContent(ctx, req, true) {
				if err != nil {
					if !yieldedData && r.cfg.isRetryable(err) {
						lastErr = err
						shouldRetry = true
						break
					}
					yield(nil, err)
					return
				}
				yieldedData = true
				if !yield(resp, nil) {
					return
				}
			}

			if !shouldRetry {
				return
			}
		}

		yield(nil, lastErr)
	}
}

// sleepCtx blocks for d or until ctx is cancelled, whichever comes first.
func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
