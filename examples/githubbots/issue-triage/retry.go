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

package main

import (
	"context"
	"crypto/rand"
	"errors"
	"iter"
	"log/slog"
	"math/big"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// Retry bounds. Deliberately small: the per-issue timeout is the real ceiling,
// and an issue that will not classify is better left for the next sweep than
// retried until the budget is gone.
const (
	maxModelAttempts = 4
	baseRetryDelay   = 2 * time.Second
)

// retryingModel retries a model call that failed transiently.
//
// Measured against the live API while testing this bot: two of four
// back-to-back classifications returned 503 "Preempted out of decode queue by a
// higher priority request", which the provider describes as temporary. Without
// a retry each one fails its issue, and because run() reports a failed issue the
// whole scheduled job goes red -- four times a day, on upstream load rather than
// on anything wrong here. That trains maintainers to ignore the bot's failures,
// which is worse than the transient error.
//
// It is a decorator rather than a loop inside run because the model call happens
// inside the framework, several layers below anything this package drives.
type retryingModel struct {
	inner    model.LLM
	log      *slog.Logger
	attempts int
	// delay is a variable so tests do not sleep.
	delay func(attempt int) time.Duration
}

func newRetryingModel(inner model.LLM, log *slog.Logger) *retryingModel {
	return &retryingModel{
		inner:    inner,
		log:      log,
		attempts: maxModelAttempts,
		delay:    backoffWithJitter,
	}
}

// backoffWithJitter grows the wait and spreads retries apart, so a provider
// already shedding load does not get every client back at the same instant.
//
// The randomness comes from crypto/rand, which is more than jitter needs -- it
// is not a security decision. The reason is narrower: gosec reports math/rand
// as G404, and the alternative is an inline suppression, which is a worse thing
// to have in a file than one extra import the module already depends on. If the
// draw fails, the retry simply loses its jitter and still backs off, which is
// the opposite of the nonce in main.go, where a failed draw is fatal because a
// predictable value is the whole attack.
func backoffWithJitter(attempt int) time.Duration {
	d := baseRetryDelay << (attempt - 1)
	spread, err := rand.Int(rand.Reader, big.NewInt(int64(d/2)+1))
	if err != nil {
		return d
	}
	return d + time.Duration(spread.Int64())
}

func (m *retryingModel) Name() string { return m.inner.Name() }

// GenerateContent forwards to the wrapped model, retrying a transient failure.
//
// It retries ONLY a call that failed before yielding anything. Once a response
// has reached the caller, restarting the stream would replay content the
// framework has already consumed, which is a worse failure than the one being
// recovered from.
func (m *retryingModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		for attempt := 1; ; attempt++ {
			delivered := false
			var failure error

			for resp, err := range m.inner.GenerateContent(ctx, req, stream) {
				if err != nil && !delivered {
					failure = err
					break
				}
				delivered = true
				if !yield(resp, err) {
					return
				}
			}

			if failure == nil {
				return
			}
			if attempt >= m.attempts || !isRetryableModelError(failure) || ctx.Err() != nil {
				yield(nil, failure)
				return
			}
			wait := m.delay(attempt)
			m.log.Warn("model call failed transiently; retrying",
				"attempt", attempt, "of", m.attempts, "wait", wait, "error", failure)
			select {
			case <-time.After(wait):
			case <-ctx.Done():
				yield(nil, failure)
				return
			}
		}
	}
}

// isRetryableModelError reports whether an error is worth trying again.
//
// Only the statuses the provider itself calls temporary. A 400 or a 403 is a
// configuration or quota problem that will fail identically every time, and
// retrying it wastes the issue's budget and hides the real cause.
func isRetryableModelError(err error) bool {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		return false
	}
	switch apiErr.Code {
	case 429, // rate limited
		500, // internal
		502, // bad gateway
		503, // overloaded: the one measured against the live API
		504: // gateway timeout
		return true
	}
	return false
}
