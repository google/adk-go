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
	"errors"
	"fmt"
	"iter"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// scriptedFailures replays a fixed sequence of outcomes, one per call.
type scriptedFailures struct {
	outcomes []error // nil means "succeed"
	calls    int
	// yieldFirst makes the call deliver a response BEFORE failing, which must
	// suppress the retry.
	yieldFirst bool
}

func (s *scriptedFailures) Name() string { return "scripted-failures" }

func (s *scriptedFailures) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		i := s.calls
		s.calls++
		var err error
		if i < len(s.outcomes) {
			err = s.outcomes[i]
		}
		if err != nil {
			if s.yieldFirst {
				if !yield(&model.LLMResponse{ModelVersion: "partial"}, nil) {
					return
				}
			}
			yield(nil, err)
			return
		}
		yield(&model.LLMResponse{ModelVersion: "ok"}, nil)
	}
}

func drain(m model.LLM) (responses int, err error) {
	for resp, e := range m.GenerateContent(context.Background(), &model.LLMRequest{}, false) {
		if e != nil {
			err = e
			continue
		}
		if resp != nil {
			responses++
		}
	}
	return responses, err
}

func noDelay(int) time.Duration { return 0 }

// The 503 measured against the live API -- "Preempted out of decode queue by a
// higher priority request" -- must be retried, not charged to the issue.
//
// Killing mutation: drop 503 from isRetryableModelError's switch.
func TestRetryingModelRecoversFromATransientFailure(t *testing.T) {
	overloaded := fmt.Errorf("failed to call model: %w", genai.APIError{
		Code:    503,
		Status:  "UNAVAILABLE",
		Message: "This model is currently experiencing high demand.",
	})
	inner := &scriptedFailures{outcomes: []error{overloaded, overloaded, nil}}
	m := &retryingModel{inner: inner, log: discardLogger(), attempts: maxModelAttempts, delay: noDelay}

	responses, err := drain(m)
	if err != nil {
		t.Fatalf("GenerateContent = %v, want the retry to have recovered", err)
	}
	if responses != 1 {
		t.Errorf("delivered %d responses, want 1", responses)
	}
	if inner.calls != 3 {
		t.Errorf("called the model %d times, want 3 (two failures then success)", inner.calls)
	}
}

// A permanent error must fail immediately. Retrying a 400 or a 403 burns the
// issue's budget and hides the real cause.
//
// Killing mutation: make isRetryableModelError return true unconditionally.
func TestRetryingModelDoesNotRetryAPermanentFailure(t *testing.T) {
	for _, code := range []int{400, 401, 403, 404} {
		t.Run(fmt.Sprint(code), func(t *testing.T) {
			permanent := fmt.Errorf("failed to call model: %w", genai.APIError{Code: code, Status: "INVALID_ARGUMENT"})
			inner := &scriptedFailures{outcomes: []error{permanent, permanent, permanent, nil}}
			m := &retryingModel{inner: inner, log: discardLogger(), attempts: maxModelAttempts, delay: noDelay}

			if _, err := drain(m); err == nil {
				t.Fatal("GenerateContent = nil, want the permanent failure surfaced")
			}
			if inner.calls != 1 {
				t.Errorf("called the model %d times, want 1: a %d is not worth retrying", inner.calls, code)
			}
		})
	}
}

// Retries are bounded. An endpoint that is down must not consume the issue's
// whole budget.
//
// Killing mutation: remove the attempt >= m.attempts check.
func TestRetryingModelGivesUp(t *testing.T) {
	overloaded := fmt.Errorf("failed to call model: %w", genai.APIError{Code: 503})
	inner := &scriptedFailures{outcomes: []error{overloaded, overloaded, overloaded, overloaded, overloaded, nil}}
	m := &retryingModel{inner: inner, log: discardLogger(), attempts: maxModelAttempts, delay: noDelay}

	if _, err := drain(m); err == nil {
		t.Fatal("GenerateContent = nil, want the failure surfaced after the attempts ran out")
	}
	if inner.calls != maxModelAttempts {
		t.Errorf("called the model %d times, want %d", inner.calls, maxModelAttempts)
	}
	var apiErr genai.APIError
	if _, err := drain(&retryingModel{inner: &scriptedFailures{outcomes: []error{overloaded}}, log: discardLogger(), attempts: 1, delay: noDelay}); !errors.As(err, &apiErr) {
		t.Error("the surfaced error no longer carries the provider's own error")
	}
}

// A stream that already delivered a response must not be restarted: replaying
// content the framework has consumed is worse than the error being recovered
// from.
//
// Killing mutation: retry whether or not anything was delivered.
func TestRetryingModelDoesNotRestartAStreamItAlreadyDelivered(t *testing.T) {
	overloaded := fmt.Errorf("failed to call model: %w", genai.APIError{Code: 503})
	inner := &scriptedFailures{outcomes: []error{overloaded, nil}, yieldFirst: true}
	m := &retryingModel{inner: inner, log: discardLogger(), attempts: maxModelAttempts, delay: noDelay}

	responses, err := drain(m)
	if err == nil {
		t.Fatal("GenerateContent = nil, want the mid-stream failure surfaced rather than replayed")
	}
	if inner.calls != 1 {
		t.Errorf("called the model %d times, want 1: the stream had already delivered", inner.calls)
	}
	if responses != 1 {
		t.Errorf("delivered %d responses, want the 1 that arrived before the failure", responses)
	}
}

// A cancelled context ends the retry loop rather than sleeping through it.
//
// Repeated, because the mutation this pins is probabilistic: without the
// explicit ctx.Err() check the loop falls through to a select between an
// already-elapsed timer and a closed Done channel, which Go resolves at random.
// One run of this would pass against the mutant about half the time.
//
// Killing mutation: drop `|| ctx.Err() != nil` from the give-up condition.
func TestRetryingModelStopsOnACancelledContext(t *testing.T) {
	overloaded := fmt.Errorf("failed to call model: %w", genai.APIError{Code: 503})
	const rounds = 50
	for round := range rounds {
		inner := &scriptedFailures{outcomes: []error{overloaded, overloaded, overloaded, overloaded}}
		m := &retryingModel{inner: inner, log: discardLogger(), attempts: maxModelAttempts, delay: noDelay}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		var err error
		for _, e := range m.GenerateContent(ctx, &model.LLMRequest{}, false) {
			if e != nil {
				err = e
			}
		}
		if err == nil {
			t.Fatalf("round %d: GenerateContent = nil, want the failure surfaced", round)
		}
		if inner.calls != 1 {
			t.Fatalf("round %d: called the model %d times after the context was cancelled, want 1", round, inner.calls)
		}
	}
}

// The backoff must grow and must not be identical across concurrent retries, so
// a provider shedding load does not get the whole sweep back at the same moment.
func TestBackoffGrowsAndIsJittered(t *testing.T) {
	var prevMin time.Duration
	for attempt := 1; attempt <= 3; attempt++ {
		floor := baseRetryDelay << (attempt - 1)
		seen := make(map[time.Duration]bool)
		for range 200 {
			d := backoffWithJitter(attempt)
			if d < floor || d > floor+floor/2 {
				t.Fatalf("attempt %d: delay %s outside [%s, %s]", attempt, d, floor, floor+floor/2)
			}
			seen[d] = true
		}
		if len(seen) < 2 {
			t.Errorf("attempt %d produced a single delay %v; concurrent retries would not be spread", attempt, seen)
		}
		if floor <= prevMin {
			t.Errorf("attempt %d floor %s did not grow past %s", attempt, floor, prevMin)
		}
		prevMin = floor
	}
}

// A non-API error (a transport failure with no status) is not retried, because
// nothing says it is transient.
func TestRetryingModelDoesNotRetryAnUnclassifiedError(t *testing.T) {
	inner := &scriptedFailures{outcomes: []error{errors.New("connection reset"), nil}}
	m := &retryingModel{inner: inner, log: discardLogger(), attempts: maxModelAttempts, delay: noDelay}
	if _, err := drain(m); err == nil {
		t.Fatal("GenerateContent = nil, want the failure surfaced")
	}
	if inner.calls != 1 {
		t.Errorf("called the model %d times, want 1", inner.calls)
	}
}
