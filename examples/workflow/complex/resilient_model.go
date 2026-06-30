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
	"iter"
	"log"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// resilientModel wraps a model.LLM to bound and, as a last resort, paper
// over flaky calls — specifically Gemini's Google Search grounding, which
// intermittently stalls with neither a response nor an error.
//
// Why it exists: a bare model call has no per-call deadline, so a stalled
// grounding request hangs forever. A workflow.RetryConfig does not help —
// it only retries calls that *return* an error, and a hang returns
// nothing. And because the pipeline fans out to three researchers behind a
// JoinNode barrier, a single stalled researcher would hang (or, with a
// plain timeout, fail) the whole run.
//
// resilientModel addresses all three:
//   - timeout: every attempt runs under its own context.WithTimeout, so a
//     stall becomes a context.DeadlineExceeded instead of an infinite wait;
//   - retry: timed-out/errored attempts are retried (a fresh deadline
//     each time) after a short fixed backoff;
//   - fail-open: if every attempt is exhausted it returns a single
//     fallback model message instead of an error, so the JoinNode still
//     fires and the synthesis agent still produces a report from whatever
//     did succeed (graceful degradation rather than a dead pipeline).
//
// It implements model.LLM, so any agent can use it transparently.
type resilientModel struct {
	inner    model.LLM
	timeout  time.Duration // per-attempt deadline
	attempts int           // total tries, including the first
	fallback string        // model text returned when every attempt fails
}

// retryBackoff is the fixed pause between attempts. It is deliberately
// small relative to the per-attempt timeout, which does the real work.
const retryBackoff = time.Second

// newResilientModel wraps inner. attempts is clamped to a minimum of 1.
func newResilientModel(inner model.LLM, timeout time.Duration, attempts int, fallback string) *resilientModel {
	if attempts < 1 {
		attempts = 1
	}
	return &resilientModel{inner: inner, timeout: timeout, attempts: attempts, fallback: fallback}
}

// Name implements model.LLM.
func (m *resilientModel) Name() string { return m.inner.Name() }

// GenerateContent implements model.LLM. Each attempt is buffered so a
// half-streamed-then-failed attempt is discarded cleanly before a retry;
// the caller only ever observes one successful attempt (or the fallback).
func (m *resilientModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		var lastErr error
		for attempt := 1; attempt <= m.attempts; attempt++ {
			responses, err := m.tryOnce(ctx, req, stream)
			if err == nil {
				for _, r := range responses {
					if !yield(r, nil) {
						return
					}
				}
				return
			}
			lastErr = err

			// A cancelled *parent* context is a real abort (the user quit
			// or a sibling node failed), not our per-attempt timeout:
			// propagate it so the node is treated as cancelled, not failed
			// over to the fallback.
			if ctx.Err() != nil {
				yield(nil, ctx.Err())
				return
			}
			if attempt < m.attempts {
				select {
				case <-time.After(retryBackoff):
				case <-ctx.Done():
					yield(nil, ctx.Err())
					return
				}
			}
		}

		log.Printf("resilientModel(%s): all %d attempts failed (%v); returning fallback",
			m.inner.Name(), m.attempts, lastErr)
		yield(m.fallbackResponse(), nil)
	}
}

// tryOnce runs a single attempt under a fresh timeout, buffering its
// responses so a mid-stream failure leaves nothing half-emitted.
func (m *resilientModel) tryOnce(ctx context.Context, req *model.LLMRequest, stream bool) ([]*model.LLMResponse, error) {
	attemptCtx, cancel := context.WithTimeout(ctx, m.timeout)
	defer cancel()

	var buffered []*model.LLMResponse
	for resp, err := range m.inner.GenerateContent(attemptCtx, req, stream) {
		if err != nil {
			return nil, err
		}
		buffered = append(buffered, resp)
	}
	return buffered, nil
}

// fallbackResponse is a plain final model message (no function calls, not
// partial), so session.Event.IsFinalResponse reports true and the
// AgentNode adopts m.fallback as the node's output.
func (m *resilientModel) fallbackResponse() *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: m.fallback}},
		},
	}
}
