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

	"google.golang.org/adk/v2/model"
)

// resilientModel wraps a model.LLM to guard against Gemini's Google Search
// grounding, which intermittently stalls with neither a response nor an
// error. A bare call has no deadline (so a stall hangs forever) and
// workflow.RetryConfig only retries calls that return an error, so neither
// helps. resilientModel instead:
//   - bounds each attempt with context.WithTimeout, turning a stall into a
//     DeadlineExceeded;
//   - retries timed-out/errored attempts after a short backoff;
//   - fails open after the last attempt, returning a fallback message so
//     the JoinNode still fires and the pipeline still produces a report.
//
// It implements model.LLM, so any agent can use it transparently.
type resilientModel struct {
	inner    model.LLM
	timeout  time.Duration // per-attempt deadline
	attempts int           // total tries, including the first
	fallback string        // model text returned when every attempt fails
}

// retryBackoff is the fixed pause between attempts.
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
// partially streamed failure is discarded cleanly before a retry.
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

			// A cancelled parent context is a real abort, not our
			// per-attempt timeout: propagate it instead of failing open.
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

// fallbackResponse is a plain final model message, so
// session.Event.IsFinalResponse reports true and the AgentNode adopts
// m.fallback as the node's output.
func (m *resilientModel) fallbackResponse() *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: m.fallback}},
		},
	}
}
