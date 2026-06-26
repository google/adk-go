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
	"errors"
	"fmt"
	"iter"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"
)

// fakeLLM records calls and returns preconfigured responses/errors per attempt.
type fakeLLM struct {
	attempts  atomic.Int32
	responses []fakeLLMResult
}

type fakeLLMResult struct {
	// For non-streaming: single response/error pair.
	resp *LLMResponse
	err  error
	// For streaming: sequence of response/error pairs.
	stream []struct {
		resp *LLMResponse
		err  error
	}
}

func (f *fakeLLM) Name() string { return "fake" }

func (f *fakeLLM) GenerateContent(_ context.Context, _ *LLMRequest, stream bool) iter.Seq2[*LLMResponse, error] {
	idx := int(f.attempts.Add(1)) - 1
	if idx >= len(f.responses) {
		idx = len(f.responses) - 1
	}
	result := f.responses[idx]

	return func(yield func(*LLMResponse, error) bool) {
		if stream && len(result.stream) > 0 {
			for _, s := range result.stream {
				if !yield(s.resp, s.err) {
					return
				}
				if s.err != nil {
					return
				}
			}
			return
		}
		yield(result.resp, result.err)
	}
}

func okResponse(text string) *LLMResponse {
	return &LLMResponse{
		Content: &genai.Content{Parts: []*genai.Part{{Text: text}}},
	}
}

func TestRetryUnary_SuccessNoRetry(t *testing.T) {
	fake := &fakeLLM{responses: []fakeLLMResult{
		{resp: okResponse("hello")},
	}}
	llm := WithRetry(fake, nil)

	var got *LLMResponse
	for r, err := range llm.GenerateContent(context.Background(), &LLMRequest{}, false) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got = r
	}

	if got.Content.Parts[0].Text != "hello" {
		t.Errorf("want 'hello', got %q", got.Content.Parts[0].Text)
	}
	if fake.attempts.Load() != 1 {
		t.Errorf("want 1 attempt, got %d", fake.attempts.Load())
	}
}

func TestRetryUnary_RetriesOnTransientError(t *testing.T) {
	fake := &fakeLLM{responses: []fakeLLMResult{
		{err: errors.New("Error 503, UNAVAILABLE")},
		{err: errors.New("Error 503, UNAVAILABLE")},
		{resp: okResponse("recovered")},
	}}
	cfg := &RetryConfig{
		MaxRetries:   3,
		InitialDelay: time.Millisecond,
		MaxDelay:     10 * time.Millisecond,
		Jitter:       time.Millisecond,
	}
	llm := WithRetry(fake, cfg)

	var got *LLMResponse
	for r, err := range llm.GenerateContent(context.Background(), &LLMRequest{}, false) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got = r
	}

	if got.Content.Parts[0].Text != "recovered" {
		t.Errorf("want 'recovered', got %q", got.Content.Parts[0].Text)
	}
	if fake.attempts.Load() != 3 {
		t.Errorf("want 3 attempts, got %d", fake.attempts.Load())
	}
}

func TestRetryUnary_NonRetryableError(t *testing.T) {
	fake := &fakeLLM{responses: []fakeLLMResult{
		{err: errors.New("invalid API key")},
	}}
	cfg := &RetryConfig{InitialDelay: time.Millisecond, Jitter: time.Millisecond}
	llm := WithRetry(fake, cfg)

	for _, err := range llm.GenerateContent(context.Background(), &LLMRequest{}, false) {
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if err.Error() != "invalid API key" {
			t.Errorf("want 'invalid API key', got %q", err.Error())
		}
	}

	if fake.attempts.Load() != 1 {
		t.Errorf("want 1 attempt (no retry), got %d", fake.attempts.Load())
	}
}

func TestRetryUnary_ExhaustsRetries(t *testing.T) {
	fake := &fakeLLM{responses: []fakeLLMResult{
		{err: errors.New("503 UNAVAILABLE")},
		{err: errors.New("503 UNAVAILABLE")},
		{err: errors.New("503 UNAVAILABLE")},
		{err: errors.New("503 UNAVAILABLE")},
	}}
	cfg := &RetryConfig{
		MaxRetries:   3,
		InitialDelay: time.Millisecond,
		MaxDelay:     10 * time.Millisecond,
		Jitter:       time.Millisecond,
	}
	llm := WithRetry(fake, cfg)

	var lastErr error
	for _, err := range llm.GenerateContent(context.Background(), &LLMRequest{}, false) {
		lastErr = err
	}

	if lastErr == nil {
		t.Fatal("expected error after exhausting retries")
	}
	// 1 initial + 3 retries = 4 total
	if fake.attempts.Load() != 4 {
		t.Errorf("want 4 attempts, got %d", fake.attempts.Load())
	}
}

func TestRetryUnary_RespectsContextCancellation(t *testing.T) {
	fake := &fakeLLM{responses: []fakeLLMResult{
		{err: errors.New("503 UNAVAILABLE")},
		{err: errors.New("503 UNAVAILABLE")},
	}}
	cfg := &RetryConfig{
		MaxRetries:   5,
		InitialDelay: time.Second, // long delay so cancellation beats it
	}
	llm := WithRetry(fake, cfg)

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel after a short time to interrupt the backoff sleep.
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	var lastErr error
	for _, err := range llm.GenerateContent(ctx, &LLMRequest{}, false) {
		lastErr = err
	}

	if lastErr == nil {
		t.Fatal("expected error after context cancellation")
	}
	// Should not have exhausted all retries.
	if fake.attempts.Load() > 3 {
		t.Errorf("expected fewer attempts due to cancellation, got %d", fake.attempts.Load())
	}
}

func TestRetryUnary_CustomIsRetryable(t *testing.T) {
	fake := &fakeLLM{responses: []fakeLLMResult{
		{err: errors.New("custom-transient")},
		{resp: okResponse("ok")},
	}}
	cfg := &RetryConfig{
		InitialDelay: time.Millisecond,
		Jitter:       time.Millisecond,
		IsRetryable: func(err error) bool {
			return err != nil && err.Error() == "custom-transient"
		},
	}
	llm := WithRetry(fake, cfg)

	var got *LLMResponse
	for r, err := range llm.GenerateContent(context.Background(), &LLMRequest{}, false) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got = r
	}

	if got.Content.Parts[0].Text != "ok" {
		t.Errorf("want 'ok', got %q", got.Content.Parts[0].Text)
	}
	if fake.attempts.Load() != 2 {
		t.Errorf("want 2 attempts, got %d", fake.attempts.Load())
	}
}

func TestRetryStream_SuccessNoRetry(t *testing.T) {
	fake := &fakeLLM{responses: []fakeLLMResult{
		{stream: []struct {
			resp *LLMResponse
			err  error
		}{
			{resp: &LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "chunk1"}}}, Partial: true}},
			{resp: &LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "chunk2"}}}, TurnComplete: true}},
		}},
	}}
	llm := WithRetry(fake, &RetryConfig{InitialDelay: time.Millisecond, Jitter: time.Millisecond})

	var chunks []string
	for r, err := range llm.GenerateContent(context.Background(), &LLMRequest{}, true) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		chunks = append(chunks, r.Content.Parts[0].Text)
	}

	if len(chunks) != 2 || chunks[0] != "chunk1" || chunks[1] != "chunk2" {
		t.Errorf("unexpected chunks: %v", chunks)
	}
	if fake.attempts.Load() != 1 {
		t.Errorf("want 1 attempt, got %d", fake.attempts.Load())
	}
}

func TestRetryStream_RetriesBeforeFirstData(t *testing.T) {
	fake := &fakeLLM{responses: []fakeLLMResult{
		{stream: []struct {
			resp *LLMResponse
			err  error
		}{
			{err: errors.New("503 UNAVAILABLE")},
		}},
		{stream: []struct {
			resp *LLMResponse
			err  error
		}{
			{resp: &LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "ok"}}}, TurnComplete: true}},
		}},
	}}
	cfg := &RetryConfig{
		MaxRetries:   2,
		InitialDelay: time.Millisecond,
		Jitter:       time.Millisecond,
	}
	llm := WithRetry(fake, cfg)

	var chunks []string
	for r, err := range llm.GenerateContent(context.Background(), &LLMRequest{}, true) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		chunks = append(chunks, r.Content.Parts[0].Text)
	}

	if len(chunks) != 1 || chunks[0] != "ok" {
		t.Errorf("unexpected chunks: %v", chunks)
	}
	if fake.attempts.Load() != 2 {
		t.Errorf("want 2 attempts, got %d", fake.attempts.Load())
	}
}

func TestRetryStream_NoRetryAfterPartialData(t *testing.T) {
	fake := &fakeLLM{responses: []fakeLLMResult{
		{stream: []struct {
			resp *LLMResponse
			err  error
		}{
			{resp: &LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "partial"}}}, Partial: true}},
			{err: errors.New("503 UNAVAILABLE")},
		}},
	}}
	cfg := &RetryConfig{
		MaxRetries:   3,
		InitialDelay: time.Millisecond,
		Jitter:       time.Millisecond,
	}
	llm := WithRetry(fake, cfg)

	var lastErr error
	var chunks []string
	for r, err := range llm.GenerateContent(context.Background(), &LLMRequest{}, true) {
		if err != nil {
			lastErr = err
			break
		}
		chunks = append(chunks, r.Content.Parts[0].Text)
	}

	if lastErr == nil {
		t.Fatal("expected error after partial data")
	}
	if len(chunks) != 1 || chunks[0] != "partial" {
		t.Errorf("unexpected chunks before error: %v", chunks)
	}
	// Only 1 attempt — no retry because data was already yielded.
	if fake.attempts.Load() != 1 {
		t.Errorf("want 1 attempt (no retry after partial), got %d", fake.attempts.Load())
	}
}

func TestRetryName_DelegatesToInner(t *testing.T) {
	fake := &fakeLLM{}
	llm := WithRetry(fake, nil)
	if llm.Name() != "fake" {
		t.Errorf("want 'fake', got %q", llm.Name())
	}
}

func TestRetryConfig_Defaults(t *testing.T) {
	var cfg *RetryConfig // nil
	if cfg.maxRetries() != 5 {
		t.Errorf("default maxRetries: want 5, got %d", cfg.maxRetries())
	}
	if cfg.initialDelay() != time.Second {
		t.Errorf("default initialDelay: want 1s, got %v", cfg.initialDelay())
	}
	if cfg.maxDelay() != 60*time.Second {
		t.Errorf("default maxDelay: want 60s, got %v", cfg.maxDelay())
	}
	if cfg.multiplier() != 2.0 {
		t.Errorf("default multiplier: want 2.0, got %f", cfg.multiplier())
	}
	if cfg.jitter() != time.Second {
		t.Errorf("default jitter: want 1s, got %v", cfg.jitter())
	}
}

func TestRetryConfig_BackoffGrowth(t *testing.T) {
	cfg := &RetryConfig{
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     10 * time.Second,
		Multiplier:   2.0,
		Jitter:       10 * time.Millisecond, // small jitter to keep growth observable
	}

	d0 := cfg.backoff(0)
	d1 := cfg.backoff(1)
	d2 := cfg.backoff(2)

	// Jitter is absolute: delay is in [base, base + 10ms).
	assertInRange(t, "backoff(0)", d0, 100*time.Millisecond, 110*time.Millisecond)
	assertInRange(t, "backoff(1)", d1, 200*time.Millisecond, 210*time.Millisecond)
	assertInRange(t, "backoff(2)", d2, 400*time.Millisecond, 410*time.Millisecond)

	if d1 <= d0 || d2 <= d1 {
		t.Errorf("backoff should increase: d0=%v, d1=%v, d2=%v", d0, d1, d2)
	}
}

func TestRetryConfig_BackoffCapsAtMax(t *testing.T) {
	cfg := &RetryConfig{
		InitialDelay: time.Second,
		MaxDelay:     5 * time.Second,
		Multiplier:   10.0,
	}
	// 1s * 10^2 = 100s → capped at 5s, plus absolute jitter up to 1s (default).
	d := cfg.backoff(2)
	assertInRange(t, "backoff(2)", d, 5*time.Second, 6*time.Second)
}

func assertInRange(t *testing.T, name string, got, lo, hi time.Duration) {
	t.Helper()
	if got < lo || got > hi {
		t.Errorf("%s: got %v, want in [%v, %v]", name, got, lo, hi)
	}
}

func TestDefaultIsRetryable(t *testing.T) {
	tests := []struct {
		err  error
		want bool
	}{
		{nil, false},
		{errors.New("invalid request"), false},
		{errors.New("408 Request Timeout"), true},
		{errors.New("429 Too Many Requests"), true},
		{errors.New("500 Internal Server Error"), true},
		{errors.New("502 Bad Gateway"), true},
		{errors.New("Error 503, Message: high demand, Status: UNAVAILABLE"), true},
		{errors.New("504 Gateway Timeout"), true},
		{errors.New("RESOURCE_EXHAUSTED: quota exceeded"), true},
		{errors.New("ResourceExhausted"), true},
		{errors.New("ServiceUnavailable"), true},
		// Network errors (aligned with adk-python httpx.NetworkError).
		{errors.New("dial tcp 127.0.0.1:443: connection refused"), true},
		{errors.New("read tcp: connection reset by peer"), true},
		{errors.New("lookup api.example.com: no such host"), true},
		{errors.New("dial tcp: i/o timeout"), true},
		{errors.New("connect: network is unreachable"), true},
		{fmt.Errorf("wrapped: %w", errors.New("503 service down")), true},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%v", tt.err), func(t *testing.T) {
			got := defaultIsRetryable(tt.err)
			if got != tt.want {
				t.Errorf("defaultIsRetryable(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}
