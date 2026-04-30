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

package retry

import (
	"context"
	"errors"
	"io"
	"iter"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type fakeLLM struct {
	name  string
	calls atomic.Int32
	// per-call programmed responses; first item per call may be (nil, err).
	scripts [][]step
}

type step struct {
	resp *model.LLMResponse
	err  error
}

func (f *fakeLLM) Name() string { return f.name }
func (f *fakeLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	idx := int(f.calls.Add(1)) - 1
	var script []step
	if idx < len(f.scripts) {
		script = f.scripts[idx]
	} else {
		script = f.scripts[len(f.scripts)-1]
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		for _, s := range script {
			if !yield(s.resp, s.err) {
				return
			}
		}
	}
}

func mustErr(t *testing.T, want error, got error) {
	t.Helper()
	if !errors.Is(got, want) {
		t.Fatalf("err = %v, want errors.Is %v", got, want)
	}
}

func collect(seq iter.Seq2[*model.LLMResponse, error]) (resps []*model.LLMResponse, errs []error) {
	for r, e := range seq {
		resps = append(resps, r)
		errs = append(errs, e)
	}
	return
}

func TestRetry_FirstErrorTransient_RetriesAndSucceeds(t *testing.T) {
	resp := &model.LLMResponse{ModelVersion: "ok"}
	transient := genai.APIError{Code: 503, Status: "Service Unavailable"}
	fake := &fakeLLM{
		name: "fake",
		scripts: [][]step{
			{{nil, transient}},
			{{resp, nil}},
		},
	}
	wrapped := Wrap(fake, Config{
		MaxAttempts:  3,
		InitialDelay: time.Microsecond,
		MaxDelay:     time.Microsecond,
		Jitter:       -1, // no jitter
	})
	resps, errs := collect(wrapped.GenerateContent(context.Background(), &model.LLMRequest{}, false))
	if len(resps) != 1 || resps[0] != resp {
		t.Fatalf("resps = %v", resps)
	}
	if errs[0] != nil {
		t.Fatalf("err = %v", errs[0])
	}
	if got := fake.calls.Load(); got != 2 {
		t.Fatalf("calls = %d, want 2", got)
	}
}

func TestRetry_NonTransient_NoRetry(t *testing.T) {
	bad := errors.New("bad request")
	fake := &fakeLLM{
		name:    "fake",
		scripts: [][]step{{{nil, bad}}},
	}
	wrapped := Wrap(fake, Config{MaxAttempts: 5, InitialDelay: time.Microsecond, Jitter: -1})
	_, errs := collect(wrapped.GenerateContent(context.Background(), &model.LLMRequest{}, false))
	if !errors.Is(errs[0], bad) {
		t.Fatalf("err = %v, want bad", errs[0])
	}
	if got := fake.calls.Load(); got != 1 {
		t.Fatalf("calls = %d, want 1", got)
	}
}

func TestRetry_ExhaustedAttempts_ReturnsLastError(t *testing.T) {
	transient := genai.APIError{Code: 429, Status: "Too Many Requests"}
	fake := &fakeLLM{
		name:    "fake",
		scripts: [][]step{{{nil, transient}}, {{nil, transient}}, {{nil, transient}}},
	}
	wrapped := Wrap(fake, Config{MaxAttempts: 3, InitialDelay: time.Microsecond, MaxDelay: time.Microsecond, Jitter: -1})
	_, errs := collect(wrapped.GenerateContent(context.Background(), &model.LLMRequest{}, false))
	if errs[0] == nil {
		t.Fatal("expected error")
	}
	var apiErr genai.APIError
	if !errors.As(errs[0], &apiErr) || apiErr.Code != 429 {
		t.Fatalf("err = %v, want APIError 429", errs[0])
	}
	if got := fake.calls.Load(); got != 3 {
		t.Fatalf("calls = %d, want 3", got)
	}
}

func TestRetry_MidStreamErrorNotRetried(t *testing.T) {
	first := &model.LLMResponse{ModelVersion: "first"}
	transient := genai.APIError{Code: 503, Status: "boom"}
	fake := &fakeLLM{
		name: "fake",
		scripts: [][]step{
			{{first, nil}, {nil, transient}},
		},
	}
	wrapped := Wrap(fake, Config{MaxAttempts: 5, InitialDelay: time.Microsecond, Jitter: -1})
	resps, errs := collect(wrapped.GenerateContent(context.Background(), &model.LLMRequest{}, false))
	if len(resps) != 2 || resps[0] != first {
		t.Fatalf("resps = %v", resps)
	}
	if !errors.As(errs[1], new(genai.APIError)) {
		t.Fatalf("err[1] = %v, want passthrough APIError", errs[1])
	}
	if got := fake.calls.Load(); got != 1 {
		t.Fatalf("calls = %d, want 1 (no mid-stream retry)", got)
	}
}

func TestRetry_ContextCanceled_StopsImmediately(t *testing.T) {
	transient := genai.APIError{Code: 503}
	fake := &fakeLLM{
		name:    "fake",
		scripts: [][]step{{{nil, transient}}},
	}
	cfg := Config{
		MaxAttempts:  10,
		InitialDelay: 200 * time.Millisecond, // long enough to trip ctx.Done
		MaxDelay:     time.Second,
		Jitter:       -1,
	}
	wrapped := Wrap(fake, cfg)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately
	_, errs := collect(wrapped.GenerateContent(ctx, &model.LLMRequest{}, false))
	// Either the very first sleep fails to context, or shouldRetry returns
	// false because ctx is canceled. Both are correct stopping paths.
	if errs[0] == nil {
		t.Fatal("expected error after canceled ctx")
	}
}

func TestRetry_OnRetryHook(t *testing.T) {
	transient := genai.APIError{Code: 503}
	resp := &model.LLMResponse{}
	fake := &fakeLLM{
		name:    "fake",
		scripts: [][]step{{{nil, transient}}, {{resp, nil}}},
	}
	var hookHits atomic.Int32
	wrapped := Wrap(fake, Config{
		MaxAttempts:  3,
		InitialDelay: time.Microsecond,
		Jitter:       -1,
		OnRetry: func(attempt int, err error, delay time.Duration) {
			hookHits.Add(1)
			if attempt != 1 {
				t.Errorf("attempt = %d, want 1", attempt)
			}
		},
	})
	collect(wrapped.GenerateContent(context.Background(), &model.LLMRequest{}, false))
	if hookHits.Load() != 1 {
		t.Fatalf("hookHits = %d, want 1", hookHits.Load())
	}
}

func TestIsTransient(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"429", genai.APIError{Code: 429}, true},
		{"500", genai.APIError{Code: 500}, true},
		{"503", genai.APIError{Code: 503}, true},
		{"599", genai.APIError{Code: 599}, true},
		{"400", genai.APIError{Code: 400}, false},
		{"401", genai.APIError{Code: 401}, false},
		{"403", genai.APIError{Code: 403}, false},
		{"404", genai.APIError{Code: 404}, false},
		{"600", genai.APIError{Code: 600}, false},
		{"unexpected EOF", io.ErrUnexpectedEOF, true},
		{"rate limit string", errors.New("upstream rate limit reached"), true},
		{"deadline exceeded string", errors.New("context: deadline exceeded"), true},
		{"connection reset", errors.New("read tcp: connection reset by peer"), true},
		{"plain error", errors.New("invalid argument"), false},
		{"too many requests", errors.New("HTTP 429: too many requests"), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsTransient(tc.err); got != tc.want {
				t.Errorf("IsTransient(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
	_ = mustErr
}
