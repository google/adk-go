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

package fallbackmodel

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"reflect"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go/v3"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropicmodel"
)

// fakeLLM is a model that yields its own name as text, and can be made to
// fail.
type fakeLLM struct {
	name string
	// err, when set, is yielded after errAfter responses, in place of the
	// final one.
	err      error
	errAfter int
	// edit, when set, edits the request when the model is called, as
	// Gemini's GenerateContent does before it returns.
	edit func(*model.LLMRequest)

	mu    sync.Mutex
	calls []call
}

// call is what a fakeLLM saw of one request, at the moment it was called.
type call struct {
	model    string
	contents []*genai.Content
	texts    []string
	config   *genai.GenerateContentConfig
	tools    map[string]any
}

func (f *fakeLLM) Name() string { return f.name }

func (f *fakeLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	c := call{
		model:    req.Model,
		contents: slices.Clone(req.Contents),
		texts:    texts(req.Contents),
		tools:    req.Tools,
	}
	if req.Config != nil {
		cfg := *req.Config
		c.config = &cfg
	}
	f.mu.Lock()
	f.calls = append(f.calls, c)
	f.mu.Unlock()
	if f.edit != nil {
		f.edit(req)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		for range f.errAfter {
			if !yield(f.response(), nil) {
				return
			}
		}
		if f.err != nil {
			yield(nil, f.err)
			return
		}
		yield(f.response(), nil)
	}
}

func (f *fakeLLM) response() *model.LLMResponse {
	return &model.LLMResponse{Content: genai.NewContentFromText(f.name, genai.RoleModel)}
}

func (f *fakeLLM) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

func healthy(name string) *fakeLLM { return &fakeLLM{name: name} }

// failing returns a model that fails with status code, wrapped the way
// model/gemini wraps the genai SDK's error. The error's message names the
// model, so a test can tell whose error reached the caller.
func failing(name string, code int) *fakeLLM {
	return &fakeLLM{name: name, err: apiError(name, code)}
}

func apiError(from string, code int) error {
	return fmt.Errorf("failed to call model: %w", genai.APIError{Code: code, Message: from})
}

func request() *model.LLMRequest {
	return &model.LLMRequest{
		Model:    "unset",
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
		Config:   &genai.GenerateContentConfig{},
	}
}

func texts(contents []*genai.Content) []string {
	var out []string
	for _, c := range contents {
		for _, p := range c.Parts {
			out = append(out, p.Text)
		}
	}
	return out
}

func newModel(t *testing.T, cfg Config) model.LLM {
	t.Helper()
	m, err := New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return m
}

func llms(fakes ...*fakeLLM) []model.LLM {
	out := make([]model.LLM, len(fakes))
	for i, f := range fakes {
		out[i] = f
	}
	return out
}

// collect returns the text of every response, stopping at the first error,
// which it returns. Like the flow, it skips a nil response.
func collect(t *testing.T, m model.LLM, req *model.LLMRequest, stream bool) ([]string, error) {
	t.Helper()
	var got []string
	for resp, err := range m.GenerateContent(t.Context(), req, stream) {
		if err != nil {
			return got, err
		}
		if resp == nil {
			continue
		}
		got = append(got, texts([]*genai.Content{resp.Content})...)
	}
	return got, nil
}

func TestGenerateContent(t *testing.T) {
	tests := []struct {
		name   string
		models []*fakeLLM
		codes  []int
		// wantTexts is the text of each response the caller gets.
		wantTexts []string
		// wantErrFrom is the index of the model whose error reaches the
		// caller unchanged, or -1 when none does.
		wantErrFrom int
		// wantCalls is how many times each model is called.
		wantCalls []int
	}{
		{
			name:        "primary success does not touch backup",
			models:      []*fakeLLM{healthy("primary"), healthy("backup")},
			wantTexts:   []string{"primary"},
			wantErrFrom: -1,
			wantCalls:   []int{1, 0},
		},
		{
			name:        "retriable status falls back",
			models:      []*fakeLLM{failing("primary", http.StatusTooManyRequests), healthy("backup")},
			wantTexts:   []string{"backup"},
			wantErrFrom: -1,
			wantCalls:   []int{1, 1},
		},
		{
			name:        "falls back through several models",
			models:      []*fakeLLM{failing("first", http.StatusTooManyRequests), failing("second", http.StatusServiceUnavailable), healthy("third")},
			wantTexts:   []string{"third"},
			wantErrFrom: -1,
			wantCalls:   []int{1, 1, 1},
		},
		{
			name:        "non-retriable status propagates",
			models:      []*fakeLLM{failing("primary", http.StatusBadRequest), healthy("backup")},
			wantErrFrom: 0,
			wantCalls:   []int{1, 0},
		},
		{
			name:        "timeout does not fall back by default",
			models:      []*fakeLLM{failing("primary", http.StatusRequestTimeout), healthy("backup")},
			wantErrFrom: 0,
			wantCalls:   []int{1, 0},
		},
		{
			name:        "all models failing returns the last error",
			models:      []*fakeLLM{failing("first", http.StatusTooManyRequests), failing("second", http.StatusServiceUnavailable)},
			wantErrFrom: 1,
			wantCalls:   []int{1, 1},
		},
		{
			name:        "status codes are configurable",
			models:      []*fakeLLM{failing("primary", http.StatusBadRequest), healthy("backup")},
			codes:       []int{http.StatusBadRequest},
			wantTexts:   []string{"backup"},
			wantErrFrom: -1,
			wantCalls:   []int{1, 1},
		},
		{
			name:        "configured status codes replace the defaults",
			models:      []*fakeLLM{failing("primary", http.StatusTooManyRequests), healthy("backup")},
			codes:       []int{http.StatusBadRequest},
			wantErrFrom: 0,
			wantCalls:   []int{1, 0},
		},
		{
			name:        "widened defaults add a status",
			models:      []*fakeLLM{failing("primary", anthropicmodel.StatusOverloaded), healthy("backup")},
			codes:       append(DefaultStatusCodes(), anthropicmodel.StatusOverloaded),
			wantTexts:   []string{"backup"},
			wantErrFrom: -1,
			wantCalls:   []int{1, 1},
		},
		{
			name:        "widened defaults keep the defaults",
			models:      []*fakeLLM{failing("primary", http.StatusTooManyRequests), healthy("backup")},
			codes:       append(DefaultStatusCodes(), anthropicmodel.StatusOverloaded),
			wantTexts:   []string{"backup"},
			wantErrFrom: -1,
			wantCalls:   []int{1, 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := newModel(t, Config{Models: llms(tt.models...), RetriableStatusCodes: tt.codes})

			got, err := collect(t, m, request(), false)

			if diff := cmp.Diff(tt.wantTexts, got); diff != "" {
				t.Errorf("response texts mismatch (-want +got):\n%s", diff)
			}
			if tt.wantErrFrom < 0 {
				if err != nil {
					t.Errorf("GenerateContent() error = %v, want nil", err)
				}
			} else if want := tt.models[tt.wantErrFrom].err; err != want {
				t.Errorf("GenerateContent() error = %v, want the error of model %q unchanged", err, tt.models[tt.wantErrFrom].name)
			}
			var calls []int
			for _, f := range tt.models {
				calls = append(calls, f.callCount())
			}
			if diff := cmp.Diff(tt.wantCalls, calls); diff != "" {
				t.Errorf("calls per model mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestErrorWithoutStatusPropagates(t *testing.T) {
	errBadRequest := errors.New("bad request")
	backup := healthy("backup")
	m := newModel(t, Config{Models: llms(&fakeLLM{name: "primary", err: errBadRequest}, backup)})

	_, err := collect(t, m, request(), false)

	if !errors.Is(err, errBadRequest) {
		t.Errorf("GenerateContent() error = %v, want %v", err, errBadRequest)
	}
	if n := backup.callCount(); n != 0 {
		t.Errorf("backup called %d times, want 0", n)
	}
}

func TestStreamingFailureAfterFirstChunkDoesNotFallBack(t *testing.T) {
	primary := &fakeLLM{name: "primary", err: apiError("primary", http.StatusTooManyRequests), errAfter: 2}
	backup := healthy("backup")
	m := newModel(t, Config{Models: llms(primary, backup)})

	got, err := collect(t, m, request(), true)

	// The chunks the primary already sent stay with the caller, and the
	// backup is never asked to finish a turn the primary began.
	if diff := cmp.Diff([]string{"primary", "primary"}, got); diff != "" {
		t.Errorf("response texts mismatch (-want +got):\n%s", diff)
	}
	if err != primary.err {
		t.Errorf("GenerateContent() error = %v, want the primary's error unchanged", err)
	}
	if n := backup.callCount(); n != 0 {
		t.Errorf("backup called %d times, want 0", n)
	}
}

func TestNilResponseDoesNotCommitTheTurn(t *testing.T) {
	primary := &nilThenFailingLLM{fakeLLM: failing("primary", http.StatusServiceUnavailable)}
	m := newModel(t, Config{Models: []model.LLM{primary, healthy("backup")}})

	got, err := collect(t, m, request(), true)
	if err != nil {
		t.Fatalf("GenerateContent() error = %v", err)
	}
	if diff := cmp.Diff([]string{"backup"}, got); diff != "" {
		t.Errorf("response texts mismatch (-want +got):\n%s", diff)
	}
}

// nilThenFailingLLM yields a nil response, as some third-party models
// do, before it fails.
type nilThenFailingLLM struct{ *fakeLLM }

func (f *nilThenFailingLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if !yield(nil, nil) {
			return
		}
		yield(nil, f.err)
	}
}

func TestStreamGoesOnAfterAnErrorThatDoesNotFallBack(t *testing.T) {
	// The flow keeps reading a stream after an error when an on-model-error
	// callback answers it, so the wrapper must not end the stream there.
	errBadRequest := errors.New("bad request")
	primary := &errorThenTextLLM{name: "primary", err: errBadRequest}
	m := newModel(t, Config{Models: []model.LLM{primary, healthy("backup")}})

	var gotErrs []error
	var gotTexts []string
	for resp, err := range m.GenerateContent(t.Context(), request(), true) {
		if err != nil {
			gotErrs = append(gotErrs, err)
			continue
		}
		gotTexts = append(gotTexts, texts([]*genai.Content{resp.Content})...)
	}

	if len(gotErrs) != 1 || gotErrs[0] != errBadRequest {
		t.Errorf("errors = %v, want [%v]", gotErrs, errBadRequest)
	}
	if diff := cmp.Diff([]string{"after the error"}, gotTexts); diff != "" {
		t.Errorf("response texts mismatch (-want +got):\n%s", diff)
	}
}

// errorThenTextLLM yields an error and then a response.
type errorThenTextLLM struct {
	name string
	err  error
}

func (f *errorThenTextLLM) Name() string { return f.name }

func (f *errorThenTextLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if !yield(nil, f.err) {
			return
		}
		yield(&model.LLMResponse{Content: genai.NewContentFromText("after the error", genai.RoleModel)}, nil)
	}
}

func TestErrorPassedOnCommitsTheTurn(t *testing.T) {
	// An on-model-error callback can turn an error into an event the caller
	// keeps, so once an error has been passed on, a later retriable error
	// must not move on either.
	errBadRequest := errors.New("bad request")
	primary := &errorsLLM{name: "primary", errs: []error{errBadRequest, apiError("primary", http.StatusServiceUnavailable)}}
	backup := healthy("backup")
	m := newModel(t, Config{Models: []model.LLM{primary, backup}})

	var gotErrs []error
	for _, err := range m.GenerateContent(t.Context(), request(), true) {
		if err != nil {
			gotErrs = append(gotErrs, err)
		}
	}

	if len(gotErrs) != 2 || gotErrs[0] != primary.errs[0] || gotErrs[1] != primary.errs[1] {
		t.Errorf("errors = %v, want both of the primary's errors unchanged", gotErrs)
	}
	if n := backup.callCount(); n != 0 {
		t.Errorf("backup called %d times, want 0", n)
	}
}

// errorsLLM yields each of its errors in turn.
type errorsLLM struct {
	name string
	errs []error
}

func (f *errorsLLM) Name() string { return f.name }

func (f *errorsLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		for _, err := range f.errs {
			if !yield(nil, err) {
				return
			}
		}
	}
}

func TestCanceledContextDoesNotFallBack(t *testing.T) {
	// The fake ignores the context and fails with a retriable status, as a
	// provider can when the caller goes away mid-request.
	primary := failing("primary", http.StatusServiceUnavailable)
	backup := healthy("backup")
	m := newModel(t, Config{Models: llms(primary, backup)})
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	var gotErr error
	for _, err := range m.GenerateContent(ctx, request(), false) {
		gotErr = err
	}

	if gotErr != primary.err {
		t.Errorf("GenerateContent() error = %v, want the primary's error unchanged", gotErr)
	}
	if n := backup.callCount(); n != 0 {
		t.Errorf("backup called %d times, want 0", n)
	}
}

func TestRequestModelPointsAtTheDelegate(t *testing.T) {
	primary := failing("primary", http.StatusTooManyRequests)
	backup := healthy("backup")
	m := newModel(t, Config{Models: llms(primary, backup)})
	req := request()

	if _, err := collect(t, m, req, false); err != nil {
		t.Fatalf("GenerateContent() error = %v", err)
	}

	if got := primary.calls[0].model; got != "primary" {
		t.Errorf("primary saw req.Model = %q, want %q", got, "primary")
	}
	if got := backup.calls[0].model; got != "backup" {
		t.Errorf("backup saw req.Model = %q, want %q", got, "backup")
	}
	// The request is left naming the model that served it.
	if req.Model != "backup" {
		t.Errorf("req.Model = %q after the call, want %q", req.Model, "backup")
	}
}

func TestNameIsThePrimarysName(t *testing.T) {
	m := newModel(t, Config{Models: llms(healthy("primary"), healthy("backup"))})

	if got := m.Name(); got != "primary" {
		t.Errorf("Name() = %q, want %q", got, "primary")
	}
}

func TestNewRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{name: "nil models", cfg: Config{}},
		{name: "empty models", cfg: Config{Models: []model.LLM{}}},
		{name: "nil model", cfg: Config{Models: []model.LLM{healthy("primary"), nil}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := New(tt.cfg); err == nil {
				t.Error("New() error = nil, want an error")
			}
		})
	}
}

func TestNewCopiesItsConfig(t *testing.T) {
	backup := healthy("backup")
	cfg := Config{
		Models:               llms(failing("primary", http.StatusBadRequest), backup),
		RetriableStatusCodes: []int{http.StatusBadRequest},
	}
	m := newModel(t, cfg)
	cfg.Models[1] = healthy("replaced")
	cfg.RetriableStatusCodes[0] = http.StatusTeapot

	got, err := collect(t, m, request(), false)
	if err != nil {
		t.Fatalf("GenerateContent() error = %v", err)
	}
	if diff := cmp.Diff([]string{"backup"}, got); diff != "" {
		t.Errorf("response texts mismatch (-want +got):\n%s", diff)
	}
}

func TestDefaultStatusCodes(t *testing.T) {
	// 408 is left out on purpose: a timeout does not say whether the request
	// was processed. Pinning the set keeps an unrelated edit from undoing
	// that.
	want := []int{429, 500, 502, 503, 504}

	got := DefaultStatusCodes()
	slices.Sort(got)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("DefaultStatusCodes() mismatch (-want +got):\n%s", diff)
	}

	got[0] = http.StatusTeapot
	if again := DefaultStatusCodes(); slices.Contains(again, http.StatusTeapot) {
		t.Errorf("DefaultStatusCodes() = %v after the caller edited an earlier result, want a fresh slice", again)
	}
}

func TestStatusCode(t *testing.T) {
	tests := []struct {
		name       string
		err        error
		wantStatus int
		wantOK     bool
	}{
		{
			name:       "genai error as model/gemini wraps it",
			err:        fmt.Errorf("failed to call model: %w", genai.APIError{Code: 503}),
			wantStatus: 503,
			wantOK:     true,
		},
		{
			name:       "openai error as model/openaimodel wraps it",
			err:        fmt.Errorf("openai: call failed: %w", &openai.Error{StatusCode: 429}),
			wantStatus: 429,
			wantOK:     true,
		},
		{
			name:       "anthropicmodel error",
			err:        &anthropicmodel.APIError{StatusCode: anthropicmodel.StatusOverloaded, Err: anthropicmodel.ErrOverloaded},
			wantStatus: anthropicmodel.StatusOverloaded,
			wantOK:     true,
		},
		{
			name:       "wrapped anthropicmodel error",
			err:        fmt.Errorf("agent: %w", &anthropicmodel.APIError{StatusCode: 500, Err: anthropicmodel.ErrAPIStatus}),
			wantStatus: 500,
			wantOK:     true,
		},
		{name: "error without a status", err: errors.New("nope")},
		{name: "context canceled", err: context.Canceled},
		{name: "deadline exceeded", err: fmt.Errorf("request: %w", context.DeadlineExceeded)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status, ok := statusCode(tt.err)
			if status != tt.wantStatus || ok != tt.wantOK {
				t.Errorf("statusCode() = (%d, %t), want (%d, %t)", status, ok, tt.wantStatus, tt.wantOK)
			}
		})
	}
}

// variantLLM is a model that reports a Google backend variant.
type variantLLM struct {
	*fakeLLM
	variant genai.Backend
}

func (v *variantLLM) GetGoogleLLMVariant() genai.Backend { return v.variant }

func TestBackendVariantComesFromThePrimary(t *testing.T) {
	tests := []struct {
		name   string
		models []model.LLM
		want   genai.Backend
	}{
		{
			name: "primary reports a variant",
			models: []model.LLM{
				&variantLLM{fakeLLM: healthy("primary"), variant: genai.BackendVertexAI},
				&variantLLM{fakeLLM: healthy("backup"), variant: genai.BackendGeminiAPI},
			},
			want: genai.BackendVertexAI,
		},
		{
			name: "primary reports none",
			models: []model.LLM{
				healthy("primary"),
				&variantLLM{fakeLLM: healthy("backup"), variant: genai.BackendGeminiAPI},
			},
			want: genai.BackendUnspecified,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := newModel(t, Config{Models: tt.models})

			if got := googlellm.GetGoogleLLMVariant(m); got != tt.want {
				t.Errorf("GetGoogleLLMVariant() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDoesNotOfferALiveClient(t *testing.T) {
	// The live flow runs only on a model with a Client method. Forwarding the
	// primary's client would run a live session with no fallback at all, so
	// the model has none, and live mode reports that it is unsupported.
	m := newModel(t, Config{Models: llms(healthy("primary"))})

	if _, ok := m.(interface{ Client() *genai.Client }); ok {
		t.Error("the fallback model has a Client method, want none")
	}
}

func TestFailedModelDoesNotLeakItsEdits(t *testing.T) {
	// Real models edit the request before sending it: Gemini appends a user
	// turn, and allocates HTTP options to write headers into. A model that then
	// fails must not hand those edits to the next one.
	primary := failing("primary", http.StatusTooManyRequests)
	primary.edit = func(req *model.LLMRequest) {
		req.Contents[0] = genai.NewContentFromText("replaced by primary", genai.RoleUser)
		req.Contents = append(req.Contents, genai.NewContentFromText("appended by primary", genai.RoleUser))
		req.Config.Temperature = new(float32(0.99))
		req.Config.HTTPOptions = &genai.HTTPOptions{Headers: http.Header{"User-Agent": {"primary"}}}
		req.Config.Labels["agent"] = "primary"
		req.Config.Tools[0].FunctionDeclarations[0].Description = "edited by primary"
	}
	backup := healthy("backup")
	m := newModel(t, Config{Models: llms(primary, backup)})
	req := request()
	req.Config.Labels = map[string]string{"agent": "caller"}
	req.Config.Tools = []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "lookup", Description: "from the caller"}}}}

	if _, err := collect(t, m, req, false); err != nil {
		t.Fatalf("GenerateContent() error = %v", err)
	}

	seen := backup.calls[0]
	if diff := cmp.Diff([]string{"hi"}, seen.texts); diff != "" {
		t.Errorf("backup saw contents mismatch (-want +got):\n%s", diff)
	}
	if seen.config.Temperature != nil {
		t.Errorf("backup saw Temperature = %v, want nil", *seen.config.Temperature)
	}
	if seen.config.HTTPOptions != nil {
		t.Errorf("backup saw HTTPOptions = %+v, want nil", seen.config.HTTPOptions)
	}
	if got := seen.config.Labels["agent"]; got != "caller" {
		t.Errorf("backup saw label %q, want %q", got, "caller")
	}
	if got := seen.config.Tools[0].FunctionDeclarations[0].Description; got != "from the caller" {
		t.Errorf("backup saw tool description %q, want %q", got, "from the caller")
	}
	if diff := cmp.Diff([]string{"hi"}, texts(req.Contents)); diff != "" {
		t.Errorf("request contents after the call mismatch (-want +got):\n%s", diff)
	}
	if req.Config.Temperature != nil {
		t.Errorf("request Temperature after the call = %v, want nil", *req.Config.Temperature)
	}
}

func TestFailedModelDoesNotLeakEditsInsideTheCallersOptions(t *testing.T) {
	// When the caller set HTTP options, Gemini writes its headers into the
	// caller's header map rather than a new one, so only a copy in depth keeps
	// them from the next model.
	primary := failing("primary", http.StatusTooManyRequests)
	primary.edit = func(req *model.LLMRequest) {
		req.Config.HTTPOptions.Headers.Set("User-Agent", "primary")
		req.Config.HTTPOptions.ExtraBody["generation"].(map[string]any)["seed"] = 2
	}
	backup := healthy("backup")
	m := newModel(t, Config{Models: llms(primary, backup)})
	req := request()
	req.Config.HTTPOptions = &genai.HTTPOptions{
		Headers:   http.Header{"X-Caller": {"yes"}},
		ExtraBody: map[string]any{"generation": map[string]any{"seed": 1}},
	}

	if _, err := collect(t, m, req, false); err != nil {
		t.Fatalf("GenerateContent() error = %v", err)
	}

	seen := backup.calls[0].config.HTTPOptions
	if diff := cmp.Diff(http.Header{"X-Caller": {"yes"}}, seen.Headers); diff != "" {
		t.Errorf("backup saw headers mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"generation": map[string]any{"seed": 1}}, seen.ExtraBody); diff != "" {
		t.Errorf("backup saw ExtraBody mismatch (-want +got):\n%s", diff)
	}
}

func TestSucceedingModelKeepsItsEdits(t *testing.T) {
	// Only a failed call is rolled back. What the model that served the call
	// sent is what traces should show.
	primary := healthy("primary")
	primary.edit = func(req *model.LLMRequest) {
		req.Contents = append(req.Contents, genai.NewContentFromText("edited", genai.RoleUser))
		req.Config.Temperature = new(float32(0.99))
	}
	m := newModel(t, Config{Models: llms(primary, healthy("backup"))})
	req := request()

	if _, err := collect(t, m, req, false); err != nil {
		t.Fatalf("GenerateContent() error = %v", err)
	}

	if diff := cmp.Diff([]string{"hi", "edited"}, texts(req.Contents)); diff != "" {
		t.Errorf("request contents mismatch (-want +got):\n%s", diff)
	}
	if req.Config.Temperature == nil || *req.Config.Temperature != 0.99 {
		t.Errorf("request Temperature = %v, want 0.99", req.Config.Temperature)
	}
}

func TestRollbackKeepsContentIdentity(t *testing.T) {
	// The rollback restores the caller's own Content values rather than
	// copies. This is the Go side of adk-python's test that a rollback keeps
	// transient contents recognizable, and it fails if the contents are ever
	// deep-copied.
	primary := failing("primary", http.StatusTooManyRequests)
	primary.edit = func(req *model.LLMRequest) {
		req.Contents = append(req.Contents, genai.NewContentFromText("appended by primary", genai.RoleUser))
	}
	backup := healthy("backup")
	m := newModel(t, Config{Models: llms(primary, backup)})
	req := request()
	first := req.Contents[0]
	second := genai.NewContentFromText("inserted by ADK", genai.RoleUser)
	req.Contents = append(req.Contents, second)

	if _, err := collect(t, m, req, false); err != nil {
		t.Fatalf("GenerateContent() error = %v", err)
	}

	seen := backup.calls[0].contents
	if len(seen) != 2 || seen[0] != first || seen[1] != second {
		t.Errorf("backup saw contents %v, want the caller's two Content values %p and %p", seen, first, second)
	}
}

// lockedTool holds a value that must not be copied, as an MCP tool's session
// does.
type lockedTool struct {
	mu sync.Mutex //nolint:unused // only its presence matters
}

func TestToolsAreSharedNotCopied(t *testing.T) {
	backup := healthy("backup")
	m := newModel(t, Config{Models: llms(failing("primary", http.StatusTooManyRequests), backup)})
	req := request()
	tool := &lockedTool{}
	req.Tools = map[string]any{"locked": tool}

	if _, err := collect(t, m, req, false); err != nil {
		t.Fatalf("GenerateContent() error = %v", err)
	}

	if got := backup.calls[0].tools["locked"]; got != tool {
		t.Errorf("backup saw tool %p, want the caller's %p", got, tool)
	}
	if got := req.Tools["locked"]; got != tool {
		t.Errorf("request tool after the call = %p, want the caller's %p", got, tool)
	}
}

func TestEveryRequestFieldIsAccountedFor(t *testing.T) {
	// A model edits the request in place, so a new LLMRequest field that a
	// model edits would leak from a failed call into the next one. This fails
	// until a field added to LLMRequest is sorted: restored by snapshot, or
	// deliberately left.
	// TestFailedModelDoesNotLeakItsEdits is what proves a restored field is
	// restored.
	restored := []string{"Contents", "Config"}
	left := []string{
		"Model",                    // Set for each call.
		"Tools",                    // Live tool values, shared rather than copied.
		"CacheConfig",              // Set by the flow, read by models.
		"CacheMetadata",            // Set by the flow, read by models.
		"PreviousPromptTokenCount", // Set by the flow, read by models.
	}

	var fields []string
	for _, f := range reflect.VisibleFields(reflect.TypeFor[model.LLMRequest]()) {
		fields = append(fields, f.Name)
	}
	want := slices.Concat(restored, left)
	slices.Sort(want)
	slices.Sort(fields)
	if diff := cmp.Diff(want, fields); diff != "" {
		t.Errorf("model.LLMRequest has fields this package does not account for (-accounted +actual):\n%s\n"+
			"If a model edits a new field, restore it in snapshot and add it to `restored`; otherwise add it to `left`.", diff)
	}
}

func TestSnapshotKeepsUnexportedFields(t *testing.T) {
	// A Google Search time range holds time.Time values, whose fields are
	// unexported. They are copied by value rather than zeroed.
	end := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	req := request()
	req.Config.Tools = []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{TimeRangeFilter: &genai.Interval{EndTime: end}}}}

	s := takeSnapshot(req)

	got := s.config.Tools[0].GoogleSearch.TimeRangeFilter
	if got == req.Config.Tools[0].GoogleSearch.TimeRangeFilter {
		t.Error("snapshot shares the time range with the request, want a copy")
	}
	if !got.EndTime.Equal(end) {
		t.Errorf("snapshot EndTime = %v, want %v", got.EndTime, end)
	}
}

func TestSnapshotCopiesBytes(t *testing.T) {
	// Bytes are copied with reflect.Copy rather than one by one, and must
	// still not be shared.
	req := request()
	req.Config.SystemInstruction = &genai.Content{Parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("caller")}}}}

	s := takeSnapshot(req)
	req.Config.SystemInstruction.Parts[0].InlineData.Data[0] = 'X'

	if got := string(s.config.SystemInstruction.Parts[0].InlineData.Data); got != "caller" {
		t.Errorf("snapshot inline data = %q after the request's bytes were edited, want %q", got, "caller")
	}
}

func TestSnapshotOfACyclicValueEnds(t *testing.T) {
	// Nothing stops a caller from putting a cyclic value in ExtraBody, and
	// copying one without a bound would overflow the stack.
	cyclic := map[string]any{}
	cyclic["self"] = cyclic
	req := request()
	req.Config.HTTPOptions = &genai.HTTPOptions{ExtraBody: cyclic}

	s := takeSnapshot(req)

	if s.config.HTTPOptions == req.Config.HTTPOptions {
		t.Error("snapshot shares HTTPOptions with the request, want a copy")
	}
}

func TestAbandoningTheStreamStopsTheDelegate(t *testing.T) {
	// The flow stops reading early when a callback fails or the client goes
	// away. The wrapper has to pass that on, so the delegate's stream, and the
	// provider connection under it, is released.
	primary := &endlessLLM{name: "primary"}
	m := newModel(t, Config{Models: []model.LLM{primary, healthy("backup")}})

	for _, err := range m.GenerateContent(t.Context(), request(), true) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		break
	}

	if !primary.stopped {
		t.Error("the primary's stream was not stopped when the caller stopped reading")
	}
}

// endlessLLM streams chunks until its consumer stops.
type endlessLLM struct {
	name    string
	stopped bool
}

func (f *endlessLLM) Name() string { return f.name }

func (f *endlessLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		defer func() { f.stopped = true }()
		for i := range 100 {
			if !yield(&model.LLMResponse{Content: genai.NewContentFromText(fmt.Sprint("chunk", i), genai.RoleModel), Partial: true}, nil) {
				return
			}
		}
	}
}

func TestLogsEachFallback(t *testing.T) {
	h := &recordingHandler{}
	m := newModel(t, Config{
		Models: llms(failing("primary", http.StatusTooManyRequests), failing("second", http.StatusServiceUnavailable), healthy("third")),
		Logger: slog.New(h),
	})

	if _, err := collect(t, m, request(), false); err != nil {
		t.Fatalf("GenerateContent() error = %v", err)
	}

	want := []string{
		"DEBUG falling back to the next model model=primary status_code=429 next_model=second",
		"DEBUG falling back to the next model model=second status_code=503 next_model=third",
	}
	if diff := cmp.Diff(want, h.records); diff != "" {
		t.Errorf("log records mismatch (-want +got):\n%s", diff)
	}
}

// recordingHandler records each record's level, message and attributes.
type recordingHandler struct {
	mu      sync.Mutex
	records []string
}

func (h *recordingHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *recordingHandler) Handle(_ context.Context, r slog.Record) error {
	line := r.Level.String() + " " + r.Message
	r.Attrs(func(a slog.Attr) bool {
		line += " " + a.String()
		return true
	})
	h.mu.Lock()
	h.records = append(h.records, line)
	h.mu.Unlock()
	return nil
}

func (h *recordingHandler) WithAttrs([]slog.Attr) slog.Handler { return h }

func (h *recordingHandler) WithGroup(string) slog.Handler { return h }

func TestConcurrentCalls(t *testing.T) {
	// One model serves every request of an agent, so calls share it. Run under
	// -race.
	m := newModel(t, Config{Models: llms(failing("primary", http.StatusTooManyRequests), healthy("backup"))})

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			got, err := collect(t, m, request(), false)
			if err != nil || len(got) != 1 || got[0] != "backup" {
				t.Errorf("GenerateContent() = %v, %v, want [backup], nil", got, err)
			}
		})
	}
	wg.Wait()
}
