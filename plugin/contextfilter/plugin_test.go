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

package contextfilter_test

import (
	"context"
	"errors"
	"math"
	"slices"
	"sync"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"google.golang.org/genai"

	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/contextfilter"
)

func u(text string) *genai.Content { return genai.NewContentFromText(text, genai.RoleUser) }

func m(text string) *genai.Content { return genai.NewContentFromText(text, genai.RoleModel) }

func uc(parts ...*genai.Part) *genai.Content { return &genai.Content{Role: "user", Parts: parts} }

func mc(parts ...*genai.Part) *genai.Content { return &genai.Content{Role: "model", Parts: parts} }

func call(id string) *genai.Part {
	return &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: "t"}}
}

func resp(id string) *genai.Part {
	return &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: id, Name: "t"}}
}

func newPlugin(t *testing.T, cfg contextfilter.Config) *plugin.Plugin {
	t.Helper()
	p, err := contextfilter.New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return p
}

// runCallback calls the plugin's before-model callback on a request holding
// in, fails the test unless it returns (nil, nil), and returns the request's
// contents afterwards.
func runCallback(t *testing.T, ctx context.Context, p *plugin.Plugin, in []*genai.Content) []*genai.Content {
	t.Helper()
	cctx := icontext.NewCallbackContextWithDelta(icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{}), nil, nil)
	req := &model.LLMRequest{Contents: in}
	resp, err := p.BeforeModelCallback()(cctx, req)
	if resp != nil || err != nil {
		t.Fatalf("BeforeModelCallback() = (%v, %v), want (nil, nil)", resp, err)
	}
	return req.Contents
}

// pick returns the elements of in at the given indices.
func pick(in []*genai.Content, idx ...int) []*genai.Content {
	out := make([]*genai.Content, 0, len(idx))
	for _, i := range idx {
		out = append(out, in[i])
	}
	return out
}

// assertSameContents fails unless got and want have the same length and hold
// the same pointers in the same order.
func assertSameContents(t *testing.T, got, want []*genai.Content) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(req.Contents) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("req.Contents[%d] = %p, want %p", i, got[i], want[i])
		}
	}
}

// spanCapture starts a recorded span and returns its context and a function
// that ends the span and returns the messages of its exception events.
func spanCapture(t *testing.T) (context.Context, func() []string) {
	t.Helper()
	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
	ctx, span := tp.Tracer("test").Start(t.Context(), "invoke_agent")
	return ctx, func() []string {
		span.End()
		var msgs []string
		for _, ev := range sr.Ended()[0].Events() {
			if ev.Name != "exception" {
				continue
			}
			for _, kv := range ev.Attributes {
				if kv.Key == attribute.Key("exception.message") {
					msgs = append(msgs, kv.Value.AsString())
				}
			}
		}
		return msgs
	}
}

func assertEvents(t *testing.T, got []string, want ...string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("exception events = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("exception event %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestConfigName_DefaultAndCustom(t *testing.T) {
	tests := []struct {
		name string
		cfg  contextfilter.Config
		want string
	}{
		{name: "empty selects default", cfg: contextfilter.Config{}, want: "context_filter_plugin"},
		{name: "custom kept", cfg: contextfilter.Config{Name: "cf_recent"}, want: "cf_recent"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, tc.cfg)
			if got := p.Name(); got != tc.want {
				t.Errorf("Name() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestNew_InvalidRemoveAmount(t *testing.T) {
	zero, minusOne, one := 0, -1, 1
	valid := []struct {
		name string
		cfg  contextfilter.Config
	}{
		{name: "one", cfg: contextfilter.Config{NumInvocationsToKeep: 1, RemoveAmount: &one}},
		{name: "nil", cfg: contextfilter.Config{NumInvocationsToKeep: 1}},
	}
	for _, tc := range valid {
		t.Run(tc.name, func(t *testing.T) {
			p, err := contextfilter.New(tc.cfg)
			if err != nil || p == nil {
				t.Fatalf("New() = (%v, %v), want a plugin and nil error", p, err)
			}
		})
	}
	invalid := []struct {
		name    string
		cfg     contextfilter.Config
		wantErr string
	}{
		{name: "zero", cfg: contextfilter.Config{NumInvocationsToKeep: 1, RemoveAmount: &zero}, wantErr: "contextfilter: RemoveAmount must be at least 1, got 0"},
		{name: "negative", cfg: contextfilter.Config{NumInvocationsToKeep: 1, RemoveAmount: &minusOne}, wantErr: "contextfilter: RemoveAmount must be at least 1, got -1"},
		{name: "zero without trimming", cfg: contextfilter.Config{NumInvocationsToKeep: 0, RemoveAmount: &zero}, wantErr: "contextfilter: RemoveAmount must be at least 1, got 0"},
	}
	for _, tc := range invalid {
		t.Run(tc.name, func(t *testing.T) {
			p, err := contextfilter.New(tc.cfg)
			if err == nil || err.Error() != tc.wantErr {
				t.Fatalf("New() error = %v, want %q", err, tc.wantErr)
			}
			if p != nil {
				t.Errorf("New() plugin = %v, want nil", p)
			}
		})
	}
}

func TestCallbackReturnsNilNil(t *testing.T) {
	in := []*genai.Content{u("u1"), m("m1"), u("u2"), m("m2")}
	p := newPlugin(t, contextfilter.Config{NumInvocationsToKeep: 1})

	got := runCallback(t, t.Context(), p, in)

	assertSameContents(t, got, in[2:])
}

func TestNegativeNumInvocationsToKeep(t *testing.T) {
	in := []*genai.Content{u("u1"), m("m1"), u("u2"), m("m2")}
	p := newPlugin(t, contextfilter.Config{NumInvocationsToKeep: -1})

	got := runCallback(t, t.Context(), p, in)

	assertSameContents(t, got, in)
}

func TestInvocationStarts_NonUserRole(t *testing.T) {
	in := []*genai.Content{
		{Role: "", Parts: []*genai.Part{{Text: "r0"}}},
		m("m0"),
		{Role: "system", Parts: []*genai.Part{{Text: "s1"}}},
		m("m1"),
		u("u1"),
		m("m2"),
	}
	p := newPlugin(t, contextfilter.Config{NumInvocationsToKeep: 1})

	got := runCallback(t, t.Context(), p, in)

	assertSameContents(t, got, in)
}

func TestBeforeModel_InvocationCountingEdges(t *testing.T) {
	tests := []struct {
		name string
		in   []*genai.Content
	}{
		{
			name: "mixed parts",
			in:   []*genai.Content{u("u1"), uc(&genai.Part{Text: "note"}, resp("c1")), u("u2"), m("m2")},
		},
		{
			name: "nil parts",
			in:   []*genai.Content{{Role: "user", Parts: nil}, m("m1"), u("u2"), m("m2")},
		},
		{
			name: "consecutive user contents",
			in:   []*genai.Content{u("u1"), m("m1"), u("u2a"), u("u2b"), m("m2")},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, contextfilter.Config{NumInvocationsToKeep: 1})

			got := runCallback(t, t.Context(), p, tc.in)

			assertSameContents(t, got, tc.in[2:])
		})
	}
}

func TestBeforeModel_OrphanAdjustmentShiftsSplitLeft(t *testing.T) {
	tests := []struct {
		name string
		in   []*genai.Content
		keep int // index of in from which the request is kept
	}{
		{
			name: "split moves to the matching call",
			in:   []*genai.Content{u("a"), mc(call("c1")), m("waiting"), u("b"), m("ok"), uc(resp("c1"))},
			keep: 1,
		},
		{
			name: "call and response in one content",
			in:   []*genai.Content{u("u1"), m("m1"), u("u2"), mc(call("c1"), resp("c1")), m("m2")},
			keep: 2,
		},
		{
			name: "response without a matching call",
			in:   []*genai.Content{u("u1"), m("m1"), u("u2"), uc(resp("unmatched"))},
			keep: 0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, contextfilter.Config{NumInvocationsToKeep: 1})

			got := runCallback(t, t.Context(), p, tc.in)

			assertSameContents(t, got, tc.in[tc.keep:])
		})
	}
}

func TestBeforeModel_EmptyIDsNotPaired(t *testing.T) {
	tests := []struct {
		name string
		in   []*genai.Content
	}{
		{
			name: "response without ID",
			in:   []*genai.Content{u("u1"), m("m1"), u("u2"), uc(resp("")), m("m2")},
		},
		{
			name: "call and response without IDs",
			in:   []*genai.Content{u("u1"), mc(call("")), u("u2"), uc(resp("")), m("m2")},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, contextfilter.Config{NumInvocationsToKeep: 1})

			got := runCallback(t, t.Context(), p, tc.in)

			assertSameContents(t, got, tc.in[2:])
		})
	}
}

func TestRemoveAmount_Threshold(t *testing.T) {
	two, three := 2, 3
	threeInvocations := func() []*genai.Content {
		return []*genai.Content{u("u1"), m("m1"), u("u2"), m("m2"), u("u3"), m("m3")}
	}
	fourInvocations := func() []*genai.Content {
		return []*genai.Content{u("u1"), m("m1"), u("u2"), m("m2"), u("u3"), m("m3"), u("u4"), m("m4")}
	}
	tests := []struct {
		name string
		cfg  contextfilter.Config
		in   []*genai.Content
		keep int // index of in from which the request is kept
	}{
		{name: "keep 1, remove 2, 3 invocations", cfg: contextfilter.Config{NumInvocationsToKeep: 1, RemoveAmount: &two}, in: threeInvocations(), keep: 4},
		{name: "keep 1, remove 3, 3 invocations", cfg: contextfilter.Config{NumInvocationsToKeep: 1, RemoveAmount: &three}, in: threeInvocations(), keep: 0},
		{name: "keep 1, remove 3, 4 invocations", cfg: contextfilter.Config{NumInvocationsToKeep: 1, RemoveAmount: &three}, in: fourInvocations(), keep: 6},
		{name: "keep 2, remove 2, 4 invocations", cfg: contextfilter.Config{NumInvocationsToKeep: 2, RemoveAmount: &two}, in: fourInvocations(), keep: 4},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, tc.cfg)

			got := runCallback(t, t.Context(), p, tc.in)

			assertSameContents(t, got, tc.in[tc.keep:])
		})
	}
}

func TestIntegerOverflowSafety(t *testing.T) {
	maxInt := math.MaxInt
	tests := []struct {
		name string
		cfg  contextfilter.Config
	}{
		{name: "keep", cfg: contextfilter.Config{NumInvocationsToKeep: math.MaxInt}},
		{name: "remove", cfg: contextfilter.Config{NumInvocationsToKeep: 1, RemoveAmount: &maxInt}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in := []*genai.Content{u("u1"), m("m1"), u("u2"), m("m2")}
			p := newPlugin(t, tc.cfg)

			got := runCallback(t, t.Context(), p, in)

			assertSameContents(t, got, in)
		})
	}
}

func TestEmptyContents(t *testing.T) {
	var argLens []int
	p := newPlugin(t, contextfilter.Config{
		NumInvocationsToKeep: 1,
		CustomFilter: func(arg []*genai.Content) ([]*genai.Content, error) {
			argLens = append(argLens, len(arg))
			return arg, nil
		},
	})

	got := runCallback(t, t.Context(), p, []*genai.Content{})

	if !slices.Equal(argLens, []int{0}) {
		t.Errorf("CustomFilter argument lengths = %v, want [0]", argLens)
	}
	if len(got) != 0 {
		t.Errorf("len(req.Contents) = %d, want 0", len(got))
	}
}

func TestCustomFilterFailure_DiscardsTrim(t *testing.T) {
	in := []*genai.Content{u("u1"), m("m1"), u("u2"), m("m2"), u("u3"), m("m3")}
	want := slices.Clone(in) // in shares its array with the request
	var argLens []int
	p := newPlugin(t, contextfilter.Config{
		NumInvocationsToKeep: 1,
		CustomFilter: func(arg []*genai.Content) ([]*genai.Content, error) {
			argLens = append(argLens, len(arg))
			arg[0] = u("corrupted")
			return nil, errors.New("boom")
		},
	})
	ctx, events := spanCapture(t)

	got := runCallback(t, ctx, p, in)

	if !slices.Equal(argLens, []int{2}) {
		t.Errorf("CustomFilter argument lengths = %v, want [2]", argLens)
	}
	assertSameContents(t, got, want)
	assertEvents(t, events(), `plugin "context_filter_plugin": custom filter failed: boom`)
}

func TestCustomFilterPanic_Absorbed(t *testing.T) {
	in := []*genai.Content{u("u1"), m("m1"), u("u2"), m("m2")}
	p := newPlugin(t, contextfilter.Config{
		NumInvocationsToKeep: 1,
		CustomFilter: func([]*genai.Content) ([]*genai.Content, error) {
			panic("filter panic")
		},
	})
	ctx, events := spanCapture(t)

	got := runCallback(t, ctx, p, in)

	assertSameContents(t, got, in)
	assertEvents(t, events(), `plugin "context_filter_plugin": custom filter panicked: filter panic`)
}

func TestNilEntriesIgnored(t *testing.T) {
	text := func(s string) *genai.Part { return &genai.Part{Text: s} }
	nilContentInput := func() []*genai.Content {
		return []*genai.Content{nil, uc(nil, text("u1")), m("m1"), nil, uc(nil, text("u2")), mc(nil, call("c1")), uc(nil, resp("c1")), m("m2")}
	}
	const (
		nilContent = `plugin "context_filter_plugin": request holds a nil content`
		nilPart    = `plugin "context_filter_plugin": request holds a nil part`
	)
	tests := []struct {
		name          string
		in            []*genai.Content
		keep          int
		withFilter    bool
		want          []int // indices of in that the request holds afterwards
		wantFilterArg []int // lengths of the filter's arguments, one per call
		wantEvents    []string
	}{
		{
			name:       "nil content",
			in:         nilContentInput(),
			keep:       1,
			want:       []int{0, 1, 2, 3, 4, 5, 6, 7},
			wantEvents: []string{nilContent},
		},
		{
			name:       "nil part in a user content",
			in:         []*genai.Content{uc(nil, text("u1")), m("m1"), u("u2"), m("m2")},
			keep:       1,
			want:       []int{0, 1, 2, 3},
			wantEvents: []string{nilPart},
		},
		{
			name:       "nil part in the kept tail",
			in:         []*genai.Content{u("u1"), m("m1"), u("u2"), mc(nil, text("x"))},
			keep:       1,
			want:       []int{0, 1, 2, 3},
			wantEvents: []string{nilPart},
		},
		{
			name: "nil part never reached",
			in:   []*genai.Content{u("u1"), mc(nil, text("x")), u("u2"), m("m2")},
			keep: 1,
			want: []int{2, 3},
		},
		{
			name:          "no trimming, nothing touched",
			in:            []*genai.Content{nil, u("u1")},
			withFilter:    true,
			want:          []int{0, 1},
			wantFilterArg: []int{2},
		},
		{
			name:       "custom filter not run after a nil failure",
			in:         nilContentInput(),
			keep:       1,
			withFilter: true,
			want:       []int{0, 1, 2, 3, 4, 5, 6, 7},
			wantEvents: []string{nilContent},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var argLens []int
			cfg := contextfilter.Config{NumInvocationsToKeep: tc.keep}
			if tc.withFilter {
				cfg.CustomFilter = func(arg []*genai.Content) ([]*genai.Content, error) {
					argLens = append(argLens, len(arg))
					return arg, nil
				}
			}
			p := newPlugin(t, cfg)
			ctx, events := spanCapture(t)

			got := runCallback(t, ctx, p, tc.in)

			assertSameContents(t, got, pick(tc.in, tc.want...))
			if !slices.Equal(argLens, tc.wantFilterArg) {
				t.Errorf("CustomFilter argument lengths = %v, want %v", argLens, tc.wantFilterArg)
			}
			assertEvents(t, events(), tc.wantEvents...)
		})
	}
}

func TestConcurrentUse(t *testing.T) {
	const workers = 16
	p := newPlugin(t, contextfilter.Config{NumInvocationsToKeep: 1})
	cctx := icontext.NewCallbackContextWithDelta(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}), nil, nil)
	ins := make([][]*genai.Content, workers)
	reqs := make([]*model.LLMRequest, workers)
	errs := make([]error, workers)
	resps := make([]*model.LLMResponse, workers)
	for i := range workers {
		ins[i] = []*genai.Content{u("u1"), m("m1"), u("u2"), m("m2")}
		reqs[i] = &model.LLMRequest{Contents: ins[i]}
	}

	var wg sync.WaitGroup
	for i := range workers {
		wg.Go(func() {
			resps[i], errs[i] = p.BeforeModelCallback()(cctx, reqs[i])
		})
	}
	wg.Wait()

	for i := range workers {
		if resps[i] != nil || errs[i] != nil {
			t.Errorf("worker %d: BeforeModelCallback() = (%v, %v), want (nil, nil)", i, resps[i], errs[i])
		}
		assertSameContents(t, reqs[i].Contents, ins[i][2:])
	}
}
