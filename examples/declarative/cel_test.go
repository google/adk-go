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
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

// TestCELResultValues checks the Go values a CEL result becomes, which is
// what a cel node stores in its output and state_delta.
func TestCELResultValues(t *testing.T) {
	tests := []struct {
		name string
		expr string
		want any
	}{
		{name: "nested map and list", expr: "{'a': [1, 2.5, null, true], 'b': {'c': 'd'}}", want: map[string]any{"a": []any{int64(1), 2.5, nil, true}, "b": map[string]any{"c": "d"}}},
		{name: "unsigned int", expr: "1u", want: uint64(1)},
		{name: "empty optional", expr: "optional.none()", want: nil},
		{name: "present optional", expr: "optional.of('x')", want: "x"},
		{name: "bytes", expr: "b'hi'", want: []byte("hi")},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e, err := compileCEL(tc.expr, nil)
			if err != nil {
				t.Fatalf("compileCEL(%q) error = %v", tc.expr, err)
			}

			got, err := e.eval(map[string]any{})
			if err != nil {
				t.Fatalf("eval(%q) error = %v", tc.expr, err)
			}

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("eval(%q) mismatch (-want +got):\n%s", tc.expr, diff)
			}
		})
	}
}

func TestCELResultErrors(t *testing.T) {
	tests := []struct {
		name       string
		expr       string
		activation map[string]any
		wantEval   bool // the error is errEval, which hides cel-go's message
	}{
		{name: "map with a non-string key", expr: "{1: 'x'}"},
		{name: "over the cost limit", expr: "input.map(x, input.map(y, x + y))", activation: map[string]any{"input": slices.Repeat([]any{int64(1)}, 2000)}, wantEval: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e, err := compileCEL(tc.expr, nil)
			if err != nil {
				t.Fatalf("compileCEL(%q) error = %v", tc.expr, err)
			}

			_, err = e.eval(tc.activation)

			if err == nil {
				t.Fatalf("eval(%q) error = nil, want an error", tc.expr)
			}
			if got := errors.Is(err, errEval); got != tc.wantEval {
				t.Errorf("eval(%q) error = %v, errors.Is(err, errEval) = %t, want %t", tc.expr, err, got, tc.wantEval)
			}
		})
	}
}

// TestPlainValue checks what a cel node sees as input and state.
func TestPlainValue(t *testing.T) {
	type pair struct {
		N int     `json:"n"`
		F float64 `json:"f"`
	}
	tests := []struct {
		name string
		in   any
		want any
	}{
		{name: "nil", in: nil, want: nil},
		{name: "int", in: 3, want: int64(3)},
		// A double one node outputs is a double in the next, whole or not.
		{name: "whole float", in: 2.0, want: 2.0},
		{name: "fractional float", in: 2.5, want: 2.5},
		{name: "content", in: genai.NewContentFromText("hi", genai.RoleUser), want: "hi"},
		{name: "struct", in: pair{N: 2, F: 0.5}, want: map[string]any{"n": int64(2), "f": 0.5}},
		{name: "nested", in: map[string]any{"l": []int{1, 2}}, want: map[string]any{"l": []any{int64(1), int64(2)}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, plainValue(tc.in)); diff != "" {
				t.Errorf("plainValue(%v) mismatch (-want +got):\n%s", tc.in, diff)
			}
		})
	}
}

// TestContextVarsConvertsStateByKey checks that a state value with no JSON
// form becomes a string on its own, rather than turning the whole state into
// one string.
// TestContextVarsConvertsStateByKey checks each state value on its own: a
// whole float64, which is what a JSON session store returns for 2, becomes an
// int64, and a value with no JSON form becomes a string without turning the
// whole state into one.
func TestContextVarsConvertsStateByKey(t *testing.T) {
	resp, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{
		AppName: "app",
		UserID:  "user",
		State: map[string]any{
			"turn": 2, "stored": 2.0, "ratio": 2.5, "huge": 1e300, "low": -1e300, "nan": math.NaN(),
			"callback": func() {},
		},
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ctx := icontext.NewReadonlyContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Session: resp.Session,
		Agent:   loadTestSystem(t, &fakeModel{})[0],
	}))

	state, ok := contextVars(ctx)["state"].(map[string]any)
	if !ok {
		t.Fatalf("state = %T, want map[string]any", contextVars(ctx)["state"])
	}
	if _, ok := state["callback"].(string); !ok {
		t.Errorf("state.callback = %T, want string", state["callback"])
	}
	delete(state, "callback")
	// 1e300 and -1e300 are whole, but no int64 holds them.
	want := map[string]any{
		"turn": int64(2), "stored": int64(2), "ratio": 2.5, "huge": 1e300, "low": -1e300, "nan": math.NaN(),
	}
	if diff := cmp.Diff(want, state, cmpopts.EquateNaNs()); diff != "" {
		t.Errorf("state mismatch (-want +got):\n%s", diff)
	}
}
