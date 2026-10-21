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

package shared

import (
	"errors"
	"testing"
	"time"

	"google.golang.org/genai"
)

// RequestTimeout must be safe on its own terms. Each endpoint package's
// applyGenerationConfig rejects a non-positive timeout before any request is
// built, so this guard is defense in depth — and it is tested directly rather
// than assumed unreachable, because a guard that only holds while callers keep
// the right order is not a guard.
func TestRequestTimeoutGuardsNonPositiveItself(t *testing.T) {
	for _, d := range []time.Duration{0, -time.Second} {
		timeout := d
		cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &timeout}}
		if got := RequestTimeout(cfg); got != 0 {
			t.Errorf("RequestTimeout(%v) = %v, want 0: a non-positive bound must not become a deadline", d, got)
		}
	}
	// The positive case still comes through, so the guard is not simply off.
	positive := time.Second
	cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &positive}}
	if got := RequestTimeout(cfg); got != positive {
		t.Errorf("RequestTimeout(%v) = %v, want it unchanged", positive, got)
	}
}

// rejectUnhandledFields stands in for genai growing a field: Later and Last
// play fields added after the handled list was written, which must be refused
// by name, first in declaration order, while zero and unexported ones pass.
func TestRejectUnhandledFieldsRefusesFieldsAddedLater(t *testing.T) {
	type knobs struct {
		Handled  *int
		Later    *bool
		Last     []int
		internal int
	}
	handled := []string{"Handled"}
	tests := []struct {
		name string
		in   *knobs
		want string // the field the error names, or "" for nil
	}{
		{name: "nil", in: nil},
		{name: "nothing set", in: &knobs{}},
		{name: "handled only", in: &knobs{Handled: new(int)}},
		{name: "unexported", in: &knobs{internal: 1}},
		{name: "added later", in: &knobs{Later: new(bool)}, want: "Knobs.Later"},
		{name: "first of two", in: &knobs{Later: new(bool), Last: []int{1}}, want: "Knobs.Later"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := rejectUnhandledFields("Knobs.", tc.in, handled)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("rejectUnhandledFields() error = %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, ErrUnsupportedConfigField) {
				t.Fatalf("rejectUnhandledFields() error = %v, want %v", err, ErrUnsupportedConfigField)
			}
			if want := ErrUnsupportedConfigField.Error() + ": " + tc.want; err.Error() != want {
				t.Errorf("rejectUnhandledFields() error = %q, want %q", err, want)
			}
		})
	}

	// s is an any, so an untyped nil and a struct passed by value must not panic.
	if err := rejectUnhandledFields("Knobs.", nil, handled); err != nil {
		t.Errorf("rejectUnhandledFields(nil) error = %v, want nil", err)
	}
	if err := rejectUnhandledFields("Knobs.", knobs{Later: new(bool)}, handled); !errors.Is(err, ErrUnsupportedConfigField) {
		t.Errorf("rejectUnhandledFields(struct value) error = %v, want %v", err, ErrUnsupportedConfigField)
	}
}

// The ThinkingConfig walk has nothing to refuse until genai grows that type, so
// narrowing the handled list stands in for the field it will one day add. It
// swaps a package variable, so it must not run in parallel with other tests.
func TestRejectUnhandledConfigFieldsWalksThinkingConfig(t *testing.T) {
	saved := HandledThinkingConfigFields
	t.Cleanup(func() { HandledThinkingConfigFields = saved })
	HandledThinkingConfigFields = []string{"IncludeThoughts", "ThinkingBudget"}

	cfg := &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow}}
	err := RejectUnhandledConfigFields(cfg, HandledConfigFields)
	if want := ErrUnsupportedConfigField.Error() + ": ThinkingConfig.ThinkingLevel"; err == nil || err.Error() != want {
		t.Errorf("RejectUnhandledConfigFields() error = %v, want %q", err, want)
	}

	// A top-level field is reported first, matching genai's declaration order.
	cfg.CachedContent = "cached"
	err = RejectUnhandledConfigFields(cfg, HandledConfigFields)
	if want := ErrUnsupportedConfigField.Error() + ": CachedContent"; err == nil || err.Error() != want {
		t.Errorf("RejectUnhandledConfigFields() error = %v, want %q", err, want)
	}
}
