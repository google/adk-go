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
	"reflect"
	"slices"
	"strings"
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

// rejectUnsupportedFields stands in for genai growing a field: Later and Last
// play fields added after the supported list was written, which must be refused
// by name, first in declaration order, while zero and unexported ones pass.
func TestRejectUnsupportedFieldsRefusesFieldsAddedLater(t *testing.T) {
	type knobs struct {
		Supported *int
		Later     *bool
		Last      []int
		internal  int
	}
	tests := []struct {
		name string
		in   any
		want string // the field the error names, or "" for no error
	}{
		{name: "untyped nil", in: nil},
		{name: "nil pointer", in: (*knobs)(nil)},
		{name: "nothing set", in: &knobs{}},
		{name: "supported only", in: &knobs{Supported: new(int)}},
		{name: "unexported", in: &knobs{internal: 1}},
		{name: "added later", in: &knobs{Later: new(bool)}, want: "Knobs.Later"},
		{name: "first of two", in: &knobs{Later: new(bool), Last: []int{1}}, want: "Knobs.Later"},
		{name: "struct value", in: knobs{Last: []int{1}}, want: "Knobs.Last"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ""
			if err := rejectUnsupportedFields("Knobs.", tc.in, []string{"Supported"}); err != nil {
				if !errors.Is(err, ErrUnsupportedConfigField) {
					t.Fatalf("rejectUnsupportedFields() error = %v, want %v", err, ErrUnsupportedConfigField)
				}
				got = strings.TrimPrefix(err.Error(), ErrUnsupportedConfigField.Error()+": ")
			}
			if got != tc.want {
				t.Errorf("rejectUnsupportedFields() named %q, want %q", got, tc.want)
			}
		})
	}
}

// The ThinkingConfig walk has nothing to refuse until genai grows that type, so
// narrowing the supported list stands in for the field it will one day add. It
// swaps a package variable, so it must not run in parallel with other tests.
func TestRejectUnsupportedConfigFieldsWalksThinkingConfig(t *testing.T) {
	saved := SupportedThinkingConfigFields
	t.Cleanup(func() { SupportedThinkingConfigFields = saved })
	SupportedThinkingConfigFields = []string{"IncludeThoughts", "ThinkingBudget"}

	cfg := &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow}}
	err := RejectUnsupportedConfigFields(cfg, SupportedConfigFields)
	if want := ErrUnsupportedConfigField.Error() + ": ThinkingConfig.ThinkingLevel"; err == nil || err.Error() != want {
		t.Errorf("RejectUnsupportedConfigFields() error = %v, want %q", err, want)
	}

	// A top-level field is reported first, matching genai's declaration order.
	cfg.CachedContent = "cached"
	err = RejectUnsupportedConfigFields(cfg, SupportedConfigFields)
	if want := ErrUnsupportedConfigField.Error() + ": CachedContent"; err == nil || err.Error() != want {
		t.Errorf("RejectUnsupportedConfigFields() error = %v, want %q", err, want)
	}
}

// A supported name genai no longer has guards nothing: the field it was renamed to
// is refused until the list follows.
func TestSupportedFieldsExistInGenai(t *testing.T) {
	for typ, names := range map[reflect.Type][]string{
		reflect.TypeFor[genai.GenerateContentConfig](): SupportedConfigFields,
		reflect.TypeFor[genai.HTTPOptions]():           slices.Concat(SupportedHTTPOptionFields, IgnoredHTTPOptionFields),
		reflect.TypeFor[genai.ThinkingConfig]():        SupportedThinkingConfigFields,
	} {
		for _, name := range names {
			if _, ok := typ.FieldByName(name); !ok {
				t.Errorf("%s has no field %s", typ, name)
			}
		}
	}
}
