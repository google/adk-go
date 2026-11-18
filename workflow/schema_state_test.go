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

package workflow_test

import (
	"errors"
	"fmt"
	"maps"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// ss3Pipeline is a state schema titled PipelineSchema with counter (integer),
// name (string), optional_field (string or null) and tags (array of string).
func ss3Pipeline(t *testing.T) *jsonschema.Resolved {
	t.Helper()
	return ss3Resolve(t, &jsonschema.Schema{
		Type:  "object",
		Title: "PipelineSchema",
		Properties: map[string]*jsonschema.Schema{
			"counter":        {Type: "integer"},
			"name":           {Type: "string"},
			"optional_field": {Types: []string{"string", "null"}},
			"tags":           {Type: "array", Items: &jsonschema.Schema{Type: "string"}},
		},
	})
}

// ss3Constrained is a state schema whose properties carry bounds: counter
// (integer, 1 to 10), name (string, at least 3 characters) and pos (integer,
// above 0).
func ss3Constrained(t *testing.T) *jsonschema.Resolved {
	t.Helper()
	return ss3Resolve(t, &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"counter": {Type: "integer", Minimum: new(1.0), Maximum: new(10.0)},
			"name":    {Type: "string", MinLength: new(3)},
			"pos":     {Type: "integer", ExclusiveMinimum: new(0.0)},
		},
	})
}

func ss3Resolve(t *testing.T, s *jsonschema.Schema) *jsonschema.Resolved {
	t.Helper()
	r, err := s.Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	return r
}

// ss3New returns NewSchemaState(cfg) and fails the test on an error.
func ss3New(t *testing.T, cfg workflow.SchemaStateConfig) *workflow.SchemaState {
	t.Helper()
	s, err := workflow.NewSchemaState(cfg)
	if err != nil {
		t.Fatalf("NewSchemaState() error = %v", err)
	}
	return s
}

// ss3WantContents checks that All and Delta of s yield all and delta.
func ss3WantContents(t *testing.T, s *workflow.SchemaState, all, delta map[string]any) {
	t.Helper()
	if diff := cmp.Diff(all, maps.Collect(s.All())); diff != "" {
		t.Errorf("All() mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(delta, s.Delta()); diff != "" {
		t.Errorf("Delta() mismatch (-want +got):\n%s", diff)
	}
}

// ss3WantGet checks that Get(key) returns want and a nil error.
func ss3WantGet(t *testing.T, s *workflow.SchemaState, key string, want any) {
	t.Helper()
	got, err := s.Get(key)
	if err != nil {
		t.Fatalf("Get(%q) error = %v, want nil", key, err)
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Get(%q) mismatch (-want +got):\n%s", key, diff)
	}
}

// ss3WantRefusal checks that err is a *StateSchemaError with reason and key.
func ss3WantRefusal(t *testing.T, err error, reason workflow.StateSchemaReason, key string) {
	t.Helper()
	se, ok := errors.AsType[*workflow.StateSchemaError](err)
	if !ok {
		t.Fatalf("error = %v, want a *workflow.StateSchemaError", err)
	}
	if se.Reason != reason || se.Key != key {
		t.Fatalf("error = %+v, want Reason %v and Key %q", se, reason, key)
	}
}

func TestSchemaState_Set_ValidWriteIsReadBack(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{Schema: ss3Pipeline(t)})

	if err := s.Set("counter", float64(5)); err != nil {
		t.Fatalf("Set(counter, 5) error = %v, want nil", err)
	}

	ss3WantGet(t, s, "counter", float64(5))
	ss3WantContents(t, s, map[string]any{"counter": float64(5)}, map[string]any{"counter": float64(5)})
}

// The value is stored as given, not in the JSON form it was validated in.
func TestSchemaState_Set_StoresValueAsGiven(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{Schema: ss3Pipeline(t)})

	if err := s.Set("tags", []string{"a"}); err != nil {
		t.Fatalf("Set(tags) error = %v, want nil", err)
	}

	ss3WantGet(t, s, "tags", []string{"a"})
}

func TestSchemaState_Set_RefusalLeavesStateUnchanged(t *testing.T) {
	tests := []struct {
		name   string
		key    string
		value  any
		reason workflow.StateSchemaReason
	}{
		{name: "undeclared key", key: "unknown_key", value: "val", reason: workflow.StateKeyUndeclared},
		{name: "string for integer", key: "counter", value: "not_an_int", reason: workflow.StateValueInvalid},
		{name: "fraction for integer", key: "counter", value: float64(5.5), reason: workflow.StateValueInvalid},
		{name: "nil for string", key: "name", value: nil, reason: workflow.StateValueInvalid},
		{name: "value with no JSON form", key: "name", value: make(chan int), reason: workflow.StateValueInvalid},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := ss3New(t, workflow.SchemaStateConfig{Schema: ss3Pipeline(t), Value: map[string]any{"counter": float64(1)}})

			err := s.Set(tc.key, tc.value)

			ss3WantRefusal(t, err, tc.reason, tc.key)
			ss3WantContents(t, s, map[string]any{"counter": float64(1)}, map[string]any{})
		})
	}
}

func TestSchemaState_NilSchema_WritesAreUnchecked(t *testing.T) {
	t.Run("Set", func(t *testing.T) {
		s := ss3New(t, workflow.SchemaStateConfig{})

		if err := s.Set("zzz", 1); err != nil {
			t.Fatalf("Set(zzz, 1) error = %v, want nil", err)
		}

		ss3WantGet(t, s, "zzz", 1)
	})
	t.Run("Update", func(t *testing.T) {
		s := ss3New(t, workflow.SchemaStateConfig{})

		if err := s.Update(map[string]any{"x": "y", "zzz2": 2}); err != nil {
			t.Fatalf("Update() error = %v, want nil", err)
		}

		if diff := cmp.Diff(map[string]any{"x": "y", "zzz2": 2}, maps.Collect(s.All())); diff != "" {
			t.Errorf("All() mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("SetDefault", func(t *testing.T) {
		s := ss3New(t, workflow.SchemaStateConfig{})

		got, err := s.SetDefault("w", "d")

		if got != "d" || err != nil {
			t.Errorf("SetDefault(w, d) = %v, %v, want d, nil", got, err)
		}
	})
}

func TestSchemaState_ZeroValueIsUsable(t *testing.T) {
	var s workflow.SchemaState

	if err := s.Set("k", 1); err != nil {
		t.Fatalf("Set(k, 1) error = %v, want nil", err)
	}

	ss3WantGet(t, &s, "k", 1)
	if diff := cmp.Diff(map[string]any{"k": 1}, s.Delta()); diff != "" {
		t.Errorf("Delta() mismatch (-want +got):\n%s", diff)
	}
	if got, err := s.SetDefault("j", 2); got != 2 || err != nil {
		t.Errorf("SetDefault(j, 2) = %v, %v, want 2, nil", got, err)
	}
}

func TestNewSchemaState_CopiesSeeds(t *testing.T) {
	cfg := workflow.SchemaStateConfig{Value: map[string]any{"a": 1}, Delta: map[string]any{"b": 2}}
	s := ss3New(t, cfg)

	cfg.Value["a"] = 99
	delete(cfg.Delta, "b")
	if diff := cmp.Diff(map[string]any{"a": 1, "b": 2}, maps.Collect(s.All())); diff != "" {
		t.Errorf("All() after the caller changed its maps mismatch (-want +got):\n%s", diff)
	}

	if err := s.Set("c", 3); err != nil {
		t.Fatalf("Set(c, 3) error = %v, want nil", err)
	}
	if _, ok := cfg.Value["c"]; ok {
		t.Errorf("cfg.Value = %v after Set(c, 3), want no key c", cfg.Value)
	}
	if _, ok := cfg.Delta["c"]; ok {
		t.Errorf("cfg.Delta = %v after Set(c, 3), want no key c", cfg.Delta)
	}
}

func TestNewSchemaState_SeedsAreNotValidated(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{
		Schema: ss3Pipeline(t),
		Value:  map[string]any{"counter": "not_an_int", "unknown": 1},
	})

	ss3WantGet(t, s, "counter", "not_an_int")
	ss3WantGet(t, s, "unknown", 1)
}

func TestNewSchemaState_UncheckableSchemaFails(t *testing.T) {
	s, err := workflow.NewSchemaState(workflow.SchemaStateConfig{Schema: ss2UnusableSchema(t)})

	if !errors.Is(err, workflow.ErrStateSchemaUnusable) {
		t.Errorf("NewSchemaState() error = %v, want ErrStateSchemaUnusable", err)
	}
	if s != nil {
		t.Errorf("NewSchemaState() = %v, want nil", s)
	}
}

// A zero *jsonschema.Resolved declares no key: every key without a colon is
// refused, and a key with a colon is not checked.
func TestNewSchemaState_ZeroSchemaDeclaresNoKey(t *testing.T) {
	s, err := workflow.NewSchemaState(workflow.SchemaStateConfig{Schema: &jsonschema.Resolved{}})
	if err != nil || s == nil {
		t.Fatalf("NewSchemaState(zero schema) = %v, %v, want a state and nil", s, err)
	}

	err = s.Set("foo", 1)
	ss3WantRefusal(t, err, workflow.StateKeyUndeclared, "foo")
	if se, _ := errors.AsType[*workflow.StateSchemaError](err); len(se.Declared) != 0 {
		t.Errorf("Declared = %v, want empty", se.Declared)
	}
	if err := s.Set("app:x", 1); err != nil {
		t.Errorf("Set(app:x, 1) error = %v, want nil", err)
	}
}

// Reads see the pending delta over the committed value.
func TestSchemaState_Reads_DeltaOverValue(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{
		Value: map[string]any{"a": 1, "b": 2},
		Delta: map[string]any{"b": 20, "c": 3},
	})

	ss3WantGet(t, s, "a", 1)
	ss3WantGet(t, s, "b", 20)
	ss3WantGet(t, s, "c", 3)
	got, err := s.Get("d")
	if !errors.Is(err, session.ErrStateKeyNotExist) || got != nil {
		t.Errorf("Get(d) = %v, %v, want nil, ErrStateKeyNotExist", got, err)
	}
	ss3WantContents(t, s, map[string]any{"a": 1, "b": 20, "c": 3}, map[string]any{"b": 20, "c": 3})
}

func TestSchemaState_Get_StoredNilIsPresent(t *testing.T) {
	t.Run("seeded", func(t *testing.T) {
		s := ss3New(t, workflow.SchemaStateConfig{Value: map[string]any{"optional_field": nil}})

		ss3WantGet(t, s, "optional_field", nil)
	})
	t.Run("written", func(t *testing.T) {
		s := ss3New(t, workflow.SchemaStateConfig{Schema: ss3Pipeline(t)})
		if err := s.Set("optional_field", nil); err != nil {
			t.Fatalf("Set(optional_field, nil) error = %v, want nil", err)
		}

		ss3WantGet(t, s, "optional_field", nil)
	})
	t.Run("updated", func(t *testing.T) {
		s := ss3New(t, workflow.SchemaStateConfig{Schema: ss3Pipeline(t)})
		if err := s.Update(map[string]any{"optional_field": nil}); err != nil {
			t.Fatalf("Update(optional_field: nil) error = %v, want nil", err)
		}

		ss3WantGet(t, s, "optional_field", nil)
		ss3WantContents(t, s, map[string]any{"optional_field": nil}, map[string]any{"optional_field": nil})
	})
}

func TestSchemaState_Delta_ReturnsACopyTheCallerOwns(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{Schema: ss3Pipeline(t)})
	if err := s.Set("counter", float64(1)); err != nil {
		t.Fatalf("Set(counter, 1) error = %v, want nil", err)
	}

	d := s.Delta()
	d["x"] = 1
	delete(d, "counter")

	if diff := cmp.Diff(map[string]any{"counter": float64(1)}, s.Delta()); diff != "" {
		t.Errorf("Delta() after the caller changed a copy mismatch (-want +got):\n%s", diff)
	}
	empty := ss3New(t, workflow.SchemaStateConfig{}).Delta()
	empty["x"] = 1 // panics if Delta returned a nil map
}

func TestSchemaState_All_SnapshotTakenAtCall(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{})
	if err := s.Set("a", 1); err != nil {
		t.Fatalf("Set(a, 1) error = %v, want nil", err)
	}

	it := s.All()
	if err := s.Set("late", 1); err != nil {
		t.Fatalf("Set(late, 1) error = %v, want nil", err)
	}

	if diff := cmp.Diff(map[string]any{"a": 1}, maps.Collect(it)); diff != "" {
		t.Errorf("ranging All() taken before Set(late) mismatch (-want +got):\n%s", diff)
	}
}

// Writes from inside a range over All do not wait for the range to end.
func TestSchemaState_All_NoLockHeldWhileRanging(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{Value: map[string]any{"a": 1}})
	done := make(chan struct{})

	go func() {
		defer close(done)
		for range s.All() {
			_ = s.Set("x", 1)
			_ = s.Update(map[string]any{"y": 2})
			_, _ = s.SetDefault("z", 3)
		}
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("writes inside a range over All() did not return within 5s")
	}
	ss3WantGet(t, s, "x", 1)
}

func TestSchemaState_ConcurrentWritesAreAllStored(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{})
	stop := make(chan struct{})
	var readers sync.WaitGroup
	for range 2 {
		readers.Go(func() {
			for {
				select {
				case <-stop:
					return
				default:
				}
				_, _ = s.Get("k0")
				for range s.All() {
				}
				_ = s.Delta()
			}
		})
	}

	var writers sync.WaitGroup
	want := map[string]any{}
	for i := range 32 {
		key := fmt.Sprintf("k%d", i)
		want[key] = i
		writers.Go(func() {
			if err := s.Set(key, i); err != nil {
				t.Errorf("Set(%s) error = %v", key, err)
			}
		})
	}
	writers.Wait()
	close(stop)
	readers.Wait()

	if diff := cmp.Diff(want, maps.Collect(s.All())); diff != "" {
		t.Errorf("All() after 32 concurrent writes mismatch (-want +got):\n%s", diff)
	}
}

func TestSchemaState_Update_WritesAllEntries(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{Schema: ss3Pipeline(t)})
	entries := map[string]any{"counter": float64(1), "name": "abc", "app:x": true}

	if err := s.Update(entries); err != nil {
		t.Fatalf("Update(%v) error = %v, want nil", entries, err)
	}
	ss3WantContents(t, s,
		map[string]any{"counter": float64(1), "name": "abc", "app:x": true},
		map[string]any{"counter": float64(1), "name": "abc", "app:x": true})

	if err := s.Update(map[string]any{"counter": float64(2)}); err != nil {
		t.Fatalf("Update(counter: 2) error = %v, want nil", err)
	}
	ss3WantGet(t, s, "counter", float64(2))
	ss3WantGet(t, s, "name", "abc")
}

// A refused update writes none of its entries, including the valid entry and
// the colon key that sort before the refused one.
func TestSchemaState_Update_RefusedWritesNothing(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{Schema: ss3Pipeline(t), Value: map[string]any{"counter": float64(1)}})

	err := s.Update(map[string]any{"app:flag": true, "counter": float64(2), "unknown": "x"})

	ss3WantRefusal(t, err, workflow.StateKeyUndeclared, "unknown")
	ss3WantContents(t, s, map[string]any{"counter": float64(1)}, map[string]any{})
}

// Of several refused entries, the first in sorted key order is reported. Map
// order is random, so the update is repeated on fresh states.
func TestSchemaState_Update_ReportsFirstFailureInSortedOrder(t *testing.T) {
	schema := ss3Pipeline(t)
	for range 20 {
		s := ss3New(t, workflow.SchemaStateConfig{Schema: schema})

		err := s.Update(map[string]any{"unknown": "x", "counter": "bad"})

		ss3WantRefusal(t, err, workflow.StateValueInvalid, "counter")
	}
}

func TestSchemaState_Update_EmptyIsNoOp(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{Delta: map[string]any{"a": 1}})

	if err := s.Update(nil); err != nil {
		t.Errorf("Update(nil) error = %v, want nil", err)
	}
	if err := s.Update(map[string]any{}); err != nil {
		t.Errorf("Update(empty) error = %v, want nil", err)
	}

	ss3WantContents(t, s, map[string]any{"a": 1}, map[string]any{"a": 1})
}

// A reader never sees one entry of an update without the others.
func TestSchemaState_Update_ConcurrentReadersSeeAllOrNothing(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{})
	stop := make(chan struct{})
	var readers sync.WaitGroup
	for range 4 {
		readers.Go(func() {
			for {
				select {
				case <-stop:
					return
				default:
				}
				if m := maps.Collect(s.All()); m["a"] != m["b"] {
					t.Errorf("All() = %v, want a and b equal", m)
					return
				}
			}
		})
	}

	for i := 1; i <= 3000; i++ {
		if err := s.Update(map[string]any{"a": i, "b": i}); err != nil {
			t.Fatalf("Update(%d) error = %v", i, err)
		}
	}
	close(stop)
	readers.Wait()

	if diff := cmp.Diff(map[string]any{"a": 3000, "b": 3000}, maps.Collect(s.All())); diff != "" {
		t.Errorf("All() after the last update mismatch (-want +got):\n%s", diff)
	}
}

func TestSchemaState_SetDefault_WritesMissingKey(t *testing.T) {
	s := ss3New(t, workflow.SchemaStateConfig{Schema: ss3Pipeline(t)})

	got, err := s.SetDefault("name", "default_name")

	if got != "default_name" || err != nil {
		t.Errorf("SetDefault(name, default_name) = %v, %v, want default_name, nil", got, err)
	}
	ss3WantGet(t, s, "name", "default_name")
	if diff := cmp.Diff(map[string]any{"name": "default_name"}, s.Delta()); diff != "" {
		t.Errorf("Delta() mismatch (-want +got):\n%s", diff)
	}
}

// When the key is present, its value is returned and the default is neither
// checked nor stored. The pending delta is read before the committed value.
func TestSchemaState_SetDefault_ExistingKeyWins(t *testing.T) {
	tests := []struct {
		name      string
		value     map[string]any
		delta     map[string]any
		set       any // written with Set before SetDefault when non-nil
		want      any
		wantDelta map[string]any
	}{
		{
			name:      "in the value",
			value:     map[string]any{"counter": float64(10)},
			want:      float64(10),
			wantDelta: map[string]any{},
		},
		{
			name:      "in the delta",
			delta:     map[string]any{"counter": float64(10)},
			want:      float64(10),
			wantDelta: map[string]any{"counter": float64(10)},
		},
		{
			name:      "stored nil",
			value:     map[string]any{"counter": nil},
			want:      nil,
			wantDelta: map[string]any{},
		},
		{
			name:      "delta over value",
			value:     map[string]any{"counter": float64(1)},
			delta:     map[string]any{"counter": float64(10)},
			want:      float64(10),
			wantDelta: map[string]any{"counter": float64(10)},
		},
		{
			name:      "written over value",
			value:     map[string]any{"counter": float64(1)},
			set:       float64(10),
			want:      float64(10),
			wantDelta: map[string]any{"counter": float64(10)},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := ss3New(t, workflow.SchemaStateConfig{Schema: ss3Pipeline(t), Value: tc.value, Delta: tc.delta})
			if tc.set != nil {
				if err := s.Set("counter", tc.set); err != nil {
					t.Fatalf("Set(counter) error = %v, want nil", err)
				}
			}

			got, err := s.SetDefault("counter", "bad")
			if err != nil {
				t.Fatalf("SetDefault(counter, bad) error = %v, want nil", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("SetDefault(counter, bad) mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantDelta, s.Delta()); diff != "" {
				t.Errorf("Delta() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSchemaState_SetDefault_RefusedDefaultWritesNothing(t *testing.T) {
	tests := []struct {
		name   string
		key    string
		def    any
		reason workflow.StateSchemaReason
	}{
		{name: "undeclared key", key: "unknown_key", def: "val", reason: workflow.StateKeyUndeclared},
		{name: "nil for string", key: "name", def: nil, reason: workflow.StateValueInvalid},
		{name: "string for integer", key: "counter", def: "bad", reason: workflow.StateValueInvalid},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := ss3New(t, workflow.SchemaStateConfig{Schema: ss3Pipeline(t)})

			got, err := s.SetDefault(tc.key, tc.def)

			ss3WantRefusal(t, err, tc.reason, tc.key)
			if got != nil {
				t.Errorf("SetDefault(%q) = %v, want nil", tc.key, got)
			}
			ss3WantContents(t, s, map[string]any{}, map[string]any{})
		})
	}
}

// Field constraints refuse a value on each of the three write paths.
func TestSchemaState_ConstraintsRefuseOnEveryPath(t *testing.T) {
	cases := []struct {
		name  string
		key   string
		value any
	}{
		{name: "below minimum", key: "counter", value: float64(0)},
		{name: "above maximum", key: "counter", value: float64(11)},
		{name: "below minLength", key: "name", value: "ab"},
		{name: "on exclusiveMinimum", key: "pos", value: float64(0)},
	}
	paths := []struct {
		name  string
		write func(s *workflow.SchemaState, key string, value any) error
	}{
		{name: "Set", write: func(s *workflow.SchemaState, key string, value any) error { return s.Set(key, value) }},
		{name: "Update", write: func(s *workflow.SchemaState, key string, value any) error {
			return s.Update(map[string]any{key: value})
		}},
		{name: "SetDefault", write: func(s *workflow.SchemaState, key string, value any) error {
			_, err := s.SetDefault(key, value)
			return err
		}},
	}
	schema := ss3Constrained(t)
	for _, p := range paths {
		for _, tc := range cases {
			t.Run(p.name+"/"+tc.name, func(t *testing.T) {
				s := ss3New(t, workflow.SchemaStateConfig{Schema: schema})

				err := p.write(s, tc.key, tc.value)

				ss3WantRefusal(t, err, workflow.StateValueInvalid, tc.key)
				ss3WantContents(t, s, map[string]any{}, map[string]any{})
			})
		}
	}
}

// Of concurrent SetDefault calls for one absent key, one stores its default
// and every call returns that default.
func TestSchemaState_SetDefault_ConcurrentStoresOneDefault(t *testing.T) {
	for range 200 {
		s := ss3New(t, workflow.SchemaStateConfig{})
		start := make(chan struct{})
		results := make([]any, 32)
		var wg sync.WaitGroup
		for i := range results {
			wg.Go(func() {
				<-start
				v, err := s.SetDefault("name", fmt.Sprintf("d%d", i))
				if err != nil {
					t.Errorf("SetDefault(name) error = %v", err)
				}
				results[i] = v
			})
		}
		close(start)
		wg.Wait()

		stored, err := s.Get("name")
		if err != nil {
			t.Fatalf("Get(name) error = %v", err)
		}
		for i, v := range results {
			if v != stored {
				t.Fatalf("SetDefault call %d returned %v, want the stored %v", i, v, stored)
			}
		}
	}
}
