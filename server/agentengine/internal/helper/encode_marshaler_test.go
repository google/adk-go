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

package helper

import (
	"encoding/json"
	"errors"
	"math/big"
	"net/http/httptest"
	"net/netip"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"

	"google.golang.org/adk/v2/session"
)

// TestStreamedOutputUsesMarshalers checks that values the field walk cannot
// represent are written the way encoding/json writes them, while the event
// and output keys stay snake_case.
func TestStreamedOutputUsesMarshalers(t *testing.T) {
	type order struct {
		OrderID uuid.UUID   `json:"orderID"`
		Ref     *uuid.UUID  `json:"ref"`
		Addr    netip.Addr  `json:"addr"`
		Amount  *big.Int    `json:"amount"`
		Counts  [3]int      `json:"counts"`
		IDs     []uuid.UUID `json:"ids"`
	}
	id := uuid.MustParse("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
	ev := session.Event{
		InvocationID: "inv",
		Output: order{
			OrderID: id,
			Ref:     &id,
			Addr:    netip.MustParseAddr("10.0.0.1"),
			Amount:  big.NewInt(12345),
			Counts:  [3]int{1, 2, 3},
			IDs:     []uuid.UUID{id},
		},
		Actions: session.EventActions{StateDelta: map[string]any{"last_order": id}},
	}
	rw := httptest.NewRecorder()
	if err := EmitJSON(rw, *EventForStream(&ev)); err != nil {
		t.Fatalf("EmitJSON() failed: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(rw.Body.Bytes(), &got); err != nil {
		t.Fatalf("json.Unmarshal(%d-byte body) failed: %v", rw.Body.Len(), err)
	}

	wantOutput := map[string]any{
		"order_id": id.String(),
		"ref":      id.String(),
		"addr":     "10.0.0.1",
		"amount":   float64(12345),
		"counts":   []any{float64(1), float64(2), float64(3)},
		"ids":      []any{id.String()},
	}
	if diff := cmp.Diff(wantOutput, got["output"]); diff != "" {
		t.Errorf("output mismatch (-want +got):\n%s", diff)
	}
	actions, ok := got["actions"].(map[string]any)
	if !ok {
		t.Fatalf("actions is %T, want map[string]any", got["actions"])
	}
	// EventActions defines MarshalJSON but has exported fields, so it is still
	// walked and keeps snake_case keys.
	if diff := cmp.Diff(map[string]any{"last_order": id.String()}, actions["state_delta"]); diff != "" {
		t.Errorf("actions.state_delta mismatch (-want +got):\n%s", diff)
	}
}

type lockedCount struct {
	sync.Mutex
	n int
}

func (c *lockedCount) MarshalJSON() ([]byte, error) { return json.Marshal(c.n) }

type hiddenText struct {
	Secret string `json:"-"`
	v      string
}

func (h hiddenText) MarshalText() ([]byte, error) { return []byte(h.v), nil }

type failingJSON struct{ _ int }

func (failingJSON) MarshalJSON() ([]byte, error) { return nil, errors.New("json boom") }

type failingText struct{ _ int }

func (failingText) MarshalText() ([]byte, error) { return nil, errors.New("text boom") }

// TestConvertSnakeOpaqueValues checks the rules for which values are handed
// to their own marshaler, and that the result matches encoding/json.
func TestConvertSnakeOpaqueValues(t *testing.T) {
	for _, tc := range []struct {
		name    string
		value   any
		want    map[string]any
		wantErr string
	}{
		{
			name:  "embedded_mutex_with_unexported_data",
			value: struct{ L *lockedCount }{&lockedCount{n: 7}},
			want:  map[string]any{"l": json.RawMessage("7")},
		},
		{
			name:  "fields_tagged_dash_are_not_walked",
			value: struct{ H hiddenText }{hiddenText{Secret: "s", v: "shown"}},
			want:  map[string]any{"h": "shown"},
		},
		{
			name:  "pointer_receiver_on_addressable_field",
			value: &struct{ Amount big.Int }{Amount: *big.NewInt(5)},
			want:  map[string]any{"amount": json.RawMessage("5")},
		},
		{
			name: "pointer_receiver_on_nested_addressable_field",
			value: &struct {
				Account struct {
					Amount big.Int `json:"amount"`
				} `json:"account"`
			}{Account: struct {
				Amount big.Int `json:"amount"`
			}{Amount: *big.NewInt(7)}},
			want: map[string]any{"account": map[string]any{"amount": json.RawMessage("7")}},
		},
		{
			name: "omitempty_keeps_non_nil_pointer_to_zero_value",
			value: struct {
				A *netip.Addr `json:"a,omitempty"`
				S *string     `json:"s,omitempty"`
				N *netip.Addr `json:"n,omitempty"`
			}{A: &netip.Addr{}, S: new(string)},
			want: map[string]any{"a": "", "s": ""},
		},
		{
			name: "omitempty_keeps_non_nil_interface_to_zero_value",
			value: struct {
				A any `json:"a,omitempty"`
				B any `json:"b,omitempty"`
			}{A: ""},
			want: map[string]any{"a": ""},
		},
		{
			name: "omitempty_keeps_marshaled_struct",
			value: struct {
				A netip.Addr `json:"a,omitempty"` //nolint:modernize // checks that omitempty has no effect, as in encoding/json
			}{},
			want: map[string]any{"a": ""},
		},
		{
			name:    "marshal_json_error",
			value:   struct{ F failingJSON }{},
			wantErr: "json boom",
		},
		{
			name:    "marshal_text_error",
			value:   struct{ F failingText }{},
			wantErr: "text boom",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := convertSnake("", tc.value)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("convertSnake() error = %v, want it to contain %q", err, tc.wantErr)
				}
				// encoding/json fails on the same value too.
				if _, err := json.Marshal(tc.value); err == nil {
					t.Errorf("json.Marshal() succeeded, want an error like convertSnake")
				}
				return
			}
			if err != nil {
				t.Fatalf("convertSnake() failed: %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("convertSnake() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

type textInner struct{ v string }

func (t textInner) MarshalText() ([]byte, error) { return []byte(t.v), nil }

type embedsTextInner struct{ textInner }

type textInnerWithField struct {
	textInner
	X int `json:"x"`
}

type selfEmbed struct{ *selfEmbed }

type anyChain struct {
	Next any `json:"next"`
	V    int `json:"v"`
}

// TestConvertSnakeGuards checks values that would crash the walk without
// its guards.
func TestConvertSnakeGuards(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value any
		want  map[string]any
	}{
		{
			// The marshaler promoted from an unexported embedding is used, as
			// encoding/json does.
			name:  "promoted_value_marshaler_from_unexported_embedding",
			value: struct{ E embedsTextInner }{embedsTextInner{textInner{v: "x"}}},
			want:  map[string]any{"e": "x"},
		},
		{
			// The walk reaches textInner through the unexported embedding, where
			// it is not interfaceable, so its MarshalText cannot be called there.
			// The outer struct has a walkable field, so it keeps snake_case keys
			// where encoding/json would use the promoted MarshalText.
			name:  "unexported_embedded_value_marshaler",
			value: struct{ E textInnerWithField }{textInnerWithField{textInner{v: "x"}, 1}},
			want:  map[string]any{"e": map[string]any{"x": int64(1)}},
		},
		{
			name:  "self_embedding_field_type",
			value: struct{ R selfEmbed }{},
			want:  map[string]any{"r": map[string]any{}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := convertSnake("", tc.value)
			if err != nil {
				t.Fatalf("convertSnake() failed: %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("convertSnake() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestStreamedOutputKeepsDeepValues checks that values nested through
// interfaces deeper than half of maxSnakeDepth still stream their output, on
// the snake_case path and on the encoding/json fallback.
func TestStreamedOutputKeepsDeepValues(t *testing.T) {
	const depth = 200
	chain := func(leaf any, wrap func(any) any) any {
		out := leaf
		for range depth - 1 {
			out = wrap(out)
		}
		return out
	}
	for _, tc := range []struct {
		name   string
		output any
		next   func(any) (any, bool)
	}{
		{
			name:   "struct_any_fields",
			output: chain(anyChain{}, func(n any) any { return anyChain{Next: n} }),
			next: func(n any) (any, bool) {
				m, ok := n.(map[string]any)
				return m["next"], ok
			},
		},
		{
			name:   "map_values",
			output: chain(map[string]any{}, func(n any) any { return map[string]any{"next": n} }),
			next: func(n any) (any, bool) {
				m, ok := n.(map[string]any)
				return m["next"], ok
			},
		},
		{
			name:   "slice_elements",
			output: chain([]any{}, func(n any) any { return []any{n} }),
			next: func(n any) (any, bool) {
				s, ok := n.([]any)
				if !ok || len(s) == 0 {
					return nil, ok
				}
				return s[0], true
			},
		},
		{
			// A netip.Addr map key fails the snake_case conversion, so the
			// output goes through the encoding/json fallback.
			name: "fallback_after_unsupported_key",
			output: chain(anyChain{Next: map[netip.Addr]int{netip.MustParseAddr("10.0.0.1"): 1}},
				func(n any) any { return anyChain{Next: n} }),
			next: func(n any) (any, bool) {
				m, ok := n.(map[string]any)
				if !ok {
					return nil, false
				}
				if v, ok := m["next"]; ok {
					return v, true
				}
				return m["Next"], true
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ev := session.Event{InvocationID: "inv", Output: tc.output}
			rw := httptest.NewRecorder()
			if err := EmitJSON(rw, *EventForStream(&ev)); err != nil {
				t.Fatalf("EmitJSON() failed: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(rw.Body.Bytes(), &got); err != nil {
				t.Fatalf("json.Unmarshal(%d-byte body) failed: %v", rw.Body.Len(), err)
			}
			n := 0
			for node, ok := got["output"], got["output"] != nil; ok && node != nil; node, ok = tc.next(node) {
				n++
			}
			if n < depth {
				t.Errorf("streamed output has %d nested levels, want at least %d", n, depth)
			}
		})
	}
}
