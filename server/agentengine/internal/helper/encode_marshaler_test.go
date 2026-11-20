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
		t.Fatalf("json.Unmarshal(%s) failed: %v", rw.Body.String(), err)
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
		t.Errorf("output mismatch (-want +got):\n%s\nbody: %s", diff, rw.Body.String())
	}
	actions, ok := got["actions"].(map[string]any)
	if !ok {
		t.Fatalf("actions is not an object: %s", rw.Body.String())
	}
	// EventActions defines MarshalJSON but has exported fields, so it is still
	// walked and keeps snake_case keys.
	if diff := cmp.Diff(map[string]any{"last_order": id.String()}, actions["state_delta"]); diff != "" {
		t.Errorf("actions.state_delta mismatch (-want +got):\n%s\nbody: %s", diff, rw.Body.String())
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
