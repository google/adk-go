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

package models

import (
	"encoding/json"
	"testing"

	"google.golang.org/adk/v2/session"
)

// The REST form of a transfer reason is adk-python's: actions.transferReason,
// left out when unset.
func TestFromSessionEvent_TransferReason(t *testing.T) {
	tests := []struct {
		name       string
		actions    session.EventActions
		wantReason any // nil: no transferReason key
	}{
		{
			name:       "reason set",
			actions:    session.EventActions{TransferToAgent: "b", TransferReason: "r"},
			wantReason: "r",
		},
		{
			name:       "reason unset",
			actions:    session.EventActions{TransferToAgent: "b"},
			wantReason: nil,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(FromSessionEvent(session.Event{Actions: tc.actions}))
			if err != nil {
				t.Fatalf("json.Marshal: %v", err)
			}
			var wire struct {
				Actions map[string]any `json:"actions"`
			}
			if err := json.Unmarshal(data, &wire); err != nil {
				t.Fatalf("json.Unmarshal: %v", err)
			}
			if got := wire.Actions["transferToAgent"]; got != "b" {
				t.Errorf("actions.transferToAgent = %v, want \"b\"", got)
			}
			if got := wire.Actions["transferReason"]; got != tc.wantReason {
				t.Errorf("actions.transferReason = %v in %s, want %v", got, data, tc.wantReason)
			}
		})
	}
}

func TestToSessionEvent_TransferReason(t *testing.T) {
	var body Event
	if err := json.Unmarshal([]byte(`{"timestamp":1767225600,"actions":{"transferToAgent":"b","transferReason":"r"}}`), &body); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	got, err := ToSessionEvent(body)
	if err != nil {
		t.Fatalf("ToSessionEvent() failed: %v", err)
	}
	if got.Actions.TransferToAgent != "b" || got.Actions.TransferReason != "r" {
		t.Errorf("ToSessionEvent() (TransferToAgent, TransferReason) = (%q, %q), want (\"b\", \"r\")", got.Actions.TransferToAgent, got.Actions.TransferReason)
	}
}
