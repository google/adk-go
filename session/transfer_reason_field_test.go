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

package session_test

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
)

// The JSON form is what the database and Redis session services store for an
// event's actions, and what adk-python writes with
// model_dump_json(exclude_none=True, by_alias=True): transferReason in
// camelCase, and no key when the reason is unset.
func TestEventActionsJSON_TransferReason(t *testing.T) {
	tests := []struct {
		name    string
		actions session.EventActions
		want    map[string]any
	}{
		{
			name:    "reason set",
			actions: session.EventActions{TransferToAgent: "a", TransferReason: "r"},
			want:    map[string]any{"transferToAgent": "a", "transferReason": "r"},
		},
		{
			name:    "reason unset",
			actions: session.EventActions{TransferToAgent: "a"},
			want:    map[string]any{"transferToAgent": "a"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.actions)
			if err != nil {
				t.Fatalf("json.Marshal: %v", err)
			}
			var wire map[string]any
			if err := json.Unmarshal(data, &wire); err != nil {
				t.Fatalf("json.Unmarshal to map: %v", err)
			}
			if diff := cmp.Diff(tc.want, wire); diff != "" {
				t.Errorf("JSON form mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("python form decodes", func(t *testing.T) {
		var got session.EventActions
		if err := json.Unmarshal([]byte(`{"transferToAgent":"a","transferReason":"r"}`), &got); err != nil {
			t.Fatalf("json.Unmarshal: %v", err)
		}
		if got.TransferToAgent != "a" || got.TransferReason != "r" {
			t.Errorf("decoded (TransferToAgent, TransferReason) = (%q, %q), want (\"a\", \"r\")", got.TransferToAgent, got.TransferReason)
		}
	})
}
