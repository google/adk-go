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

package llminternal

import "testing"

func TestEffectiveIncludeContents(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name            string
		mode            Mode
		includeContents string
		want            string
	}{
		{name: "unset_chat_defaults_default", mode: ModeChat, want: "default"},
		{name: "unset_single_turn_defaults_none", mode: ModeSingleTurn, want: "none"},
		{name: "explicit_default_preserved_on_single_turn", mode: ModeSingleTurn, includeContents: "default", want: "default"},
		{name: "explicit_none_preserved", mode: ModeChat, includeContents: "none", want: "none"},
		{name: "unset_mode_unset_defaults_default", mode: ModeUnset, want: "default"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := &State{Mode: tc.mode, IncludeContents: tc.includeContents}
			if got := s.EffectiveIncludeContents(); got != tc.want {
				t.Fatalf("EffectiveIncludeContents() = %q, want %q", got, tc.want)
			}
			if s.IncludeContents != tc.includeContents {
				t.Fatalf("IncludeContents mutated: got %q, want %q", s.IncludeContents, tc.includeContents)
			}
		})
	}
}
