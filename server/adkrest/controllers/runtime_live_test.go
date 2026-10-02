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

package controllers

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/gorilla/websocket"
)

func TestTruncateCloseReason(t *testing.T) {
	longErr := "live session: reconnect budget exhausted: 5 consecutive attempts delivered no content: " +
		"failed to receive message: websocket: close 1006 (abnormal closure): unexpected EOF"

	tests := []struct {
		name   string
		reason string
	}{
		{name: "short reason is untouched", reason: "agent not found"},
		{name: "exactly at the limit", reason: strings.Repeat("a", maxCloseReason)},
		{name: "over the limit", reason: longErr},
		{name: "multi-byte rune straddling the cut", reason: strings.Repeat("é", maxCloseReason)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := truncateCloseReason(tc.reason)
			if len(got) > maxCloseReason {
				t.Errorf("len = %d, want at most %d: gorilla drops the whole frame", len(got), maxCloseReason)
			}
			if !utf8.ValidString(got) {
				t.Error("result is not valid UTF-8, which a close reason must be")
			}
			if !strings.HasPrefix(tc.reason, got) {
				t.Errorf("result %q is not a prefix of the reason", got)
			}
			if len(tc.reason) <= maxCloseReason && got != tc.reason {
				t.Errorf("a reason that already fits was altered: %q", got)
			}
		})
	}

	// The frame gorilla actually builds must be acceptable to it.
	if got := len(websocket.FormatCloseMessage(websocket.CloseInternalServerErr, truncateCloseReason(longErr))); got > 125 {
		t.Errorf("close frame payload = %d bytes, want at most 125", got)
	}
}
