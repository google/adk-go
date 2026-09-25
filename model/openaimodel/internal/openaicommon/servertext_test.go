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

package openaicommon

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// TestClipServerText pins the cap's arithmetic, which counts runes while the
// strings it guards are bytes.
func TestClipServerText(t *testing.T) {
	tests := []struct {
		name      string
		in        string
		wantRunes int
		wantMark  bool
	}{
		{name: "short", in: "upstream exploded", wantRunes: 17},
		{name: "blank", in: "  \n\t ", wantRunes: 0},
		{
			// Exactly at the cap: nothing was dropped, so nothing may claim it
			// was. Paired with the row below, it pins the boundary at exactly
			// MaxServerTextRunes, catching an off-by-one on either side.
			name:      "exactly the cap",
			in:        strings.Repeat("A", MaxServerTextRunes),
			wantRunes: MaxServerTextRunes,
		},
		{
			name:      "one past the cap",
			in:        strings.Repeat("A", MaxServerTextRunes+1),
			wantRunes: MaxServerTextRunes + 1, // the cap plus the marker
			wantMark:  true,
		},
		{
			// Multi-byte, so slicing by bytes rather than runes would sever a
			// rune and emit invalid UTF-8.
			name:      "multi-byte past the cap",
			in:        strings.Repeat("世", MaxServerTextRunes+10),
			wantRunes: MaxServerTextRunes + 1,
			wantMark:  true,
		},
		{
			name:      "astral past the cap",
			in:        strings.Repeat("🙂", MaxServerTextRunes+10),
			wantRunes: MaxServerTextRunes + 1,
			wantMark:  true,
		},
		{
			// The cap lands inside a run of spaces, so the marker would
			// otherwise be pushed out behind them.
			name:      "cut inside whitespace",
			in:        strings.Repeat("A", 250) + strings.Repeat(" ", 10) + strings.Repeat("B", 10),
			wantRunes: 251, // 250 kept, the spaces dropped, plus the marker
			wantMark:  true,
		},
	}
	// Asserted absolutely, once. Every other bound in this file is written in
	// terms of MaxServerTextRunes, so raising the constant would otherwise slip
	// past all of them at once.
	if MaxServerTextRunes != 256 {
		t.Errorf("MaxServerTextRunes = %d, want 256; raising the cap is a deliberate change, so update this line and the bounds written against it", MaxServerTextRunes)
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ClipServerText(tc.in)
			if n := utf8.RuneCountInString(got); n != tc.wantRunes {
				t.Errorf("ClipServerText() returned %d runes, want %d", n, tc.wantRunes)
			}
			if !utf8.ValidString(got) {
				t.Errorf("ClipServerText() returned invalid UTF-8: %q", got)
			}
			if marked := strings.HasSuffix(got, "…"); marked != tc.wantMark {
				t.Errorf("ClipServerText() truncation marked = %v, want %v", marked, tc.wantMark)
			}
		})
	}
}

// FuzzClipServerText pins the invariants the cap exists for, on input no table
// would think to write. Severing a multi-byte rune shows up here as invalid
// UTF-8, which is what covers the boundary arithmetic.
func FuzzClipServerText(f *testing.F) {
	for _, seed := range []string{
		"", "  ", "upstream exploded", "line\r\nforged", "\x1b[2J", "\u2028sep",
		// An ellipsis the server itself sent, which is not a truncation marker.
		"rate limited, retrying …",
		strings.Repeat("A", MaxServerTextRunes),
		strings.Repeat("世", MaxServerTextRunes+1),
		strings.Repeat("🙂", MaxServerTextRunes+1),
		// Clipped mid-whitespace, so the marker assertion below has something
		// to bite on without waiting for the fuzzer to find it.
		strings.Repeat("A", 250) + strings.Repeat(" ", 10) + strings.Repeat("B", 10),
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, in string) {
		got := ClipServerText(in)
		if n := utf8.RuneCountInString(got); n > MaxServerTextRunes+1 {
			t.Errorf("ClipServerText(%q) returned %d runes, want at most %d", in, n, MaxServerTextRunes+1)
		}
		if utf8.ValidString(in) && !utf8.ValidString(got) {
			t.Errorf("ClipServerText(%q) turned valid UTF-8 into %q", in, got)
		}
		if strings.TrimSpace(got) != got {
			t.Errorf("ClipServerText(%q) = %q, want no leading or trailing space", in, got)
		}
		// The marker sits at the end, so trailing space hides behind it. Gated
		// on the input having actually been clipped, because a trailing "…" the
		// server sent is its own text and the space before it is not ours to
		// judge.
		if utf8.RuneCountInString(strings.TrimSpace(in)) > MaxServerTextRunes {
			if body, marked := strings.CutSuffix(got, "…"); marked && strings.TrimSpace(body) != body {
				t.Errorf("ClipServerText(%q) = %q, want no space before the truncation marker", in, got)
			}
		}
	})
}
