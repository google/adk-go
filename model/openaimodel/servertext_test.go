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

package openaimodel

import (
	"errors"
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"

	"github.com/openai/openai-go/v3/responses"
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
			// was. A >= in place of the > would break this and nothing else.
			name:      "exactly the cap",
			in:        strings.Repeat("A", maxServerTextRunes),
			wantRunes: maxServerTextRunes,
		},
		{
			name:      "one past the cap",
			in:        strings.Repeat("A", maxServerTextRunes+1),
			wantRunes: maxServerTextRunes + 1, // the cap plus the marker
			wantMark:  true,
		},
		{
			// Multi-byte, so slicing by bytes rather than runes would sever a
			// rune and emit invalid UTF-8.
			name:      "multi-byte past the cap",
			in:        strings.Repeat("世", maxServerTextRunes+10),
			wantRunes: maxServerTextRunes + 1,
			wantMark:  true,
		},
		{
			name:      "astral past the cap",
			in:        strings.Repeat("🙂", maxServerTextRunes+10),
			wantRunes: maxServerTextRunes + 1,
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
	// terms of maxServerTextRunes, so raising the constant would otherwise slip
	// past all of them at once.
	if maxServerTextRunes != 256 {
		t.Errorf("maxServerTextRunes = %d, want 256; raising the cap is a deliberate change, so update this line and the bounds written against it", maxServerTextRunes)
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := clipServerText(tc.in)
			if n := utf8.RuneCountInString(got); n != tc.wantRunes {
				t.Errorf("clipServerText() returned %d runes, want %d", n, tc.wantRunes)
			}
			if !utf8.ValidString(got) {
				t.Errorf("clipServerText() returned invalid UTF-8: %q", got)
			}
			if marked := strings.HasSuffix(got, "…"); marked != tc.wantMark {
				t.Errorf("clipServerText() truncation marked = %v, want %v", marked, tc.wantMark)
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
		strings.Repeat("A", maxServerTextRunes),
		strings.Repeat("世", maxServerTextRunes+1),
		strings.Repeat("🙂", maxServerTextRunes+1),
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, in string) {
		got := clipServerText(in)
		if n := utf8.RuneCountInString(got); n > maxServerTextRunes+1 {
			t.Errorf("clipServerText(%q) returned %d runes, want at most %d", in, n, maxServerTextRunes+1)
		}
		if utf8.ValidString(in) && !utf8.ValidString(got) {
			t.Errorf("clipServerText(%q) turned valid UTF-8 into %q", in, got)
		}
		if strings.TrimSpace(got) != got {
			t.Errorf("clipServerText(%q) = %q, want no leading or trailing space", in, got)
		}
		// The marker sits at the end, so trailing space hides behind it.
		if body, marked := strings.CutSuffix(got, "…"); marked && strings.TrimSpace(body) != body {
			t.Errorf("clipServerText(%q) = %q, want no space before the truncation marker", in, got)
		}
	})
}

// FuzzFailedResponseError pins what no server-chosen string may do to the error
// built from it: carry something into an operator's log that splits a line, or
// grow without bound. %q is what enforces the first, so this is the test that
// notices if a field stops being quoted.
func FuzzFailedResponseError(f *testing.F) {
	f.Add("upstream exploded", "server_error", "resp_123")
	f.Add("line\r\nERROR forged", "code\nnope", "id\r\nforged")
	f.Add("\x1b[2J\x1b[1;1H", "\x7f", "\x00")
	f.Add("sep\u2028sep\u2029", "\u0085", "🙂")
	f.Add(strings.Repeat("A", 1<<12), strings.Repeat("B", 1<<12), strings.Repeat("C", 1<<12))
	// The worst case for the bound below: an unprintable astral rune is the
	// longest thing %q can produce from one rune.
	f.Add(strings.Repeat("\U0010ffff", 1<<10), strings.Repeat("\U0010ffff", 1<<10), strings.Repeat("\U0010ffff", 1<<10))
	f.Fuzz(func(t *testing.T, msg, code, id string) {
		err := failedResponseError(&responses.Response{
			ID:     id,
			Status: responses.ResponseStatusFailed,
			Error:  responses.ResponseError{Code: responses.ResponseErrorCode(code), Message: msg},
		})
		got := err.Error()
		for _, r := range got {
			// U+2028 and U+2029 are not IsControl, but split a line all the
			// same. %q escapes both, so neither may survive.
			if unicode.IsControl(r) || r == '\u2028' || r == '\u2029' {
				t.Errorf("failedResponseError(%q, %q, %q) = %q, which carries %U", msg, code, id, got, r)
			}
		}
		// Derived rather than guessed: each field is clipped to the cap plus
		// its truncation marker, %q renders the worst rune it can be made of —
		// an unprintable astral one — as the ten characters of \U0010ffff, and
		// wraps the result in two quotes. Three such fields, plus the labels
		// and the sentinel, is the ceiling.
		const maxRendered = 3*(10*(maxServerTextRunes+1)+2) + 64
		if n := utf8.RuneCountInString(got); n > maxRendered {
			t.Errorf("failedResponseError() produced %d runes, want at most %d", n, maxRendered)
		}
		if !errors.Is(err, ErrResponseFailed) {
			t.Errorf("failedResponseError() = %v, want it to wrap ErrResponseFailed", err)
		}
	})
}

// TestFailedResponseError_ServerText spells out on worked examples what the
// fuzz targets above assert in general.
func TestFailedResponseError_ServerText(t *testing.T) {
	t.Run("control characters cannot forge a line", func(t *testing.T) {
		err := failedResponseError(&responses.Response{
			ID:     "resp_1\r\nid forged",
			Status: responses.ResponseStatusFailed,
			Error: responses.ResponseError{
				Code:    "server_error\x1b[2J",
				Message: "line1\r\nERROR forged line 2\tand a tab\x1b[2J",
			},
		})
		got := err.Error()
		if strings.ContainsAny(got, "\r\n\t\x1b") {
			t.Errorf("error %q still carries a raw control character", got)
		}
		for _, want := range []string{`\r\n`, `\t`, `\x1b`} {
			if !strings.Contains(got, want) {
				t.Errorf("error %q does not render %s visibly", got, want)
			}
		}
	})

	t.Run("line separators cannot forge a line either", func(t *testing.T) {
		err := failedResponseError(&responses.Response{
			Status: responses.ResponseStatusFailed,
			Error:  responses.ResponseError{Message: "before\u2028after\u2029end"},
		})
		if got := err.Error(); strings.ContainsRune(got, '\u2028') || strings.ContainsRune(got, '\u2029') {
			t.Errorf("error %q still carries a raw line separator", got)
		}
	})

	t.Run("a huge message is capped", func(t *testing.T) {
		err := failedResponseError(&responses.Response{
			Status: responses.ResponseStatusFailed,
			Error:  responses.ResponseError{Message: strings.Repeat("A", 1<<20)},
		})
		// Runes, because that is the unit the cap counts in.
		if got, max := utf8.RuneCountInString(err.Error()), maxServerTextRunes+64; got > max {
			t.Errorf("error is %d runes, want at most %d", got, max)
		}
		if !strings.HasSuffix(err.Error(), `…"`) {
			t.Errorf("error %q does not mark that it was truncated", err.Error())
		}
	})

	t.Run("a huge id and code are capped", func(t *testing.T) {
		err := failedResponseError(&responses.Response{
			ID:     strings.Repeat("🙂", 1<<16),
			Status: responses.ResponseStatusFailed,
			Error:  responses.ResponseError{Code: responses.ResponseErrorCode(strings.Repeat("世", 1<<16))},
		})
		// Non-ASCII on purpose. Both runes are printable, so %q emits them as
		// themselves — one rune, several bytes each — and a bound counted in
		// bytes would trip on the encoding rather than on any real growth.
		if got, max := utf8.RuneCountInString(err.Error()), 3*(maxServerTextRunes+1)+64; got > max {
			t.Errorf("error is %d runes, want at most %d", got, max)
		}
	})

	t.Run("a short message is quoted, not altered", func(t *testing.T) {
		const msg = `upstream exploded (code 429): retry`
		err := failedResponseError(&responses.Response{
			Status: responses.ResponseStatusFailed,
			Error:  responses.ResponseError{Message: msg},
		})
		if want := `openai: response failed: "` + msg + `"`; err.Error() != want {
			t.Errorf("error = %q, want %q", err.Error(), want)
		}
	})
}

// TestReportsFailure_WhitespaceOnlyError pins the predicate to the renderer. A
// status-less body whose error is nothing but whitespace says nothing, so it
// is a turn: reading the raw value here while the renderer reads the clipped
// one would discard the output and report a failure naming nothing.
func TestReportsFailure_WhitespaceOnlyError(t *testing.T) {
	resp := &responses.Response{
		ID: "resp_123",
		Error: responses.ResponseError{
			Code:    " \t ",
			Message: "  \n ",
		},
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "output_text", Text: "a whole answer"},
				},
			},
		},
	}
	if reportsFailure(resp) {
		t.Error("reportsFailure() = true for an error object of nothing but whitespace, want false")
	}
	got, err := convertResponse(resp)
	if err != nil {
		t.Fatalf("convertResponse() err = %v, want the turn converted", err)
	}
	if text := got.Candidates[0].Content.Parts[0].Text; text != "a whole answer" {
		t.Errorf("text = %q, want the output the server produced", text)
	}
}

// TestStreamTranslator_ErrorEvent_ServerText pins the generic stream error to
// the same treatment as a failed response body: the text is the server's, so it
// is capped and quoted rather than interpolated bare.
func TestStreamTranslator_ErrorEvent_ServerText(t *testing.T) {
	t.Run("nothing that splits a line survives", func(t *testing.T) {
		event := decodeEvent(t, `{"type":"error","message":"boom\r\nERROR forged\u001b[2J\u2028sep\u0085"}`)
		_, err := newStreamTranslator().process(event)
		if err == nil {
			t.Fatal("process() err = nil, want the stream error")
		}
		for _, r := range err.Error() {
			if unicode.IsControl(r) || r == '\u2028' || r == '\u2029' {
				t.Errorf("process() err = %q carries %U", err, r)
			}
		}
	})

	t.Run("an error event with no message still errors", func(t *testing.T) {
		event := decodeEvent(t, `{"type":"error"}`)
		_, err := newStreamTranslator().process(event)
		if err == nil {
			t.Fatal("process() err = nil, want the stream error even with nothing to quote")
		}
	})

	t.Run("a huge message is capped", func(t *testing.T) {
		// Non-ASCII, so the bound is measured against the cap's own unit
		// rather than against an encoding that happens to be one byte wide.
		event := decodeEvent(t, `{"type":"error","message":"`+strings.Repeat("世", 1<<12)+`"}`)
		_, err := newStreamTranslator().process(event)
		// Guarded rather than dereferenced: were this path ever to stop
		// erroring, a nil dereference would take the whole package's results
		// down with it instead of failing this one assertion.
		if err == nil {
			t.Fatal("process() err = nil, want the stream error")
		}
		if got, max := utf8.RuneCountInString(err.Error()), maxServerTextRunes+64; got > max {
			t.Errorf("process() err is %d runes, want at most %d", got, max)
		}
	})
}
