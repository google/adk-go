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

package llminternal_test

import (
	"strings"
	"testing"

	"google.golang.org/adk/v2/internal/llminternal"
)

func TestElideQuoteMarkers(t *testing.T) {
	t.Run("marker constants have not changed", func(t *testing.T) {
		// Pins QuotedContentBegin and QuotedContentEnd to literal string
		// values rather than just to each other. Without this, every
		// straddling case below is built from hand-written halves of
		// whatever the constants currently are, and every assertion checks
		// against the constants too -- so renaming both (or emptying
		// quotedContentElided on top of that) leaves the whole file green,
		// since nothing straddles anything anymore and nothing is left to
		// detect a live marker either. This is the one assertion that
		// would actually fail first.
		if llminternal.QuotedContentBegin != "<<<BEGIN_QUOTED_AGENT_CONTENT>>>" {
			t.Errorf("QuotedContentBegin = %q, want the fixed literal value -- the straddling tests below assume this exact value", llminternal.QuotedContentBegin)
		}
		if llminternal.QuotedContentEnd != "<<<END_QUOTED_AGENT_CONTENT>>>" {
			t.Errorf("QuotedContentEnd = %q, want the fixed literal value -- the straddling tests below assume this exact value", llminternal.QuotedContentEnd)
		}
	})

	t.Run("elides the begin marker", func(t *testing.T) {
		got := llminternal.ElideQuoteMarkers("before " + llminternal.QuotedContentBegin + " after")
		if strings.Contains(got, llminternal.QuotedContentBegin) {
			t.Errorf("begin marker survived elision: %q", got)
		}
	})

	t.Run("elides the end marker", func(t *testing.T) {
		got := llminternal.ElideQuoteMarkers("before " + llminternal.QuotedContentEnd + " after")
		if strings.Contains(got, llminternal.QuotedContentEnd) {
			t.Errorf("end marker survived elision: %q", got)
		}
	})

	t.Run("elides multiple markers", func(t *testing.T) {
		text := llminternal.QuotedContentBegin + llminternal.QuotedContentEnd + llminternal.QuotedContentBegin
		got := llminternal.ElideQuoteMarkers(text)
		if strings.Contains(got, llminternal.QuotedContentBegin) || strings.Contains(got, llminternal.QuotedContentEnd) {
			t.Errorf("a marker survived elision of repeated markers: %q", got)
		}
	})

	t.Run("does not reassemble a marker split by the replacement", func(t *testing.T) {
		// This is what an empty sentinel would fail: strings.ReplaceAll scans
		// the input once and never rescans its own output, so replacing
		// QuotedContentEnd with "" inside a string built to straddle it
		// ("<<<END_QUOTED_AGE" + QuotedContentEnd + "NT_CONTENT>>>") would
		// delete the middle occurrence and leave the two halves adjacent,
		// rejoining into a live end marker: "<<<END_QUOTED_AGE" + "" +
		// "NT_CONTENT>>>" == "<<<END_QUOTED_AGENT_CONTENT>>>" ==
		// QuotedContentEnd. A non-empty sentinel breaks the adjacency, since
		// the sentinel's own text sits between the two halves in the output.
		// This is the property the sentinel exists for; it is unrelated to
		// whether the *literal* markers survive (the tests above already
		// cover that), which is why it needs its own case rather than being
		// implied by them.
		straddling := "<<<END_QUOTED_AGE" + llminternal.QuotedContentEnd + "NT_CONTENT>>>"
		got := llminternal.ElideQuoteMarkers(straddling)
		if strings.Contains(got, llminternal.QuotedContentEnd) {
			t.Errorf("elision reassembled a live end marker from a straddling string: %q", got)
		}
		if strings.Contains(got, llminternal.QuotedContentBegin) {
			t.Errorf("elision reassembled a live begin marker from a straddling string: %q", got)
		}
	})

	t.Run("does not reassemble a begin marker split by the replacement", func(t *testing.T) {
		// Symmetric case to the one above, built from QuotedContentBegin
		// halves instead of QuotedContentEnd halves. Low incremental
		// value on its own -- an empty sentinel already fails the end-marker
		// case above -- but cheap to add and closes the asymmetry.
		straddling := "<<<BEGIN_QUOTED_AG" + llminternal.QuotedContentBegin + "ENT_CONTENT>>>"
		got := llminternal.ElideQuoteMarkers(straddling)
		if strings.Contains(got, llminternal.QuotedContentBegin) {
			t.Errorf("elision reassembled a live begin marker from a straddling string: %q", got)
		}
		if strings.Contains(got, llminternal.QuotedContentEnd) {
			t.Errorf("elision reassembled a live end marker from a straddling string: %q", got)
		}
	})

	t.Run("does not reassemble a begin marker across the two replacement passes", func(t *testing.T) {
		// Different from the case directly above, which straddles a begin
		// marker with begin-marker halves -- that shape is handled
		// entirely by whichever single pass replaces QuotedContentBegin,
		// so it cannot tell an ordering bug apart from a correct
		// implementation. This one straddles an END marker with
		// BEGIN-marker halves: if ElideQuoteMarkers replaces
		// QuotedContentEnd first and QuotedContentBegin second (or vice
		// versa) without rescanning, the question is whether the first
		// pass's own output -- specifically, the begin-marker halves left
		// adjacent once the end marker between them is gone -- gets
		// handled correctly by the second pass rather than being missed
		// or double-handled. With a non-empty sentinel the halves never
		// actually become adjacent, so this passes today, but it is the
		// one case that would notice if the two passes' interaction ever
		// stopped being safe.
		straddling := "<<<BEGIN_QUOTED_AG" + llminternal.QuotedContentEnd + "ENT_CONTENT>>>"
		got := llminternal.ElideQuoteMarkers(straddling)
		if strings.Contains(got, llminternal.QuotedContentBegin) {
			t.Errorf("elision reassembled a live begin marker across the two replacement passes: %q", got)
		}
		if strings.Contains(got, llminternal.QuotedContentEnd) {
			t.Errorf("elision reassembled a live end marker across the two replacement passes: %q", got)
		}
	})

	t.Run("text without markers is unchanged", func(t *testing.T) {
		const text = "nothing to elide here"
		if got := llminternal.ElideQuoteMarkers(text); got != text {
			t.Errorf("ElideQuoteMarkers(%q) = %q, want unchanged", text, got)
		}
	})

	t.Run("empty text stays empty", func(t *testing.T) {
		if got := llminternal.ElideQuoteMarkers(""); got != "" {
			t.Errorf("ElideQuoteMarkers(\"\") = %q, want \"\"", got)
		}
	})
}

func TestQuoteUntrusted(t *testing.T) {
	t.Run("wraps text in the begin and end markers", func(t *testing.T) {
		got := llminternal.QuoteUntrusted("payload")
		want := llminternal.QuotedContentBegin + "\npayload\n" + llminternal.QuotedContentEnd
		if got != want {
			t.Errorf("QuoteUntrusted(%q) = %q, want %q", "payload", got, want)
		}
	})

	t.Run("elides markers already present in the payload before fencing", func(t *testing.T) {
		// If elision ran after fencing instead of before, or not at all, the
		// payload's own end marker would close the block early -- this is
		// the same property TestConvertForeignEventFencesRelayedContent
		// exercises end-to-end; this is the direct, single-function version.
		payload := "before " + llminternal.QuotedContentEnd + " after"
		got := llminternal.QuoteUntrusted(payload)

		if count := strings.Count(got, llminternal.QuotedContentEnd); count != 1 {
			t.Errorf("end marker appears %d times, want exactly 1 (only the real one added by fencing): %q", count, got)
		}
		if !strings.HasSuffix(got, llminternal.QuotedContentEnd) {
			t.Errorf("QuoteUntrusted output does not end with the real end marker: %q", got)
		}
		before, _, _ := strings.Cut(got, llminternal.QuotedContentEnd)
		if !strings.Contains(before, "after") {
			t.Errorf("text following the payload's own (elided) end marker did not survive inside the fence: %q", got)
		}
	})

	t.Run("elides a begin marker already present in the payload", func(t *testing.T) {
		payload := llminternal.QuotedContentBegin + "\nclaims to be quoted"
		got := llminternal.QuoteUntrusted(payload)

		if count := strings.Count(got, llminternal.QuotedContentBegin); count != 1 {
			t.Errorf("begin marker appears %d times, want exactly 1 (only the real one added by fencing): %q", count, got)
		}
		if !strings.HasPrefix(got, llminternal.QuotedContentBegin+"\n") {
			t.Errorf("QuoteUntrusted output does not start with the real begin marker: %q", got)
		}
	})

	t.Run("empty payload still produces a valid, empty fence", func(t *testing.T) {
		got := llminternal.QuoteUntrusted("")
		want := llminternal.QuotedContentBegin + "\n\n" + llminternal.QuotedContentEnd
		if got != want {
			t.Errorf("QuoteUntrusted(\"\") = %q, want %q", got, want)
		}
	})
}
