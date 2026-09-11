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

package gcp

import (
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// This file is the scrub: everything that turns service-controlled text into
// text an error may carry. It shares auth/gcp with the credential client only
// because the client is its one caller — nothing in it is specific to GCP.

// maxErrorBody caps service-controlled text carried into an error.
const maxErrorBody = 1024

// withheldText replaces service text that could not be shown free of the caller's
// own identifiers. It names no secret and is constant, so it leaks nothing itself.
const withheldText = "[withheld: could not be shown free of the request's own identifiers]"

// redactedMarker is what redact writes in place of a removed value.
const redactedMarker = "[redacted]"

// maxScrubbableSecret bounds the caller-supplied values this package will match
// against a response. Above it, nothing is shown at all.
//
// Matching costs O(body x value), and the body reaches doPost's 1 MiB read cap
// while Request.UserID and Request.ContinueURI are bounded by nothing here — a
// 512 KiB value against a megabyte of response measured 9.1s of uninterruptible
// CPU, and decodeFully is quadratic on the same input. Half-scrubbing is not an
// option, so the choice is between a bound and a denial of service, and the
// bound fails closed.
//
// 4 KiB rather than something tighter because it has to clear every real
// identifier by a wide margin: an address is at most 320 bytes and a redirect URI
// is held under 2 KiB by what servers and browsers accept. At this bound the
// worst case is milliseconds.
//
// The bound is measured on the RAW value while matching runs on the lowered copy,
// and the two differ: strings.ToLower expands a byte that is not valid UTF-8 into
// a three-byte U+FFFD, so a value at this bound can be matched as up to three
// times its length. Left as it is because the factor is constant and the result
// stays inside the budget — the same shape measures 38 ms in valid UTF-8 and
// 230 ms built from invalid bytes, against a 5-second ceiling — while folding the
// value first to measure it would reject callers this bound is not aimed at.
const maxScrubbableSecret = 4096

// redactedForError prepares service-controlled text so that an error may carry
// it: the caller's own identifiers removed and the source capped — or, where
// neither can be shown to have worked, none of the response at all. What comes
// back is therefore not the service's text, and on that last path bears no
// relation to it.
//
// Contract: no secret is recoverable from the returned string by this package's
// decoder, or nothing is returned. A caller may quote the result into an error a
// model reads and a session stores.
//
// Not covered, and the boundary is exact: an identifier a service mangles into a
// form no decoder reconstructs and a human reads anyway — split with a \n,
// percent-encoded as %40, or written \U0040 with the capital U this package's
// decoder does not accept, or echoed in half. The guarantee is that WE add no
// identifier, not that we can launder one back out of arbitrary text.
//
// A value longer than [maxScrubbableSecret] is not matched at all and nothing is
// returned, because scrubbing it costs more than the diagnostic is worth.
//
// A CANDIDATE longer than [maxStraddleWindow] is withheld whenever the visible
// cap cut it, because past that length an occurrence cannot be seen whole. The
// candidate is what is measured, not the response: the second one is decoded a
// pass before it is offered, so a response that falls under the threshold once
// decoded is examined in full and can still be shown. 120,000 bytes of \u0041
// decode to 20,000 and come back scrubbed rather than withheld.
//
// The ContinueURI is covered less thoroughly than the UserID in one shape. Where
// the UserID is a substring of the URI and the service escapes the URI so the
// scrub cannot match it, the UserID's marker lands inside the URI and the rest of
// the URI stays visible around it. The contract holds as stated, since this
// package's decoder does not reconstruct the URI from what is returned, but a
// reader sees most of it: https://github.com/google/adk-go/issues/1539.
//
// Two things here look like they could be simpler and cannot be.
//
// The choice between the two candidates is made on the OUTPUT, never on a property
// of s and u. The service writes the text both candidates are measured from, so any
// test over those inputs is one it controls: adding an occurrence the decode
// destroys costs it nothing and moves the comparison wherever it likes. Three
// revisions were broken that way before this one.
//
// The cap is applied to the text being READ, not to the scrubbed result, and
// [redactWithinLimit] carries why both of the simpler orders leak.
func redactedForError(s string, secrets ...string) string {
	for _, v := range secrets {
		if len(v) > maxScrubbableSecret {
			return withheldText
		}
	}
	if out, ok := showable(s, secrets); ok {
		return out
	}
	if u := unescapeJSON(s); u != s {
		if out, ok := showable(u, secrets); ok {
			return out
		}
	}
	// Neither candidate can be shown clean. The status code, the resource and the
	// sentinel all survive in the error around this.
	return withheldText
}

// maxStraddleWindow bounds how much of the response the whole-text check reads.
//
// Only an occurrence that STRADDLES the cut can be hidden by it — bytes wholly
// past the cut are never shown, so they cannot leak — and an occurrence has to
// start before the cut to straddle it. So the check needs the visible window plus
// room for one occurrence's escaped spelling, not the whole megabyte doPost
// admits. Eight times the value bound covers any singly escaped spelling of a
// bounded value with room to spare.
//
// It is NOT a bound on how inflated a spelling may be, and no constant is one.
// The service chooses the spelling: a character at nesting level k costs
// 2^(k-1)+5 bytes and needs k decode passes, so a spelling that runs past any
// fixed window is still reconstructed by decodeFully inside its own pass bound.
// Reading part of such an occurrence answers the question wrong rather than
// conservatively — the check sees a proper prefix, matches nothing, and the
// visible kilobyte goes out carrying a decodable fragment of the value. So a
// response running past the window is withheld rather than examined in part,
// which is also cheaper than examining it.
//
// Without a window the check ran redact over the full body and split the result
// on the marker: a one-character value against a megabyte measured 39.8 MiB
// allocated and 524,289 parts, to compute one boolean.
const maxStraddleWindow = maxErrorBody + 8*maxScrubbableSecret

// fitsStraddleWindow reports whether the whole-text check can examine s whole.
//
// It is a threshold rather than a slice. An earlier version handed back the
// prefix, which no caller can use: the part of an occurrence that fits says
// nothing about the occurrence, and reading it is what returned a decodable
// fragment of a value.
func fitsStraddleWindow(s string) bool {
	return len(s) <= maxStraddleWindow
}

// showable scrubs s for an error and reports whether the result can be shown.
func showable(s string, secrets []string) (string, bool) {
	out, truncated := redactWithinLimit(s, secrets...)
	// Asked of the WHOLE text as well, and only when the cap actually cut, because
	// the visible part alone cannot answer it. An occurrence the scrub could not
	// match — an escaped spelling — that straddles the cut is sliced in half, and
	// half an identifier matches nothing, so what is shown looks clean while the
	// response plainly carried the identifier and this package's own decoder gets
	// it back out.
	//
	// Past the window there is no answer to give, only a guess, so the response is
	// withheld unexamined — see [maxStraddleWindow]. The redact below therefore
	// runs only on text already known to be within that bound.
	if truncated && (!fitsStraddleWindow(s) || recoverable(redact(s, secrets...), secrets)) {
		return "", false
	}
	// Checked AFTER the cap, so what is examined is exactly what is returned. The
	// cap is not neutral: its ellipsis is appended text, and appended text can
	// finish a secret the untruncated string only started — an identifier ending
	// in a dot is completed by the first character of "...".
	if recoverable(out, secrets) {
		return "", false
	}
	return out, true
}

// recoverable reports whether any secret can be read out of x, either literally or
// once this package's decoder has been run over it to a fixpoint.
//
// Decoding x rather than trusting redact's own bookkeeping is the point: it asks
// what an attacker gets from the bytes being returned, so it cannot be steered by
// what the service put in the bytes that were measured.
//
// The marker this scrub writes must be kept out of the search, or a one-character
// user id of "e" occurs inside "[redacted]" and every response that redacted
// anything is suppressed. It is kept out by POSITION: an occurrence belongs to us
// only when every byte of it came from a marker. Excluding it by splitting x on
// the marker instead excluded whole occurrences rather than marker bytes, so a
// value containing or abutting the marker spanned a split, landed whole in no
// part, and was matched by nothing — `a[redacted]b` came back verbatim.
func recoverable(x string, secrets []string) bool {
	// Built once, outside the search, and the decoded spelling too, because
	// decoding is what mangles a secret. An identifier containing \/ survives a
	// decoded copy's scrub as the slash it decodes to, which is not the secret and
	// is still the identity.
	var forms []string
	for _, v := range secrets {
		if v == "" {
			continue
		}
		decoded, done := decodeFully(v)
		if !done {
			// Fails closed, here and below. Not finishing the decode means not
			// knowing what this text says, and "unknown" has to answer the same
			// way as "yes" when the question is whether a secret is readable.
			return true
		}
		for _, form := range []string{strings.ToLower(v), strings.ToLower(decoded)} {
			if form != "" {
				forms = append(forms, form)
			}
		}
	}
	if len(forms) == 0 {
		return false
	}
	// Decoded once for the whole text rather than once per marker-separated part.
	// The SERVICE picked how many parts there were by writing the marker into its
	// own response — a kilobyte of markers is about a hundred — and per-part work
	// is what measured 3m33.6s for one call.
	decoded, done := decodeFully(x)
	if !done {
		return true
	}
	for _, text := range []string{strings.ToLower(x), strings.ToLower(decoded)} {
		// Lowered first, then scanned: the marker is already lower case, so it
		// survives the fold unchanged and its offsets are the folded copy's own.
		ours := markerBytes(text)
		for _, form := range forms {
			if readableOutsideMarkers(text, ours, form) {
				return true
			}
		}
	}
	return false
}

// markerBytes marks every byte of text belonging to an occurrence of the marker,
// and returns nil when there is none — the common case, which then costs one
// search rather than an allocation.
func markerBytes(text string) []bool {
	i := strings.Index(text, redactedMarker)
	if i < 0 {
		return nil
	}
	ours := make([]bool, len(text))
	for i >= 0 {
		for k := i; k < i+len(redactedMarker); k++ {
			ours[k] = true
		}
		next := strings.Index(text[i+len(redactedMarker):], redactedMarker)
		if next < 0 {
			break
		}
		i += len(redactedMarker) + next
	}
	return ours
}

// readableOutsideMarkers reports whether form occurs anywhere in text other than
// wholly inside bytes this scrub wrote.
//
// Overlapping occurrences are walked one byte at a time rather than skipped past,
// since a masked hit says nothing about the next one. Stepping by len(form)
// instead passes every test here, and that is a property of the marker rather
// than a gap in them: "[redacted]" has no non-empty border, so its occurrences
// never overlap and a fully masked hit always sits inside a run periodic in ten.
// The byte step costs nothing at these sizes and does not rest on that argument.
func readableOutsideMarkers(text string, ours []bool, form string) bool {
	for i := 0; i+len(form) <= len(text); {
		j := strings.Index(text[i:], form)
		if j < 0 {
			return false
		}
		at := i + j
		if !allOurs(ours, at, at+len(form)) {
			return true
		}
		i = at + 1
	}
	return false
}

// allOurs reports whether every byte of [lo,hi) came from a marker. A text with
// no marker in it owns nothing, so nothing is excused.
func allOurs(ours []bool, lo, hi int) bool {
	if ours == nil {
		return false
	}
	for i := lo; i < hi; i++ {
		if !ours[i] {
			return false
		}
	}
	return true
}

// maxDecodePasses bounds decodeFully. Legitimate text needs one pass, or two
// where a body was JSON-encoded twice; this is two orders of magnitude above
// that, so reaching it means the text was built to be expensive rather than to
// be read.
//
// It counts READS. A text is reported finished only by a read that changes
// nothing, so the last read has nothing after it to confirm the result and the
// most changing passes that can be reported finished is one fewer. The direction
// is fail-closed — text that needed exactly this many changing passes is called
// unfinished — and the boundary is pinned by TestDecodeFullyStopsAtItsPassBound.
const maxDecodePasses = 64

// decodeFully applies unescapeJSON until the text stops changing, and reports
// whether it got there.
//
// One pass is not enough. A doubly escaped identifier decodes to a singly escaped
// one, which still hides it from a substring scrub and which this same function
// will happily decode the rest of the way for anyone who asks twice — so checking
// only the first pass leaves an identifier the package's own decoder recovers.
//
// Termination alone is not enough either, which is what the pass bound is for.
// Every branch that changes anything writes fewer bytes than it consumed, so a
// changing pass strictly shortens — but it can shorten by as little as five
// bytes, and `\u005c` is the input that does exactly that while re-forming the
// introducer for the next one. Unbounded, that is one pass per five bytes of a
// response this package reads a megabyte of: 20 KB measured 233ms, 80 KB 3.5s,
// and 1 MiB over nine minutes of CPU no caller's deadline could cancel.
//
// TestDecodeFullyShrinksWheneverItChanges pins the per-pass lemma the argument
// rests on, over unescapeJSON rather than over this loop, and
// TestDecodeFullyStopsAtItsPassBound pins the bound.
//
// The bound is reported rather than swallowed because the answer this feeds is a
// disclosure decision. A caller that cannot finish decoding has not shown the
// text to be clean, and must treat it as though a secret were in there.
func decodeFully(s string) (string, bool) {
	for range maxDecodePasses {
		u := unescapeJSON(s)
		if u == s {
			return s, true
		}
		s = u
	}
	return s, false
}

// unescapeJSON decodes the JSON escapes that can hide a caller-supplied value
// from a substring scrub, and leaves every other one alone — a malformed escape
// included, which stays verbatim rather than being dropped.
//
// Three are decoded, and on valid UTF-8 built from those three plus ordinary
// text this agrees with encoding/json, which is what the differential test
// asserts. On a byte that is not valid UTF-8 the two part company on purpose:
// encoding/json replaces it with U+FFFD, and this copies it through untouched,
// because the input here is an arbitrary response body rather than a decoded
// string and substituting a byte the service sent would be the fabrication this
// scrub is trying to avoid.
//
// \uXXXX, because a JSON encoder commonly escapes non-ASCII, and a surrogate
// PAIR as one rune rather than two halves. A non-BMP identifier arrives as two
// \uXXXX escapes. Decoding each alone yields U+FFFD twice, so the identifier is
// in neither the decoded copy nor the original and survives the scrub whole.
// That is not an encoding this function declines to handle — it is two of the
// escapes it says it decodes.
//
// \/, because RFC 8259 permits it and several encoders emit it by default,
// which is enough to hide every slash in an echoed ContinueURI.
//
// A doubled backslash, because it is what stops a literal one from introducing
// either of the other two. Skipping it is not neutral: the second backslash then
// opens an escape JSON says is not there, so \\uXXXX decoded to a backslash
// followed by the rune rather than to the six literal characters.
func unescapeJSON(s string) string {
	if !strings.Contains(s, `\`) {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); {
		switch {
		case hasEscape(s, i, '\\'):
			b.WriteByte('\\')
			i += shortEscapeLen
		case hasEscape(s, i, '/'):
			b.WriteByte('/')
			i += shortEscapeLen
		default:
			r, ok := decodeUnicodeEscape(s, i)
			if !ok {
				// Not an escape this function decodes, or a malformed one. Either
				// way the byte is copied through rather than dropped.
				b.WriteByte(s[i])
				i++
				continue
			}
			// A non-BMP rune arrives as a surrogate PAIR, and the two halves have
			// to be decoded together: apart, each is an unpaired surrogate that
			// becomes U+FFFD, so the rune would be in neither the decoded copy nor
			// the original and would survive the scrub whole.
			if utf16.IsSurrogate(r) {
				if lo, ok := decodeUnicodeEscape(s, i+unicodeEscapeLen); ok {
					if paired := utf16.DecodeRune(r, lo); paired != utf8.RuneError {
						b.WriteRune(paired)
						i += surrogatePairLen
						continue
					}
				}
			}
			b.WriteRune(r)
			i += unicodeEscapeLen
		}
	}
	return b.String()
}

// Widths in bytes of the escapes unescapeJSON decodes. They are the reason for
// every bounds check in it: reading a \uXXXX needs six bytes to be there.
const (
	shortEscapeLen   = 2                    // \\ and \/
	unicodeEscapeLen = 6                    // \uXXXX
	surrogatePairLen = 2 * unicodeEscapeLen // \uD83D\uDE00
)

// hasEscape reports whether s[i:] begins with a backslash followed by c.
func hasEscape(s string, i int, c byte) bool {
	return i+shortEscapeLen <= len(s) && s[i] == '\\' && s[i+1] == c
}

// decodeUnicodeEscape decodes the \uXXXX at s[i:], if that is what is there.
//
// The four digits are read with an explicit base, which is what keeps this to
// exactly the escapes JSON defines: at base 16 strconv admits no sign, no 0x
// prefix and no underscores, so nothing but four hex digits parses.
func decodeUnicodeEscape(s string, i int) (rune, bool) {
	if !hasEscape(s, i, 'u') || i+unicodeEscapeLen > len(s) {
		return 0, false
	}
	n, err := strconv.ParseUint(s[i+shortEscapeLen:i+unicodeEscapeLen], 16, 32)
	if err != nil {
		return 0, false
	}
	return rune(n), true
}

// visibleLimit reports how many bytes of s an error may show, and whether s ran
// past that. Factored out of truncateForError so the scan can stop at the same
// place instead of the text being cut after it — see redactWithinLimit.
func visibleLimit(s string) (cut int, truncated bool) {
	const max = maxErrorBody
	if len(s) <= max {
		return len(s), false
	}
	// Back up to a rune boundary so a multi-byte rune straddling the cap isn't
	// sliced into a mangled partial rune. Bounded: the body need not be UTF-8 at
	// all, and an unbounded scan over continuation bytes would walk to 0 and
	// discard every byte of diagnostic context.
	cut = max
	for i := 0; i < utf8.UTFMax-1 && cut > 0 && !utf8.RuneStart(s[cut]); i++ {
		cut--
	}
	if !utf8.RuneStart(s[cut]) {
		cut = max
	}
	return cut, true
}

// truncateForError caps an error body so a large (e.g. HTML gateway) response
// doesn't bloat the returned error, and reports whether it cut.
//
// It returns both because a caller that takes the flag from one string and the
// text from another can be handed a disagreement: strings.ToLower folds U+0130
// from two bytes to one, so a lowered copy sits under the cap while the original
// crosses it. That flag is what arms the whole-text check in showable, and with
// it false the check never runs — measured at 14 of an 18-byte address in
// cleartext. Returning the pair from one place makes the pairing structural.
func truncateForError(s string) (string, bool) {
	cut, truncated := visibleLimit(s)
	if !truncated {
		return s, false
	}
	return s[:cut] + "...", true
}

// redact removes caller-supplied values from a service-controlled string,
// ignoring case. Where it removes anything, the text it returns is lowercased.
//
// Case-insensitively because the two sides need not agree on it: a server taking
// session.UserID from an OIDC email claim keeps whatever case the provider sent,
// while the services lowercase an address before echoing it, so a literal match
// misses the echo entirely. Unlike an encoding, that is not something the
// best-effort caveat above covers — it is the plain identifier, spelled the same.
//
// Lowercased rather than spliced back into the original because there is no cheap
// way to map an offset in the lowered copy onto the original. Lowercasing changes
// byte length in both directions — Go folds U+0130 to a one-byte "i" and U+023A
// to a three-byte U+2C65 — so a body carrying equal numbers of each has the same
// total length lowered as unlowered while every offset inside it has moved. A
// guard comparing totals sees nothing wrong, the splice lands short, and the
// identifier survives in full. That was a real bug here, and keeping the
// service's capitalisation is not worth another.
//
// Every value is matched in ONE left-to-right pass rather than one pass each.
// Sequential passes let the second value match inside the marker the first
// inserted — a user id of "e" turns "[redacted]" into "[r[redacted]dact[redacted]d]"
// — and let a short value break a longer one that contains it, so the longer one
// then matches nothing and its remainder survives. Overlapping candidates are
// resolved earliest-first, then longest, which is what keeps a ContinueURI
// carrying the user id from being split by the user id.
//
// Text with no match is returned untouched, so an error carrying no secret keeps
// its case. Empty values are dropped, and that is load-bearing rather than
// cosmetic: the empty string matches at the cursor forever, so admitting one
// would leave the cursor where it is and the loop would never finish.
func redact(s string, values ...string) string {
	lowered := loweredValues(values)
	if len(lowered) == 0 {
		return s
	}
	ls := strings.ToLower(s)
	out, hit := redactLowered(ls, len(ls), lowered)
	if !hit {
		return s
	}
	return out
}

// redactWithinLimit is redact with the error-body cap applied to the text it READS
// rather than to the text it returns.
//
// The order matters and the obvious one is wrong in both directions. Capping the
// scrubbed text lets redaction's own shortening pull bytes into view that the
// same cap on the raw response would have hidden: a body that echoes the
// ContinueURI thirty times and then names the user id past the kilobyte mark
// collapses every echo to a ten-byte marker, and the id rides up into the
// window. Capping the raw text instead would slice an occurrence straddling the
// cut in half, and half an identifier matches nothing, so the surviving prefix
// would be copied straight out.
//
// So the cap bounds which bytes of s may be EMITTED while matching still runs
// over the whole of s. A value that starts before the cut and ends after it is
// removed whole, and nothing at or past the cut is shown either way.
//
// The bound is measured on the lowered copy, because that is what the result is
// built from and lowercasing does not preserve byte offsets. When nothing
// matches there is no lowered copy in play and s is capped on its own bytes.
//
// Bounding the source bytes does not bound the result, so the result is capped
// on its own length as well. One matched run becomes a ten-byte marker however
// short the run was, and the SERVICE picks how many runs there are: `u u u …`
// against a user id of "u" produced 5635 bytes from a kilobyte of source, and
// what comes back is quoted into a prompt and stored in a session.
func redactWithinLimit(s string, values ...string) (string, bool) {
	lowered := loweredValues(values)
	if len(lowered) == 0 {
		return truncateForError(s)
	}
	ls := strings.ToLower(s)
	limit, truncated := visibleLimit(ls)
	out, hit := redactLowered(ls, limit, lowered)
	if !hit {
		// s is what is returned here, so s is what the flag has to describe. The
		// lowered copy's own limit says nothing about it.
		return truncateForError(s)
	}
	// One ellipsis however many bounds cut. Appending before the second cut lets
	// that cut land inside the dots just written and the next append stack more on
	// top, which measured five of them.
	cut, over := visibleLimit(out)
	if over {
		out = out[:cut]
	}
	if truncated || over {
		out += "..."
	}
	return out, truncated || over
}

// loweredValues drops the empty values and lowercases the rest. Empty values are
// dropped, and that is load-bearing rather than cosmetic: the empty string
// matches at the cursor forever, so admitting one would leave the cursor where
// it is and the loop would never finish.
func loweredValues(values []string) []string {
	lowered := make([]string, 0, len(values))
	for _, v := range values {
		if v != "" {
			lowered = append(lowered, strings.ToLower(v))
		}
	}
	return lowered
}

// redactLowered runs the scan over ls, which is already lowered, with lowered
// already non-empty and lowercased. It emits no byte of ls at or past limit, and
// reports whether it wrote a marker — if it did not, the caller returns the
// original text rather than the lowered copy.
//
// Cost is O(len(ls) x total value length): the outer scan and the next[] refresh
// are linear in len(ls) per value, but the extension walk below re-compares a
// value at every position of a range it covers. redactedForError bounds the second
// factor by refusing to scrub a value longer than maxScrubbableSecret.
func redactLowered(ls string, limit int, lowered []string) (string, bool) {
	if limit > len(ls) {
		limit = len(ls)
	}
	if limit <= 0 {
		return "", false
	}

	// next[i] is where lowered[i] matches at or after pos, or -1 once exhausted.
	// Refreshed only for values whose match pos has passed, and pos only moves
	// forward, so this part of the scan is linear in len(ls) per value.
	next := make([]int, len(lowered))
	for i, lv := range lowered {
		next[i] = strings.Index(ls, lv)
	}

	var b strings.Builder
	pos, hit := 0, false
	for {
		at, which := earliestMatch(next, lowered)
		if at < 0 || at >= limit {
			// Nothing left to redact, or what is left starts at or past the bound
			// and so cannot reach the output at all.
			break
		}
		end := endOfRun(ls, at, at+len(lowered[which]), lowered)

		// One marker per redacted run, not per match. Adjacent ranges are the
		// common case for a short secret — a megabyte of the acting user's single
		// initial would otherwise emit a megabyte of markers, ten times the input,
		// to produce a kilobyte of error.
		//
		// hit doubles as "something is already emitted", which is what makes
		// at == pos mean "this range touches the last one" rather than "this is
		// the first range and it starts at zero".
		if at > pos || !hit {
			b.WriteString(ls[pos:at])
			b.WriteString(redactedMarker)
		}
		pos, hit = end, true
		if pos >= limit {
			// The run reaches the bound, so nothing after it may be shown. It was
			// removed whole rather than cut, which is the point of matching over
			// the whole of ls while emitting only up to limit.
			return b.String(), true
		}
		advanceMatches(ls, pos, lowered, next)
	}
	if !hit {
		return "", false
	}
	b.WriteString(ls[pos:limit])
	return b.String(), true
}

// earliestMatch picks the next range to redact: the leftmost pending match, and
// on a tie the longest, so a value contained in another does not split it.
// It returns -1, -1 once every value is exhausted.
func earliestMatch(next []int, lowered []string) (at, which int) {
	at, which = -1, -1
	for i, n := range next {
		if n < 0 {
			continue
		}
		if at < 0 || n < at || (n == at && len(lowered[i]) > len(lowered[which])) {
			at, which = n, i
		}
	}
	return at, which
}

// endOfRun extends a redacted range while ANY occurrence starts inside it, and
// returns where the run ends.
//
// Choosing the earliest match covers a contained occurrence and an equal start,
// but not one that straddles the far edge: the caller's next[] only looks forward
// from the cursor, so a straddler would be neither redacted nor found again and
// its tail would be copied straight out.
//
// Scanned position by position rather than off next[], which holds only the FIRST
// occurrence of each value at or after the cursor. A second occurrence of the same
// value can start inside the range and straddle it, and next[] cannot see it.
// Measured with an earlier version:
// redact("https://cb.test/u/alice@example.test/alice@example.test",
// ..... "alice@example.test", "https://cb.test/u/alice@example.test/a")
// returned "[redacted]lice@example.test", keeping 17 of the address's 18 bytes.
//
// The condition re-reads end each step, so an extension made at k is picked up by
// the same walk — which is what makes one pass enough.
func endOfRun(ls string, at, end int, lowered []string) int {
	for k := at + 1; k < end; k++ {
		for _, lv := range lowered {
			if k+len(lv) > end && strings.HasPrefix(ls[k:], lv) {
				end = k + len(lv)
			}
		}
	}
	return end
}

// advanceMatches moves every pending match the cursor has passed forward to its
// next occurrence, or retires it. Only the overtaken ones are re-searched, and
// pos only moves forward, which is what keeps the outer scan linear per value.
func advanceMatches(ls string, pos int, lowered []string, next []int) {
	for i, lv := range lowered {
		if next[i] < 0 || next[i] >= pos {
			continue
		}
		if j := strings.Index(ls[pos:], lv); j < 0 {
			next[i] = -1
		} else {
			next[i] = pos + j
		}
	}
}
