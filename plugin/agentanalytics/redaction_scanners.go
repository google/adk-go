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

package agentanalytics

import (
	"bytes"
	"encoding/base64"
	"net/url"
	"strings"
	"unicode"
	"unicode/utf8"

	"google.golang.org/adk/v2/session"
)

// The linear-time text scanners: quoted-string scanning shared by them, and
// the Authorization and API-key header, bearer, basic, quoted JSON and
// key=value or key: value passes of scrubTextPatterns.

func scrubErrorTokens(msg string) string {
	return scrubErrorTokensTraced(msg, nil)
}

// scrubErrorTokensTraced is scrubErrorTokens, recording in tr, when it is not
// nil, the runs of the result copied from msg (see passTrace).
func scrubErrorTokensTraced(msg string, tr *passTrace) string {
	return scrubErrorTokensTracedWith(msg, redactURIComponent, tr)
}

// scrubErrorTokensTracedWith is scrubErrorTokensTraced with scrub, which
// returns what redactURIComponent does, in its place (see
// scrubMemo.componentScrub).
func scrubErrorTokensTracedWith(msg string, scrub uriComponentScrub, tr *passTrace) string {
	c := startPasses(msg, tr)
	c.run(func(s string, tr *passTrace) string {
		return scrubURITokens(s, scrub, scannerURIKeys, tr)
	})
	c.run(scrubTextPatterns)
	return c.end(tr)
}

// scrubTextPatterns redacts Authorization and API-key headers, bearer and
// basic credentials, and the values of sensitive key-value pairs in s,
// recording in tr, when it is not nil, the runs of the result copied from s.
func scrubTextPatterns(s string, tr *passTrace) string {
	c := startPasses(s, tr)
	if !mayMatchTextPatterns(s) {
		return c.end(tr)
	}
	hasColonOrEq := strings.ContainsAny(s, ":=")
	if hasColonOrEq && strings.Contains(s, ":") {
		c.run(redactAuthHeaders)
	}
	if strings.ContainsAny(c.s, "bB") {
		c.run(redactBearerTokens)
		c.run(redactBasicCredentials)
	}
	if hasColonOrEq && strings.Contains(c.s, `"`) && strings.Contains(c.s, ":") {
		c.run(redactJSONKeyValues)
		if c.s == redactedSensitiveTextPlaceholder {
			return c.end(tr)
		}
	}
	if strings.ContainsAny(c.s, "=:") {
		c.run(redactTextKeyValues)
	}
	return c.end(tr)
}

// openerScanBudgetMultiplier and minOpenerScanBudget bound the bytes that the
// lookahead scans of one quoteScanState read past the value they classify
// (the runs of glued values that extendGluedValueEnd and nestedPairEndWithState
// follow, and the pairs that innerContinuesIntoGluedToken reads) over a text s
// to max(openerScanBudgetMultiplier*len(s), minOpenerScanBudget), so that
// their total work stays linear in len(s). Text that is not built to defeat
// them reads each byte a few times at most; text that runs the scans out of
// budget fails closed (see consumeOpenerBudget).
const (
	minOpenerScanBudget        = 8192
	openerScanBudgetMultiplier = 8
)

type quoteScanState struct {
	unclosedDoubleFrom    int
	unclosedDoubleUntil   int
	unclosedSingleFrom    int
	unclosedSingleUntil   int
	failedEscDoubleFrom   int
	failedEscDoublePos    int
	failedEscSingleFrom   int
	failedEscSinglePos    int
	closedDoubleFrom      int
	closedDoubleUntil     int
	closedSingleFrom      int
	closedSingleUntil     int
	closedEscDoubleFrom   int
	closedEscDoubleUntil  int
	closedEscSingleFrom   int
	closedEscSingleUntil  int
	openerScanBudget      int
	openerBudgetExhausted bool
}

// newQuoteScanState returns the scan state of a text of textLen bytes, with
// empty caches and the scan budget of that text (see
// openerScanBudgetMultiplier).
func newQuoteScanState(textLen int) quoteScanState {
	return quoteScanState{
		unclosedDoubleUntil:  -1,
		unclosedSingleUntil:  -1,
		failedEscDoublePos:   -1,
		failedEscSinglePos:   -1,
		closedDoubleUntil:    -1,
		closedSingleUntil:    -1,
		closedEscDoubleUntil: -1,
		closedEscSingleUntil: -1,
		openerScanBudget:     max(openerScanBudgetMultiplier*textLen, minOpenerScanBudget),
	}
}

// consumeOpenerBudget takes steps bytes of lookahead scan work from the budget
// of st and reports whether the scan may go on. Once the budget runs out, st
// stays exhausted and the scans fail closed: the value being classified runs
// to the end of s, and redactJSONKeyValues and redactTextKeyValues return
// redactedSensitiveTextPlaceholder for the whole text, so that running out of
// budget never keeps a value. A nil st has no budget.
func (st *quoteScanState) consumeOpenerBudget(steps int) bool {
	if st == nil {
		return true
	}
	if st.openerBudgetExhausted {
		return false
	}
	if steps >= st.openerScanBudget {
		st.openerScanBudget = 0
		st.openerBudgetExhausted = true
		return false
	}
	st.openerScanBudget -= steps
	return true
}

// isRegexpSpace reports whether c is one of the bytes that \s matches in a Go
// regular expression. It is used where a scan must agree with such an
// expression: the value stops below and the gap before a following string
// that they end (see closesOpenQuoteAndOpensNext), the spaces around the ":"
// of a JSON key and after Bearer or Basic, and the end of a redacted URI.
// Container, member and opener checks use isDelimSpace instead.
func isRegexpSpace(c byte) bool {
	return c == '\t' || c == '\n' || c == '\f' || c == '\r' || c == ' '
}

// isDelimSpace reports whether c is delimiter whitespace: a space or any other
// ASCII control byte, as a vertical tab, so that such a byte separates the
// tokens of a key, a value or a container as a space does.
func isDelimSpace(c byte) bool {
	return c <= 0x20 || c == 0x7f
}

func isTextValueStop(c byte) bool {
	return isRegexpSpace(c) || c == ',' || c == ';' || c == '&'
}

func isJSONValueStop(c byte) bool {
	return isRegexpSpace(c) || c == ',' || c == '}'
}

func isBearerBareStop(c byte) bool {
	return isRegexpSpace(c) || c == ',' || c == ';' || c == ')' || c == '"' || c == '\'' || c == ']'
}

func isBearerBracketTailStop(c byte) bool {
	return isRegexpSpace(c) || c == ',' || c == ';' || c == ')' || c == '"' || c == '\''
}

func isTempKeyBodyByte(c byte) bool {
	switch c {
	case '=', ':', '"', '\'', ' ', '\t', '\r', '\n', '\f', ',', '}':
		return false
	default:
		return true
	}
}

func trimInlineSpaces(s string) string {
	return strings.TrimFunc(s, isInlineSpaceRune)
}

func isInlineSpaceRune(r rune) bool {
	return unicode.IsSpace(r) || unicode.In(r, unicode.Z)
}

// trimmedSpan returns the span of s[from:to] that trimInlineSpaces keeps.
func trimmedSpan(s string, from, to int) (int, int) {
	left := strings.TrimLeftFunc(s[from:to], isInlineSpaceRune)
	from = to - len(left)
	return from, from + len(strings.TrimRightFunc(left, isInlineSpaceRune))
}

func skipInlineSpacesFwd(s string, pos, limit int) int {
	for pos < limit {
		r, size := utf8.DecodeRuneInString(s[pos:limit])
		if size == 0 || (!unicode.IsSpace(r) && !unicode.In(r, unicode.Z)) {
			break
		}
		pos += size
	}
	return pos
}

// recordFailedQuoteSpan updates [from, until) with a failed quote scan from
// start to failEnd so forward scans inside [from, until) short-circuit in O(1)
// while backward re-scans starting before from are never falsely rejected.
func recordFailedQuoteSpan(from, until *int, start, failEnd int) {
	if *until <= 0 || start > *until {
		*from, *until = start, failEnd
		return
	}
	if start >= *from && failEnd > *until {
		*until = failEnd
	}
}

// recordClosedQuoteSpan updates [from, until) with a closed quote scan from
// start to end so scans starting inside [from, until-len(quote)) return end in
// O(1) without rescanning the interior.
func recordClosedQuoteSpan(from, until *int, start, end int) {
	if *until <= 0 || end > *until {
		*from, *until = start, end
		return
	}
	if end == *until && start < *from {
		*from = start
	}
}

func scanSimpleQuoted(s string, start int, q byte, st *quoteScanState) (int, bool) {
	if start >= len(s) || s[start] != q {
		return 0, false
	}
	if st != nil {
		if q == '"' {
			if start >= st.unclosedDoubleFrom && start < st.unclosedDoubleUntil {
				return 0, false
			}
			if start >= st.closedDoubleFrom && start+1 < st.closedDoubleUntil {
				return st.closedDoubleUntil, true
			}
		} else {
			if start >= st.unclosedSingleFrom && start < st.unclosedSingleUntil {
				return 0, false
			}
			if start >= st.closedSingleFrom && start+1 < st.closedSingleUntil {
				return st.closedSingleUntil, true
			}
		}
	}
	for i := start + 1; i < len(s); i++ {
		if st != nil {
			if q == '"' && st.closedDoubleUntil > 0 && i == st.closedDoubleFrom+1 {
				st.closedDoubleFrom = start
				return st.closedDoubleUntil, true
			}
			if q == '\'' && st.closedSingleUntil > 0 && i == st.closedSingleFrom+1 {
				st.closedSingleFrom = start
				return st.closedSingleUntil, true
			}
		}
		c := s[i]
		if c == '\\' {
			if i+1 >= len(s) {
				if st != nil {
					if q == '"' {
						recordFailedQuoteSpan(&st.unclosedDoubleFrom, &st.unclosedDoubleUntil, start, len(s))
					} else {
						recordFailedQuoteSpan(&st.unclosedSingleFrom, &st.unclosedSingleUntil, start, len(s))
					}
				}
				return 0, false
			}
			i++
			continue
		}
		if c == q {
			if st != nil {
				if q == '"' {
					recordClosedQuoteSpan(&st.closedDoubleFrom, &st.closedDoubleUntil, start, i+1)
				} else {
					recordClosedQuoteSpan(&st.closedSingleFrom, &st.closedSingleUntil, start, i+1)
				}
			}
			return i + 1, true
		}
	}
	if st != nil {
		if q == '"' {
			recordFailedQuoteSpan(&st.unclosedDoubleFrom, &st.unclosedDoubleUntil, start, len(s))
		} else {
			recordFailedQuoteSpan(&st.unclosedSingleFrom, &st.unclosedSingleUntil, start, len(s))
		}
	}
	return 0, false
}

func scanEscapedQuoted(s string, start int, q byte, st *quoteScanState) (int, bool) {
	if start+1 >= len(s) || s[start] != '\\' || s[start+1] != q {
		return 0, false
	}
	if st != nil {
		if q == '"' {
			if start >= st.failedEscDoubleFrom && start+2 <= st.failedEscDoublePos {
				return 0, false
			}
			if start >= st.closedEscDoubleFrom && start+2 < st.closedEscDoubleUntil {
				return st.closedEscDoubleUntil, true
			}
		} else {
			if start >= st.failedEscSingleFrom && start+2 <= st.failedEscSinglePos {
				return 0, false
			}
			if start >= st.closedEscSingleFrom && start+2 < st.closedEscSingleUntil {
				return st.closedEscSingleUntil, true
			}
		}
	}
	recordFail := func(failRunStart int) {
		if st == nil {
			return
		}
		if q == '"' {
			recordFailedQuoteSpan(&st.failedEscDoubleFrom, &st.failedEscDoublePos, start, failRunStart)
		} else {
			recordFailedQuoteSpan(&st.failedEscSingleFrom, &st.failedEscSinglePos, start, failRunStart)
		}
	}
	recordClose := func(end int) {
		if st == nil {
			return
		}
		if q == '"' {
			recordClosedQuoteSpan(&st.closedEscDoubleFrom, &st.closedEscDoubleUntil, start, end)
		} else {
			recordClosedQuoteSpan(&st.closedEscSingleFrom, &st.closedEscSingleUntil, start, end)
		}
	}
	i := start + 2
	for i < len(s) {
		if st != nil {
			if q == '"' {
				if st.closedEscDoubleUntil > 0 && i == st.closedEscDoubleFrom+2 {
					st.closedEscDoubleFrom = start
					return st.closedEscDoubleUntil, true
				}
				if st.failedEscDoublePos > 0 && i >= st.failedEscDoubleFrom+2 && i <= st.failedEscDoublePos {
					recordFail(st.failedEscDoublePos)
					return 0, false
				}
			} else {
				if st.closedEscSingleUntil > 0 && i == st.closedEscSingleFrom+2 {
					st.closedEscSingleFrom = start
					return st.closedEscSingleUntil, true
				}
				if st.failedEscSinglePos > 0 && i >= st.failedEscSingleFrom+2 && i <= st.failedEscSinglePos {
					recordFail(st.failedEscSinglePos)
					return 0, false
				}
			}
		}
		runStart := i
		m := 0
		for i < len(s) && s[i] == '\\' {
			m++
			i++
		}
		if i == len(s) {
			recordFail(len(s))
			return 0, false
		}
		c := s[i]
		i++
		if c == q {
			switch m & 3 {
			case 1:
				recordClose(i)
				return i, true
			case 3:
				continue
			default:
				recordFail(runStart)
				return 0, false
			}
		}
		if m&3 == 3 {
			recordFail(runStart)
			return 0, false
		}
	}
	recordFail(len(s))
	return 0, false
}

func scanQuotedAt(s string, start int, st *quoteScanState) (int, bool) {
	if start >= len(s) {
		return 0, false
	}
	switch s[start] {
	case '"', '\'':
		return scanSimpleQuoted(s, start, s[start], st)
	case '\\':
		if start+1 < len(s) && (s[start+1] == '"' || s[start+1] == '\'') {
			return scanEscapedQuoted(s, start, s[start+1], st)
		}
	}
	return 0, false
}

func scanValueEnd(s string, valStart int, stop func(byte) bool, st *quoteScanState) int {
	if end, ok := scanQuotedAt(s, valStart, st); ok {
		return end
	}
	end := valStart + 1
	for end < len(s) && !stop(s[end]) {
		end++
	}
	return end
}

var authHeaderNames = [...]string{
	"proxy-authorization",
	"proxy_authorization",
	"authorization",
	"x-api-key",
	"x-api_key",
	"x_api-key",
	"x_api_key",
	"api-key",
	"api_key",
}

func matchSuffixFoldBack(s string, floor, hdrEnd int, target string) (int, bool) {
	si := hdrEnd
	for ti := len(target) - 1; ti >= 0; ti-- {
		if si <= floor {
			return 0, false
		}
		c := s[si-1]
		if c < utf8.RuneSelf {
			if 'A' <= c && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c != target[ti] {
				return 0, false
			}
			si--
			continue
		}
		r, size := utf8.DecodeLastRuneInString(s[floor:si])
		var fc byte
		switch r {
		case '\u017f':
			fc = 's'
		case '\u212a':
			fc = 'k'
		default:
			return 0, false
		}
		if fc != target[ti] {
			return 0, false
		}
		si -= size
	}
	if si > 0 && isWordByte(s[si-1]) {
		return 0, false
	}
	return si, true
}

func matchAuthHeaderAtEnd(s string, last, hdrEnd int) int {
	for _, name := range authHeaderNames {
		if start, ok := matchSuffixFoldBack(s, last, hdrEnd, name); ok {
			return start
		}
	}
	return -1
}

func redactAuthHeaders(s string, tr *passTrace) string {
	w := spanWriter{s: s, tr: tr}
	last, pos := 0, 0
	// deferUntil is the end of a line that ends in a nested key (see
	// nestedKeyBeforeNextLine), whose later headers are part of the
	// redacted value or of the pair that redactTextKeyValues redacts, so
	// that the line is read once however many headers it holds.
	deferUntil := -1
	for pos < len(s) {
		rel := strings.IndexByte(s[pos:], ':')
		if rel < 0 {
			break
		}
		colonPos := pos + rel
		hdrEnd := colonPos
		for hdrEnd > last && (s[hdrEnd-1] == ' ' || s[hdrEnd-1] == '\t') {
			hdrEnd--
		}
		hdrStart := matchAuthHeaderAtEnd(s, last, hdrEnd)
		if hdrStart < 0 || colonPos < deferUntil {
			pos = colonPos + 1
			continue
		}
		valEnd := lineEnd(s, colonPos+1)
		valStart := colonPos + 1
		for valStart < valEnd && (s[valStart] == ' ' || s[valStart] == '\t') {
			valStart++
		}
		if valStart == valEnd && valEnd < len(s) && valueStartsAt(s, valEnd, nil) {
			// The value starts on a following line, as in "api_key:\n  SECRET",
			// and is redacted as the value of a key-value pair (see
			// redactTextKeyValues, which keeps the line break).
			pos = colonPos + 1
			continue
		}
		valEnd = max(valEnd, authHeaderValueEnd(s, valStart))
		if keyStart, ok := nestedKeyBeforeNextLine(s, valStart, valEnd); ok {
			// The header line ends in a nested key whose value starts on
			// the next line, as in api_key: x "token":\n  SECRET. The
			// header value is redacted up to that key, and the pair it
			// starts is left to the key-value passes, which redact it
			// together with that value and keep its line break (see
			// headerLineBreak), so that the header value stays the
			// placeholder on a later pass; later headers on the line are
			// part of the redacted value or of that pair.
			deferUntil = valEnd
			pos = valEnd
			vEnd := keyStart
			for vEnd > valStart && (s[vEnd-1] == ' ' || s[vEnd-1] == '\t') {
				vEnd--
			}
			if inner, _ := unquoteValue(s[valStart:vEnd]); vEnd == valStart || isRedactedMarker(inner) {
				continue
			}
			w.copy(last, hdrEnd)
			w.lit(": " + redactedValue)
			if vEnd == keyStart {
				// The key was joined to the redacted text, as the token of
				// SECRET/token:, and is now written apart from it.
				w.lit(" ")
			}
			last = vEnd
			continue
		}
		if inner, _ := unquoteValue(s[valStart:valEnd]); isRedactedMarker(inner) {
			pos = valEnd
			continue
		}
		w.copy(last, hdrEnd)
		w.lit(": " + redactedValue)
		last = valEnd
		pos = valEnd
	}
	return w.done(last)
}

// nestedKeyBeforeNextLine reports whether the header value
// s[valStart:valEnd], which ends at a line break, ends in a name that holds a
// sensitive key, possibly quoted and padded, and a delimiter, with a value
// starting on a following line (see nestedKeyName and valueStartsAt), and
// returns where that key starts: at the token of SECRET/token:,
// v1\vSECRET/token: or sk_v1_token:, or at the opening quote of
// " token ":. Only a fixed sensitive key at the end of the name counts (see
// fixedKeySuffixStart), and a quoted name only when it is such a key as a
// whole, possibly after a "user:" or "app:" prefix: the rest of a compound
// or "temp:" name, as the v1 of temp:v1: or "dev-v1-password":,
// may be part of the header value, so that the whole line is redacted then.
func nestedKeyBeforeNextLine(s string, valStart, valEnd int) (int, bool) {
	end := valEnd
	for end > valStart && (s[end-1] == ' ' || s[end-1] == '\t') {
		end--
	}
	if valEnd == len(s) || end <= valStart || (s[end-1] != ':' && s[end-1] != '=') || !valueStartsAt(s, valEnd, nil) {
		return 0, false
	}
	nameEnd := end - 1
	for nameEnd > valStart && (s[nameEnd-1] == ' ' || s[nameEnd-1] == '\t') {
		nameEnd--
	}
	if nameEnd > valStart && isQuoteByte(s[nameEnd-1]) {
		// A quoted name runs back to the quote that opens it, over any
		// padding, as in " token ".
		if open := strings.LastIndexByte(s[valStart:nameEnd-1], s[nameEnd-1]); open >= 0 {
			nameStart := valStart + open
			innerStart, innerEnd := nameStart+1, nameEnd-1
			if nameStart > valStart && s[nameStart-1] == '\\' {
				nameStart--
			}
			if innerEnd > innerStart && s[innerEnd-1] == '\\' {
				innerEnd--
			}
			if kind, _ := nestedKeyName(trimInlineSpaces(s[innerStart:innerEnd]), innerStart, false); kind == nestedKey {
				return nameStart, true
			}
			return 0, false
		}
	}
	nameStart := nameEnd
	for nameStart > valStart && s[nameStart-1] != ' ' && s[nameStart-1] != '\t' {
		nameStart--
	}
	kind, pair := nestedKeyName(s[nameStart:nameEnd], nameStart, false)
	if kind == notNestedKey {
		return 0, false
	}
	// A name that ends in a stray closing quote, as the password" of
	// password":, names the key before it (see nestedKeyName).
	keyEnd := nameEnd
	if keyEnd-pair >= 2 && isQuoteByte(s[keyEnd-1]) && s[keyEnd-2] != '\\' {
		keyEnd--
	}
	j := fixedKeySuffixStart(s[pair:keyEnd])
	if j < 0 {
		return 0, false
	}
	return pair + j, true
}

// isQuoteByte reports whether c is a double or single quote.
func isQuoteByte(c byte) bool {
	return c == '"' || c == '\''
}

// lineEnd returns the index of the first "\r" or "\n" at or after i in s, or
// len(s) if there is none.
func lineEnd(s string, i int) int {
	if n := strings.IndexAny(s[i:], "\r\n"); n >= 0 {
		return i + n
	}
	return len(s)
}

// authHeaderValueEnd returns where the header value that starts at s[start]
// ends: at the end of its line, or, when a quoted string opened in it runs on
// past a backslash-escaped line break, as in
// `api_key: "a\<newline>b SECRET"`, at the end of the line on which that
// string closes or the text ends. A quote left open at an unescaped line
// break ends the value at that line break.
func authHeaderValueEnd(s string, start int) int {
	var q byte
	for i := start; i < len(s); i++ {
		switch c := s[i]; {
		case c == '\r' || c == '\n':
			return i
		case c == '\\':
			if i+1 < len(s) && (q != 0 || (s[i+1] != '\r' && s[i+1] != '\n')) {
				if i+2 < len(s) && s[i+1] == '\r' && s[i+2] == '\n' {
					i++
				}
				i++
			}
		case q == 0 && isQuoteByte(c):
			q = c
		case c == q:
			q = 0
		}
	}
	return len(s)
}

func matchWordFoldASCII(s string, pos int, word string) bool {
	if pos+len(word) > len(s) {
		return false
	}
	if pos > 0 && isWordByte(s[pos-1]) {
		return false
	}
	return strings.EqualFold(s[pos:pos+len(word)], word)
}

func redactBearerTokens(s string, tr *passTrace) string {
	qState := newQuoteScanState(len(s))
	w := spanWriter{s: s, tr: tr}
	last, pos := 0, 0
	for pos+7 <= len(s) {
		rel := strings.IndexAny(s[pos:], "bB")
		if rel < 0 {
			break
		}
		i := pos + rel
		if !matchWordFoldASCII(s, i, "bearer") {
			pos = i + 1
			continue
		}
		afterWord := i + len("bearer")
		valStart := afterWord
		for valStart < len(s) && isRegexpSpace(s[valStart]) {
			valStart++
		}
		if valStart == afterWord || valStart >= len(s) {
			pos = afterWord
			continue
		}
		valEnd := scanBearerValueEnd(s, valStart, &qState)
		if valEnd <= valStart {
			pos = afterWord
			continue
		}
		if rest := s[valStart:]; len(rest) >= 2 && strings.EqualFold(rest[:2], "of") {
			if r, _ := utf8.DecodeRuneInString(rest[2:]); len(rest) == 2 || unicode.IsSpace(r) {
				pos = valEnd
				continue
			}
		}
		if inner, _ := unquoteValue(s[valStart:valEnd]); isRedactedValue(inner) {
			pos = valEnd
			continue
		}
		w.copy(last, afterWord)
		w.lit(" " + redactedValue)
		last = valEnd
		pos = valEnd
	}
	return w.done(last)
}

func scanBearerValueEnd(s string, valStart int, st *quoteScanState) int {
	switch s[valStart] {
	case '"', '\'':
		if end, ok := scanSimpleQuoted(s, valStart, s[valStart], st); ok {
			return end
		}
		return 0
	case '[':
		j := valStart + 1
		for j < len(s) && s[j] != ']' && s[j] != '\r' && s[j] != '\n' && !isRegexpSpace(s[j]) {
			j++
		}
		if j > valStart+1 && j < len(s) && s[j] == ']' {
			j++
			for j < len(s) && !isBearerBracketTailStop(s[j]) {
				j++
			}
			return j
		}
	}
	end := valStart
	for end < len(s) && !isBearerBareStop(s[end]) {
		end++
	}
	return end
}

// maxBasicCredentialBytes is adk-python's _MAX_BASIC_AUTH_TOKEN_CHARS.
const maxBasicCredentialBytes = 16 * 1024

func matchBasicPrefix(s string, i int) (after int, ok bool) {
	if i > 0 && isWordByte(s[i-1]) {
		return 0, false
	}
	if i+5 <= len(s) && strings.EqualFold(s[i:i+5], "basic") {
		return i + 5, true
	}
	// Handle Unicode U+017F (ſ) case fold matching (?i)basic.
	if i+6 <= len(s) && (s[i] == 'b' || s[i] == 'B') && (s[i+1] == 'a' || s[i+1] == 'A') &&
		s[i+2] == '\xc5' && s[i+3] == '\xbf' &&
		(s[i+4] == 'i' || s[i+4] == 'I') && (s[i+5] == 'c' || s[i+5] == 'C') {
		return i + 6, true
	}
	return 0, false
}

func isBase64BodyByte(c byte) bool {
	return ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z') || ('0' <= c && c <= '9') || c == '+' || c == '/'
}

func redactBasicCredentials(s string, tr *passTrace) string {
	w := spanWriter{s: s, tr: tr}
	last, pos := 0, 0
	for pos+6 <= len(s) {
		rel := strings.IndexAny(s[pos:], "bB")
		if rel < 0 {
			break
		}
		i := pos + rel
		afterWord, ok := matchBasicPrefix(s, i)
		if !ok {
			pos = i + 1
			continue
		}
		valStart := afterWord
		for valStart < len(s) && isRegexpSpace(s[valStart]) {
			valStart++
		}
		if valStart == afterWord || valStart >= len(s) {
			pos = afterWord
			continue
		}
		end := valStart
		for end < len(s) && isBase64BodyByte(s[end]) {
			end++
		}
		if end == valStart {
			pos = afterWord
			continue
		}
		for pad := 0; pad < 2 && end < len(s) && s[end] == '='; pad++ {
			end++
		}
		if end < len(s) && isBase64Byte(s[end]) {
			pos = end
			continue
		}
		encoded := s[valStart:end]
		if len(encoded) <= maxBasicCredentialBytes {
			decoded, err := base64.StdEncoding.DecodeString(encoded)
			if err != nil || (!strings.Contains(encoded, "=") && !bytes.Contains(decoded, []byte(":"))) {
				pos = end
				continue
			}
		}
		w.copy(last, afterWord)
		w.lit(" " + redactedValue)
		last = end
		pos = end
	}
	return w.done(last)
}

func isBase64Byte(c byte) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || '0' <= c && c <= '9' || c == '+' || c == '/' || c == '='
}

func isUnescapedQuote(s string, floor, quotePos int) bool {
	bs := 0
	for p := quotePos; p > floor && s[p-1] == '\\'; p-- {
		bs++
	}
	return bs%2 == 0
}

func redactJSONKeyValues(s string, tr *passTrace) string {
	qState := newQuoteScanState(len(s))
	var brackets openBrackets
	var quotes openQuotes
	w := spanWriter{s: s, tr: tr}
	last, pos := 0, 0
	for pos < len(s) {
		rel := strings.IndexByte(s[pos:], ':')
		if rel < 0 {
			break
		}
		colonPos := pos + rel
		p := colonPos
		for p > last && isRegexpSpace(s[p-1]) {
			p--
		}
		if p <= last || s[p-1] != '"' || !isUnescapedQuote(s, last, p-1) {
			pos = colonPos + 1
			continue
		}
		closeQuote := p - 1
		openQuote := -1
		for q := closeQuote - 1; q >= last; q-- {
			c := s[q]
			if c == '"' {
				if isUnescapedQuote(s, last, q) {
					openQuote = q
				}
				break
			}
			if c == '\'' || c == ',' || c == '}' {
				break
			}
		}
		if openQuote < 0 {
			pos = colonPos + 1
			continue
		}
		keyStart := skipInlineSpacesFwd(s, openQuote+1, closeQuote)
		keyEnd := skipInlineSpacesBack(s, keyStart, closeQuote)
		if !isValidJSONPatternKey(s[keyStart:keyEnd]) {
			pos = colonPos + 1
			continue
		}
		valStart := colonPos + 1
		for valStart < len(s) && isRegexpSpace(s[valStart]) {
			valStart++
		}
		if valStart >= len(s) || isJSONValueStop(s[valStart]) {
			pos = colonPos + 1
			continue
		}
		end := scanValueEnd(s, valStart, isJSONValueStop, &qState)
		open := brackets.before(s, openQuote)
		openQ := quotes.before(s, colonPos)
		kind, pair := nestsSensitiveKey(s, valStart, end, startsLine(s, colonPos+1, valStart), openQ, &qState)
		if qState.openerBudgetExhausted {
			return w.failClosed()
		}
		switch kind {
		case nestedKey:
			pos = valStart
			continue
		case valueEndingInKeyName, gluedAfterValue:
			if kind == gluedAfterValue {
				end = max(end, extendGluedValueEnd(s, pair, open, openQ, &qState))
			} else {
				end = max(end, nestedPairEndWithState(s, pair, open, openQ, &qState))
			}
			if qState.openerBudgetExhausted {
				return w.failClosed()
			}
			brackets.skip(end)
			writeJSONPair(&w, last, openQuote, keyStart, keyEnd, closeQuote, colonPos, valStart)
			last = end
			pos = end
			continue
		}
		origEnd := end
		_, origQuote := unquoteValue(s[valStart:origEnd])
		if origQuote != "" {
			if hasGluedSuffixAfterQuotedValue(s, valStart, end, openQ, &qState) {
				end = max(end, extendGluedValueEnd(s, end, open, openQ, &qState))
			}
			if qState.openerBudgetExhausted {
				return w.failClosed()
			}
		} else {
			end = sensitiveValueEnd(s, valStart, end)
		}
		brackets.skip(end)
		if end == origEnd || origQuote == "" {
			if inner, _ := unquoteValue(s[valStart:end]); isRedactedValue(inner) {
				pos = end
				continue
			}
		}
		writeJSONPair(&w, last, openQuote, keyStart, keyEnd, closeQuote, colonPos, valStart)
		last = end
		pos = end
	}
	if qState.openerBudgetExhausted {
		return w.failClosed()
	}
	return w.done(last)
}

func isCompoundKeyRuneBack(s string, floor, pos int) int {
	if pos <= floor {
		return 0
	}
	c := s[pos-1]
	if c < utf8.RuneSelf {
		if isCompoundKeyByte(c) {
			return 1
		}
		return 0
	}
	r, size := utf8.DecodeLastRuneInString(s[floor:pos])
	if r == '\u0130' || r == '\u017f' || r == '\u212a' {
		return size
	}
	return 0
}

func findTextKeyStart(s string, last, keyEnd int) int {
	// Shape A: temp:<non-delimiter/non-space> key preceded by a word boundary.
	tk := keyEnd
	for tk > last && isTempKeyBodyByte(s[tk-1]) {
		tk--
	}
	if tk < keyEnd && tk-len(session.KeyPrefixTemp) >= last &&
		strings.EqualFold(s[tk-len(session.KeyPrefixTemp):tk], session.KeyPrefixTemp) {
		tempStart := tk - len(session.KeyPrefixTemp)
		if tempStart == 0 || !isWordByte(s[tempStart-1]) {
			return tempStart
		}
	}
	// Shape B: compound or fixed sensitive key made of compound key runes.
	fullStart := keyEnd
	for fullStart > last {
		step := isCompoundKeyRuneBack(s, last, fullStart)
		if step == 0 {
			break
		}
		fullStart -= step
	}
	if fullStart >= keyEnd {
		return -1
	}
	return sensitiveKeyStart(s, fullStart, keyEnd)
}

func redactTextKeyValues(s string, tr *passTrace) string {
	qState := newQuoteScanState(len(s))
	var brackets openBrackets
	var quotes openQuotes
	w := spanWriter{s: s, tr: tr}
	last, pos := 0, 0
	for pos < len(s) {
		rel := strings.IndexAny(s[pos:], "=:")
		if rel < 0 {
			break
		}
		delimPos := pos + rel
		valStart := delimPos + 1
		for valStart < len(s) && isDelimSpace(s[valStart]) {
			valStart++
		}
		if (valStart >= len(s) || isTextValueStop(s[valStart])) && valStart > delimPos+1 && !isTextValueStop(s[valStart-1]) {
			valStart--
		}
		if valStart >= len(s) || isTextValueStop(s[valStart]) {
			pos = delimPos + 1
			continue
		}
		p := delimPos
		for p > last && isDelimSpace(s[p-1]) {
			p--
		}
		quoteStart, keyEnd := p, p
		if p-2 >= last && s[p-2] == '\\' && (s[p-1] == '"' || s[p-1] == '\'') && !(p-3 >= last && s[p-3] == '\\') {
			quoteStart, keyEnd = p-2, p-2
		} else if p-1 >= last && (s[p-1] == '"' || s[p-1] == '\'') {
			quoteStart, keyEnd = p-1, p-1
		}
		keyStart := findTextKeyStart(s, last, keyEnd)
		if keyStart < 0 && quoteStart == p-2 {
			quoteStart, keyEnd = p-1, p-1
			keyStart = findTextKeyStart(s, last, keyEnd)
		}
		// rewritten holds the spans of a key whose padding or stray quote
		// the redacted pair drops.
		var rewritten keyCopy
		if keyStart < 0 && quoteStart < p {
			keyStart, rewritten = matchPaddedQuotedTextKey(s, last, p)
		}
		if keyStart < 0 && p-2 > last && isQuoteByte(s[p-1]) && isQuoteByte(s[p-2]) {
			// A stray quote after the closing quote of a key, as in
			// "client_secret"':SECRET or " password"": SECRET, which is
			// dropped from the redacted pair.
			keyStart, rewritten = matchPaddedQuotedTextKey(s, last, p-1)
			if keyStart < 0 && s[p-3] != '\\' {
				if ks := findTextKeyStart(s, last, p-2); ks >= 0 {
					keyStart, rewritten = ks, keyCopy{spans: [3][2]int{{ks, p - 1}}, n: 1}
				}
			}
		}
		if keyStart < 0 || (quoteStart == p && isPathErrorColon(s, delimPos) && isPathPrefixedDottedKey(s, last, keyStart, keyEnd)) {
			pos = delimPos + 1
			continue
		}
		end := scanValueEnd(s, valStart, isTextValueStop, &qState)
		open := brackets.before(s, keyStart)
		openQ := quotes.before(s, delimPos)
		kind, pair := nestsSensitiveKey(s, valStart, end, startsLine(s, delimPos+1, valStart), openQ, &qState)
		if qState.openerBudgetExhausted {
			return w.failClosed()
		}
		switch kind {
		case nestedKey:
			pos = valStart
			continue
		case valueEndingInKeyName, gluedAfterValue:
			_, quote := unquoteValue(s[valStart:end])
			if kind == gluedAfterValue {
				end = max(end, extendGluedValueEnd(s, pair, open, openQ, &qState))
			} else {
				end = max(end, nestedPairEndWithState(s, pair, open, openQ, &qState))
			}
			if qState.openerBudgetExhausted {
				return w.failClosed()
			}
			brackets.skip(end)
			writeTextPairKey(&w, last, keyStart, p, rewritten)
			w.copy(delimPos, keyLineBreakEnd(s, last, keyStart, p, delimPos, valStart, quoteStart == p))
			w.lit(quote + redactedValue + quote)
			last = end
			pos = end
			continue
		}
		if rewritten.n == 0 && quoteStart == p && s[delimPos] == ':' && valStart == delimPos+1 &&
			hasFoldPrefix(s[keyStart:keyEnd], session.KeyPrefixTemp) {
			if stackedValPos, ok := scanStackedBareTempTail(s, valStart, openQ, &qState); ok {
				end = max(end, extendGluedValueEnd(s, stackedValPos, open, openQ, &qState))
			}
			if qState.openerBudgetExhausted {
				return w.failClosed()
			}
		}
		origEnd := end
		_, origQuote := unquoteValue(s[valStart:origEnd])
		// s[valStart:gapEnd] is the text between the delimiter and a value
		// in the next string, which is kept (see isQuoteGapBeforeNextString).
		gapEnd := valStart
		if origQuote != "" {
			if nextOpen := end - len(origQuote); isQuoteGapBeforeNextString(s, valStart, end, openQ, &qState) {
				// The key ends the enclosing string, and its value is the
				// next string, as in msg: "password=" "SECRET", which is
				// redacted as msg: "password=" "[REDACTED]", keeping the
				// quotes balanced.
				gapEnd = nextOpen
				nEnd, _ := scanQuotedAt(s, nextOpen, &qState)
				if inner, _ := unquoteValue(s[nextOpen:nEnd]); isRedactedValue(inner) &&
					!hasGluedSuffixAfterQuotedValue(s, nextOpen, nEnd, openQuoteState{}, &qState) {
					pos = nEnd
					continue
				}
				end = nEnd
				if hasGluedSuffixAfterQuotedValue(s, nextOpen, end, openQuoteState{}, &qState) {
					end = max(end, extendGluedValueEnd(s, end, open, openQuoteState{}, &qState))
				}
			} else if hasGluedSuffixAfterQuotedValue(s, valStart, end, openQ, &qState) {
				end = max(end, extendGluedValueEnd(s, end, open, openQ, &qState))
			}
			if qState.openerBudgetExhausted {
				return w.failClosed()
			}
		} else {
			end = textValueEnd(s, valStart, end, open)
			if (s[valStart] == '{' || s[valStart] == '[') && end > origEnd && end < len(s) &&
				!isTextValueStop(s[end]) && s[end] != '}' && s[end] != ']' && s[end] != ')' {
				tailEnd := scanValueEnd(s, end, isTextValueStop, &qState)
				{
					switch tKind, tPair := nestsSensitiveKey(s, end, tailEnd, false, openQ, &qState); tKind {
					case notNestedKey:
						end = textValueEnd(s, end, tailEnd, open)
					case gluedAfterValue:
						end = max(tailEnd, extendGluedValueEnd(s, tPair, open, openQ, &qState))
					case valueEndingInKeyName:
						// A nested key after the container, as the password: of
						// dbPassword={a: b}/password: v, is redacted with the
						// value of the pair it starts, as a second pass over
						// "[REDACTED]/password: v" does: a placeholder glued to
						// other text is no whole value (see classifyNestedKey).
						end = max(tailEnd, nestedPairEndWithState(s, tPair, open, openQ, &qState))
					}
					if qState.openerBudgetExhausted {
						return w.failClosed()
					}
				}
			}
		}
		brackets.skip(end)
		if end == valStart {
			// Only closing brackets follow the delimiter, as in {Password:}.
			pos = end
			continue
		}
		quote := origQuote
		if end == origEnd || origQuote == "" {
			var inner string
			inner, quote = unquoteValue(s[valStart:end])
			if isRedactedValue(inner) {
				pos = end
				continue
			}
		}
		writeTextPairKey(&w, last, keyStart, p, rewritten)
		w.copy(delimPos, keyLineBreakEnd(s, last, keyStart, p, delimPos, valStart, quoteStart == p))
		w.copy(valStart, gapEnd)
		w.lit(quote + redactedValue + quote)
		last = end
		pos = end
	}
	if qState.openerBudgetExhausted {
		return w.failClosed()
	}
	return w.done(last)
}

// keyLineBreakEnd returns where the line break to keep before the redacted
// value of a text pair, which follows the delimiter at s[delimPos], ends: at
// valStart when it is kept (see authHeaderLineBreak and headerLineBreak), and
// right after the delimiter otherwise.
func keyLineBreakEnd(s string, last, keyStart, keyEnd, delimPos, valStart int, unquoted bool) int {
	if authHeaderLineBreak(s, last, keyEnd, delimPos, valStart, unquoted) || headerLineBreak(s, keyStart, delimPos, valStart) {
		return valStart
	}
	return delimPos + 1
}

// writeTextPairKey writes, after the text from last on, the key of a text
// pair that starts at s[keyStart] and, with its closing quote, ends at s[p]:
// as it is, or without the padding or stray quote that rewritten drops.
func writeTextPairKey(w *spanWriter, last, keyStart, p int, rewritten keyCopy) {
	w.copy(last, keyStart)
	if rewritten.n == 0 {
		w.copy(keyStart, p)
		return
	}
	for _, sp := range rewritten.spans[:rewritten.n] {
		w.copy(sp[0], sp[1])
	}
}

// writeJSONPair writes, after the text from last on, the quoted JSON key
// whose name s[keyStart:keyEnd] stands between the quotes at s[openQuote]
// and s[closeQuote], without its padding, the ":" at s[colonPos], the line
// break that headerLineBreak keeps and the placeholder in quotes.
func writeJSONPair(w *spanWriter, last, openQuote, keyStart, keyEnd, closeQuote, colonPos, valStart int) {
	w.copy(last, openQuote+1)
	w.copy(keyStart, keyEnd)
	w.copy(closeQuote, closeQuote+1)
	lbEnd := colonPos + 1
	if headerLineBreak(w.s, openQuote, colonPos, valStart) {
		lbEnd = valStart
	}
	w.copy(colonPos, lbEnd)
	w.lit(`"` + redactedValue + `"`)
}

// headerLineBreak reports whether to keep the whitespace between the
// delimiter at s[delimPos] and a redacted value that starts at valStart on a
// following line: the key, which starts at keyStart, ends a line that holds
// an Authorization or API-key header before it, as the "token" of
// api_key: x "token":\n  SECRET other=ok (see nestedKeyBeforeNextLine).
// redactAuthHeaders redacts the header value up to such a key;
// writing the redacted value on the line of the header, as in
// api_key: [REDACTED] "token":"[REDACTED]" other=ok, would make a second pass
// redact the whole rest of the line, the text after the value too.
func headerLineBreak(s string, keyStart, delimPos, valStart int) bool {
	if !startsLine(s, delimPos+1, valStart) {
		return false
	}
	ls := strings.LastIndexAny(s[:keyStart], "\r\n") + 1
	for c := ls; c < keyStart; c++ {
		if s[c] != ':' {
			continue
		}
		hdrEnd := c
		for hdrEnd > ls && (s[hdrEnd-1] == ' ' || s[hdrEnd-1] == '\t') {
			hdrEnd--
		}
		if matchAuthHeaderAtEnd(s, ls, hdrEnd) >= 0 {
			return true
		}
	}
	return false
}

// authHeaderLineBreak reports whether to keep the whitespace between the
// delimiter at s[delimPos] and a redacted value that starts at valStart: the
// key, which ends at keyEnd and is unquoted when unquoted is true, is an
// Authorization or API-key header name (see redactAuthHeaders) followed by a
// ":" and its value starts a line, as in "api_key:\n  SECRET other=ok".
// redactAuthHeaders redacts
// the whole rest of the line after such a header, so writing the redacted
// value on the line of the header, as in "api_key:[REDACTED] other=ok", would
// make a second pass redact the text after it too.
func authHeaderLineBreak(s string, last, keyEnd, delimPos, valStart int, unquoted bool) bool {
	if !unquoted || s[delimPos] != ':' || !startsLine(s, delimPos+1, valStart) {
		return false
	}
	for i := keyEnd; i < delimPos; i++ {
		if s[i] != ' ' && s[i] != '\t' {
			return false
		}
	}
	return matchAuthHeaderAtEnd(s, last, keyEnd) >= 0
}

// isPaddedQuotedTextKey reports whether trimmed, the space-trimmed interior of
// a quoted text key, is a valid JSON pattern key or a single-line compound/path
// key whose suffix findTextKeyStart matches.
func isPaddedQuotedTextKey(trimmed string) bool {
	if isValidJSONPatternKey(trimmed) {
		return true
	}
	return trimmed != "" && !strings.ContainsAny(trimmed, "\"'\r\n,{}") && findTextKeyStart(trimmed, 0, len(trimmed)) >= 0
}

// keyCopy holds the spans of s, written in order, that make up a key written
// without its padding or a stray quote (see redactTextKeyValues).
type keyCopy struct {
	spans [3][2]int
	n     int
}

// matchPaddedQuotedTextKey matches a quoted sensitive key (", ', \" or \')
// ending at s[:p] whose interior has whitespace padding (such as
// ' client_secret ' or " password ") that findTextKeyStart rejected, returning
// the index of its opening quote and its canonical trimmed quoted form as the
// spans of s that it copies: the opening quote, the trimmed name and the
// closing quote.
func matchPaddedQuotedTextKey(s string, last, p int) (int, keyCopy) {
	if p-2 >= last && s[p-2] == '\\' && (s[p-1] == '"' || s[p-1] == '\'') && isUnescapedQuote(s, last, p-2) {
		q := s[p-1]
		closePos := p - 2
		for openIdx := closePos - 2; openIdx >= last; openIdx-- {
			c := s[openIdx]
			if c == '\\' && s[openIdx+1] == q {
				if isUnescapedQuote(s, last, openIdx) {
					if from, to := trimmedSpan(s, openIdx+2, closePos); isPaddedQuotedTextKey(s[from:to]) {
						return openIdx, keyCopy{spans: [3][2]int{{openIdx, openIdx + 2}, {from, to}, {closePos, p}}, n: 3}
					}
				}
				break
			}
			if c == ',' || c == '}' {
				break
			}
		}
		return -1, keyCopy{}
	}
	if p-1 >= last && (s[p-1] == '"' || s[p-1] == '\'') && isUnescapedQuote(s, last, p-1) {
		q := s[p-1]
		other := byte('"')
		if q == '"' {
			other = '\''
		}
		closePos := p - 1
		for openIdx := closePos - 1; openIdx >= last; openIdx-- {
			c := s[openIdx]
			if c == q {
				if isUnescapedQuote(s, last, openIdx) {
					if from, to := trimmedSpan(s, openIdx+1, closePos); isPaddedQuotedTextKey(s[from:to]) {
						return openIdx, keyCopy{spans: [3][2]int{{openIdx, openIdx + 1}, {from, to}, {closePos, p}}, n: 3}
					}
				}
				break
			}
			if c == other || c == ',' || c == '}' {
				break
			}
		}
	}
	return -1, keyCopy{}
}

// scanStackedBareTempTail checks whether s[pos:], immediately after a bare
// "temp:<seg>:" prefix, continues with one or more "<seg>:"/"<seg>=" segments
// followed by a redaction value (as in temp:a:b: SECRET), returning where that
// trailing value starts.
func scanStackedBareTempTail(s string, pos int, openQ openQuoteState, st *quoteScanState) (int, bool) {
	cur := pos
	bestValPos := 0
	found := false
	for cur < len(s) {
		// A segment ends at a text-value stop too, so that a value and
		// the pairs after it, as in temp:a:[REDACTED];token:[REDACTED],
		// are not read as segments.
		j := cur
		for j < len(s) && isTempKeyBodyByte(s[j]) && !isTextValueStop(s[j]) {
			j++
		}
		if j == cur {
			break
		}
		afterSeg := skipDelimSpaceFwd(s, j)
		if afterSeg+1 < len(s) && s[afterSeg] == '\\' && (s[afterSeg+1] == '"' || s[afterSeg+1] == '\'') {
			afterSeg += 2
		} else if afterSeg < len(s) && (s[afterSeg] == '"' || s[afterSeg] == '\'') {
			afterSeg++
		}
		dPos := skipDelimSpaceFwd(s, afterSeg)
		if dPos < len(s) && (s[dPos] == ':' || s[dPos] == '=') {
			if okVal, gluedEnd := nestedValueAfter(s, dPos+1, openQ, st); okVal &&
				(afterSeg == skipDelimSpaceFwd(s, j) || !startsWithRedactedValue(s[dPos+1:])) {
				if gluedEnd > 0 {
					bestValPos = gluedEnd
				} else {
					bestValPos = valAfterDelim(s, dPos+1)
				}
				found = true
			}
		}
		if j < len(s) && s[j] == ':' {
			cur = j + 1
			continue
		}
		break
	}
	return bestValPos, found
}

func skipInlineSpacesBack(s string, last, pos int) int {
	for pos > last {
		r, size := utf8.DecodeLastRuneInString(s[last:pos])
		if size == 0 || (!unicode.IsSpace(r) && !unicode.In(r, unicode.Z)) {
			break
		}
		pos -= size
	}
	return pos
}

// textValueEnd returns the end of the value matched at s[start:end] after a
// key: a quoted value ends at its closing quote, a value that opens a JSON
// object or array is extended as in sensitiveValueEnd, and a bare value ends
// before the "}" and "]" at its end that close a bracket opened before the
// key, and the punctuation after them (see trimEnclosingClosers). open is the
// number of brackets opened before the key.
func textValueEnd(s string, start, end, open int) int {
	if _, quote := unquoteValue(s[start:end]); quote != "" {
		return end
	}
	// rest is where the text after a JSON object or array value starts.
	rest := sensitiveValueEnd(s, start, start)
	return rest + len(trimEnclosingClosers(s[rest:max(end, rest)], open))
}

// startsLine reports whether the value at s[start:] starts a line, that is,
// whether the whitespace between it and the end of the delimiter before it,
// at delim, holds a line break.
func startsLine(s string, delim, start int) bool {
	return strings.ContainsAny(s[delim:start], "\r\n")
}

// openQuoteState records an unclosed quote opened earlier on the current line.
type openQuoteState struct {
	quote   byte
	escaped bool
}

// openQuotes tracks whether a string quote is open at monotonically advancing
// byte offsets in s, resetting at line breaks.
type openQuotes struct {
	pos   int
	state openQuoteState
}

func (q *openQuotes) before(s string, limit int) openQuoteState {
	for q.pos < limit {
		c := s[q.pos]
		if c == '\r' || c == '\n' {
			q.state = openQuoteState{}
			q.pos++
			continue
		}
		switch {
		case q.state.quote != 0 && q.state.escaped:
			if c != '\\' {
				q.pos++
				continue
			}
			m := 0
			for q.pos < limit && s[q.pos] == '\\' {
				m++
				q.pos++
			}
			if q.pos < limit && (s[q.pos] == '\r' || s[q.pos] == '\n') {
				q.state = openQuoteState{}
				q.pos++
				continue
			}
			if q.pos < limit && s[q.pos] == q.state.quote {
				q.pos++
				if m&3 != 3 {
					q.state = openQuoteState{}
				}
				continue
			}
			if m&3 == 3 {
				q.state = openQuoteState{}
			}
			if q.pos < limit {
				q.pos++
			}
		case q.state.quote != 0:
			if c == '\\' {
				q.pos++
				if q.pos < limit {
					if s[q.pos] == '\r' || s[q.pos] == '\n' {
						q.state = openQuoteState{}
					}
					q.pos++
				}
				continue
			}
			if c == q.state.quote {
				q.state = openQuoteState{}
			}
			q.pos++
		default:
			switch c {
			case '\\':
				m := 0
				for q.pos < limit && s[q.pos] == '\\' {
					m++
					q.pos++
				}
				if q.pos < limit && (s[q.pos] == '"' || s[q.pos] == '\'') {
					ch := s[q.pos]
					q.pos++
					if m%2 == 0 {
						q.state = openQuoteState{quote: ch}
					} else if m&3 == 1 {
						q.state = openQuoteState{quote: ch, escaped: true}
					}
				}
				continue
			case '"':
				q.state = openQuoteState{quote: '"'}
			case '\'':
				if !(q.pos > 0 && isWordByte(s[q.pos-1]) && q.pos+1 < len(s) && isWordByte(s[q.pos+1])) {
					q.state = openQuoteState{quote: '\''}
				}
			}
			q.pos++
		}
	}
	return q.state
}

// quotedValueEnds reports whether the quoted value s[:n] at the start of s
// ends there: s ends after it, or goes on with a character that is not a
// letter, a digit, "_" or the quote that the value starts with. Otherwise
// its quotes close one string and open the next rather than enclose a
// value, as do the quotes around " for user " in the text
// "password=abc-token: " for user "bob"".
func quotedValueEnds(s string, n int) bool {
	after := s[n:]
	if after == "" {
		return true
	}
	r, _ := utf8.DecodeRuneInString(after)
	if r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) {
		return false
	}
	_, quote := unquoteValue(s[:n])
	return !strings.HasPrefix(after, quote)
}

// openBrackets counts the brackets, "{" or "[", that are open in a text read
// from its start. A closing bracket with none open is ignored.
type openBrackets struct {
	pos   int // s[:pos] has been read or skipped
	depth int
}

// before reads s up to i and returns the number of brackets open there.
func (b *openBrackets) before(s string, i int) int {
	for ; b.pos < i; b.pos++ {
		switch s[b.pos] {
		case '{', '[':
			b.depth++
		case '}', ']':
			b.depth = max(b.depth-1, 0)
		}
	}
	return b.depth
}

// skip moves past s[:i] without reading it, as for a redacted value, so that
// scrubbing redacted text again changes nothing.
func (b *openBrackets) skip(i int) {
	b.pos = max(b.pos, i)
}

// closingPunctuation holds the closing brackets, closing quotes and
// punctuation that may follow the text of a bare value, as in the %q output
// "{APIKey:...}".
const closingPunctuation = `}])"'\:.`

// trimEnclosingClosers returns v, the text of a value, without the "}" and
// "]" at its end that close brackets opened before it, at most open of them,
// and without the punctuation after them, such as the closing quote in the
// %q output "{APIKey:...}". A "}" or "]" followed by anything else, as in
// "ab}rest", or one that closes a bracket opened in v, stays in v.
func trimEnclosingClosers(v string, open int) string {
	tail := len(strings.TrimRight(v, closingPunctuation))
	var inner openBrackets
	depth := inner.before(v, tail)
	var closers []int
	for i := tail; i < len(v); i++ {
		if v[i] == '}' || v[i] == ']' {
			closers = append(closers, i)
		}
	}
	// The first depth closers close brackets opened in v; at most open of
	// the others close brackets opened before it.
	if j := max(depth, len(closers)-open); j < len(closers) {
		return v[:closers[j]]
	}
	return v
}

// sensitiveValueEnd returns the end of the value matched at s[start:end]. A
// value that opens a JSON object or array, such as the one in
// {"token": {"value": "..."}}, is extended to its closing bracket, or to the
// end of s if it is not closed, so that the fields nested in it are redacted
// together with it. Brackets inside strings do not count, including strings
// delimited by escaped quotes, as in JSON text formatted with %q.
func sensitiveValueEnd(s string, start, end int) int {
	if c := s[start]; c != '{' && c != '[' {
		return end
	}
	if cEnd, closed := scanBalancedContainer(s, start, len(s)); closed {
		return max(end, cEnd)
	}
	return len(s)
}

// unquoteValue strips one pair of quotes (", ', \" or \') from a matched
// value and returns the content and the quote.
func unquoteValue(v string) (inner, quote string) {
	for _, q := range []string{`\"`, `\'`, `"`, `'`} {
		if len(v) >= 2*len(q) && strings.HasPrefix(v, q) && strings.HasSuffix(v, q) &&
			isUnescapedQuote(v, 0, len(v)-len(q)) {
			return v[len(q) : len(v)-len(q)], q
		}
	}
	return v, ""
}

// isRedactedMarker reports whether v, possibly percent-encoded, is a
// placeholder written by an earlier redaction pass.
func isRedactedMarker(v string) bool {
	if isKnownSentinel(v) {
		return true
	}
	if decoded, err := url.PathUnescape(v); err == nil && decoded != v {
		return isKnownSentinel(decoded)
	}
	return false
}

// isRedactedValue reports whether v, a matched value without its quotes, is
// a placeholder written by an earlier redaction pass, possibly followed by
// punctuation that ends the sentence, as in "[REDACTED]." or "[REDACTED])".
// Decoding a redacted URI can leave a placeholder next to such punctuation,
// which is not part of a credential.
func isRedactedValue(v string) bool {
	return isRedactedMarker(v) || isRedactedMarker(trimTrailingPunctuation(v))
}
