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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf16"
	"unicode/utf8"
)

const (
	redactedValue                    = "[REDACTED]"
	redactedURIPlaceholder           = "[REDACTED_SENSITIVE_URI]"
	redactedSensitiveTextPlaceholder = "[REDACTED_SENSITIVE_TEXT]"
	formattingFailedPlaceholder      = "[FORMATTER_FAILED]"
	unparseableJSONPlaceholder       = "[UNPARSEABLE_JSON_BLOB]"
)

const (
	maxInspectableJSONTextBytes  = 4_000_000 // adk-python _MAX_JSON_INSPECT_CHARS
	maxCanonicalUnescapePasses   = 4
	maxQuotedFragmentsPerInspect = 64
)

// hasQuotedSensitiveKey reports whether s contains a quoted sensitive key
// followed by optional spaces and a key-value delimiter (":" or "=").
func hasQuotedSensitiveKey(s string) bool {
	return hasQuotedSensitiveKeyWithRedacted(s, false)
}

func hasQuotedSensitiveKeyWithRedacted(s string, skipRedacted bool) bool {
	if !mayMatchTextPatterns(s) || !strings.ContainsAny(s, "\"'") || !strings.ContainsAny(s, ":=") {
		return false
	}
	prevQuote := -1
	for i := 0; i < len(s); i++ {
		if s[i] != '"' && s[i] != '\'' {
			continue
		}
		if prevQuote >= 0 {
			inner := trimInlineSpaces(s[prevQuote+1 : i])
			if isValidJSONPatternKey(inner) && followedByKeyDelimiterWithRedacted(s[i+1:], skipRedacted) {
				return true
			}
		}
		prevQuote = i
	}
	return false
}

func trimBOMAndSpace(s string) string {
	return strings.TrimFunc(s, func(r rune) bool {
		return unicode.IsSpace(r) || r == '\ufeff'
	})
}

// knownSentinels are the placeholders that this package writes in place of
// redacted, dropped or unformattable content (see isKnownSentinel).
var knownSentinels = [...]string{
	redactedValue,
	redactedURIPlaceholder,
	redactedSensitiveTextPlaceholder,
	formattingFailedPlaceholder,
	unparseableJSONPlaceholder,
	inlineDataText,
	binaryDataPlaceholder,
	circularRefPlaceholder,
	maxDepthPlaceholder,
	sanitizeBudgetPlaceholder,
	unsupportedKeyPlaceholder,
	unsupportedObjPlaceholder,
}

// maxKnownSentinelLen is the length of the longest of knownSentinels, so
// that isKnownSentinel compares no longer text.
var maxKnownSentinelLen = func() int {
	n := 0
	for _, s := range knownSentinels {
		n = max(n, len(s))
	}
	return n
}()

func isKnownSentinel(s string) bool {
	if len(s) > maxKnownSentinelLen {
		return false
	}
	for _, k := range knownSentinels {
		if s == k {
			return true
		}
	}
	return false
}

// sanitizeRawText inspects raw user or model text parts for JSON-encoded
// credentials while preserving positively identified harmless prose (such as
// markdown links "[docs](https://...)", log tags "[INFO] ready", or template
// text "{{template}} rendered") when it is not valid JSON and contains no
// raw or encoded credential markers.
func sanitizeRawText(raw string) (string, bool) {
	out, lost, settled := sanitizeRawTextOnce(raw, false)
	for cur, pass := raw, 1; !settled && out != cur && !isScrubbedPlainText(out); pass++ {
		if pass == maxRedactionPasses {
			return redactedSensitiveTextPlaceholder, true
		}
		var next string
		var nextLost bool
		next, nextLost, settled = sanitizeRawTextOnce(out, true)
		if next == out {
			break
		}
		cur, out, lost = out, next, lost || nextLost
	}
	return out, lost
}

// expressionsKeepStringsApart reports whether the expressions that scrubbed
// text on main, with and without the compound keys of the scanners (see
// legacyScrubTokensTraced), remove from doc, a JSON document or a JSON string
// literal that holds one, no part of more than one value, as the text spells
// it and as main read it with its escapes decoded (see keyRedactionHidesRunOn).
// As the text spells it, what they remove, but for the quotes and spaces at
// its ends (see trimRemovedEdges), holds the start of no value, a string,
// number, true, false or null, of doc, or of a JSON document in a string of it
// at any level, but for the value it starts in, as the value of a header, or
// of a key whose quote a later string closes, as the access_token=' of
// ["access_token='", "SECRET", "'"], does. Read with its escapes decoded (see
// escapedReadings), as main read text before it failed it closed, what they
// remove, unless it is a placeholder that they take again, as the [REDACTED]
// that a decoded \u0026 or %26 puts after an "&access_token=", may hold no
// first or last byte of a value that they keep in doc as written, but for
// those of one value that the redaction of sensitive keys replaces, which it
// holds as a whole (see keepsValueEdge): the quote after Bearer in
// ["Bearer '","SECRET","\u0027"] opens a value that the decoded \u0027
// closes, and the decoded \" after Bearer in ["Bearer \"","SECRET","\""]
// opens one that the closing quote of its string closes.
func expressionsKeepStringsApart(doc string) bool {
	var values []jsonValueSpan
	if !appendValueSpans(doc, nil, &values, 0) {
		return false
	}
	starts := make([]int, len(values))
	for k, v := range values {
		starts[k] = v.start
	}
	slices.Sort(starts)
	readings := escapedReadings(doc)
	var edges []valueEdge
	if len(readings) > 0 {
		edges = valueEdges(values)
	}
	for _, compound := range [...]bool{false, true} {
		var tr passTrace
		legacyScrubTokensTraced(doc, compound, &tr)
		for _, sp := range tr.removed {
			from, to := trimRemovedEdges(doc, sp[0], sp[1])
			if k, _ := slices.BinarySearch(starts, from+1); k < len(starts) && starts[k] < to {
				return false
			}
		}
		if len(readings) == 0 {
			continue
		}
		removed := mergeSpans(tr.removed)
		for _, r := range readings {
			var rt passTrace
			legacyScrubTokensTraced(r.text, compound, &rt)
			// What one pass rewrites and a later pass removes, as the
			// placeholder of a rewritten URI value and the quote after it,
			// is one span.
			for _, sp := range mergeSpans(rt.removed) {
				if isPlaceholderSpan(r.text, sp[0], sp[1]) {
					continue
				}
				// A bare value, which starts with no quote, ends at a
				// space, ",", ";" or "&", or at the end of the text.
				c := r.text[sp[0]]
				bare := !isQuoteByte(c) && (c != '\\' || sp[0]+1 >= sp[1] || !isQuoteByte(r.text[sp[0]+1]))
				if keepsValueEdge(values, edges, removed, r.pos[sp[0]], r.pos[sp[1]], bare) {
					return false
				}
			}
		}
	}
	return true
}

// jsonValueSpan is where a value of a JSON document starts and ends in the
// outermost text: its first byte and its last, the opening and the closing
// quote of a string, however escaped (see appendValueSpans). closeEnd is the
// end of the source of its last byte: end+1, or, for a closing quote that the
// outermost text spells as an escape, as the \" that closes a string of a
// document in a JSON string literal, the end of that escape. redacted
// reports that redactJSONValuePass replaces the value: it is the value of a
// sensitive key, which, when it is an object or an array, is one value from
// its first byte to the "}" or "]" that closes it.
type jsonValueSpan struct {
	start, end, closeEnd int
	redacted             bool
}

// valueEdge is the first or the last byte of the value at index v of the
// spans that valueEdges reads.
type valueEdge struct{ pos, v int }

// valueEdges returns the first and the last byte of each value of values,
// sorted by position.
func valueEdges(values []jsonValueSpan) []valueEdge {
	edges := make([]valueEdge, 0, 2*len(values))
	for k, v := range values {
		edges = append(edges, valueEdge{v.start, k})
		if v.end != v.start {
			edges = append(edges, valueEdge{v.end, k})
		}
	}
	slices.SortFunc(edges, func(a, b valueEdge) int { return a.pos - b.pos })
	return edges
}

// keepsValueEdge reports whether doc[from:to], the source of what the
// expressions remove from a reading of doc with its escapes decoded, holds
// the first or the last byte of a value that they keep in doc as written (see
// removed, the sorted spans that they remove from it). It may hold both of
// one value that the redaction of sensitive keys replaces (see
// jsonValueSpan), which it then holds as a whole; a value that the redaction
// keeps, as the SECRET of ["Authorization%3A x\t","SECRET"], whose decoded
// header runs on into it, or of {"temp%3aa":"SECRET"}, whose key only main
// read decoded, is a value that main read as part of a credential, and
// failed closed.
// A bare value (see expressionsKeepStringsApart) may also end at the closing
// quote of the string that it starts in, as the value of password in the
// decoded "retry\npassword=SECRET" does: the expressions as written read that
// quote as part of a bare value too. A value that a quote opens, though, may
// not: the decoded \" after Bearer in ["Bearer \"","SECRET","\""] opens a
// value that the closing quote of its string closes. Nor may it where the
// expressions as written take that closing quote too but keep a part of the
// value, as they keep the ;key=b; of {"m":"access_token=%22a;key=b;token=c"},
// whose decoded %22 opens a value that runs on to the end of the string.
// The closing quote of a value is the whole source of that quote (see
// jsonValueSpan): a span of a reading that ends at the decoded quote of a
// \" ends past both of its bytes, and still at the quote, as in a JSON
// string literal that holds {"m":"Bearer \"SECRET;key=b;token=c"}.
func keepsValueEdge(values []jsonValueSpan, edges []valueEdge, removed [][2]int, from, to int, bare bool) bool {
	whole, kept := -1, 0
	k, _ := slices.BinarySearchFunc(edges, from, func(e valueEdge, p int) int { return e.pos - p })
	for ; k < len(edges) && edges[k].pos < to; k++ {
		e := edges[k]
		if j, _ := slices.BinarySearchFunc(removed, e.pos+1, func(sp [2]int, p int) int { return sp[0] - p }); j > 0 && e.pos < removed[j-1][1] {
			if v := values[e.v]; !bare && v.end == e.pos && to <= v.closeEnd && v.start < from && removed[j-1][0] > from {
				return true
			}
			continue
		}
		if whole >= 0 && whole != e.v {
			return true
		}
		whole = e.v
		kept++
	}
	if whole < 0 {
		return false
	}
	v := values[whole]
	if v.start >= from && v.end < to {
		return !v.redacted
	}
	return !bare || kept != 1 || v.start >= from || v.end != to-1
}

// mergeSpans returns spans sorted and with the spans that overlap or touch
// merged.
func mergeSpans(spans [][2]int) [][2]int {
	merged := slices.Clone(spans)
	slices.SortFunc(merged, func(a, b [2]int) int { return a[0] - b[0] })
	out := merged[:0]
	for _, sp := range merged {
		if sp[0] >= sp[1] {
			continue
		}
		if n := len(out); n > 0 && sp[0] <= out[n-1][1] {
			out[n-1][1] = max(out[n-1][1], sp[1])
			continue
		}
		out = append(out, sp)
	}
	return out
}

// isPlaceholderSpan reports whether s[from:to], but for the quotes and
// spaces at its ends (see trimRemovedEdges), is a placeholder, written in
// any case: the expressions take a placeholder in text again when a closing
// quote follows it, as in "&access_token=[REDACTED]"}, which is no credential.
func isPlaceholderSpan(s string, from, to int) bool {
	from, to = trimRemovedEdges(s, from, to)
	for _, ph := range redactionPlaceholders {
		if strings.EqualFold(s[from:to], ph) {
			return true
		}
	}
	return false
}

// textReading is a text decoded from doc, with, for each of its bytes and
// for its end, the offset in doc of the byte or escape that it is decoded
// from.
type textReading struct {
	text string
	pos  []int
}

// escapedReadings returns the texts that scrubErrorText on main decoded doc
// into before it failed doc closed (see exposesEncodedPattern and
// hasBackslashEscapedCredentialWith): doc with its \uXXXX, \xXX, percent and
// escaped-quote escapes decoded (see canonicalizeCredentialMarker), and with
// its standard backslash escapes decoded and then those, as many times as the
// escapes nest (see unescapeCredentialStep). Each holds no text twice, and
// none is doc.
func escapedReadings(doc string) []textReading {
	if !strings.ContainsAny(doc, `\%`) {
		return nil
	}
	var readings []textReading
	add := func(text string, pos []int) {
		if text == doc {
			return
		}
		for _, r := range readings {
			if r.text == text {
				return
			}
		}
		readings = append(readings, textReading{text, pos})
	}
	if hasCanonicalizableEscape(doc) {
		add(canonicalizeMapped(doc))
	}
	cur := textReading{doc, nil}
	for range 2 * maxCanonicalUnescapePasses {
		if !strings.ContainsAny(cur.text, `\%`) {
			break
		}
		next := unescapeCredentialStepMapped(cur.text)
		if next.text == cur.text {
			break
		}
		if cur.pos != nil {
			for k, p := range next.pos {
				next.pos[k] = cur.pos[p]
			}
		}
		add(next.text, next.pos)
		cur = next
	}
	return readings
}

// unescapeCredentialStepMapped is unescapeCredentialStep, which also maps each
// byte of the result, and its end, to the offset in s that it is decoded from.
func unescapeCredentialStepMapped(s string) textReading {
	r := textReading{s, nil}
	if strings.Contains(s, `\`) {
		r.text, r.pos = unescapeBackslashControlsMapped(s)
	}
	if strings.ContainsAny(r.text, "%xXuU") {
		text, pos := canonicalizeMapped(r.text)
		if r.pos != nil {
			for k, p := range pos {
				pos[k] = r.pos[p]
			}
		}
		r.text, r.pos = text, pos
	}
	if r.pos == nil {
		r.pos = make([]int, len(s)+1)
		for k := range r.pos {
			r.pos[k] = k
		}
	}
	return r
}

// canonicalizeMapped is canonicalizeCredentialMarker, which also returns, for
// each byte of the result and for its end, the offset in s of the byte or
// escape that it is decoded from.
func canonicalizeMapped(s string) (string, []int) {
	out := make([]byte, 0, len(s))
	pos := make([]int, 0, len(s)+1)
	for i := 0; i < len(s); i++ {
		out = append(out, s[i])
		pos = append(pos, i)
		for {
			n := len(out)
			if n >= 2 && out[n-2] == '\\' && (out[n-1] == '"' || out[n-1] == '\'') {
				out = append(out[:n-2], out[n-1])
				pos = pos[:n-1]
				continue
			}
			escapeStart := -1
			var digits []byte
			if n >= 6 && out[n-6] == '\\' && (out[n-5] == 'u' || out[n-5] == 'U') &&
				isHexByte(out[n-4]) && isHexByte(out[n-3]) && isHexByte(out[n-2]) && isHexByte(out[n-1]) {
				escapeStart = n - 6
				digits = out[n-4:]
			} else if n >= 4 && out[n-4] == '\\' && (out[n-3] == 'x' || out[n-3] == 'X') &&
				isHexByte(out[n-2]) && isHexByte(out[n-1]) {
				escapeStart = n - 4
				digits = out[n-2:]
			} else if n >= 3 && out[n-3] == '%' &&
				isHexByte(out[n-2]) && isHexByte(out[n-1]) {
				escapeStart = n - 3
				digits = out[n-2:]
			}
			if escapeStart < 0 {
				break
			}
			decoded, err := strconv.ParseUint(string(digits), 16, 32)
			if err != nil || !isFoldedEscapeRune(rune(decoded), escapeStart == n-6) {
				break
			}
			escapePos := pos[escapeStart]
			out = utf8.AppendRune(out[:escapeStart], rune(decoded))
			pos = pos[:escapeStart]
			for len(pos) < len(out) {
				pos = append(pos, escapePos)
			}
		}
	}
	return string(out), append(pos, len(s))
}

// unescapeBackslashControlsMapped is unescapeBackslashControls, which also
// returns, for each byte of the result and for its end, the offset in s of
// the byte, or of the run of backslashes, that it is decoded from.
func unescapeBackslashControlsMapped(s string) (string, []int) {
	b := make([]byte, 0, len(s))
	pos := make([]int, 0, len(s)+1)
	for i := 0; i < len(s); i++ {
		if s[i] != '\\' {
			b = append(b, s[i])
			pos = append(pos, i)
			continue
		}
		j := i + 1
		for j < len(s) && s[j] == '\\' {
			j++
		}
		if j == len(s) {
			b = append(b, '\\')
			pos = append(pos, i)
			break
		}
		switch c := s[j]; c {
		case 'b':
			b = append(b, '\b')
		case 'f':
			b = append(b, '\f')
		case 'n':
			b = append(b, '\n')
		case 'r':
			b = append(b, '\r')
		case 't':
			b = append(b, '\t')
		case '/', '"', '\'':
			b = append(b, c)
		default:
			b = append(b, '\\', c)
			pos = append(pos, i)
		}
		pos = append(pos, i)
		i = j
	}
	return string(b), append(pos, len(s))
}

// appendValueSpans appends to values where each value of text, a JSON
// document or a JSON string literal at depth, and of each JSON document in a
// string of it, at any level, starts and ends in the outermost text: at the
// opening and the closing quote of a string, however escaped, or the first
// and the last byte of a number, true, false or null, with the end of the
// source of that last byte, and whether the redaction of sensitive keys
// replaces it (see jsonValueSpan). The object or array of a sensitive key,
// which the redaction replaces as a whole, is one value, and the values
// inside it are not read: in
// {"Credentials":{"AccessKeyId":"a","SecretAccessKey":"b"},"Message":"ok"}
// the expressions remove a part of the object that holds the start of the
// strings inside it, none of which the redaction keeps. pos maps each byte
// of text, and its end, to the byte of the outermost text that it is decoded
// from, or is nil when text is the outermost text. It reports false when
// depth runs out or text holds a string, or the object or array of a
// sensitive key, that does not end or decode.
func appendValueSpans(text string, pos []int, values *[]jsonValueSpan, depth int) bool {
	if depth >= maxSanitizeDepth {
		return false
	}
	at := func(i int) int {
		if pos == nil {
			return i
		}
		return pos[i]
	}
	const structural = " \t\r\n,:[]{}\""
	// sensitiveKey reports that the last string was a sensitive key, whose
	// value redactJSONValuePass replaces.
	sensitiveKey := false
	for i := 0; i < len(text); i++ {
		if c := text[i]; c != '"' {
			if strings.IndexByte(structural, c) < 0 {
				start := i
				for i+1 < len(text) && strings.IndexByte(structural, text[i+1]) < 0 {
					i++
				}
				*values = append(*values, jsonValueSpan{at(start), at(i), at(i + 1), sensitiveKey})
				sensitiveKey = false
			} else if c == '{' || c == '[' {
				if sensitiveKey {
					// The object or array of a sensitive key is one value,
					// which the redaction replaces as a whole.
					end := jsonContainerEnd(text, i)
					if end < 0 {
						return false
					}
					*values = append(*values, jsonValueSpan{at(i), at(end), at(end + 1), true})
					i = end
				}
				sensitiveKey = false
			}
			continue
		}
		j := i + 1
		for j < len(text) && text[j] != '"' {
			if text[j] == '\\' {
				j++
			}
			j++
		}
		if j >= len(text) {
			return false
		}
		isKey := false
		if k := skipJSONSpace(text, j+1); k < len(text) && text[k] == ':' {
			isKey = true
		}
		content, inner, innerPos := text[i+1:j], text[i+1:j], []int(nil)
		// Only a key or a string that may hold a JSON document of its own
		// needs its escapes decoded.
		if strings.IndexByte(content, '\\') >= 0 {
			var ok bool
			if inner, innerPos, ok = decodeJSONStringMapped(content); !ok {
				return false
			}
		}
		if isKey {
			*values = append(*values, jsonValueSpan{at(i), at(j), at(j + 1), false})
			sensitiveKey = isSensitiveKey(inner)
		} else {
			*values = append(*values, jsonValueSpan{at(i), at(j), at(j + 1), sensitiveKey})
			sensitiveKey = false
		}
		if innerPos != nil {
			if t := trimBOMAndSpace(inner); t != "" && strings.IndexByte(`{["`, t[0]) >= 0 && json.Valid([]byte(t)) {
				for k, p := range innerPos {
					innerPos[k] = at(i + 1 + p)
				}
				if !appendValueSpans(inner, innerPos, values, depth+1) {
					return false
				}
			}
		}
		i = j
	}
	return true
}

// skipJSONSpace returns the offset of the first byte at or after i in text
// that is no JSON whitespace.
func skipJSONSpace(text string, i int) int {
	for i < len(text) && strings.IndexByte(" \t\r\n", text[i]) >= 0 {
		i++
	}
	return i
}

// jsonContainerEnd returns the offset of the "}" or "]" that closes the
// object or array that text[open] opens, or -1 when none does.
func jsonContainerEnd(text string, open int) int {
	depth := 0
	for i := open; i < len(text); i++ {
		switch text[i] {
		case '"':
			for i++; i < len(text) && text[i] != '"'; i++ {
				if text[i] == '\\' {
					i++
				}
			}
		case '{', '[':
			depth++
		case '}', ']':
			if depth--; depth == 0 {
				return i
			}
		}
	}
	return -1
}

// decodeJSONStringMapped decodes raw, the text between the quotes of a JSON
// string, and returns, for each byte of the result and for its end, the
// offset in raw of the byte or escape that it is decoded from. It reports
// false for an escape that JSON does not have.
func decodeJSONStringMapped(raw string) (string, []int, bool) {
	b := make([]byte, 0, len(raw))
	pos := make([]int, 0, len(raw)+1)
	for i := 0; i < len(raw); {
		c := raw[i]
		if c != '\\' {
			b = append(b, c)
			pos = append(pos, i)
			i++
			continue
		}
		if i+1 == len(raw) {
			return "", nil, false
		}
		n := 2
		switch e := raw[i+1]; e {
		case '"', '\\', '/':
			b = append(b, e)
		case 'b':
			b = append(b, '\b')
		case 'f':
			b = append(b, '\f')
		case 'n':
			b = append(b, '\n')
		case 'r':
			b = append(b, '\r')
		case 't':
			b = append(b, '\t')
		case 'u':
			r, ok := jsonHexRune(raw[i+2:])
			if !ok {
				return "", nil, false
			}
			n = 6
			if utf16.IsSurrogate(r) {
				// A surrogate that no low surrogate follows decodes to
				// U+FFFD, as encoding/json decodes it.
				pair := utf8.RuneError
				if strings.HasPrefix(raw[i+6:], `\u`) {
					if lo, ok := jsonHexRune(raw[i+8:]); ok {
						if pair = utf16.DecodeRune(r, lo); pair != utf8.RuneError {
							n = 12
						}
					}
				}
				r = pair
			}
			start := len(b)
			b = utf8.AppendRune(b, r)
			for range len(b) - start - 1 {
				pos = append(pos, i)
			}
		default:
			return "", nil, false
		}
		pos = append(pos, i)
		i += n
	}
	return string(b), append(pos, len(raw)), true
}

// jsonHexRune returns the rune that the four hex digits that s starts with
// spell, as the \u escape of a JSON string does.
func jsonHexRune(s string) (rune, bool) {
	if len(s) < 4 {
		return 0, false
	}
	var r rune
	for k := range 4 {
		if !isHexByte(s[k]) {
			return 0, false
		}
		r = r<<4 | rune(unhexByte(s[k]))
	}
	return r, true
}

// hasHexOrPercentEscape reports whether s holds a percent escape or a \u, \U,
// \x or \X escape, any of which may hide text from the expressions of main.
func hasHexOrPercentEscape(s string) bool {
	return strings.IndexByte(s, '%') >= 0 ||
		strings.Contains(s, `\u`) || strings.Contains(s, `\U`) ||
		strings.Contains(s, `\x`) || strings.Contains(s, `\X`)
}

// keyRedactionHidesRunOn reports whether the expressions of main read a value
// across the strings of s, a JSON document or a JSON string literal that holds
// one at depth (see expressionsKeepStringsApart, which reads escapes decoded
// too), in the JSON text that main's structural redactor passes to its text
// expressions (see legacyRedactedJSONTextDepth). A document whose redaction of
// sensitive keys changes it then fails closed when main's structural redactor
// leaves a value that runs on across strings, as a header in the value of a
// compound key, which main keeps, as the Authorization\u003a\n of
// {"zz_secret":"Authorization\u003a\n","b":"SECRET"}, or writes it before
// other strings than in s, and would otherwise keep a string that main redacts
// with it, while keeping sibling fields when main's structural redactor
// already replaced the field holding the header with "[REDACTED]".
func keyRedactionHidesRunOn(s string, depth int) bool {
	legacy, changed, sawAmbiguous, ok := legacyRedactedJSONTextDepth(s, depth, false, &nodeBudget{remaining: maxSanitizeNodes})
	if !ok || !expressionsKeepStringsApart(legacy) {
		return true
	}
	if !changed && sawAmbiguous {
		sorted, _, _, ok := legacyRedactedJSONTextDepth(s, depth, true, &nodeBudget{remaining: maxSanitizeNodes})
		return !ok || (sorted != legacy && !expressionsKeepStringsApart(sorted))
	}
	return false
}

// legacyRedactedJSONTextDepth returns the JSON text that main's structural
// redactor produces for s at depth before main runs its text expressions:
// values under isLegacySensitiveKey are replaced with "[REDACTED]", and each
// container or JSON string literal that changes (or when forceSort is set) is
// marshaled with json.Marshal, while unchanged levels keep their original text
// and key order. sawAmbiguous reports an already-redacted legacy key or a
// non-JSON container-led string leaf whose re-serialization order is checked
// both ways when changed is false. It reports ok=false when s does not parse,
// depth or budget runs out, or the marshaled text is too long (see
// rewritesTooLong).
func legacyRedactedJSONTextDepth(s string, depth int, forceSort bool, budget *nodeBudget) (out string, changed, sawAmbiguous, ok bool) {
	if depth >= maxSanitizeDepth {
		return "", false, false, false
	}
	trimmed := trimBOMAndSpace(s)
	if strings.HasPrefix(trimmed, `"`) {
		if budget != nil && !budget.take() {
			return "", false, false, false
		}
		var decoded string
		if err := json.Unmarshal([]byte(trimmed), &decoded); err != nil {
			return "", false, false, false
		}
		innerTrimmed := trimBOMAndSpace(decoded)
		if innerTrimmed == "" || strings.IndexByte(`{["`, innerTrimmed[0]) < 0 || !json.Valid([]byte(innerTrimmed)) {
			amb := innerTrimmed != "" && strings.IndexByte(`{["`, innerTrimmed[0]) >= 0
			return s, false, amb, true
		}
		innerOut, innerChanged, innerAmb, innerOK := legacyRedactedJSONTextDepth(decoded, depth+1, forceSort, budget)
		if !innerOK {
			return "", false, false, false
		}
		if !innerChanged && !forceSort {
			return s, false, innerAmb, true
		}
		b, err := json.Marshal(innerOut)
		if err != nil || rewritesTooLong(trimmed, b) {
			return "", false, false, false
		}
		return string(b), innerChanged, innerAmb, true
	}
	v, sawDup, exceeded, err := unmarshalJSONStrictWithBudget(trimmed, budget)
	if err != nil || exceeded {
		return "", false, false, false
	}
	redacted, valChanged, valAmb, valOK := legacyRedactedJSONValue(v, depth+1, forceSort, budget)
	if !valOK {
		return "", false, false, false
	}
	changed = valChanged || sawDup
	if !changed && !forceSort {
		return s, false, valAmb, true
	}
	b, err := json.Marshal(redacted)
	if err != nil || rewritesTooLong(trimmed, b) {
		return "", false, false, false
	}
	return string(b), changed, valAmb, true
}

// legacyRedactedJSONValue is legacyRedactedJSONTextDepth for v, a decoded JSON
// value at depth, updated in place.
func legacyRedactedJSONValue(v any, depth int, forceSort bool, budget *nodeBudget) (out any, changed, sawAmbiguous, ok bool) {
	if depth >= maxSanitizeDepth {
		return nil, false, false, false
	}
	switch x := v.(type) {
	case map[string]any:
		for k, val := range x {
			if isLegacySensitiveKey(k) {
				if str, isStr := val.(string); !isStr || str != redactedValue {
					changed = true
				} else {
					sawAmbiguous = true
				}
				x[k] = redactedValue
				continue
			}
			child, c, amb, childOK := legacyRedactedJSONValue(val, depth+1, forceSort, budget)
			if !childOK {
				return nil, false, false, false
			}
			x[k] = child
			changed = changed || c
			sawAmbiguous = sawAmbiguous || amb
		}
	case []any:
		for i, val := range x {
			child, c, amb, childOK := legacyRedactedJSONValue(val, depth+1, forceSort, budget)
			if !childOK {
				return nil, false, false, false
			}
			x[i] = child
			changed = changed || c
			sawAmbiguous = sawAmbiguous || amb
		}
	case string:
		if t := trimBOMAndSpace(x); t != "" && strings.IndexByte(`{["`, t[0]) >= 0 {
			if json.Valid([]byte(t)) {
				return legacyRedactedJSONTextDepth(x, depth+1, forceSort, budget)
			}
			return v, false, true, true
		}
	}
	return v, changed, sawAmbiguous, true
}

// isScrubbedPlainText reports whether out, text that a pass of
// redactErrorMessage, sanitizeRawText or redactStringValue changed, needs no
// further pass: it does not start with "{", "[" or a quote, so the pass that
// changed it returned it from scrubErrorText, whose result a further
// scrubErrorText, and so a further pass over such text, leaves unchanged (see
// scrubErrorText).
func isScrubbedPlainText(out string) bool {
	trimmed := trimBOMAndSpace(out)
	return trimmed != "" && trimmed[0] != '{' && trimmed[0] != '[' && trimmed[0] != '"'
}

// maxRedactionPasses bounds how many times redactErrorMessage, sanitizeRawText,
// redactStringValue and scrubErrorText run their pass over text, each time
// over the output of the last one, to reach text that a further pass leaves
// unchanged. A pass that redacts one value may only expose another to the
// next, as when a key whose value an earlier pass replaced reads as a nested
// key. Text that still changes after maxRedactionPasses passes fails closed.
// A pass after the first reads the result of the pass before it, and keeps
// what that pass left malformed (see redactJSONTextPass). A new call reads
// the same text as it reads any other, and fails it closed where main does:
// redacting redacted text changes nothing unless that text is JSON-like and
// no longer parses.
const maxRedactionPasses = 4

// redactStringValue redacts str, a string value that SmartTruncate sanitizes
// at depth, with redactJSONTextPass and scrubErrorText, and reports
// whether content was lost. A value that starts with "{" or "[" and is no
// JSON document fails closed whatever it holds, as on main: sanitizeRawText
// and redactErrorMessage keep such text when it is prose (see
// isPositivelyHarmlessProse), and SmartTruncate keeps none. As in
// sanitizeRawText, the pass is run again over its result until that no
// longer changes (see maxRedactionPasses), each further pass with a budget of
// its own.
func redactStringValue(str string, depth int, budget *nodeBudget) (string, bool) {
	out, lost, settled := redactStringValueOnce(str, depth, false, budget)
	for cur, pass := str, 1; !settled && out != cur && !isScrubbedPlainText(out); pass++ {
		if pass == maxRedactionPasses {
			return redactedSensitiveTextPlaceholder, true
		}
		var next string
		var nextLost bool
		next, nextLost, settled = redactStringValueOnce(out, depth, true, &nodeBudget{remaining: maxSanitizeNodes})
		if next == out {
			break
		}
		cur, out, lost = out, next, lost || nextLost
	}
	return out, lost
}

// redactStringValueOnce is one pass of redactStringValue. laterPass reports
// that str is the result of an earlier pass (see redactJSONTextPass), which
// is kept rather than failed closed when it reads as malformed JSON and is
// prose (see isPositivelyHarmlessProse): the scrub of a JSON document as text
// may take the closing quote of a value with a credential that it redacts, as
// the header value of {"log": "x-api-key: SECRET\nrequest id: 42"}, which the
// text spells with the escape \n, runs on to the end of the text. The next
// pass reads what is left as malformed JSON, and keeps it, as main returns
// it, rather than failing the whole value closed. settled reports that the
// result is already a fixed point of redactStringValueOnce and needs no
// further pass.
func redactStringValueOnce(str string, depth int, laterPass bool, budget *nodeBudget) (out string, lost, settled bool) {
	redacted, changed, lost := redactJSONTextPass(str, depth, laterPass, budget)
	if laterPass && changed && redacted == unparseableJSONPlaceholder && isPositivelyHarmlessProse(str, trimBOMAndSpace(str), true) {
		return str, false, true
	}
	if scrubbed := scrubErrorText(redacted); scrubbed != redacted {
		return scrubbed, true, false
	}
	return redacted, lost, !lost
}

// sanitizeRawTextOnce is one pass of sanitizeRawText. laterPass reports that
// raw is the result of an earlier pass (see redactJSONTextPass).
func sanitizeRawTextOnce(raw string, laterPass bool) (out string, lost, settled bool) {
	budget := &nodeBudget{remaining: maxSanitizeNodes}
	sanitized, changed, lost := redactJSONTextPass(raw, 0, laterPass, budget)
	if !changed {
		scrubbed := scrubErrorText(raw)
		if scrubbed != raw {
			return scrubbed, true, false
		}
		return raw, false, true
	}
	if sanitized != unparseableJSONPlaceholder {
		scrubbed := scrubErrorText(sanitized)
		if scrubbed != sanitized {
			return scrubbed, true, false
		}
		return sanitized, lost, !lost
	}
	trimmed := trimBOMAndSpace(raw)
	if isPositivelyHarmlessProse(raw, trimmed, laterPass) {
		return raw, false, true
	}
	return sanitized, lost, true
}

func hasCandidateQuotedSensitiveKey(s string, depth int) bool {
	return hasCandidateQuotedSensitiveKeyWithRedacted(s, depth, false)
}

func hasCandidateQuotedSensitiveKeyWithRedacted(s string, depth int, skipRedacted bool) bool {
	if depth > 4 || s == "" {
		return false
	}
	if len(s) > maxInspectableJSONTextBytes {
		s = s[:maxInspectableJSONTextBytes]
	}
	if hasQuotedSensitiveKeyWithRedacted(s, skipRedacted) {
		return true
	}
	if canon := canonicalizeCredentialMarker(s); canon != s {
		canonSkip := skipRedacted && !strings.ContainsAny(canon, "{[")
		if hasQuotedSensitiveKeyWithRedacted(canon, canonSkip) {
			return true
		}
	}

	inQuote := false
	quoteChar := byte(0)
	start := 0
	escaped := false

	for i := range len(s) {
		b := s[i]
		if !inQuote {
			if b == '"' || b == '\'' {
				inQuote = true
				quoteChar = b
				start = i
				escaped = false
			}
		} else {
			if escaped {
				escaped = false
				continue
			}
			if b == '\\' {
				escaped = true
				continue
			}
			if b == quoteChar {
				inQuote = false
				rawCandidate := s[start+1 : i]
				var candidate string
				if quoteChar == '"' {
					var unquoted string
					if err := json.Unmarshal([]byte(s[start:i+1]), &unquoted); err == nil {
						candidate = unquoted
					} else {
						candidate = canonicalizeCredentialMarker(rawCandidate)
					}
				} else {
					candidate = canonicalizeCredentialMarker(rawCandidate)
				}
				if isSensitiveKey(candidate) || isSensitiveKey(canonicalizeCredentialMarker(candidate)) {
					if followedByKeyDelimiterWithRedacted(s[i+1:], skipRedacted) {
						return true
					}
				}
				if strings.ContainsAny(candidate, `"'`) {
					childSkip := skipRedacted && !strings.ContainsAny(candidate, "{[")
					if hasCandidateQuotedSensitiveKeyWithRedacted(candidate, depth+1, childSkip) {
						return true
					}
				}
			}
		}
	}
	return false
}

// followedByKeyDelimiterWithRedacted reports whether rest, the text after a
// quoted sensitive key, starts with a key-value delimiter (":" or "=") after
// optional whitespace, ASCII control bytes, or Unicode Cf format characters,
// optionally skipping already-redacted placeholder values.
func followedByKeyDelimiterWithRedacted(rest string, skipRedacted bool) bool {
	sawObfuscatedSep := false
	for i, r := range rest {
		if r <= 0x20 || r == 0x7f || unicode.IsSpace(r) || unicode.In(r, unicode.Z, unicode.Cf) {
			if !isASCIIInlineOrNewlineSpace(r) {
				sawObfuscatedSep = true
			}
			continue
		}
		if r != ':' && r != '=' {
			return false
		}
		if sawObfuscatedSep {
			return true
		}
		return !skipRedacted || !startsWithRedactedValue(rest[i+1:])
	}
	return false
}

func isASCIIInlineOrNewlineSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\f'
}

// startsWithRedactedValue reports whether afterDelim, the text after a ":" or
// "=" delimiter, begins (after optional whitespace) with a complete "[REDACTED]"
// placeholder token at a real token boundary rather than a secret that shares
// its prefix or has a value glued after closing punctuation.
func startsWithRedactedValue(afterDelim string) bool {
	val := strings.TrimLeftFunc(afterDelim, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.In(r, unicode.Z)
	})
	if val == "" {
		return false
	}
	for _, q := range []string{`\"`, `\'`, `"`, `'`} {
		lit := q + redactedValue + q
		if strings.HasPrefix(val, lit) {
			return isRedactedPlaceholderBoundary(val[len(lit):])
		}
	}
	if strings.HasPrefix(val, redactedValue) {
		return isRedactedPlaceholderBoundary(val[len(redactedValue):])
	}
	return false
}

// isRedactedPlaceholderBoundary reports whether tail, the text immediately
// following a bare or quoted "[REDACTED]" placeholder, begins at a clean token
// boundary: end of input, whitespace or ",", ";" or "&", closing brackets or
// "." not followed by a glued value token, or a trailing ":" with no value
// after it.
func isRedactedPlaceholderBoundary(tail string) bool {
	if tail == "" {
		return true
	}
	c := tail[0]
	if isTextValueStop(c) {
		return true
	}
	if c == ':' {
		rest := strings.TrimLeftFunc(tail[1:], func(r rune) bool {
			return unicode.IsSpace(r) || unicode.In(r, unicode.Z) || (r <= 0x20 || r == 0x7f)
		})
		return rest == ""
	}
	if c == '}' || c == ']' || c == ')' || c == '.' {
		k := 0
		for k < len(tail) {
			switch tail[k] {
			case '}', ']', ')', '.', '"', '\'':
				k++
				continue
			case '\\':
				if k+1 < len(tail) && (tail[k+1] == '"' || tail[k+1] == '\'') {
					k += 2
					continue
				}
			}
			break
		}
		return k == len(tail) || isTextValueStop(tail[k])
	}
	return false
}

// isBalancedAlreadyRedactedProse reports whether raw, quote-led text, may be
// already-redacted prose such as "api.key":"[REDACTED]" other=ok: it holds no
// "{", no "[" but that of a "[REDACTED]" placeholder and no escaped quote,
// which may hide a quoted key, and its double quotes pair up into strings that
// end at clean boundaries (see hasMalformedQuotedMembers).
func isBalancedAlreadyRedactedProse(raw string) bool {
	doubleQuotes := 0
	for i := 0; i < len(raw); i++ {
		switch raw[i] {
		case '"':
			doubleQuotes++
		case '{':
			return false
		case '[':
			if !strings.HasPrefix(raw[i:], redactedValue) {
				return false
			}
		case '\\':
			if i+1 < len(raw) && isQuoteByte(raw[i+1]) {
				return false
			}
		}
	}
	return doubleQuotes%2 == 0 && !hasMalformedQuotedMembers(raw)
}

// hasMalformedQuotedMembers reports whether raw, read by pairing its double
// quotes from the start into strings, holds a string followed by a word byte
// ("..."x) or by another string, double- or single-quoted, with no delimiter
// in between ("..." "..."), or holds, outside the strings, a "," followed by
// a quote, escaped or not, as the members of a JSON object are, whatever the
// value before the "," ("...": "...", "..." or "...": [REDACTED], '...'), or
// a quote of any kind anywhere after a "[REDACTED]" placeholder, unquoted or
// in quotes, with or without other text in between ("...": [REDACTED] "...",
// "...": "[REDACTED]"; '...' or "...": '[REDACTED]' 1 "..."), as in the
// members of a JSON object, or an unquoted placeholder as the value of a
// quoted sensitive key and ":" (see isAfterQuotedSensitiveKeyColon). The
// quoted value of an unquoted key right after its delimiter, as the
// other="ok" of "token": "[REDACTED]"; other="ok", is prose, as is unquoted
// text. Such text is not prose but malformed JSON text, which fails closed.
// A complete "[REDACTED]" string right after a string that ends in a
// delimiter, as in "password=" "[REDACTED]", is text that an earlier pass
// redacted, and so is a placeholder glued between a string that ends in a
// delimiter and the next, as the redacted nested value of
// "S0:password="[REDACTED]"".
func hasMalformedQuotedMembers(raw string) bool {
	open := -1
	// lastOpen and lastEnd are where the last string opened and closed.
	lastOpen, lastEnd := -1, -1
	// afterValue reports that a "[REDACTED]" placeholder, unquoted or in
	// quotes, stands before raw[i] outside the strings.
	afterValue := false
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if open >= 0 {
			if c != '"' {
				continue
			}
			if isMalformedAfterString(raw, open, i) {
				return true
			}
			if raw[open+1:i] == redactedValue {
				afterValue = true
			}
			lastOpen, lastEnd = open, i
			open = -1
			continue
		}
		if afterValue && (isQuoteByte(c) || (c == '\\' && i+1 < len(raw) && isQuoteByte(raw[i+1]))) && !isUnquotedMemberValueAt(raw, i) {
			return true
		}
		switch c {
		case '"':
			open = i
		case '[':
			if !strings.HasPrefix(raw[i:], redactedValue) {
				continue
			}
			k := i + len(redactedValue)
			switch {
			case i > 0 && raw[i-1] == '\'' && k < len(raw) && raw[k] == '\'':
				afterValue = true
				i = k
			case lastEnd >= 0 && lastEnd == i-1 && k < len(raw) && raw[k] == '"' && endsInDelimiter(raw[lastOpen+1:lastEnd]):
				i = k - 1
			case isAfterQuotedSensitiveKeyColon(raw, i, lastOpen, lastEnd):
				return true
			default:
				afterValue = true
				i = k - 1
			}
		case ',':
			j := skipInlineSpacesFwd(raw, i+1, len(raw))
			if j < len(raw) && (isQuoteByte(raw[j]) || (raw[j] == '\\' && j+1 < len(raw) && isQuoteByte(raw[j+1]))) {
				return true
			}
		}
	}
	return false
}

// isUnquotedMemberValueAt reports whether the quote at raw[i], which may be
// escaped, opens the value of an unquoted key: a word byte, then a ":" or
// "=" and inline spaces stand right before it, as in other="ok".
func isUnquotedMemberValueAt(raw string, i int) bool {
	j := i
	for j > 0 && (raw[j-1] == ' ' || raw[j-1] == '\t') {
		j--
	}
	return j >= 2 && (raw[j-1] == ':' || raw[j-1] == '=') && isWordByte(raw[j-2])
}

// isAfterQuotedSensitiveKeyColon reports whether the unquoted placeholder at
// raw[i] is the value of a quoted sensitive key and ":", as in
// "token": [REDACTED], where raw[lastOpen:lastEnd+1] is the last string
// before it: the passes write the redacted value of such a key in quotes
// ("token":"[REDACTED]"), so an unquoted one is not text that they redacted.
func isAfterQuotedSensitiveKeyColon(raw string, i, lastOpen, lastEnd int) bool {
	if lastEnd < 0 {
		return false
	}
	j := i
	for j > lastEnd+1 && isDelimSpace(raw[j-1]) {
		j--
	}
	if j == lastEnd+1 || raw[j-1] != ':' {
		return false
	}
	j--
	for j > lastEnd+1 && isDelimSpace(raw[j-1]) {
		j--
	}
	return j == lastEnd+1 && isSensitiveKey(trimInlineSpaces(raw[lastOpen+1:lastEnd]))
}

// endsInDelimiter reports whether s, the inside of a string, ends in ":" or
// "=" before any trailing spaces or tabs.
func endsInDelimiter(s string) bool {
	s = trimInlineSpaces(s)
	return s != "" && (s[len(s)-1] == ':' || s[len(s)-1] == '=')
}

// isMalformedAfterString reports whether the string raw[open:end+1] is
// followed by a word byte, or by another string, but for a complete
// "[REDACTED]" string when it ends in ":" or "=" (see
// hasMalformedQuotedMembers).
func isMalformedAfterString(raw string, open, end int) bool {
	if end+1 < len(raw) && isWordByte(raw[end+1]) {
		return true
	}
	j := skipInlineSpacesFwd(raw, end+1, len(raw))
	if j >= len(raw) || !isQuoteByte(raw[j]) {
		return false
	}
	lit := raw[j:j+1] + redactedValue + raw[j:j+1]
	if endsInDelimiter(raw[open+1:end]) && strings.HasPrefix(raw[j:], lit) && isRedactedPlaceholderBoundary(raw[j+len(lit):]) {
		return false
	}
	return true
}

// isPositivelyHarmlessProse reports whether raw, text that starts with "{",
// "[" or a quote once its padding is trimmed (see trimmed) and that fails
// closed as malformed JSON, is prose that sanitizeRawText and
// redactErrorMessage keep, as on main: it holds no escaped quote, nothing
// that the scrub or the legacy expressions redact, as written or with its
// escapes decoded, and no quoted sensitive key before a delimiter. laterPass
// reports that raw is the result of an earlier pass (see
// redactJSONTextPass). Only then is quote-led text whose quoted sensitive
// keys all hold a "[REDACTED]" placeholder read as redacted text and kept,
// and quote-led text that starts like an encoded container failed closed.
func isPositivelyHarmlessProse(raw, trimmed string, laterPass bool) bool {
	if len(raw) > maxInspectableJSONTextBytes {
		return false
	}
	if laterPass && strings.HasPrefix(trimmed, `"`) && startsLikeEncodedContainer(trimmed) {
		return false
	}
	// An escaped quote can hide a quoted credential key from the pattern
	// checks below; other backslashes (paths, regexes) are harmless.
	if strings.Contains(raw, `\"`) {
		return false
	}
	if scrubErrorText(raw) != raw || legacyScrubChanges(raw) || hasEncodedCredentialMarker(raw) {
		return false
	}
	// Malformed "{" / "[" JSON containers must always check quoted sensitive
	// keys strictly without skipping "[REDACTED]" values, whereas quote-led
	// prose with balanced double quotes and no JSON container openers that
	// an earlier pass returned may be redacted output (e.g.
	// `"api_key":"[REDACTED]" other=ok`).
	allowRedactedProse := laterPass && strings.HasPrefix(trimmed, `"`) && isBalancedAlreadyRedactedProse(raw)
	if hasCandidateQuotedSensitiveKeyWithRedacted(raw, 0, allowRedactedProse) {
		return false
	}
	canon := canonicalizeCredentialMarker(raw)
	if canon != raw && strings.ContainsAny(canon, "{[") {
		allowRedactedProse = false
	}
	if hasQuotedSensitiveKeyWithRedacted(canon, allowRedactedProse) {
		return false
	}
	// Bracket-, brace- or quote-led text that failed JSON parsing and
	// carries no credential pattern is prose (e.g.
	// `[INFO] user clicked "Submit"` or `"Hello," she said [laughing].`),
	// unless a quoted string in it decodes to a credential: the text may
	// hold a JSON string whose escapes the checks above do not all decode.
	return (strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, `"`)) &&
		!hasJSONEscapedCredential(raw)
}

// hasJSONEscapedCredential reports whether the text between two consecutive
// quotes in s, read as a JSON string, holds a credential pattern once
// decoded. canonicalizeCredentialMarker leaves JSON escapes such as \t alone,
// so "Bearer\tSECRET" hides a bearer token from the checks on the raw text.
// Each pair of consecutive quotes is tried, whichever quote opens a string;
// s must not contain an escaped quote, so the work stays linear in len(s).
func hasJSONEscapedCredential(s string) bool {
	if !strings.Contains(s, `\`) {
		return false
	}
	open := strings.IndexByte(s, '"')
	for open >= 0 {
		n := strings.IndexByte(s[open+1:], '"')
		if n < 0 {
			return false
		}
		closing := open + 1 + n
		if quoted := s[open : closing+1]; strings.Contains(quoted, `\`) {
			var decoded string
			if json.Unmarshal([]byte(quoted), &decoded) == nil && hasAnyCredentialMarker(decoded) {
				return true
			}
		}
		open = closing
	}
	return false
}

// maxRewriteGrowth and maxRewriteSlack bound how much longer than the text
// that it replaces redactJSONTextPass writes a JSON value again (see
// rewritesTooLong). json.Marshal writes "<", ">" and "&" as six bytes each, and
// each level of JSON string encoding around a document, as a JSON string
// literal or a string leaf of another document that holds it, escapes the
// escapes inside it once more: the "&" of a document whose sensitive key is
// redacted takes 6 bytes written again, 7 in a literal, 9 in a literal in a
// literal, and 2^k+5 at k levels, so that a document nested in many levels
// would grow by a factor for each level as it is written again.
const (
	maxRewriteGrowth = 8
	maxRewriteSlack  = 4096
)

// rewritesTooLong reports whether out, the JSON text that the redaction writes
// in place of in, is longer than maxRewriteGrowth times in and
// maxRewriteSlack bytes, or than maxInspectableJSONTextBytes. Such a value
// fails closed, as text longer than maxInspectableJSONTextBytes does.
func rewritesTooLong(in string, out []byte) bool {
	return len(out) > maxInspectableJSONTextBytes || len(out) > maxRewriteGrowth*len(in)+maxRewriteSlack
}

// redactJSONTextPass inspects s if it is a JSON object, array, or
// JSON-encoded string literal, recursively redacting values under sensitive
// keys. When a JSON-like blob cannot be parsed or exceeds maxSanitizeDepth or
// budget, it fails closed and reports contentLost = true, as it does when the
// redaction would write it again too long (see rewritesTooLong).
//
// It is one pass of redactErrorMessage, sanitizeRawText or redactStringValue
// over s (see maxRedactionPasses). laterPass reports that s is the result of
// an earlier pass of the same call rather than text as it was received. Text
// as it was received fails closed wherever it does on main: no part of it is
// read as text that is redacted already, however much it looks like the
// result of a redaction, since a credential that the scrub does not read may
// stand next to a placeholder that the text came with. Only the result of an
// earlier pass, which that pass scrubbed and main returns as it is, is read
// as redacted text: a "[REDACTED_SENSITIVE_URI]" placeholder that it starts
// with (see startsWithRedactedURI), a "[REDACTED]" placeholder in the text
// after or inside a JSON string literal (see hasRedactableContainerText and
// startsLikeEncodedContainer), and quote-led text whose quoted sensitive keys
// all hold one (see isPositivelyHarmlessProse).
func redactJSONTextPass(s string, depth int, laterPass bool, budget *nodeBudget) (sanitized string, changed, contentLost bool) {
	sanitized, changed, contentLost, _ = redactJSONTextInner(s, depth, false, true, laterPass, budget)
	return sanitized, changed, contentLost
}

// literalRunsIntoSuffix reports whether the expressions on main, with or
// without compound keys, remove any byte after litEnd (in the suffix after a
// leading JSON string literal) from doc as written or with its escapes
// decoded (see escapedReadings).
func literalRunsIntoSuffix(doc string, litEnd int) bool {
	readings := escapedReadings(doc)
	for _, compound := range [...]bool{false, true} {
		var tr passTrace
		legacyScrubTokensTraced(doc, compound, &tr)
		for _, sp := range tr.removed {
			if sp[1] > litEnd {
				return true
			}
		}
		for _, r := range readings {
			var rt passTrace
			legacyScrubTokensTraced(r.text, compound, &rt)
			for _, sp := range mergeSpans(rt.removed) {
				if !isPlaceholderSpan(r.text, sp[0], sp[1]) && r.pos[sp[1]] > litEnd {
					return true
				}
			}
		}
	}
	return false
}

func redactJSONTextInner(s string, depth int, inContainer, allowCandidateRecheck, laterPass bool, budget *nodeBudget) (sanitized string, changed, contentLost, rewriteExceeded bool) {
	trimmed := trimBOMAndSpace(s)
	if trimmed == "" || isKnownSentinel(trimmed) {
		return s, false, false, false
	}
	if strings.HasPrefix(trimmed, `"`) {
		if depth >= maxSanitizeDepth {
			return maxDepthPlaceholder, true, true, false
		}
		if budget != nil && !budget.take() {
			return sanitizeBudgetPlaceholder, true, true, false
		}
		if len(trimmed) > maxInspectableJSONTextBytes {
			return unparseableJSONPlaceholder, true, true, false
		}
		dec := json.NewDecoder(strings.NewReader(trimmed))
		var decoded string
		if err := dec.Decode(&decoded); err != nil {
			afterQuote := trimBOMAndSpace(trimmed[1:])
			if strings.HasPrefix(afterQuote, "{") || strings.HasPrefix(afterQuote, "[") ||
				strings.HasPrefix(afterQuote, `\`) || strings.HasPrefix(afterQuote, `"`) ||
				hasAnyCredentialMarker(trimmed) {
				return unparseableJSONPlaceholder, true, true, false
			}
			return s, false, false, false
		}
		litEnd := int(dec.InputOffset())
		suffix := trimmed[litEnd:]
		trimmedSuffix := trimBOMAndSpace(suffix)
		if trimmedSuffix != "" {
			if hasRedactableContainerText(suffix, trimmed, laterPass) || strings.HasPrefix(trimmedSuffix, `\`) ||
				isUnbalancedQuotedKeyText(trimmed, decoded, suffix) ||
				(hasMalformedQuotedMembers(trimmed) && hasCandidateQuotedSensitiveKey(trimmed, 0)) ||
				hasAnyCredentialMarker(suffix) || inspectQuotedFragments(suffix, depth+1, laterPass, budget) {
				return unparseableJSONPlaceholder, true, true, false
			}
			// Scrubbing the text may leave it malformed, as when a redacted
			// value takes a quote with it; it then fails closed here already,
			// so that a pass over the scrubbed text agrees. Non-quote results
			// (such as redactedSensitiveTextPlaceholder) are checked only after
			// inspecting decoded below so malformed/credential-bearing literals
			// still return unparseableJSONPlaceholder.
			if scrubbed := scrubErrorText(trimmed); scrubbed != trimmed && strings.HasPrefix(scrubbed, `"`) {
				if again, _, _ := redactJSONTextPass(scrubbed, depth+1, true, budget); again == unparseableJSONPlaceholder {
					return unparseableJSONPlaceholder, true, true, false
				}
			}
		}
		trimmedInner := trimBOMAndSpace(decoded)
		var innerSanitized string
		var innerChanged, innerLost, innerExceeded bool
		switch {
		case strings.HasPrefix(trimmedInner, "{") || strings.HasPrefix(trimmedInner, "[") || strings.HasPrefix(trimmedInner, `"`):
			innerSanitized, innerChanged, innerLost, innerExceeded = redactJSONTextInner(decoded, depth+1, true, true, laterPass, budget)
		case hasRedactableContainerText(decoded, decoded, laterPass) || hasAnyCredentialMarker(decoded) || inspectQuotedFragments(decoded, depth+1, laterPass, budget):
			return unparseableJSONPlaceholder, true, true, false
		}
		// A document in the literal that fails closed, as one whose redaction
		// of sensitive keys would hide a value that runs on (see
		// keyRedactionHidesRunOn), fails the literal closed too, rather than
		// standing in it as a JSON string.
		if innerChanged && (innerSanitized == unparseableJSONPlaceholder || innerSanitized == maxDepthPlaceholder ||
			innerSanitized == sanitizeBudgetPlaceholder || innerSanitized == redactedSensitiveTextPlaceholder) {
			return innerSanitized, true, true, innerExceeded
		}
		if !innerChanged {
			if trimmedSuffix != "" {
				if scrubbed := scrubErrorText(trimmed); scrubbed != trimmed && (scrubbed == redactedSensitiveTextPlaceholder || !strings.HasPrefix(scrubbed, `"`)) {
					return redactedSensitiveTextPlaceholder, true, true, false
				}
			}
			return s, false, false, false
		}
		b, err := json.Marshal(innerSanitized)
		if err != nil || rewritesTooLong(trimmed[:litEnd], b) {
			return unparseableJSONPlaceholder, true, true, true
		}
		if trimmedSuffix != "" {
			if !inContainer && keyRedactionHidesRunOn(trimmed[:litEnd], depth) {
				return redactedSensitiveTextPlaceholder, true, true, false
			}
			if literalRunsIntoSuffix(trimmed, litEnd) {
				return redactedSensitiveTextPlaceholder, true, true, false
			}
			if scrubbed := scrubErrorText(trimmed); scrubbed != trimmed && (scrubbed == redactedSensitiveTextPlaceholder || !strings.HasPrefix(scrubbed, `"`)) {
				return redactedSensitiveTextPlaceholder, true, true, false
			}
			candidate := string(b) + suffix
			// The candidate, and its scrub below, are read as the pass
			// after this one reads them (see redactJSONTextPass), so that
			// what that pass would change or fail closed is returned here.
			if allowCandidateRecheck {
				freshBudget := &nodeBudget{remaining: maxSanitizeNodes}
				if again, againChanged, againLost, _ := redactJSONTextInner(candidate, depth, true, false, true, freshBudget); againChanged || againLost {
					return again, true, true, false
				}
			}
			if scrubbed := scrubErrorText(candidate); scrubbed != candidate {
				if scrubbed == redactedSensitiveTextPlaceholder || !strings.HasPrefix(scrubbed, `"`) {
					return redactedSensitiveTextPlaceholder, true, true, false
				}
				freshBudget := &nodeBudget{remaining: maxSanitizeNodes}
				if again, _, _, _ := redactJSONTextInner(scrubbed, depth, true, false, true, freshBudget); again == unparseableJSONPlaceholder {
					return unparseableJSONPlaceholder, true, true, false
				}
			}
			return candidate, true, innerLost, innerExceeded
		}
		if !inContainer && keyRedactionHidesRunOn(trimmed, depth) {
			return redactedSensitiveTextPlaceholder, true, true, false
		}
		return string(b), true, innerLost, innerExceeded
	}

	if !strings.HasPrefix(trimmed, "{") && !strings.HasPrefix(trimmed, "[") {
		return s, false, false, false
	}
	if laterPass && startsWithRedactedURI(trimmed) {
		// Text that an earlier pass left starting with a redacted URI is not
		// a JSON array but text, which is kept when redactErrorMessage and
		// sanitizeRawText would both keep it as malformed JSON (see
		// redactErrorMessageOnce and isPositivelyHarmlessProse), so that a
		// later pass over it changes nothing. Text that starts so as it is
		// received fails closed like any other "["-led text that does not
		// parse.
		if hasAnyCredentialMarker(trimmed) || hasEncodedCredentialMarker(trimmed) || hasCandidateQuotedSensitiveKey(trimmed, 0) ||
			!isPositivelyHarmlessProse(s, trimmed, true) {
			return unparseableJSONPlaceholder, true, true, false
		}
		return s, false, false, false
	}
	if depth >= maxSanitizeDepth {
		return maxDepthPlaceholder, true, true, false
	}
	if budget != nil && !budget.take() {
		return sanitizeBudgetPlaceholder, true, true, false
	}
	if len(trimmed) > maxInspectableJSONTextBytes {
		return unparseableJSONPlaceholder, true, true, false
	}
	decoded, sawDuplicateKey, budgetExceeded, err := unmarshalJSONStrictWithBudget(trimmed, budget)
	if budgetExceeded {
		return sanitizeBudgetPlaceholder, true, true, false
	}
	if err != nil {
		return unparseableJSONPlaceholder, true, true, false
	}
	redacted, valChanged, valLost, valExceeded := redactJSONValueInner(decoded, depth+1, true, laterPass, budget)
	if str, ok := redacted.(string); ok && str == sanitizeBudgetPlaceholder {
		return sanitizeBudgetPlaceholder, true, true, false
	}
	if !valChanged && !sawDuplicateKey && !valLost {
		return s, false, false, false
	}
	if valExceeded {
		return redactedSensitiveTextPlaceholder, true, true, true
	}
	b, err := json.Marshal(redacted)
	if err != nil || rewritesTooLong(trimmed, b) {
		return unparseableJSONPlaceholder, true, true, true
	}
	if !inContainer {
		if keyRedactionHidesRunOn(trimmed, depth) {
			return redactedSensitiveTextPlaceholder, true, true, false
		}
	} else if hasHexOrPercentEscape(trimmed) && keyRedactionHidesRunOn(trimmed, depth) {
		return redactedSensitiveTextPlaceholder, true, true, false
	}
	return string(b), true, valLost, false
}

// startsWithRedactedURI reports whether s starts with the placeholder that
// scrubURITokens writes for a URI with credentials, followed by the end of
// s or a byte that ends a URI token there: whitespace, a quote, "<", ">" or
// trailing punctuation (see uriTokenRe and trimTrailingPunctuation).
func startsWithRedactedURI(s string) bool {
	if !strings.HasPrefix(s, redactedURIPlaceholder) {
		return false
	}
	rest := s[len(redactedURIPlaceholder):]
	return rest == "" || isRegexpSpace(rest[0]) || strings.IndexByte("\"'<>.,;:)]}", rest[0]) >= 0
}

// isUnbalancedQuotedKeyText reports whether trimmed, quote-led text whose
// first JSON string decodes to decoded and is followed by suffix, starts with
// a sensitive key and a delimiter, as in "api.key": failed: "..., and holds an
// odd number of double quotes. Such text is a truncated or malformed run of
// key-value pairs rather than a JSON string literal, so it fails closed: a
// quote left open in it may run on over the value of any key, and text that
// a pass redacts stays malformed, so that every pass over it agrees.
func isUnbalancedQuotedKeyText(trimmed, decoded, suffix string) bool {
	if strings.Count(trimmed, `"`)%2 == 0 || !followedByKeyDelimiterWithRedacted(suffix, false) {
		return false
	}
	key := trimInlineSpaces(decoded)
	return isSensitiveKey(key) || isValidJSONPatternKey(key) || isSensitiveKey(canonicalizeCredentialMarker(key))
}

// hasRedactableContainerText reports whether s, part of the text whole, holds
// a "{" or "[" that may open a JSON container, which fails the text closed:
// any "{" or "[" does, as on main, unless laterPass reports that whole is the
// result of an earlier pass (see redactJSONTextPass). Then the "[" of a
// complete "[REDACTED]" placeholder does not count when whole is balanced
// already-redacted prose (see isBalancedAlreadyRedactedProse) whose quoted
// sensitive keys all have redacted values, as in "api.key":"[REDACTED]", so
// that a later pass over it changes nothing, while truncated or suffixed
// text, as in "token": "[REDACTED]", "note": "v, or text with a quoted
// sensitive key whose value the passes do not read, as in
// "user:token"<U+00A0>:v [REDACTED], still fails closed.
func hasRedactableContainerText(s, whole string, laterPass bool) bool {
	if !strings.ContainsAny(s, "{[") {
		return false
	}
	return !laterPass || hasContainerOpener(s) || !isBalancedAlreadyRedactedProse(whole) || hasCandidateQuotedSensitiveKeyWithRedacted(whole, 0, true)
}

// hasContainerOpener reports whether s holds a "{" or a "[" that may open a
// JSON container, that is, any "{" or "[" other than the "[" of a complete
// "[REDACTED]" placeholder (see isCompleteRedactedPlaceholderAt), so that text
// an earlier pass redacted, as in "api.key":"[REDACTED]", reads the same on a
// second pass, while a placeholder with text glued after it, as in
// "token":"[REDACTED]":SECRET, still counts.
func hasContainerOpener(s string) bool {
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '{':
			return true
		case '[':
			if !isCompleteRedactedPlaceholderAt(s, i) {
				return true
			}
		}
	}
	return false
}

// isCompleteRedactedPlaceholderAt reports whether a "[REDACTED]" placeholder
// starts at s[i] and, after any closing quotes, ends at a clean token boundary
// (see endsAtPlaceholderBoundary).
func isCompleteRedactedPlaceholderAt(s string, i int) bool {
	return strings.HasPrefix(s[i:], redactedValue) && endsAtPlaceholderBoundary(s[i+len(redactedValue):])
}

// endsAtPlaceholderBoundary reports whether tail, the text after a
// placeholder, begins at a clean token boundary (see
// isRedactedPlaceholderBoundary) after any closing quotes (", ', \" or \').
func endsAtPlaceholderBoundary(tail string) bool {
	k := 0
	for k < len(tail) {
		if tail[k] == '"' || tail[k] == '\'' {
			k++
			continue
		}
		if k+1 < len(tail) && tail[k] == '\\' && (tail[k+1] == '"' || tail[k+1] == '\'') {
			k += 2
			continue
		}
		break
	}
	return isRedactedPlaceholderBoundary(tail[k:])
}

// inspectQuotedFragments reports whether a quoted fragment of s decodes to
// text that needs redaction, or could not be decoded and starts like an
// encoded container. The callers have already scanned all of s for
// credential markers, so the work here stays linear in len(s).
func inspectQuotedFragments(s string, depth int, laterPass bool, budget *nodeBudget) bool {
	if depth >= maxSanitizeDepth {
		return true
	}
	fragments := 0
	// A quote before failedUntil was read as an escaped quote by a decode
	// that started at an earlier quote and failed at failedUntil. A decode
	// starting at it would fail at the same point, so it is not repeated.
	failedUntil := 0
	for i := 0; i < len(s); i++ {
		if s[i] != '"' {
			continue
		}
		fragments++
		if fragments > maxQuotedFragmentsPerInspect {
			return true
		}
		if budget != nil && !budget.take() {
			return true
		}
		sub := s[i:]
		if i < failedUntil {
			if startsLikeEncodedContainer(sub) {
				return true
			}
			continue
		}
		dec := json.NewDecoder(strings.NewReader(sub))
		var decoded string
		if err := dec.Decode(&decoded); err != nil {
			var syntaxErr *json.SyntaxError
			switch {
			case errors.Is(err, io.ErrUnexpectedEOF):
				failedUntil = len(s)
			case errors.As(err, &syntaxErr) && syntaxErr.Offset > 1:
				// Offset counts the byte that caused the error, which may
				// itself be a quote that starts a decodable fragment.
				failedUntil = i + int(syntaxErr.Offset) - 1
			}
			if startsLikeEncodedContainer(sub) {
				return true
			}
			continue
		}
		if _, changed, lost := redactJSONTextPass(sub[:dec.InputOffset()], depth+1, laterPass, budget); changed || lost {
			return true
		}
		if int(dec.InputOffset()) > 1 {
			i += int(dec.InputOffset()) - 1
		}
	}
	return false
}

// startsLikeEncodedContainer reports whether s, a quoted fragment that could
// not be decoded, starts like an encoded JSON container. Like adk-python, it
// looks only at the first character after the quote. A "[REDACTED]"
// placeholder starts none: only the result of an earlier pass gets here with
// one (see redactJSONTextPass), since text as it was received fails closed
// on any "[" before its fragments are read (see hasRedactableContainerText).
func startsLikeEncodedContainer(s string) bool {
	afterQuote := strings.TrimLeftFunc(s[1:], func(r rune) bool {
		return unicode.IsSpace(r) || r == '\ufeff'
	})
	if strings.HasPrefix(afterQuote, "[") && isCompleteRedactedPlaceholderAt(afterQuote, 0) {
		// A placeholder that an earlier pass wrote, as in =\"[REDACTED]\".
		return false
	}
	return strings.HasPrefix(afterQuote, "{") || strings.HasPrefix(afterQuote, "[") ||
		strings.HasPrefix(afterQuote, `\`) || strings.HasPrefix(afterQuote, `"`)
}

// redactJSONValuePass redacts sensitive keys in one pass over v (see
// redactJSONTextPass).
func redactJSONValuePass(v any, depth int, laterPass bool, budget *nodeBudget) (out any, changed, contentLost bool) {
	out, changed, contentLost, _ = redactJSONValueInner(v, depth, false, laterPass, budget)
	return out, changed, contentLost
}

func budgetTruncatedMap(outMap map[string]any, lastKey string, hasLastKey, rewriteExceeded bool) (any, bool, bool, bool) {
	if len(outMap) == 0 {
		return sanitizeBudgetPlaceholder, true, true, rewriteExceeded
	}
	if hasLastKey {
		delete(outMap, lastKey)
	}
	outMap[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
	return outMap, true, true, rewriteExceeded
}

func redactJSONValueInner(v any, depth int, inContainer, laterPass bool, budget *nodeBudget) (out any, changed, contentLost, rewriteExceeded bool) {
	if depth >= maxSanitizeDepth {
		return maxDepthPlaceholder, true, true, false
	}
	if budget != nil && !budget.take() {
		return sanitizeBudgetPlaceholder, true, true, false
	}
	switch x := v.(type) {
	case map[string]any:
		outMap := make(map[string]any, min(len(x), maxContainerAlloc))
		var lastKey string
		var hasLastKey bool
		for k, val := range x {
			if budget != nil && budget.remaining <= 0 {
				return budgetTruncatedMap(outMap, lastKey, hasLastKey, rewriteExceeded)
			}
			if isSensitiveKey(k) {
				if budget != nil && !budget.take() {
					return budgetTruncatedMap(outMap, lastKey, hasLastKey, rewriteExceeded)
				}
				if str, ok := val.(string); !ok || str != redactedValue {
					changed = true
				}
				outMap[k] = redactedValue
				lastKey, hasLastKey = k, true
				continue
			}
			newVal, c, lost, exc := redactJSONValueInner(val, depth+1, inContainer, laterPass, budget)
			if str, ok := newVal.(string); ok && str == sanitizeBudgetPlaceholder && budget != nil && budget.remaining <= 0 {
				return budgetTruncatedMap(outMap, lastKey, hasLastKey, rewriteExceeded || exc)
			}
			outMap[k] = newVal
			lastKey, hasLastKey = k, true
			changed = changed || c
			contentLost = contentLost || lost
			rewriteExceeded = rewriteExceeded || exc
		}
		return outMap, changed, contentLost, rewriteExceeded
	case []any:
		outSlice := make([]any, 0, min(len(x), maxContainerAlloc))
		for _, val := range x {
			if budget != nil && budget.remaining <= 0 {
				if len(outSlice) == 0 {
					return sanitizeBudgetPlaceholder, true, true, rewriteExceeded
				}
				outSlice[len(outSlice)-1] = sanitizeBudgetPlaceholder
				return outSlice, true, true, rewriteExceeded
			}
			newVal, c, lost, exc := redactJSONValueInner(val, depth+1, inContainer, laterPass, budget)
			outSlice = append(outSlice, newVal)
			changed = changed || c
			contentLost = contentLost || lost
			rewriteExceeded = rewriteExceeded || exc
			if str, ok := newVal.(string); ok && str == sanitizeBudgetPlaceholder && budget != nil && budget.remaining <= 0 {
				if len(outSlice) <= 1 {
					return sanitizeBudgetPlaceholder, true, true, rewriteExceeded
				}
				outSlice[len(outSlice)-2] = sanitizeBudgetPlaceholder
				return outSlice[:len(outSlice)-1], true, true, rewriteExceeded
			}
		}
		return outSlice, changed, contentLost, rewriteExceeded
	case string:
		san, c, lost, exc := redactJSONTextInner(x, depth+1, inContainer, true, laterPass, budget)
		return san, c, lost, exc
	default:
		return v, false, false, false
	}
}

func unmarshalJSONStrictWithBudget(s string, budget *nodeBudget) (val any, sawDuplicateKey, budgetExceeded bool, err error) {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	val, sawDuplicateKey, budgetExceeded, err = decodeJSONValue(dec, 0, budget)
	if err != nil || budgetExceeded {
		return nil, false, budgetExceeded, err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, false, false, errors.New("unexpected data after JSON value")
	}
	return val, sawDuplicateKey, false, nil
}

func decodeJSONValue(dec *json.Decoder, depth int, budget *nodeBudget) (any, bool, bool, error) {
	if depth > maxSanitizeDepth*2 {
		return nil, false, false, errors.New("JSON nesting exceeds limit")
	}
	if budget != nil && !budget.take() {
		return nil, false, true, errors.New("JSON node budget exceeded")
	}
	tok, err := dec.Token()
	if err != nil {
		return nil, false, false, err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return tok, false, false, nil
	}
	switch delim {
	case '{':
		m := make(map[string]any)
		sawDup := false
		for dec.More() {
			keyTok, err := dec.Token()
			if err != nil {
				return nil, false, false, err
			}
			key, ok := keyTok.(string)
			if !ok {
				return nil, false, false, errors.New("expected string key in JSON object")
			}
			val, childDup, exceeded, err := decodeJSONValue(dec, depth+1, budget)
			if err != nil || exceeded {
				return nil, false, exceeded, err
			}
			if _, exists := m[key]; exists {
				sawDup = true
			}
			sawDup = sawDup || childDup
			m[key] = val
		}
		endTok, err := dec.Token()
		if err != nil || endTok != json.Delim('}') {
			return nil, false, false, errors.New("unclosed JSON object")
		}
		return m, sawDup, false, nil
	case '[':
		var s []any
		sawDup := false
		for dec.More() {
			val, childDup, exceeded, err := decodeJSONValue(dec, depth+1, budget)
			if err != nil || exceeded {
				return nil, false, exceeded, err
			}
			sawDup = sawDup || childDup
			s = append(s, val)
		}
		endTok, err := dec.Token()
		if err != nil || endTok != json.Delim(']') {
			return nil, false, false, errors.New("unclosed JSON array")
		}
		if s == nil {
			s = []any{}
		}
		return s, sawDup, false, nil
	default:
		return nil, false, false, errors.New("unexpected JSON delimiter")
	}
}

// redactErrorMessage scrubs credentials from an error string before it is
// written to error_message: full or embedded JSON values, URIs with query,
// path or userinfo credentials, Authorization/API-key headers, Bearer/Basic
// authorization tokens, and sensitive key=value or key: value pairs.
func redactErrorMessage(msg string) string {
	out, settled := redactErrorMessageOnce(msg, false)
	for cur, pass := msg, 1; !settled && out != cur && !isScrubbedPlainText(out); pass++ {
		if pass == maxRedactionPasses {
			return redactedSensitiveTextPlaceholder
		}
		var next string
		next, settled = redactErrorMessageOnce(out, true)
		if next == out {
			break
		}
		cur, out = out, next
	}
	return out
}

// redactErrorMessageOnce is one pass of redactErrorMessage (see
// maxRedactionPasses). laterPass reports that msg is the result of an earlier
// pass (see redactJSONTextPass).
func redactErrorMessageOnce(msg string, laterPass bool) (out string, settled bool) {
	if msg == "" || isKnownSentinel(msg) {
		return msg, true
	}
	budget := &nodeBudget{remaining: maxSanitizeNodes}
	trimmed := trimBOMAndSpace(msg)
	if strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
		if decoded, sawDuplicateKey, exceeded, err := unmarshalJSONStrictWithBudget(trimmed, budget); err == nil && !exceeded {
			redacted, valChanged, valLost := redactJSONValuePass(decoded, 1, laterPass, budget)
			scrubbed, strChanged := scrubErrorJSONStrings(redacted, 1)
			if valChanged || sawDuplicateKey || valLost || strChanged {
				if b, err := json.Marshal(scrubbed); err == nil {
					return string(b), !strChanged && !valLost
				}
			}
			return msg, true
		}
		// Malformed JSON document: if it contains any sensitive key pattern or credential marker,
		// fail closed rather than falling through to partial regex rewriting.
		if hasAnyCredentialMarker(trimmed) || hasEncodedCredentialMarker(trimmed) || hasCandidateQuotedSensitiveKey(trimmed, 0) {
			return unparseableJSONPlaceholder, true
		}
		scrubbed := scrubErrorText(msg)
		return scrubbed, scrubbed == msg
	}
	if strings.HasPrefix(trimmed, `"`) {
		if redacted, ok, _ := redactJSONTextPass(msg, 0, laterPass, budget); ok {
			// As in sanitizeRawText, quote-led prose that only failed JSON
			// parsing is kept.
			if redacted == unparseableJSONPlaceholder && isPositivelyHarmlessProse(msg, trimmed, laterPass) {
				return msg, true
			}
			scrubbed := scrubErrorText(redacted)
			return scrubbed, scrubbed == redacted
		}
	}
	scrubbed := scrubErrorText(msg)
	return scrubbed, scrubbed == msg
}

func scrubErrorJSONStrings(v any, depth int) (any, bool) {
	if depth >= maxSanitizeDepth {
		return v, false
	}
	switch x := v.(type) {
	case map[string]any:
		changed := false
		out := make(map[string]any, len(x))
		for k, val := range x {
			newVal, c := scrubErrorJSONStrings(val, depth+1)
			out[k] = newVal
			changed = changed || c
		}
		return out, changed
	case []any:
		changed := false
		out := make([]any, len(x))
		for i, val := range x {
			newVal, c := scrubErrorJSONStrings(val, depth+1)
			out[i] = newVal
			changed = changed || c
		}
		return out, changed
	case string:
		if isKnownSentinel(x) {
			return x, false
		}
		scrubbed := scrubErrorText(x)
		return scrubbed, scrubbed != x
	default:
		return v, false
	}
}

// scrubErrorText redacts the credentials in msg with scrubErrorTokens, run
// until the text stops changing, and fails msg closed when decoding its
// escapes exposes a credential, when the scrub keeps a byte that the legacy
// expressions remove (see leavesLegacyValue), or when it rewrites a URI whose
// bytes it cannot map back to msg within the budget of the alignment (see
// uriAligner). Its result r is a placeholder, or text that scrubErrorTokens
// leaves unchanged and that passes those checks as r itself, so that
// scrubErrorText(r) == r.
func scrubErrorText(msg string) string {
	var m scrubMemo
	return m.scrubText(msg)
}

// scrubURIComponentText is scrubErrorText for one decoded URI component (see
// redactURIComponent): the same passes and checks, with scrubTextPatterns and
// the legacy expressions over the component as text (see
// legacyScrubComponentText) in place of scrubErrorTokens and the legacy
// scrubs of whole texts, so that a URI in it is not parsed again, which keeps
// the work linear. Its result r is a placeholder or a fixed point, so that
// scrubbing the component of a sanitized URI again changes nothing.
func scrubURIComponentText(s string) string {
	m := scrubMemo{component: true}
	return m.scrubText(s)
}

// scrubText is scrubErrorText with the scrubs of m.
func (m *scrubMemo) scrubText(msg string) string {
	first := m.tokenRuns.do(msg, m.tokenPass())
	scrubbed := first.out
	if scrubbed == redactedSensitiveTextPlaceholder || first.trace.unaligned {
		return redactedSensitiveTextPlaceholder
	}
	if exposesEncodedPattern(scrubbed, m.tokens) || hasBackslashEscapedCredentialWith(msg, scrubbed, m.tokens) {
		return redactedSensitiveTextPlaceholder
	}
	if scrubbed == msg {
		if m.leavesLegacyValue(msg, scrubbed, identityRuns(len(msg))) {
			return redactedSensitiveTextPlaceholder
		}
		return scrubbed
	}
	// Scrub the scrubbed text again until it no longer changes (see
	// maxRedactionPasses), and check the result as a second scrub of it
	// would, so that scrubbing it again changes nothing. kept maps the bytes
	// of the result copied from msg back to msg.
	kept := first.trace.runs
	for prev, pass := msg, 1; scrubbed != prev; pass++ {
		if pass == maxRedactionPasses {
			return redactedSensitiveTextPlaceholder
		}
		next := m.tokenRuns.do(scrubbed, m.tokenPass())
		prev, scrubbed = scrubbed, next.out
		if scrubbed == redactedSensitiveTextPlaceholder || next.trace.unaligned {
			return redactedSensitiveTextPlaceholder
		}
		if scrubbed != prev {
			kept = composeRuns(next.trace.runs, kept)
		}
	}
	if exposesEncodedPattern(scrubbed, m.tokens) || hasBackslashEscapedCredentialWith(scrubbed, scrubbed, m.tokens) ||
		m.leavesLegacyValue(msg, scrubbed, kept) || m.leavesLegacyValue(scrubbed, scrubbed, identityRuns(len(scrubbed))) {
		return redactedSensitiveTextPlaceholder
	}
	return scrubbed
}

// scrubMemo remembers, within one scrubErrorText, what scrubErrorTokens and
// the two legacy scrubs (see leavesLegacyValue) returned for the last texts
// they were given, with their traces (see passTrace). Its checks scrub the
// same texts, and the same decoded texts, more than once, and each scrub
// reads the whole text.
type scrubMemo struct {
	// component selects the scrubs of one decoded URI component, which read
	// no URI (see scrubURIComponentText).
	component                           bool
	tokenRuns, legacyRuns, compoundRuns memoRuns
	// components remembers what the scrubs of the components of the URIs
	// in the texts returned for each component (see componentScrub).
	components map[componentKey]componentResult
}

// componentKind tells which scrub of a decoded URI component a componentKey
// holds the result of: redactURIComponent, legacyRedactURIComponent or
// compoundLegacyRedactURIComponent.
type componentKind uint8

const (
	scannerComponent componentKind = iota
	legacyComponent
	compoundLegacyComponent
)

type componentKey struct {
	kind componentKind
	s    string
}

type componentResult struct {
	out     string
	changed bool
}

// componentScrub returns scrub, the scrub of a decoded URI component of
// kind, remembering its result for each component in m. The checks of one
// scrubErrorText read the URIs of a text again, decoded and scrubbed, and
// with each legacy scrub, and the pairs of a URI may repeat, so that each
// component would be scrubbed many times; the scrubs depend on the
// component alone.
func (m *scrubMemo) componentScrub(kind componentKind, scrub uriComponentScrub) uriComponentScrub {
	return func(s string) (string, bool) {
		k := componentKey{kind, s}
		if r, ok := m.components[k]; ok {
			return r.out, r.changed
		}
		out, changed := scrub(s)
		if m.components == nil {
			m.components = make(map[componentKey]componentResult)
		}
		m.components[k] = componentResult{out, changed}
		return out, changed
	}
}

// memoEntry is the result of one scrub of in, with its trace.
type memoEntry struct {
	in, out string
	trace   passTrace
}

// memoRuns holds the last scrubs of one function.
type memoRuns struct {
	e [4]memoEntry
	n int
}

func (r *memoRuns) do(s string, scrub tracedPass) memoEntry {
	for k := range min(r.n, len(r.e)) {
		if r.e[k].in == s {
			return r.e[k]
		}
	}
	e := memoEntry{in: s}
	e.out = scrub(s, &e.trace)
	r.e[r.n%len(r.e)] = e
	r.n++
	return e
}

// tokenPass returns the scrub of m: scrubErrorTokens, or scrubTextPatterns
// for one decoded URI component.
func (m *scrubMemo) tokenPass() tracedPass {
	if m.component {
		return scrubTextPatterns
	}
	scrub := m.componentScrub(scannerComponent, redactURIComponent)
	return func(s string, tr *passTrace) string { return scrubErrorTokensTracedWith(s, scrub, tr) }
}

func (m *scrubMemo) tokens(s string) string {
	return m.tokenRuns.do(s, m.tokenPass()).out
}

func (m *scrubMemo) legacy(s string) string {
	return m.legacyRuns.do(s, m.legacyPass(false)).out
}

func (m *scrubMemo) compound(s string) string {
	return m.compoundRuns.do(s, m.legacyPass(true)).out
}

// keptLiterals returns the non-redacted literal slices left in scrubbed after
// removing redaction placeholders, in order.
func keptLiterals(scrubbed string) []string {
	parts := []string{scrubbed}
	for _, sentinel := range []string{redactedURIPlaceholder, redactedValue, url.QueryEscape(redactedValue)} {
		var next []string
		for _, part := range parts {
			for piece := range strings.SplitSeq(part, sentinel) {
				if piece != "" {
					next = append(next, piece)
				}
			}
		}
		parts = next
	}
	return parts
}

// keptInOrder reports whether every part appears in s in order.
func keptInOrder(parts []string, s string) bool {
	rest := s
	for _, part := range parts {
		i := strings.Index(rest, part)
		if i < 0 {
			return false
		}
		rest = rest[i+len(part):]
	}
	return true
}

func unescapeCredentialStep(s string) string {
	return unescapeCredentialStepWith(s, canonicalizeCredentialMarker)
}

func unescapeCredentialStepWith(s string, canonFn func(string) string) string {
	if strings.Contains(s, `\`) {
		s = unescapeBackslashControls(s)
	}
	if strings.ContainsAny(s, "%xXuU") {
		s = canonFn(s)
	}
	return s
}

// hasBackslashEscapedCredentialWith reports whether unescaping standard
// backslash escapes (\b, \f, \n, \r, \t, \/, \", \' and \\), or
// canonicalizing percent, \x or \u escapes, exposes a credential that scrub
// catches, as is or through exposesEncodedPattern, including when scrub on raw
// consumed an escape inside an earlier value and left a following secret in
// scrubbed, the result of scrub on raw. scrub is scrubErrorTokens or a scrub
// that scrubText or leavesLegacyValue uses in its place. Each pass decodes a
// run of backslashes in one step (see unescapeBackslashControls), so passes
// are needed only where percent, \x or \u escapes and backslash escapes nest
// inside each other; up to 2*maxCanonicalUnescapePasses are run, failing
// closed if escapes still change past that depth.
func hasBackslashEscapedCredentialWith(raw, scrubbed string, scrub func(string) string) bool {
	return hasBackslashEscapedCredentialFull(raw, scrubbed, scrub, canonicalizeCredentialMarker)
}

// hasLegacyBackslashEscapedCredentialWith is hasBackslashEscapedCredentialWith
// using the merge-base ASCII-only canonicalizer so the legacy parity backstop
// preserves merge-base escape semantics.
func hasLegacyBackslashEscapedCredentialWith(raw, scrubbed string, scrub func(string) string) bool {
	return hasBackslashEscapedCredentialFull(raw, scrubbed, scrub, canonicalizeLegacyCredentialMarker)
}

func hasBackslashEscapedCredentialFull(raw, scrubbed string, scrub, canonFn func(string) string) bool {
	var parts []string
	curRaw, curScrubbed := raw, scrubbed
	for range 2 * maxCanonicalUnescapePasses {
		if !strings.ContainsAny(curRaw, `\%`) {
			return false
		}
		nextRaw := unescapeCredentialStepWith(curRaw, canonFn)
		if nextRaw == curRaw {
			return false
		}
		if curScrubbed == curRaw {
			// Nothing was redacted, so the scrubbed text decodes to nextRaw
			// too and one scrub of it answers both checks below.
			if scrub(nextRaw) != nextRaw || exposesEncodedPatternWith(nextRaw, scrub, canonFn) {
				return true
			}
			curRaw, curScrubbed = nextRaw, nextRaw
			continue
		}
		if parts == nil {
			parts = keptLiterals(curScrubbed)
		}
		nextScrubbed := unescapeCredentialStepWith(curScrubbed, canonFn)
		for j, part := range parts {
			parts[j] = unescapeCredentialStepWith(part, canonFn)
		}
		if scrub(nextScrubbed) != nextScrubbed || exposesEncodedPatternWith(nextScrubbed, scrub, canonFn) {
			return true
		}
		if !keptInOrder(parts, scrub(nextRaw)) {
			return true
		}
		curRaw, curScrubbed = nextRaw, nextScrubbed
	}
	return unescapeCredentialStepWith(curRaw, canonFn) != curRaw
}

// unescapeBackslashControls decodes the standard backslash escapes in s. A
// run of backslashes, which each level of nested escaping doubles, is decoded
// in one step as the escape its last backslash starts, so that the depth of
// the nesting does not set the number of passes.
func unescapeBackslashControls(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		if s[i] != '\\' {
			// Copy the run of bytes up to the next backslash at once.
			n := strings.IndexByte(s[i:], '\\')
			if n < 0 {
				n = len(s) - i
			}
			b.WriteString(s[i : i+n])
			i += n - 1
			continue
		}
		j := i + 1
		for j < len(s) && s[j] == '\\' {
			j++
		}
		if j == len(s) {
			b.WriteByte('\\')
			break
		}
		switch c := s[j]; c {
		case 'b':
			b.WriteByte('\b')
		case 'f':
			b.WriteByte('\f')
		case 'n':
			b.WriteByte('\n')
		case 'r':
			b.WriteByte('\r')
		case 't':
			b.WriteByte('\t')
		case '/', '"', '\'':
			b.WriteByte(c)
		default:
			b.WriteByte('\\')
			b.WriteByte(c)
		}
		i = j
	}
	return b.String()
}

func hasEncodedCredentialMarker(msg string) bool {
	return exposesEncodedPattern(msg, scrubErrorTokens)
}

// exposesEncodedPattern reports whether decoding the escapes in msg exposes a
// pattern that scrub would redact.
func exposesEncodedPattern(msg string, scrub func(string) string) bool {
	return exposesEncodedPatternWith(msg, scrub, canonicalizeCredentialMarker)
}

// exposesLegacyEncodedPattern is exposesEncodedPattern using the merge-base
// ASCII-only canonicalizer so the legacy parity backstop preserves merge-base
// escape semantics.
func exposesLegacyEncodedPattern(msg string, scrub func(string) string) bool {
	return exposesEncodedPatternWith(msg, scrub, canonicalizeLegacyCredentialMarker)
}

func exposesEncodedPatternWith(msg string, scrub, canonFn func(string) string) bool {
	if msg == "" || isKnownSentinel(msg) {
		return false
	}
	cleaned := strings.ReplaceAll(msg, "%5BREDACTED%5D", redactedValue)
	cleaned = strings.ReplaceAll(cleaned, "%5bredacted%5d", redactedValue)
	canon := canonFn(cleaned)
	if canon == cleaned {
		return false
	}
	return scrub(canon) != canon
}

// hasAnyCredentialMarker reports whether s holds a credential that the
// scanners or the legacy expressions redact (see legacyScrubChanges), or a
// quoted sensitive key, possibly escaped.
func hasAnyCredentialMarker(s string) bool {
	if s == "" {
		return false
	}
	if scrubErrorText(s) != s || legacyScrubChanges(s) {
		return true
	}
	if hasCandidateQuotedSensitiveKey(s, 0) {
		return true
	}
	canon := canonicalizeCredentialMarker(s)
	if hasQuotedSensitiveKey(canon) {
		return true
	}
	return canon != s && scrubErrorTokens(canon) != canon
}

// applyContentFormatter runs formatter on payload. It fails closed: if the
// formatter returns an error or panics, it returns the "[FORMATTER_FAILED]"
// placeholder, a safe type-only diagnostic note for error_message, and true.
// Raw error strings or panic values are never logged because they may contain
// unredacted payload data.
func applyContentFormatter(formatter ContentFormatter, logger Logger, payload any, eventType string, debugErrors bool) (result any, errNote string, failed bool) {
	if formatter == nil {
		return payload, "", false
	}
	defer func() {
		if r := recover(); r != nil {
			if logger != nil {
				logger.Printf("agentanalytics: ContentFormatter panicked for event %s (type=%T); content replaced", eventType, r)
			}
			result, errNote, failed = formattingFailedPlaceholder, "content_formatter panicked", true
		}
	}()
	formatted, err := formatter(payload, eventType)
	if err != nil {
		if logger != nil {
			if debugErrors {
				logger.Printf("agentanalytics: ContentFormatter failed for event %s (type=%T, err_len=%d); content replaced", eventType, err, len(err.Error()))
			} else {
				logger.Printf("agentanalytics: ContentFormatter failed for event %s (%T); content replaced", eventType, err)
			}
		}
		return formattingFailedPlaceholder, fmt.Sprintf("content_formatter raised %T", err), true
	}
	return formatted, "", false
}

// Payload columns that PayloadColumnDenylist may suppress.
const (
	columnContent      = "content"
	columnContentParts = "content_parts"
	columnAttributes   = "attributes"
	columnLatencyMS    = "latency_ms"
)

// parsePayloadColumnDenylist validates cols and returns them as a set.
func parsePayloadColumnDenylist(cols []string) (map[string]struct{}, error) {
	denied := make(map[string]struct{}, len(cols))
	for _, c := range cols {
		switch c {
		case columnContent, columnContentParts, columnAttributes, columnLatencyMS:
			denied[c] = struct{}{}
		default:
			return nil, fmt.Errorf("unsupported PayloadColumnDenylist entry %q: allowed values are %q, %q, %q and %q",
				c, columnContent, columnContentParts, columnAttributes, columnLatencyMS)
		}
	}
	return denied, nil
}

// customMetadataFilter selects CustomMetadata keys by exact name or by prefix
// (entries ending in "*").
type customMetadataFilter struct {
	exact    map[string]struct{}
	prefixes []string
}

func newCustomMetadataFilter(allowlist []string) customMetadataFilter {
	f := customMetadataFilter{exact: make(map[string]struct{})}
	for _, entry := range allowlist {
		if prefix, ok := strings.CutSuffix(entry, "*"); ok {
			f.prefixes = append(f.prefixes, prefix)
		} else {
			f.exact[entry] = struct{}{}
		}
	}
	return f
}

func (f customMetadataFilter) allows(key string) bool {
	if _, ok := f.exact[key]; ok {
		return true
	}
	for _, p := range f.prefixes {
		if strings.HasPrefix(key, p) {
			return true
		}
	}
	return false
}

// apply returns the allowed entries of md, or nil if there are none.
func (f customMetadataFilter) apply(md map[string]any) map[string]any {
	var out map[string]any
	for k, v := range md {
		if !f.allows(k) {
			continue
		}
		if out == nil {
			out = make(map[string]any)
		}
		out[k] = v
	}
	return out
}
