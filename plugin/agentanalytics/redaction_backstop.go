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
	"strings"
	"unicode"
	"unicode/utf8"

	"google.golang.org/adk/v2/session"
)

// The legacy-pattern backstop. Earlier versions of this package redacted text
// with regular expressions: Authorization and API-key header lines, bearer
// and basic credentials, quoted JSON key-value pairs, and key=value or
// key: value pairs of the sensitiveKeys names (each "_" also written as "-"
// or left out) and of "temp:" keys, leaving a value that is itself the name
// of a nested pair for that pair, and failing text closed when decoding its
// escapes changed what they redacted (see hasBackslashEscapedCredentialWith).
// The scanners that replaced them also read compound and state-prefixed keys
// and nested keys that the expressions did not, and so take other paths
// through the text. legacyScrubErrorTokens redoes what the expressions did,
// with linear-time matchers in place of the expressions, and
// compoundLegacyScrubErrorTokens does the same with every key that
// isSensitiveKey accepts. Both record the spans of the text that they remove
// (see spanWriter), and leavesLegacyValue checks the scanners' result against
// them byte by byte: text in which the scanners keep a byte that either
// removes, or that either fails closed while the scanners keep a byte of it,
// fails closed. So whatever the expressions removed stays removed, whatever
// path the scanners take through quotes, escapes and nested keys, and
// whatever the removed bytes spell: punctuation, or a word that also stands
// elsewhere in the text, counts as any other byte.

// leavesLegacyValue reports whether scrubbed, the scrub of raw whose bytes
// copied from raw kept maps back to raw (see copyRun), keeps a byte that the
// legacy expressions remove from raw, counting every byte as removed when they
// fail raw closed (see scrubErrorText and keepsRemovedByte).
func (m *scrubMemo) leavesLegacyValue(raw, scrubbed string, kept []copyRun) bool {
	if !strings.ContainsAny(raw, `\%`) && !mayMatchTextPatterns(raw) {
		return false
	}
	var c *spanCheck
	for _, compound := range []bool{false, true} {
		legacy, removed := m.legacyRemoved(raw, compound)
		if legacy == raw || legacy == scrubbed {
			continue
		}
		if c == nil {
			c = newSpanCheck(raw, scrubbed, kept)
		}
		if c.keepsRemovedByte(removed, compound) {
			return true
		}
	}
	return false
}

// legacyRemoved returns the legacy scrub of raw (see legacyScrubErrorTokens,
// or compoundLegacyScrubErrorTokens when compound is true) and the spans of
// raw that it removes: all of raw when decoding an escape in it exposes a
// pattern that the scrub redacts, as the expressions on main failed such text
// closed.
func (m *scrubMemo) legacyRemoved(raw string, compound bool) (string, [][2]int) {
	scrub := m.legacy
	runs := &m.legacyRuns
	if compound {
		scrub = m.compound
		runs = &m.compoundRuns
	}
	e := runs.do(raw, m.legacyPass(compound))
	if exposesLegacyEncodedPattern(e.out, scrub) || hasLegacyBackslashEscapedCredentialWith(raw, e.out, scrub) {
		return redactedSensitiveTextPlaceholder, [][2]int{{0, len(raw)}}
	}
	return e.out, e.trace.removed
}

// legacyPass returns the legacy scrub of m (see legacyScrubTokensTraced).
func (m *scrubMemo) legacyPass(compound bool) tracedPass {
	if m.component {
		return func(s string, tr *passTrace) string { return legacyScrubComponentTextTraced(s, compound, tr) }
	}
	kind, scrub := legacyComponent, legacyRedactURIComponent
	if compound {
		kind, scrub = compoundLegacyComponent, compoundLegacyRedactURIComponent
	}
	scrub = m.componentScrub(kind, scrub)
	return func(s string, tr *passTrace) string { return legacyScrubTokensTracedWith(s, compound, scrub, tr) }
}

// redactedKeyBytes marks the bytes of s that belong to a sensitive key whose
// value is a redaction placeholder ("[REDACTED]" or, for a URI with
// credentials, "[REDACTED_SENSITIVE_URI]"): the key with its quotes, possibly
// escaped, and padding, and the delimiter, whitespace and opening quote of
// the value before the placeholder.
// When only the end of the name is a sensitive key, as the token of
// x/token:[REDACTED] or of "x/token":[REDACTED], only that end is marked:
// the words before it may be the value of an earlier pair, on a line of its
// own too. So is the fixed key at the end of a compound or "temp:" name
// within a line (see exemptKeyStart), as the token of api_key: [REDACTED]
// sk_v1_token: [REDACTED].
func redactedKeyBytes(s string) []bool {
	return redactedKeyBytesWith(s, false)
}

// redactedKeyBytesWith is redactedKeyBytes, which, when compoundKeys is
// true, marks a compound or "temp:" name within a line as a whole, as
// compoundLegacyScrubErrorTokens, which reads such names as keys, does, so
// that the name of a nested pair in the redacted value of an earlier pair,
// as the "<long>-password" of dbPassword=[REDACTED]/"<long>-password":
// "[REDACTED]", does not count against it. Against the expressions on main,
// which read only fixed keys, only the fixed key at the end of such a name
// counts.
func redactedKeyBytesWith(s string, compoundKeys bool) []bool {
	marked := make([]bool, len(s))
	for p := 0; p < len(s); {
		i := strings.IndexByte(s[p:], '[')
		if i < 0 {
			break
		}
		p += i
		n := 0
		if strings.HasPrefix(s[p:], redactedValue) {
			n = len(redactedValue)
		} else if strings.HasPrefix(s[p:], redactedURIPlaceholder) {
			n = len(redactedURIPlaceholder)
		}
		if n == 0 {
			p++
			continue
		}
		phStart := p
		k := p
		for k > 0 && (isQuoteByte(s[k-1]) || s[k-1] == '\\' || isDelimSpace(s[k-1])) {
			k--
		}
		p += n
		if k == 0 || (s[k-1] != ':' && s[k-1] != '=') {
			continue
		}
		end := k - 1
		for end > 0 && (s[end-1] == ' ' || s[end-1] == '\t') {
			end--
		}
		start, ok := redactedKeyStart(s, end)
		if !ok {
			continue
		}
		if end > start && isQuoteByte(s[end-1]) {
			end--
		}
		key := strings.TrimRight(trimInlineSpaces(s[start:end]), `\`)
		if key == "" {
			continue
		}
		keyAt := strings.Index(s[start:end], key) + start
		from := keyAt
		if isSensitiveKey(key) {
			if !compoundKeys && !startsLineAt(s, start) {
				// Within a line, only the fixed key at the end of a compound
				// or "temp:" name counts, as the token of
				// sk_v1_token:[REDACTED] (see exemptKeyStart).
				off := exemptKeyStart(key)
				if off < 0 {
					continue
				}
				from = keyAt + off
			}
		} else {
			// Only the longest name after a "/", "\", ":" or "=" that is a
			// key counts, on a line of its own too: the scanners keep the
			// words before the key of such a name, as the /etc/ssl/ of
			// credentials:\n  /etc/ssl/password: v, only where the
			// expressions on main read the name as a nested key and kept them
			// (see mainReadsNestedKey), so that none of them is removed.
			at := -1
			for j := len(key) - 1; j > 0 && len(key)-j <= maxNestedKeyLen; j-- {
				if strings.IndexByte(`/\:=`, key[j-1]) >= 0 && isSensitiveKey(key[j:]) {
					at = j
				}
			}
			if at < 0 {
				continue
			}
			from = keyAt + at
			if !compoundKeys {
				off := exemptKeyStart(key[at:])
				if off < 0 {
					continue
				}
				from += off
			}
		}
		if from == keyAt {
			// The whole name counts, with its padding and opening quote,
			// possibly escaped.
			from = start
			if from > 0 && isQuoteByte(s[from-1]) {
				from--
				if from > 0 && s[from-1] == '\\' {
					from--
				}
			}
		}
		for b := from; b < phStart; b++ {
			marked[b] = true
		}
	}
	return marked
}

// exemptKeyStart returns where the part of key, a sensitive key name within
// a line, that the backstop reads as a key starts: key as a whole when it is
// a fixed sensitive key, possibly after a "user:" or "app:" prefix, as
// user:private.key is, or else only the fixed key at its end (see
// fixedKeySuffixStart), as the token of sk_v1_token, or -1 when it ends in
// none, as temp:v1 does: the words before that key may be the value of an
// earlier pair or header, as the v1 of
// api_key: [REDACTED] temp:v1:[REDACTED], which the expressions on main
// redacted with the header.
func exemptKeyStart(key string) int {
	if kind, _ := nestedKeyName(key, 0, false); kind == nestedKey {
		return 0
	}
	return fixedKeySuffixStart(key)
}

// startsLineAt reports whether only spaces or tabs, and an opening quote,
// possibly escaped, stand between the start of a line of s and s[i], as
// before the DB_PASSWORD of credentials:\n  \"DB_PASSWORD\"=... (see
// redactedKeyStart, which starts a quoted name right after its opening
// quote). Only one backslash is stepped over, so a name after an escaped
// backslash, as in \\"DB_PASSWORD", does not start a line.
func startsLineAt(s string, i int) bool {
	if i > 0 && isQuoteByte(s[i-1]) {
		i--
		if i > 0 && s[i-1] == '\\' {
			i--
		}
	}
	for i > 0 && (s[i-1] == ' ' || s[i-1] == '\t') {
		i--
	}
	return i == 0 || s[i-1] == '\n' || s[i-1] == '\r'
}

// redactedKeyStart returns where the name of the key that ends at s[end],
// before the delimiter of a redacted pair, starts. A quoted name, as in
// "token":[REDACTED], starts after its opening quote on the same line and
// holds no other quote, whitespace but padding, placeholder or value stop
// (see isTextValueStop). An unquoted name runs back to whitespace, a control
// byte, a non-ASCII space, a value stop, a bracket or a quote, or a backslash
// escape of a control byte, as the \n of SECRET\ntoken, which separates
// words.
func redactedKeyStart(s string, end int) (int, bool) {
	if end > 0 && isQuoteByte(s[end-1]) {
		q := s[end-1]
		open := end - 2
		for open >= 0 && s[open] != q {
			if c := s[open]; c == '\r' || c == '\n' || isQuoteByte(c) || c == '[' || c == ']' || c == ',' || c == ';' || c == '&' {
				return 0, false
			}
			open--
		}
		if open < 0 {
			return 0, false
		}
		if inner := trimInlineSpaces(strings.TrimRight(s[open+1:end-1], `\`)); strings.IndexFunc(inner, func(r rune) bool {
			return unicode.IsSpace(r) || unicode.In(r, unicode.Z)
		}) >= 0 {
			return 0, false
		}
		return open + 1, true
	}
	start := end
	for start > 0 {
		c := s[start-1]
		if isDelimSpace(c) || isTextValueStop(c) || strings.IndexByte("{}[]()\"'", c) >= 0 ||
			(start >= 2 && s[start-2] == '\\' && strings.IndexByte("bfnrt", c) >= 0) {
			break
		}
		if c >= utf8.RuneSelf {
			if r, size := utf8.DecodeLastRuneInString(s[:start]); unicode.IsSpace(r) || unicode.In(r, unicode.Z) {
				break
			} else {
				start -= size
				continue
			}
		}
		start--
	}
	return start, true
}

// legacyKeyNames holds the lowercased names in sensitiveKeys, but for the
// secret_key and signing_key that came with the scanners, with each "_" kept,
// written as "-" or left out: the fixed key names of the expressions. Unlike
// bareFixedSensitiveKeys it has no "." variants.
var legacyKeyNames = func() map[string]struct{} {
	m := make(map[string]struct{}, 256)
	for k := range sensitiveKeys {
		if k == "secret_key" || k == "signing_key" {
			continue
		}
		parts := strings.Split(k, "_")
		variants := []string{parts[0]}
		for _, p := range parts[1:] {
			next := make([]string, 0, len(variants)*3)
			for _, v := range variants {
				next = append(next, v+"_"+p, v+"-"+p, v+p)
			}
			variants = next
		}
		for _, v := range variants {
			m[v] = struct{}{}
		}
	}
	return m
}()

var maxLegacyKeyLen = func() int {
	n := 0
	for k := range legacyKeyNames {
		n = max(n, len(k))
	}
	return n
}()

// isLegacyFixedKey reports whether k is one of legacyKeyNames, ignoring case
// as the expressions did (see legacyFoldAt).
func isLegacyFixedKey(k string) bool {
	if len(k) == 0 || len(k) > 2*maxLegacyKeyLen || strings.ContainsRune(k, 'İ') {
		return false
	}
	_, ok := legacyKeyNames[foldKeyASCII(k)]
	return ok
}

// legacyFoldAt returns the lowercase ASCII letter or other ASCII byte that
// the character at s[i] matches in a case-insensitive expression, and the
// length of that character in bytes. Besides ASCII letters, RE2 (?i) matches
// U+017F (long s) to "s" and U+212A (Kelvin sign) to "k", but not U+0130
// (dotted capital I) to "i", which isSensitiveKey reads so. Any other
// non-ASCII character returns 0.
func legacyFoldAt(s string, i int) (byte, int) {
	c := s[i]
	if c < utf8.RuneSelf {
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		return c, 1
	}
	r, size := utf8.DecodeRuneInString(s[i:])
	switch r {
	case 'ſ':
		return 's', size
	case 'K':
		return 'k', size
	}
	return 0, size
}

// isLegacyLetterAt reports whether a letter, as legacyFoldAt reads it,
// starts at s[i].
func isLegacyLetterAt(s string, i int) bool {
	if c := s[i]; c < utf8.RuneSelf {
		c |= 0x20
		return 'a' <= c && c <= 'z'
	}
	c, _ := legacyFoldAt(s, i)
	return c != 0
}

// legacyFoldPrefixEnd returns where word, in lowercase ASCII, ends if it
// starts at s[i], ignoring case as legacyFoldAt does, or -1.
func legacyFoldPrefixEnd(s string, i int, word string) int {
	for k := 0; k < len(word); k++ {
		if i >= len(s) {
			return -1
		}
		c, size := legacyFoldAt(s, i)
		if c != word[k] {
			return -1
		}
		i += size
	}
	return i
}

// isLegacyBoundary reports whether a word boundary of the expressions (\b)
// holds before s[i], where a word starts. RE2 reads only ASCII letters,
// digits and "_" as word characters, so before a non-ASCII letter that folds
// onto an ASCII one, as the long s of a key that starts with one, the
// boundary holds only after a word character.
func isLegacyBoundary(s string, i int) bool {
	if i < len(s) && s[i] >= utf8.RuneSelf {
		return i > 0 && isWordByte(s[i-1])
	}
	return i == 0 || !isWordByte(s[i-1])
}

// legacyScrubErrorTokens is scrubErrorTokens as the legacy expressions did
// it, including in the components of the URIs in msg, which the expressions
// on main scrubbed as text too (see legacyRedactURIComponent).
func legacyScrubErrorTokens(msg string) string {
	return legacyScrubTokensTraced(msg, false, nil)
}

// compoundLegacyScrubErrorTokens is legacyScrubErrorTokens with the
// expressions also matching every key that isSensitiveKey accepts, such as
// DB_PASSWORD, dbPassword or api.key, as a whole, but for a path-prefixed
// ".key" file name before the ":" of a Go path error (see
// isSkippedPathDottedPair).
func compoundLegacyScrubErrorTokens(msg string) string {
	return legacyScrubTokensTraced(msg, true, nil)
}

// legacyScrubTokensTraced is legacyScrubErrorTokens, or
// compoundLegacyScrubErrorTokens when compound is true, recording in tr, when
// it is not nil, the runs of the result copied from msg and the spans of msg
// that it removes: the bytes of each URI that its rewrite does not keep (see
// scrubURITokens), and each value, header value or credential that the
// expressions replaced.
func legacyScrubTokensTraced(msg string, compound bool, tr *passTrace) string {
	scrub := legacyRedactURIComponent
	if compound {
		scrub = compoundLegacyRedactURIComponent
	}
	return legacyScrubTokensTracedWith(msg, compound, scrub, tr)
}

// legacyScrubTokensTracedWith is legacyScrubTokensTraced with scrub, which
// returns what legacyRedactURIComponent, or compoundLegacyRedactURIComponent
// when compound is true, does, in its place (see scrubMemo.componentScrub).
func legacyScrubTokensTracedWith(msg string, compound bool, scrub uriComponentScrub, tr *passTrace) string {
	keys := legacyURIKeys
	if compound {
		keys = compoundLegacyURIKeys
	}
	c := startPasses(msg, tr)
	c.run(func(s string, tr *passTrace) string { return scrubURITokens(s, scrub, keys, tr) })
	c.run(func(s string, tr *passTrace) string { return legacyScrubComponentTextTraced(s, compound, tr) })
	return c.end(tr)
}

// legacyScrubComponentText is legacyScrubErrorTokens without reading URIs,
// for one decoded URI component or text whose URIs are scrubbed already.
func legacyScrubComponentText(s string) string {
	return legacyScrubComponentTextTraced(s, false, nil)
}

// compoundLegacyScrubComponentText is compoundLegacyScrubErrorTokens without
// reading URIs (see legacyScrubComponentText).
func compoundLegacyScrubComponentText(s string) string {
	return legacyScrubComponentTextTraced(s, true, nil)
}

// legacyScrubTextPatterns is scrubTextPatterns as the legacy expressions did
// it, with every key that isSensitiveKey accepts when compound is set.
func legacyScrubTextPatterns(s string, compound bool) string {
	return legacyScrubComponentTextTraced(s, compound, nil)
}

// legacyScrubComponentTextTraced is legacyScrubComponentText, or
// compoundLegacyScrubComponentText when compound is true, recording what it
// does in tr as legacyScrubTokensTraced does.
func legacyScrubComponentTextTraced(s string, compound bool, tr *passTrace) string {
	c := startPasses(s, tr)
	if !mayMatchTextPatterns(s) {
		return c.end(tr)
	}
	budget := 4*len(s) + 4096
	c.run(legacyRedactHeaders)
	for _, m := range []struct {
		starts string
		match  legacyMatcher
	}{
		{"bB", (*legacyText).bearerAt},
		{"bB", (*legacyText).basicAt},
		{`"`, (*legacyText).jsonPairAt},
		{"", (*legacyText).textPairAt},
	} {
		c.run(func(s string, tr *passTrace) string {
			return legacyReplace(s, tr, &budget, compound, m.starts, m.match)
		})
	}
	return c.end(tr)
}

// legacyScrubChanges reports whether the legacy expressions redact a part of
// s (see legacyScrubErrorTokens). Malformed JSON text and the text after a
// quoted string fail closed when a scrub changes them (see
// hasAnyCredentialMarker and isPositivelyHarmlessProse), and the expressions
// changed text that the scanners keep, such as an Authorization or API-key
// header whose value starts on a following line and reads as redacted, as
// in {api_key:\n  "[REDACTED]" "SECRET"}, which the header expression
// rewrote, or a pair the scanners read as nested whose value the expressions
// redacted. Such text still fails closed.
func legacyScrubChanges(s string) bool {
	return legacyScrubErrorTokens(s) != s
}

// legacySensitiveKeyNames holds the names in sensitiveKeyNames but for the
// secret_key and signing_key that came with the scanners: the names that
// isSensitiveKey read on main.
var legacySensitiveKeyNames = func() map[string]struct{} {
	m := make(map[string]struct{}, len(sensitiveKeys))
	for k := range sensitiveKeys {
		if k != "secret_key" && k != "signing_key" {
			m[strings.ReplaceAll(k, "_", "")] = struct{}{}
		}
	}
	return m
}()

// isLegacySensitiveKey is isSensitiveKey as it was on main: a "temp:" key,
// or a name in legacySensitiveKeyNames after a "user:" or "app:" prefix,
// compared ignoring case and the "_" and "-" separators, but not compound
// names or "." separators.
func isLegacySensitiveKey(k string) bool {
	n := strings.ToLower(strings.TrimSpace(k))
	if strings.HasPrefix(n, session.KeyPrefixTemp) {
		return true
	}
	if rest, ok := strings.CutPrefix(n, session.KeyPrefixUser); ok {
		n = rest
	} else {
		n = strings.TrimPrefix(n, session.KeyPrefixApp)
	}
	n = strings.Map(func(r rune) rune {
		if r == '_' || r == '-' {
			return -1
		}
		return r
	}, strings.TrimSpace(n))
	_, ok := legacySensitiveKeyNames[n]
	return ok
}

// legacyHeaderNames are the header names of the expressions, longest first.
var legacyHeaderNames = [...]string{
	"proxy-authorization", "proxy_authorization", "authorization",
	"x-api-key", "x-api_key", "x_api-key", "x_api_key", "api-key", "api_key",
}

// legacyRedactHeaders replaces each header name followed by optional spaces
// or tabs and ":" with the name, ": " and the placeholder, through the end of
// the line, recording the value after the ":" and the spaces or tabs as
// removed.
func legacyRedactHeaders(s string, tr *passTrace) string {
	w := spanWriter{s: s, tr: tr}
	last := 0
	for d := 0; d < len(s); {
		i := strings.IndexByte(s[d:], ':')
		if i < 0 {
			break
		}
		d += i
		e := d
		for e > last && (s[e-1] == ' ' || s[e-1] == '\t') {
			e--
		}
		d++
		// Every header name ends in "key" or "authorization".
		if e == last || (s[e-1]|0x20 != 'y' && s[e-1]|0x20 != 'n') {
			continue
		}
		for _, name := range legacyHeaderNames {
			// matchSuffixFoldBack folds U+017F and U+212A as RE2 (?i) does
			// and checks the word boundary before the name.
			if _, ok := matchSuffixFoldBack(s, last, e, name); !ok {
				continue
			}
			valStart := d
			for valStart < len(s) && (s[valStart] == ' ' || s[valStart] == '\t') {
				valStart++
			}
			end := valStart
			for end < len(s) && s[end] != '\r' && s[end] != '\n' {
				end++
			}
			w.copy(last, e)
			w.lit(": " + redactedValue)
			w.remove(valStart, end)
			last, d = end, end
			break
		}
	}
	return w.done(last)
}

// legacyRepl is the text that replaces a match of a legacy pass that ends at
// s[end]: pre, the copy of s[from:to], the copy of the delimiter at s[delim]
// when delim >= 0, and lit. The pass removes s[core[0]:core[1]], the value
// that the placeholder in lit stands for.
type legacyRepl struct {
	pre, lit string
	from, to int
	delim    int
	core     [2]int
	end      int
}

// keepRepl is the replacement of a match at s[i:end] that keeps it as it is.
func keepRepl(i, end int) legacyRepl {
	return legacyRepl{from: i, to: end, delim: -1, end: end}
}

// legacyMatcher reports whether a match of a legacy pass starts at s[i] (see
// legacyReplace) and returns its replacement.
type legacyMatcher func(t *legacyText, i, pos int, brackets *openBrackets) (legacyRepl, bool)

// legacyReplace replaces the matches of match in s, searching from the start
// of s and, after each match, from where the text it replaced or kept ends,
// as replaceMatchesFunc on main did, recording in tr what it copies and
// removes. match reports whether a match starts at s[i], given that the
// search resumed at pos (a word boundary holds there). A match starts with a
// byte in starts, or with a letter when starts is empty.
func legacyReplace(s string, tr *passTrace, budget *int, compound bool, starts string, match legacyMatcher) string {
	w := spanWriter{s: s, tr: tr}
	if starts != "" && !strings.ContainsAny(s, starts) {
		return w.done(0)
	}
	t := &legacyText{s: s, budget: budget, compound: compound}
	var brackets openBrackets
	last, pos := 0, 0
	for i := 0; i < len(s); i++ {
		if starts != "" {
			k := strings.IndexAny(s[i:], starts)
			if k < 0 {
				break
			}
			i += k
		} else if c := s[i]; c < utf8.RuneSelf {
			// A match that starts with a letter starts a word, or where
			// the search resumed, so the letters inside a word are
			// skipped.
			if c |= 0x20; c < 'a' || c > 'z' || (i != pos && i > 0 && isWordByte(s[i-1])) {
				continue
			}
		} else if !isLegacyLetterAt(s, i) {
			continue
		}
		r, ok := match(t, i, pos, &brackets)
		if !ok {
			continue
		}
		w.copy(last, i)
		w.lit(r.pre)
		w.copy(r.from, r.to)
		if r.delim >= 0 {
			w.copy(r.delim, r.delim+1)
		}
		w.lit(r.lit)
		w.remove(r.core[0], r.core[1])
		last = r.end
		pos = max(r.end, i+1)
		i = pos - 1
	}
	return w.done(last)
}

// legacyText is a text being matched as the expressions did. dq, sq, edq and
// esq memoize the ends of quoted strings (see quotedEnd).
type legacyText struct {
	s                string
	budget           *int
	compound         bool
	dq, sq, edq, esq []int32
	// keys caches what isSensitiveKey reports for the names that keyEndsAt
	// reads, which a text often repeats.
	keys map[string]bool
}

// isSensitiveKey is isSensitiveKey, cached in t.keys.
func (t *legacyText) isSensitiveKey(k string) bool {
	sensitive, ok := t.keys[k]
	if !ok {
		if t.keys == nil {
			t.keys = make(map[string]bool)
		}
		sensitive = isSensitiveKey(k)
		t.keys[k] = sensitive
	}
	return sensitive
}

// bearerAt matches "bearer" and whitespace followed by a quoted value, a
// bracketed value or a run of bytes other than whitespace, ",", ";", ")",
// quotes and "]", as redactBearerToken on main did.
func (t *legacyText) bearerAt(i, pos int, _ *openBrackets) (legacyRepl, bool) {
	s := t.s
	if (s[i] != 'b' && s[i] != 'B') || (i != pos && !isLegacyBoundary(s, i)) {
		return legacyRepl{}, false
	}
	word := legacyFoldPrefixEnd(s, i, "bearer")
	if word < 0 || word >= len(s) {
		return legacyRepl{}, false
	}
	j := word
	for j < len(s) && isRegexpSpace(s[j]) {
		j++
	}
	if j == word || j == len(s) {
		return legacyRepl{}, false
	}
	end := t.bearerValueEnd(j)
	if end < 0 {
		return legacyRepl{}, false
	}
	if rest := s[j:]; len(rest) >= 2 && strings.EqualFold(rest[:2], "of") {
		if r, _ := utf8.DecodeRuneInString(rest[2:]); len(rest) == 2 || unicode.IsSpace(r) {
			return keepRepl(i, end), true
		}
	}
	if inner, _ := legacyUnquote(s[j:end]); isRedactedValue(inner) {
		return keepRepl(i, end), true
	}
	return legacyRepl{from: i, to: word, delim: -1, lit: " " + redactedValue, core: [2]int{j, end}, end: end}, true
}

func (t *legacyText) bearerValueEnd(j int) int {
	s := t.s
	switch s[j] {
	case '"', '\'':
		return t.quotedEnd(j)
	case '[':
		k := j + 1
		for k < len(s) && s[k] != ']' && !isRegexpSpace(s[k]) {
			k++
		}
		if k > j+1 && k < len(s) && s[k] == ']' {
			for k++; k < len(s) && !isBearerBracketTailStop(s[k]); k++ {
			}
			return k
		}
	}
	k := j
	for k < len(s) && !isBearerBareStop(s[k]) {
		k++
	}
	if k == j {
		return -1
	}
	return k
}

// basicAt matches "basic", whitespace and a base64 token, replaced as
// redactBasicCredential on main did when it holds a basic credential.
func (t *legacyText) basicAt(i, pos int, _ *openBrackets) (legacyRepl, bool) {
	s := t.s
	if (s[i] != 'b' && s[i] != 'B') || (i != pos && !isLegacyBoundary(s, i)) {
		return legacyRepl{}, false
	}
	word := legacyFoldPrefixEnd(s, i, "basic")
	if word < 0 || word >= len(s) {
		return legacyRepl{}, false
	}
	j := word
	for j < len(s) && isRegexpSpace(s[j]) {
		j++
	}
	k := j
	for k < len(s) && isBase64BodyByte(s[k]) {
		k++
	}
	if j == word || k == j {
		return legacyRepl{}, false
	}
	for n := 0; n < 2 && k < len(s) && s[k] == '='; n++ {
		k++
	}
	if k < len(s) && isBase64Byte(s[k]) {
		return keepRepl(i, k), true
	}
	if encoded := s[j:k]; len(encoded) <= maxBasicCredentialBytes {
		decoded, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil || (!strings.Contains(encoded, "=") && !bytes.Contains(decoded, []byte(":"))) {
			return keepRepl(i, k), true
		}
	}
	return legacyRepl{from: i, to: word, delim: -1, lit: " " + redactedValue, core: [2]int{j, k}, end: k}, true
}

// jsonPairAt matches a quoted sensitive key, ":" and a quoted value or a run
// of bytes other than whitespace, "," and "}", replaced as
// redactJSONKeyValue on main did.
func (t *legacyText) jsonPairAt(i, _ int, _ *openBrackets) (legacyRepl, bool) {
	s := t.s
	if s[i] != '"' {
		return legacyRepl{}, false
	}
	k := skipInlineSpacesFwd(s, i+1, len(s))
	for _, keyEnd := range t.keyEndsAt(k, true) {
		j := skipInlineSpacesFwd(s, keyEnd, len(s))
		if j >= len(s) || s[j] != '"' {
			continue
		}
		for j++; j < len(s) && isRegexpSpace(s[j]); j++ {
		}
		if j >= len(s) || s[j] != ':' {
			continue
		}
		for j++; j < len(s) && isRegexpSpace(s[j]); j++ {
		}
		if j >= len(s) {
			continue
		}
		end := t.quotedEnd(j)
		if end < 0 {
			for end = j; end < len(s) && !isJSONValueStop(s[end]); end++ {
			}
			if end == j {
				continue
			}
		}
		end = t.valueEnd(j, end)
		if inner, _ := legacyUnquote(s[j:end]); isRedactedValue(inner) {
			return keepRepl(i, end), true
		}
		return legacyRepl{pre: `"`, from: k, to: keyEnd, delim: -1, lit: `":"` + redactedValue + `"`, core: [2]int{j, end}, end: end}, true
	}
	return legacyRepl{}, false
}

// textPairAt matches a sensitive key, an optional (escaped) quote, ":" or
// "=" between optional whitespace or control bytes, and a quoted value or a
// run of bytes other than whitespace, ",", ";" and "&", replaced as
// redactTextKeyValue on main did.
func (t *legacyText) textPairAt(i, pos int, brackets *openBrackets) (legacyRepl, bool) {
	s := t.s
	if !isLegacyLetterAt(s, i) || (i != pos && !isLegacyBoundary(s, i)) {
		return legacyRepl{}, false
	}
	_, quoteEnd, delim, start, end, ok := t.textPairFields(i)
	if !ok {
		return legacyRepl{}, false
	}
	open := brackets.before(s, i)
	switch kind, pair := t.nestedKind(start, end, startsLine(s, delim+1, start)); kind {
	case nestedKey:
		return keepRepl(i, start), true
	case valueEndingInKeyName:
		_, quote := legacyUnquote(s[start:end])
		end = t.nestedPairEnd(pair, open)
		brackets.skip(end)
		return legacyRepl{from: i, to: quoteEnd, delim: delim, lit: quote + redactedValue + quote, core: [2]int{start, end}, end: end}, true
	}
	end = t.textValueEnd(start, end, open)
	brackets.skip(end)
	if end == start {
		return keepRepl(i, end), true
	}
	inner, quote := legacyUnquote(s[start:end])
	if isRedactedValue(inner) {
		return keepRepl(i, end), true
	}
	return legacyRepl{from: i, to: quoteEnd, delim: delim, lit: quote + redactedValue + quote, core: [2]int{start, end}, end: end}, true
}

func (t *legacyText) textPairFields(i int) (keyEnd, quoteEnd, delim, valStart, valEnd int, ok bool) {
	s := t.s
	for _, keyEnd := range t.keyEndsAt(i, false) {
		j := keyEnd
		if j+1 < len(s) && s[j] == '\\' && (s[j+1] == '"' || s[j+1] == '\'') {
			j += 2
		} else if j < len(s) && (s[j] == '"' || s[j] == '\'') {
			j++
		}
		quoteEnd := j
		for j < len(s) && isDelimSpace(s[j]) {
			j++
		}
		if j >= len(s) || (s[j] != ':' && s[j] != '=') {
			continue
		}
		delim := j
		for j++; j < len(s) && isDelimSpace(s[j]); j++ {
		}
		if j >= len(s) {
			continue
		}
		end := t.quotedEnd(j)
		if end < 0 {
			for end = j; end < len(s) && !isTextValueStop(s[end]); end++ {
			}
			if end == j {
				continue
			}
		}
		return keyEnd, quoteEnd, delim, j, end, true
	}
	return 0, 0, 0, 0, 0, false
}

// keyEndsAt returns where the sensitive key names that start at s[i] end: a
// "temp:" key, whose name runs on to the first byte that cannot be in it (in
// a quoted JSON key, "=" and ":" can), or any fixed key name.
func (t *legacyText) keyEndsAt(i int, quoted bool) []int {
	s := t.s
	if i >= len(s) {
		return nil
	}
	if hasFoldPrefix(s[i:], session.KeyPrefixTemp) {
		j := i + len(session.KeyPrefixTemp)
		for j < len(s) && (isTempKeyBodyByte(s[j]) || quoted && (s[j] == '=' || s[j] == ':')) {
			j++
		}
		if j > i+len(session.KeyPrefixTemp) {
			return []int{j}
		}
		return nil
	}
	// A key name is followed by a quote, whitespace or a delimiter, none of
	// which can be in it, so it ends where the run of bytes that can be in
	// it does.
	j := i
	if t.compound {
		for j < len(s) && j-i <= maxNestedKeyLen {
			if c := s[j]; c < utf8.RuneSelf {
				if !isCompoundKeyByte(c) {
					break
				}
				j++
			} else if f, size := legacyFoldAt(s, j); f != 0 {
				j += size
			} else {
				break
			}
		}
		if j > i && j-i <= maxNestedKeyLen && t.keyEndFollows(j, quoted) && t.isSensitiveKey(s[i:j]) && (quoted || !isSkippedPathDottedPair(s, i)) {
			return []int{j}
		}
		return nil
	}
	var buf [64]byte
	n := 0
	for j < len(s) && n <= maxLegacyKeyLen && n < len(buf) {
		c, size := legacyFoldAt(s, j)
		if !isWordByte(c) && c != '-' {
			break
		}
		buf[n] = c
		n++
		j += size
	}
	if n == 0 || n > maxLegacyKeyLen || !t.keyEndFollows(j, quoted) {
		return nil
	}
	if _, ok := legacyKeyNames[string(buf[:n])]; ok {
		return []int{j}
	}
	return nil
}

// keyEndFollows reports whether what may end a key name follows s[:j]: a
// quote, whitespace, a control byte or a delimiter, and, in a quoted JSON
// key, the end of the key.
func (t *legacyText) keyEndFollows(j int, quoted bool) bool {
	if j >= len(t.s) {
		return false
	}
	c := t.s[j]
	return isQuoteByte(c) || c == '\\' || c == ':' || c == '=' || isDelimSpace(c) || (quoted && c >= utf8.RuneSelf)
}

// valueEnd extends the value at s[start:end] that opens a JSON object or
// array to its closing bracket, as sensitiveValueEnd does, or to the end of
// s if it is not closed or the matchers have used up their budget.
func (t *legacyText) valueEnd(start, end int) int {
	if c := t.s[start]; c != '{' && c != '[' {
		return end
	}
	limit := min(len(t.s), start+max(*t.budget, 0))
	cEnd, closed := scanBalancedContainer(t.s, start, limit)
	*t.budget -= cEnd - start
	if !closed {
		return len(t.s)
	}
	return max(end, cEnd)
}

func (t *legacyText) textValueEnd(start, end, open int) int {
	if _, quote := legacyUnquote(t.s[start:end]); quote != "" {
		return end
	}
	rest := t.valueEnd(start, start)
	return rest + len(trimEnclosingClosers(t.s[rest:max(end, rest)], open))
}

// nestedPairEnd returns where the text redacted with a value that only ends
// in a key name ends: at the end of the value of the pair that the key name
// starts at s[pair], or of the pair that that value names, and so on.
func (t *legacyText) nestedPairEnd(pair, open int) int {
	for {
		_, _, delim, start, end, ok := t.textPairFields(pair)
		if !ok {
			return pair
		}
		kind, next := t.nestedKind(start, end, startsLine(t.s, delim+1, start))
		if kind == notNestedKey || next <= pair {
			return t.textValueEnd(start, end, open)
		}
		pair = next
	}
}

// nestedKind tells whether the value at s[start:end] is the name of a nested
// pair, as nestsSensitiveKey did.
func (t *legacyText) nestedKind(start, end int, lineStart bool) (nestedKeyKind, int) {
	s := t.s
	if end-start > maxNestedKeyLen || s[start] == '{' || s[start] == '[' {
		return notNestedKey, 0
	}
	v := s[start:end]
	if inner, quote := legacyUnquote(v); quote != "" {
		if !t.valueFollowsDelimiter(end) {
			return notNestedKey, 0
		}
		return t.nestedKeyName(inner, start+len(quote), lineStart)
	}
	if c := v[len(v)-1]; c == ':' || c == '=' {
		if !t.valueStartsAt(end) {
			return notNestedKey, 0
		}
		return t.nestedKeyName(v[:len(v)-1], start, lineStart)
	}
	if j := skipDelimSpaceFwd(s, end); j < len(s) && (s[j] == ':' || s[j] == '=') {
		if !t.valueStartsAt(j + 1) {
			return notNestedKey, 0
		}
		return t.nestedKeyName(v, start, lineStart)
	}
	if q := strings.IndexAny(v, `"'\`); q > 0 && (v[q-1] == ':' || v[q-1] == '=') && t.quotedEnd(start+q) > end && t.valueStartsAt(start+q) {
		if kind, pair := t.nestedKeyName(v[:q-1], start, lineStart); kind != notNestedKey {
			return kind, pair
		}
	}
	i := strings.LastIndexAny(v, ":=")
	if i <= 0 {
		return notNestedKey, 0
	}
	if inner, _ := legacyUnquote(trimTrailingPunctuation(v[i+1:])); !isRedactedValue(inner) {
		return notNestedKey, 0
	}
	return t.nestedKeyName(v[:i], start, lineStart)
}

func (t *legacyText) valueFollowsDelimiter(i int) bool {
	j := skipDelimSpaceFwd(t.s, i)
	return j < len(t.s) && (t.s[j] == ':' || t.s[j] == '=') && t.valueStartsAt(j+1)
}

// valueStartsAt reports whether a value that can be redacted starts at s[i],
// possibly after whitespace: a quoted value that ends there and holds a byte
// other than whitespace and closing punctuation, or else a run of bytes up to
// the next whitespace, ",", ";" or "&" that holds such a byte.
func (t *legacyText) valueStartsAt(i int) bool {
	s := t.s
	j := skipDelimSpaceFwd(s, i)
	if j < len(s) && (s[j] == '"' || s[j] == '\'' || s[j] == '\\') {
		if end := t.quotedEnd(j); end > 0 && legacyQuotedValueEnds(s, j, end) {
			inner, _ := legacyUnquote(s[j:end])
			return strings.Trim(inner, delimSpaceAndClosers) != ""
		}
	}
	for ; j < len(s) && !isTextValueStop(s[j]); j++ {
		if !isDelimSpace(s[j]) && strings.IndexByte(closingPunctuation, s[j]) < 0 {
			return true
		}
	}
	return false
}

const delimSpaceAndClosers = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\v\f\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f \x7f" + closingPunctuation

// legacyQuotedValueEnds reports whether the quoted value s[start:end] ends
// there rather than running on into a word or the next string.
func legacyQuotedValueEnds(s string, start, end int) bool {
	if end == len(s) {
		return true
	}
	r, _ := utf8.DecodeRuneInString(s[end:])
	if r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) {
		return false
	}
	_, quote := legacyUnquote(s[start:end])
	return !strings.HasPrefix(s[end:], quote)
}

// legacyUnquote strips one pair of quotes (", ', \" or \') from v.
func legacyUnquote(v string) (inner, quote string) {
	for _, q := range []string{`\"`, `\'`, `"`, `'`} {
		if len(v) >= 2*len(q) && strings.HasPrefix(v, q) && strings.HasSuffix(v, q) {
			return v[len(q) : len(v)-len(q)], q
		}
	}
	return v, ""
}

// legacyNestedKeyName classifies name, the name of a pair that a value may
// start at offset at of the text, as nestedKeyName did: a fixed key name,
// possibly after a quote and a "user:" or "app:" prefix, is a nested key; a
// name that only ends in a key name after a byte other than a letter, a
// digit or "_", such as db/password, is a nested key on a line of its own,
// and otherwise a value that ends in a key name.
func legacyNestedKeyName(name string, at int, lineStart bool) (nestedKeyKind, int) {
	off := leadingOptionalQuoteLen(name)
	rest := name[off:]
	keyAt := 0
	if hasFoldPrefix(rest, session.KeyPrefixUser) {
		keyAt = len(session.KeyPrefixUser)
	} else if hasFoldPrefix(rest, session.KeyPrefixApp) {
		keyAt = len(session.KeyPrefixApp)
	}
	if isLegacyFixedKey(rest[keyAt:]) {
		return nestedKey, at + off + keyAt
	}
	// The longest prefix that ends in a byte other than a letter, a digit,
	// "_", whitespace, a quote or a backslash, followed by a key name.
	prefix := -1
	for n := 1; n <= len(rest) && n <= maxLegacyKeyLen; n++ {
		if isLegacyFixedKey(rest[len(rest)-n:]) && isLegacyKeyPrefix(rest[:len(rest)-n]) {
			prefix = len(rest) - n
			break
		}
	}
	if c := strings.LastIndexByte(rest, ':'); c+1 < len(rest) && c-len(session.KeyPrefixTemp)+1 > prefix {
		tStart := c - len(session.KeyPrefixTemp) + 1
		if hasFoldPrefix(rest[tStart:], session.KeyPrefixTemp) && isLegacyKeyPrefix(rest[:tStart]) &&
			strings.IndexFunc(rest[c+1:], func(r rune) bool { return r < utf8.RuneSelf && !isTempKeyBodyByte(byte(r)) }) < 0 {
			prefix = tStart
		}
	}
	switch {
	case prefix < 0:
		return notNestedKey, 0
	case lineStart:
		return nestedKey, at + off + prefix
	default:
		return valueEndingInKeyName, at + off + prefix
	}
}

// nestedKeyName is legacyNestedKeyName, where, for compound expressions, a
// name that isSensitiveKey accepts as a whole is a key name too.
func (t *legacyText) nestedKeyName(name string, at int, lineStart bool) (nestedKeyKind, int) {
	kind, pair := legacyNestedKeyName(name, at, lineStart)
	if kind != notNestedKey || !t.compound {
		return kind, pair
	}
	off := leadingOptionalQuoteLen(name)
	rest := name[off:]
	if rest == "" || strings.ContainsAny(rest, "\t\n\f\r \"'\\") || !isSensitiveKey(rest) {
		return notNestedKey, 0
	}
	if lineStart {
		return nestedKey, at + off
	}
	return valueEndingInKeyName, at + off
}

func isLegacyKeyPrefix(p string) bool {
	if p == "" {
		return true
	}
	return !strings.ContainsAny(p, "\t\n\f\r \"'\\") && !isWordByte(p[len(p)-1])
}

// quotedEnd returns the end of the quoted string that starts at s[i] as the
// expressions read it, or -1: inside "..." or '...', a backslash escapes any
// byte but a line break; inside \"...\" or \'...\', a doubled backslash
// starts an escape of the inner text, and the string ends at an unpaired \"
// or \' (see escapedQuotedRegex on main).
func (t *legacyText) quotedEnd(i int) int {
	s := t.s
	switch {
	case s[i] == '"':
		return quotedEndFrom(&t.dq, s, i+1, '"', false)
	case s[i] == '\'':
		return quotedEndFrom(&t.sq, s, i+1, '\'', false)
	case s[i] == '\\' && i+1 < len(s) && s[i+1] == '"':
		return quotedEndFrom(&t.edq, s, i+2, '"', true)
	case s[i] == '\\' && i+1 < len(s) && s[i+1] == '\'':
		return quotedEndFrom(&t.esq, s, i+2, '\'', true)
	}
	return -1
}

// quotedEndFrom returns the end of the quoted string whose body starts at
// s[p], filling memo with the ends for every start, from the end of s back,
// when first called.
func quotedEndFrom(memo *[]int32, s string, p int, q byte, escaped bool) int {
	if *memo == nil {
		m := make([]int32, len(s)+1)
		m[len(s)] = -1
		at := func(j int) int32 {
			if j > len(s) {
				return -1
			}
			return m[j]
		}
		for i := len(s) - 1; i >= 0; i-- {
			c := s[i]
			switch {
			case !escaped && c == q:
				m[i] = int32(i + 1)
			case !escaped && c == '\\':
				m[i] = -1
				if i+1 < len(s) && s[i+1] != '\n' {
					m[i] = at(i + 2)
				}
			case !escaped:
				m[i] = at(i + 1)
			case c == q || (c == '\\' && i+1 >= len(s)):
				m[i] = -1
			case c != '\\':
				m[i] = at(i + 1)
			case s[i+1] == q:
				m[i] = int32(i + 2)
			case s[i+1] != '\\':
				m[i] = at(i + 2)
			case i+3 < len(s) && s[i+2] == '\\' && (s[i+3] == '\\' || s[i+3] == q):
				m[i] = at(i + 4)
			case i+2 < len(s) && s[i+2] != '\\' && s[i+2] != q:
				m[i] = at(i + 3)
			default:
				m[i] = -1
			}
		}
		*memo = m
	}
	return int((*memo)[p])
}
