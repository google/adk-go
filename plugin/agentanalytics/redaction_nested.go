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
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"google.golang.org/adk/v2/session"
)

// Nested keys: whether the value of a pair is itself the name of a pair, or
// runs on past where it seems to end, and where the text to redact with it
// ends (see nestsSensitiveKey).

// maxNestedKeyLen is the length of the longest value that nestsSensitiveKey
// may take for a key. A longer value is redacted, so that it is not matched
// again from each key in it.
const maxNestedKeyLen = 128

// nestedKeyKind tells how redactJSONKeyValues and redactTextKeyValues treat a
// value that may be the name of a nested key-value pair (see
// nestsSensitiveKey).
type nestedKeyKind int

const (
	// notNestedKey is a value, which is redacted.
	notNestedKey nestedKeyKind = iota
	// nestedKey is a sensitive key, which is left in place, so that the
	// pair that it starts is matched and redacted on its own.
	nestedKey
	// valueEndingInKeyName is a value that only ends in a sensitive key
	// name, which is redacted together with the value of the pair that it
	// starts.
	valueEndingInKeyName
	// gluedAfterValue is a value whose matched span stops at an embedded
	// delimiter, quote boundary or maxNestedKeyLen cutoff while a trailing
	// value continues at pair, so the outer replacement consumes through the
	// end of that trailing value as well.
	gluedAfterValue
)

// nestsSensitiveKey tells whether the value matched at s[start:end] is a
// sensitive key rather than a value, as in the YAML "credentials:\n  token:
// ..." or the text "credentials: token = ...", and if so returns nestedKey
// and where the key name in the value starts. That is the case only when
// the whole value is a sensitive key name that ends with a delimiter or is
// followed by one, and a value follows the delimiter, so that the pair that
// the name starts is matched and redacted on its own; a quoted name (including
// a padded or state-prefixed JSON key accepted by isValidJSONPatternKey) must
// be followed by a delimiter. A value made of such a name and a delimiter
// followed by the start of a quoted value or an unclosed JSON object/array
// ("{" or "[") that runs on past end, as in credentials: password="a b" or
// "app:temp:conn": secret:{"k": "v"}, is a key too. So is a value made of such
// a name, a delimiter and a redacted value, as in "token:[REDACTED]", so that
// scrubbing redacted text again changes nothing. Any other value, such as
// the quoted 'token: abc def', abc-token:def, or an abc-token: with no value
// after it, is not a key (notNestedKey), and neither is a value that opens an
// unclosed JSON object or array (unless a closed "[" array is immediately
// joined to a nested sensitive pair, as in [abc];"token": ...), so that a
// credentials envelope written that way is redacted as a whole. Only the
// value, if it is at most maxNestedKeyLen bytes long, and the whitespace, the
// delimiter and the start of the next value after it, up to the next
// whitespace, ",", ";" or "&", or to the character after the closing quote of
// a quoted value that starts in the value or after it, are read.
//
// The name is a sensitive key name when it is itself one (see
// nestedKeyNameRe). A name that only ends in one (see compoundKeyNameRe and
// trailingQuotedSensitiveKey), such as X-Amz-Security-Token, db/password or
// abc&"token", is nestedKey only when the whole value starts a line without a
// preceding joined value, as a nested key does in YAML: lineStart reports
// that. On the line of the key before it (or after a joined value or a
// path-prefixed dotted segment that redactTextKeyValues skips), such a value
// may begin with a credential before the nested key, as in the Go error
// "password=dev-SECRET-password: dial tcp ...", so nestsSensitiveKey then
// returns valueEndingInKeyName, and where the key name in it starts, so that
// the value is redacted together with the value of the pair it starts.
func nestsSensitiveKey(s string, start, end int, lineStart bool, openQ openQuoteState, st *quoteScanState) (nestedKeyKind, int) {
	kind, pair := classifyNestedKey(s, start, end, lineStart, openQ, st)
	if kind == nestedKey && isSkippedPathDottedPair(s, pair) {
		return valueEndingInKeyName, pair
	}
	return kind, pair
}

func classifyNestedKey(s string, start, end int, lineStart bool, openQ openQuoteState, st *quoteScanState) (nestedKeyKind, int) {
	cur := start
	for cur < end && (s[cur] == '{' || s[cur] == '[') {
		cEnd, closed := scanBalancedContainer(s, cur, end)
		if !closed || cEnd == end {
			return notNestedKey, 0
		}
		cur = cEnd
	}
	if cur > start {
		var kind nestedKeyKind
		var pair int
		if end-cur > maxNestedKeyLen {
			if s[start:cur] == redactedValue && hasRedactedOverBudgetTail(s, cur, end, st) {
				return nestedKey, cur
			}
			kind, pair = classifyOverBudgetNestedKey(s, cur, end, openQ, st)
		} else {
			kind, pair = classifyNestedKey(s, cur, end, false, openQ, st)
		}
		if kind == notNestedKey {
			return notNestedKey, 0
		}
		if kind == gluedAfterValue {
			return gluedAfterValue, pair
		}
		if s[start:cur] == redactedValue && !isSkippedPathDottedPair(s, pair) && !hasUnredactedDelimiter(s, cur, pair) {
			return nestedKey, cur
		}
		return valueEndingInKeyName, pair
	}
	if end-start > maxNestedKeyLen {
		return classifyOverBudgetNestedKey(s, start, end, openQ, st)
	}
	v := s[start:end]
	if inner, quote := unquoteValue(v); quote != "" {
		okVal, gluedEnd := nestedValueFollowsDelimiter(s, end, openQ, st)
		if !okVal && end < len(s) && isQuoteByte(s[end]) && (openQ.quote != s[end] || openQ.escaped) {
			if ok, _ := nestedValueFollowsDelimiter(s, end+1, openQ, st); ok {
				// A stray quote after a quoted key, as in
				// \"temp:a\"": SECRET, that does not close the enclosing
				// string: the value is redacted with the glued text from
				// that quote on, which takes the value after the
				// delimiter (see extendGluedValueEnd).
				return gluedAfterValue, end
			}
		}
		if !okVal {
			if nextVal, ok := stolenQuotedKeyAfterQuoteEnd(s, end, openQ, st); ok {
				return gluedAfterValue, nextVal
			}
			if isGluedTokenAfterQuote(s, end) && innerContinuesIntoGluedToken(inner, st) {
				return gluedAfterValue, end
			}
			return notNestedKey, 0
		}
		if kind, pair := nestedKeyName(inner, start+len(quote), lineStart); kind != notNestedKey {
			if gluedEnd > 0 {
				return gluedAfterValue, gluedEnd
			}
			if kind == valueEndingInKeyName && pair == start+len(quote) {
				pair = start
			}
			return kind, pair
		}
		trimmedInner := trimInlineSpaces(inner)
		// A padded quoted key at the start of a line, as the "   DB_PASSWORD   "
		// of credentials:\n  "   DB_PASSWORD   "="[REDACTED]", is a nested
		// key whatever its quote (", ', \" or \') and delimiter: the key-value
		// passes match it on its own (see matchPaddedQuotedTextKey).
		paddedLineKey := lineStart && isPaddedQuotedTextKey(trimmedInner)
		if isValidJSONPatternKey(trimmedInner) {
			if gluedEnd > 0 {
				return gluedAfterValue, gluedEnd
			}
			if paddedLineKey || (lineStart && quote == `"` && isJSONColonAfter(s, end)) {
				return nestedKey, start
			}
			return valueEndingInKeyName, start
		}
		if kind, _ := nestedKeyName(trimmedInner, 0, false); kind != notNestedKey {
			if gluedEnd > 0 {
				return gluedAfterValue, gluedEnd
			}
			if paddedLineKey {
				return nestedKey, start
			}
			return valueEndingInKeyName, start
		}
		if pair, ok := quotedValueEndsInKey(s, start+len(quote), end-len(quote)); ok {
			// The closing quote closes a key that the last word of the
			// value starts, as the secret" of "abc; secret": SECRET.
			if gluedEnd > 0 {
				return gluedAfterValue, gluedEnd
			}
			return valueEndingInKeyName, pair
		}
		if nextVal, ok := stolenQuotedKeyAfterQuoteEnd(s, end, openQ, st); ok {
			return gluedAfterValue, nextVal
		}
		return notNestedKey, 0
	}
	strayEscDoubleEnd, strayEscSingleEnd := 0, 0
	strayOuter := -1
	// unplaced reports that a quote in the value opens a string that runs
	// past end and that none of the checks below places.
	unplaced := false
	for idx := 0; idx < len(v); idx++ {
		switch v[idx] {
		case '"', '\'', '\\':
			// A quote escaped by a backslash, such as the one of an escaped
			// quote \" that does not close (see scanEscapedQuoted), opens no
			// string, so it cannot skip brackets or delimiters after it.
			if v[idx] == '\\' {
				if start+idx+1 >= len(s) || !isQuoteByte(s[start+idx+1]) {
					continue
				}
				if (s[start+idx+1] == '"' && start+idx+2 < strayEscDoubleEnd) ||
					(s[start+idx+1] == '\'' && start+idx+2 < strayEscSingleEnd) {
					continue
				}
			} else if !isUnescapedQuote(v, 0, idx) {
				continue
			}
			if valPos, ok := danglingQuoteKeyAt(s, start, start+idx, openQ, st); ok {
				// The quote closes a key rather than opening a string, as
				// the token" of DB_PASSWORD: token":{k: "SECRET"}.
				return gluedAfterValue, valPos
			}
			qEnd, qOK := scanQuotedAt(s, start+idx, st)
			if !qOK {
				continue
			}
			if outer := strayQuoteOuter(s, start+idx, qEnd, end, st); outer >= 0 {
				// The quote is stray, as the one of x"S0/creds:{"k":
				// "SECRET"}, x"S0/creds:{ "k": "SECRET"} or
				// x"S0/creds:[1,"SECRET"], whose closing quote opens a
				// member of a container that opens in the value, also
				// when the value seems to end before that quote, at the
				// space or "," in the container, so the container after it
				// is read, as in classifyOverBudgetNestedKey. When the
				// string encloses that container and others after it, as
				// in x":{'a': "1"} {'b': 'SECRET'}", they are all redacted
				// with the value.
				if last, _ := enclosedContainerRun(s, start+idx, qEnd, outer, st); last > outer {
					return gluedAfterValue, last
				}
				if st != nil && st.openerBudgetExhausted {
					return gluedAfterValue, len(s)
				}
				if v[idx] == '\\' {
					if s[start+idx+1] == '"' {
						strayEscDoubleEnd = max(strayEscDoubleEnd, qEnd)
					} else {
						strayEscSingleEnd = max(strayEscSingleEnd, qEnd)
					}
				}
				strayOuter = outer
				idx = outer - start - 1
				continue
			}
			if st != nil && st.openerBudgetExhausted {
				return gluedAfterValue, len(s)
			}
			if qEnd <= end {
				idx = qEnd - start - 1
				continue
			}
			if kind, pair := classifySpanningQuoteInNestedKey(s, v, start, idx, qEnd, lineStart, openQ, st); kind != notNestedKey {
				return kind, pair
			}
			if kind, pair := trailingDelimitedKeyInValue(s, v, start, end, openQ, st); kind != notNestedKey {
				return kind, pair
			}
			if idx > 0 && enclosesMixedQuoteContainer(s, start+idx, qEnd, end, st) {
				if st != nil && st.openerBudgetExhausted {
					return gluedAfterValue, len(s)
				}
				return gluedAfterValue, start + idx
			}
			if last := spacedStrayQuoteContainer(s, start+idx, qEnd, end, st); last >= 0 {
				// The quoted string encloses containers after a delimiter
				// in it and seems to end inside them, as the one of
				// x": {'msg': "it's ok", 'k': 'SECRET'}" at the member in
				// the same quote, or the one of x":{'msg': 'say "hi"'}" at
				// the quote in the member in other quotes: the containers
				// are redacted with the value.
				return gluedAfterValue, last
			}
			if st != nil && st.openerBudgetExhausted {
				return gluedAfterValue, len(s)
			}
			// No check places the string that the quote opens, as the
			// apostrophe of user's": {'msg': "it's ok", 'k': 'SECRET'}" or
			// the ' of a'x": {'a': "1", 'b': 'SECRET'}", whose string seems
			// to close in the container: the scan goes on with the byte
			// after the quote, so that a later quote that opens the string
			// around the container, as the " of both, is still read. Only
			// a later quote is read, since the string of such a quote may
			// enclose the rest of the value: a container that no stray
			// quote leads to, as the {a: it's} of x"S0/creds:{a: it's}",
			// ends the scan, and the value stays a value, so that a word in
			// it that runs the container to the end of the text, as it's,
			// does not take the text after it.
			unplaced = true
			continue
		case '{', '[':
			if unplaced && start+idx != strayOuter {
				return notNestedKey, 0
			}
			if idx > 0 && end < len(s) {
				dIdx := delimBeforePos(v, idx)
				if dIdx > 0 || start+idx == strayOuter {
					cEnd, closed := scanBalancedContainer(s, start+idx, end)
					if closed {
						idx = cEnd - start - 1
						continue
					}
					if dIdx > 0 {
						if kind, pair := nestedKeyName(v[:dIdx], start, lineStart); kind != notNestedKey {
							return kind, pair
						}
					}
					// A container after a delimiter in the value that is
					// still open at end, as the {"k": "SECRET"} of
					// x/creds:{"k": "SECRET"}, runs past the value and is
					// redacted with it, as in classifyOverBudgetNestedKey.
					return gluedAfterValue, start + idx
				}
			}
		}
	}
	if unplaced {
		return notNestedKey, 0
	}
	if strings.HasSuffix(v, ":") || strings.HasSuffix(v, "=") {
		okVal, gluedEnd := nestedValueAfter(s, end, openQ, st)
		if !okVal {
			return notNestedKey, 0
		}
		kind, pair := nestedKeyName(v[:len(v)-1], start, lineStart)
		if kind != notNestedKey && gluedEnd > 0 {
			return gluedAfterValue, gluedEnd
		}
		return kind, pair
	}
	if followedByDelimiter(s, end) {
		okVal, gluedEnd := nestedValueFollowsDelimiter(s, end, openQ, st)
		if !okVal {
			return notNestedKey, 0
		}
		kind, pair := nestedKeyName(v, start, lineStart)
		if kind != notNestedKey && gluedEnd > 0 {
			return gluedAfterValue, gluedEnd
		}
		return kind, pair
	}
	// A pair that an earlier pass redacted, as in "token:[REDACTED]", or
	// token:"[REDACTED]"} in the %+v output of a struct.
	i := strings.LastIndexAny(v, ":=")
	if i <= 0 {
		return notNestedKey, 0
	}
	if inner, _ := unquoteValue(trimTrailingPunctuation(v[i+1:])); !isRedactedValue(inner) {
		return notNestedKey, 0
	}
	return nestedKeyName(v[:i], start, lineStart)
}

// isOverBudgetKeyEnd reports whether s[floor:end], the text before a
// delimiter in an over-budget value, ends in a sensitive key name, quoted or
// not. In strict mode, used to tell whether an over-budget tail is already
// redacted (see hasRedactedOverBudgetTail), a quoted key name must be a single
// token, without inner whitespace, so that a quoted value ending in a key
// word, as in "SECRET password", is not taken for a key, and a path-prefixed
// ".key" filename that redactTextKeyValues skips is not a key either.
func isOverBudgetKeyEnd(s string, floor, end int, strict bool) bool {
	for end > floor && isDelimSpace(s[end-1]) {
		end--
	}
	if end <= floor {
		return false
	}
	if inner, quote := unquoteValue(s[floor:end]); quote != "" {
		return isOverBudgetQuotedKey(trimInlineSpaces(inner), strict)
	}
	if end-2 >= floor && s[end-2] == '\\' && (s[end-1] == '"' || s[end-1] == '\'') && isUnescapedQuote(s, floor, end-2) {
		if kStart, key := matchPaddedQuotedTextKey(s, floor, end); kStart >= 0 {
			return !strict || !hasInnerSpace(key)
		}
		q := s[end-1]
		for openIdx := end - 4; openIdx >= floor; openIdx-- {
			if s[openIdx] == '\\' && s[openIdx+1] == q && isUnescapedQuote(s, floor, openIdx) {
				return (openIdx == floor || isJoinBeforeQuotedKey(s[openIdx-1])) &&
					isOverBudgetQuotedKey(trimInlineSpaces(s[openIdx+2:end-2]), strict)
			}
		}
		return false
	}
	if end-1 >= floor && (s[end-1] == '"' || s[end-1] == '\'') && isUnescapedQuote(s, floor, end-1) {
		if kStart, key := matchPaddedQuotedTextKey(s, floor, end); kStart >= 0 {
			return !strict || !hasInnerSpace(key)
		}
		q := s[end-1]
		for openIdx := end - 2; openIdx >= floor; openIdx-- {
			if s[openIdx] == q && isUnescapedQuote(s, floor, openIdx) {
				return isOverBudgetQuotedKey(trimInlineSpaces(s[openIdx+1:end-1]), strict)
			}
		}
		return false
	}
	kStart := findTextKeyStart(s, floor, end)
	if kStart < 0 {
		return false
	}
	return !strict || !isSkippedPathDottedPair(s, kStart)
}

// isOverBudgetQuotedKey reports whether trimmed, the trimmed interior of a
// quoted string, is a key name (see isPaddedQuotedTextKey), which in strict
// mode must also hold no inner whitespace.
func isOverBudgetQuotedKey(trimmed string, strict bool) bool {
	return isPaddedQuotedTextKey(trimmed) && (!strict || !hasInnerSpace(trimmed))
}

// hasInnerSpace reports whether s holds whitespace or a control byte.
func hasInnerSpace(s string) bool {
	return strings.IndexFunc(s, func(r rune) bool {
		return r <= 0x20 || r == 0x7f || unicode.IsSpace(r) || unicode.In(r, unicode.Z)
	}) >= 0
}

// hasUnredactedDelimiter reports whether s[from:to] holds a ":" or "=" that
// is not followed by a redacted placeholder (see startsWithRedactedValue): an
// assignment whose value may be a credential, as the =SECRET in
// [REDACTED]/x=SECRET/password:[REDACTED], so that the text around it is not
// taken for text that an earlier pass redacted. A delimiter followed only by
// whitespace and quotes up to to, which joins the text before it to a key
// that starts at to, as in [REDACTED]:"api.key": ..., does not count.
func hasUnredactedDelimiter(s string, from, to int) bool {
	for i := from; i < to; i++ {
		if (s[i] != ':' && s[i] != '=') || startsWithRedactedValue(s[i+1:]) {
			continue
		}
		j := i + 1
		for j < to && (isDelimSpace(s[j]) || s[j] == '"' || s[j] == '\'' || s[j] == '\\') {
			j++
		}
		if j < to {
			return true
		}
	}
	return false
}

// hasRedactedOverBudgetTail reports whether s[cur:end], the over-budget tail
// of a value after a redacted placeholder, as in
// dbPassword=[REDACTED]/<long>-password: "[REDACTED]", is already redacted
// text that a second pass leaves alone: it ends in a single sensitive key
// whose value is a redacted placeholder, and every other ":" or "=" in it is
// followed by a redacted placeholder too. Any other assignment in the tail,
// as in [REDACTED]/<long>=SECRET/password:[REDACTED], may hold a credential,
// so the tail is not skipped and the whole value is redacted.
func hasRedactedOverBudgetTail(s string, cur, end int, st *quoteScanState) bool {
	if hasUnredactedDelimiter(s, cur, end) {
		return false
	}
	v := s[cur:end]
	for idx := cur; idx < end; idx++ {
		switch s[idx] {
		case '"', '\'', '\\':
			if s[idx] != '\\' && !isUnescapedQuote(s, cur, idx) {
				continue
			}
			qEnd, ok := scanQuotedAt(s, idx, st)
			if !ok {
				continue
			}
			if qEnd <= end {
				idx = qEnd - 1
				continue
			}
			dPos := skipDelimSpaceFwd(s, qEnd)
			if dPos < len(s) && (s[dPos] == ':' || s[dPos] == '=') && startsWithRedactedValue(s[dPos+1:]) {
				inner, quote := unquoteValue(s[idx:qEnd])
				return quote != "" && isOverBudgetQuotedKey(trimInlineSpaces(inner), true)
			}
			return false
		case '{', '[':
			if idx > cur && delimBeforePos(s[cur:idx], idx-cur) > 0 {
				cEnd, closed := scanBalancedContainer(s, idx, end)
				if closed {
					idx = cEnd - 1
					continue
				}
				return false
			}
		}
	}
	if strings.HasSuffix(v, ":") || strings.HasSuffix(v, "=") {
		return startsWithRedactedValue(s[end:]) && isOverBudgetKeyEnd(s, cur, end-1, true)
	}
	dPos := skipDelimSpaceFwd(s, end)
	if dPos < len(s) && (s[dPos] == ':' || s[dPos] == '=') {
		return startsWithRedactedValue(s[dPos+1:]) && isOverBudgetKeyEnd(s, cur, end, true)
	}
	i := strings.LastIndexAny(v, ":=")
	if i <= 0 {
		return false
	}
	return startsWithRedactedValue(s[cur+i+1:]) && isOverBudgetKeyEnd(s, cur, cur+i, true)
}

// trailingDelimitedKeyInValue handles a value v = s[start:end] in which a
// quote that is not the quote of a key (see classifySpanningQuoteInNestedKey)
// runs on past end, pairing up with the quote of a later value. When the
// text before the last delimiter in v, or all of v when a delimiter follows
// it, ends in a sensitive key name, with a value after that delimiter, as in
// dbPassword:foo"x/token:\n" SECRET", DB_PASSWORD: client_secret':" SECRET"
// or DB_PASSWORD: foo"x/\"db.password\" = SECRET, the quote closes no string
// of v and the pair at the end of v takes that value, so v is redacted
// together with it (valueEndingInKeyName or gluedAfterValue). Otherwise v is
// a value.
func trailingDelimitedKeyInValue(s, v string, start, end int, openQ openQuoteState, st *quoteScanState) (nestedKeyKind, int) {
	name := v
	var okVal bool
	var gluedEnd int
	if followedByDelimiter(s, end) {
		okVal, gluedEnd = nestedValueFollowsDelimiter(s, end, openQ, st)
	} else {
		d := strings.LastIndexAny(v, ":=")
		if d <= 0 {
			return notNestedKey, 0
		}
		name = v[:d]
		okVal, gluedEnd = nestedValueAfter(s, start+d+1, openQ, st)
	}
	if !okVal {
		return notNestedKey, 0
	}
	kind, pair := nestedKeyName(name, start, false)
	if kind == notNestedKey {
		return notNestedKey, 0
	}
	if gluedEnd > 0 {
		return gluedAfterValue, gluedEnd
	}
	return valueEndingInKeyName, pair
}

func classifySpanningQuoteInNestedKey(s, v string, start, q, qEnd int, lineStart bool, openQ openQuoteState, st *quoteScanState) (nestedKeyKind, int) {
	if dIdx := delimBeforePos(v, q); dIdx > 0 {
		if okVal, gluedEnd := nestedValueAfter(s, start+q, openQ, st); okVal {
			if kind, pair := nestedKeyName(v[:dIdx], start, lineStart); kind != notNestedKey {
				if gluedEnd > 0 {
					// The nested value is a quoted string with text glued
					// after it, whose closing quote may really open a value
					// nested in it, as the "a token: "x SECRET" of
					// temp:a="a token: "x SECRET"; the value goes on to
					// where it really ends (see quotedValueRealEnd).
					if realEnd, ok := quotedValueRealEnd(s, start+q, qEnd, 0, openQ, st); ok && realEnd > gluedEnd {
						return gluedAfterValue, realEnd
					}
					return gluedAfterValue, gluedEnd
				}
				return kind, pair
			}
		}
	}
	canStartKeyQuote := false
	if v[q] == '"' || v[q] == '\'' {
		canStartKeyQuote = isUnescapedQuote(v, 0, q)
	} else if q == 0 || isJoinBeforeQuotedKey(v[q-1]) {
		canStartKeyQuote = true
	}
	if canStartKeyQuote {
		if okVal, gluedEnd := nestedValueFollowsDelimiter(s, qEnd, openQ, st); okVal {
			inner, quote := unquoteValue(s[start+q : qEnd])
			if quote != "" {
				trimmedInner := trimInlineSpaces(inner)
				if qEnd-(start+q) > maxNestedKeyLen {
					if isPaddedQuotedTextKey(trimmedInner) {
						if gluedEnd > 0 {
							return gluedAfterValue, gluedEnd
						}
						return valueEndingInKeyName, start + q
					}
					return notNestedKey, 0
				}
				if isValidJSONPatternKey(trimmedInner) {
					if gluedEnd > 0 {
						return gluedAfterValue, gluedEnd
					}
					if q == 0 && lineStart && quote == `"` && isJSONColonAfter(s, qEnd) {
						return nestedKey, start
					}
					return valueEndingInKeyName, start + q
				}
				if kind, _ := nestedKeyName(trimmedInner, 0, false); kind != notNestedKey {
					if gluedEnd > 0 {
						return gluedAfterValue, gluedEnd
					}
					return valueEndingInKeyName, start + q
				}
			}
		} else if isGluedTokenAfterQuote(s, qEnd) &&
			(q == 0 || (!isWordByte(v[q-1]) && v[q-1] != '"' && v[q-1] != '\'' && v[q-1] != '\\')) {
			if inner, quote := unquoteValue(s[start+q : qEnd]); quote != "" && innerContinuesIntoGluedToken(inner, st) {
				return gluedAfterValue, qEnd
			}
		}
	}
	return notNestedKey, 0
}

// classifyOverBudgetNestedKey fails closed when s[start:end] exceeds
// maxNestedKeyLen: if an unclosed container or quote inside the over-budget
// span carries a trailing value past end, or a delimiter and value follow end,
// it returns gluedAfterValue so the outer replacement consumes that value too.
func classifyOverBudgetNestedKey(s string, start, end int, openQ openQuoteState, st *quoteScanState) (nestedKeyKind, int) {
	v := s[start:end]
	if inner, quote := unquoteValue(v); quote != "" {
		if okVal, gluedEnd := nestedValueFollowsDelimiter(s, end, openQ, st); okVal {
			if _, endsInKey := quotedValueEndsInKey(s, start+len(quote), end-len(quote)); endsInKey || isPaddedQuotedTextKey(trimInlineSpaces(inner)) {
				if gluedEnd > 0 {
					return gluedAfterValue, gluedEnd
				}
				return gluedAfterValue, valAfterDelim(s, skipDelimSpaceFwd(s, end)+1)
			}
		}
		if nextVal, ok := stolenQuotedKeyAfterQuoteEnd(s, end, openQ, st); ok {
			return gluedAfterValue, nextVal
		}
		if hasGluedSuffixAfterQuotedValue(s, start, end, openQ, st) {
			return gluedAfterValue, end
		}
		return notNestedKey, 0
	}
	strayEscDoubleEnd, strayEscSingleEnd := 0, 0
	strayOuter := -1
	// unplaced is as in classifyNestedKey.
	unplaced := false
	for idx := start; idx < end; idx++ {
		switch s[idx] {
		case '"', '\'', '\\':
			if s[idx] == '\\' {
				if idx+1 >= len(s) || !isQuoteByte(s[idx+1]) {
					continue
				}
				if (s[idx+1] == '"' && idx+2 < strayEscDoubleEnd) ||
					(s[idx+1] == '\'' && idx+2 < strayEscSingleEnd) {
					continue
				}
			} else if !isUnescapedQuote(s, start, idx) {
				continue
			}
			if valPos, ok := danglingQuoteKeyAt(s, start, idx, openQ, st); ok {
				past, closedEnd := valueEndsPast(s, valPos, end, st)
				if st != nil && st.openerBudgetExhausted {
					return gluedAfterValue, len(s)
				}
				if past {
					// The quote closes a key whose value starts after the
					// value, as the password" of password": {"k": "SECRET"},
					// or is a quoted string or a container that runs past
					// it, as the "a SECRET" of
					// <130a>=S0/my_api_key":"a SECRET" or the
					// {"value": "SECRET"} of <130a>=S0/credentials":{...}.
					// Any other value that starts inside it is redacted
					// with it, so that the scan moves past end and chains
					// stay linear.
					return gluedAfterValue, valPos
				}
				// The quote closes a key, so it opens no string that could
				// pair with a later quote and skip the containers and quotes
				// after it, as the password" of
				// password":S0/<130a>/credentials:{"k": "SECRET"}. A bare
				// value, or a quoted string that opens a member of a
				// container (see opensContainerMember), is read on from the
				// quote. A quoted string or a container that closes inside
				// the value is skipped as a whole, so that it is read once.
				// The skip keeps the scan linear: a container that closes at
				// the end of the text, which the "{" and "[" case below does
				// not skip, would otherwise be scanned again from each
				// dangling key quote nested in it, as in
				// <130a>/password":x/token":{/password":x/token":{...}},
				// which takes quadratic time.
				if closedEnd > 0 {
					idx = closedEnd - 1
				}
				continue
			}
			qEnd, ok := scanQuotedAt(s, idx, st)
			if !ok {
				continue
			}
			if outer := strayQuoteOuter(s, idx, qEnd, end, st); outer >= 0 {
				// The quote is stray, as the one of x"y/credentials:{"k":
				// "SECRET"}, x"y/credentials:{\n  "k": "SECRET"} or
				// x"y/credentials:[true, "SECRET"], whose closing quote
				// opens a member of a container that opens in the value,
				// also past end, so the container after it is read, with
				// the containers after it that the string encloses, as in
				// classifyNestedKey.
				if last, _ := enclosedContainerRun(s, idx, qEnd, outer, st); last > outer {
					return gluedAfterValue, last
				}
				if st != nil && st.openerBudgetExhausted {
					return gluedAfterValue, len(s)
				}
				if s[idx] == '\\' {
					if s[idx+1] == '"' {
						strayEscDoubleEnd = max(strayEscDoubleEnd, qEnd)
					} else {
						strayEscSingleEnd = max(strayEscSingleEnd, qEnd)
					}
				}
				strayOuter = outer
				idx = outer - 1
				continue
			}
			if st != nil && st.openerBudgetExhausted {
				return gluedAfterValue, len(s)
			}
			if qEnd <= end {
				idx = qEnd - 1
				continue
			}
			if okVal, gluedEnd := nestedValueFollowsDelimiter(s, qEnd, openQ, st); okVal {
				if inner, quote := unquoteValue(s[idx:qEnd]); quote != "" && isPaddedQuotedTextKey(trimInlineSpaces(inner)) {
					if gluedEnd > 0 {
						return gluedAfterValue, gluedEnd
					}
					return gluedAfterValue, valAfterDelim(s, skipDelimSpaceFwd(s, qEnd)+1)
				}
			}
			if idx > start {
				if dRel := delimBeforePos(s[start:idx], idx-start); dRel > 0 && isOverBudgetKeyEnd(s, start, start+dRel, false) {
					if okVal, gluedEnd := nestedValueAfter(s, idx, openQ, st); okVal {
						if gluedEnd > 0 {
							return gluedAfterValue, gluedEnd
						}
						return gluedAfterValue, idx
					}
				}
			}
			if isGluedTokenAfterQuote(s, qEnd) &&
				(idx == start || (!isWordByte(s[idx-1]) && s[idx-1] != '"' && s[idx-1] != '\'' && s[idx-1] != '\\')) {
				if inner, quote := unquoteValue(s[idx:qEnd]); quote != "" && innerContinuesIntoGluedToken(inner, st) {
					return gluedAfterValue, qEnd
				}
			}
			if idx > start && enclosesMixedQuoteContainer(s, idx, qEnd, end, st) {
				if st != nil && st.openerBudgetExhausted {
					return gluedAfterValue, len(s)
				}
				return gluedAfterValue, idx
			}
			if last := spacedStrayQuoteContainer(s, idx, qEnd, end, st); last >= 0 {
				// The quoted string encloses the containers after a
				// delimiter in it, as in classifyNestedKey.
				return gluedAfterValue, last
			}
			if st != nil && st.openerBudgetExhausted {
				return gluedAfterValue, len(s)
			}
			// The scan goes on past a quote that no check places, as in
			// classifyNestedKey.
			unplaced = true
			continue
		case '{', '[':
			if unplaced && idx != strayOuter {
				return notNestedKey, 0
			}
			if idx > start && end < len(s) && (delimBeforePos(s[start:idx], idx-start) > 0 || idx == strayOuter) {
				cEnd, closed := scanBalancedContainer(s, idx, end)
				if closed {
					idx = cEnd - 1
					continue
				}
				return gluedAfterValue, idx
			}
		}
	}
	if unplaced {
		return notNestedKey, 0
	}
	if strings.HasSuffix(v, ":") || strings.HasSuffix(v, "=") {
		if okVal, gluedEnd := nestedValueAfter(s, end, openQ, st); okVal {
			if gluedEnd > 0 {
				return gluedAfterValue, gluedEnd
			}
			return gluedAfterValue, valAfterDelim(s, end)
		}
	}
	if followedByDelimiter(s, end) {
		if okVal, gluedEnd := nestedValueFollowsDelimiter(s, end, openQ, st); okVal {
			if gluedEnd > 0 {
				return gluedAfterValue, gluedEnd
			}
			return gluedAfterValue, valAfterDelim(s, skipDelimSpaceFwd(s, end)+1)
		}
	}
	return notNestedKey, 0
}

func delimBeforePos(v string, idx int) int {
	p := idx
	for p > 0 && isDelimSpace(v[p-1]) {
		p--
	}
	if p > 0 && (v[p-1] == ':' || v[p-1] == '=') {
		return p - 1
	}
	return -1
}

func isGluedTokenAfterQuote(s string, end int) bool {
	if end >= len(s) {
		return false
	}
	c := s[end]
	return !isTextValueStop(c) && c != '}' && c != ']' && c != ')' && c != '.'
}

// innerContinuesIntoGluedToken reports whether inner, the text in the quotes of
// a string glued to the word after its closing quote, ends in a pair whose
// text before the ":" or "=" ends in a key name and whose value holds no
// text-value stop, so that the value runs on into the glued word. The text
// before each such delimiter is read in full, so its length is charged to st,
// and once st runs out of budget it reports true, so that the caller fails
// closed.
func innerContinuesIntoGluedToken(inner string, st *quoteScanState) bool {
	// lastStop is where the last text-value stop in inner is: no stop
	// follows a delimiter whose value starts past it.
	lastStop := len(inner) - 1
	for lastStop >= 0 && !isTextValueStop(inner[lastStop]) {
		lastStop--
	}
	for d := 1; d < len(inner); d++ {
		if inner[d] != ':' && inner[d] != '=' {
			continue
		}
		if skipDelimSpaceFwd(inner, d+1) <= lastStop {
			continue
		}
		left := inner[:d]
		for len(left) > 0 && isDelimSpace(left[len(left)-1]) {
			left = left[:len(left)-1]
		}
		if left == "" {
			continue
		}
		if !st.consumeOpenerBudget(len(left)) {
			return true
		}
		if kind, _ := nestedKeyName(left, 0, true); kind != notNestedKey {
			return true
		}
		unq := left
		strippedEscQuote := false
		if strings.HasSuffix(unq, `\"`) || strings.HasSuffix(unq, `\'`) {
			unq = unq[:len(unq)-2]
			strippedEscQuote = true
		} else if unq[len(unq)-1] == '"' || unq[len(unq)-1] == '\'' {
			unq = unq[:len(unq)-1]
		}
		if isValidJSONPatternKey(trimInlineSpaces(unq)) || findTextKeyStart(unq, 0, len(unq)) >= 0 {
			return true
		}
		if strippedEscQuote && findTextKeyStart(left[:len(left)-1], 0, len(left)-1) >= 0 {
			return true
		}
	}
	return false
}

// isDottedKeyFilename reports whether s[keyStart:keyEnd] is a dotted ".key"
// filename segment (such as "private.key" or "secret.key") rather than a
// "temp:" key or a dotted compound credential ending in a sensitive key word
// (such as "db.password", "github.token" or "client.secret").
func isDottedKeyFilename(s string, keyStart, keyEnd int) bool {
	k := s[keyStart:keyEnd]
	if hasFoldPrefix(k, session.KeyPrefixTemp) {
		return false
	}
	lastDot := strings.LastIndexByte(k, '.')
	return lastDot > 0 && strings.EqualFold(k[lastDot+1:], "key")
}

// isPathErrorOpBefore reports whether s[floor:keyStart] ends with a Go
// *os.PathError or file-operation verb followed by horizontal spaces directly
// before a relative ".key" filename (such as "open private.key: no such file").
func isPathErrorOpBefore(s string, floor, keyStart int) bool {
	if keyStart <= floor {
		return false
	}
	wEnd := keyStart
	for wEnd > floor && (s[wEnd-1] == ' ' || s[wEnd-1] == '\t') {
		wEnd--
	}
	if wEnd == keyStart {
		return false
	}
	wStart := wEnd
	for wStart > floor && (('a' <= s[wStart-1] && s[wStart-1] <= 'z') || ('A' <= s[wStart-1] && s[wStart-1] <= 'Z')) {
		wStart--
	}
	if wStart == wEnd {
		return false
	}
	if wStart > floor {
		prev := s[wStart-1]
		if isWordByte(prev) || prev == '.' || prev == '-' || prev == '/' || prev == '\\' {
			return false
		}
	}
	switch strings.ToLower(s[wStart:wEnd]) {
	case "open", "openat", "create", "stat", "lstat", "fstat",
		"read", "write", "readfile", "writefile", "readlink",
		"remove", "removeall", "rename", "chmod", "chown", "lchown",
		"chtimes", "truncate", "mkdir", "mkdirall", "rmdir", "chdir",
		"readdir", "symlink", "link", "load", "parse":
		return true
	default:
		return false
	}
}

// isPathPrefixedDottedKey reports whether the bare key s[keyStart:keyEnd] is a
// dotted ".key" filename segment in a file path (immediately preceded by "/" or
// "\", excluding URI "://", or directly following a Go file-operation verb such
// as "open " in an *os.PathError), such as "/etc/ssl/private.key" or
// "open private.key".
func isPathPrefixedDottedKey(s string, floor, keyStart, keyEnd int) bool {
	if !isDottedKeyFilename(s, keyStart, keyEnd) {
		return false
	}
	if keyStart > 0 && (s[keyStart-1] == '/' || s[keyStart-1] == '\\') {
		return !(s[keyStart-1] == '/' && keyStart >= 3 && s[keyStart-3:keyStart] == "://")
	}
	return isPathErrorOpBefore(s, floor, keyStart)
}

// isSkippedPathDottedPair reports whether the pair starting at pair in s is an
// unquoted path-prefixed dotted ".key" filename with a ":" delimiter that
// redactTextKeyValues skips (see isPathPrefixedDottedKey).
func isSkippedPathDottedPair(s string, pair int) bool {
	if pair <= 0 || pair >= len(s) || s[pair] == '"' || s[pair] == '\'' || s[pair] == '\\' {
		return false
	}
	end := pair
	for end < len(s) {
		if c := s[end]; c < utf8.RuneSelf {
			if !isCompoundKeyByte(c) {
				break
			}
			end++
			continue
		}
		r, size := utf8.DecodeRuneInString(s[end:])
		if r != '\u0130' && r != '\u017f' && r != '\u212a' {
			break
		}
		end += size
	}
	if end == pair || !isPathPrefixedDottedKey(s, 0, pair, end) {
		return false
	}
	return isPathErrorColon(s, skipDelimSpaceFwd(s, end))
}

// isPathErrorColon reports whether s[i] is a ":" that may end the path of a
// Go *os.PathError, "op path: err": one not directly followed by the quote
// of a quoted value (", ', \" or \') or by a container ({ or [), as in
// /etc/ssl/private.key:'SECRET', which is a key-value pair instead.
func isPathErrorColon(s string, i int) bool {
	if i >= len(s) || s[i] != ':' {
		return false
	}
	if i+1 < len(s) {
		switch s[i+1] {
		case '"', '\'', '{', '[':
			return false
		case '\\':
			return i+2 >= len(s) || (s[i+2] != '"' && s[i+2] != '\'')
		}
	}
	return true
}

// isBareKeyPairAt reports whether a pair whose key is made only of compound
// key bytes (see isCompoundKeyByte), with no quote, starts at s[pair], so
// that redactTextKeyValues matches it on its own.
func isBareKeyPairAt(s string, pair int) bool {
	i := pair
	for i < len(s) && isCompoundKeyByte(s[i]) {
		i++
	}
	return i > pair && followedByDelimiter(s, i)
}

// isJSONColonAfter reports whether s[i:] begins with ":" after optional JSON
// whitespace, so redactJSONKeyValues will match a preceding double-quoted key.
func isJSONColonAfter(s string, i int) bool {
	for i < len(s) && isRegexpSpace(s[i]) {
		i++
	}
	return i < len(s) && s[i] == ':'
}

// scanBalancedContainer scans the "{" or "[" container starting at s[start] up
// to limit, ignoring brackets inside strings (including \" and \'), and returns
// the index immediately after its matching closer and true when it closes
// within s[start:limit], or (limit, false) when it remains open at limit.
func scanBalancedContainer(s string, start, limit int) (end int, closed bool) {
	depth := 0
	// quote is the quote of the string being skipped. escaped reports that
	// the string was opened with an escaped quote (\" or \'); only the same
	// escaped quote closes it (see scanEscapedQuoted).
	var quote byte
	escaped := false
	for i := start; i < limit; i++ {
		c := s[i]
		switch {
		case escaped:
			if c != '\\' || i+1 == limit {
				break
			}
			switch s[i+1] {
			case quote:
				quote, escaped = 0, false
				i++
			case '\\':
				// An escape of the inner text. The escaped character may
				// itself be escaped, as in \\\" or \\\\.
				i += 2
				if i < limit && s[i] == '\\' {
					i++
				}
			default:
				i++
			}
		case c == '\\':
			if quote == 0 && i+1 < limit && (s[i+1] == '"' || s[i+1] == '\'') {
				quote, escaped = s[i+1], true
			}
			i++
		case quote != 0:
			if c == quote {
				quote = 0
			}
		case c == '"' || c == '\'':
			quote = c
		case c == '{' || c == '[':
			depth++
		case c == '}' || c == ']':
			depth--
			if depth == 0 {
				return i + 1, true
			}
		}
	}
	return limit, false
}

// skipStatePrefixes advances i past any leading "user:" or "app:" session state
// prefixes in name.
func skipStatePrefixes(name string, i int) int {
	for i < len(name) {
		if hasFoldPrefix(name[i:], session.KeyPrefixUser) {
			i += strings.IndexByte(name[i:], ':') + 1
			continue
		}
		if hasFoldPrefix(name[i:], session.KeyPrefixApp) {
			i += strings.IndexByte(name[i:], ':') + 1
			continue
		}
		break
	}
	return i
}

// hasValidCompoundPrefix reports whether prefix is a non-empty prefix before a
// nested compound or path-like key: free of delimiter whitespace and ending in
// a non-word byte (including "/" or "\" path separators).
func hasValidCompoundPrefix(prefix string) bool {
	if prefix == "" || isWordByte(prefix[len(prefix)-1]) {
		return false
	}
	for i := 0; i < len(prefix); i++ {
		if isDelimSpace(prefix[i]) {
			return false
		}
	}
	return true
}

func isControlDelimSpace(c byte) bool {
	return isDelimSpace(c) && !isRegexpSpace(c)
}

func trimControlDelimSpaceSpan(name string, at int) (string, int) {
	lead := 0
	for lead < len(name) && isControlDelimSpace(name[lead]) {
		lead++
	}
	end := len(name)
	for end > lead && isControlDelimSpace(name[end-1]) {
		end--
	}
	return name[lead:end], at + lead
}

func isStackedBareTempKey(s string) bool {
	if !hasFoldPrefix(s, session.KeyPrefixTemp) {
		return false
	}
	rest := s[len(session.KeyPrefixTemp):]
	if rest == "" || strings.HasPrefix(rest, ":") || strings.HasSuffix(rest, ":") || strings.Contains(rest, "::") {
		return false
	}
	hasInnerColon := false
	for i := 0; i < len(rest); i++ {
		if rest[i] == ':' {
			hasInnerColon = true
			continue
		}
		if !isTempKeyBodyByte(rest[i]) || isDelimSpace(rest[i]) {
			return false
		}
	}
	return hasInnerColon
}

// nestedKeyName classifies a value that may start a nested pair (see
// nestsSensitiveKey): name is the name of that pair, which begins at offset
// at of the text, and lineStart reports whether the value starts a line.
// Unless the value is notNestedKey, nestedKeyName also returns where the
// sensitive key name in name, and so the pair that it starts, begins in the
// text.
func nestedKeyName(name string, at int, lineStart bool) (nestedKeyKind, int) {
	return nestedKeyNameStrip(name, at, lineStart, true)
}

// nestedKeyNameStrip is nestedKeyName, which, when stripQuotes is true, also
// reads a name that ends in closing quotes that close no quoted name in it
// (see the end of the function).
func nestedKeyNameStrip(name string, at int, lineStart, stripQuotes bool) (nestedKeyKind, int) {
	name, at = trimControlDelimSpaceSpan(name, at)
	if name == "" {
		return notNestedKey, 0
	}
	if _, ok := sensitiveKeys[name]; ok {
		return nestedKey, at
	}
	if m := nestedKeyNameRe.FindStringSubmatchIndex(name); m != nil {
		return nestedKey, at + m[2]
	}
	q := leadingOptionalQuoteLen(name)
	stateStart := skipStatePrefixes(name, q)
	if m := compoundKeyNameRe.FindStringSubmatchIndex(name); m != nil {
		prefixAfterState := name[min(stateStart, m[2]):m[2]]
		// The expression reads only ASCII whitespace as \s, so its prefix
		// may hold another space or control character, as the U+00A0 of
		// SECRET<U+00A0>x/signing_key; a key name on a line of its own holds
		// none, so such a name only ends in a key name.
		if lineStart && (q == 0 || m[2] == q) && !strings.ContainsAny(name[:m[2]], "&;|") && !hasSpaceOrControl(name[:m[2]]) &&
			(!strings.Contains(prefixAfterState, ":") || hasFoldPrefix(prefixAfterState, session.KeyPrefixTemp) || (q == 0 && isPathLikePrefix(name[:m[2]]))) {
			return nestedKey, at + m[2]
		}
		return valueEndingInKeyName, at + m[2]
	}
	if kind, pair := trailingQuotedSensitiveKey(name, at); kind != notNestedKey {
		return kind, pair
	}
	if isStackedBareTempKey(name[stateStart:]) && (stateStart == q || hasValidCompoundPrefix(name[q:stateStart])) {
		if lineStart && (q == 0 || stateStart == q) && !strings.ContainsAny(name[:stateStart], "&;|") {
			return nestedKey, at + stateStart
		}
		return valueEndingInKeyName, at + stateStart
	}
	endTrim := len(name)
	if endTrim > q && (name[endTrim-1] == '"' || name[endTrim-1] == '\'') {
		endTrim--
	}
	tk := endTrim
	for tk > q && isTempKeyBodyByte(name[tk-1]) && !isDelimSpace(name[tk-1]) {
		tk--
	}
	if tk < endTrim && tk-len(session.KeyPrefixTemp) >= q &&
		hasFoldPrefix(name[tk-len(session.KeyPrefixTemp):tk], session.KeyPrefixTemp) {
		tStart := tk - len(session.KeyPrefixTemp)
		if tStart == stateStart || hasValidCompoundPrefix(name[q:tStart]) {
			if lineStart && endTrim == len(name) && (tStart == stateStart || (q == 0 && isPathLikePrefix(name[:tStart]))) {
				return nestedKey, at + tStart
			}
			return valueEndingInKeyName, at + tStart
		}
	}
	off := len(name)
	for off > q {
		step := isCompoundKeyRuneBack(name, q, off)
		if step == 0 {
			break
		}
		off -= step
	}
	if off < len(name) && (off == stateStart || hasValidCompoundPrefix(name[q:off])) {
		if kStart := sensitiveKeyStart(name, off, len(name)); kStart >= 0 {
			if lineStart && (off == stateStart || (q == 0 && isPathLikePrefix(name[:kStart]) && isBareSensitiveKeyName(name[kStart:]))) {
				return nestedKey, at + kStart
			}
			return valueEndingInKeyName, at + kStart
		}
	}
	// A name that ends in an unescaped closing quote that closes no quoted
	// name in it (see trailingQuotedSensitiveKey), as the password" of
	// DB_PASSWORD: password": SECRET, names the key before that quote, as the
	// optional quote after the key in the legacy regular expression did. So
	// does a quoted name followed by a stray closing quote, as the " token""
	// of " token"": SECRET. At most one such quote is stripped, once, so that
	// the name is read at most twice. A name ending in an escaped quote, as
	// in abc\"password\", is left to the escaped-quote checks of
	// scrubErrorText, which fail closed.
	if n := len(name); stripQuotes && n > q+1 && isQuoteByte(name[n-1]) && name[n-2] != '\\' {
		return nestedKeyNameStrip(name[:n-1], at, lineStart, false)
	}
	return notNestedKey, 0
}

// hasSpaceOrControl reports whether s holds whitespace, a control character
// or another Unicode space.
func hasSpaceOrControl(s string) bool {
	for i := 0; i < len(s); i++ {
		if c := s[i]; isDelimSpace(c) {
			return true
		} else if c >= utf8.RuneSelf {
			return strings.IndexFunc(s[i:], func(r rune) bool { return unicode.IsSpace(r) || unicode.In(r, unicode.Z) }) >= 0
		}
	}
	return false
}

func isPathLikePrefix(prefix string) bool {
	if prefix == "" || strings.ContainsAny(prefix, "&;|\"'=") {
		return false
	}
	if prefix[0] == '/' || prefix[0] == '\\' {
		return true
	}
	return len(prefix) >= 3 &&
		(('a' <= prefix[0] && prefix[0] <= 'z') || ('A' <= prefix[0] && prefix[0] <= 'Z')) &&
		prefix[1] == ':' && (prefix[2] == '\\' || prefix[2] == '/')
}

// isJoinBeforeQuotedKey reports whether c is a non-word punctuation joiner
// (such as "&", ";", "|", "/" or "+") that separates an earlier value from a
// trailing escaped-quoted key, matching compoundKeyNameRe's [^\w\s"'\\] boundary.
func isJoinBeforeQuotedKey(c byte) bool {
	return !isWordByte(c) && !isDelimSpace(c) && c != '"' && c != '\'' && c != '\\'
}

// trailingQuotedSensitiveKey checks whether name ends with a quoted sensitive
// key (", ', \" or \'), such as abc&"token", abcX"token" or ;'client_secret',
// and if so returns valueEndingInKeyName and the offset of the opening quote
// in s.
func trailingQuotedSensitiveKey(name string, at int) (nestedKeyKind, int) {
	n := len(name)
	if n >= 4 && name[n-2] == '\\' && (name[n-1] == '"' || name[n-1] == '\'') && isUnescapedQuote(name, 0, n-2) {
		q := name[n-1]
		for openIdx := n - 4; openIdx >= 0; openIdx-- {
			if name[openIdx] == '\\' && name[openIdx+1] == q && isUnescapedQuote(name, 0, openIdx) {
				if openIdx == 0 || isJoinBeforeQuotedKey(name[openIdx-1]) {
					trimmed := trimInlineSpaces(name[openIdx+2 : n-2])
					if isValidJSONPatternKey(trimmed) {
						return valueEndingInKeyName, at + openIdx
					}
					if kind, _ := nestedKeyName(trimmed, 0, false); kind != notNestedKey {
						return valueEndingInKeyName, at + openIdx
					}
				}
				break
			}
		}
		return notNestedKey, 0
	}
	if n >= 2 && (name[n-1] == '"' || name[n-1] == '\'') && isUnescapedQuote(name, 0, n-1) {
		q := name[n-1]
		for openIdx := n - 2; openIdx >= 0; openIdx-- {
			if name[openIdx] == q && isUnescapedQuote(name, 0, openIdx) {
				trimmed := trimInlineSpaces(name[openIdx+1 : n-1])
				if isValidJSONPatternKey(trimmed) {
					return valueEndingInKeyName, at + openIdx
				}
				if kind, _ := nestedKeyName(trimmed, 0, false); kind != notNestedKey {
					return valueEndingInKeyName, at + openIdx
				}
				break
			}
		}
	}
	return notNestedKey, 0
}

func leadingOptionalQuoteLen(name string) int {
	if strings.HasPrefix(name, `\"`) || strings.HasPrefix(name, `\'`) {
		return 2
	}
	if len(name) > 0 && (name[0] == '"' || name[0] == '\'') {
		return 1
	}
	return 0
}

// nestedPairEndWithState returns where the text to redact with a value that
// only ends in a key name ends (see nestsSensitiveKey), given pair, where the
// pair that the key name starts begins in s: at the end of the value of that
// pair, or, if that value is itself the name of a pair, as in
// "password=a-token: token: x", at the end of the value of that pair, and so
// on. open is the number of brackets opened before the key of the value, openQ
// the string open around it, and st the scan state of s; once st runs out of
// scan budget, the text to redact runs to the end of s.
func nestedPairEndWithState(s string, pair, open int, openQ openQuoteState, st *quoteScanState) int {
	for {
		if st != nil && st.openerBudgetExhausted {
			return len(s)
		}
		delimAfter, start, end, ok := scanTextKeyValAt(s, pair, st)
		if !ok {
			return failClosedNestedPairEnd(s, pair, open, openQ, st)
		}
		kind, next := nestsSensitiveKey(s, start, end, startsLine(s, delimAfter, start), openQ, st)
		if st != nil && st.openerBudgetExhausted {
			return len(s)
		}
		switch kind {
		case notNestedKey:
			if realEnd, ok := quotedValueRealEnd(s, start, end, open, openQ, st); ok {
				return max(end, realEnd)
			}
			if _, quote := unquoteValue(s[start:end]); quote != "" &&
				hasGluedSuffixAfterQuotedValue(s, start, end, openQ, st) {
				return max(end, extendGluedValueEnd(s, end, open, openQ, st))
			}
			return textValueEnd(s, start, end, open)
		case gluedAfterValue:
			return max(end, extendGluedValueEnd(s, next, open, openQ, st))
		default:
			pair = next
		}
	}
}

func valAfterDelim(s string, afterDelim int) int {
	vPos := skipDelimSpaceFwd(s, afterDelim)
	if (vPos >= len(s) || isTextValueStop(s[vPos])) && vPos > afterDelim && !isTextValueStop(s[vPos-1]) {
		vPos--
	}
	return vPos
}

func stolenQuotedKeyAfterQuoteEnd(s string, cur int, openQ openQuoteState, st *quoteScanState) (int, bool) {
	if cur <= 0 || cur >= len(s) || (s[cur-1] != '"' && s[cur-1] != '\'') {
		return 0, false
	}
	closePos := cur - 1
	if cur >= 2 && s[cur-2] == '\\' {
		bs := 0
		for k := cur - 2; k >= 0 && s[k] == '\\'; k-- {
			bs++
		}
		if bs&1 == 1 {
			closePos = cur - 2
		}
	}
	keyQEnd, ok := scanQuotedAt(s, closePos, st)
	if !ok || keyQEnd <= cur {
		return 0, false
	}
	inner, quote := unquoteValue(s[closePos:keyQEnd])
	if quote == "" || !isPaddedQuotedTextKey(trimInlineSpaces(inner)) {
		return 0, false
	}
	okVal, gluedEnd := nestedValueFollowsDelimiter(s, keyQEnd, openQ, st)
	if !okVal {
		return 0, false
	}
	if gluedEnd > cur {
		return gluedEnd, true
	}
	nextVal := valAfterDelim(s, skipDelimSpaceFwd(s, keyQEnd)+1)
	if nextVal <= cur {
		return 0, false
	}
	return nextVal, true
}

func danglingClosedKeyAfterQuoteEnd(s string, cur int, openQ openQuoteState, st *quoteScanState) (int, bool) {
	if cur <= 0 || cur >= len(s) || (s[cur-1] != '"' && s[cur-1] != '\'') {
		return 0, false
	}
	i := cur
	var keyEnd int
	if hasFoldPrefix(s[i:], session.KeyPrefixTemp) {
		i += len(session.KeyPrefixTemp)
		bodyStart := i
		for i < len(s) && isTempKeyBodyByte(s[i]) {
			i++
		}
		if i == bodyStart {
			return 0, false
		}
		keyEnd = i
		if i+1 < len(s) && s[i] == '\\' && (s[i+1] == '"' || s[i+1] == '\'') {
			i += 2
		} else if i < len(s) && (s[i] == '"' || s[i] == '\'') {
			i++
		} else {
			return 0, false
		}
	} else {
		// The key may be joined to the text before it by a path separator,
		// other punctuation or a quote, escaped or not, that opens no string,
		// as in 'foo 'x/token": SECRET or 'foo 'a\"token': SECRET. Its closing
		// quote, possibly escaped, is the one right before the delimiter, and
		// findTextKeyStart requires a sensitive key name at a word boundary
		// before it, as the \b of the legacy regular expression did.
		for i < len(s) && isDanglingKeyPathByte(s[i]) {
			i++
		}
		keyEnd = i
		if keyEnd <= cur || (s[keyEnd-1] != '"' && s[keyEnd-1] != '\'') {
			return 0, false
		}
		keyEnd--
		if !isUnescapedQuote(s, cur, keyEnd) {
			keyEnd--
		}
		if keyEnd <= cur {
			return 0, false
		}
	}
	if findTextKeyStart(s, cur, keyEnd) < 0 {
		return 0, false
	}
	okVal, gluedEnd := nestedValueFollowsDelimiter(s, i, openQ, st)
	if !okVal {
		return 0, false
	}
	if gluedEnd > cur {
		return gluedEnd, true
	}
	nextVal := valAfterDelim(s, skipDelimSpaceFwd(s, i)+1)
	if nextVal <= cur {
		return 0, false
	}
	return nextVal, true
}

// quotedValueRealEnd handles a quoted value s[valStart:end] whose closing
// quote is really the opening quote of a key or a value nested in it, or a
// quote inside the containers that the value encloses, returning where the
// value really ends and true:
//
//   - when the closing quote opens a member of a "{" or "[" container that
//     the value opens (see containerMemberOpener), as the {"k": of
//     "a:token:{"k": "SECRET"}, the { "k": of "a:token:{ "k": "SECRET"} or
//     the [1,"SECRET"] of "a:token:[1,"SECRET"], the value runs on to the end
//     of that container (or of s, when it is not closed); when the opening
//     quote closes the string right after the outermost container open at
//     that quote and the containers that follow it (see
//     enclosedContainerRun), as in "{'a': "1"} {'b': 'SECRET'}", the value
//     runs on to that quote;
//   - when a token is glued to the closing quote (see isGluedTokenAfterQuote)
//     or the closing quote comes inside a member of a container that opens
//     in the value (see closingQuoteInMember), which then comes inside
//     containers that open right after the opening quote or after a ":" or
//     "=" in the value, and the opening quote closes the string right after
//     them (see quotedValueContainerRun), as the "hi" of
//     "{'msg': 'say "hi"', 'k': 'SECRET'}", the " hi" of
//     "{'msg': 'say " hi"', 'k': 'SECRET'}" or the it's of
//     'S0/x: {"msg": "it's ok", "k": "SECRET"}', the value runs on to that
//     quote; a value such as "{abc" that a stop follows is not scanned, so
//     that many of them stay linear;
//   - when the value ends in a sensitive key name, possibly quoted and padded
//     as ' api.key ', and a delimiter followed by a line break, as in
//     "a token:\n  " SECRET"tail, or by optional spaces and the closing quote
//     with text glued after it, as in "a password="SECRET" or
//     "a token: "x SECRET", the quote opens the value of that key, and the
//     value runs on to the end of that nested value (see
//     extendGluedValueEnd); when another string follows the closing quote
//     after spaces, as in "' api.key ':\t" "SECRET", that string is the
//     value.
//
// Otherwise it returns false.
func quotedValueRealEnd(s string, valStart, end, open int, openQ openQuoteState, st *quoteScanState) (int, bool) {
	inner, quote := unquoteValue(s[valStart:end])
	if quote == "" {
		return 0, false
	}
	t := len(inner)
	for t > 0 && isDelimSpace(inner[t-1]) {
		t--
	}
	if t == 0 {
		return 0, false
	}
	qPos := end - len(quote)
	brace, outer := quotedContainerOpeners(s, valStart, end, st)
	last := -1
	if brace >= 0 {
		last, _ = enclosedContainerRun(s, valStart, end, outer, st)
	} else if o, ok := closingQuoteInMember(s, valStart, end, st); ok {
		// The containers start at the outermost one open at the closing
		// quote, wherever it opens in the value, as the { of
		// "prefix {'a': 'say " hi', 'k': 'SECRET'}" or of
		// "({'a': 'say " hi', 'k': 'SECRET'})".
		last, _ = enclosedContainerRun(s, valStart, end, o, st)
	} else if isGluedTokenAfterQuote(s, end) {
		last = quotedValueContainerRun(s, valStart, end, st)
	}
	if st != nil && st.openerBudgetExhausted {
		return len(s), true
	}
	if last >= 0 {
		// The opening quote closes the string right after the last
		// container, possibly after delimiter whitespace and the ")" of
		// groups around the containers (see enclosedContainerRun), and that
		// quote ends the value, so that it cannot open a string with a later
		// quote.
		cEnd, _ := scanBalancedContainer(s, last, len(s))
		closeQuote := skipGroupClosers(s, cEnd)
		if !st.consumeOpenerBudget(closeQuote - last) {
			return len(s), true
		}
		return closeQuote + len(quote), true
	}
	if brace >= 0 {
		cEnd, _ := scanBalancedContainer(s, brace, len(s))
		return cEnd, true
	}
	if c := inner[t-1]; c != ':' && c != '=' {
		return 0, false
	}
	if !strings.ContainsAny(inner[t:], "\r\n") && !isGluedTokenAfterQuote(s, end) && !isStringAfterSpaces(s, end) {
		return 0, false
	}
	nameEnd := t - 1
	for nameEnd > 0 && isDelimSpace(inner[nameEnd-1]) {
		nameEnd--
	}
	nameStart := nameEnd
	for nameStart > 0 && !isDelimSpace(inner[nameStart-1]) {
		nameStart--
	}
	kind, _ := nestedKeyName(inner[nameStart:nameEnd], 0, false)
	if kind == notNestedKey && nameEnd > 0 && isQuoteByte(inner[nameEnd-1]) {
		// A quoted key padded with spaces, as ' api.key '.
		if open := strings.LastIndexByte(inner[:nameEnd-1], inner[nameEnd-1]); open >= 0 {
			if name := trimInlineSpaces(inner[open+1 : nameEnd-1]); name != "" {
				kind, _ = nestedKeyName(name, 0, false)
			}
		}
	}
	if kind == notNestedKey {
		return 0, false
	}
	if next := skipInlineSpacesFwd(s, end, len(s)); next > end && isStringAfterSpaces(s, end) {
		// The value is the next string, as in "' api.key ':\t" "SECRET".
		if nEnd, ok := scanQuotedAt(s, next, st); ok {
			return max(nEnd, extendGluedValueEnd(s, nEnd, open, openQ, st)), true
		}
	}
	return extendGluedValueEnd(s, qPos, open, openQ, st), true
}

// quotedValueContainerRun returns where the last of the "{" or "["
// containers that the quoted value s[q:qEnd] encloses starts, when the first
// of them opens before the closing quote of s[q:qEnd], right after the opening
// quote or after a ":" or "=" in the value, in either case possibly after
// delimiter whitespace, and the value seems to close inside them (see
// enclosedContainerRun), as the { of "{'msg': 'say "hi"', 'k': 'SECRET'}", of
// 'S0/x: {"msg": "it's ok", "k": "SECRET"}' or of ": ['say "hi"', 'SECRET']";
// otherwise it returns -1. Containers that stay open, that the opening quote
// does not close the string right after, or that it closes right after, as in
// "{a: 1}", leave the value as it is. The openers are tried in order, each
// past the first container of the one before, and every byte read is charged
// to st, so that repeated unclosed containers stay linear; once st runs out of
// budget it returns -1 and the caller fails closed.
func quotedValueContainerRun(s string, q, qEnd int, st *quoteScanState) int {
	_, quote := unquoteValue(s[q:qEnd])
	if quote == "" {
		return -1
	}
	from, seemsClose := q+len(quote), qEnd-len(quote)
	if !st.consumeOpenerBudget(seemsClose - from) {
		return -1
	}
	opener := skipDelimSpaceFwd(s, from)
	for opener < seemsClose {
		if s[opener] == '{' || s[opener] == '[' {
			last, firstEnd := enclosedContainerRun(s, q, qEnd, opener, st)
			if last >= 0 || firstEnd < 0 || firstEnd >= seemsClose {
				return last
			}
			opener = firstEnd
		}
		d := strings.IndexAny(s[opener:seemsClose], ":=")
		if d < 0 {
			return -1
		}
		opener = skipDelimSpaceFwd(s, opener+d+1)
	}
	return -1
}

// closingQuoteInMember reports whether the closing quote of the quoted string
// s[q:qEnd] comes inside a member of a "{" or "[" container that opens in the
// string, after any text, and is still open at that quote (see
// opensMemberAfter), and returns where the outermost container open at that
// quote starts: inside a string in other quotes that opens a member or a part
// of one there and closes past that quote, as the 'say " hi"' of
// "{'msg': 'say " hi"', 'k': 'SECRET'}", the 'say " hi' of
// "{'a': ('say " hi'), 'k': 'SECRET'}" or of
// "{'a': 'x' + 'say " hi', 'k': 'SECRET'}", or right where a member or a part
// of one starts, also after a string prefix, the closing quote then opening a
// string, as the " 1\n2" of "{'a': " 1\n2", 'k': 'SECRET'}", the " a" of
// "{ " a", 'SECRET'}" or the b" SECRET" of "{'k': b" SECRET", 'm': 1}" or of
// "{'a': 'x' b" SECRET", 'm': 1}". Either string may hold line breaks and must
// end a member or a part of one (see memberStringEnds). Strings that close
// before the closing quote are skipped, a backslash escapes only a quote or a
// backslash, so that the { of "\{'a': 1}" opens a container, and a quote
// joined to a word, as in it's, opens no string unless that word is a string
// prefix (see isStringPrefixQuote), so that "{abc", "{it's}" and the "{" of
// "{" user=bob are not such values, and their quoted values are not scanned
// for containers. Every byte read is charged to st.
func closingQuoteInMember(s string, q, qEnd int, st *quoteScanState) (int, bool) {
	_, quote := unquoteValue(s[q:qEnd])
	if quote == "" {
		return -1, false
	}
	from, qPos := q+len(quote), qEnd-len(quote)
	if !st.consumeOpenerBudget(qPos - from) {
		return -1, false
	}
	outerQuote, outerEscaped := quote[len(quote)-1], len(quote) == 2
	var stack []byte
	// outer is where the outermost container in stack starts.
	outer := -1
	// prev is the last byte read other than delimiter whitespace, or
	// closedString after a string, and wordPrev is the one before the word
	// that ends at prev.
	prev, wordPrev := byte(0), byte(0)
	for i := from; i < qPos; i++ {
		switch c := s[i]; c {
		case '\\':
			if i+1 < qPos && (isQuoteByte(s[i+1]) || s[i+1] == '\\') {
				i++
			}
			prev = '\\'
		case '{', '[':
			if len(stack) == 0 {
				outer = i
			}
			stack = append(stack, c)
			prev = c
		case '}', ']':
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
			prev = c
		case '"', '\'':
			memberPrev := prev
			if i > from && isWordByte(s[i-1]) {
				if !isStringPrefixQuote(s, from, i, st) {
					prev = c
					continue
				}
				memberPrev = wordPrev
			}
			if c == outerQuote && !outerEscaped {
				prev = c
				continue
			}
			end, ok := scanSimpleQuoted(s, i, c, st)
			if !ok {
				prev = c
				continue
			}
			if end > qPos {
				if opensMemberAfter(stack, memberPrev) && memberStringEnds(s, qPos, end, st) {
					return outer, true
				}
				return -1, false
			}
			i = end - 1
			prev = closedString
		default:
			if !isDelimSpace(c) {
				if isWordByte(c) && (i == from || !isWordByte(s[i-1])) {
					wordPrev = prev
				}
				prev = c
			}
		}
	}
	memberPrev := prev
	if qPos > from && isWordByte(s[qPos-1]) {
		// The closing quote right after a string prefix, as the b" of
		// "{'k': b" SECRET", 'm': 1}", starts a member after the byte before
		// that prefix.
		if stringPrefixStart(s, from, qPos) < 0 {
			return -1, false
		}
		memberPrev = wordPrev
	}
	if !opensMemberAfter(stack, memberPrev) {
		return -1, false
	}
	if end, ok := scanQuotedAt(s, qPos, st); ok && memberStringEnds(s, qPos, end, st) {
		return outer, true
	}
	return -1, false
}

// closedString stands for the closing quote of a string that
// closingQuoteInMember skips, as the last byte it read: that byte is never
// delimiter whitespace, which takes in every control byte, so closedString
// is never a byte that it reads.
const closedString byte = 0x01

// opensMemberAfter reports whether a member of the innermost of the open
// containers in stack, or a part of one, starts after prev, the last byte
// before it other than delimiter whitespace: after the "{" or "[" that opens
// it, a "," or a "(", which may open a tuple, after the ":" or "=" of a key of
// a "{" container, or after a "+" or another string (closedString), which
// add the part to the member, as the 'say " hi' of 'x' + 'say " hi' and of
// the implicit concatenation 'x' 'say " hi' does.
func opensMemberAfter(stack []byte, prev byte) bool {
	if len(stack) == 0 {
		return false
	}
	switch prev {
	case '{', '[', ',', '(', '+', closedString:
		return true
	case ':', '=':
		return stack[len(stack)-1] == '{'
	}
	return false
}

// memberStringEnds reports whether s[:end], the end of a string that runs on
// past from, ends a member or a part of one, possibly before delimiter
// whitespace: whether a ",", ":", "=", "}", "]" or ")", a "+" or another string,
// with or without a string prefix, follows it, as the ) of (b'x}y'), the + of
// b'x}y' + 'z' or the 'z' of b'x}y' 'z' do, or, right after its closing
// quote, a subscript or an attribute, as the [0] of b'x}y'[0] or the .decode
// of b'x}y'.decode() does. The bytes from from on are charged to st.
func memberStringEnds(s string, from, end int, st *quoteScanState) bool {
	d := skipDelimSpaceFwd(s, end)
	if !st.consumeOpenerBudget(d - from) {
		return false
	}
	if d >= len(s) {
		return false
	}
	if strings.IndexByte(",:=}])+", s[d]) >= 0 || isQuoteByte(s[d]) {
		return true
	}
	if d == end && (s[d] == '[' || (s[d] == '.' && d+1 < len(s) && isWordByte(s[d+1]))) {
		return true
	}
	q := d
	for q < len(s) && q-d < 3 && isWordByte(s[q]) {
		q++
	}
	return q < len(s) && isQuoteByte(s[q]) && stringPrefixStart(s, d, q) == d
}

// isStringPrefixQuote reports whether the quote at s[i] follows a Python string
// prefix that makes up the whole word before it (see stringPrefixStart) and
// opens a string that closes, possibly past line breaks, as other strings in a
// container may, and that ends a member or a part of one (see
// memberStringEnds), as the b'x}SECRET'} of {k: b'x}SECRET'}, the
// b'x}\nSECRET', of {k: b'x}\nSECRET', m: 1}, the u'k': of {u'k': 1} or the
// b'x}y') of {k: (b'x}y')}, so that the apostrophe of B's in Plan B's budget,
// whose string runs on to a word, opens no string. The string is read with st
// (see scanSimpleQuoted), and the bytes read are charged to it.
func isStringPrefixQuote(s string, from, i int, st *quoteScanState) bool {
	if stringPrefixStart(s, from, i) < 0 {
		return false
	}
	end, ok := scanSimpleQuoted(s, i, s[i], st)
	return ok && memberStringEnds(s, i, end, st)
}

// stringPrefixStart returns where the word right before s[i] starts when that
// word, which starts at from or later, is a Python string prefix (b, r, u or
// f, or br, rb, fr or rf, in either case), and -1 otherwise.
func stringPrefixStart(s string, from, i int) int {
	start := i
	for start > from && i-start <= 2 && isWordByte(s[start-1]) {
		start--
	}
	if start == i || i-start > 2 {
		return -1
	}
	for _, p := range [...]string{"b", "r", "u", "f", "br", "rb", "fr", "rf"} {
		if strings.EqualFold(s[start:i], p) {
			return start
		}
	}
	return -1
}

// isStringAfterSpaces reports whether horizontal spaces and then a quoted
// string follow s[:i].
func isStringAfterSpaces(s string, i int) bool {
	j := skipInlineSpacesFwd(s, i, len(s))
	return j > i && j < len(s) && (isQuoteByte(s[j]) || (s[j] == '\\' && j+1 < len(s) && isQuoteByte(s[j+1])))
}

// quotedValueEndsInKey reports whether s[from:to], the inner text of a quoted
// value followed by a delimiter, ends, possibly before spaces, in a sensitive
// key name at any boundary that findTextKeyStart accepts, as the secret of
// "abc; secret", the token of "abc; /x/token" or "err next-page-token", the
// temp:a.b of "ctx: temp:a.b" or the password of "invalid key=password", and
// returns where that key name starts.
func quotedValueEndsInKey(s string, from, to int) (int, bool) {
	end := skipInlineSpacesBack(s, from, to)
	if end <= from {
		return 0, false
	}
	if kStart := findTextKeyStart(s, from, end); kStart > from {
		return kStart, true
	}
	return 0, false
}

// danglingQuoteKeyAt reports whether the quote at s[q] closes a sensitive key
// name that ends right before it, after floor (see findTextKeyStart), with a
// delimiter and a value after the quote, as the token" of token":{k: "SECRET"}
// or the password" of password"={"k": "SECRET"}, and returns where that value
// starts.
func danglingQuoteKeyAt(s string, floor, q int, openQ openQuoteState, st *quoteScanState) (int, bool) {
	if q <= floor || !isQuoteByte(s[q]) || findTextKeyStart(s, floor, q) < 0 {
		return 0, false
	}
	d := skipDelimSpaceFwd(s, q+1)
	if d >= len(s) || (s[d] != ':' && s[d] != '=') {
		return 0, false
	}
	if ok, _ := nestedValueAfter(s, d+1, openQ, st); !ok {
		return 0, false
	}
	return valAfterDelim(s, d+1), true
}

// valueEndsPast reports whether the value that starts at s[valPos] runs past
// end: at or past end itself, a quoted string, possibly with escaped quotes,
// that closes past end, or a "{" or "[" container that is still open at end
// (see scanBalancedContainer). For a container or a quoted string that closes
// at or before end, it also returns the index right after its closer or
// closing quote, so that the caller skips it as a whole, and 0 otherwise, as
// for a quoted string whose closing quote opens a member of a container (see
// opensContainerMember).
func valueEndsPast(s string, valPos, end int, st *quoteScanState) (bool, int) {
	if valPos >= end {
		return true, 0
	}
	switch c := s[valPos]; {
	case c == '{' || c == '[':
		cEnd, closed := scanBalancedContainer(s, valPos, end)
		if !closed {
			return true, 0
		}
		return false, cEnd
	case isQuoteByte(c) || c == '\\':
		qEnd, ok := scanQuotedAt(s, valPos, st)
		if !ok {
			return false, 0
		}
		if qEnd > end {
			return true, 0
		}
		if opensContainerMember(s, valPos, qEnd, st) {
			return false, 0
		}
		if st != nil && st.openerBudgetExhausted {
			return true, 0
		}
		return false, qEnd
	}
	return false, 0
}

// opensContainerMember reports whether the closing quote of the quoted string
// s[q:qEnd] opens a member of a "{" or "[" container that opens in that
// string (see containerMemberOpener), as the "k": of "S0/x:{"k": "SECRET"},
// the "a", of "S0/x:[ "a", "SECRET"] or the "SECRET"] of
// "S0/x:[1,"SECRET"]: that quote then opens a member of a container rather
// than closing the string (see quotedValueRealEnd), so that string is not
// skipped as a whole, which would skip the container too.
func opensContainerMember(s string, q, qEnd int, st *quoteScanState) bool {
	return containerMemberOpener(s, q, qEnd, st) >= 0
}

// containerMemberOpener returns where the "{" or "[" container whose member
// the closing quote of the quoted string s[q:qEnd] opens starts, or -1. The
// container opens in the string and is still open at that quote, which
// follows, after optional delimiter whitespace, line breaks and other control
// bytes included (see isDelimSpace), either the opener, as in "S0/x:{ "k" or
// "S0/x:[\n  "a", or a "," after other members, as in "S0/x:[1,"SECRET" or
// "S0/x:[true, "SECRET", and opens a quoted key or array element there (see
// isQuotedMemberAt); or the ":" or "=" of a key of a "{" container, as in
// "S0/x:{k: "SECRET", and opens its quoted value (see
// isQuotedMemberValueAt). Brackets inside quoted strings in s[q:qEnd] are
// skipped, and st caches the scan per closing quote and charges it to its
// budget, so repeated checks stay linear; once that budget runs out, it
// returns -1 and the caller fails closed (see consumeOpenerBudget).
func containerMemberOpener(s string, q, qEnd int, st *quoteScanState) int {
	innerOpener, _ := quotedContainerOpeners(s, q, qEnd, st)
	return innerOpener
}

// strayQuoteOuter returns where the outermost "{" or "[" container opened in
// s[q:qEnd] before end starts when the closing quote of s[q:qEnd] opens a
// member of a container opened in it (see containerMemberOpener), as the {"k"
// of x"S0/creds:[1, {"k": "SECRET"}], whose value ends at the first ",", or -1.
func strayQuoteOuter(s string, q, qEnd, end int, st *quoteScanState) int {
	_, outerOpener := quotedContainerOpeners(s, q, qEnd, st)
	if outerOpener >= 0 && outerOpener < end {
		return outerOpener
	}
	return -1
}

// spacedStrayQuoteContainer returns where the last of the "{" or "["
// containers that the quoted string s[q:qEnd] encloses starts, when they follow
// the quote that opens the string, or a ":" or "=" after it in the value
// s[:end], and delimiter whitespace, or when the string seems to close in a
// member of them, wherever they open in it, and the string seems to close
// inside them (see enclosedContainerRun); otherwise it returns -1. Containers
// that the string seems to close in a member of, as the { of
// x"{'msg': 'say " hi"', 'k': 'SECRET'}" or of
// x"prefix {'msg': 'say " hi"', 'k': 'SECRET'}", start at the outermost one
// open at the closing quote (see closingQuoteInMember), and containers right
// after the quote are also read when the string seems to close before a glued
// token, as for a quoted value (see quotedValueRealEnd), so that values such
// as x"{abc" that a stop follows are not scanned. The containers open past end
// when the value ends at the whitespace after the delimiter, as the { of
// x": {'msg': "it's ok", 'k': 'SECRET'}" or the [ of x": [1, "SECRET"]", or
// in the value, as the { of x":{'msg': 'say "hi"', 'k': 'SECRET'}", whose
// string seems to close at the quote in a member in other quotes. The value is
// then redacted together with the containers that the string encloses, as when
// the string seems to close at a member of the first one (see
// strayQuoteOuter), while containers that stay open, that the string does not
// close right after, or that it seems to close after leave the text after the
// value as it is. The delimiters are tried in order, each past the first
// container of the one before, and every byte read is charged to st, so that
// repeated unclosed containers stay linear; once st runs out of budget it
// returns -1 and the caller fails closed.
func spacedStrayQuoteContainer(s string, q, qEnd, end int, st *quoteScanState) int {
	_, quote := unquoteValue(s[q:qEnd])
	from := q + len(quote)
	if quote == "" || end >= len(s) || !st.consumeOpenerBudget(end-from) {
		return -1
	}
	if strings.ContainsAny(s[from:qEnd-len(quote)], "{[") {
		if o, ok := closingQuoteInMember(s, q, qEnd, st); ok {
			if last, _ := enclosedContainerRun(s, q, qEnd, o, st); last >= 0 {
				return last
			}
		}
	}
	if opener := skipDelimSpaceFwd(s, from); opener < len(s) && (s[opener] == '{' || s[opener] == '[') && isGluedTokenAfterQuote(s, qEnd) {
		// Containers right after the quote are also read before a glued
		// token, behind the same gate as those of a quoted value.
		if !st.consumeOpenerBudget(max(opener-end, 0)) {
			return -1
		}
		if last, _ := enclosedContainerRun(s, q, qEnd, opener, st); last >= 0 {
			return last
		}
	}
	for d := from; d < end; d++ {
		if s[d] != ':' && s[d] != '=' {
			continue
		}
		opener := skipDelimSpaceFwd(s, d+1)
		if !st.consumeOpenerBudget(max(opener-end, 0)) {
			return -1
		}
		if opener >= len(s) || (s[opener] != '{' && s[opener] != '[') {
			continue
		}
		last, firstEnd := enclosedContainerRun(s, q, qEnd, opener, st)
		if last >= 0 || firstEnd < 0 {
			return last
		}
		d = firstEnd - 1
	}
	return -1
}

// enclosedContainerRun scans the "{" or "[" container at s[opener], which opens
// after the quote that opens the quoted string s[q:qEnd], and the containers
// that follow it with nothing but delimiter whitespace, possibly around a ",",
// between them. When that quote closes the string right after the last of
// them, possibly after delimiter whitespace and the ")" of groups around them
// (see skipGroupClosers), and the closing quote of s[q:qEnd], where the
// string seems to close, lies inside them, as the one of the "1" of
// x": {'a': "1"} {'b': 'SECRET'}" or
// x": {'a': "1"}, {'b': 'SECRET'}" or of the "hi" of x": {'msg': 'say "hi"'}"
// or x"({'msg': 'say "hi"'})", it returns where the last of them starts;
// otherwise it returns -1. It also
// returns where the first container ends, or -1 when that container stays
// open. Strings in either quote inside the containers are skipped (see
// scanBalancedContainer). Every byte read is charged to st, and once st runs
// out of budget it returns -1 for both.
func enclosedContainerRun(s string, q, qEnd, opener int, st *quoteScanState) (last, firstEnd int) {
	_, quote := unquoteValue(s[q:qEnd])
	if quote == "" {
		return -1, -1
	}
	seemsClose := qEnd - len(quote)
	firstEnd = -1
	for cur := opener; cur < len(s) && (s[cur] == '{' || s[cur] == '['); {
		cEnd, closed := scanBalancedContainer(s, cur, len(s))
		if !st.consumeOpenerBudget(cEnd - cur) {
			return -1, -1
		}
		if !closed {
			return -1, firstEnd
		}
		if firstEnd < 0 {
			firstEnd = cEnd
		}
		next := skipDelimSpaceFwd(s, cEnd)
		closeQuote := skipGroupClosers(s, cEnd)
		if !st.consumeOpenerBudget(closeQuote - cEnd) {
			return -1, -1
		}
		if strings.HasPrefix(s[closeQuote:], quote) {
			if opener < seemsClose && seemsClose < cEnd {
				return cur, firstEnd
			}
			return -1, firstEnd
		}
		if next < len(s) && s[next] == ',' {
			// A "," between containers, as in {'a': 1}, {'b': 'SECRET'},
			// goes on with the run only when another container follows it.
			after := skipDelimSpaceFwd(s, next+1)
			if !st.consumeOpenerBudget(after - next) {
				return -1, -1
			}
			if after >= len(s) || (s[after] != '{' && s[after] != '[') {
				return -1, firstEnd
			}
			next = after
		}
		cur = next
	}
	return -1, firstEnd
}

// skipGroupClosers returns where the first byte after s[:i] other than
// delimiter whitespace and the ")" that close groups around containers, as
// the ) of ({'k': 'v'}) or of (1, {'k': 'v'}), is.
func skipGroupClosers(s string, i int) int {
	i = skipDelimSpaceFwd(s, i)
	for i < len(s) && s[i] == ')' {
		i = skipDelimSpaceFwd(s, i+1)
	}
	return i
}

// enclosesMixedQuoteContainer reports whether the spanning quoted string
// s[q:qEnd] (with qEnd > end) closes right after one or more delimited "{" or
// "[" containers whose sequence opened after a delimiter before end, or right
// after the opening quote, possibly after delimiter whitespace, as in
// x"{'k': 'SECRET'}", or after any other text, from the first container in
// the string on, possibly in groups whose ")" come before the closing quote,
// as in x"prefix {'k': 'SECRET'}" or x"({'k': 'SECRET'})", and holds a member
// in other quotes, as in x"S0/creds:{ 'k': 'SECRET'}",
// x": {'k': 'SECRET'}", x"S0/creds:{ k1: 1, 'k2': 'SECRET'}",
// x"S0/creds:[1, 2, 'SECRET']", x"S0/creds:{ a: { 'k': 'SECRET' }}",
// x"S0/creds:{ 'k': 'SECRET'} {a:1}" or x": {'a': '1'}, {'k': 'SECRET'}"
// (see scanMixedQuoteContainers). The delimiter may be the first byte in
// the quotes. Its scans are charged to st, and once st runs out of budget it
// reports true, so that the caller fails closed.
func enclosesMixedQuoteContainer(s string, q, qEnd, end int, st *quoteScanState) bool {
	if st != nil && st.openerBudgetExhausted {
		return true
	}
	inner, quote := unquoteValue(s[q:qEnd])
	if quote == "" {
		return false
	}
	from := q + len(quote)
	qPos := from + len(inner)
	t := qPos
	for t > from && isDelimSpace(s[t-1]) {
		t--
	}
	// The containers end at tg when ")" close groups around them right
	// before the closing quote, as in x"({'k': 'SECRET'})".
	tg := t
	for tg > from && s[tg-1] == ')' {
		for tg--; tg > from && isDelimSpace(s[tg-1]); tg-- {
		}
	}
	if tg <= from || (s[tg-1] != '}' && s[tg-1] != ']') {
		return false
	}
	escaped := len(quote) == 2
	qByte := quote[len(quote)-1]
	if tg < t {
		return mixedQuoteContainersAfterText(s, from, tg, qEnd, qByte, escaped, st)
	}
	if cand := skipDelimSpaceFwd(s, from); cand < t && (s[cand] == '{' || s[cand] == '[') {
		// The containers open right after the quote, as the { of
		// x"{'k': 'SECRET'}".
		if !st.consumeOpenerBudget(cand - from) {
			return true
		}
		if mixedQuoteContainersCached(s, cand, t, qEnd, qByte, escaped, st) {
			return true
		}
	}
	dLimit := min(end, t)
	cStart := -1
	for dPos := from; dPos < dLimit; dPos++ {
		if !st.consumeOpenerBudget(1) {
			return true
		}
		if s[dPos] != ':' && s[dPos] != '=' {
			continue
		}
		cand := skipDelimSpaceFwd(s, dPos+1)
		if !st.consumeOpenerBudget(cand - (dPos + 1)) {
			return true
		}
		if cand >= t || (s[cand] != '{' && s[cand] != '[') {
			dPos = cand - 1
			continue
		}
		cStart = cand
		cur := cand
		for {
			cEnd, closed := scanBalancedContainer(s, cur, dLimit)
			if !st.consumeOpenerBudget(max(cEnd-cur, 1)) {
				return true
			}
			if !closed {
				dPos = dLimit
				break
			}
			next := cEnd
			for next < t && isDelimSpace(s[next]) {
				next++
			}
			if next < t && s[next] == ',' {
				// A "," goes on with the run only when another container
				// follows it.
				after := next + 1
				for after < t && isDelimSpace(s[after]) {
					after++
				}
				if after < t && (s[after] == '{' || s[after] == '[') {
					next = after
				}
			}
			if !st.consumeOpenerBudget(next - cEnd) {
				return true
			}
			if next == t || (next < t && (s[next] == '{' || s[next] == '[')) {
				if next >= dLimit {
					dPos = dLimit
					break
				}
				cur = next
				continue
			}
			cStart = -1
			dPos = cEnd - 1
			break
		}
	}
	if cStart >= 0 {
		return mixedQuoteContainersCached(s, cStart, t, qEnd, qByte, escaped, st)
	}
	return mixedQuoteContainersAfterText(s, from, t, qEnd, qByte, escaped, st)
}

// mixedQuoteContainersAfterText is mixedQuoteContainersCached for the
// containers that the first "{" or "[" in s[from:limit], the inside of the
// spanning quoted string that ends at qEnd, starts after any other text, as
// the { of x"prefix {'k': 'SECRET'}" or of x"({'k': 'SECRET'})". The bytes
// before that container are charged to st.
func mixedQuoteContainersAfterText(s string, from, limit, qEnd int, qByte byte, escaped bool, st *quoteScanState) bool {
	c := strings.IndexAny(s[from:limit], "{[")
	if c < 0 {
		return false
	}
	if !st.consumeOpenerBudget(c) {
		return true
	}
	return mixedQuoteContainersCached(s, from+c, limit, qEnd, qByte, escaped, st)
}

// mixedQuoteContainersCached is scanMixedQuoteContainers for the containers
// from s[cStart] to limit in the spanning quoted string that ends at qEnd,
// with its result cached in st per string and start. Once st runs out of
// budget it reports true, so that the caller fails closed.
func mixedQuoteContainersCached(s string, cStart, limit, qEnd int, qByte byte, escaped bool, st *quoteScanState) bool {
	var cache *quoteContainerCache
	if st != nil {
		cache = &st.containers[quoteKindIndex(qByte, escaped)]
		if cache.mixedQEnd == qEnd && cache.mixedFrom == cStart {
			return cache.mixedResult
		}
	}
	res := scanMixedQuoteContainers(s, cStart, limit, qByte, escaped, st)
	if st != nil && st.openerBudgetExhausted {
		return true
	}
	if cache != nil {
		cache.mixedQEnd = qEnd
		cache.mixedFrom = cStart
		cache.mixedResult = res
	}
	return res
}

// scanMixedQuoteContainers reports whether s[cStart:limit] is a run of one or
// more balanced "{" or "[" containers, separated only by delimiter whitespace
// (see isDelimSpace), as the vertical tab of {'a': '1'}\v{'b': 'SECRET'}, or
// by a "," with such whitespace around it, as in {'a': '1'}, {'b': 'SECRET'},
// any of which holds, at any depth, a member in quotes other than the
// spanning quote (outerQuote, outerEscaped) around them (see
// isMixedContainerMemberQuoteAt), as the 'k2' of { k1: 1, 'k2': 'SECRET'} or
// the 'SECRET' of [1, 2, 'SECRET']. Delimiter whitespace also separates a
// member from the bytes around it. Brackets inside inner quoted strings are
// skipped, and a quote joined to a word, as in it's, opens no string unless
// that word is a string prefix (see isStringPrefixQuote), as the b of
// b'x}SECRET'. Every byte it reads is charged to st, and once st runs out of
// budget it reports true, so that the caller fails closed.
func scanMixedQuoteContainers(s string, cStart, limit int, outerQuote byte, outerEscaped bool, st *quoteScanState) bool {
	foundQuotedMember := false
	pos := cStart
	var stack []byte
	for pos < limit {
		if s[pos] != '{' && s[pos] != '[' {
			return false
		}
		stack = append(stack[:0], s[pos])
		// wordPrev is the byte other than delimiter whitespace before the
		// word that ends at prevNonSpace.
		prevNonSpace, wordPrev := s[pos], s[pos]
		if !st.consumeOpenerBudget(1) {
			return true
		}
		i := pos + 1
		for i < limit && len(stack) > 0 {
			topOpen := stack[len(stack)-1]
			c := s[i]
			switch c {
			case '\\':
				m := 1
				for i+m < limit && s[i+m] == '\\' {
					m++
				}
				next := i + m
				if m%2 == 0 {
					if !st.consumeOpenerBudget(m) {
						return true
					}
					prevNonSpace = '\\'
					i = next
					continue
				}
				if next < limit && isQuoteByte(s[next]) && !isWordByte(s[i-1]) {
					q := s[next]
					if m&3 == 1 && (q != outerQuote || !outerEscaped) {
						if isMixedContainerMemberQuoteAt(s, next-1, topOpen, prevNonSpace) {
							foundQuotedMember = true
						}
						if end, ok := scanEscapedQuoted(s, next-1, q, st); ok && end <= limit {
							if !st.consumeOpenerBudget(end - i) {
								return true
							}
							prevNonSpace = q
							i = end
							continue
						}
					} else if outerEscaped && q == outerQuote && m&3 == 3 {
						if !st.consumeOpenerBudget(next + 1 - i) {
							return true
						}
						if end := scanTripleEscapedQuoteEnd(s, next+1, limit, q, st); end > 0 {
							if isTripleEscapedMemberAt(s, i, end, limit, topOpen, prevNonSpace) {
								foundQuotedMember = true
							}
							prevNonSpace = q
							i = end
							continue
						}
						if st != nil && st.openerBudgetExhausted {
							return true
						}
						prevNonSpace = q
						i = next + 1
						continue
					}
				}
				step := min(next+1, limit) - i
				if !st.consumeOpenerBudget(step) {
					return true
				}
				prevNonSpace = s[min(next, limit-1)]
				i = min(next+1, limit)
			case '"', '\'':
				joined := isWordByte(s[i-1])
				if (!outerEscaped && c == outerQuote) || (joined && !isStringPrefixQuote(s, pos, i, st)) {
					if !st.consumeOpenerBudget(1) {
						return true
					}
					prevNonSpace = c
					i++
					continue
				}
				memberPrev := prevNonSpace
				if joined {
					// The quote after a string prefix, as the b' of
					// b'x}SECRET', starts a member after the byte before
					// that prefix.
					memberPrev = wordPrev
				}
				if isMixedContainerMemberQuoteAt(s, i, topOpen, memberPrev) {
					foundQuotedMember = true
				}
				if end, ok := scanSimpleQuoted(s, i, c, st); ok && end <= limit {
					if !st.consumeOpenerBudget(end - i) {
						return true
					}
					prevNonSpace = c
					i = end
					continue
				}
				if !st.consumeOpenerBudget(1) {
					return true
				}
				prevNonSpace = c
				i++
			case '{', '[':
				if !st.consumeOpenerBudget(1) {
					return true
				}
				stack = append(stack, c)
				prevNonSpace = c
				i++
			case '}', ']':
				if !st.consumeOpenerBudget(1) {
					return true
				}
				if (c == '}' && topOpen != '{') || (c == ']' && topOpen != '[') {
					return false
				}
				stack = stack[:len(stack)-1]
				prevNonSpace = c
				i++
			default:
				if !st.consumeOpenerBudget(1) {
					return true
				}
				if !isDelimSpace(c) {
					if isWordByte(c) && !isWordByte(s[i-1]) {
						wordPrev = prevNonSpace
					}
					prevNonSpace = c
				}
				i++
			}
		}
		if len(stack) > 0 {
			return false
		}
		pos = i
		for pos < limit && isDelimSpace(s[pos]) {
			if !st.consumeOpenerBudget(1) {
				return true
			}
			pos++
		}
		if pos < limit && s[pos] == ',' {
			// A "," between containers, as in {'a': '1'}, {'b': 'SECRET'},
			// needs another container after it.
			for pos++; pos < limit && isDelimSpace(s[pos]); pos++ {
				if !st.consumeOpenerBudget(1) {
					return true
				}
			}
			if pos >= limit {
				return false
			}
			if !st.consumeOpenerBudget(1) {
				return true
			}
		}
	}
	return foundQuotedMember
}

// isMixedContainerMemberQuoteAt reports whether the quote at s[q], which
// follows prevNonSpace in a container that topOpen opens, starts a member of
// that container: in an object, a quoted key after the "{" or a "," (see
// isQuotedMemberAt) or a quoted value after the ":" or "=" of a key (see
// isQuotedMemberValueAt); in an array, a quoted element after the "[" or a
// ",".
func isMixedContainerMemberQuoteAt(s string, q int, topOpen, prevNonSpace byte) bool {
	if topOpen == '{' {
		switch prevNonSpace {
		case '{', ',':
			return isQuotedMemberAt(s, q, '{')
		case ':', '=':
			return isQuotedMemberValueAt(s, q)
		}
		return false
	}
	if prevNonSpace == '[' || prevNonSpace == ',' {
		return isQuotedMemberAt(s, q, '[')
	}
	return false
}

// isTripleEscapedMemberAt is isMixedContainerMemberQuoteAt for a string in
// triple-escaped quotes (\\\" or \\\'), the inner quotes of a string in
// escaped quotes, that runs from s[qStart] to s[end]: whether the byte after
// it, past delimiter whitespace and before limit, ends a non-empty key (":" or
// "="), a value ("," or "}") or an element ("," or "]") where that string
// starts.
func isTripleEscapedMemberAt(s string, qStart, end, limit int, topOpen, prevNonSpace byte) bool {
	empty := end-(qStart+4) == 4
	d := end
	for d < limit && isDelimSpace(s[d]) {
		d++
	}
	if d >= limit {
		return false
	}
	if topOpen == '{' {
		switch prevNonSpace {
		case '{', ',':
			return !empty && (s[d] == ':' || s[d] == '=')
		case ':', '=':
			return s[d] == ',' || s[d] == '}'
		}
		return false
	}
	if prevNonSpace == '[' || prevNonSpace == ',' {
		return s[d] == ',' || s[d] == ']' || (!empty && (s[d] == ':' || s[d] == '='))
	}
	return false
}

const (
	memberFlagObj uint8 = 1 << iota
	memberFlagArr
	memberFlagVal
	memberFlagsComputed
)

func quoteKindIndex(q byte, escaped bool) int {
	if escaped {
		if q == '"' {
			return 2
		}
		return 3
	}
	if q == '"' {
		return 0
	}
	return 1
}

func insideInnerQuote(pos int, starts, ends []int) bool {
	i, found := slices.BinarySearch(starts, pos)
	if found {
		return false
	}
	return i > 0 && pos < ends[i-1]
}

func memberFlagsAt(s string, qPos int, cache *quoteContainerCache) uint8 {
	if cache != nil && cache.memberFlags&memberFlagsComputed != 0 && cache.memberFlagsQPos == qPos {
		return cache.memberFlags
	}
	flags := memberFlagsComputed
	if end, empty := quotedMemberEnd(s, qPos); end >= 0 {
		d := end
		for d < len(s) && isDelimSpace(s[d]) {
			d++
		}
		if d < len(s) {
			switch s[d] {
			case ':', '=':
				if !empty {
					flags |= memberFlagObj | memberFlagArr
				}
			case ',':
				flags |= memberFlagArr | memberFlagVal
			case ']':
				flags |= memberFlagArr
			case '}':
				flags |= memberFlagVal
			}
		}
	}
	if cache != nil {
		cache.memberFlags = flags
		cache.memberFlagsQPos = qPos
	}
	return flags
}

// quotedContainerOpeners scans s[q:qEnd] for unclosed "{" and "[" openers
// (ignoring brackets inside inner quoted strings) and returns the innermost
// and outermost openers when s[qEnd-len(quote)] starts a container member, or
// a part of one after a "(", a "+" or another string, as the b" of
// "{'a': (1 + b" SECRET"), 'k': 'v'}" or of "{'a': 'x' b" SECRET", 'k': 'v'}"
// does when its string ends a member or a part of one (see memberStringEnds).
func quotedContainerOpeners(s string, q, qEnd int, st *quoteScanState) (innerOpener, outerOpener int) {
	if st != nil && st.openerBudgetExhausted {
		return -1, -1
	}
	inner, quote := unquoteValue(s[q:qEnd])
	if quote == "" {
		return -1, -1
	}
	from := q + len(quote)
	qPos := from + len(inner)
	t := qPos
	for t > from && isDelimSpace(s[t-1]) {
		t--
	}
	if p := stringPrefixStart(s, from, qPos); p >= 0 {
		// The closing quote right after a string prefix, as the b" of
		// "{'k': b" SECRET", 'm': 1}", starts a member where that prefix
		// starts.
		t = p
		for t > from && isDelimSpace(s[t-1]) {
			t--
		}
	}
	if t <= from {
		return -1, -1
	}
	c := s[t-1]
	// part reports that the closing quote opens a part of a member, and
	// limit is where the span scan stops: past the closing quote of another
	// string right before, so that the scan reads that string as one.
	part, limit := false, t-1
	switch c {
	case '{', '[', ',', ':', '=':
	case '(', '+':
		part = true
	default:
		if !isQuoteByte(c) {
			return -1, -1
		}
		part, limit = true, t
	}
	escaped := len(quote) == 2
	qByte := quote[len(quote)-1]
	var cache *quoteContainerCache
	if st != nil {
		cache = &st.containers[quoteKindIndex(qByte, escaped)]
	}
	var flags uint8
	if part {
		if end, ok := scanQuotedAt(s, qPos, st); !ok || !memberStringEnds(s, qPos, end, st) {
			return -1, -1
		}
	} else {
		flags = memberFlagsAt(s, qPos, cache)
	}
	if c == '{' && flags&memberFlagObj == 0 {
		return -1, -1
	}
	if c == '[' && flags&memberFlagArr == 0 {
		return -1, -1
	}
	if c == ',' && flags&(memberFlagObj|memberFlagArr) == 0 {
		return -1, -1
	}
	if (c == ':' || c == '=') && flags&memberFlagVal == 0 {
		return -1, -1
	}
	var openers []int
	// ends lists where the strings that the span scan skipped end.
	var ends []int
	hit := false
	if cache != nil {
		if cache.checkedQEnd != qEnd {
			cache.checkedQEnd = qEnd
			cache.numSpans = 0
		}
		for i := range cache.numSpans {
			sp := &cache.spans[i]
			if from >= sp.checkedFrom && !insideInnerQuote(from, sp.iqStarts, sp.iqEnds) {
				idx, _ := slices.BinarySearch(sp.openers, from)
				openers = sp.openers[idx:]
				ends = sp.iqEnds
				hit = true
				break
			}
		}
	}
	if !hit {
		var iqStarts []int
		openers, iqStarts, ends = scanOpenersInSpan(s, from, limit, qByte, escaped, st)
		if st != nil && st.openerBudgetExhausted {
			return -1, -1
		}
		if c == '{' || c == '[' {
			openers = append(openers, t-1)
		}
		if cache != nil {
			slot := cache.numSpans
			if slot < len(cache.spans) {
				cache.numSpans++
			} else {
				slot = len(cache.spans) - 1
			}
			cache.spans[slot] = quoteContainerSpan{
				checkedFrom: from,
				openers:     openers,
				iqStarts:    iqStarts,
				iqEnds:      ends,
			}
		}
	}
	if len(openers) == 0 {
		return -1, -1
	}
	o := openers[len(openers)-1]
	switch c {
	case '{', '[', '(', '+':
		return o, openers[0]
	case ',':
		if (s[o] == '{' && flags&memberFlagObj != 0) || (s[o] == '[' && flags&memberFlagArr != 0) {
			return o, openers[0]
		}
	case ':', '=':
		if s[o] == '{' && flags&memberFlagVal != 0 {
			return o, openers[0]
		}
	default:
		// The quote right before closes another string only when the span
		// scan skipped a string that ends there, so that the apostrophe of
		// it's " SECRET" starts no part.
		if _, found := slices.BinarySearch(ends, t); found {
			return o, openers[0]
		}
	}
	return -1, -1
}

// scanOpenersInSpan scans s[from:limit], the interior prefix of a quoted
// string opened by (outerQuote, outerEscaped), and returns the positions of
// all "{" and "[" brackets that remain open at limit, ignoring brackets inside
// closed inner quoted strings, along with the skipped inner-quote spans.
func scanOpenersInSpan(s string, from, limit int, outerQuote byte, outerEscaped bool, st *quoteScanState) (openers, iqStarts, iqEnds []int) {
	noMoreTripleEsc := false
	for i := from; i < limit; i++ {
		c := s[i]
		switch c {
		case '\\':
			m := 1
			for i+m < limit && s[i+m] == '\\' {
				m++
			}
			next := i + m
			if m%2 == 0 {
				if !st.consumeOpenerBudget(m) {
					return nil, nil, nil
				}
				i = next - 1
				continue
			}
			if next < limit && isQuoteByte(s[next]) && (i == from || !isWordByte(s[i-1])) {
				q := s[next]
				if m&3 == 1 && (q != outerQuote || (!outerEscaped && len(openers) > 0)) {
					if end, ok := scanEscapedQuoted(s, next-1, q, st); ok && end <= limit {
						if !st.consumeOpenerBudget(end - i) {
							return nil, nil, nil
						}
						iqStarts = append(iqStarts, next-1)
						iqEnds = append(iqEnds, end)
						i = end - 1
						continue
					}
				} else if outerEscaped && q == outerQuote && m&3 == 3 && len(openers) > 0 && !noMoreTripleEsc {
					if !st.consumeOpenerBudget(next + 1 - i) {
						return nil, nil, nil
					}
					if end := scanTripleEscapedQuoteEnd(s, next+1, limit, q, st); end > 0 {
						iqStarts = append(iqStarts, i)
						iqEnds = append(iqEnds, end)
						i = end - 1
						continue
					}
					if st != nil && st.openerBudgetExhausted {
						return nil, nil, nil
					}
					noMoreTripleEsc = true
					i = next
					continue
				}
			}
			if !st.consumeOpenerBudget(min(next+1, limit) - i) {
				return nil, nil, nil
			}
			i = next
		case '"', '\'':
			if (!outerEscaped && c == outerQuote) || (i > from && isWordByte(s[i-1]) && !isStringPrefixQuote(s, from, i, st)) {
				if !st.consumeOpenerBudget(1) {
					return nil, nil, nil
				}
				continue
			}
			if end, ok := scanSimpleQuoted(s, i, c, st); ok && end <= limit {
				if !st.consumeOpenerBudget(end - i) {
					return nil, nil, nil
				}
				iqStarts = append(iqStarts, i)
				iqEnds = append(iqEnds, end)
				i = end - 1
				continue
			}
			if !st.consumeOpenerBudget(1) {
				return nil, nil, nil
			}
		case '{', '[':
			if !st.consumeOpenerBudget(1) {
				return nil, nil, nil
			}
			openers = append(openers, i)
		case '}', ']':
			if !st.consumeOpenerBudget(1) {
				return nil, nil, nil
			}
			if len(openers) > 0 {
				openers = openers[:len(openers)-1]
			}
		default:
			if !st.consumeOpenerBudget(1) {
				return nil, nil, nil
			}
		}
	}
	return openers, iqStarts, iqEnds
}

func scanTripleEscapedQuoteEnd(s string, pos, limit int, q byte, st *quoteScanState) int {
	for i := pos; i < limit; {
		m := 0
		for i < limit && s[i] == '\\' {
			m++
			i++
		}
		if m > 0 && !st.consumeOpenerBudget(m) {
			return -1
		}
		if i >= limit {
			return -1
		}
		if !st.consumeOpenerBudget(1) {
			return -1
		}
		c := s[i]
		if c < 0x20 || c == 0x7f {
			return -1
		}
		i++
		if c == q && m&3 == 3 {
			return i
		}
	}
	return -1
}

// isQuotedMemberAt reports whether a quoted member of a container that open
// ("{" or "[") opens starts at s[q]: a quoted string (see quotedMemberEnd)
// followed, after optional delimiter whitespace, by a ":" or "=" when it is a
// non-empty key, as the "k": of {"k": 1}, the "user:foo": of
// {"user:foo": 1}, the "a b"= of {"a b"= 1} or the \'k\': of {\'k\': 1}, or,
// in an array, by a "," or "]" when it is an element, as the "a", of
// ["a", 1]. A key may hold any other byte, as a JSON key may, so that a key
// such as "a/b" is read as one.
func isQuotedMemberAt(s string, q int, open byte) bool {
	end, empty := quotedMemberEnd(s, q)
	if end < 0 {
		return false
	}
	d := end
	for d < len(s) && isDelimSpace(s[d]) {
		d++
	}
	if d >= len(s) {
		return false
	}
	switch s[d] {
	case ':', '=':
		return !empty
	case ',', ']':
		return open == '['
	}
	return false
}

// isQuotedMemberValueAt reports whether the quoted value of a member of a "{"
// container starts at s[q]: a quoted string (see quotedMemberEnd) followed,
// after optional delimiter whitespace, by a "," or the "}" that closes the
// container, as the "SECRET"} of {k: "SECRET"}.
func isQuotedMemberValueAt(s string, q int) bool {
	end, _ := quotedMemberEnd(s, q)
	if end < 0 {
		return false
	}
	for end < len(s) && isDelimSpace(s[end]) {
		end++
	}
	return end < len(s) && (s[end] == ',' || s[end] == '}')
}

// quotedMemberEnd returns the index right after the quoted string that starts
// at s[q], and whether that string is empty, or -1: a string opened by a quote
// that may be escaped by a backslash (", ', \" or \') and closed by the same
// quote, that may escape a byte with a backslash (with m&3 backslash parity
// inside \" and \') and may span lines and hold control bytes, as the strings
// that the container scans skip may (see scanSimpleQuoted and
// scanEscapedQuoted).
func quotedMemberEnd(s string, q int) (int, bool) {
	end, ok := scanQuotedAt(s, q, nil)
	if !ok {
		return -1, false
	}
	if s[q] == '\\' {
		return end, end-q == 4
	}
	return end, end-q == 2
}

// isDanglingKeyPathByte reports whether c may be part of a key, possibly
// joined to earlier text by a path separator, punctuation or a quote, that
// runs up to the delimiter after it (see danglingClosedKeyAfterQuoteEnd): any
// byte other than delimiter whitespace, a text-value stop, ":" or "=".
func isDanglingKeyPathByte(c byte) bool {
	return !isDelimSpace(c) && !isTextValueStop(c) && c != ':' && c != '='
}

func extendGluedValueEnd(s string, valPos, open int, openQ openQuoteState, st *quoteScanState) int {
	cur := valPos
	for cur < len(s) {
		if st != nil && st.openerBudgetExhausted {
			return len(s)
		}
		if nextVal, ok := stolenQuotedKeyAfterQuoteEnd(s, cur, openQ, st); ok {
			cur = nextVal
			continue
		}
		if isTextValueStop(s[cur]) {
			return cur
		}
		if isQuoteByte(s[cur]) && followedByDelimiter(s, cur+1) && (openQ.quote != s[cur] || openQ.escaped) {
			// A quote right before a delimiter that does not close the
			// enclosing string, and opens no string or only one that holds
			// the delimiter and whitespace up to the quote of the next
			// value, closes a key, as the second quote of
			// "temp:a"": SECRET or "temp:a"": " SECRET", so the value after
			// the delimiter goes on.
			qEnd, ok := scanQuotedAt(s, cur, st)
			if !ok || !hasNonEmptyTextValue(s[skipDelimSpaceFwd(s, cur+1)+1:qEnd-1], false) {
				if next := valAfterDelim(s, skipDelimSpaceFwd(s, cur+1)+1); next > cur && next < len(s) {
					cur = next
					continue
				}
			}
		}
		vEnd := scanValueEnd(s, cur, isTextValueStop, st)
		kind, next := nestsSensitiveKey(s, cur, vEnd, false, openQ, st)
		if st != nil && st.openerBudgetExhausted {
			return len(s)
		}
		switch kind {
		case notNestedKey:
			if realEnd, ok := quotedValueRealEnd(s, cur, vEnd, open, openQ, st); ok {
				return max(vEnd, realEnd)
			}
			if _, quote := unquoteValue(s[cur:vEnd]); quote != "" &&
				hasGluedSuffixAfterQuotedValue(s, cur, vEnd, openQ, st) {
				cur = vEnd
				continue
			}
			if nextVal, ok := danglingClosedKeyAfterQuoteEnd(s, cur, openQ, st); ok {
				cur = nextVal
				continue
			}
			return textValueEnd(s, cur, vEnd, open)
		case gluedAfterValue:
			if realEnd, ok := quotedValueRealEnd(s, cur, vEnd, open, openQ, st); ok {
				return max(vEnd, realEnd)
			}
			if next <= cur {
				return textValueEnd(s, cur, vEnd, open)
			}
			cur = next
		default:
			return max(vEnd, nestedPairEndWithState(s, next, open, openQ, st))
		}
	}
	return cur
}

// failClosedNestedPairEnd extends past a nested key and its value when
// scanTextKeyValAt cannot re-parse pair directly (for example, an inner quoted
// key ending before an outer closing quote, such as 'token'": ...), while
// returning pair unchanged if no delimiter follows the key.
func failClosedNestedPairEnd(s string, pair, open int, openQ openQuoteState, st *quoteScanState) int {
	if pair >= len(s) {
		return pair
	}
	i := pair
	if qEnd, qOK := scanQuotedAt(s, i, nil); qOK {
		i = qEnd
	} else {
		for i < len(s) && !isDelimSpace(s[i]) && s[i] != ':' && s[i] != '=' {
			i++
		}
	}
	for i < len(s) {
		if i+1 < len(s) && s[i] == '\\' && (s[i+1] == '"' || s[i+1] == '\'') {
			i += 2
			continue
		}
		if s[i] == '"' || s[i] == '\'' {
			i++
			continue
		}
		break
	}
	i = skipDelimSpaceFwd(s, i)
	if i >= len(s) || (s[i] != ':' && s[i] != '=') {
		return pair
	}
	valStart := valAfterDelim(s, i+1)
	if valStart >= len(s) || isTextValueStop(s[valStart]) {
		return valStart
	}
	return extendGluedValueEnd(s, valStart, open, openQ, st)
}

func scanTextKeyValAt(s string, pos int, st *quoteScanState) (delimAfter, valStart, valEnd int, ok bool) {
	if pos >= len(s) {
		return 0, 0, 0, false
	}
	i := pos
	if s[i] == '"' || s[i] == '\'' || (i+1 < len(s) && s[i] == '\\' && (s[i+1] == '"' || s[i+1] == '\'')) {
		qEnd, qOK := scanQuotedAt(s, i, nil)
		if !qOK {
			return 0, 0, 0, false
		}
		i = qEnd
	} else {
		if hasFoldPrefix(s[i:], session.KeyPrefixTemp) {
			i += len(session.KeyPrefixTemp)
			bodyStart := i
			for i < len(s) && isTempKeyBodyByte(s[i]) {
				i++
			}
			if i == bodyStart {
				return 0, 0, 0, false
			}
			for i+1 < len(s) && s[i] == ':' && isTempKeyBodyByte(s[i+1]) {
				j := i + 1
				for j < len(s) && isTempKeyBodyByte(s[j]) {
					j++
				}
				k := skipDelimSpaceFwd(s, j)
				if k < len(s) && (s[k] == ':' || s[k] == '=') {
					i = j
				} else {
					break
				}
			}
		} else {
			for i < len(s) {
				if c := s[i]; c < utf8.RuneSelf {
					if !isCompoundKeyByte(c) {
						break
					}
					i++
					continue
				}
				r, size := utf8.DecodeRuneInString(s[i:])
				if r != '\u0130' && r != '\u017f' && r != '\u212a' {
					break
				}
				i += size
			}
			if i == pos {
				return 0, 0, 0, false
			}
		}
		// Optional closing quote on the key when pair started after an opening quote.
		if i+1 < len(s) && s[i] == '\\' && (s[i+1] == '"' || s[i+1] == '\'') {
			i += 2
		} else if i < len(s) && (s[i] == '"' || s[i] == '\'') {
			i++
		}
	}
	// A stray quote after the closing quote of the key, as in
	// "temp:a"": SECRET (see redactTextKeyValues).
	if i+1 < len(s) && i > pos && isQuoteByte(s[i]) && isQuoteByte(s[i-1]) {
		if j := skipDelimSpaceFwd(s, i+1); j < len(s) && (s[j] == ':' || s[j] == '=') {
			i++
		}
	}
	for i < len(s) && isDelimSpace(s[i]) {
		i++
	}
	if i >= len(s) || (s[i] != ':' && s[i] != '=') {
		return 0, 0, 0, false
	}
	delimAfter = i + 1
	valStart = delimAfter
	for valStart < len(s) && isDelimSpace(s[valStart]) {
		valStart++
	}
	if (valStart >= len(s) || isTextValueStop(s[valStart])) && valStart > delimAfter && !isTextValueStop(s[valStart-1]) {
		valStart--
	}
	if valStart >= len(s) || isTextValueStop(s[valStart]) {
		return 0, 0, 0, false
	}
	return delimAfter, valStart, scanValueEnd(s, valStart, isTextValueStop, st), true
}

func skipDelimSpaceFwd(s string, i int) int {
	for i < len(s) && isDelimSpace(s[i]) {
		i++
	}
	return i
}

// followedByDelimiter reports whether a key-value delimiter, ":" or "=",
// follows s[:i], possibly after whitespace.
func followedByDelimiter(s string, i int) bool {
	j := skipDelimSpaceFwd(s, i)
	return j < len(s) && (s[j] == ':' || s[j] == '=')
}

// nestedValueFollowsDelimiter reports whether a key-value delimiter follows
// s[:i], possibly after whitespace, and a value starts after the delimiter
// (see nestedValueAfter), also returning the end of any quoted value that
// continues into a glued suffix.
func nestedValueFollowsDelimiter(s string, i int, openQ openQuoteState, st *quoteScanState) (bool, int) {
	j := skipDelimSpaceFwd(s, i)
	if j >= len(s) || (s[j] != ':' && s[j] != '=') {
		return false, 0
	}
	return nestedValueAfter(s, j+1, openQ, st)
}

// isClosingPunctByte reports whether c is one of closingPunctuation.
func isClosingPunctByte(c byte) bool {
	switch c {
	case '}', ']', ')', '"', '\'', '\\', ':', '.':
		return true
	default:
		return false
	}
}

// hasNonEmptyTextValue reports whether s contains at least one byte before any
// optional text-value stop byte that is neither delimiter whitespace nor
// closingPunctuation.
func hasNonEmptyTextValue(s string, stopOnTextStop bool) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if stopOnTextStop && isTextValueStop(c) {
			return false
		}
		if !isDelimSpace(c) && !isClosingPunctByte(c) {
			return true
		}
	}
	return false
}

// valueStartsAt reports whether a value that can be redacted starts in s at
// i, possibly after whitespace, when no outer string quote is open on the same
// line (see nestedValueAfter).
func valueStartsAt(s string, i int, st *quoteScanState) bool {
	ok, _ := nestedValueAfter(s, i, openQuoteState{}, st)
	return ok
}

// nestedValueAfter reports whether a value that can be redacted starts in s at
// i, possibly after whitespace. When a closed quoted literal starts there and
// ends cleanly (see quotedValueEnds), it must hold a non-empty text value; when
// a token is glued directly after its closing quote without closing an
// enclosing same-line quote before a space-separated next quoted string (as in
// `password=" secret"tail`), nestedValueAfter returns (true, qEnd) so the
// caller consumes the quoted value and its glued suffix together.
func nestedValueAfter(s string, i int, openQ openQuoteState, st *quoteScanState) (bool, int) {
	valStart := skipDelimSpaceFwd(s, i)
	if valStart >= len(s) {
		return false, 0
	}
	rest := s[valStart:]
	if c := rest[0]; c == '"' || c == '\'' || c == '\\' {
		if qEnd, ok := scanQuotedAt(s, valStart, st); ok {
			if quotedValueEnds(rest, qEnd-valStart) {
				return hasNonEmptyTextValue(s[valStart:qEnd], false), 0
			}
			if !closesOpenQuoteAndOpensNext(s, valStart, qEnd, openQ, st) {
				return true, qEnd
			}
		}
	}
	return hasNonEmptyTextValue(rest, true), 0
}

// closesOpenQuoteAndOpensNext reports whether s[qPos:qEnd] spans the prose
// between an enclosing string opened on the same line (openQ) and a following
// closed quoted string preceded by whitespace, as in
// `failed: "password=SECRETP-token: " for user "bob"` or
// `got "password=SECRETP-token=", want "abc"`.
func closesOpenQuoteAndOpensNext(s string, qPos, qEnd int, openQ openQuoteState, st *quoteScanState) bool {
	if openQ.quote == 0 {
		return false
	}
	var ch byte
	var escaped bool
	var qLen int
	if s[qPos] == '\\' {
		if qPos+1 >= len(s) {
			return false
		}
		ch, escaped, qLen = s[qPos+1], true, 2
	} else {
		ch, escaped, qLen = s[qPos], false, 1
	}
	if openQ.quote != ch || openQ.escaped != escaped || qEnd <= qPos+2*qLen {
		return false
	}
	nextOpen := qEnd - qLen
	if !isRegexpSpace(s[nextOpen-1]) {
		return false
	}
	_, nextOK := scanQuotedAt(s, nextOpen, st)
	return nextOK
}

// isQuoteGapBeforeNextString reports whether the quoted value s[qPos:qEnd]
// holds only whitespace between the closing quote of an enclosing string and
// the opening quote of the next string (see closesOpenQuoteAndOpensNext), as
// the " " in msg: "password=" "SECRET" does, so that the value of the key at
// the end of the enclosing string is that next string.
func isQuoteGapBeforeNextString(s string, qPos, qEnd int, openQ openQuoteState, st *quoteScanState) bool {
	if !closesOpenQuoteAndOpensNext(s, qPos, qEnd, openQ, st) {
		return false
	}
	inner, quote := unquoteValue(s[qPos:qEnd])
	if quote == "" {
		return false
	}
	for i := 0; i < len(inner); i++ {
		if !isDelimSpace(inner[i]) {
			return false
		}
	}
	return true
}

// isTrailingColonBoundary reports whether s[i:] begins with ":" followed by
// whitespace or the end of s, as in `password: "abc": invalid`.
func isTrailingColonBoundary(s string, i int) bool {
	return i < len(s) && s[i] == ':' && (i+1 >= len(s) || isDelimSpace(s[i+1]))
}

// isEscapedSpaceAt reports whether s[i:] begins with a single backslash-escaped
// whitespace sequence (\t, \n, \f, \r), as in `token: \" secret\"\nretry`.
func isEscapedSpaceAt(s string, i int) bool {
	if i+1 >= len(s) || s[i] != '\\' {
		return false
	}
	switch s[i+1] {
	case 't', 'n', 'f', 'r':
		return true
	default:
		return false
	}
}

// hasGluedSuffixAfterQuotedValue reports whether a closed quoted value
// s[valStart:end] is followed directly by a glued token that should be
// redacted together with it rather than a value/prose boundary.
func hasGluedSuffixAfterQuotedValue(s string, valStart, end int, openQ openQuoteState, st *quoteScanState) bool {
	return isGluedTokenAfterQuote(s, end) &&
		!isTrailingColonBoundary(s, end) &&
		!isEscapedSpaceAt(s, end) &&
		!closesEnclosingQuoteOnly(s, end, openQ) &&
		!closesOpenQuoteAndOpensNext(s, valStart, end, openQ, st)
}

// closesEnclosingQuoteOnly reports whether the only text glued to the quoted
// value ending at s[end-1] is the closing quote of the string opened earlier
// on the same line (openQ), with the same quote byte and escaping, followed by
// a value boundary: the end of s, a text-value stop, "}", "]", ")", ".", an
// escaped whitespace sequence, or a colon followed by whitespace or the end of
// s, as in `failed: "password='abc'" for user` or
// `parsing "password='abc'": invalid`. Such a suffix only finishes the
// surrounding prose string, so callers do not extend the value past end. No
// quoted key can start at s[end-1] here, since its span would contain the
// enclosing quote, and any key after the boundary is scanned again from end by
// the caller.
func closesEnclosingQuoteOnly(s string, end int, openQ openQuoteState) bool {
	if openQ.quote == 0 || end >= len(s) {
		return false
	}
	k := end
	if openQ.escaped {
		if k+1 >= len(s) || s[k] != '\\' || s[k+1] != openQ.quote {
			return false
		}
		k += 2
	} else {
		if s[k] != openQ.quote {
			return false
		}
		k++
	}
	if k >= len(s) || !isGluedTokenAfterQuote(s, k) {
		return true
	}
	return isTrailingColonBoundary(s, k) || isEscapedSpaceAt(s, k)
}
