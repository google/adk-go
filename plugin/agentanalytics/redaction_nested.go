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
// that. Words before its key that main read as part of a value, as a path
// with a backslash or before a key that main did not have, keep it from being
// one (see mainReadsNestedKey). On the line of the key before it (or after a
// joined value or a path-prefixed dotted segment that redactTextKeyValues
// skips), or after such words, such a value may begin with a credential
// before the nested key, as in the Go error "password=dev-SECRET-password:
// dial tcp ...", so nestsSensitiveKey then returns valueEndingInKeyName, and
// where the key name in it starts, so that the value is redacted together
// with the value of the pair it starts.
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
		// The text after the closed containers may end in a key, as the
		// token of [abc];token: v, whose pair is redacted with them. A
		// container is never a key, and neither is a placeholder that an
		// earlier pass wrote, glued to other text as in [REDACTED]x/token:,
		// so the value is never left for a later pass.
		var kind nestedKeyKind
		var pair int
		if end-cur > maxNestedKeyLen {
			kind, pair = classifyOverBudgetNestedKey(s, cur, end, openQ, st)
		} else {
			kind, pair = classifyNestedKey(s, cur, end, false, openQ, st)
		}
		switch kind {
		case notNestedKey:
			return notNestedKey, 0
		case gluedAfterValue:
			return gluedAfterValue, pair
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
	// unplaced reports that a quote in the value opens a string that runs
	// past end and that none of the checks below places.
	unplaced := false
	for idx := 0; idx < len(v); idx++ {
		switch v[idx] {
		case '"', '\'', '\\':
			// A backslash before a quote starts a string quoted with escaped
			// quotes (see scanEscapedQuoted); before anything else it is
			// skipped, and a quote that a backslash escapes is read with it.
			if v[idx] == '\\' {
				if start+idx+1 >= len(s) || !isQuoteByte(s[start+idx+1]) {
					continue
				}
			} else if !isUnescapedQuote(v, 0, idx) {
				continue
			}
			if valPos, ok := danglingQuoteKeyAt(s, start, start+idx, openQ, st); ok {
				// The quote closes a key rather than opening a string, as
				// the token" of DB_PASSWORD: token": SECRET.
				return gluedAfterValue, valPos
			}
			qEnd, qOK := scanQuotedAt(s, start+idx, st)
			if !qOK {
				continue
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
			// No check places the string that the quote opens: the scan
			// goes on with the byte after the quote, so that a later quote
			// is still read, and a container after it ends the scan.
			unplaced = true
			continue
		case '{', '[':
			if unplaced {
				return notNestedKey, 0
			}
			if idx > 0 && end < len(s) {
				if dIdx := delimBeforePos(v, idx); dIdx > 0 {
					cEnd, closed := scanBalancedContainer(s, start+idx, end)
					if closed {
						idx = cEnd - start - 1
						continue
					}
					if kind, pair := nestedKeyName(v[:dIdx], start, lineStart); kind != notNestedKey {
						return kind, pair
					}
					// A container after a delimiter in the value that is
					// still open at end, as the {"k": "SECRET"} of
					// x/creds:{"k": "SECRET"}, runs past the value and is
					// redacted with it.
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

// classifyOverBudgetNestedKey classifies s[start:end], a value longer than
// maxNestedKeyLen, which is redacted as a value, as on main, without reading
// the keys in it, so that a value is never read again from each key in it.
// Only the end of the value is read: a quoted value that ends in a key name
// or is a padded quoted key (see isPaddedQuotedTextKey) and that a delimiter
// follows, or a bare value that ends in a delimiter or that a delimiter
// follows, with a value after it, may end in a key, and is redacted together
// with that value (gluedAfterValue at where it starts), so that such a key is
// never redacted without its value; so is a quoted value with text glued to
// it, or a quoted key after its closing quote (see
// stolenQuotedKeyAfterQuoteEnd).
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
	if strings.HasSuffix(v, ":") || strings.HasSuffix(v, "=") {
		if okVal, gluedEnd := nestedValueAfter(s, end, openQ, st); okVal {
			if gluedEnd > 0 {
				return gluedAfterValue, gluedEnd
			}
			return gluedAfterValue, valAfterDelim(s, end)
		}
	}
	if okVal, gluedEnd := nestedValueFollowsDelimiter(s, end, openQ, st); okVal {
		if gluedEnd > 0 {
			return gluedAfterValue, gluedEnd
		}
		return gluedAfterValue, valAfterDelim(s, skipDelimSpaceFwd(s, end)+1)
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
		// none, so such a name only ends in a key name. Words before the key
		// are part of a nested key only where main read them so too (see
		// mainReadsNestedKey).
		if lineStart && (q == 0 || m[2] == q) && !strings.ContainsAny(name[:m[2]], "&;|") && !hasSpaceOrControl(name[:m[2]]) &&
			(!strings.Contains(prefixAfterState, ":") || hasFoldPrefix(prefixAfterState, session.KeyPrefixTemp) || (q == 0 && isPathLikePrefix(name[:m[2]]))) &&
			(m[2] <= stateStart || mainReadsNestedKey(name)) {
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
			if lineStart && endTrim == len(name) && (tStart == stateStart || (q == 0 && isPathLikePrefix(name[:tStart]) && mainReadsNestedKey(name))) {
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
			if lineStart && (off == stateStart || (q == 0 && isPathLikePrefix(name[:kStart]) && isBareSensitiveKeyName(name[kStart:]) && mainReadsNestedKey(name))) {
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

// mainReadsNestedKey reports whether the expressions on main read name, the
// name of a pair that a value on a line of its own starts, as a nested key,
// whose pair they redacted on its own, keeping the words before its key, as
// the /etc/ssl/ of credentials:\n  /etc/ssl/password: v (see
// legacyNestedKeyName). They read no other words before a key so: not a path
// that holds a backslash, as C:\ssl\password, nor a path before a key that
// they did not have, as /etc/ssl/secret_key, which they redacted as the value
// of the pair on the line above. nestedKeyName keeps the words before the key
// of such a name only where they did, and redacts the name otherwise, with
// the value of the pair that it starts (see valueEndingInKeyName).
func mainReadsNestedKey(name string) bool {
	kind, _ := legacyNestedKeyName(name, 0, true)
	return kind == nestedKey
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
		if !st.consumeOpenerBudget(end - pair) {
			return len(s)
		}
		kind, next := nestsSensitiveKey(s, start, end, startsLine(s, delimAfter, start), openQ, st)
		if st != nil && st.openerBudgetExhausted {
			return len(s)
		}
		switch kind {
		case notNestedKey:
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

// isDanglingKeyPathByte reports whether c may be part of a key, possibly
// joined to earlier text by a path separator, punctuation or a quote, that
// runs up to the delimiter after it (see danglingClosedKeyAfterQuoteEnd): any
// byte other than delimiter whitespace, a text-value stop, ":" or "=".
func isDanglingKeyPathByte(c byte) bool {
	return !isDelimSpace(c) && !isTextValueStop(c) && c != ':' && c != '='
}

// extendGluedValueEnd returns where the text to redact with a value ends when
// the value goes on at s[valPos], glued to it: past each value that a key
// glued to the one before it starts, to the end of the first one that does
// not go on. Each step reads one value, from cur to the next value stop; a
// value longer than maxNestedKeyLen is taken as a whole (see
// classifyOverBudgetNestedKey), so that a step past the budget moves past
// what it read, and a step within it reads at most maxNestedKeyLen bytes.
// Every byte read is also charged to st, and once st runs out of budget the
// text to redact runs to the end of s.
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
		if !st.consumeOpenerBudget(vEnd - cur) {
			return len(s)
		}
		if vEnd-cur > maxNestedKeyLen {
			// A value past the budget is redacted as a value as a whole,
			// without reading the keys in it, and only its end may run on
			// into the value after a delimiter (see
			// classifyOverBudgetNestedKey), so that a run of glued pairs
			// with no value stop, as "token":"x"/ repeated, is read once.
			if kind, next := classifyOverBudgetNestedKey(s, cur, vEnd, openQ, st); kind == gluedAfterValue {
				cur = max(next, vEnd)
				continue
			}
			return textValueEnd(s, cur, vEnd, open)
		}
		kind, next := nestsSensitiveKey(s, cur, vEnd, false, openQ, st)
		if st != nil && st.openerBudgetExhausted {
			return len(s)
		}
		switch kind {
		case notNestedKey:
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
