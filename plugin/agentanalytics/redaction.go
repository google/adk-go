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
	"strings"
	"unicode"
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
// that a scan for a placeholder in text (see knownSentinelLenAt) reads no
// further than any placeholder may reach.
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
	out, lost := sanitizeRawTextOnce(raw)
	for cur, pass := raw, 1; out != cur && !isScrubbedPlainText(out); pass++ {
		if pass == maxRedactionPasses {
			return redactedSensitiveTextPlaceholder, true
		}
		next, nextLost := sanitizeRawTextOnce(out)
		if next == out {
			break
		}
		cur, out, lost = out, next, lost || nextLost
	}
	return out, lost
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

// maxRedactionPasses bounds how many times redactErrorMessage, sanitizeRawText
// and scrubErrorText run their pass over text, each time over the output of
// the last one, to reach text that a further pass leaves unchanged. A pass
// that redacts one value may only expose another to the next, as when a key
// whose value an earlier pass replaced reads as a nested key; running the pass
// again until nothing changes makes redacting redacted text change nothing.
// Text that still changes after maxRedactionPasses passes fails closed.
const maxRedactionPasses = 4

// redactStringValue redacts str, a string value that SmartTruncate sanitizes
// at depth, with redactJSONTextWithBudget and scrubErrorText, and reports
// whether content was lost. As in sanitizeRawText, the pass is run again over
// its result until that no longer changes (see maxRedactionPasses), each
// further pass with a budget of its own, so that sanitizing a sanitized value
// changes nothing.
func redactStringValue(str string, depth int, budget *nodeBudget) (string, bool) {
	out, lost := redactStringValueOnce(str, depth, budget)
	for cur, pass := str, 1; out != cur && !isScrubbedPlainText(out); pass++ {
		if pass == maxRedactionPasses {
			return redactedSensitiveTextPlaceholder, true
		}
		next, nextLost := redactStringValueOnce(out, depth, &nodeBudget{remaining: maxSanitizeNodes})
		if next == out {
			break
		}
		cur, out, lost = out, next, lost || nextLost
	}
	return out, lost
}

// redactStringValueOnce is one pass of redactStringValue.
func redactStringValueOnce(str string, depth int, budget *nodeBudget) (string, bool) {
	str, _, lost := redactJSONTextWithBudget(str, depth, budget)
	if scrubbed := scrubErrorText(str); scrubbed != str {
		return scrubbed, true
	}
	return str, lost
}

// sanitizeRawTextOnce is one pass of sanitizeRawText.
func sanitizeRawTextOnce(raw string) (string, bool) {
	budget := &nodeBudget{remaining: maxSanitizeNodes}
	sanitized, changed, lost := redactJSONTextWithBudget(raw, 0, budget)
	if !changed {
		scrubbed := scrubErrorText(raw)
		if scrubbed != raw {
			return scrubbed, true
		}
		return raw, false
	}
	if sanitized != unparseableJSONPlaceholder {
		scrubbed := scrubErrorText(sanitized)
		if scrubbed != sanitized {
			return scrubbed, true
		}
		return sanitized, lost
	}
	trimmed := trimBOMAndSpace(raw)
	if isPositivelyHarmlessProse(raw, trimmed) {
		return raw, false
	}
	return sanitized, lost
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

	for i := 0; i < len(s); i++ {
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
// optional whitespace, optionally skipping already-redacted placeholder values.
func followedByKeyDelimiterWithRedacted(rest string, skipRedacted bool) bool {
	for i, r := range rest {
		if unicode.IsSpace(r) || unicode.In(r, unicode.Z) {
			continue
		}
		if r != ':' && r != '=' {
			return false
		}
		return !skipRedacted || !startsWithRedactedValue(rest[i+1:])
	}
	return false
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

func isPositivelyHarmlessProse(raw, trimmed string) bool {
	if len(raw) > maxInspectableJSONTextBytes {
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
	// prose with balanced double quotes and no JSON container openers may be
	// already-redacted output (e.g. `"api_key":"[REDACTED]" other=ok`).
	allowRedactedProse := strings.HasPrefix(trimmed, `"`) && isBalancedAlreadyRedactedProse(raw)
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
	// Bracket- or brace-led text that failed JSON parsing and carries no
	// credential pattern is prose (e.g. `[INFO] user clicked "Submit"`).
	if strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "{") {
		return true
	}
	// Quote-led text is prose on the same basis (e.g. `"Hello," she said
	// [laughing]. "Bye."` or `"What does [this] mean?"`), unless a quoted
	// string in it decodes to a credential: the text may be a JSON string
	// literal, whose escapes the checks above do not all decode.
	return strings.HasPrefix(trimmed, `"`) && !hasJSONEscapedCredential(raw)
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

// redactJSONTextWithBudget inspects s if it is a JSON object, array, or
// JSON-encoded string literal, recursively redacting values under sensitive
// keys. When a JSON-like blob cannot be parsed or exceeds maxSanitizeDepth or
// budget, it fails closed and reports contentLost = true.
func redactJSONTextWithBudget(s string, depth int, budget *nodeBudget) (sanitized string, changed, contentLost bool) {
	trimmed := trimBOMAndSpace(s)
	if trimmed == "" || isKnownSentinel(trimmed) {
		return s, false, false
	}
	if strings.HasPrefix(trimmed, `"`) {
		if depth >= maxSanitizeDepth {
			return maxDepthPlaceholder, true, true
		}
		if budget != nil && !budget.take() {
			return sanitizeBudgetPlaceholder, true, true
		}
		if len(trimmed) > maxInspectableJSONTextBytes {
			return unparseableJSONPlaceholder, true, true
		}
		dec := json.NewDecoder(strings.NewReader(trimmed))
		var decoded string
		if err := dec.Decode(&decoded); err != nil {
			afterQuote := trimBOMAndSpace(trimmed[1:])
			if strings.HasPrefix(afterQuote, "{") || strings.HasPrefix(afterQuote, "[") ||
				strings.HasPrefix(afterQuote, `\`) || strings.HasPrefix(afterQuote, `"`) ||
				hasAnyCredentialMarker(trimmed) {
				return unparseableJSONPlaceholder, true, true
			}
			return s, false, false
		}
		suffix := trimmed[dec.InputOffset():]
		trimmedSuffix := trimBOMAndSpace(suffix)
		if trimmedSuffix != "" {
			if hasRedactableContainerText(suffix, trimmed) || strings.HasPrefix(trimmedSuffix, `\`) ||
				isUnbalancedQuotedKeyText(trimmed, decoded, suffix) ||
				(hasMalformedQuotedMembers(trimmed) && hasCandidateQuotedSensitiveKey(trimmed, 0)) ||
				hasAnyCredentialMarker(suffix) || inspectQuotedFragments(suffix, depth+1, budget) {
				return unparseableJSONPlaceholder, true, true
			}
			// Scrubbing the text may leave it malformed, as when a redacted
			// value takes a quote with it; it then fails closed here
			// already, so that a pass over the scrubbed text agrees.
			if scrubbed := scrubErrorText(trimmed); scrubbed != trimmed && strings.HasPrefix(scrubbed, `"`) {
				if again, _, _ := redactJSONTextWithBudget(scrubbed, depth+1, budget); again == unparseableJSONPlaceholder {
					return unparseableJSONPlaceholder, true, true
				}
			}
		}
		trimmedInner := trimBOMAndSpace(decoded)
		var innerSanitized string
		var innerChanged, innerLost bool
		switch {
		case strings.HasPrefix(trimmedInner, "{") || strings.HasPrefix(trimmedInner, "[") || strings.HasPrefix(trimmedInner, `"`):
			innerSanitized, innerChanged, innerLost = redactJSONTextWithBudget(decoded, depth+1, budget)
		case hasRedactableContainerText(decoded, decoded) || hasAnyCredentialMarker(decoded) || inspectQuotedFragments(decoded, depth+1, budget):
			return unparseableJSONPlaceholder, true, true
		}
		if innerChanged && (innerSanitized == unparseableJSONPlaceholder || innerSanitized == maxDepthPlaceholder || innerSanitized == sanitizeBudgetPlaceholder) {
			return innerSanitized, true, true
		}
		if !innerChanged {
			return s, false, false
		}
		b, err := json.Marshal(innerSanitized)
		if err != nil {
			return unparseableJSONPlaceholder, true, true
		}
		return string(b) + suffix, true, innerLost
	}

	if !strings.HasPrefix(trimmed, "{") && !strings.HasPrefix(trimmed, "[") {
		return s, false, false
	}
	if startsWithRedactedURI(trimmed) {
		// Text that an earlier pass left starting with a redacted URI is not
		// a JSON array but text, which is kept when redactErrorMessage and
		// sanitizeRawText would both keep it as malformed JSON (see
		// redactErrorMessageOnce and isPositivelyHarmlessProse), so that a
		// pass over it changes nothing.
		if hasAnyCredentialMarker(trimmed) || hasEncodedCredentialMarker(trimmed) || hasCandidateQuotedSensitiveKey(trimmed, 0) ||
			!isPositivelyHarmlessProse(s, trimmed) {
			return unparseableJSONPlaceholder, true, true
		}
		return s, false, false
	}
	if depth >= maxSanitizeDepth {
		return maxDepthPlaceholder, true, true
	}
	if budget != nil && !budget.take() {
		return sanitizeBudgetPlaceholder, true, true
	}
	if len(trimmed) > maxInspectableJSONTextBytes {
		return unparseableJSONPlaceholder, true, true
	}
	decoded, sawDuplicateKey, budgetExceeded, err := unmarshalJSONStrictWithBudget([]byte(trimmed), budget)
	if budgetExceeded {
		return sanitizeBudgetPlaceholder, true, true
	}
	if err != nil {
		return unparseableJSONPlaceholder, true, true
	}
	redacted, valChanged, valLost := redactJSONValue(decoded, depth+1, budget)
	if str, ok := redacted.(string); ok && str == sanitizeBudgetPlaceholder {
		return sanitizeBudgetPlaceholder, true, true
	}
	if !valChanged && !sawDuplicateKey && !valLost {
		return s, false, false
	}
	b, err := json.Marshal(redacted)
	if err != nil {
		return unparseableJSONPlaceholder, true, true
	}
	return string(b), true, valLost
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
// a "{" or "[" that may open a JSON container, which fails the text closed.
// The "[" of a complete "[REDACTED]" placeholder does not count when whole is
// balanced already-redacted prose (see isBalancedAlreadyRedactedProse) whose
// quoted sensitive keys all have redacted values, as in
// "api.key":"[REDACTED]", so that a second pass over redacted text changes
// nothing, while truncated or suffixed text, as in
// "token": "[REDACTED]", "note": "v, or text with a quoted sensitive key whose
// value the passes do not read, as in "user:token"<U+00A0>:v [REDACTED],
// still fails closed.
func hasRedactableContainerText(s, whole string) bool {
	if !strings.ContainsAny(s, "{[") {
		return false
	}
	return hasContainerOpener(s) || !isBalancedAlreadyRedactedProse(whole) || hasCandidateQuotedSensitiveKeyWithRedacted(whole, 0, true)
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
// starts at s[i] and, after any closing quotes (", ', \" or \'), ends at a
// clean token boundary (see isRedactedPlaceholderBoundary).
func isCompleteRedactedPlaceholderAt(s string, i int) bool {
	if !strings.HasPrefix(s[i:], redactedValue) {
		return false
	}
	k := i + len(redactedValue)
	for k < len(s) {
		if s[k] == '"' || s[k] == '\'' {
			k++
			continue
		}
		if k+1 < len(s) && s[k] == '\\' && (s[k+1] == '"' || s[k+1] == '\'') {
			k += 2
			continue
		}
		break
	}
	return isRedactedPlaceholderBoundary(s[k:])
}

// inspectQuotedFragments reports whether a quoted fragment of s decodes to
// text that needs redaction, or could not be decoded and starts like an
// encoded container. The callers have already scanned all of s for
// credential markers, so the work here stays linear in len(s).
func inspectQuotedFragments(s string, depth int, budget *nodeBudget) bool {
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
		if _, changed, lost := redactJSONTextWithBudget(sub[:dec.InputOffset()], depth+1, budget); changed || lost {
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
// looks only at the first character after the quote.
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

func redactJSONValue(v any, depth int, budget *nodeBudget) (out any, changed, contentLost bool) {
	if depth >= maxSanitizeDepth {
		return maxDepthPlaceholder, true, true
	}
	if budget != nil && !budget.take() {
		return sanitizeBudgetPlaceholder, true, true
	}
	switch x := v.(type) {
	case map[string]any:
		outMap := make(map[string]any, min(len(x), maxContainerAlloc))
		for k, val := range x {
			if budget != nil && budget.remaining <= 0 {
				outMap[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
				return outMap, true, true
			}
			if isSensitiveKey(k) {
				if budget != nil && !budget.take() {
					outMap[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
					return outMap, true, true
				}
				if str, ok := val.(string); !ok || str != redactedValue {
					changed = true
				}
				outMap[k] = redactedValue
				continue
			}
			newVal, c, lost := redactJSONValue(val, depth+1, budget)
			if str, ok := newVal.(string); ok && str == sanitizeBudgetPlaceholder && budget != nil && budget.remaining <= 0 {
				outMap[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
				return outMap, true, true
			}
			outMap[k] = newVal
			changed = changed || c
			contentLost = contentLost || lost
		}
		return outMap, changed, contentLost
	case []any:
		outSlice := make([]any, 0, min(len(x), maxContainerAlloc))
		for _, val := range x {
			if budget != nil && budget.remaining <= 0 {
				outSlice = append(outSlice, sanitizeBudgetPlaceholder)
				return outSlice, true, true
			}
			newVal, c, lost := redactJSONValue(val, depth+1, budget)
			outSlice = append(outSlice, newVal)
			changed = changed || c
			contentLost = contentLost || lost
			if str, ok := newVal.(string); ok && str == sanitizeBudgetPlaceholder && budget != nil && budget.remaining <= 0 {
				return outSlice, true, true
			}
		}
		return outSlice, changed, contentLost
	case string:
		return redactJSONTextWithBudget(x, depth+1, budget)
	default:
		return v, false, false
	}
}

func unmarshalJSONStrictWithBudget(b []byte, budget *nodeBudget) (val any, sawDuplicateKey, budgetExceeded bool, err error) {
	dec := json.NewDecoder(strings.NewReader(string(b)))
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
	out := redactErrorMessageOnce(msg)
	for cur, pass := msg, 1; out != cur && !isScrubbedPlainText(out); pass++ {
		if pass == maxRedactionPasses {
			return redactedSensitiveTextPlaceholder
		}
		next := redactErrorMessageOnce(out)
		if next == out {
			break
		}
		cur, out = out, next
	}
	return out
}

// redactErrorMessageOnce is one pass of redactErrorMessage (see
// maxRedactionPasses).
func redactErrorMessageOnce(msg string) string {
	if msg == "" || isKnownSentinel(msg) {
		return msg
	}
	budget := &nodeBudget{remaining: maxSanitizeNodes}
	trimmed := trimBOMAndSpace(msg)
	if strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
		if decoded, sawDuplicateKey, exceeded, err := unmarshalJSONStrictWithBudget([]byte(trimmed), budget); err == nil && !exceeded {
			redacted, valChanged, valLost := redactJSONValue(decoded, 1, budget)
			scrubbed, strChanged := scrubErrorJSONStrings(redacted, 1)
			if valChanged || sawDuplicateKey || valLost || strChanged {
				if b, err := json.Marshal(scrubbed); err == nil {
					return string(b)
				}
			}
			return msg
		}
		// Malformed JSON document: if it contains any sensitive key pattern or credential marker,
		// fail closed rather than falling through to partial regex rewriting.
		if hasAnyCredentialMarker(trimmed) || hasEncodedCredentialMarker(trimmed) || hasCandidateQuotedSensitiveKey(trimmed, 0) {
			return unparseableJSONPlaceholder
		}
		return scrubErrorText(msg)
	}
	if strings.HasPrefix(trimmed, `"`) {
		if redacted, ok, _ := redactJSONTextWithBudget(msg, 0, budget); ok {
			// As in sanitizeRawText, quote-led prose that only failed JSON
			// parsing is kept.
			if redacted == unparseableJSONPlaceholder && isPositivelyHarmlessProse(msg, trimmed) {
				return msg
			}
			return scrubErrorText(redacted)
		}
	}
	return scrubErrorText(msg)
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
// escapes exposes a credential or the scrub keeps a word that the legacy
// expressions redact (see leavesLegacyValue). Its result r is a placeholder,
// or text that scrubErrorTokens leaves unchanged and that passes those checks
// as r itself, so that scrubErrorText(r) == r.
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
	scrubbed := m.tokens(msg)
	if scrubbed == redactedSensitiveTextPlaceholder {
		return redactedSensitiveTextPlaceholder
	}
	if exposesEncodedPattern(scrubbed, m.tokens) || hasBackslashEscapedCredentialWith(msg, scrubbed, m.tokens) {
		return redactedSensitiveTextPlaceholder
	}
	if scrubbed == msg {
		if m.leavesLegacyValue(msg, scrubbed) {
			return redactedSensitiveTextPlaceholder
		}
		return scrubbed
	}
	// Scrub the scrubbed text again until it no longer changes (see
	// maxRedactionPasses), and check the result as a second scrub of it
	// would, so that scrubbing it again changes nothing.
	for prev, pass := msg, 1; scrubbed != prev; pass++ {
		if pass == maxRedactionPasses {
			return redactedSensitiveTextPlaceholder
		}
		prev, scrubbed = scrubbed, m.tokens(scrubbed)
		if scrubbed == redactedSensitiveTextPlaceholder {
			return redactedSensitiveTextPlaceholder
		}
	}
	if exposesEncodedPattern(scrubbed, m.tokens) || hasBackslashEscapedCredentialWith(scrubbed, scrubbed, m.tokens) ||
		m.leavesLegacyValue(msg, scrubbed) || m.leavesLegacyValue(scrubbed, scrubbed) {
		return redactedSensitiveTextPlaceholder
	}
	return scrubbed
}

// scrubMemo remembers, within one scrubErrorText, what scrubErrorTokens and
// the two legacy scrubs (see leavesLegacyValue) returned for the last texts
// they were given. Its checks scrub the same texts, and the same decoded
// texts, more than once, and each scrub reads the whole text.
type scrubMemo struct {
	// component selects the scrubs of one decoded URI component, which read
	// no URI (see scrubURIComponentText).
	component                           bool
	tokenRuns, legacyRuns, compoundRuns memoRuns
}

// memoRuns holds the last scrubs of one function.
type memoRuns struct {
	in, out [4]string
	n       int
}

func (r *memoRuns) do(s string, scrub func(string) string) string {
	for k := range min(r.n, len(r.in)) {
		if r.in[k] == s {
			return r.out[k]
		}
	}
	out := scrub(s)
	k := r.n % len(r.in)
	r.in[k], r.out[k] = s, out
	r.n++
	return out
}

func (m *scrubMemo) tokens(s string) string {
	if m.component {
		return m.tokenRuns.do(s, scrubTextPatterns)
	}
	return m.tokenRuns.do(s, scrubErrorTokens)
}

func (m *scrubMemo) legacy(s string) string {
	if m.component {
		return m.legacyRuns.do(s, legacyScrubComponentText)
	}
	return m.legacyRuns.do(s, legacyScrubErrorTokens)
}

func (m *scrubMemo) compound(s string) string {
	if m.component {
		return m.compoundRuns.do(s, compoundLegacyScrubComponentText)
	}
	return m.compoundRuns.do(s, compoundLegacyScrubErrorTokens)
}

// keptLiterals returns the non-redacted literal slices left in scrubbed after
// removing redaction placeholders, in order.
func keptLiterals(scrubbed string) []string {
	parts := []string{scrubbed}
	for _, sentinel := range []string{redactedURIPlaceholder, redactedValue, url.QueryEscape(redactedValue)} {
		var next []string
		for _, part := range parts {
			for _, piece := range strings.Split(part, sentinel) {
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
	if strings.Contains(s, `\`) {
		s = unescapeBackslashControls(s)
	}
	if strings.ContainsAny(s, "%xXuU") {
		s = canonicalizeCredentialMarker(s)
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
	var parts []string
	curRaw, curScrubbed := raw, scrubbed
	for range 2 * maxCanonicalUnescapePasses {
		if !strings.ContainsAny(curRaw, `\%`) {
			return false
		}
		nextRaw := unescapeCredentialStep(curRaw)
		if nextRaw == curRaw {
			return false
		}
		if curScrubbed == curRaw {
			// Nothing was redacted, so the scrubbed text decodes to nextRaw
			// too and one scrub of it answers both checks below.
			if scrub(nextRaw) != nextRaw || exposesEncodedPattern(nextRaw, scrub) {
				return true
			}
			curRaw, curScrubbed = nextRaw, nextRaw
			continue
		}
		if parts == nil {
			parts = keptLiterals(curScrubbed)
		}
		nextScrubbed := unescapeCredentialStep(curScrubbed)
		for j, part := range parts {
			parts[j] = unescapeCredentialStep(part)
		}
		if scrub(nextScrubbed) != nextScrubbed || exposesEncodedPattern(nextScrubbed, scrub) {
			return true
		}
		if !keptInOrder(parts, scrub(nextRaw)) {
			return true
		}
		curRaw, curScrubbed = nextRaw, nextScrubbed
	}
	return unescapeCredentialStep(curRaw) != curRaw
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
	if msg == "" || isKnownSentinel(msg) {
		return false
	}
	cleaned := strings.ReplaceAll(msg, "%5BREDACTED%5D", redactedValue)
	cleaned = strings.ReplaceAll(cleaned, "%5bredacted%5d", redactedValue)
	canon := canonicalizeCredentialMarker(cleaned)
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
