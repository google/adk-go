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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"google.golang.org/adk/v2/session"
)

const (
	redactedValue                    = "[REDACTED]"
	redactedURIPlaceholder           = "[REDACTED_SENSITIVE_URI]"
	redactedSensitiveTextPlaceholder = "[REDACTED_SENSITIVE_TEXT]"
	formattingFailedPlaceholder      = "[FORMATTER_FAILED]"
	unparseableJSONPlaceholder       = "[UNPARSEABLE_JSON_BLOB]"

	maxInspectableJSONTextBytes  = 4_000_000 // adk-python _MAX_JSON_INSPECT_CHARS
	maxCanonicalUnescapePasses   = 4
	maxQuotedFragmentsPerInspect = 64
	maxRedactionPasses           = 4
	maxBasicCredentialBytes      = 16 * 1024
	maxNestedKeyLen              = 128
	maxRewriteGrowth             = 8
	maxRewriteSlack              = 4096

	regexpSpace        = "\t\n\f\r "
	delimSpace         = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\v\f\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f \x7f"
	closingPunctuation = `}])"'\:.`
)

var (
	authHeaderRe    = regexp.MustCompile(`(?im)\b((?:proxy[_-])?author[i\x{0130}]zat[i\x{0130}]on|x[_-]ap[i\x{0130}][_-]key|ap[i\x{0130}][_-]key)[ \t]*:[ \t]*[^\s\r\n][^\r\n]*`)
	authHeaderKeyRe = regexp.MustCompile(`(?i)\b(?:(?:proxy[_-])?author[i\x{0130}]zat[i\x{0130}]on|x[_-]ap[i\x{0130}][_-]key|ap[i\x{0130}][_-]key)[ \t]*:`)
	bearerTokenRe   = regexp.MustCompile(`(?i)\b(bearer)\s+("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\[[^\]\r\n\s]+\][^\s,;)"']*|[^\s,;)"'\]]+)`)
	basicTokenRe    = regexp.MustCompile(`(?i)\b(basic)\s+([A-Za-z0-9+/]+={0,2})`)

	// jsonKeyPrefixRe stops right at ":" so FindAllStringSubmatchIndex never
	// consumes adjacent keys; Go-side isValidJSONPatternKey filters candidates.
	jsonKeyPrefixRe   = regexp.MustCompile(`"[\s\p{Z}]*([^"'\s\r\n,}](?:[^"'\r\n,}]*?[^"'\s\r\n,}])?)[\s\p{Z}]*"\s*:`)
	nestedKeyNameRe   = regexp.MustCompile(`(?i)^(?:\\?["'])?(?:(?:` + regexp.QuoteMeta(session.KeyPrefixUser) + `|` + regexp.QuoteMeta(session.KeyPrefixApp) + `)*)(` + fixedSensitiveKeyRegex() + `)$`)
	compoundKeyNameRe = regexp.MustCompile(`(?i)^(?:\\?["'])?(?:[^\s]*[^\w\s])?(` + bareCandidateKeyRegex() + `)(?:\\?["'])?$`)
	quotedValueRe     = regexp.MustCompile(`^(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|` + escapedQuotedRegex(`"`) + `|` + escapedQuotedRegex(`'`) + `)`)
)

func escapedQuotedRegex(q string) string {
	return `\\` + q + `(?:\\\\(?:\\\\|\\` + q + `|[^\\` + q + `])|\\[^\\` + q + `]|[^\\` + q + `])*\\` + q
}

func trimBOMAndSpace(s string) string {
	return strings.TrimFunc(s, func(r rune) bool { return unicode.IsSpace(r) || r == '\ufeff' })
}

func isKnownSentinel(s string) bool {
	switch s {
	case redactedValue, redactedURIPlaceholder, redactedSensitiveTextPlaceholder,
		formattingFailedPlaceholder, unparseableJSONPlaceholder, inlineDataText,
		binaryDataPlaceholder, circularRefPlaceholder, maxDepthPlaceholder,
		sanitizeBudgetPlaceholder, unsupportedKeyPlaceholder, unsupportedObjPlaceholder:
		return true
	}
	return false
}

// runFixedPoint repeats step while its output changes and still starts with
// "{", "[", or "\"", up to maxRedactionPasses times, failing closed to
// "[REDACTED_SENSITIVE_TEXT]" if it does not settle.
//
// Pass 2 is required when Pass 1 re-serializes nested JSON (with keys sorted)
// and text scrubbing then breaks the inner JSON leaf while leaving the outer
// container valid; for example:
//
//	{"k":"{\"zz_a\":\"access_token='\",\"b_sec\":\"SECRETZQ\",\"c_close\":\"'\",\"z\":\"{\\\"DB_PASSWORD\\\":\\\"p\\\"}\"}"}
//
// Pass 2 fails the corrupted inner leaf closed to "[UNPARSEABLE_JSON_BLOB]"
// instead of leaving "SECRETZQ" exposed. It also ensures idempotency when
// Pass 1 scrubs a URI inside a JSON string to "[REDACTED_SENSITIVE_URI] ...",
// such as {"log":"https://u:p@h/a \"token\":\"SECRET\""}.
func runFixedPoint(in string, step func(s string, laterPass bool) (out string, lost, settled bool)) (string, bool) {
	out, lost, settled := step(in, false)
	for cur, pass := in, 1; !settled && out != cur; pass++ {
		if t := trimBOMAndSpace(out); t != "" && t[0] != '{' && t[0] != '[' && t[0] != '"' {
			break
		}
		if pass == maxRedactionPasses {
			return redactedSensitiveTextPlaceholder, true
		}
		next, nextLost, nextSettled := step(out, true)
		if next == out {
			break
		}
		cur = out
		out = next
		lost = lost || nextLost
		settled = nextSettled
	}
	return out, lost
}

func redactStringValue(str string, depth int, budget *nodeBudget) (string, bool) {
	return redactTextEntry(str, depth, budget, false)
}

func sanitizeRawText(raw string) (string, bool) {
	return redactTextEntry(raw, 0, nil, true)
}

// redactTextEntry runs JSON-aware redaction followed by text-pattern scrubbing
// until fixed point. sanitizeRawText sets checkHarmlessOnFirstPass so malformed
// JSON-like prose (such as "[INFO] ready") stays readable on pass 1, whereas
// redactStringValue (used by smartTruncate) fails malformed "{"/"["/"\"" strings
// closed on pass 1 and only relaxes that check on later passes over output that
// pass 1 already scrubbed.
func redactTextEntry(in string, depth int, budget *nodeBudget, checkHarmlessOnFirstPass bool) (string, bool) {
	return runFixedPoint(in, func(s string, laterPass bool) (string, bool, bool) {
		b := budget
		if b == nil || laterPass {
			b = &nodeBudget{remaining: maxSanitizeNodes}
		}
		redacted, changed, lost := redactJSONTextPass(s, depth, laterPass, b)
		if !changed {
			if scrubbed := scrubErrorText(s); scrubbed != s {
				return scrubbed, true, false
			}
			return s, false, true
		}
		if redacted == unparseableJSONPlaceholder {
			if (checkHarmlessOnFirstPass || laterPass) && isPositivelyHarmlessProse(s, trimBOMAndSpace(s), laterPass) {
				return s, false, true
			}
			return redacted, lost, true
		}
		if scrubbed := scrubErrorText(redacted); scrubbed != redacted {
			if startsWithJSONContainer(s) && !json.Valid([]byte(scrubbed)) {
				return redactedSensitiveTextPlaceholder, true, true
			}
			return scrubbed, true, false
		}
		return redacted, lost, !lost
	})
}

// redactErrorMessage scrubs credentials from an error string before it is
// written to error_message. Valid JSON documents are redacted by key and then
// scrubbed leaf-by-leaf via scrubErrorJSONStrings so a header in one string
// leaf does not erase sibling JSON fields.
func redactErrorMessage(msg string) string {
	out, _ := runFixedPoint(msg, func(s string, laterPass bool) (string, bool, bool) {
		if s == "" || isKnownSentinel(s) {
			return s, false, true
		}
		budget := &nodeBudget{remaining: maxSanitizeNodes}
		trimmed := trimBOMAndSpace(s)
		if strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
			if decoded, sawDup, exceeded, err := unmarshalJSONStrictWithBudget(trimmed, budget); err == nil && !exceeded {
				redacted, valChanged, valLost := redactJSONValue(decoded, 1, laterPass, true, budget)
				scrubbed, strChanged := scrubErrorJSONStrings(redacted, 1)
				if valChanged || sawDup || valLost || strChanged {
					b, err := json.Marshal(scrubbed)
					if err != nil {
						return unparseableJSONPlaceholder, true, true
					}
					return string(b), true, !strChanged && !valLost
				}
				return s, false, true
			}
			if hasAnyCredentialMarker(trimmed) || hasEncodedCredentialMarker(trimmed) || hasCandidateQuotedSensitiveKey(trimmed, 0, false) {
				return unparseableJSONPlaceholder, true, true
			}
		} else if strings.HasPrefix(trimmed, `"`) {
			if redacted, ok, _ := redactJSONTextPass(s, 0, laterPass, budget); ok {
				if redacted == unparseableJSONPlaceholder && isPositivelyHarmlessProse(s, trimmed, laterPass) {
					return s, false, true
				}
				scrubbed := scrubErrorText(redacted)
				return scrubbed, true, scrubbed == redacted
			}
		}
		scrubbed := scrubErrorText(s)
		return scrubbed, scrubbed != s, scrubbed == s
	})
	return out
}

func startsWithJSONContainer(s string) bool {
	t := trimBOMAndSpace(s)
	return strings.HasPrefix(t, "{") || strings.HasPrefix(t, "[")
}

func hasQuotedSensitiveKey(s string, skipRedacted bool) bool {
	if !mayMatchTextPatterns(s) || !strings.ContainsAny(s, "\"'") || !strings.ContainsAny(s, ":=") {
		return false
	}
	for i, prev := 0, -1; i < len(s); i++ {
		if s[i] != '"' && s[i] != '\'' {
			continue
		}
		if prev >= 0 && isValidJSONPatternKey(trimBOMAndSpace(s[prev+1:i])) && followedByKeyDelimiter(s[i+1:], skipRedacted) {
			return true
		}
		prev = i
	}
	return false
}

func hasCandidateQuotedSensitiveKey(s string, depth int, skipRedacted bool) bool {
	if depth > 4 || s == "" {
		return false
	}
	if len(s) > maxInspectableJSONTextBytes {
		s = s[:maxInspectableJSONTextBytes]
	}
	if hasQuotedSensitiveKey(s, skipRedacted) {
		return true
	}
	if canon := canonicalizeCredentialMarker(s); canon != s && hasQuotedSensitiveKey(canon, skipRedacted && !strings.ContainsAny(canon, "{[")) {
		return true
	}
	var quoteChar byte
	start, escaped := 0, false
	for i, b := range []byte(s) {
		if quoteChar == 0 {
			if b == '"' || b == '\'' {
				quoteChar, start, escaped = b, i, false
			}
			continue
		}
		if escaped || b == '\\' {
			escaped = !escaped
			continue
		}
		if b != quoteChar {
			continue
		}
		q, candidate := quoteChar, canonicalizeCredentialMarker(s[start+1:i])
		quoteChar = 0
		if q == '"' {
			var unquoted string
			if json.Unmarshal([]byte(s[start:i+1]), &unquoted) == nil {
				candidate = unquoted
			}
		}
		if (isSensitiveKey(candidate) || isSensitiveKey(canonicalizeCredentialMarker(candidate))) && followedByKeyDelimiter(s[i+1:], skipRedacted) {
			return true
		}
		if strings.ContainsAny(candidate, `"'`) && hasCandidateQuotedSensitiveKey(candidate, depth+1, skipRedacted && !strings.ContainsAny(candidate, "{[")) {
			return true
		}
	}
	return false
}

func followedByKeyDelimiter(rest string, skipRedacted bool) bool {
	sawObfuscated := false
	for i, r := range rest {
		if r <= 0x20 || r == 0x7f || unicode.IsSpace(r) || unicode.In(r, unicode.Z, unicode.Cf) {
			sawObfuscated = sawObfuscated || (r != ' ' && r != '\t' && r != '\n' && r != '\r' && r != '\f')
			continue
		}
		return (r == ':' || r == '=') && (sawObfuscated || !skipRedacted || !isRedactedPlaceholderPrefix(rest[i+1:], true))
	}
	return false
}

func stripLeadingClosers(s, chars string) string {
	for len(s) > 0 {
		if strings.IndexByte(chars, s[0]) >= 0 {
			s = s[1:]
		} else if len(s) >= 2 && s[0] == '\\' && (s[1] == '"' || s[1] == '\'') {
			s = s[2:]
		} else {
			break
		}
	}
	return s
}

func isRedactedPlaceholderPrefix(s string, allowLeadingSpace bool) bool {
	if allowLeadingSpace {
		s = strings.TrimLeftFunc(s, func(r rune) bool { return unicode.IsSpace(r) || unicode.In(r, unicode.Z) })
	}
	for _, q := range []string{`\"`, `\'`, `"`, `'`, ""} {
		if !allowLeadingSpace && q != "" {
			continue
		}
		tail, ok := strings.CutPrefix(s, q+redactedValue+q)
		if !ok {
			continue
		}
		if !allowLeadingSpace {
			tail = stripLeadingClosers(tail, `"'`)
		}
		if tail == "" || strings.IndexByte(regexpSpace, tail[0]) >= 0 || strings.IndexByte(",;&", tail[0]) >= 0 {
			return true
		}
		if tail[0] == ':' {
			return strings.TrimLeftFunc(tail[1:], func(r rune) bool { return r <= 0x20 || r == 0x7f || unicode.IsSpace(r) || unicode.In(r, unicode.Z) }) == ""
		}
		if strings.IndexByte("}].)", tail[0]) >= 0 {
			tail = stripLeadingClosers(tail, `}].)"'`)
			return tail == "" || strings.IndexByte(regexpSpace, tail[0]) >= 0 || strings.IndexByte(",;&", tail[0]) >= 0
		}
	}
	return false
}

func hasUnescapedQuoteEscape(s string) bool {
	bs := 0
	for i := range len(s) {
		if s[i] == '\\' {
			bs++
		} else if s[i] == '"' && bs%2 == 1 {
			return true
		} else {
			bs = 0
		}
	}
	return false
}

func isBalancedAlreadyRedactedProse(raw string) bool {
	quotes := 0
	for i, c := range []byte(raw) {
		switch {
		case c == '"':
			quotes++
		case c == '{', c == '[' && !isRedactedPlaceholderPrefix(raw[i:], false),
			c == '\\' && i+1 < len(raw) && (raw[i+1] == '"' || raw[i+1] == '\''):
			return false
		}
	}
	return quotes%2 == 0
}

// isPositivelyHarmlessProse reports whether malformed "{", "[", or "\""-led
// text is safe to return verbatim instead of "[UNPARSEABLE_JSON_BLOB]". It
// requires the text to be within maxInspectableJSONTextBytes and free of
// unescaped quote escapes, text/encoded/JSON-escaped credentials, and quoted
// sensitive keys (except already-redacted values in balanced quote-led output
// produced by an earlier pass).
func isPositivelyHarmlessProse(raw, trimmed string, laterPass bool) bool {
	if len(raw) > maxInspectableJSONTextBytes || hasUnescapedQuoteEscape(raw) ||
		(laterPass && strings.HasPrefix(trimmed, `"`) && startsLikeEncodedContainer(trimmed)) ||
		scrubErrorText(raw) != raw || hasEncodedCredentialMarker(raw) {
		return false
	}
	allowRedacted := laterPass && strings.HasPrefix(trimmed, `"`) && isBalancedAlreadyRedactedProse(raw)
	if hasCandidateQuotedSensitiveKey(raw, 0, allowRedacted) {
		return false
	}
	canon := canonicalizeCredentialMarker(raw)
	if hasQuotedSensitiveKey(canon, allowRedacted && (canon == raw || !strings.ContainsAny(canon, "{["))) {
		return false
	}
	return (strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, `"`)) && !hasJSONEscapedCredential(raw)
}

func hasJSONEscapedCredential(s string) bool {
	if !strings.Contains(s, `\`) {
		return false
	}
	for open := strings.IndexByte(s, '"'); open >= 0; {
		n := strings.IndexByte(s[open+1:], '"')
		if n < 0 {
			return false
		}
		closing := open + 1 + n
		var decoded string
		if q := s[open : closing+1]; strings.Contains(q, `\`) && json.Unmarshal([]byte(q), &decoded) == nil && hasAnyCredentialMarker(decoded) {
			return true
		}
		open = closing
	}
	return false
}

func rewritesTooLong(in string, out []byte) bool {
	return len(out) > maxInspectableJSONTextBytes || len(out) > maxRewriteGrowth*len(in)+maxRewriteSlack
}

func startsWithRedactedURI(s string) bool {
	rest, ok := strings.CutPrefix(s, redactedURIPlaceholder)
	return ok && (rest == "" || strings.IndexByte(regexpSpace, rest[0]) >= 0 || strings.IndexByte("\"'<>.,;:)]}", rest[0]) >= 0)
}

func hasRedactableContainerText(s, whole string, laterPass bool) bool {
	if !strings.ContainsAny(s, "{[") {
		return false
	}
	if !laterPass || !isBalancedAlreadyRedactedProse(whole) || hasCandidateQuotedSensitiveKey(whole, 0, true) {
		return true
	}
	for i, c := range []byte(s) {
		if c == '{' || (c == '[' && !isRedactedPlaceholderPrefix(s[i:], false)) {
			return true
		}
	}
	return false
}

// redactJSONTextPass parses and redacts a "{", "[", or "\""-led string under
// depth and node-budget limits, failing closed to a placeholder on malformed
// or oversized inputs. laterPass indicates s came from an earlier redaction
// pass of the same call, allowing "[REDACTED_SENSITIVE_URI]"-led text and
// balanced already-redacted fragments to settle without failing closed.
func redactJSONTextPass(s string, depth int, laterPass bool, budget *nodeBudget) (sanitized string, changed, contentLost bool) {
	trimmed := trimBOMAndSpace(s)
	if trimmed == "" || isKnownSentinel(trimmed) || strings.IndexByte(`{["`, trimmed[0]) < 0 {
		return s, false, false
	}
	if laterPass && trimmed[0] != '"' && startsWithRedactedURI(trimmed) {
		if hasAnyCredentialMarker(trimmed) || hasEncodedCredentialMarker(trimmed) || hasCandidateQuotedSensitiveKey(trimmed, 0, false) || !isPositivelyHarmlessProse(s, trimmed, true) {
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
	if trimmed[0] == '"' {
		dec := json.NewDecoder(strings.NewReader(trimmed))
		var decoded string
		if err := dec.Decode(&decoded); err != nil {
			if startsLikeEncodedContainer(trimmed) || hasAnyCredentialMarker(trimmed) {
				return unparseableJSONPlaceholder, true, true
			}
			return s, false, false
		}
		suffix := trimmed[dec.InputOffset():]
		if ts := trimBOMAndSpace(suffix); ts != "" && (hasRedactableContainerText(suffix, trimmed, laterPass) ||
			strings.HasPrefix(ts, `\`) || hasAnyCredentialMarker(suffix) || inspectQuotedFragments(suffix, depth+1, laterPass, budget)) {
			return unparseableJSONPlaceholder, true, true
		}
		trimmedInner := trimBOMAndSpace(decoded)
		var innerSan string
		var innerChanged, innerLost bool
		if trimmedInner != "" && strings.IndexByte(`{["`, trimmedInner[0]) >= 0 {
			innerSan, innerChanged, innerLost = redactJSONTextPass(decoded, depth+1, laterPass, budget)
		} else if hasRedactableContainerText(decoded, decoded, laterPass) || hasAnyCredentialMarker(decoded) || inspectQuotedFragments(decoded, depth+1, laterPass, budget) {
			return unparseableJSONPlaceholder, true, true
		}
		if innerChanged && isKnownSentinel(innerSan) && innerSan != redactedValue {
			return innerSan, true, true
		}
		if !innerChanged {
			return s, false, false
		}
		b, err := json.Marshal(innerSan)
		if err != nil || rewritesTooLong(trimmed[:dec.InputOffset()], b) {
			return unparseableJSONPlaceholder, true, true
		}
		return string(b) + suffix, true, innerLost
	}
	decoded, sawDup, exceeded, err := unmarshalJSONStrictWithBudget(trimmed, budget)
	if exceeded {
		return sanitizeBudgetPlaceholder, true, true
	}
	if err != nil {
		return unparseableJSONPlaceholder, true, true
	}
	redacted, valChanged, valLost := redactJSONValue(decoded, depth+1, laterPass, false, budget)
	if str, ok := redacted.(string); ok && (str == sanitizeBudgetPlaceholder || (str == redactedSensitiveTextPlaceholder && valLost)) {
		return str, true, true
	}
	if !valChanged && !sawDup && !valLost {
		return s, false, false
	}
	b, err := json.Marshal(redacted)
	if err != nil || rewritesTooLong(trimmed, b) {
		return unparseableJSONPlaceholder, true, true
	}
	return string(b), true, valLost
}

// inspectQuotedFragments reports whether any quoted JSON string inside s
// requires redaction, fails closed, or exceeds maxQuotedFragmentsPerInspect.
// failedUntil caches failed decode spans so unclosed quotes stay O(n).
func inspectQuotedFragments(s string, depth int, laterPass bool, budget *nodeBudget) bool {
	if depth >= maxSanitizeDepth {
		return true
	}
	for i, fragments, failedUntil := 0, 0, 0; i < len(s); i++ {
		if s[i] != '"' {
			continue
		}
		if fragments++; fragments > maxQuotedFragmentsPerInspect || (budget != nil && !budget.take()) {
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
			if errors.Is(err, io.ErrUnexpectedEOF) {
				failedUntil = len(s)
			} else if errors.As(err, &syntaxErr) && syntaxErr.Offset > 1 {
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
		if n := int(dec.InputOffset()); n > 1 {
			i += n - 1
		}
	}
	return false
}

func startsLikeEncodedContainer(s string) bool {
	after := strings.TrimLeftFunc(s[1:], func(r rune) bool { return unicode.IsSpace(r) || r == '\ufeff' })
	if strings.HasPrefix(after, "[") && isRedactedPlaceholderPrefix(after, false) {
		return false
	}
	return len(after) > 0 && strings.IndexByte(`{[\"`, after[0]) >= 0
}

func budgetTruncatedMap(outMap map[string]any, lastKey string, hasLastKey bool) (any, bool, bool) {
	if len(outMap) == 0 {
		return sanitizeBudgetPlaceholder, true, true
	}
	if hasLastKey {
		delete(outMap, lastKey)
	}
	outMap[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
	return outMap, true, true
}

// redactJSONValue redacts the values of sensitive keys in a decoded JSON value
// and the JSON text held in its strings.
//
// In v0.1.0, compound sensitive keys were not recognized as JSON keys, so a
// header pattern inside a compound key's string value ran to end-of-line during
// the whole-text scrub. To preserve that v0.1.0 redaction floor on text entry
// points (keepSiblings=false), a non-base sensitive key whose string holds a
// header or Bearer marker, or any string child that collapses to
// "[REDACTED_SENSITIVE_TEXT]", replaces the whole document with
// "[REDACTED_SENSITIVE_TEXT]". redactErrorMessage sets keepSiblings=true and
// scrubs each string independently so sibling fields remain readable.
func redactJSONValue(v any, depth int, laterPass, keepSiblings bool, budget *nodeBudget) (out any, changed, contentLost bool) {
	if depth >= maxSanitizeDepth {
		return maxDepthPlaceholder, true, true
	}
	if budget != nil && !budget.take() {
		return sanitizeBudgetPlaceholder, true, true
	}
	switch x := v.(type) {
	case map[string]any:
		outMap := make(map[string]any, min(len(x), maxContainerAlloc))
		var lastKey string
		var hasLastKey bool
		for k, val := range x {
			if budget != nil && budget.remaining <= 0 {
				return budgetTruncatedMap(outMap, lastKey, hasLastKey)
			}
			if isSensitiveKey(k) {
				if budget != nil && !budget.take() {
					return budgetTruncatedMap(outMap, lastKey, hasLastKey)
				}
				if str, ok := val.(string); ok {
					if !keepSiblings && !isBaseSensitiveKey(k, true) && hasCrossFieldRunOnValue(str) {
						return redactedSensitiveTextPlaceholder, true, true
					}
					changed = changed || str != redactedValue
				} else {
					changed = true
				}
				outMap[k] = redactedValue
				lastKey = k
				hasLastKey = true
				continue
			}
			newVal, c, lost := redactJSONValue(val, depth+1, laterPass, keepSiblings, budget)
			if str, ok := newVal.(string); ok {
				if !keepSiblings && str == redactedSensitiveTextPlaceholder && lost {
					return redactedSensitiveTextPlaceholder, true, true
				}
				if str == sanitizeBudgetPlaceholder && budget != nil && budget.remaining <= 0 {
					return budgetTruncatedMap(outMap, lastKey, hasLastKey)
				}
			}
			outMap[k] = newVal
			lastKey = k
			hasLastKey = true
			changed = changed || c
			contentLost = contentLost || lost
		}
		return outMap, changed, contentLost
	case []any:
		outSlice := make([]any, 0, min(len(x), maxContainerAlloc))
		for _, val := range x {
			if budget != nil && budget.remaining <= 0 {
				if len(outSlice) == 0 {
					return sanitizeBudgetPlaceholder, true, true
				}
				outSlice[len(outSlice)-1] = sanitizeBudgetPlaceholder
				return outSlice, true, true
			}
			newVal, c, lost := redactJSONValue(val, depth+1, laterPass, keepSiblings, budget)
			if str, ok := newVal.(string); ok && !keepSiblings && str == redactedSensitiveTextPlaceholder && lost {
				return redactedSensitiveTextPlaceholder, true, true
			}
			outSlice = append(outSlice, newVal)
			changed = changed || c
			contentLost = contentLost || lost
			if str, ok := newVal.(string); ok && str == sanitizeBudgetPlaceholder && budget != nil && budget.remaining <= 0 {
				if len(outSlice) <= 1 {
					return sanitizeBudgetPlaceholder, true, true
				}
				outSlice[len(outSlice)-2] = sanitizeBudgetPlaceholder
				return outSlice[:len(outSlice)-1], true, true
			}
		}
		return outSlice, changed, contentLost
	case string:
		return redactJSONTextPass(x, depth+1, laterPass, budget)
	default:
		return v, false, false
	}
}

func unmarshalJSONStrictWithBudget(s string, budget *nodeBudget) (val any, sawDup, exceeded bool, err error) {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	if val, sawDup, exceeded, err = decodeJSONValue(dec, 0, budget); err != nil || exceeded {
		return nil, false, exceeded, err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, false, false, errors.New("unexpected data after JSON value")
	}
	return val, sawDup, false, nil
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
	if delim == '{' {
		m, sawDup := make(map[string]any), false
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
			_, exists := m[key]
			sawDup, m[key] = sawDup || childDup || exists, val
		}
		if endTok, err := dec.Token(); err != nil || endTok != json.Delim('}') {
			return nil, false, false, errors.New("unclosed JSON object")
		}
		return m, sawDup, false, nil
	}
	if delim == '[' {
		s, sawDup := []any{}, false
		for dec.More() {
			val, childDup, exceeded, err := decodeJSONValue(dec, depth+1, budget)
			if err != nil || exceeded {
				return nil, false, exceeded, err
			}
			sawDup, s = sawDup || childDup, append(s, val)
		}
		if endTok, err := dec.Token(); err != nil || endTok != json.Delim(']') {
			return nil, false, false, errors.New("unclosed JSON array")
		}
		return s, sawDup, false, nil
	}
	return nil, false, false, errors.New("unexpected JSON delimiter")
}

func scrubErrorJSONStrings(v any, depth int) (any, bool) {
	if depth >= maxSanitizeDepth {
		return v, false
	}
	switch x := v.(type) {
	case map[string]any:
		changed, out := false, make(map[string]any, len(x))
		for k, val := range x {
			newVal, c := scrubErrorJSONStrings(val, depth+1)
			out[k], changed = newVal, changed || c
		}
		return out, changed
	case []any:
		changed, out := false, make([]any, len(x))
		for i, val := range x {
			newVal, c := scrubErrorJSONStrings(val, depth+1)
			out[i], changed = newVal, changed || c
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

func hasCrossFieldRunOnValue(str string) bool {
	if !mayMatchTextPatterns(str) {
		return false
	}
	if authHeaderKeyRe.MatchString(str) {
		return true
	}
	canon := canonicalizeCredentialMarker(str)
	return authHeaderKeyRe.MatchString(canon) || (strings.ContainsAny(canon, `"'`) && strings.Contains(strings.ToLower(canon), "bearer"))
}

// scrubErrorText runs scrubErrorTokens until it reaches a fixed point (up to
// maxRedactionPasses times, failing closed to "[REDACTED_SENSITIVE_TEXT]") and
// fails closed if encoded or backslash-escaped credentials survive.
//
// The loop is required when Pass 1's key=value pass collapses a multiline
// header onto one line (for example, "Authorization:\n  Bearer SECRET123" ->
// "Authorization:[REDACTED] [REDACTED]", or "api_key:\n  SECRET123\nregion: us"
// -> "api_key:[REDACTED]\nregion: us") so Pass 2's header pattern normalizes it
// to "Authorization: [REDACTED]" / "api_key: [REDACTED]\nregion: us".
//
// The asciiOnly backslash retry preserves v0.1.0 fail-closed behavior when a
// double-escaped Unicode format rune sits next to a URI token, as in:
// `{"s":"https\\ufeff://h/a;Bearer+SECRETZQ+token=x"}`.
func scrubErrorText(msg string) string {
	scrubbed := scrubErrorTokens(msg)
	if exposesEncodedPattern(scrubbed, scrubErrorTokens) || hasBackslashEscapedCredentialWith(msg, scrubbed, canonicalizeCredentialMarker) ||
		(scrubbed != msg && strings.Contains(msg, `\\u`) && hasBackslashEscapedCredentialWith(msg, scrubbed, func(s string) string {
			return canonicalizeCredentialMarkerWith(s, true)
		})) {
		return redactedSensitiveTextPlaceholder
	}
	for prev, pass := msg, 1; scrubbed != prev; pass++ {
		if pass == maxRedactionPasses {
			return redactedSensitiveTextPlaceholder
		}
		prev = scrubbed
		scrubbed = scrubErrorTokens(scrubbed)
	}
	return scrubbed
}

// keptLiterals splits scrubbed around redaction placeholders to obtain the
// non-redacted fragments that must still appear in order after unescaping.
func keptLiterals(scrubbed string) []string {
	parts := []string{scrubbed}
	for _, sentinel := range []string{redactedURIPlaceholder, url.QueryEscape(redactedURIPlaceholder), redactedValue, url.QueryEscape(redactedValue)} {
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

func keptInOrder(parts []string, s string) bool {
	for _, part := range parts {
		i := strings.Index(s, part)
		if i < 0 {
			return false
		}
		s = s[i+len(part):]
	}
	return true
}

func unescapeCredentialStep(s string, canonFn func(string) string) string {
	if strings.Contains(s, `\`) {
		s = unescapeBackslashControls(s)
	}
	if strings.ContainsAny(s, "%xXuU") {
		s = canonFn(s)
	}
	return s
}

// hasBackslashEscapedCredentialWith reports whether iteratively unescaping
// backslash controls and percent/\x/\u sequences (up to
// 2*maxCanonicalUnescapePasses passes, failing closed if escapes keep changing)
// exposes a credential missed by scrubErrorTokens, or breaks the in-order
// survival of scrubbed's non-redacted literals. Preserved from v0.1.0 for
// inputs such as `Bearer\tsecret`, `password=\"SECRET\\\ax monkey=ok tail\"`,
// and `{"s":"https\\ufeff://h/a;Bearer+SECRETZQ+token=x"}`.
func hasBackslashEscapedCredentialWith(raw, scrubbed string, canonFn func(string) string) bool {
	var parts []string
	curRaw, curScrubbed := raw, scrubbed
	for range 2 * maxCanonicalUnescapePasses {
		if !strings.ContainsAny(curRaw, `\%`) {
			return false
		}
		nextRaw := unescapeCredentialStep(curRaw, canonFn)
		if nextRaw == curRaw {
			return false
		}
		if curScrubbed == curRaw {
			if scrubErrorTokens(nextRaw) != nextRaw || exposesEncodedPattern(nextRaw, scrubErrorTokens) {
				return true
			}
			curRaw, curScrubbed = nextRaw, nextRaw
			continue
		}
		if parts == nil {
			parts = keptLiterals(curScrubbed)
		}
		nextScrubbed := unescapeCredentialStep(curScrubbed, canonFn)
		for j, part := range parts {
			parts[j] = unescapeCredentialStep(part, canonFn)
		}
		if scrubErrorTokens(nextScrubbed) != nextScrubbed || exposesEncodedPattern(nextScrubbed, scrubErrorTokens) || !keptInOrder(parts, scrubErrorTokens(nextRaw)) {
			return true
		}
		curRaw, curScrubbed = nextRaw, nextScrubbed
	}
	return unescapeCredentialStep(curRaw, canonFn) != curRaw
}

var backslashControlEscapes = [128]byte{'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t', '/': '/', '"': '"', '\'': '\''}

func unescapeBackslashControls(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		if s[i] != '\\' {
			n := strings.IndexByte(s[i:], '\\')
			if n < 0 {
				b.WriteString(s[i:])
				break
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
		if c := s[j]; c < 128 && backslashControlEscapes[c] != 0 {
			b.WriteByte(backslashControlEscapes[c])
		} else {
			b.WriteByte('\\')
			b.WriteByte(c)
		}
		i = j
	}
	return b.String()
}

func scrubErrorTokens(msg string) string { return scrubTextPatterns(scrubURITokens(msg)) }

func scrubTextPatterns(s string) string {
	folded, ok := foldForTextPatterns(s)
	if !ok {
		return s
	}
	if strings.Contains(s, ":") && (strings.Contains(folded, "author") || strings.Contains(folded, "apikey")) {
		s = authHeaderRe.ReplaceAllString(s, "${1}: "+redactedValue)
	}
	if strings.Contains(folded, "bearer") {
		s = replaceSinglePass(s, bearerTokenRe, redactBearerToken)
	}
	if strings.Contains(folded, "basic") {
		s = replaceSinglePass(s, basicTokenRe, redactBasicCredential)
	}
	if q := strings.IndexByte(s, '"'); q >= 0 && strings.IndexByte(s[q+1:], ':') >= 0 {
		s = redactJSONPatternPairs(s)
	}
	if strings.ContainsAny(s, ":=") {
		s = redactTextPatternPairs(s)
	}
	return s
}

func replaceSinglePass(s string, re *regexp.Regexp, repl func(s string, loc []int) string) string {
	matches := re.FindAllStringSubmatchIndex(s, -1)
	if len(matches) == 0 {
		return s
	}
	var b strings.Builder
	last := 0
	for _, loc := range matches {
		if loc[0] >= last {
			if out := repl(s, loc); out != s[loc[0]:loc[1]] {
				b.WriteString(s[last:loc[0]])
				b.WriteString(out)
				last = loc[1]
			}
		}
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

func redactBearerToken(s string, loc []int) string {
	if rest := s[loc[4]:]; len(rest) >= 2 && strings.EqualFold(rest[:2], "of") {
		if r, _ := utf8.DecodeRuneInString(rest[2:]); len(rest) == 2 || unicode.IsSpace(r) {
			return s[loc[0]:loc[5]]
		}
	}
	if inner, _ := unquoteValue(s[loc[4]:loc[5]]); isRedactedValue(inner) {
		return s[loc[0]:loc[5]]
	}
	return s[loc[2]:loc[3]] + " " + redactedValue
}

func redactBasicCredential(s string, loc []int) string {
	encoded := s[loc[4]:loc[5]]
	if loc[5] < len(s) && isBase64Byte(s[loc[5]]) {
		return s[loc[0]:loc[5]]
	}
	if len(encoded) <= maxBasicCredentialBytes {
		if decoded, err := base64.StdEncoding.DecodeString(encoded); err != nil || (!strings.Contains(encoded, "=") && !bytes.Contains(decoded, []byte(":"))) {
			return s[loc[0]:loc[5]]
		}
	}
	return s[loc[2]:loc[3]] + " " + redactedValue
}

func isBase64Byte(c byte) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || '0' <= c && c <= '9' || c == '+' || c == '/' || c == '='
}

func skipSpace(s string, i int, space string) int {
	for i < len(s) && strings.IndexByte(space, s[i]) >= 0 {
		i++
	}
	return i
}

func bareValueEnd(s string, start int, stop string, cache *int) int {
	if cache != nil && start < *cache {
		return *cache
	}
	end := start
	for end < len(s) && strings.IndexByte(stop, s[end]) < 0 {
		end++
	}
	if cache != nil {
		*cache = end
	}
	return end
}

// matchValueToken returns the end offset of a quoted or bare value at valStart,
// extending past a closing quote when glued to word bytes.
func matchValueToken(s string, valStart int, bareStop string) int {
	return matchValueTokenCached(s, valStart, bareStop, nil)
}

func matchValueTokenCached(s string, valStart int, bareStop string, bareCache *int) int {
	if valStart < len(s) && (s[valStart] == '"' || s[valStart] == '\'' || s[valStart] == '\\') {
		if q := quotedValueRe.FindString(s[valStart:]); q != "" {
			if quotedValueEnds(s[valStart:], len(q)) {
				return valStart + len(q)
			}
			return bareValueEnd(s, valStart+len(q), bareStop, bareCache)
		}
	}
	return bareValueEnd(s, valStart, bareStop, bareCache)
}

func redactJSONPatternPairs(s string) string {
	matches := jsonKeyPrefixRe.FindAllStringSubmatchIndex(s, -1)
	if len(matches) == 0 {
		return s
	}
	var b strings.Builder
	followedByAssign := func(pos int) bool {
		d := skipSpace(s, skipOneClosingQuote(s, pos), delimSpace)
		return d < len(s) && (s[d] == ':' || s[d] == '=')
	}
	last, coveredEnd := 0, 0
	nextQuoteOrBarrier := [4]int{-1, -1, -1, -1}
	nextTempQuote := [4]int{-1, -1, -1, -1}
	for i, loc := range matches {
		if loc[0] < coveredEnd {
			continue
		}
		trimmedKey := trimBOMAndSpace(s[loc[2]:loc[3]])
		if !isValidJSONPatternKey(trimmedKey) {
			continue
		}
		valStart := skipSpace(s, loc[1], regexpSpace)
		if valStart >= len(s) {
			continue
		}
		tokEnd := matchValueToken(s, valStart, regexpSpace+",}")
		valEnd := sensitiveValueEnd(s, valStart, tokEnd)
		if valEnd == valStart {
			continue
		}
		lineStart := strings.ContainsAny(s[loc[1]:valStart], "\r\n")
		isContainer := s[valStart] == '{' || (s[valStart] == '[' && !isRedactedPlaceholderPrefix(s[valStart:], false))
		cEnd := valStart
		if isContainer {
			cEnd = sensitiveValueEnd(s, valStart, valStart)
		}
		tailStart, tailIsContainer, tailCEnd, jStart := valStart, isContainer, cEnd, i+1
		if isContainer {
			tailStart = cEnd
		}
		nextIsValidKey := false
		if i+1 < len(matches) && matches[i+1][0] == valStart {
			if nextKey := trimBOMAndSpace(s[matches[i+1][2]:matches[i+1][3]]); isValidJSONPatternKey(nextKey) {
				nextIsValidKey = true
				if lineStart && !hasStackedTempColon(nextKey) && (!isBaseSensitiveKey(trimmedKey, false) || isBaseSensitiveKey(nextKey, false)) {
					continue
				}
				jStart = i + 2
				if nv := skipSpace(s, matches[i+1][1], regexpSpace); nv < len(s) {
					nvIsContainer := s[nv] == '{' || (s[nv] == '[' && !isRedactedPlaceholderPrefix(s[nv:], false))
					nvCEnd := nv
					if nvIsContainer {
						nvCEnd = sensitiveValueEnd(s, nv, nv)
					}
					valEnd = max(valEnd, sensitiveValueEnd(s, nv, matchValueToken(s, nv, regexpSpace+",}")))
					tailIsContainer, tailCEnd = nvIsContainer, nvCEnd
					tailStart = nv
					if nvIsContainer {
						tailStart = nvCEnd
					}
				} else {
					tailStart = valEnd
				}
			}
		}
		if tailStart < valEnd {
			if kind, _ := nestsSensitiveKeyInMatches(s, tailStart, valEnd, !nextIsValidKey && !tailIsContainer && lineStart && !isBaseSensitiveKey(trimmedKey, false), nil, 0); !nextIsValidKey && !tailIsContainer && kind == nestedKey {
				continue
			} else if kind != notNestedKey {
				valEnd = extendOverlappingTextPair(s, tailStart, valEnd, &nextQuoteOrBarrier, &nextTempQuote)
			} else if valEnd < len(s) {
				if !isCompleteQuotedValue(s[tailStart:valEnd]) || followedByAssign(valEnd) {
					valEnd = extendOverlappingTextPair(s, tailStart, valEnd, &nextQuoteOrBarrier, &nextTempQuote)
				}
			}
		}
		for j, bareEnd := jStart, 0; valEnd < len(s); {
			prevEnd, extStart := valEnd, -1
			for ; j < len(matches) && matches[j][0] < valEnd && valEnd < len(s); j++ {
				if tailIsContainer && matches[j][0] < tailCEnd {
					continue
				}
				jk := trimBOMAndSpace(s[matches[j][2]:matches[j][3]])
				if !isValidJSONPatternKey(jk) {
					if kind, _ := classifyNestedCandidate(s, s[matches[j][2]:matches[j][3]], matches[j][2], false, nil, 0); kind == notNestedKey {
						continue
					}
				}
				nv := skipSpace(s, matches[j][1], regexpSpace)
				if nv >= len(s) {
					continue
				}
				if s[nv] == '{' || s[nv] == '[' {
					ncEnd := sensitiveValueEnd(s, nv, nv)
					ntokEnd := bareValueEnd(s, nv, regexpSpace+",}", &bareEnd)
					if cand := max(ncEnd, ntokEnd); cand > valEnd {
						valEnd = cand
						if valEnd > ncEnd && extStart < 0 {
							extStart = ncEnd
						}
					}
					for j+1 < len(matches) && matches[j+1][0] < ncEnd {
						j++
					}
					continue
				}
				if s[nv] == '"' || s[nv] == '\'' || s[nv] == '\\' {
					if cand := matchValueTokenCached(s, nv, regexpSpace+",}", &bareEnd); cand > valEnd {
						valEnd = cand
						if extStart < 0 && (!isCompleteQuotedValue(s[nv:valEnd]) || followedByAssign(valEnd)) {
							extStart = nv
						}
					}
					continue
				}
				if nv < bareEnd {
					continue
				}
				if cand := bareValueEnd(s, nv, regexpSpace+",}", &bareEnd); cand > valEnd {
					valEnd = cand
					if extStart < 0 {
						extStart = nv
					}
				}
			}
			if valEnd <= prevEnd || valEnd >= len(s) || extStart < 0 || extStart >= valEnd {
				break
			}
			ext := extendOverlappingTextPair(s, extStart, valEnd, &nextQuoteOrBarrier, &nextTempQuote)
			if ext <= valEnd {
				break
			}
			valEnd = ext
		}
		if inner, q := unquoteValue(s[valStart:valEnd]); isRedactedValue(inner) {
			coveredEnd = valEnd
			continue
		} else if q != "" && !isBaseSensitiveKey(trimmedKey, false) && hasCrossFieldRunOnValue(inner) {
			return redactedSensitiveTextPlaceholder
		}
		b.WriteString(s[last:loc[0]])
		b.WriteString(`"` + trimmedKey + `":"` + redactedValue + `"`)
		last = valEnd
		coveredEnd = valEnd
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

type textKeyMatch struct {
	openStart, start, keyEnd, quoteEnd, end int
}

func prevKeyRuneSize(s string, pos, floor int) int {
	if pos <= floor {
		return 0
	}
	if c := s[pos-1]; c < utf8.RuneSelf {
		if isWordByte(c) || c == '.' || c == '-' {
			return 1
		}
		return 0
	}
	r, sz := utf8.DecodeLastRuneInString(s[floor:pos])
	if r == '\u0130' || r == '\u017f' || r == '\u212a' {
		return sz
	}
	return 0
}

func stripClosingKeyQuote(s string, quoteEnd, floor int) int {
	keyEnd := quoteEnd
	if keyEnd > floor && (s[keyEnd-1] == '"' || s[keyEnd-1] == '\'') {
		keyEnd--
		bs := 0
		for p := keyEnd; p > floor && s[p-1] == '\\'; p-- {
			bs++
		}
		if bs%2 == 1 {
			keyEnd--
		}
	}
	return keyEnd
}

func hasMatchingOpeningQuote(s string, pos, floor int, q string) bool {
	if len(q) == 2 {
		return pos >= floor+2 && s[pos-2:pos] == q
	}
	if pos <= floor || s[pos-1] != q[0] {
		return false
	}
	bs := 0
	for i := pos - 1; i > floor && s[i-1] == '\\'; i-- {
		bs++
	}
	return bs%2 == 0
}

func prevAnySpaceSize(s string, pos, floor int) int {
	if pos <= floor {
		return 0
	}
	if c := s[pos-1]; c < utf8.RuneSelf {
		if c == ' ' || c == '\t' || c == '\v' || c == '\f' || c == '\r' || c == '\n' {
			return 1
		}
		return 0
	}
	r, sz := utf8.DecodeLastRuneInString(s[floor:pos])
	if r == '\ufeff' || unicode.IsSpace(r) {
		return sz
	}
	return 0
}

func isUnescapedQuoteAt(s string, floor, pos int) bool {
	bs := 0
	for i := pos; i > floor && s[i-1] == '\\'; i-- {
		bs++
	}
	return bs%2 == 0
}

func hasAnySpace(s string) bool {
	for i := 0; i < len(s); {
		if c := s[i]; c < utf8.RuneSelf {
			if c <= 0x20 || c == 0x7f {
				return true
			}
			i++
			continue
		}
		r, sz := utf8.DecodeRuneInString(s[i:])
		if r == '\ufeff' || unicode.IsSpace(r) {
			return true
		}
		i += sz
	}
	return false
}

func validQuotedTempKey(s string) bool {
	if !hasFoldPrefix(s, session.KeyPrefixTemp) || len(s) <= len(session.KeyPrefixTemp) {
		return false
	}
	rest := s[len(session.KeyPrefixTemp):]
	return !strings.ContainsAny(rest, "=\"' \t\r\n\f,}") && !strings.HasSuffix(rest, ":") && !strings.Contains(rest, "::")
}

func isCompleteQuotedValue(v string) bool {
	return v != "" && (v[0] == '"' || v[0] == '\'' || v[0] == '\\') && len(quotedValueRe.FindString(v)) == len(v)
}

func skipOneClosingQuote(s string, pos int) int {
	if pos < len(s) && (s[pos] == '\'' || s[pos] == '"') {
		return pos + 1
	}
	if pos+1 < len(s) && s[pos] == '\\' && (s[pos+1] == '\'' || s[pos+1] == '"') {
		return pos + 2
	}
	return pos
}

func quotedAssignSpan(s string, start, end int) (qEnd, afterQuote int, ok bool) {
	if start >= end || (s[start] != '"' && s[start] != '\'' && s[start] != '\\') {
		return 0, 0, false
	}
	q := quotedValueRe.FindString(s[start:end])
	if q == "" {
		return 0, 0, false
	}
	qEnd = start + len(q)
	if qEnd == end {
		return qEnd, skipOneClosingQuote(s, end), true
	}
	if skipOneClosingQuote(s, qEnd) == end {
		return qEnd, end, true
	}
	return 0, 0, false
}

func matchQuotedKeySpan(s string, quoteStart, quoteEnd, sensFloor int, noQuoteBefore *int) (openStart, keyStart, keyEnd int, ok bool) {
	q := s[quoteStart:quoteEnd]
	if len(q) == 2 && !isUnescapedQuoteAt(s, sensFloor, quoteStart) {
		return 0, 0, 0, false
	}
	limit := max(sensFloor, *noQuoteBefore)
	openIdx := -1
	if len(q) == 2 {
		for i := quoteStart - 2; i >= limit; i-- {
			c := s[i]
			if c == ',' || c == '{' || c == '}' {
				break
			}
			if c == '\\' && s[i+1] == q[1] && isUnescapedQuoteAt(s, sensFloor, i) {
				openIdx = i
				break
			}
		}
	} else {
		for i := quoteStart - 1; i >= limit; i-- {
			c := s[i]
			if c == ',' || c == '{' || c == '}' {
				break
			}
			if c == q[0] {
				if isUnescapedQuoteAt(s, sensFloor, i) {
					openIdx = i
				}
				break
			}
		}
	}
	*noQuoteBefore = max(*noQuoteBefore, quoteStart)
	if openIdx < 0 {
		return 0, 0, 0, false
	}
	kFrom, kTo := openIdx+len(q), quoteStart
	for kFrom < kTo {
		if c := s[kFrom]; c < utf8.RuneSelf {
			if c == ' ' || c == '\t' || c == '\v' || c == '\f' || c == '\r' || c == '\n' {
				kFrom++
				continue
			}
			break
		}
		r, sz := utf8.DecodeRuneInString(s[kFrom:kTo])
		if r == '\ufeff' || unicode.IsSpace(r) {
			kFrom += sz
			continue
		}
		break
	}
	for kTo > kFrom {
		sz := prevAnySpaceSize(s, kTo, kFrom)
		if sz == 0 {
			break
		}
		kTo -= sz
	}
	if kFrom >= kTo {
		return 0, 0, 0, false
	}
	sub := s[kFrom:kTo]
	if stripped, isT := stripStatePrefixes(sub); validQuotedTempKey(sub) || (isT && validQuotedTempKey(stripped)) {
		return openIdx, kFrom, kTo, true
	}
	for i := range len(sub) {
		if sub[i] == ':' && i >= 4 && strings.EqualFold(sub[i-4:i], "temp") &&
			(i == 4 || sub[i-5] == '.' || sub[i-5] == '-' || sub[i-5] == '/' || sub[i-5] == '\\') &&
			!hasAnySpace(sub[:i-4]) && validQuotedTempKey(sub[i-4:]) {
			return openIdx, kFrom + i - 4, kTo, true
		}
	}
	if hasAnySpace(sub) {
		return 0, 0, 0, false
	}
	if ks := sensitiveKeyStart(s, kFrom, kTo); ks >= 0 {
		return openIdx, ks, kTo, true
	}
	if slash := strings.LastIndexAny(sub, `/\`); slash >= 0 && slash+1 < len(sub) {
		if ks := sensitiveKeyStart(s, kFrom+slash+1, kTo); ks >= 0 {
			return openIdx, ks, kTo, true
		}
	}
	return 0, 0, 0, false
}

func quoteTypeIdx(v string) int {
	if len(v) >= 2 && v[0] == '\\' {
		switch v[1] {
		case '\'':
			return 2
		case '"':
			return 3
		}
	}
	if len(v) >= 1 {
		switch v[0] {
		case '\'':
			return 0
		case '"':
			return 1
		}
	}
	return -1
}

func quotedValueEndsInSensitiveKey(s string, start, end int) bool {
	_, q := unquoteValue(s[start:end])
	if q == "" {
		return false
	}
	localFloor := start
	if _, _, _, ok := matchQuotedKeySpan(s, end-len(q), end, start, &localFloor); ok {
		return true
	}
	kTo := end - len(q)
	kFrom := min(skipAnySpace(s, start+len(q)), kTo)
	for kTo > kFrom {
		sz := prevAnySpaceSize(s, kTo, kFrom)
		if sz == 0 {
			break
		}
		kTo -= sz
	}
	if kFrom >= kTo {
		return false
	}
	segStart := kTo
	for segStart > kFrom {
		if sz := prevAnySpaceSize(s, segStart, kFrom); sz > 0 {
			break
		}
		if c := s[segStart-1]; c < utf8.RuneSelf {
			if c == '/' || c == '\\' {
				break
			}
			segStart--
			continue
		}
		_, sz := utf8.DecodeLastRuneInString(s[kFrom:segStart])
		segStart -= sz
	}
	return segStart < kTo && (sensitiveKeyStart(s, segStart, kTo) >= 0 || validQuotedTempKey(s[segStart:kTo]))
}

func skipAnySpace(s string, pos int) int {
	for pos < len(s) {
		if c := s[pos]; c < utf8.RuneSelf {
			if c == ' ' || c == '\t' || c == '\v' || c == '\f' || c == '\r' || c == '\n' {
				pos++
				continue
			}
			break
		}
		r, sz := utf8.DecodeRuneInString(s[pos:])
		if r == '\ufeff' || unicode.IsSpace(r) {
			pos += sz
			continue
		}
		break
	}
	return pos
}

func isTempSegByte(c byte) bool {
	return strings.IndexByte("=:\"' \t\r\n\f,}", c) < 0
}

// textKeyPrefixSpans finds sensitive bare key=value and key: value prefixes by
// jumping straight to "=" and ":" delimiters via strings.IndexAny, walking back
// over the key name, and filtering with sensitiveKeyStart before appending.
func textKeyPrefixSpans(s string) []textKeyMatch {
	var out []textKeyMatch
	var quoteFloor [4]int
	failedTempUntil := 0
	for from, floor, sensFloor := 0, 0, 0; from < len(s); {
		rel := strings.IndexAny(s[from:], "=:")
		if rel < 0 {
			break
		}
		delim := from + rel
		quoteEnd := delim
		for quoteEnd > floor && strings.IndexByte(delimSpace, s[quoteEnd-1]) >= 0 {
			quoteEnd--
		}
		keyEnd := stripClosingKeyQuote(s, quoteEnd, floor)
		quoteStart := keyEnd
		padded := false
		if keyEnd < quoteEnd {
			for sz := prevAnySpaceSize(s, keyEnd, floor); sz > 0; sz = prevAnySpaceSize(s, keyEnd, floor) {
				keyEnd -= sz
				padded = true
			}
		}
		start := keyEnd
		for start > floor {
			sz := prevKeyRuneSize(s, start, floor)
			if sz == 0 {
				break
			}
			start -= sz
		}
		var savedQuoteFloor [4]int
		if quoteStart < quoteEnd {
			savedQuoteFloor = quoteFloor
		}
		if keyEnd < quoteEnd && (padded || start == keyEnd || (start > floor && (s[start-1] == '/' || s[start-1] == '\\' || s[start-1] == '\v'))) {
			qIdx := 0
			if s[quoteEnd-1] == '"' {
				qIdx = 1
			}
			if quoteEnd-quoteStart == 2 {
				qIdx += 2
			}
			if os, ks, ke, ok := matchQuotedKeySpan(s, quoteStart, quoteEnd, sensFloor, &quoteFloor[qIdx]); ok {
				out = append(out, textKeyMatch{os, ks, ke, quoteEnd, delim + 1})
				sensFloor = delim + 1
				from = delim + 1
				floor = delim + 1
				continue
			}
		}
		if quoteStart < quoteEnd && quoteStart > floor && (s[quoteStart-1] == '\'' || s[quoteStart-1] == '"') &&
			!(quoteStart-1 > floor && s[quoteStart-2] == '\\') {
			q2End := quoteStart
			q2Start := q2End - 1
			q2KeyEnd := q2Start
			q2Padded := false
			for sz := prevAnySpaceSize(s, q2KeyEnd, floor); sz > 0; sz = prevAnySpaceSize(s, q2KeyEnd, floor) {
				q2KeyEnd -= sz
				q2Padded = true
			}
			q2From := q2KeyEnd
			for q2From > floor {
				sz := prevKeyRuneSize(s, q2From, floor)
				if sz == 0 {
					break
				}
				q2From -= sz
			}
			if q2Padded || q2From == q2KeyEnd || (q2From > floor && (s[q2From-1] == '/' || s[q2From-1] == '\\' || s[q2From-1] == '\v')) ||
				(q2From == floor && floor > sensFloor && s[floor-1] == ':') {
				q2Idx := 0
				if s[q2Start] == '"' {
					q2Idx = 1
				}
				retryFloor := savedQuoteFloor[q2Idx]
				os, ks, ke, ok := matchQuotedKeySpan(s, q2Start, q2End, sensFloor, &retryFloor)
				quoteFloor[q2Idx] = max(quoteFloor[q2Idx], retryFloor)
				if ok {
					out = append(out, textKeyMatch{os, ks, ke, q2End, delim + 1})
					sensFloor = delim + 1
					from = delim + 1
					floor = delim + 1
					continue
				}
			} else if q2From < q2KeyEnd {
				if ks := sensitiveKeyStart(s, q2From, q2KeyEnd); ks >= 0 {
					openAfter := q2From
					for sz := prevAnySpaceSize(s, openAfter, floor); sz > 0; sz = prevAnySpaceSize(s, openAfter, floor) {
						openAfter -= sz
					}
					if q := s[q2KeyEnd:q2End]; hasMatchingOpeningQuote(s, openAfter, floor, q) {
						openStart := openAfter - len(q)
						out = append(out, textKeyMatch{openStart, ks, q2KeyEnd, q2End, delim + 1})
						sensFloor = delim + 1
						from = delim + 1
						floor = delim + 1
						continue
					}
				}
			}
		}
		if keyEnd < quoteEnd && (padded || start == keyEnd) {
			from = delim + 1
			floor = delim + 1
			continue
		}
		if start == keyEnd {
			from = delim + 1
			floor = delim + 1
			continue
		}
		if s[delim] == ':' && quoteEnd == delim && keyEnd == delim &&
			keyEnd-start >= 4 && strings.EqualFold(s[keyEnd-4:keyEnd], "temp") &&
			(keyEnd-4 == start || s[keyEnd-5] == '.' || s[keyEnd-5] == '-' || s[keyEnd-5] == ':') {
			tempStart := keyEnd - 4
			nextClosingQuoteEnd := func(pos int) (int, int, bool) {
				p := skipAnySpace(s, pos)
				qStart := p
				if p < len(s) && (s[p] == '"' || s[p] == '\'') {
					p++
				} else if p+1 < len(s) && s[p] == '\\' && (s[p+1] == '"' || s[p+1] == '\'') {
					p += 2
				} else {
					return pos, tempStart, true
				}
				openIdx := tempStart
				openStart := start
				openFloor := floor
				walkFloor := max(sensFloor, start-maxNestedKeyLen)
				for openStart > walkFloor {
					c := s[openStart-1]
					if c < utf8.RuneSelf {
						if c == '"' || c == '\'' || c == '=' || c == ',' || c == '{' || c == '}' ||
							c == ' ' || c == '\t' || c == '\v' || c == '\f' || c == '\r' || c == '\n' {
							break
						}
						openStart--
						continue
					}
					r, sz := utf8.DecodeLastRuneInString(s[walkFloor:openStart])
					if r == '\ufeff' || unicode.IsSpace(r) {
						break
					}
					openStart -= sz
				}
				if openStart < start {
					openFloor = sensFloor
				}
				openAfter := openStart
				for sz := prevAnySpaceSize(s, openAfter, openFloor); sz > 0; sz = prevAnySpaceSize(s, openAfter, openFloor) {
					openAfter -= sz
				}
				if hasMatchingOpeningQuote(s, openAfter, openFloor, s[qStart:p]) {
					openIdx = openAfter - (p - qStart)
				} else if qStart > pos {
					return pos, tempStart, false
				}
				return p, openIdx, true
			}
			kEnd := delim + 1
			for kEnd < len(s) && isTempSegByte(s[kEnd]) {
				kEnd++
			}
			if kEnd > delim+1 {
				matchedQEnd, matchedOpen := 0, tempStart
				if kEnd >= failedTempUntil {
					look := kEnd
					for look < len(s) && s[look] == ':' {
						next := look + 1
						for next < len(s) && isTempSegByte(s[next]) {
							next++
						}
						if next == look+1 {
							break
						}
						nq, nOpen, ok := nextClosingQuoteEnd(next)
						if !ok {
							break
						}
						nd := skipSpace(s, nq, delimSpace)
						if nd >= len(s) {
							look = len(s)
							break
						}
						if nq > next && nOpen < tempStart && nd == nq && (s[nd] == '\'' || s[nd] == '"') {
							if nd2 := skipSpace(s, nd+1, delimSpace); nd2 < len(s) && (s[nd2] == '=' || s[nd2] == ':') {
								nd = nd2
							}
						}
						if nq > next && (s[nd] == '=' || s[nd] == ':') {
							kEnd = next
							matchedQEnd, matchedOpen = nq, nOpen
							break
						}
						if s[nd] != ':' || (nd+2 < len(s) && s[nd+1:nd+3] == "//") {
							break
						}
						look = next
					}
					if matchedQEnd == 0 {
						failedTempUntil = max(failedTempUntil, look)
					}
				}
				qEnd, openStart := matchedQEnd, matchedOpen
				if qEnd == 0 {
					var ok bool
					if qEnd, openStart, ok = nextClosingQuoteEnd(kEnd); !ok {
						qEnd, openStart = kEnd, tempStart
					}
				}
				d := skipSpace(s, qEnd, delimSpace)
				if qEnd > kEnd && openStart < tempStart && d == qEnd && d < len(s) && (s[d] == '\'' || s[d] == '"') {
					d = skipSpace(s, d+1, delimSpace)
				}
				if d < len(s) && (s[d] == '=' || s[d] == ':') {
					out = append(out, textKeyMatch{openStart, tempStart, kEnd, qEnd, d + 1})
					from = d + 1
					floor = d + 1
					sensFloor = d + 1
					continue
				}
			}
		}
		if ks := sensitiveKeyStart(s, start, keyEnd); ks >= 0 {
			openStart := ks
			if keyEnd < quoteEnd {
				openAfter := start
				for sz := prevAnySpaceSize(s, openAfter, floor); sz > 0; sz = prevAnySpaceSize(s, openAfter, floor) {
					openAfter -= sz
				}
				if q := s[keyEnd:quoteEnd]; hasMatchingOpeningQuote(s, openAfter, floor, q) {
					openStart = openAfter - len(q)
				}
			}
			out = append(out, textKeyMatch{openStart, ks, keyEnd, quoteEnd, delim + 1})
			sensFloor = delim + 1
		}
		from = delim + 1
		floor = delim + 1
	}
	return out
}

func extendOverlappingTextPair(s string, valStart, valEnd int, nextQuoteOrBarrier, nextTempQuote *[4]int) int {
	if nextQuoteOrBarrier == nil {
		local := [4]int{-1, -1, -1, -1}
		nextQuoteOrBarrier = &local
	}
	if nextTempQuote == nil {
		local := [4]int{-1, -1, -1, -1}
		nextTempQuote = &local
	}
	windowEndFor := func(cur, pos int) int {
		w := min(len(s), pos+maxNestedKeyLen*2)
		qIdx, qPos, qLen, qByte := -1, -1, 0, byte(0)
		for i := pos - 1; i >= cur; i-- {
			c := s[i]
			if c == ',' || c == '{' || c == '}' {
				break
			}
			if (c == '\'' || c == '"') && isUnescapedQuoteAt(s, cur, i) {
				if c == '\'' {
					qIdx = 0
				} else {
					qIdx = 1
				}
				qPos, qLen, qByte = i, 1, c
				break
			}
			if i > cur && s[i-1] == '\\' && (c == '\'' || c == '"') && isUnescapedQuoteAt(s, cur, i-1) {
				if c == '\'' {
					qIdx = 2
				} else {
					qIdx = 3
				}
				qPos, qLen, qByte = i-1, 2, c
				break
			}
		}
		if qPos >= 0 && !strings.ContainsAny(s[pos:w], ",{}") {
			if nextQuoteOrBarrier[qIdx] < w {
				k := w
				if qLen == 2 && k > pos && k < len(s) && s[k-1] == '\\' && s[k] == qByte && isUnescapedQuoteAt(s, 0, k-1) {
					k--
				}
				for k < len(s) {
					c := s[k]
					if c == ',' || c == '{' || c == '}' {
						break
					}
					if qLen == 1 {
						if c == qByte && isUnescapedQuoteAt(s, 0, k) {
							break
						}
					} else if c == '\\' && k+1 < len(s) && s[k+1] == qByte && isUnescapedQuoteAt(s, 0, k) {
						break
					}
					k++
				}
				nextQuoteOrBarrier[qIdx] = k
			}
			if k := nextQuoteOrBarrier[qIdx]; k < len(s) && s[k] != ',' && s[k] != '{' && s[k] != '}' {
				w = k + qLen
				if w < len(s) && (s[w] == '\'' || s[w] == '"') {
					w++
				} else if w+1 < len(s) && s[w] == '\\' && (s[w+1] == '\'' || s[w+1] == '"') {
					w += 2
				}
			}
		}
		w = skipAnySpace(s, w)
		for range 2 {
			if w < len(s) && (s[w] == '"' || s[w] == '\'') {
				w++
			} else if w+1 < len(s) && s[w] == '\\' && (s[w+1] == '"' || s[w+1] == '\'') {
				w += 2
			} else {
				break
			}
		}
		return min(len(s), skipSpace(s, w, delimSpace)+1)
	}
	bareCache, tempCache := 0, 0
	var quotedAssignEnd [4]int
	extendQuotedAssign := func(start, end int) int {
		for {
			qEnd, afterQuote, ok := quotedAssignSpan(s, start, end)
			if !ok {
				break
			}
			qIdx := quoteTypeIdx(s[start:qEnd])
			if qIdx < 0 || qEnd <= quotedAssignEnd[qIdx] {
				break
			}
			quotedAssignEnd[qIdx] = qEnd
			d := skipSpace(s, afterQuote, delimSpace)
			if d >= len(s) || (s[d] != ':' && s[d] != '=') || !quotedValueEndsInSensitiveKey(s, start, qEnd) {
				break
			}
			nv2 := skipSpace(s, d+1, delimSpace)
			if nv2 >= len(s) {
				return nv2
			}
			var next int
			if s[nv2] == '{' || (s[nv2] == '[' && !isRedactedPlaceholderPrefix(s[nv2:], false)) {
				next = max(end, sensitiveValueEnd(s, nv2, nv2), matchValueTokenCached(s, nv2, regexpSpace+",;&", &bareCache))
			} else if valueStartsAt(s, nv2) {
				next = max(end, matchValueTokenCached(s, nv2, regexpSpace+",;&", &bareCache))
			} else {
				break
			}
			if next <= end {
				break
			}
			start, end = nv2, next
		}
		return end
	}
	enclosingQuoteStart, enclosingQuoteEnd := -1, -1
	if qEnd, _, ok := quotedAssignSpan(s, valStart, valEnd); ok {
		enclosingQuoteStart, enclosingQuoteEnd = valStart, qEnd
	}
	valEnd = extendQuotedAssign(valStart, valEnd)
	base := valStart
	windowEnd := windowEndFor(valStart, valEnd)
	spans := textKeyPrefixSpans(s[base:windowEnd])
	for idx, cur := 0, valStart; cur < valEnd && valEnd < len(s); {
		if enclosingQuoteStart >= 0 && cur >= enclosingQuoteEnd {
			enclosingQuoteStart, enclosingQuoteEnd = -1, -1
		}
		for idx < len(spans) && base+spans[idx].openStart < cur {
			idx++
		}
		if idx >= len(spans) {
			nextWin := windowEndFor(cur, valEnd)
			if nextWin <= windowEnd {
				break
			}
			base = cur
			if enclosingQuoteStart >= 0 && cur < enclosingQuoteEnd {
				base = min(cur, enclosingQuoteStart)
			}
			windowEnd = nextWin
			spans = textKeyPrefixSpans(s[base:windowEnd])
			idx = 0
			continue
		}
		sp := spans[idx]
		idx++
		if base+sp.openStart >= valEnd {
			break
		}
		nv := skipSpace(s, base+sp.end, delimSpace)
		if nv >= len(s) {
			return nv
		}
		isContainer := s[nv] == '{' || (s[nv] == '[' && !isRedactedPlaceholderPrefix(s[nv:], false))
		isBareTemp := sp.keyEnd == sp.quoteEnd && s[base+sp.end-1] == ':' && hasFoldPrefix(s[base+sp.start:base+sp.keyEnd], session.KeyPrefixTemp)
		tokEnd := matchValueTokenCached(s, nv, regexpSpace+",;&", &bareCache)
		nextEnd := tokEnd
		if isContainer {
			cEnd := sensitiveValueEnd(s, nv, nv)
			nextEnd = max(nextEnd, cEnd)
			if isBareTemp && nextEnd < len(s) {
				if stackedEnd := stackedBareTempTailEnd(s, nv, 0, &tempCache, nextTempQuote); stackedEnd > nextEnd {
					nextEnd = stackedEnd
				}
			}
			cur = max(base+sp.end, cEnd)
		} else if !valueStartsAt(s, nv) {
			cur = nv
			continue
		} else {
			if qEnd, _, ok := quotedAssignSpan(s, nv, tokEnd); ok {
				if qEnd > enclosingQuoteEnd {
					enclosingQuoteStart, enclosingQuoteEnd = nv, qEnd
				}
				nextEnd = max(nextEnd, extendQuotedAssign(nv, tokEnd))
			}
			if isBareTemp && nextEnd < len(s) {
				if stackedEnd := stackedBareTempTailEnd(s, nv, 0, &tempCache, nextTempQuote); stackedEnd > nextEnd {
					nextEnd = stackedEnd
				}
			}
			cur = nv
		}
		valEnd = max(valEnd, nextEnd)
	}
	return valEnd
}

// isPathPrefixedDottedKey skips bare Go *os.PathError filenames such as
// "open /etc/ssl/private.key: no such file" while still redacting assignments.
func isPathPrefixedDottedKey(s string, floor, keyStart, keyEnd int, delim byte, afterDelim int) bool {
	k := s[keyStart:keyEnd]
	if delim != ':' || hasFoldPrefix(k, session.KeyPrefixTemp) {
		return false
	}
	if dot := strings.LastIndexByte(k, '.'); dot <= 0 || !strings.EqualFold(k[dot+1:], "key") {
		return false
	}
	if afterDelim < len(s) && (strings.IndexByte(`"'{[`, s[afterDelim]) >= 0 ||
		(s[afterDelim] == '\\' && afterDelim+1 < len(s) && (s[afterDelim+1] == '"' || s[afterDelim+1] == '\''))) {
		return false
	}
	if keyStart > 0 && (s[keyStart-1] == '/' || s[keyStart-1] == '\\') {
		return !(s[keyStart-1] == '/' && keyStart >= 3 && s[keyStart-3:keyStart] == "://")
	}
	wEnd := keyStart
	for wEnd > floor && (s[wEnd-1] == ' ' || s[wEnd-1] == '\t') {
		wEnd--
	}
	wStart := wEnd
	for wStart > floor && ('a' <= s[wStart-1] && s[wStart-1] <= 'z' || 'A' <= s[wStart-1] && s[wStart-1] <= 'Z') {
		wStart--
	}
	if wEnd == keyStart || wStart == wEnd || (wStart > floor && (isWordByte(s[wStart-1]) || strings.IndexByte(`.-/\`, s[wStart-1]) >= 0)) {
		return false
	}
	switch strings.ToLower(s[wStart:wEnd]) {
	case "open", "openat", "create", "stat", "lstat", "fstat", "read", "write", "readfile",
		"writefile", "readlink", "remove", "removeall", "rename", "chmod", "chown", "lchown",
		"chtimes", "truncate", "mkdir", "mkdirall", "rmdir", "chdir", "readdir", "symlink", "link", "load", "parse":
		return true
	}
	return false
}

func stackedBareTempTailEnd(s string, valStart, open int, tempCache *int, nextQuote *[4]int) int {
	if valStart >= len(s) || (tempCache != nil && valStart < *tempCache) {
		return 0
	}
	if nextQuote == nil {
		local := [4]int{-1, -1, -1, -1}
		nextQuote = &local
	}
	cur, best, bestRaw, bareCache, containerCache, lastContainerStart, lastContainerEnd := valStart, 0, 0, 0, 0, -1, 0
	for cur < len(s) {
		j := cur
		if s[cur] == '{' || (s[cur] == '[' && !isRedactedPlaceholderPrefix(s[cur:], false)) {
			if cur == lastContainerStart {
				j = lastContainerEnd
			} else if cur >= containerCache {
				j = sensitiveValueEnd(s, cur, cur)
				lastContainerStart, lastContainerEnd = cur, j
				containerCache = max(containerCache, j)
			}
		} else {
			for j < len(s) {
				if s[j] == '\\' && j+1 < len(s) && (s[j+1] == '\'' || s[j+1] == '"') && isUnescapedQuoteAt(s, cur, j) {
					qi, qByte := 2, s[j+1]
					if qByte == '"' {
						qi = 3
					}
					if nextQuote[qi] <= j {
						k := j + 2
						for k < len(s) {
							if s[k] == '\\' && k+1 < len(s) {
								if s[k+1] == qByte {
									break
								}
								if s[k+1] == '\\' {
									k += 2
									if k < len(s) && s[k] == '\\' {
										k += 2
									} else if k < len(s) && s[k] != qByte {
										k++
									}
									continue
								}
								k += 2
								continue
							}
							k++
						}
						nextQuote[qi] = min(k, len(s))
					}
					if nextQuote[qi] < len(s) {
						break
					}
					j += 2
					continue
				}
				if s[j] == '\'' || s[j] == '"' {
					qi := 0
					if s[j] == '"' {
						qi = 1
					}
					if nextQuote[qi] <= j {
						k := j + 1
						for k < len(s) {
							if s[k] == '\\' {
								k += 2
								continue
							}
							if s[k] == s[j] {
								break
							}
							k++
						}
						nextQuote[qi] = min(k, len(s))
					}
					if nextQuote[qi] < len(s) {
						break
					}
					j++
					continue
				}
				if !isTempSegByte(s[j]) || strings.IndexByte(regexpSpace+",;&", s[j]) >= 0 {
					break
				}
				j++
			}
		}
		if j == cur {
			break
		}
		spaceEnd := skipAnySpace(s, j)
		afterSeg := spaceEnd
		if afterSeg+1 < len(s) && s[afterSeg] == '\\' && (s[afterSeg+1] == '"' || s[afterSeg+1] == '\'') {
			afterSeg += 2
		} else if afterSeg < len(s) && (s[afterSeg] == '"' || s[afterSeg] == '\'') {
			afterSeg++
		}
		dPos := skipSpace(s, j, " \t")
		if afterSeg > spaceEnd {
			dPos = skipSpace(s, afterSeg, delimSpace)
		}
		if dPos < len(s) && (s[dPos] == ':' || (s[dPos] == '=' && !(dPos+1 < len(s) && s[dPos+1] == '='))) {
			nv := skipSpace(s, dPos+1, delimSpace)
			if nv < len(s) && valueStartsAt(s, nv) && (afterSeg == spaceEnd || !isRedactedPlaceholderPrefix(s[dPos+1:], false)) {
				if s[nv] == '{' || (s[nv] == '[' && !isRedactedPlaceholderPrefix(s[nv:], false)) {
					if nv >= containerCache {
						vEnd := matchValueTokenCached(s, nv, regexpSpace+",;&", &bareCache)
						cEnd := sensitiveValueEnd(s, nv, nv)
						lastContainerStart, lastContainerEnd = nv, cEnd
						containerCache = max(containerCache, cEnd)
						best = max(best, cEnd)
						if vEnd > cEnd && vEnd > bestRaw {
							bestRaw = vEnd
							best = max(best, cEnd+len(trimEnclosingClosers(s[cEnd:vEnd], open)))
						}
					}
				} else {
					vEnd := matchValueTokenCached(s, nv, regexpSpace+",;&", &bareCache)
					if vEnd > bestRaw {
						bestRaw = vEnd
						best = max(best, textValueEnd(s, nv, vEnd, open))
					}
				}
				if best >= len(s) {
					if tempCache != nil {
						*tempCache = len(s)
					}
					return len(s)
				}
			}
		}
		if j < len(s) && s[j] == ':' {
			cur = j + 1
			continue
		}
		break
	}
	if tempCache != nil {
		*tempCache = max(*tempCache, cur)
	}
	return best
}

func redactTextPatternPairs(s string) string {
	matches := textKeyPrefixSpans(s)
	if len(matches) == 0 {
		return s
	}
	var b strings.Builder
	var brackets openBrackets
	last, coveredEnd := 0, 0
	nextTempQuote := [4]int{-1, -1, -1, -1}
	for i, loc := range matches {
		keyStart := loc.start
		if keyStart < coveredEnd {
			continue
		}
		valStart := skipSpace(s, loc.end, delimSpace)
		if valStart >= len(s) || (loc.keyEnd == loc.quoteEnd && isPathPrefixedDottedKey(s, last, keyStart, loc.keyEnd, s[loc.end-1], loc.end)) {
			continue
		}
		valEnd := matchValueToken(s, valStart, regexpSpace+",;&")
		if valEnd == valStart {
			continue
		}
		open := brackets.before(s, keyStart)
		kind, pairIdx := nestsSensitiveKeyInMatches(s, valStart, valEnd, strings.ContainsAny(s[loc.end:valStart], "\r\n"), matches, i+1)
		if kind == nestedKey {
			continue
		}
		var end, containerEnd int
		jStart := i + 1
		bareCache, checkedBareEnd, tempCache := 0, 0, 0
		_, valQuote := unquoteValue(s[valStart:valEnd])
		gluedQuoted := false
		if kind == valueEndingInKeyName {
			var lastIdx int
			end, lastIdx, containerEnd = nestedPairEndInMatches(s, pairIdx, valEnd, open, matches, nil, nil, &tempCache, &nextTempQuote)
			jStart = max(jStart, lastIdx+1)
		} else {
			if s[valStart] == '{' || (s[valStart] == '[' && !isRedactedPlaceholderPrefix(s[valStart:], false)) {
				containerEnd = sensitiveValueEnd(s, valStart, valStart)
				end = containerEnd + len(trimEnclosingClosers(s[containerEnd:max(valEnd, containerEnd)], open))
			} else {
				end = textValueEnd(s, valStart, valEnd, open)
			}
			if kind == notNestedKey && valQuote != "" && end < len(s) && (s[end] == '=' || s[end] == ':') &&
				end+1 < len(s) && strings.IndexByte(regexpSpace+",;&}])", s[end+1]) < 0 {
				gv := end + 1
				if s[gv] == '{' || (s[gv] == '[' && !isRedactedPlaceholderPrefix(s[gv:], false)) {
					cEnd := sensitiveValueEnd(s, gv, gv)
					containerEnd = max(containerEnd, cEnd)
					tokEnd := bareValueEnd(s, gv, regexpSpace+",;&", &bareCache)
					end = max(cEnd, cEnd+len(trimEnclosingClosers(s[cEnd:max(tokEnd, cEnd)], open)))
				} else {
					vEnd := matchValueTokenCached(s, gv, regexpSpace+",;&", &bareCache)
					end = textValueEnd(s, gv, vEnd, open)
				}
				gluedQuoted = true
			}
			if loc.keyEnd == loc.quoteEnd && s[loc.end-1] == ':' && hasFoldPrefix(s[keyStart:loc.keyEnd], session.KeyPrefixTemp) && end < len(s) {
				if stackedEnd := stackedBareTempTailEnd(s, valStart, open, &tempCache, &nextTempQuote); stackedEnd > end {
					end = stackedEnd
				}
			}
		}
		for j := jStart; j < len(matches) && matches[j].openStart < end && end < len(s); j++ {
			if matches[j].start < containerEnd || !valueStartsAt(s, matches[j].end) {
				continue
			}
			// Preserve bare tokens immediately preceding an escaped-quoted key
			// (such as `token=abc\"password\": SECRET123`) so scrubErrorText
			// catches the escaped quote boundary fail-closed.
			if valQuote == "" && matches[j].start > valStart &&
				(s[matches[j].start-1] == '"' || s[matches[j].start-1] == '\'') {
				bs := 0
				for p := matches[j].start - 1; p > valStart && s[p-1] == '\\'; p-- {
					bs++
				}
				if bs%2 == 1 {
					continue
				}
			}
			isBareTemp := matches[j].keyEnd == matches[j].quoteEnd && s[matches[j].end-1] == ':' &&
				hasFoldPrefix(s[matches[j].start:matches[j].keyEnd], session.KeyPrefixTemp)
			if nv := skipSpace(s, matches[j].end, delimSpace); nv < checkedBareEnd && checkedBareEnd-nv > maxNestedKeyLen &&
				(!isBareTemp || nv < tempCache) &&
				strings.IndexByte(`{["'`, s[nv]) < 0 &&
				!(s[nv] == '\\' && nv+1 < len(s) && (s[nv+1] == '"' || s[nv+1] == '\'')) {
				continue
			}
			ext, lastIdx, cEnd := nestedPairEndInMatches(s, j, end, open, matches, &bareCache, &checkedBareEnd, &tempCache, &nextTempQuote)
			if ext > end {
				end = ext
			}
			if lastIdx > j {
				j = lastIdx
			}
			for cEnd > 0 && j+1 < len(matches) && matches[j+1].start < cEnd {
				j++
			}
		}
		brackets.skip(end)
		if end == valStart {
			continue
		}
		valSpan := s[valStart:end]
		if kind == valueEndingInKeyName {
			valSpan = s[valStart:valEnd]
		}
		inner, quote := unquoteValue(valSpan)
		if quote == "" && gluedQuoted {
			quote = valQuote
		}
		if kind != valueEndingInKeyName && isRedactedValue(inner) {
			coveredEnd = end
			continue
		}
		b.WriteString(s[last:keyStart])
		b.WriteString(s[keyStart:loc.quoteEnd] + s[loc.end-1:loc.end] + quote + redactedValue + quote)
		last = end
		coveredEnd = end
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

type nestedKeyKind int

const (
	notNestedKey nestedKeyKind = iota
	nestedKey
	valueEndingInKeyName
)

// nestsSensitiveKeyInMatches classifies whether s[start:end] (at most
// maxNestedKeyLen bytes and not a "{" or "[" container) introduces another
// sensitive key with a non-empty value after it.
//
// It returns nestedKey when the inner pair should be redacted on its own while
// keeping the outer key (for example, "credentials:\n  password: SECRET"), and
// valueEndingInKeyName when a same-line value ends in a sensitive key name (for
// example, "password=dev-SECRET-password: dial tcp") so both pairs are redacted
// together.
func nestsSensitiveKeyInMatches(s string, start, end int, lineStart bool, matches []textKeyMatch, nextIdx int) (nestedKeyKind, int) {
	if end-start > maxNestedKeyLen || s[start] == '{' || (s[start] == '[' && !strings.HasPrefix(s[start:], redactedValue)) {
		return notNestedKey, -1
	}
	v := s[start:end]
	if inner, quote := unquoteValue(v); quote != "" {
		rest := strings.TrimLeft(s[end:], delimSpace)
		if rest == "" || (rest[0] != ':' && rest[0] != '=') || !valueStartsAt(rest, 1) {
			return notNestedKey, -1
		}
		return classifyNestedCandidate(s, inner, start+len(quote), lineStart && !(quote == `"` && rest[0] == '=' && trimBOMAndSpace(inner) == inner), matches, nextIdx)
	}
	if lineStart {
		if eq := strings.IndexAny(v, ":="); eq > 0 && eq < len(v)-1 && !hasFoldPrefix(v, session.KeyPrefixTemp) &&
			isSensitiveKey(v[:eq]) && valueStartsAt(s, start+eq+1) {
			return nestedKey, -1
		}
	}
	if strings.HasSuffix(v, ":") || strings.HasSuffix(v, "=") {
		if valueStartsAt(s, end) {
			if kind, idx := classifyNestedCandidate(s, strings.TrimRight(v[:len(v)-1], delimSpace), start, lineStart, matches, nextIdx); kind != notNestedKey {
				return kind, idx
			}
		}
	}
	if rest := strings.TrimLeft(s[end:], delimSpace); rest != "" && (rest[0] == ':' || rest[0] == '=') {
		if valueStartsAt(rest, 1) {
			if kind, idx := classifyNestedCandidate(s, v, start, lineStart, matches, nextIdx); kind != notNestedKey {
				return kind, idx
			}
		}
	}
	for q := 1; q < len(v); q++ {
		if (v[q-1] == ':' || v[q-1] == '=') && strings.IndexByte("\"'\\{[", v[q]) >= 0 {
			isContainer := v[q] == '{' || (v[q] == '[' && !isRedactedPlaceholderPrefix(s[start+q:], false))
			if (isContainer || len(quotedValueRe.FindString(s[start+q:])) > end-start-q) && valueStartsAt(s, start+q) {
				if kind, idx := classifyNestedCandidate(s, v[:q-1], start, lineStart, matches, nextIdx); kind != notNestedKey {
					return kind, idx
				}
			}
		}
	}
	if i := strings.LastIndexAny(v, ":="); i > 0 {
		if inner, _ := unquoteValue(trimTrailingPunctuation(v[i+1:])); isRedactedValue(inner) {
			return classifyNestedCandidate(s, v[:i], start, lineStart, matches, nextIdx)
		}
	}
	return notNestedKey, -1
}

// classifyNestedCandidate resolves a candidate key name found inside another
// sensitive key's value into nestedKey, valueEndingInKeyName, or notNestedKey,
// and returns its index in matches (or -1 if not pre-indexed). Exact keys
// always classify as nestedKey; compound keys classify as nestedKey only when
// lineStart is true (as in YAML blocks) and as valueEndingInKeyName inline.
func isNestedGapRune(r rune) bool {
	return unicode.IsSpace(r) || r == '\ufeff' || (r < utf8.RuneSelf && strings.IndexByte(delimSpace, byte(r)) >= 0)
}

func classifyNestedCandidate(s, name string, at int, lineStart bool, matches []textKeyMatch, nextIdx int) (nestedKeyKind, int) {
	if !strings.HasPrefix(name, `\"`) && !strings.HasPrefix(name, `\'`) &&
		(strings.Contains(name, `\"`) || strings.Contains(name, `\'`)) {
		return notNestedKey, -1
	}
	quoted := at > 0 && at <= len(s) && (s[at-1] == '"' || s[at-1] == '\'')
	trimmedRight := strings.TrimRightFunc(name, func(r rune) bool { return unicode.IsSpace(r) || r == '\ufeff' })
	trimmed := strings.TrimLeftFunc(trimmedRight, func(r rune) bool { return unicode.IsSpace(r) || r == '\ufeff' })
	at += len(trimmedRight) - len(trimmed)
	name = trimmed
	unquoted, offset := name, 0
	for _, q := range []string{`\"`, `\'`, `"`, `'`} {
		if strings.HasPrefix(unquoted, q) {
			unquoted, offset = unquoted[len(q):], len(q)
			break
		}
	}
	idx := -1
	for j := max(0, nextIdx-1); j < len(matches) && matches[j].start <= at+len(name); j++ {
		if matches[j].start >= at+offset && (matches[j].keyEnd >= at+len(name) && strings.TrimRightFunc(s[at+len(name):matches[j].keyEnd], isNestedGapRune) == "" ||
			matches[j].keyEnd == matches[j].quoteEnd && matches[j].end <= len(s) && s[matches[j].end-1] == ':' &&
				hasFoldPrefix(s[matches[j].start:matches[j].keyEnd], session.KeyPrefixTemp) &&
				hasStackedTempColon(s[matches[j].start:at+len(name)])) {
			idx = j
			break
		}
	}
	if _, ok := sensitiveKeys[unquoted]; ok || ((lineStart || (!quoted && offset == 0)) && nestedKeyNameRe.MatchString(name)) {
		return nestedKey, idx
	}
	if hasStackedTempColon(unquoted) {
		return valueEndingInKeyName, idx
	}
	if isSensitiveKey(unquoted) {
		if lineStart {
			return nestedKey, idx
		}
		return valueEndingInKeyName, idx
	}
	m := compoundKeyNameRe.FindStringSubmatchIndex(name)
	if m == nil || sensitiveKeyStart(name[m[2]:m[3]], 0, m[3]-m[2]) < 0 {
		return notNestedKey, -1
	}
	if lineStart && !strings.Contains(name, `\`) && (strings.HasPrefix(name[offset:], name[m[2]:m[3]]) || isBaseSensitiveKey(name[m[2]:m[3]], false)) {
		return nestedKey, idx
	}
	return valueEndingInKeyName, idx
}

// nestedPairEndInMatches follows a chain of nested sensitive keys starting at
// matches[idx] and returns the terminal value end offset, the last match index
// visited (so callers do not re-walk the chain), and the container end offset
// when the terminal value starts with "{" or "[". When bareCache and
// checkedBareEnd are non-nil (quoted-value scan), repeated bare scans and
// dominated closer trims are skipped in O(1).
func nestedPairEndInMatches(s string, idx, fallback, open int, matches []textKeyMatch, bareCache, checkedBareEnd, tempCache *int, nextQuote *[4]int) (end, lastIdx, containerEnd int) {
	if idx < 0 || idx >= len(matches) {
		return fallback, idx, 0
	}
	lastIdx = idx
	for idx >= 0 && idx < len(matches) {
		lastIdx = idx
		loc := matches[idx]
		valStart := skipSpace(s, loc.end, delimSpace)
		if valStart >= len(s) {
			return valStart, lastIdx, 0
		}
		valEnd := matchValueTokenCached(s, valStart, regexpSpace+",;&", bareCache)
		if valEnd == valStart {
			return valStart, lastIdx, 0
		}
		kind, next := nestsSensitiveKeyInMatches(s, valStart, valEnd, strings.ContainsAny(s[loc.end:valStart], "\r\n"), matches, idx+1)
		if kind == notNestedKey || next <= idx {
			isBareTemp := loc.keyEnd == loc.quoteEnd && s[loc.end-1] == ':' && hasFoldPrefix(s[loc.start:loc.keyEnd], session.KeyPrefixTemp)
			if s[valStart] == '{' || s[valStart] == '[' {
				containerEnd = sensitiveValueEnd(s, valStart, valStart)
				if bareCache != nil {
					if max(valEnd, containerEnd) <= fallback {
						if isBareTemp && valEnd < len(s) && (tempCache == nil || valStart >= *tempCache) {
							if stackedEnd := stackedBareTempTailEnd(s, valStart, open, tempCache, nextQuote); stackedEnd > fallback {
								return stackedEnd, lastIdx, containerEnd
							}
						}
						return fallback, lastIdx, containerEnd
					}
					if checkedBareEnd != nil && valEnd > containerEnd && containerEnd < *checkedBareEnd && valEnd == *checkedBareEnd {
						if isBareTemp && valEnd < len(s) && (tempCache == nil || valStart >= *tempCache) {
							if stackedEnd := stackedBareTempTailEnd(s, valStart, open, tempCache, nextQuote); stackedEnd > fallback {
								return stackedEnd, lastIdx, containerEnd
							}
						}
						return fallback, lastIdx, containerEnd
					}
				}
				end = containerEnd + len(trimEnclosingClosers(s[containerEnd:max(valEnd, containerEnd)], open))
				if isBareTemp && end < len(s) {
					if stackedEnd := stackedBareTempTailEnd(s, valStart, open, tempCache, nextQuote); stackedEnd > end {
						end = stackedEnd
					}
				}
				if checkedBareEnd != nil && valEnd > *checkedBareEnd {
					*checkedBareEnd = valEnd
				}
				return end, lastIdx, containerEnd
			}
			if bareCache != nil && valEnd <= fallback {
				if isBareTemp && valEnd < len(s) && (tempCache == nil || valStart >= *tempCache) {
					if stackedEnd := stackedBareTempTailEnd(s, valStart, open, tempCache, nextQuote); stackedEnd > fallback {
						return stackedEnd, lastIdx, 0
					}
				}
				if checkedBareEnd != nil && valEnd > *checkedBareEnd && valEnd-valStart > maxNestedKeyLen && strings.IndexByte(`"'\`, s[valStart]) < 0 {
					*checkedBareEnd = valEnd
				}
				return fallback, lastIdx, 0
			}
			end = textValueEnd(s, valStart, valEnd, open)
			if isBareTemp && end < len(s) {
				if stackedEnd := stackedBareTempTailEnd(s, valStart, open, tempCache, nextQuote); stackedEnd > end {
					end = stackedEnd
				}
			}
			if checkedBareEnd != nil && valEnd > *checkedBareEnd && valEnd-valStart > maxNestedKeyLen && strings.IndexByte(`"'\`, s[valStart]) < 0 {
				*checkedBareEnd = valEnd
			}
			if bareCache != nil && end <= fallback {
				return fallback, lastIdx, 0
			}
			return end, lastIdx, 0
		}
		idx = next
	}
	return fallback, lastIdx, 0
}

func textValueEnd(s string, start, end, open int) int {
	if _, quote := unquoteValue(s[start:end]); quote != "" {
		return end
	}
	rest := sensitiveValueEnd(s, start, start)
	return rest + len(trimEnclosingClosers(s[rest:max(end, rest)], open))
}

// hasNonEmptyTextValue reports whether s contains at least one byte before any
// optional text-value stop byte (whitespace, ",", ";", "&") that is neither
// delimiter whitespace nor closingPunctuation.
func hasNonEmptyTextValue(s string, stopOnTextStop bool) bool {
	for _, c := range []byte(s) {
		if stopOnTextStop && (strings.IndexByte(regexpSpace, c) >= 0 || c == ',' || c == ';' || c == '&') {
			return false
		}
		if strings.IndexByte(delimSpace, c) < 0 && strings.IndexByte(closingPunctuation, c) < 0 {
			return true
		}
	}
	return false
}

// valueStartsAt reports whether a redactable value starts at s[i] (after
// optional whitespace), containing at least one byte outside delimSpace and
// closingPunctuation (so "{Password:abc-token: }" and empty quotes return false).
func valueStartsAt(s string, i int) bool {
	rest := strings.TrimLeft(s[i:], delimSpace)
	if rest != "" && strings.IndexByte(`"'\`, rest[0]) >= 0 {
		if q := quotedValueRe.FindString(rest); q != "" && quotedValueEnds(rest, len(q)) {
			inner, _ := unquoteValue(q)
			return hasNonEmptyTextValue(inner, false)
		}
	}
	return hasNonEmptyTextValue(rest, true)
}

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

// openBrackets tracks unmatched "{" and "[" depth monotonically across a
// string so trimEnclosingClosers knows how many trailing closers belong to
// surrounding containers rather than the redacted value.
type openBrackets struct{ pos, depth int }

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

func (b *openBrackets) skip(i int) { b.pos = max(b.pos, i) }

// trimEnclosingClosers strips trailing "}" and "]" (and punctuation after them)
// from v when they close brackets opened before v (at most open of them),
// keeping brackets opened inside v or followed by non-punctuation bytes.
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
	if j := max(depth, len(closers)-open); j < len(closers) {
		return v[:closers[j]]
	}
	return v
}

// sensitiveValueEnd returns the end of a "{" or "[" container value starting
// at s[start] (respecting quoted and backslash-quoted strings), or len(s) if
// unclosed; for non-container values it returns end unchanged.
func sensitiveValueEnd(s string, start, end int) int {
	if start >= len(s) || (s[start] != '{' && s[start] != '[') {
		return end
	}
	depth, quote, escaped := 0, byte(0), false
	for i := start; i < len(s); i++ {
		c := s[i]
		switch {
		case escaped:
			if c != '\\' || i+1 == len(s) {
				break
			}
			switch s[i+1] {
			case quote:
				quote, escaped = 0, false
				i++
			case '\\':
				i += 2
				if i < len(s) && s[i] == '\\' {
					i++
				}
			default:
				i++
			}
		case c == '\\':
			if quote == 0 && i+1 < len(s) && (s[i+1] == '"' || s[i+1] == '\'') {
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
			if depth--; depth == 0 {
				return max(end, i+1)
			}
		}
	}
	return len(s)
}

func unquoteValue(v string) (inner, quote string) {
	for _, q := range []string{`\"`, `\'`, `"`, `'`} {
		if len(v) >= 2*len(q) && strings.HasPrefix(v, q) && strings.HasSuffix(v, q) {
			return v[len(q) : len(v)-len(q)], q
		}
	}
	return v, ""
}

func isRedactedMarker(v string) bool {
	if isKnownSentinel(v) {
		return true
	}
	decoded, err := url.PathUnescape(v)
	return err == nil && decoded != v && isKnownSentinel(decoded)
}

func isRedactedValue(v string) bool {
	return isRedactedMarker(v) || isRedactedMarker(trimTrailingPunctuation(v))
}

func hasEncodedCredentialMarker(msg string) bool { return exposesEncodedPattern(msg, scrubErrorTokens) }

func exposesEncodedPattern(msg string, scrub func(string) string) bool {
	if msg == "" || isKnownSentinel(msg) {
		return false
	}
	cleaned := strings.ReplaceAll(strings.ReplaceAll(msg, "%5BREDACTED%5D", redactedValue), "%5bredacted%5d", redactedValue)
	canon := canonicalizeCredentialMarker(cleaned)
	if canon != cleaned && scrub(canon) != canon {
		return true
	}
	canonASCII := canonicalizeCredentialMarkerWith(cleaned, true)
	return canonASCII != cleaned && canonASCII != canon && scrub(canonASCII) != canonASCII
}

func hasAnyCredentialMarker(s string) bool {
	if s == "" {
		return false
	}
	if scrubErrorText(s) != s || hasCandidateQuotedSensitiveKey(s, 0, false) {
		return true
	}
	canon := canonicalizeCredentialMarker(s)
	return hasQuotedSensitiveKey(canon, false) || (canon != s && scrubErrorTokens(canon) != canon)
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
