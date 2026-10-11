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
	"maps"
	"net/url"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"

	"google.golang.org/adk/v2/session"
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

// sensitiveKeys holds normalized (lowercase, "-" replaced by "_") map keys
// whose values are always redacted. It is the adk-python list plus a few
// common credential names.
var sensitiveKeys = map[string]struct{}{
	"access_token":          {},
	"api_key":               {},
	"authorization":         {},
	"bearer":                {},
	"client_secret":         {},
	"credential":            {},
	"credentials":           {},
	"google_access_id":      {},
	"id_token":              {},
	"passphrase":            {},
	"password":              {},
	"private_key":           {},
	"proxy_authorization":   {},
	"refresh_token":         {},
	"secret":                {},
	"sig":                   {},
	"signature":             {},
	"token":                 {},
	"x_amz_credential":      {},
	"x_amz_signature":       {},
	"x_api_key":             {},
	"x_goog_credential":     {},
	"x_goog_security_token": {},
	"x_goog_signature":      {},
}

var (
	uriTokenRe           = regexp.MustCompile(`[a-zA-Z][a-zA-Z0-9+.-]*://[^\s"'<>]+`)
	authHeaderRe         = regexp.MustCompile(`(?im)\b((?:proxy[-_])?authorization|x[-_]api[-_]key|api[-_]key)[ \t]*:[ \t]*[^\r\n]*`)
	authSchemeRe         = regexp.MustCompile(`(?i)\b(bearer|basic)\s+("[^"]*"|'[^']*'|\[[^\]\r\n\s]+\][^\s,;)"']*|[^\s,;)"'\]]+)`)
	jsonKeyValTokenRe    = regexp.MustCompile(`(?i)"[\s\p{Z}]*(` + sensitiveKeyRegex() + `)[\s\p{Z}]*"\s*:\s*("([^"\\]|\\.)*"|[^\s,}\]]+)`)
	textKeyValTokenRe    = regexp.MustCompile(`(?i)\b(` + bareSensitiveKeyRegex() + `)\s*([=:])\s*("[^"]*"|'[^']*'|\[[^\]\r\n\s]+\][^\s,;)"'\]&]*|[^\s,;)"'\]&]+)`)
	quotedSensitiveKeyRe = regexp.MustCompile(`(?i)["'][\s\p{Z}]*(` + sensitiveKeyRegex() + `)[\s\p{Z}]*["']`)
)

// sensitiveKeyRegex returns an alternation of the sensitive key names for
// keys read inside quotes, where a "temp:" key name may contain "=" and ":".
func sensitiveKeyRegex() string {
	return sensitiveKeyAlternation(`temp:[^"'\s\r\n,}]+`)
}

// bareSensitiveKeyRegex returns an alternation of the sensitive key names for
// unquoted key=value and key: value text. A "temp:" key name ends before the
// first "=" or ":", so neither the assignment delimiter nor an "=" or ":"
// inside the value can be read as part of the key and leave the value
// unredacted. Unlike adk-python's temp:[A-Za-z0-9_.-]+, any other key
// character, such as "/", is accepted, so "temp:cache/val=..." is redacted too.
func bareSensitiveKeyRegex() string {
	return sensitiveKeyAlternation(`temp:[^=:"'\s\r\n,}]+`)
}

func sensitiveKeyAlternation(tempKey string) string {
	parts := make([]string, 0, len(sensitiveKeys)+1)
	parts = append(parts, tempKey)
	for _, k := range slices.Sorted(maps.Keys(sensitiveKeys)) {
		parts = append(parts, strings.ReplaceAll(regexp.QuoteMeta(k), "_", "[_-]"))
	}
	return strings.Join(parts, "|")
}

func normalizeKey(k string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(k)), "-", "_")
}

// isSensitiveKey reports whether the value stored under k must be redacted:
// k names a credential, or it is a session "temp:" key, which holds
// invocation-scoped scratch data.
func isSensitiveKey(k string) bool {
	n := normalizeKey(k)
	if _, ok := sensitiveKeys[n]; ok {
		return true
	}
	return strings.HasPrefix(n, session.KeyPrefixTemp)
}

func canonicalizeCredentialMarker(s string) string {
	var output []rune
	for _, r := range s {
		output = append(output, r)
		for {
			escapeStart := -1
			var digits string
			n := len(output)
			if n >= 6 && output[n-6] == '\\' && (output[n-5] == 'u' || output[n-5] == 'U') &&
				isHexRune(output[n-4]) && isHexRune(output[n-3]) && isHexRune(output[n-2]) && isHexRune(output[n-1]) {
				escapeStart = n - 6
				digits = string(output[n-4:])
			} else if n >= 4 && output[n-4] == '\\' && (output[n-3] == 'x' || output[n-3] == 'X') &&
				isHexRune(output[n-2]) && isHexRune(output[n-1]) {
				escapeStart = n - 4
				digits = string(output[n-2:])
			} else if n >= 3 && output[n-3] == '%' &&
				isHexRune(output[n-2]) && isHexRune(output[n-1]) {
				escapeStart = n - 3
				digits = string(output[n-2:])
			}
			if escapeStart < 0 {
				break
			}
			decoded, err := strconv.ParseUint(digits, 16, 32)
			if err != nil || decoded > 0x7F {
				break
			}
			output = append(output[:escapeStart], rune(decoded))
		}
	}
	return string(output)
}

func isHexRune(r rune) bool {
	return (r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')
}

// sanitizeURI redacts credentials carried in a URI. A URI with user
// information, or one that cannot be parsed, is replaced as a whole. Path
// segments matching a sensitive key name redact both that segment and the
// following value segment. Values of sensitive query parameters or fragment
// key-value pairs, and of the "key" parameter used for API keys, are replaced
// with "[REDACTED]". The bool result reports whether the URI was changed.
func sanitizeURI(raw string) (string, bool) {
	if raw == "" {
		return raw, false
	}
	u, err := url.Parse(raw)
	if err != nil || u.User != nil {
		return redactedURIPlaceholder, true
	}
	pathChanged := false
	if escapedPath := u.EscapedPath(); escapedPath != "" {
		if sanitizedPath, unescapedPath, ok := sanitizePathSegments(escapedPath); ok {
			u.Path = unescapedPath
			u.RawPath = sanitizedPath
			pathChanged = true
		}
	}
	queryChanged := false
	if u.RawQuery != "" {
		u.RawQuery, queryChanged = sanitizeQueryPairs(u.RawQuery, false)
	}
	fragmentChanged := false
	if u.Fragment != "" {
		rawFragment := u.EscapedFragment()
		if sanitizedFrag, ok := sanitizeQueryPairs(rawFragment, true); ok {
			if unescaped, err := url.PathUnescape(sanitizedFrag); err == nil {
				u.Fragment = unescaped
			} else {
				u.Fragment = sanitizedFrag
			}
			u.RawFragment = sanitizedFrag
			fragmentChanged = true
		}
	}
	if !pathChanged && !queryChanged && !fragmentChanged {
		return raw, false
	}
	return u.String(), true
}

func sanitizePathSegments(escapedPath string) (escaped, unescaped string, changed bool) {
	segments := strings.Split(escapedPath, "/")
	escapedRedacted := url.QueryEscape(redactedValue)
	redactNext := false
	for i, rawSeg := range segments {
		if rawSeg == "" {
			continue
		}
		seg := canonicalizeCredentialMarker(rawSeg)
		if redactNext {
			segments[i] = escapedRedacted
			changed = true
			redactNext = false
			continue
		}
		if isSensitiveKey(seg) {
			segments[i] = escapedRedacted
			changed = true
			redactNext = true
			continue
		}
		if rawKey, _, hasEq := strings.Cut(seg, "="); hasEq {
			canonKey := canonicalizeCredentialMarker(rawKey)
			if isSensitiveKey(canonKey) || normalizeKey(canonKey) == "key" {
				if rawLeft, _, hasRawEq := strings.Cut(rawSeg, "="); hasRawEq {
					segments[i] = rawLeft + "=" + escapedRedacted
				} else {
					segments[i] = url.PathEscape(canonKey) + "=" + escapedRedacted
				}
				changed = true
			}
		}
	}
	if !changed {
		return escapedPath, "", false
	}
	escaped = strings.Join(segments, "/")
	unescaped, err := url.PathUnescape(escaped)
	if err != nil {
		unescaped = escaped
	}
	return escaped, unescaped, true
}

func sanitizeQueryPairs(raw string, requireEquals bool) (string, bool) {
	pairs := strings.Split(raw, "&")
	changed := false
	for i, pair := range pairs {
		rawKey, _, hasEq := strings.Cut(pair, "=")
		if requireEquals && !hasEq {
			continue
		}
		key := canonicalizeCredentialMarker(rawKey)
		if isSensitiveKey(key) || normalizeKey(key) == "key" {
			pairs[i] = rawKey + "=" + url.QueryEscape(redactedValue)
			changed = true
		}
	}
	if !changed {
		return raw, false
	}
	return strings.Join(pairs, "&"), true
}

func trimBOMAndSpace(s string) string {
	return strings.TrimFunc(s, func(r rune) bool {
		return unicode.IsSpace(r) || r == '\ufeff'
	})
}

func isKnownSentinel(s string) bool {
	switch s {
	case redactedValue,
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
		unsupportedObjPlaceholder:
		return true
	default:
		return false
	}
}

// sanitizeRawText inspects raw user or model text parts for JSON-encoded
// credentials while preserving positively identified harmless prose (such as
// markdown links "[docs](https://...)", log tags "[INFO] ready", or template
// text "{{template}} rendered") when it is not valid JSON and contains no
// raw or encoded credential markers.
func sanitizeRawText(raw string) (string, bool) {
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
	if depth > 4 || s == "" {
		return false
	}
	if len(s) > maxInspectableJSONTextBytes {
		s = s[:maxInspectableJSONTextBytes]
	}
	if quotedSensitiveKeyRe.MatchString(s) {
		return true
	}
	canon := canonicalizeCredentialMarker(s)
	if quotedSensitiveKeyRe.MatchString(canon) {
		return true
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
				if isSensitiveKey(candidate) {
					return true
				}
				if canonKey := canonicalizeCredentialMarker(candidate); isSensitiveKey(canonKey) {
					return true
				}
				if strings.ContainsAny(candidate, `"'`) {
					if hasCandidateQuotedSensitiveKey(candidate, depth+1) {
						return true
					}
				}
			}
		}
	}
	return false
}

func hasCredentialPattern(s string) bool {
	if scrubErrorText(s) != s {
		return true
	}
	if hasCandidateQuotedSensitiveKey(s, 0) {
		return true
	}
	return quotedSensitiveKeyRe.MatchString(canonicalizeCredentialMarker(s))
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
	if hasCredentialPattern(raw) || hasEncodedCredentialMarker(raw) || hasCandidateQuotedSensitiveKey(raw, 0) {
		return false
	}
	// Bracket- or brace-led text that failed JSON parsing and carries no
	// credential pattern is prose (e.g. `[INFO] user clicked "Submit"`).
	if strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "{") {
		return true
	}
	// Quote-led prose (e.g. `"see {this}" she said`) is only harmless prose when it is a single
	// quoted phrase followed by prose words, not a quoted JSON literal or nested quoted fragment.
	if strings.HasPrefix(trimmed, `"`) && strings.Count(trimmed, `"`) == 2 && !strings.HasSuffix(trimmed, `"`) {
		return true
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
			if strings.ContainsAny(suffix, "{[") || strings.HasPrefix(trimmedSuffix, `\`) ||
				hasAnyCredentialMarker(suffix) || inspectQuotedFragments(suffix, depth+1, budget) {
				return unparseableJSONPlaceholder, true, true
			}
		}
		trimmedInner := trimBOMAndSpace(decoded)
		var innerSanitized string
		var innerChanged, innerLost bool
		switch {
		case strings.HasPrefix(trimmedInner, "{") || strings.HasPrefix(trimmedInner, "[") || strings.HasPrefix(trimmedInner, `"`):
			innerSanitized, innerChanged, innerLost = redactJSONTextWithBudget(decoded, depth+1, budget)
		case strings.ContainsAny(decoded, "{[") || hasAnyCredentialMarker(decoded) || inspectQuotedFragments(decoded, depth+1, budget):
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

func inspectQuotedFragments(s string, depth int, budget *nodeBudget) bool {
	if depth >= maxSanitizeDepth {
		return true
	}
	fragments := 0
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
		dec := json.NewDecoder(strings.NewReader(sub))
		var decoded string
		if err := dec.Decode(&decoded); err != nil {
			afterQuote := trimBOMAndSpace(sub[1:])
			if strings.HasPrefix(afterQuote, "{") || strings.HasPrefix(afterQuote, "[") ||
				strings.HasPrefix(afterQuote, `\`) || strings.HasPrefix(afterQuote, `"`) ||
				hasAnyCredentialMarker(sub) {
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

func scrubErrorText(msg string) string {
	msg = scrubErrorTokens(msg)
	if hasEncodedCredentialMarker(msg) {
		return redactedSensitiveTextPlaceholder
	}
	return msg
}

func scrubErrorTokens(msg string) string {
	msg = uriTokenRe.ReplaceAllStringFunc(msg, func(token string) string {
		trimmed := strings.TrimRight(token, ".,;:)]}")
		suffix := token[len(trimmed):]
		sanitized, changed := sanitizeURI(trimmed)
		if !changed {
			return token
		}
		return sanitized + suffix
	})
	msg = authHeaderRe.ReplaceAllString(msg, "${1}: "+redactedValue)
	msg = authSchemeRe.ReplaceAllStringFunc(msg, func(match string) string {
		sub := authSchemeRe.FindStringSubmatch(match)
		if len(sub) < 3 {
			return redactedValue
		}
		val := strings.TrimSpace(sub[2])
		cleanVal := strings.Trim(val, `"'`)
		if cleanVal == redactedValue {
			return match
		}
		return sub[1] + " " + redactedValue
	})
	msg = jsonKeyValTokenRe.ReplaceAllStringFunc(msg, func(match string) string {
		sub := jsonKeyValTokenRe.FindStringSubmatch(match)
		if len(sub) < 2 || !isSensitiveKey(sub[1]) {
			return match
		}
		return `"` + sub[1] + `":"` + redactedValue + `"`
	})
	msg = textKeyValTokenRe.ReplaceAllStringFunc(msg, func(match string) string {
		sub := textKeyValTokenRe.FindStringSubmatch(match)
		if len(sub) < 4 || !isSensitiveKey(sub[1]) {
			return match
		}
		val := strings.TrimSpace(sub[3])
		cleanVal := strings.Trim(val, `"'`)
		if cleanVal == redactedValue || strings.EqualFold(cleanVal, "%5bredacted%5d") || cleanVal == redactedURIPlaceholder {
			return match
		}
		if sub[2] == ":" && strings.HasPrefix(sub[3], "//") {
			return match
		}
		quoteChar := ""
		if strings.HasPrefix(val, `"`) && strings.HasSuffix(val, `"`) {
			quoteChar = `"`
		} else if strings.HasPrefix(val, `'`) && strings.HasSuffix(val, `'`) {
			quoteChar = `'`
		}
		return sub[1] + sub[2] + quoteChar + redactedValue + quoteChar
	})
	return msg
}

func hasEncodedCredentialMarker(msg string) bool {
	if msg == "" || isKnownSentinel(msg) {
		return false
	}
	cleaned := strings.ReplaceAll(msg, "%5BREDACTED%5D", redactedValue)
	cleaned = strings.ReplaceAll(cleaned, "%5bredacted%5d", redactedValue)
	canon := canonicalizeCredentialMarker(cleaned)
	if canon == cleaned {
		return false
	}
	return scrubErrorTokens(canon) != canon
}

func hasAnyCredentialMarker(s string) bool {
	if s == "" {
		return false
	}
	if scrubErrorText(s) != s {
		return true
	}
	if hasCandidateQuotedSensitiveKey(s, 0) {
		return true
	}
	canon := canonicalizeCredentialMarker(s)
	return quotedSensitiveKeyRe.MatchString(canon) || scrubErrorTokens(canon) != canon
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
