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
	"net/url"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

var uriTokenRe = regexp.MustCompile(`[a-zA-Z][a-zA-Z0-9+.-]*://[^\s"'<>]+`)

// sanitizeURI redacts credentials carried in a URI: userinfo and unparseable
// URIs are replaced whole; exact sensitive path segments redact the segment
// and its value while keeping hyphenated route slugs and dotted filenames;
// sensitive matrix, query, and fragment parameters are redacted; and remaining
// components are scrubbed with the text patterns.
func sanitizeURI(raw string) (string, bool) {
	if raw == "" {
		return raw, false
	}
	u, err := url.Parse(raw)
	if err != nil || u.User != nil {
		return redactedURIPlaceholder, true
	}
	changed := false
	if ep := u.EscapedPath(); ep != "" {
		if sanitized, unescaped, ok := sanitizePathSegments(ep); ok {
			u.Path, u.RawPath, changed = unescaped, sanitized, true
		}
	}
	if u.RawQuery != "" {
		if q, ok := sanitizeQueryPairs(u.RawQuery, false); ok {
			u.RawQuery, changed = q, true
		}
	}
	if u.Fragment != "" {
		if sf, ok := sanitizeQueryPairs(u.EscapedFragment(), true); ok {
			u.Fragment, u.RawFragment, changed = unescapePathOrRaw(sf), sf, true
		}
		if safe, ok := redactURIComponent(u.Fragment); ok {
			u.Fragment, u.RawFragment, changed = safe, "", true
		}
	}
	if !changed {
		return raw, false
	}
	return u.String(), true
}

// sanitizePathSegments redacts sensitive URI path segments and the value
// segment that follows each one. A v0.1.0 base key (fromBase) redacts only the
// single following segment (so "/api_key/secret/items" keeps "/items"), whereas
// post-v0.1.0 keys (secret_key, signing_key, stacked user:/app:/temp:, or
// non-ASCII escaped keys) chain across consecutive sensitive path segments
// until the value segment ends the run.
func sanitizePathSegments(escapedPath string) (escaped, unescaped string, changed bool) {
	segments := strings.Split(escapedPath, "/")
	escapedRedacted := url.QueryEscape(redactedValue)
	redactNext := false
	fromBase := false
	for i, rawSeg := range segments {
		if rawSeg == "" {
			continue
		}
		seg := canonicalizeCredentialMarker(unescapePathOrRaw(rawSeg))
		if redactNext {
			// Leave a final "key:" segment (such as ".../secret_key/api_key:?x")
			// intact so the query/text pass can redact its paired value.
			if !fromBase && i == len(segments)-1 && strings.HasSuffix(seg, ":") && isSensitivePathSegment(strings.TrimSuffix(seg, ":")) {
				redactNext = false
				fromBase = false
				continue
			}
			segments[i] = escapedRedacted
			changed = true
			if !fromBase && isSensitivePathSegment(seg) {
				fromBase = isBaseSensitiveKey(canonicalizeCredentialMarkerWith(rawSeg, true), true)
			} else {
				redactNext = false
				fromBase = false
			}
			continue
		}
		if isSensitivePathSegment(seg) {
			segments[i] = escapedRedacted
			changed = true
			redactNext = true
			fromBase = isBaseSensitiveKey(canonicalizeCredentialMarkerWith(rawSeg, true), true)
			continue
		}
		if left, rawVal, hasEq := strings.Cut(rawSeg, "="); hasEq {
			if ck := canonicalizeCredentialMarker(unescapePathOrRaw(left)); isSensitiveKey(ck) || normalizeKey(ck) == "key" {
				safeLeft, keyChanged := redactEscapedComponent(left, url.PathUnescape, url.PathEscape)
				if !isRedactedMarker(canonicalizeCredentialMarker(rawVal)) {
					segments[i], changed = safeLeft+"="+escapedRedacted, true
				} else if keyChanged {
					segments[i], changed = safeLeft+"="+rawVal, true
				}
				continue
			}
		} else if rawKey, val, hasEq := strings.Cut(seg, "="); hasEq {
			if ck := canonicalizeCredentialMarker(rawKey); isSensitiveKey(ck) || normalizeKey(ck) == "key" {
				safeLeft, keyChanged := redactEscapedComponent(url.PathEscape(ck), url.PathUnescape, url.PathEscape)
				if !isRedactedMarker(val) || keyChanged {
					segments[i], changed = safeLeft+"="+escapedRedacted, true
				}
				continue
			}
		}
		if safeSeg, ok := redactEscapedComponent(rawSeg, url.PathUnescape, url.PathEscape); ok {
			segments[i], changed = safeSeg, true
		}
	}
	if !changed {
		return escapedPath, "", false
	}
	escaped = strings.Join(segments, "/")
	return escaped, unescapePathOrRaw(escaped), true
}

func isSensitivePathSegment(k string) bool {
	rest, isTemp := stripStatePrefixes(k)
	if isTemp {
		return true
	}
	if strings.ContainsRune(rest, '.') {
		return false
	}
	_, ok := sensitiveKeyNames[compactKey(rest)]
	return ok
}

// sanitizeQueryPairs redacts "&"-separated key=value pairs in a query string
// (requireEquals=false, decoding "+" as space and scrubbing non-sensitive
// keys/values too) or a fragment (requireEquals=true, using path escaping so
// plain fragments like "#section-1" are left untouched).
func sanitizeQueryPairs(raw string, requireEquals bool) (string, bool) {
	unescape, escape := url.QueryUnescape, url.QueryEscape
	if requireEquals {
		unescape, escape = url.PathUnescape, url.PathEscape
	}
	pairs, changed := strings.Split(raw, "&"), false
	for i, pair := range pairs {
		rawKey, rawValue, hasEq := strings.Cut(pair, "=")
		if requireEquals && !hasEq {
			continue
		}
		keyText := unescapeQueryOrRaw(rawKey)
		if requireEquals {
			keyText = unescapePathOrRaw(rawKey)
		}
		key := canonicalizeCredentialMarker(keyText)
		if isSensitiveKey(key) || normalizeKey(key) == "key" || (!requireEquals && endsInSensitiveTextKey(key)) {
			safeKey, keyChanged := redactEscapedComponent(rawKey, unescape, escape)
			if hasEq && (rawValue == "" || isRedactedMarker(rawValue)) {
				if keyChanged {
					pairs[i], changed = safeKey+"="+rawValue, true
				}
				continue
			}
			pairs[i], changed = safeKey+"="+url.QueryEscape(redactedValue), true
			continue
		}
		if requireEquals {
			continue
		}
		safeKey, keyChanged := redactEscapedComponent(rawKey, unescape, escape)
		safeValue, valueChanged := redactEscapedComponent(rawValue, unescape, escape)
		if keyChanged || valueChanged {
			pairs[i], changed = safeKey, true
			if hasEq {
				pairs[i] += "=" + safeValue
			}
		}
	}
	if !changed {
		return raw, false
	}
	return strings.Join(pairs, "&"), true
}

func unescapePathOrRaw(s string) string {
	if strings.IndexByte(s, '%') >= 0 {
		if decoded, err := url.PathUnescape(s); err == nil {
			return decoded
		}
	}
	return s
}

func unescapeQueryOrRaw(s string) string {
	if strings.IndexByte(s, '%') < 0 && strings.IndexByte(s, '+') < 0 {
		return s
	}
	if decoded, err := url.QueryUnescape(s); err == nil {
		return decoded
	}
	return strings.ReplaceAll(s, "+", " ")
}

func endsInSensitiveTextKey(key string) bool {
	for {
		trimmed := trimBOMAndSpace(key)
		end := len(trimmed)
		for end > 0 && (trimmed[end-1] <= 0x20 || trimmed[end-1] == 0x7f) {
			end--
		}
		if end == len(key) {
			break
		}
		key = trimmed[:end]
	}
	if key == "" {
		return false
	}
	start := len(key)
	for start > 0 {
		if sz := prevKeyRuneSize(key, start, 0); sz > 0 {
			start -= sz
			continue
		}
		break
	}
	if start < len(key) && sensitiveKeyStart(key, start, len(key)) >= 0 {
		return true
	}
	if key[len(key)-1] == ':' {
		return false
	}
	minI := 4
	for p := len(key) - 1; p >= 0; p-- {
		if strings.IndexByte("=\"' \t\r\n\f,}", key[p]) >= 0 {
			minI = max(minI, p)
			break
		}
		if key[p] == ':' && p+1 < len(key) && key[p+1] == ':' {
			minI = max(minI, p+1)
			break
		}
	}
	for i := minI; i+1 < len(key); i++ {
		if key[i] == ':' && strings.EqualFold(key[i-4:i], "temp") {
			if i == 4 {
				return true
			}
			if key[i-5] < utf8.RuneSelf {
				if !isWordByte(key[i-5]) {
					return true
				}
				continue
			}
			if r, _ := utf8.DecodeLastRuneInString(key[:i-4]); !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' {
				return true
			}
		}
	}
	return false
}

func redactEscapedComponent(raw string, unescape func(string) (string, error), escape func(string) string) (string, bool) {
	decoded, err := unescape(raw)
	if err != nil {
		decoded = raw
	}
	safe, changed := redactURIComponent(decoded)
	if !changed {
		return raw, false
	}
	return escape(safe), true
}

func redactURIComponent(s string) (string, bool) {
	if s == "" || isKnownSentinel(s) {
		return s, false
	}
	scrubbed := scrubTextPatterns(scrubNestedUserinfoURIs(s))
	if exposesEncodedPattern(scrubbed, scrubErrorTokens) ||
		hasBackslashEscapedCredentialWith(s, scrubbed, canonicalizeCredentialMarker) ||
		(scrubbed != s && strings.Contains(s, `\\u`) && hasBackslashEscapedCredentialWith(s, scrubbed, func(v string) string {
			return canonicalizeCredentialMarkerWith(v, true)
		})) {
		return redactedSensitiveTextPlaceholder, true
	}
	for prev, pass := s, 1; scrubbed != prev; pass++ {
		if pass == maxRedactionPasses {
			return redactedSensitiveTextPlaceholder, true
		}
		prev = scrubbed
		scrubbed = scrubTextPatterns(scrubbed)
	}
	return scrubbed, scrubbed != s
}

func scrubNestedUserinfoURIs(s string) string {
	if !strings.Contains(s, "://") {
		return s
	}
	spans := uriTokenRe.FindAllStringIndex(s, -1)
	if len(spans) == 0 {
		return s
	}
	var b strings.Builder
	last := 0
	for _, loc := range spans {
		trimmed := trimTrailingPunctuation(s[loc[0]:loc[1]])
		if u, err := url.Parse(trimmed); err != nil || u.User != nil {
			b.WriteString(s[last:loc[0]])
			b.WriteString(redactedURIPlaceholder)
			last = loc[0] + len(trimmed)
		}
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

func scrubURITokens(msg string) string {
	if !strings.Contains(msg, "://") {
		return msg
	}
	spans := uriTokenRe.FindAllStringIndex(msg, -1)
	if len(spans) == 0 {
		return msg
	}
	var b strings.Builder
	last := 0
	for _, loc := range spans {
		trimmed := trimTrailingPunctuation(msg[loc[0]:loc[1]])
		if sanitized, changed := sanitizeURI(trimmed); changed {
			b.WriteString(msg[last:loc[0]])
			b.WriteString(sanitized)
			last = loc[0] + len(trimmed)
		}
	}
	if last == 0 {
		return msg
	}
	b.WriteString(msg[last:])
	return b.String()
}

// trimTrailingPunctuation strips sentence punctuation that follows a URI in
// prose while preserving a closing "]" that completes a known placeholder
// sentinel (such as "[REDACTED]") or an IPv6 host literal (such as
// "https://[2001:db8::1]").
func trimTrailingPunctuation(s string) string {
	end := len(s)
	for end > 0 && strings.IndexByte(".,;:)}", s[end-1]) >= 0 {
		end--
	}
	if end > 0 && s[end-1] == ']' {
		if open := strings.LastIndexByte(s[:end], '['); open >= 0 {
			if closeRel := strings.IndexByte(s[open+1:end], ']'); closeRel >= 0 {
				matchEnd := open + 1 + closeRel + 1
				if strings.TrimRight(s[matchEnd:end], ".,;:)]}") == "" &&
					(isKnownSentinel(s[open:matchEnd]) || isIPv6HostBracket(s[:matchEnd], open)) {
					return s[:matchEnd]
				}
			}
		}
	}
	return strings.TrimRight(s, ".,;:)]}")
}

func isIPv6HostBracket(s string, open int) bool {
	schemeEnd := strings.LastIndex(s[:open], "://")
	if schemeEnd < 0 || !strings.Contains(s[open:], ":") {
		return false
	}
	return open == schemeEnd+3 || (open > schemeEnd+3 && s[open-1] == '@' && !strings.ContainsAny(s[schemeEnd+3:open], "/?#"))
}

// stripValidIPv6ZoneEscapes removes RFC 6874 "%25<zone>" (and percent-encoded
// "%2525<zone>" inside "%3A%2F%2F%5B...%5D") suffixes from bracketed IPv6 hosts
// that carry no userinfo so url.Parse accepts valid zoned IPv6 URIs during
// canonicalized credential checks. Both branches scan bounded spans in O(N).
func stripValidIPv6ZoneEscapes(s string) string {
	if !strings.Contains(s, "%25") {
		return s
	}
	var b strings.Builder
	last := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '[' {
			if i >= 4 && s[i-3:i] == "://" && hasValidURISchemeBefore(s, i-3) {
				if pctPos, closePos, ok := matchRawIPv6ZoneSpan(s, i); ok {
					b.WriteString(s[last:pctPos])
					last = closePos
					i = closePos
					continue
				}
			}
		} else if i+12 <= len(s) && s[i] == '%' && strings.EqualFold(s[i:i+12], "%3A%2F%2F%5B") && hasValidURISchemeBefore(s, i) {
			if pctPos, closePos, ok := matchEncodedIPv6ZoneSpan(s, i+12); ok {
				b.WriteString(s[last:pctPos])
				last = closePos
				i = closePos + 2
				continue
			}
		}
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

func matchRawIPv6ZoneSpan(s string, open int) (pctPos, closePos int, ok bool) {
	hasColon := false
	j := open + 1
	for j < len(s) {
		c := s[j]
		if c == '%' {
			if j+3 <= len(s) && s[j+1] == '2' && s[j+2] == '5' {
				break
			}
			return 0, 0, false
		}
		if c == ':' {
			hasColon = true
			j++
			continue
		}
		if isHexByte(c) || c == '.' {
			j++
			continue
		}
		return 0, 0, false
	}
	if !hasColon || j == open+1 || j+3 > len(s) || s[j:j+3] != "%25" {
		return 0, 0, false
	}
	pctPos = j
	j += 3
	zoneStart := j
	for j < len(s) {
		c := s[j]
		if c == ']' {
			if j > zoneStart {
				return pctPos, j, true
			}
			return 0, 0, false
		}
		if !isRFC6874UnreservedByte(c) {
			return 0, 0, false
		}
		j++
	}
	return 0, 0, false
}

func matchEncodedIPv6ZoneSpan(s string, openEnd int) (pctPos, closePos int, ok bool) {
	hasColon := false
	j := openEnd
	for j < len(s) {
		if s[j] == '%' {
			if j+5 <= len(s) && s[j:j+5] == "%2525" {
				break
			}
			if j+3 <= len(s) && strings.EqualFold(s[j:j+3], "%3A") {
				hasColon = true
				j += 3
				continue
			}
			return 0, 0, false
		}
		if isHexByte(s[j]) || s[j] == '.' {
			j++
			continue
		}
		return 0, 0, false
	}
	if !hasColon || j == openEnd || j+5 > len(s) || s[j:j+5] != "%2525" {
		return 0, 0, false
	}
	pctPos = j
	j += 5
	zoneStart := j
	for j < len(s) {
		if s[j] == '%' {
			if j > zoneStart && j+3 <= len(s) && strings.EqualFold(s[j:j+3], "%5D") {
				return pctPos, j, true
			}
			return 0, 0, false
		}
		if !isRFC6874UnreservedByte(s[j]) {
			return 0, 0, false
		}
		j++
	}
	return 0, 0, false
}

func hasValidURISchemeBefore(s string, schemeEnd int) bool {
	start := schemeEnd
	for start > 0 {
		c := s[start-1]
		if ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9') || c == '+' || c == '.' || c == '-' {
			start--
			continue
		}
		break
	}
	return start < schemeEnd && (('a' <= s[start] && s[start] <= 'z') || ('A' <= s[start] && s[start] <= 'Z'))
}

func isRFC6874UnreservedByte(c byte) bool {
	return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9') || c == '.' || c == '_' || c == '-' || c == '~'
}
