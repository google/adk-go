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
	loc := textKeyEndRe.FindStringSubmatchIndex(key)
	return loc != nil && sensitiveKeyStart(key, loc[2], loc[3]) >= 0
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
	scrubbed := scrubTextPatterns(s)
	if exposesEncodedPattern(scrubbed, scrubErrorTokens) {
		return redactedSensitiveTextPlaceholder, true
	}
	return scrubbed, scrubbed != s
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
// sentinel (such as "[REDACTED]").
func trimTrailingPunctuation(s string) string {
	trimmed := strings.TrimRight(s, ".,;:)]}")
	if withBracket := s[:min(len(trimmed)+1, len(s))]; strings.HasSuffix(withBracket, "]") {
		if open := strings.LastIndexByte(trimmed, '['); open >= 0 && isKnownSentinel(withBracket[open:]) {
			return withBracket
		}
	}
	return trimmed
}
