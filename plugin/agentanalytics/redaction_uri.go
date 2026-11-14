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
	"strings"

	"google.golang.org/adk/v2/session"
)

// Credentials in URIs: user information, sensitive path segments, and
// sensitive query, matrix and fragment parameters.

// sanitizeURI redacts credentials carried in a URI. A URI with user
// information, or one that cannot be parsed, is replaced as a whole. Path
// segments matching an exact sensitive key name (see isSensitivePathSegment)
// redact both that segment and the following value segment, while keeping
// compound route slugs such as /account/reset-password/confirm intact. Values
// of sensitive matrix (seg=value), query or fragment key-value pairs, and of
// the "key" parameter used for API keys, are replaced with "[REDACTED]". As in
// adk-python, every other path segment, query key and value, and the fragment
// are scrubbed with the text patterns, so credentials in a nested URL
// (?next=https://host/cb?access_token=...), a fragment (#/cb?access_token=...)
// or a segment (/token:...) are redacted too. The bool result reports whether
// the URI was changed.
func sanitizeURI(raw string) (string, bool) {
	return sanitizeURIWith(raw, redactURIComponent, true)
}

// uriComponentScrub scrubs one decoded URI component (a path segment, a
// query key or value, or the fragment) and reports whether it changed it
// (see redactURIComponent).
type uriComponentScrub func(string) (string, bool)

// sanitizeURIWith is sanitizeURI with scrub in place of redactURIComponent,
// so that the legacy-pattern backstop reads the components of a URI as the
// expressions on main did (see legacyRedactURIComponent). decodeQueryKeys
// reads each query key as redactQueryComponent reads its text, with a "+" as
// a space, as sanitizeURI does (see sanitizeQueryPairs); the legacy scrubs
// read a "+" in a key as itself, as main did.
func sanitizeURIWith(raw string, scrub uriComponentScrub, decodeQueryKeys bool) (string, bool) {
	if raw == "" {
		return raw, false
	}
	u, err := url.Parse(raw)
	if err != nil || u.User != nil {
		return redactedURIPlaceholder, true
	}
	pathChanged := false
	if escapedPath := u.EscapedPath(); escapedPath != "" {
		if sanitizedPath, unescapedPath, ok := sanitizePathSegments(escapedPath, scrub); ok {
			u.Path = unescapedPath
			u.RawPath = sanitizedPath
			pathChanged = true
		}
	}
	queryChanged := false
	if u.RawQuery != "" {
		u.RawQuery, queryChanged = sanitizeQueryPairs(u.RawQuery, false, scrub, decodeQueryKeys)
	}
	fragmentChanged := false
	if u.Fragment != "" {
		rawFragment := u.EscapedFragment()
		if sanitizedFrag, ok := sanitizeQueryPairs(rawFragment, true, scrub, false); ok {
			if unescaped, err := url.PathUnescape(sanitizedFrag); err == nil {
				u.Fragment = unescaped
			} else {
				u.Fragment = sanitizedFrag
			}
			u.RawFragment = sanitizedFrag
			fragmentChanged = true
		}
		if safeFragment, ok := scrub(u.Fragment); ok {
			u.Fragment = safeFragment
			u.RawFragment = ""
			fragmentChanged = true
		}
	}
	if !pathChanged && !queryChanged && !fragmentChanged {
		return raw, false
	}
	return u.String(), true
}

func sanitizePathSegments(escapedPath string, scrub uriComponentScrub) (escaped, unescaped string, changed bool) {
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
		if isSensitivePathSegment(seg) {
			segments[i] = escapedRedacted
			changed = true
			redactNext = true
			continue
		}
		if rawKey, value, hasEq := strings.Cut(seg, "="); hasEq {
			canonKey := canonicalizeCredentialMarker(rawKey)
			if isSensitiveKey(canonKey) || normalizeKey(canonKey) == "key" {
				if isRedactedMarker(value) {
					// Already redacted, as when redacted text is
					// scrubbed again.
					continue
				}
				if rawLeft, _, hasRawEq := strings.Cut(rawSeg, "="); hasRawEq {
					segments[i] = rawLeft + "=" + escapedRedacted
				} else {
					segments[i] = url.PathEscape(canonKey) + "=" + escapedRedacted
				}
				changed = true
				continue
			}
		}
		decoded, err := url.PathUnescape(rawSeg)
		if err != nil {
			decoded = rawSeg
		}
		if safeSeg, ok := scrub(decoded); ok {
			segments[i] = url.PathEscape(safeSeg)
			changed = true
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

// isSensitivePathSegment reports whether seg is a standalone URL path segment
// that introduces a /<key>/<value> credential pair: a session "temp:" key or an
// exact sensitive key name (ignoring case, "_", "-", and "user:"/"app:" state
// prefixes), without matching hyphenated route slugs like "reset-password" or
// dotted filenames like "private.key".
func isSensitivePathSegment(k string) bool {
	k = strings.TrimSpace(k)
	if strings.IndexByte(k, ':') >= 0 {
		for {
			if hasFoldPrefix(k, session.KeyPrefixTemp) {
				return true
			}
			if !hasFoldPrefix(k, session.KeyPrefixUser) && !hasFoldPrefix(k, session.KeyPrefixApp) {
				break
			}
			k = strings.TrimSpace(k[strings.IndexByte(k, ':')+1:])
		}
	}
	b0, b1, ok := last2FoldedKeyBytes(k)
	if !ok || !validSensitiveLast2[b0][b1] {
		return false
	}
	stripped := strings.Map(func(r rune) rune {
		if r == '_' || r == '-' {
			return -1
		}
		return r
	}, foldKeyASCII(k))
	_, ok = sensitiveKeyNames[stripped]
	return ok
}

// sanitizeQueryPairs redacts the values of sensitive keys in raw, a query
// string or a fragment of "&"-separated pairs. For a fragment
// (requireEquals), pairs without "=" are skipped and the caller scrubs the
// whole fragment as text; for a query, the key and value of every other pair
// are scrubbed as text with scrub. With decodeQueryKeys, a key is read with
// each "+" as a space, as redactQueryComponent and url.QueryUnescape read a
// query, so that the +token of ?+token=SECRET is the key " token", as the
// %20token of ?%20token=SECRET is, and a key that ends in a sensitive key
// name after a space or another byte that no key name holds counts, as the
// text scrub of key=value reads it (see endsInSensitiveTextKey), so that
// these values are redacted, as the text scrub of an error message redacts
// them; a "+" in a key such as a+b, which is no sensitive key either way, is
// kept as it is.
func sanitizeQueryPairs(raw string, requireEquals bool, scrub uriComponentScrub, decodeQueryKeys bool) (string, bool) {
	pairs := strings.Split(raw, "&")
	changed := false
	for i, pair := range pairs {
		rawKey, rawValue, hasEq := strings.Cut(pair, "=")
		if requireEquals && !hasEq {
			continue
		}
		keyText := rawKey
		if decodeQueryKeys {
			keyText = strings.ReplaceAll(rawKey, "+", " ")
		}
		key := canonicalizeCredentialMarker(keyText)
		if isSensitiveKey(key) || normalizeKey(key) == "key" || (decodeQueryKeys && endsInSensitiveTextKey(key)) {
			if hasEq && isRedactedMarker(rawValue) {
				// Already redacted, as when redacted text is scrubbed
				// again: re-encoding "[REDACTED]" is not a change.
				continue
			}
			pairs[i] = rawKey + "=" + url.QueryEscape(redactedValue)
			changed = true
			continue
		}
		if requireEquals {
			continue
		}
		safeKey, keyChanged := redactQueryComponent(rawKey, scrub)
		safeValue, valueChanged := redactQueryComponent(rawValue, scrub)
		if keyChanged || valueChanged {
			pairs[i] = safeKey
			if hasEq {
				pairs[i] += "=" + safeValue
			}
			changed = true
		}
	}
	if !changed {
		return raw, false
	}
	return strings.Join(pairs, "&"), true
}

// endsInSensitiveTextKey reports whether key, a decoded query key, ends in a
// sensitive key name, possibly before delimiter whitespace (see isDelimSpace),
// at a boundary that the text scrub of key=value accepts (see
// findTextKeyStart), as the token of "my token", which my+token decodes to, of
// "+token", which %2Btoken decodes to, or of "my token\t", which my+token%09
// decodes to.
func endsInSensitiveTextKey(key string) bool {
	end := len(key)
	for end > 0 && isDelimSpace(key[end-1]) {
		end--
	}
	return end > 0 && findTextKeyStart(key, 0, end) >= 0
}

// redactQueryComponent scrubs one escaped query key or value as text with
// scrub and returns it escaped again. The scrub reads the decoded text, in
// which a "+" is a space, as in password+%3D+SECRET, so the space-delimited
// pairs, bearer and basic credentials in it are read as in any other text;
// url.QueryEscape writes a space as "+" again.
func redactQueryComponent(raw string, scrub uriComponentScrub) (string, bool) {
	decoded, err := url.QueryUnescape(raw)
	if err != nil {
		decoded = raw
	}
	safe, changed := scrub(decoded)
	if !changed {
		return raw, false
	}
	return url.QueryEscape(safe), true
}

// redactURIComponent scrubs one decoded URI component (a path segment, a
// query key or value, or the fragment) with the text patterns, as adk-python
// does, run until the text stops changing, and fails closed when decoding an
// escape in it exposes a credential pattern, or when the scrub keeps a word
// that the legacy expressions redact, as scrubErrorText does for text (see
// scrubURIComponentText). So a URI that is formatted on its own, as the
// file URI of a content part, is checked as text in an error message is. A
// URL nested in the component is not parsed again, which keeps the work
// linear; its key-value pairs are still matched as text.
func redactURIComponent(s string) (string, bool) {
	if s == "" || isKnownSentinel(s) {
		return s, false
	}
	scrubbed := scrubURIComponentText(s)
	return scrubbed, scrubbed != s
}

// legacyRedactURIComponent is redactURIComponent as the expressions on main
// did it: one pass of the legacy expressions over the component (see
// legacyScrubComponentText), failing closed when decoding an escape in it
// exposes a pattern that they redact.
func legacyRedactURIComponent(s string) (string, bool) {
	return legacyRedactURIComponentWith(s, legacyScrubComponentText)
}

// compoundLegacyRedactURIComponent is legacyRedactURIComponent with every key
// that isSensitiveKey accepts (see compoundLegacyScrubErrorTokens).
func compoundLegacyRedactURIComponent(s string) (string, bool) {
	return legacyRedactURIComponentWith(s, compoundLegacyScrubComponentText)
}

func legacyRedactURIComponentWith(s string, scrub func(string) string) (string, bool) {
	if s == "" || isKnownSentinel(s) {
		return s, false
	}
	scrubbed := scrub(s)
	if exposesEncodedPattern(scrubbed, scrub) {
		return redactedSensitiveTextPlaceholder, true
	}
	return scrubbed, scrubbed != s
}

// scrubURITokens redacts the credentials in the URIs in msg (see
// sanitizeURI).
func scrubURITokens(msg string) string {
	return scrubURITokensWith(msg, redactURIComponent, true)
}

// scrubURITokensWith is scrubURITokens with scrub in place of
// redactURIComponent (see sanitizeURIWith).
func scrubURITokensWith(msg string, scrub uriComponentScrub, decodeQueryKeys bool) string {
	if strings.Contains(msg, "://") {
		msg = uriTokenRe.ReplaceAllStringFunc(msg, func(token string) string {
			trimmed := trimTrailingPunctuation(token)
			suffix := token[len(trimmed):]
			sanitized, changed := sanitizeURIWith(trimmed, scrub, decodeQueryKeys)
			if !changed {
				return token
			}
			return sanitized + suffix
		})
	}
	return msg
}

// trimTrailingPunctuation strips the punctuation that may follow a URI or a
// value in text, such as a closing parenthesis or a full stop, from s. A "]"
// that closes a redaction placeholder, as in access_token=[REDACTED], is
// kept, so that scrubbing redacted text again changes nothing.
func trimTrailingPunctuation(s string) string {
	trimmed := strings.TrimRight(s, ".,;:)]}")
	// A placeholder holds no trimmed punctuation, so at most its closing
	// "]" was trimmed.
	if withBracket := s[:min(len(trimmed)+1, len(s))]; strings.HasSuffix(withBracket, "]") {
		if open := strings.LastIndexByte(trimmed, '['); open >= 0 && isKnownSentinel(withBracket[open:]) {
			return withBracket
		}
	}
	return trimmed
}
