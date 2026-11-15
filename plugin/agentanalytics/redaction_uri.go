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
	return sanitizeURIWith(raw, redactURIComponent, scannerURIKeys)
}

// uriKeys tells sanitizeURIWith which names of a URI are sensitive: the path
// segments that start a /<key>/<value> pair, and the keys of path segment,
// query and fragment pairs whose values are redacted. decodeQueryKeys reads
// each query key with each "+" as a space, as redactQueryComponent and
// url.QueryUnescape read a query, and also counts a key that ends in a
// sensitive key name after a space or another byte that no key name holds
// (see sanitizeQueryPairs). scrubKeys scrubs such a key as text too, as any
// other key is, so that a credential written before the key name, as the
// Bearer SECRET of ?Bearer+SECRET+token=..., which the text scrub of a key
// that is no sensitive key redacts, is redacted with it.
type uriKeys struct {
	pathSegment, key func(string) bool
	decodeQueryKeys  bool
	scrubKeys        bool
}

var (
	// scannerURIKeys are the names that sanitizeURI reads as sensitive.
	scannerURIKeys = uriKeys{isSensitivePathSegment, isSensitiveKey, true, true}
	// legacyURIKeys are the names that the expressions on main read as
	// sensitive in a URI, with a "+" in a key read as itself (see
	// isLegacySensitiveKey), and whose own text main kept, so that the
	// legacy scrub leaves for its text passes what main left (see
	// legacyScrubTokensTraced).
	legacyURIKeys = uriKeys{isLegacySensitiveKey, isLegacySensitiveKey, false, false}
	// compoundLegacyURIKeys are scannerURIKeys with a "+" in a key read as
	// itself and the text of a sensitive key kept, for
	// compoundLegacyScrubErrorTokens.
	compoundLegacyURIKeys = uriKeys{isSensitivePathSegment, isSensitiveKey, false, false}
)

// uriComponentScrub scrubs one decoded URI component (a path segment, a
// query key or value, or the fragment) and reports whether it changed it
// (see redactURIComponent).
type uriComponentScrub func(string) (string, bool)

// sanitizeURIWith is sanitizeURI with scrub in place of redactURIComponent
// and keys in place of scannerURIKeys, so that the legacy-pattern backstop
// reads a URI as the expressions on main did (see legacyRedactURIComponent
// and legacyURIKeys).
func sanitizeURIWith(raw string, scrub uriComponentScrub, keys uriKeys) (string, bool) {
	if raw == "" {
		return raw, false
	}
	u, err := url.Parse(raw)
	if err != nil || u.User != nil {
		return redactedURIPlaceholder, true
	}
	pathChanged := false
	if escapedPath := u.EscapedPath(); escapedPath != "" {
		if sanitizedPath, unescapedPath, ok := sanitizePathSegments(escapedPath, scrub, keys); ok {
			u.Path = unescapedPath
			u.RawPath = sanitizedPath
			pathChanged = true
		}
	}
	queryChanged := false
	if u.RawQuery != "" {
		u.RawQuery, queryChanged = sanitizeQueryPairs(u.RawQuery, false, scrub, keys)
	}
	fragmentChanged := false
	if u.Fragment != "" {
		rawFragment := u.EscapedFragment()
		fragKeys := keys
		fragKeys.decodeQueryKeys = false
		if sanitizedFrag, ok := sanitizeQueryPairs(rawFragment, true, scrub, fragKeys); ok {
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

func sanitizePathSegments(escapedPath string, scrub uriComponentScrub, keys uriKeys) (escaped, unescaped string, changed bool) {
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
		if keys.pathSegment(seg) {
			segments[i] = escapedRedacted
			changed = true
			redactNext = true
			continue
		}
		if rawKey, value, hasEq := strings.Cut(seg, "="); hasEq {
			canonKey := canonicalizeCredentialMarker(rawKey)
			if keys.key(canonKey) || normalizeKey(canonKey) == "key" {
				left, rawValue, hasRawEq := strings.Cut(rawSeg, "=")
				if !hasRawEq {
					left, rawValue = url.PathEscape(canonKey), escapedRedacted
				}
				keyChanged := false
				if keys.scrubKeys {
					left, keyChanged = redactPathComponent(left, scrub)
				}
				if isRedactedMarker(value) {
					// Already redacted, as when redacted text is
					// scrubbed again.
					if keyChanged {
						segments[i] = left + "=" + rawValue
						changed = true
					}
					continue
				}
				segments[i] = left + "=" + escapedRedacted
				changed = true
				continue
			}
		}
		if safeSeg, ok := redactPathComponent(rawSeg, scrub); ok {
			segments[i] = safeSeg
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

// redactPathComponent scrubs one escaped path segment, or the key of a
// key=value segment, as text with scrub and returns it escaped again.
func redactPathComponent(raw string, scrub uriComponentScrub) (string, bool) {
	decoded, err := url.PathUnescape(raw)
	if err != nil {
		decoded = raw
	}
	safe, changed := scrub(decoded)
	if !changed {
		return raw, false
	}
	return url.PathEscape(safe), true
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
// kept as it is. With scrubKeys, a sensitive key is scrubbed as text as
// every other key is (see redactQueryComponent), also when its value is
// redacted already.
func sanitizeQueryPairs(raw string, requireEquals bool, scrub uriComponentScrub, keys uriKeys) (string, bool) {
	pairs := strings.Split(raw, "&")
	changed := false
	for i, pair := range pairs {
		rawKey, rawValue, hasEq := strings.Cut(pair, "=")
		if requireEquals && !hasEq {
			continue
		}
		keyText := rawKey
		if keys.decodeQueryKeys {
			keyText = strings.ReplaceAll(rawKey, "+", " ")
		}
		key := canonicalizeCredentialMarker(keyText)
		if keys.key(key) || normalizeKey(key) == "key" || (keys.decodeQueryKeys && endsInSensitiveTextKey(key)) {
			safeKey, keyChanged := rawKey, false
			if keys.scrubKeys {
				safeKey, keyChanged = redactQueryComponent(rawKey, scrub)
			}
			if hasEq && isRedactedMarker(rawValue) {
				// Already redacted, as when redacted text is scrubbed
				// again: re-encoding "[REDACTED]" is not a change.
				if keyChanged {
					pairs[i] = safeKey + "=" + rawValue
					changed = true
				}
				continue
			}
			pairs[i] = safeKey + "=" + url.QueryEscape(redactedValue)
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
// escape in it exposes a credential pattern, or when the scrub keeps a byte
// that the legacy expressions remove (see leavesLegacyValue), as
// scrubErrorText does for text (see scrubURIComponentText). So a URI that is
// formatted on its own, as the file URI of a content part, is checked as text
// in an error message is. A URL nested in the component is not parsed again,
// which keeps the work linear; its key-value pairs are still matched as text.
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
// sanitizeURIWith) with scrub and keys, which scrubErrorTokens sets to
// redactURIComponent and scannerURIKeys, recording in tr, when it is not
// nil, the runs of the result copied from msg and the bytes of each
// rewritten URI that its rewrite does not keep (see spanWriter.rewriteURI),
// so that what the legacy scrubs remove from a URI, as a credential in a
// component or a value of a sensitive key, counts as removed by them (see
// leavesLegacyValue).
func scrubURITokens(msg string, scrub uriComponentScrub, keys uriKeys, tr *passTrace) string {
	w := spanWriter{s: msg, tr: tr}
	if !strings.Contains(msg, "://") {
		return w.done(0)
	}
	last := 0
	for _, loc := range uriTokenRe.FindAllStringIndex(msg, -1) {
		trimmed := trimTrailingPunctuation(msg[loc[0]:loc[1]])
		sanitized, changed := sanitizeURIWith(trimmed, scrub, keys)
		if !changed {
			continue
		}
		w.copy(last, loc[0])
		w.rewriteURI(loc[0], loc[0]+len(trimmed), sanitized)
		last = loc[0] + len(trimmed)
	}
	return w.done(last)
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
