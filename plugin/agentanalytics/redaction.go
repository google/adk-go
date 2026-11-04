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
	"maps"
	"net/url"
	"regexp"
	"slices"
	"strconv"
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
)

const (
	maxInspectableJSONTextBytes  = 4_000_000 // adk-python _MAX_JSON_INSPECT_CHARS
	maxCanonicalUnescapePasses   = 4
	maxQuotedFragmentsPerInspect = 64
)

// sensitiveKeys holds normalized (lowercase, "-" replaced by "_") map keys
// whose values are always redacted. It is the adk-python list plus
// "api_key_secret", "api_key_string", "auth_code", "aws_secret_access_key",
// "aws_session_token", "bearer", "code_verifier", "credential", "credentials",
// "passphrase", "secret_access_key", "session_token", "x_auth_token" and
// "x_goog_api_key". A value under "credential" or "credentials" in a map, in
// JSON text, in a Python dict or in a Go struct printed with %+v is redacted
// as a whole, so that no field of a credentials envelope, such as the AWS STS
// result {"Credentials": {"AccessKeyId": ..., "SecretAccessKey": ...,
// "SessionToken": ...}}, is logged, whatever its name. In YAML or other
// key: value text, the fields under "credentials:" are not redacted as a
// whole: each is redacted only if its own name is sensitive (see
// nestsSensitiveKey). isSensitiveKey and the text patterns also match these
// names without separators, as in the camelCase apiKey or the Go field name
// APIKey.
var sensitiveKeys = map[string]struct{}{
	"access_token":          {},
	"api_key":               {},
	"api_key_secret":        {},
	"api_key_string":        {},
	"auth_code":             {},
	"authorization":         {},
	"aws_secret_access_key": {},
	"aws_session_token":     {},
	"bearer":                {},
	"client_secret":         {},
	"code_verifier":         {},
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
	"secret_access_key":     {},
	"session_token":         {},
	"sig":                   {},
	"signature":             {},
	"token":                 {},
	"x_amz_credential":      {},
	"x_amz_signature":       {},
	"x_api_key":             {},
	"x_auth_token":          {},
	"x_goog_api_key":        {},
	"x_goog_credential":     {},
	"x_goog_security_token": {},
	"x_goog_signature":      {},
}

// sensitiveKeyNames holds the names in sensitiveKeys with "_" removed, the
// form in which isSensitiveKey compares keys.
var sensitiveKeyNames = func() map[string]struct{} {
	names := make(map[string]struct{}, len(sensitiveKeys))
	for k := range sensitiveKeys {
		names[removeKeySeparators(k)] = struct{}{}
	}
	return names
}()

var (
	uriTokenRe   = regexp.MustCompile(`[a-zA-Z][a-zA-Z0-9+.-]*://[^\s"'<>]+`)
	authHeaderRe = regexp.MustCompile(`(?im)\b((?:proxy[-_])?authorization|x[-_]api[-_]key|api[-_]key)[ \t]*:[ \t]*[^\r\n]*`)
	// bearerTokenRe matches a bearer token. A bearer token (RFC 6750) never
	// contains quotes, ")" or "]", so the bare value stops there and keeps
	// the surrounding JSON or prose intact.
	bearerTokenRe = regexp.MustCompile(`(?i)\b(bearer)\s+("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\[[^\]\r\n\s]+\][^\s,;)"']*|[^\s,;)"'\]]+)`)
	// basicTokenRe matches a candidate basic credential, which
	// redactBasicCredential redacts only if it is valid base64.
	basicTokenRe      = regexp.MustCompile(`(?i)\b(basic)\s+([A-Za-z0-9+/]+={0,2})`)
	jsonKeyValTokenRe = regexp.MustCompile(`(?i)"[\s\p{Z}]*(` + sensitiveKeyRegex() + `)[\s\p{Z}]*"\s*:\s*(` + sensitiveValueRegex(`\s,}`) + `)`)
	// textKeyValTokenRe matches key=value and key: value pairs. As in
	// adk-python, the key may be quoted (Python dict reprs use single
	// quotes); it may also be quoted with escaped quotes, as in JSON text
	// formatted with %q. Unlike adk-python's, a bare value does not stop at
	// "}", so a value such as "ab}rest" is redacted as a whole;
	// redactTextKeyValue leaves out the "}" and "]" that end the value and
	// close a bracket opened before the key, as in the %+v output
	// {APIKey:...}.
	textKeyValTokenRe   = regexp.MustCompile(`(?i)\b(` + bareSensitiveKeyRegex() + `)((?:\\?["'])?)[\s[:cntrl:]]*([=:])[\s[:cntrl:]]*(` + sensitiveValueRegex(`\s,;&`) + `)`)
	textKeyValTokenAtRe = regexp.MustCompile(`(?i)^(` + bareSensitiveKeyRegex() + `)((?:\\?["'])?)[\s[:cntrl:]]*([=:])[\s[:cntrl:]]*(` + sensitiveValueRegex(`\s,;&`) + `)`)
	// nestedKeyNameRe matches a value that is itself a fixed sensitive key
	// name (see nestsSensitiveKey): the name, possibly after a quote,
	// which may be escaped, and a "user:" or "app:" state prefix, as
	// isSensitiveKey reads it. The submatch is the key name.
	nestedKeyNameRe = regexp.MustCompile(`(?i)^(?:\\?["'])?(?:` + regexp.QuoteMeta(session.KeyPrefixUser) + `|` + regexp.QuoteMeta(session.KeyPrefixApp) + `)?(` + fixedSensitiveKeyRegex() + `)$`)
	// compoundKeyNameRe matches a value that is a "temp:" key name or only
	// ends in a sensitive key name (see nestsSensitiveKey): the name,
	// possibly after a quote, which may be escaped, and optional words
	// before it that end in a character other than a letter, a digit or
	// "_", such as the "user:" of user:temp:foo, the "X-Amz-Security-" of
	// X-Amz-Security-Token, the "spring.datasource." of
	// spring.datasource.password or the "db/" of db/password, but also the
	// "dev-SECRET-" of dev-SECRET-password. The submatch is the key name.
	compoundKeyNameRe = regexp.MustCompile(`(?i)^(?:\\?["'])?(?:[^\s"'\\]*[^\w\s"'\\])?(` + bareSensitiveKeyRegex() + `)$`)
	// quotedValueRe matches a quoted value (see quotedValueRegex) at the
	// start of a text.
	quotedValueRe = regexp.MustCompile(`^(?:` + quotedValueRegex() + `)`)
	// quotedSensitiveKeyRe detects a quoted sensitive key that is followed by
	// a key-value delimiter. A quoted sensitive word in prose, such as
	// `What does "token" mean?`, is not a key.
	quotedSensitiveKeyRe = regexp.MustCompile(`(?i)["'][\s\p{Z}]*(` + sensitiveKeyRegex() + `)[\s\p{Z}]*["'][\s\p{Z}]*[:=]`)
)

// patternWords holds words of which every match of authHeaderRe,
// bearerTokenRe, basicTokenRe, jsonKeyValTokenRe, textKeyValTokenRe and
// quotedSensitiveKeyRe contains at least one, once the text is folded by
// foldPatternText: the sensitive key names without separators (which include
// "authorization", "apikey" and "bearer"), the "temp:" prefix and the "basic"
// scheme. A name that contains another, such as "accesstoken", which contains
// "token", is left out, since the text then contains the shorter name too.
var patternWords = func() []string {
	words := append([]string{"basic", session.KeyPrefixTemp}, slices.Collect(maps.Keys(sensitiveKeyNames))...)
	minimal := slices.DeleteFunc(slices.Clone(words), func(w string) bool {
		return slices.ContainsFunc(words, func(other string) bool {
			return other != w && strings.Contains(w, other)
		})
	})
	slices.Sort(minimal)
	return minimal
}()

// mayMatchTextPatterns reports whether s contains one of patternWords,
// ignoring case. When it does not, none of the credential text patterns can
// match s, so the regular expressions, which are slow on long text, need
// not run over it.
func mayMatchTextPatterns(s string) bool {
	folded := foldPatternText(s)
	for _, w := range patternWords {
		if strings.Contains(folded, w) {
			return true
		}
	}
	return false
}

// foldPatternText lowercases the ASCII letters of s and removes "-" and "_",
// as the patterns accept key names with either separator or none (api_key,
// api-key, apiKey). It also replaces the two non-ASCII letters that (?i)
// matches against ASCII ones, U+017F (long s) and U+212A (Kelvin sign), with
// "s" and "k".
func foldPatternText(s string) string {
	b := make([]byte, 0, len(s))
	for i := 0; i < len(s); {
		c := s[i]
		if c < utf8.RuneSelf {
			switch {
			case 'A' <= c && c <= 'Z':
				b = append(b, c+'a'-'A')
			case c != '-' && c != '_':
				b = append(b, c)
			}
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		switch r {
		case '\u017f':
			b = append(b, 's')
		case '\u212a':
			b = append(b, 'k')
		default:
			b = append(b, s[i:i+size]...)
		}
		i += size
	}
	return string(b)
}

// hasQuotedSensitiveKey reports whether quotedSensitiveKeyRe matches s.
func hasQuotedSensitiveKey(s string) bool {
	return mayMatchTextPatterns(s) && quotedSensitiveKeyRe.MatchString(s)
}

// sensitiveValueRegex returns the pattern for the value that follows a
// sensitive key: a double- or single-quoted string, a string quoted with
// escaped quotes (\" or \'), or a run of characters other than those in
// bareStop. For key=value text, bareStop is `\s,;&`, adk-python's class
// without "}", so that a value such as "abc)rest", "ab}rest" or "//rest" is
// redacted as a whole.
func sensitiveValueRegex(bareStop string) string {
	return quotedValueRegex() + `|[^` + bareStop + `]+`
}

// quotedValueRegex returns the pattern for a quoted value: a double- or
// single-quoted string, or a string quoted with escaped quotes (\" or \').
func quotedValueRegex() string {
	return `"(?:[^"\\]|\\.)*"` +
		`|'(?:[^'\\]|\\.)*'` +
		`|` + escapedQuotedRegex(`"`) +
		`|` + escapedQuotedRegex(`'`)
}

// escapedQuotedRegex returns the pattern for a string delimited by \q, as
// produced by quoting JSON text again (for example with %q). Inside it, a
// doubled backslash starts an escape of the inner text (\\\" is an escaped
// quote within the value), so the string ends only at an unpaired \q.
func escapedQuotedRegex(q string) string {
	return `\\` + q + `(?:\\\\(?:\\\\|\\` + q + `|[^\\` + q + `])|\\[^\\` + q + `]|[^\\` + q + `])*\\` + q
}

// sensitiveKeyRegex returns an alternation of the sensitive key names for
// keys read inside quotes, where a "temp:" key name may contain "=" and ":".
func sensitiveKeyRegex() string {
	return `temp:[^"'\s\r\n,}]+|` + fixedSensitiveKeyRegex()
}

// bareSensitiveKeyRegex returns an alternation of the sensitive key names for
// unquoted key=value and key: value text. A "temp:" key name ends before the
// first "=" or ":", so neither the assignment delimiter nor an "=" or ":"
// inside the value can be read as part of the key and leave the value
// unredacted. Unlike adk-python's temp:[A-Za-z0-9_.-]+, any other key
// character, such as "/", is accepted, so "temp:cache/val=..." is redacted too.
func bareSensitiveKeyRegex() string {
	return `temp:[^=:"'\s\r\n,}]+|` + fixedSensitiveKeyRegex()
}

// fixedSensitiveKeyRegex returns an alternation of the names in sensitiveKeys,
// in which each "_" may also be written as "-" or left out, so that api_key,
// api-key, apiKey and APIKey all match.
func fixedSensitiveKeyRegex() string {
	parts := make([]string, 0, len(sensitiveKeys))
	for _, k := range slices.Sorted(maps.Keys(sensitiveKeys)) {
		parts = append(parts, strings.ReplaceAll(regexp.QuoteMeta(k), "_", "[_-]?"))
	}
	return strings.Join(parts, "|")
}

func normalizeKey(k string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(k)), "-", "_")
}

// removeKeySeparators returns k without its "_" and "-" separators.
func removeKeySeparators(k string) string {
	return strings.Map(func(r rune) rune {
		if r == '_' || r == '-' {
			return -1
		}
		return r
	}, k)
}

// isSensitiveKey reports whether the value stored under k must be redacted:
// k names a credential, or it is a session "temp:" key, which holds
// invocation-scoped scratch data. Credential names are compared ignoring
// case, the "_" and "-" separators, and a "user:" or "app:" state prefix, so
// that api_key, api-key, apiKey, the Go field name APIKey and the state key
// user:api_key all match.
func isSensitiveKey(k string) bool {
	n := strings.ToLower(strings.TrimSpace(k))
	if strings.HasPrefix(n, session.KeyPrefixTemp) {
		return true
	}
	if rest, ok := strings.CutPrefix(n, session.KeyPrefixUser); ok {
		n = rest
	} else {
		n = strings.TrimPrefix(n, session.KeyPrefixApp)
	}
	_, ok := sensitiveKeyNames[removeKeySeparators(strings.TrimSpace(n))]
	return ok
}

// canonicalizeCredentialMarker decodes nested ASCII \u, \x and percent
// escapes, and escaped quotes (\" and \'), in one pass. The result is used
// for detection only: callers keep the original text unless the canonical
// form exposes a credential construct.
func canonicalizeCredentialMarker(s string) string {
	if !strings.ContainsAny(s, `\%`) {
		// Every escape starts with "\" or "%".
		return s
	}
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
			} else if n >= 2 && output[n-2] == '\\' && (output[n-1] == '"' || output[n-1] == '\'') {
				output = append(output[:n-2], output[n-1])
				continue
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
// with "[REDACTED]". As in adk-python, every other path segment, query key and
// value, and the fragment are scrubbed with the text patterns, so credentials
// in a nested URL (?next=https://host/cb?access_token=...), a fragment
// (#/cb?access_token=...) or a segment (/token:...) are redacted too. The bool
// result reports whether the URI was changed.
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
		if safeFragment, ok := redactURIComponent(u.Fragment); ok {
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
		if safeSeg, ok := redactURIComponent(decoded); ok {
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

// sanitizeQueryPairs redacts the values of sensitive keys in raw, a query
// string or a fragment of "&"-separated pairs. For a fragment
// (requireEquals), pairs without "=" are skipped and the caller scrubs the
// whole fragment as text; for a query, the key and value of every other pair
// are scrubbed as text.
func sanitizeQueryPairs(raw string, requireEquals bool) (string, bool) {
	pairs := strings.Split(raw, "&")
	changed := false
	for i, pair := range pairs {
		rawKey, rawValue, hasEq := strings.Cut(pair, "=")
		if requireEquals && !hasEq {
			continue
		}
		key := canonicalizeCredentialMarker(rawKey)
		if isSensitiveKey(key) || normalizeKey(key) == "key" {
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
		safeKey, keyChanged := redactQueryComponent(rawKey)
		safeValue, valueChanged := redactQueryComponent(rawValue)
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

// redactQueryComponent scrubs one escaped query key or value as text and
// returns it escaped again.
func redactQueryComponent(raw string) (string, bool) {
	decoded, err := url.QueryUnescape(raw)
	if err != nil {
		decoded = raw
	}
	safe, changed := redactURIComponent(decoded)
	if !changed {
		return raw, false
	}
	return url.QueryEscape(safe), true
}

// redactURIComponent scrubs one decoded URI component (a path segment, a
// query key or value, or the fragment) with the text patterns, as adk-python
// does, and fails closed when decoding an escape in it exposes a credential
// pattern. A URL nested in the component is not parsed again, which keeps
// the work linear; its key-value pairs are still matched as text.
func redactURIComponent(s string) (string, bool) {
	if s == "" || isKnownSentinel(s) {
		return s, false
	}
	scrubbed := scrubTextPatterns(s)
	if exposesEncodedPattern(scrubbed, scrubTextPatterns) {
		return redactedSensitiveTextPlaceholder, true
	}
	return scrubbed, scrubbed != s
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
	if hasQuotedSensitiveKey(s) {
		return true
	}
	if canon := canonicalizeCredentialMarker(s); canon != s && hasQuotedSensitiveKey(canon) {
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
				if isSensitiveKey(candidate) || isSensitiveKey(canonicalizeCredentialMarker(candidate)) {
					if followedByKeyDelimiter(s[i+1:]) {
						return true
					}
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

// followedByKeyDelimiter reports whether rest, the text after a quoted
// string, starts with a key-value delimiter (":" or "=") after optional
// whitespace.
func followedByKeyDelimiter(rest string) bool {
	for _, r := range rest {
		if unicode.IsSpace(r) || unicode.In(r, unicode.Z) {
			continue
		}
		return r == ':' || r == '='
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
	return hasQuotedSensitiveKey(canonicalizeCredentialMarker(s))
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

func scrubErrorText(msg string) string {
	scrubbed := scrubErrorTokens(msg)
	if hasEncodedCredentialMarker(scrubbed) || hasBackslashEscapedCredential(msg, scrubbed) {
		return redactedSensitiveTextPlaceholder
	}
	return scrubbed
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

// hasBackslashEscapedCredential reports whether unescaping standard backslash
// escapes (\b, \f, \n, \r, \t, \/, \", \' and \\), or canonicalizing percent,
// \x or \u escapes, exposes a credential that scrubErrorTokens or
// hasEncodedCredentialMarker catches, including when scrubErrorTokens on raw
// consumed an escape inside an earlier value and left a following secret in
// scrubbed. Each pass decodes a run of backslashes in one step (see
// unescapeBackslashControls), so passes are needed only where percent, \x or
// \u escapes and backslash escapes nest inside each other; up to
// 2*maxCanonicalUnescapePasses are run, failing closed if escapes still change
// past that depth.
func hasBackslashEscapedCredential(raw, scrubbed string) bool {
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
			if scrubErrorTokens(nextRaw) != nextRaw || hasEncodedCredentialMarker(nextRaw) {
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
		if scrubErrorTokens(nextScrubbed) != nextScrubbed || hasEncodedCredentialMarker(nextScrubbed) {
			return true
		}
		if !keptInOrder(parts, scrubErrorTokens(nextRaw)) {
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
			b.WriteByte(s[i])
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

func scrubErrorTokens(msg string) string {
	if strings.Contains(msg, "://") {
		msg = uriTokenRe.ReplaceAllStringFunc(msg, func(token string) string {
			trimmed := trimTrailingPunctuation(token)
			suffix := token[len(trimmed):]
			sanitized, changed := sanitizeURI(trimmed)
			if !changed {
				return token
			}
			return sanitized + suffix
		})
	}
	return scrubTextPatterns(msg)
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

// scrubTextPatterns redacts Authorization and API-key headers, bearer and
// basic credentials, and the values of sensitive key-value pairs in s.
func scrubTextPatterns(s string) string {
	if !mayMatchTextPatterns(s) {
		return s
	}
	s = authHeaderRe.ReplaceAllString(s, "${1}: "+redactedValue)
	s = replaceMatches(s, bearerTokenRe, redactBearerToken)
	s = replaceMatches(s, basicTokenRe, redactBasicCredential)
	s = replaceMatches(s, jsonKeyValTokenRe, redactJSONKeyValue)
	var brackets openBrackets
	return replaceMatchesFunc(s, findKeyValToken, func(s string, loc []int) (string, int) {
		return redactTextKeyValue(s, loc, &brackets)
	})
}

func findKeyValToken(s string) []int {
	if loc := textKeyValTokenAtRe.FindStringSubmatchIndex(s); loc != nil {
		return loc
	}
	return textKeyValTokenRe.FindStringSubmatchIndex(s)
}

// replaceMatches replaces each match of re in s with the text returned by
// repl, which receives the submatch indexes of the match and also returns
// where the replaced text ends; it may extend past the match. The search
// resumes there, so the replaced spans never overlap.
func replaceMatches(s string, re *regexp.Regexp, repl func(s string, loc []int) (string, int)) string {
	return replaceMatchesFunc(s, re.FindStringSubmatchIndex, repl)
}

func replaceMatchesFunc(s string, find func(string) []int, repl func(s string, loc []int) (string, int)) string {
	var b strings.Builder
	last, pos := 0, 0
	for pos < len(s) {
		loc := find(s[pos:])
		if loc == nil {
			break
		}
		for i, v := range loc {
			if v >= 0 {
				loc[i] = v + pos
			}
		}
		replacement, end := repl(s, loc)
		b.WriteString(s[last:loc[0]])
		b.WriteString(replacement)
		last = end
		pos = max(end, loc[0]+1)
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

// redactBearerToken replaces a bearerTokenRe match. As in adk-python, "bearer
// of" followed by a space is prose ("the bearer of bad news"), not a token.
func redactBearerToken(s string, loc []int) (string, int) {
	end := loc[5]
	value := s[loc[4]:end]
	if rest := s[loc[4]:]; len(rest) >= 2 && strings.EqualFold(rest[:2], "of") {
		if r, _ := utf8.DecodeRuneInString(rest[2:]); len(rest) == 2 || unicode.IsSpace(r) {
			return s[loc[0]:end], end
		}
	}
	if inner, _ := unquoteValue(value); isRedactedValue(inner) {
		return s[loc[0]:end], end
	}
	return s[loc[2]:loc[3]] + " " + redactedValue, end
}

// maxBasicCredentialBytes is adk-python's _MAX_BASIC_AUTH_TOKEN_CHARS.
const maxBasicCredentialBytes = 16 * 1024

// redactBasicCredential replaces a basicTokenRe match when it holds a basic
// credential, using adk-python's test: the base64 token must not run on
// into more base64 characters and must decode, and it must be padded or
// decode to "user:password". An over-long token is always redacted. Prose
// such as "a basic HTTP server" or "basic auth" is kept.
func redactBasicCredential(s string, loc []int) (string, int) {
	end := loc[5]
	encoded := s[loc[4]:end]
	if end < len(s) && isBase64Byte(s[end]) {
		return s[loc[0]:end], end
	}
	if len(encoded) <= maxBasicCredentialBytes {
		decoded, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil || (!strings.Contains(encoded, "=") && !bytes.Contains(decoded, []byte(":"))) {
			return s[loc[0]:end], end
		}
	}
	return s[loc[2]:loc[3]] + " " + redactedValue, end
}

func isBase64Byte(c byte) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || '0' <= c && c <= '9' || c == '+' || c == '/' || c == '='
}

// redactJSONKeyValue replaces a jsonKeyValTokenRe match with
// "key":"[REDACTED]".
func redactJSONKeyValue(s string, loc []int) (string, int) {
	end := sensitiveValueEnd(s, loc[4], loc[5])
	if inner, _ := unquoteValue(s[loc[4]:end]); isRedactedValue(inner) {
		return s[loc[0]:end], end
	}
	return `"` + s[loc[2]:loc[3]] + `":"` + redactedValue + `"`, end
}

// redactTextKeyValue replaces a textKeyValTokenRe match, keeping the key, its
// quote, the delimiter and the quotes around the value. A value that is itself
// a sensitive key, as in the YAML "credentials:\n  token: ...", is left in
// place, so that the pair it starts is matched and redacted on its own (see
// nestsSensitiveKey). A value that only ends in a key name, as in
// "password=dev-SECRET-password: dial tcp ...", is redacted together with the
// value of the pair it starts (see nestedPairEnd). A quoted value ends at its
// closing quote. A value that opens a JSON object or array is extended as in
// sensitiveValueEnd. A bare value runs on past any "}" or "]" in it and is
// redacted as a whole, except for the "}" and "]" at its end that close a
// bracket opened before the key, and the punctuation after them (see
// trimEnclosingClosers), as in the %+v output {APIKey:...} or
// [{APIKey:...}]. brackets counts the brackets opened before the key.
func redactTextKeyValue(s string, loc []int, brackets *openBrackets) (string, int) {
	open := brackets.before(s, loc[0])
	start, end := loc[8], loc[9]
	switch kind, pair := nestsSensitiveKey(s, start, end, startsLine(s, loc[7], start)); kind {
	case nestedKey:
		return s[loc[0]:start], start
	case valueEndingInKeyName:
		_, quote := unquoteValue(s[start:end])
		end = nestedPairEnd(s, pair, open)
		brackets.skip(end)
		return s[loc[2]:loc[3]] + s[loc[4]:loc[5]] + s[loc[6]:loc[7]] + quote + redactedValue + quote, end
	}
	end = textValueEnd(s, start, end, open)
	brackets.skip(end)
	if end == start {
		// Only closing brackets follow the delimiter, as in {Password:}.
		return s[loc[0]:end], end
	}
	inner, quote := unquoteValue(s[start:end])
	if isRedactedValue(inner) {
		return s[loc[0]:end], end
	}
	return s[loc[2]:loc[3]] + s[loc[4]:loc[5]] + s[loc[6]:loc[7]] + quote + redactedValue + quote, end
}

// textValueEnd returns the end of the value matched at s[start:end] after a
// key: a quoted value ends at its closing quote, a value that opens a JSON
// object or array is extended as in sensitiveValueEnd, and a bare value ends
// before the "}" and "]" at its end that close a bracket opened before the
// key, and the punctuation after them (see trimEnclosingClosers). open is the
// number of brackets opened before the key.
func textValueEnd(s string, start, end, open int) int {
	if _, quote := unquoteValue(s[start:end]); quote != "" {
		return end
	}
	// rest is where the text after a JSON object or array value starts.
	rest := sensitiveValueEnd(s, start, start)
	return rest + len(trimEnclosingClosers(s[rest:max(end, rest)], open))
}

// startsLine reports whether the value at s[start:] starts a line, that is,
// whether the whitespace between it and the end of the delimiter before it,
// at delim, holds a line break.
func startsLine(s string, delim, start int) bool {
	return strings.ContainsAny(s[delim:start], "\r\n")
}

// maxNestedKeyLen is the length of the longest value that nestsSensitiveKey
// may take for a key. A longer value is redacted, so that it is not matched
// again from each key in it.
const maxNestedKeyLen = 128

// nestedKeyKind tells how redactTextKeyValue treats a value that may be the
// name of a nested key-value pair (see nestsSensitiveKey).
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
)

// nestsSensitiveKey tells whether the value matched at s[start:end] is a
// sensitive key rather than a value, as in the YAML "credentials:\n  token:
// ..." or the text "credentials: token = ...", and if so returns nestedKey
// and where the key name in the value starts. That is the case only when
// the whole value is a sensitive key name that ends with a delimiter or is
// followed by one, and a value follows the delimiter, so that the pair that
// the name starts is matched and redacted on its own; a quoted name must be
// followed by a delimiter. A value made of such a name and a delimiter
// followed by the start of a quoted value that runs on past it, as in
// credentials: password="a b", is a key too. So is a value made of such a
// name, a delimiter and a redacted value, as in "token:[REDACTED]", so that
// scrubbing redacted text again changes nothing. Any other value, such as
// the quoted 'token: abc def', abc-token:def, or an abc-token: with no value
// after it, is not a key (notNestedKey), and neither is a value that opens a
// JSON object or array, so that a credentials envelope written that way is
// redacted as a whole. Only the value, if it is at most maxNestedKeyLen bytes
// long, and the whitespace, the delimiter and the start of the next value
// after it, up to the next whitespace, ",", ";" or "&", or to the character
// after the closing quote of a quoted value that starts in the value or after
// it, are read.
//
// The name is a sensitive key name when it is itself one (see
// nestedKeyNameRe). A name that only ends in one (see compoundKeyNameRe),
// such as X-Amz-Security-Token or db/password, is one only when the value
// starts a line, as a nested key does in YAML: lineStart reports that. On
// the line of the key before it, such a value may just as well be a
// credential that ends in a key name, as in the Go error
// "password=dev-SECRET-password: dial tcp ...", so nestsSensitiveKey then
// returns valueEndingInKeyName, and where the key name in it starts, so that
// the value is redacted together with the value of the pair it starts.
func nestsSensitiveKey(s string, start, end int, lineStart bool) (nestedKeyKind, int) {
	if end-start > maxNestedKeyLen || s[start] == '{' || s[start] == '[' {
		return notNestedKey, 0
	}
	v := s[start:end]
	if inner, quote := unquoteValue(v); quote != "" {
		if !valueFollowsDelimiter(s, end) {
			return notNestedKey, 0
		}
		return nestedKeyName(inner, start+len(quote), lineStart)
	}
	if strings.HasSuffix(v, ":") || strings.HasSuffix(v, "=") {
		if !valueStartsAt(s, end) {
			return notNestedKey, 0
		}
		return nestedKeyName(v[:len(v)-1], start, lineStart)
	}
	if followedByDelimiter(s, end) {
		if !valueFollowsDelimiter(s, end) {
			return notNestedKey, 0
		}
		return nestedKeyName(v, start, lineStart)
	}
	// A name and a delimiter followed by a quoted value that holds
	// whitespace, ",", ";" or "&", as in credentials: password="a b", where
	// the value matched ends inside the quoted value.
	if q := strings.IndexAny(v, `"'\`); q > 0 && (v[q-1] == ':' || v[q-1] == '=') &&
		len(quotedValueRe.FindString(s[start+q:])) > end-start-q && valueStartsAt(s, start+q) {
		if kind, pair := nestedKeyName(v[:q-1], start, lineStart); kind != notNestedKey {
			return kind, pair
		}
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

// nestedKeyName classifies a value that may start a nested pair (see
// nestsSensitiveKey): name is the name of that pair, which begins at offset
// at of the text, and lineStart reports whether the value starts a line.
// Unless the value is notNestedKey, nestedKeyName also returns where the
// sensitive key name in name, and so the pair that it starts, begins in the
// text.
func nestedKeyName(name string, at int, lineStart bool) (nestedKeyKind, int) {
	if _, ok := sensitiveKeys[name]; ok {
		return nestedKey, at
	}
	if m := nestedKeyNameRe.FindStringSubmatchIndex(name); m != nil {
		return nestedKey, at + m[2]
	}
	m := compoundKeyNameRe.FindStringSubmatchIndex(name)
	switch {
	case m == nil:
		return notNestedKey, 0
	case lineStart:
		return nestedKey, at + m[2]
	default:
		return valueEndingInKeyName, at + m[2]
	}
}

// nestedPairEnd returns where the text to redact with a value that only ends
// in a key name ends (see nestsSensitiveKey), given pair, where the pair that
// the key name starts begins in s: at the end of the value of that pair, or,
// if that value is itself the name of a pair, as in "password=a-token: token:
// x", at the end of the value of that pair, and so on. open is the number of
// brackets opened before the key of the value.
func nestedPairEnd(s string, pair, open int) int {
	for {
		loc := textKeyValTokenAtRe.FindStringSubmatchIndex(s[pair:])
		if loc == nil {
			// Not reached: nestsSensitiveKey has read a delimiter and a
			// value after the key name.
			return pair
		}
		start, end := pair+loc[8], pair+loc[9]
		kind, next := nestsSensitiveKey(s, start, end, startsLine(s, pair+loc[7], start))
		if kind == notNestedKey {
			return textValueEnd(s, start, end, open)
		}
		pair = next
	}
}

// regexpSpace holds the bytes that \s matches in a regular expression such
// as textKeyValTokenRe.
const regexpSpace = "\t\n\f\r "

// delimSpace holds the bytes that [\s[:cntrl:]] matches around delimiters in
// textKeyValTokenRe.
const delimSpace = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\v\f\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f \x7f"

// followedByDelimiter reports whether a key-value delimiter, ":" or "=",
// follows s[:i], possibly after whitespace.
func followedByDelimiter(s string, i int) bool {
	rest := strings.TrimLeft(s[i:], delimSpace)
	return rest != "" && (rest[0] == ':' || rest[0] == '=')
}

// valueFollowsDelimiter reports whether a key-value delimiter follows s[:i],
// possibly after whitespace, and a value starts after the delimiter (see
// valueStartsAt).
func valueFollowsDelimiter(s string, i int) bool {
	rest := strings.TrimLeft(s[i:], delimSpace)
	return rest != "" && (rest[0] == ':' || rest[0] == '=') && valueStartsAt(rest, 1)
}

// valueStartsAt reports whether a value that can be redacted starts in s at
// i, possibly after whitespace. The value read is a quoted value (see
// quotedValueRegex) that ends there (see quotedValueEnds), which runs on
// past any whitespace in it to its closing quote, or else the first run of
// bytes, up to the next whitespace, ",", ";" or "&". It must hold a byte
// other than whitespace and the closing brackets, quotes and punctuation
// that may end a value (see trimEnclosingClosers), unlike the "}" that
// closes the struct in the text {Password:abc-token: }, the unclosed quote
// in {Password:abc-token: "} or the empty quotes in {Password:abc-token: ""}.
func valueStartsAt(s string, i int) bool {
	rest := strings.TrimLeft(s[i:], delimSpace)
	if rest != "" && (rest[0] == '"' || rest[0] == '\'' || rest[0] == '\\') {
		if q := quotedValueRe.FindString(rest); q != "" && quotedValueEnds(rest, len(q)) {
			rest, _ = unquoteValue(q)
			return strings.Trim(rest, delimSpace+closingPunctuation) != ""
		}
	}
	if n := strings.IndexAny(rest, regexpSpace+",;&"); n >= 0 {
		rest = rest[:n]
	}
	return strings.Trim(rest, delimSpace+closingPunctuation) != ""
}

// quotedValueEnds reports whether the quoted value s[:n] at the start of s
// ends there: s ends after it, or goes on with a character that is not a
// letter, a digit, "_" or the quote that the value starts with. Otherwise
// its quotes close one string and open the next rather than enclose a
// value, as do the quotes around " for user " in the text
// "password=abc-token: " for user "bob"".
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

// openBrackets counts the brackets, "{" or "[", that are open in a text read
// from its start. A closing bracket with none open is ignored.
type openBrackets struct {
	pos   int // s[:pos] has been read or skipped
	depth int
}

// before reads s up to i and returns the number of brackets open there.
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

// skip moves past s[:i] without reading it, as for a redacted value, so that
// scrubbing redacted text again changes nothing.
func (b *openBrackets) skip(i int) {
	b.pos = max(b.pos, i)
}

// closingPunctuation holds the closing brackets, closing quotes and
// punctuation that may follow the text of a bare value, as in the %q output
// "{APIKey:...}".
const closingPunctuation = `}])"'\:.`

// trimEnclosingClosers returns v, the text of a value, without the "}" and
// "]" at its end that close brackets opened before it, at most open of them,
// and without the punctuation after them, such as the closing quote in the
// %q output "{APIKey:...}". A "}" or "]" followed by anything else, as in
// "ab}rest", or one that closes a bracket opened in v, stays in v.
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
	// The first depth closers close brackets opened in v; at most open of
	// the others close brackets opened before it.
	if j := max(depth, len(closers)-open); j < len(closers) {
		return v[:closers[j]]
	}
	return v
}

// sensitiveValueEnd returns the end of the value matched at s[start:end]. A
// value that opens a JSON object or array, such as the one in
// {"token": {"value": "..."}}, is extended to its closing bracket, or to the
// end of s if it is not closed, so that the fields nested in it are redacted
// together with it. Brackets inside strings do not count, including strings
// delimited by escaped quotes, as in JSON text formatted with %q.
func sensitiveValueEnd(s string, start, end int) int {
	if c := s[start]; c != '{' && c != '[' {
		return end
	}
	depth := 0
	// quote is the quote of the string being skipped. escaped reports that
	// the string was opened with an escaped quote (\" or \'); only the same
	// escaped quote closes it (see escapedQuotedRegex).
	var quote byte
	escaped := false
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
				// An escape of the inner text. The escaped character may
				// itself be escaped, as in \\\" or \\\\.
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
			depth--
			if depth == 0 {
				return max(end, i+1)
			}
		}
	}
	return len(s)
}

// unquoteValue strips one pair of quotes (", ', \" or \') from a matched
// value and returns the content and the quote.
func unquoteValue(v string) (inner, quote string) {
	for _, q := range []string{`\"`, `\'`, `"`, `'`} {
		if len(v) >= 2*len(q) && strings.HasPrefix(v, q) && strings.HasSuffix(v, q) {
			return v[len(q) : len(v)-len(q)], q
		}
	}
	return v, ""
}

// isRedactedMarker reports whether v, possibly percent-encoded, is a
// placeholder written by an earlier redaction pass.
func isRedactedMarker(v string) bool {
	if isKnownSentinel(v) {
		return true
	}
	if decoded, err := url.PathUnescape(v); err == nil && decoded != v {
		return isKnownSentinel(decoded)
	}
	return false
}

// isRedactedValue reports whether v, a matched value without its quotes, is
// a placeholder written by an earlier redaction pass, possibly followed by
// punctuation that ends the sentence, as in "[REDACTED]." or "[REDACTED])".
// Decoding a redacted URI can leave a placeholder next to such punctuation,
// which is not part of a credential.
func isRedactedValue(v string) bool {
	return isRedactedMarker(v) || isRedactedMarker(trimTrailingPunctuation(v))
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
