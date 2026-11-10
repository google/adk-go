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
// "passphrase", "secret_access_key", "secret_key", "session_token",
// "signing_key", "x_auth_token" and "x_goog_api_key". A value under "credential" or "credentials" in a map, in
// JSON text, in a Python dict or in a Go struct printed with %+v is redacted
// as a whole, so that no field of a credentials envelope, such as the AWS STS
// result {"Credentials": {"AccessKeyId": ..., "SecretAccessKey": ...,
// "SessionToken": ...}}, is logged, whatever its name. In YAML or other
// key: value text, the fields under "credentials:" are not redacted as a
// whole: each is redacted only if its own name is sensitive (see
// nestsSensitiveKey). isSensitiveKey and the text patterns also match these
// names without separators, as in the camelCase apiKey or the Go field name
// APIKey, and at the end of a compound name, as in DB_PASSWORD, dbPassword
// or GITHUB_TOKEN (see isCompoundSensitiveKey).
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
	"secret_key":            {},
	"session_token":         {},
	"sig":                   {},
	"signature":             {},
	"signing_key":           {},
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
	uriTokenRe = regexp.MustCompile(`[a-zA-Z][a-zA-Z0-9+.-]*://[^\s"'<>]+`)
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
)

// bareFixedSensitiveKeys holds every variant of the names in sensitiveKeys
// where each "_" is kept as "_", replaced by "-" or ".", or removed.
var bareFixedSensitiveKeys = func() map[string]struct{} {
	m := make(map[string]struct{}, 512)
	for k := range sensitiveKeys {
		parts := strings.Split(k, "_")
		variants := []string{parts[0]}
		for _, p := range parts[1:] {
			next := make([]string, 0, len(variants)*4)
			for _, v := range variants {
				next = append(next, v+"_"+p, v+"-"+p, v+"."+p, v+p)
			}
			variants = next
		}
		for _, v := range variants {
			m[v] = struct{}{}
		}
	}
	return m
}()

// patternWords holds words of which every match of redactAuthHeaders,
// redactBearerTokens, redactBasicCredentials, redactJSONKeyValues,
// redactTextKeyValues and hasQuotedSensitiveKey contains at least one, once
// the text is folded by foldPatternText: the sensitive key names without
// separators (which include "authorization", "apikey" and "bearer"), the
// "temp:" prefix and the "basic" scheme. A name that contains another, such as
// "accesstoken", which contains "token", is left out, since the text then
// contains the shorter name too.
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
// ignoring case. When it does not, none of the credential text scanners can
// match s, so scrubTextPatterns and hasQuotedSensitiveKey need not scan it.
func mayMatchTextPatterns(s string) bool {
	folded := foldPatternText(s)
	for _, w := range patternWords {
		if strings.Contains(folded, w) {
			return true
		}
	}
	return false
}

// foldPatternText lowercases the ASCII letters of s and removes "-", "_" and ".",
// as the scanners accept key names with any of those separators or none
// (api_key, api-key, api.key, apiKey). It also folds the three non-ASCII
// letters that Go case conversion or RE2 (?i) maps onto ASCII letters: U+0130
// (dotted capital I -> "i"), U+017F (long s -> "s") and U+212A (Kelvin sign ->
// "k").
func foldPatternText(s string) string {
	needsFold := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if ('A' <= c && c <= 'Z') || c == '-' || c == '_' || c == '.' || c >= utf8.RuneSelf {
			needsFold = true
			break
		}
	}
	if !needsFold {
		return s
	}
	b := make([]byte, 0, len(s))
	for i := 0; i < len(s); {
		c := s[i]
		if c < utf8.RuneSelf {
			switch {
			case 'A' <= c && c <= 'Z':
				b = append(b, c+'a'-'A')
			case c != '-' && c != '_' && c != '.':
				b = append(b, c)
			}
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		switch r {
		case '\u0130':
			b = append(b, 'i')
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

// hasQuotedSensitiveKey reports whether s contains a quoted sensitive key
// followed by optional spaces and a key-value delimiter (":" or "=").
func hasQuotedSensitiveKey(s string) bool {
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
			if isValidJSONPatternKey(inner) && followedByKeyDelimiter(s[i+1:]) {
				return true
			}
		}
		prevQuote = i
	}
	return false
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
// in which each "_" may also be written as "-" or ".", or left out, so that
// api_key, api-key, api.key, apiKey and APIKey all match.
func fixedSensitiveKeyRegex() string {
	parts := make([]string, 0, len(sensitiveKeys))
	for _, k := range slices.Sorted(maps.Keys(sensitiveKeys)) {
		parts = append(parts, strings.ReplaceAll(regexp.QuoteMeta(k), "_", "[._-]?"))
	}
	return strings.Join(parts, "|")
}

func normalizeKey(k string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(k)), "-", "_")
}

// removeKeySeparators returns k without its "_", "-" and "." separators.
func removeKeySeparators(k string) string {
	if !strings.ContainsAny(k, "_-.") {
		return k
	}
	return strings.Map(func(r rune) rune {
		if r == '_' || r == '-' || r == '.' {
			return -1
		}
		return r
	}, k)
}

var validSensitiveLast2 = func() [128][128]bool {
	var table [128][128]bool
	for name := range sensitiveKeyNames {
		if len(name) >= 2 && name[len(name)-2] < 128 && name[len(name)-1] < 128 {
			table[name[len(name)-2]][name[len(name)-1]] = true
		}
	}
	return table
}()

var compoundKeySuffixesByLast2 = func() map[uint16][]string {
	m := make(map[uint16][]string, len(sensitiveKeyNames))
	for name := range sensitiveKeyNames {
		if _, low := lowPrecisionKeySuffixes[name]; low || len(name) < 2 {
			continue
		}
		code := uint16(name[len(name)-2])<<8 | uint16(name[len(name)-1])
		m[code] = append(m[code], name)
	}
	return m
}()

func last2FoldedKeyBytes(k string) (byte, byte, bool) {
	var b [2]byte
	found := 0
	for i := len(k); i > 0 && found < 2; {
		c := k[i-1]
		if c < utf8.RuneSelf {
			switch {
			case c == '_' || c == '-' || c == '.':
				i--
				continue
			case 'A' <= c && c <= 'Z':
				c += 'a' - 'A'
			case ('a' <= c && c <= 'z') || ('0' <= c && c <= '9'):
			default:
				return 0, 0, false
			}
			b[found] = c
			found++
			i--
			continue
		}
		r, size := utf8.DecodeLastRuneInString(k[:i])
		var fc byte
		switch r {
		case '\u0130':
			fc = 'i'
		case '\u017f':
			fc = 's'
		case '\u212a':
			fc = 'k'
		default:
			return 0, 0, false
		}
		b[found] = fc
		found++
		i -= size
	}
	if found < 2 {
		return 0, 0, false
	}
	return b[1], b[0], true
}

func hasCompoundWordBoundary(k string) bool {
	for i := 0; i < len(k); {
		c := k[i]
		if c < utf8.RuneSelf {
			if c == '_' || c == '-' || c == '.' || ('A' <= c && c <= 'Z') {
				return true
			}
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(k[i:])
		if r == '\u0130' || r == '\u212a' {
			return true
		}
		i += size
	}
	return false
}

// isSensitiveKey reports whether the value stored under k must be redacted:
// k names a credential, or it is a session "temp:" key, which holds
// invocation-scoped scratch data. Credential names are compared ignoring
// case, the "_", "-" and "." separators, and the "user:" and "app:" state
// prefixes, so that api_key, api-key, api.key, apiKey, the Go field name
// APIKey and the state key user:api_key all match. A "temp:" key is one after
// a state prefix too, as user:temp:access_token. A compound name that ends in
// a credential name, as an environment variable or a prefixed field does, such
// as DB_PASSWORD, GITHUB_TOKEN, dbPassword or spring.datasource.password, is
// sensitive too (see isCompoundSensitiveKey).
func isSensitiveKey(k string) bool {
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
	if _, ok := sensitiveKeyNames[removeKeySeparators(foldKeyASCII(k))]; ok {
		return true
	}
	return isCompoundSensitiveKey(k)
}

// foldKeyASCII lowercases ASCII letters in k and folds U+0130 ("i"),
// U+017F ("s") and U+212A ("k").
func foldKeyASCII(k string) string {
	hasUpperOrNonASCII := false
	for i := 0; i < len(k); i++ {
		c := k[i]
		if ('A' <= c && c <= 'Z') || c >= utf8.RuneSelf {
			hasUpperOrNonASCII = true
			break
		}
	}
	if !hasUpperOrNonASCII {
		return k
	}
	b := make([]byte, 0, len(k))
	for i := 0; i < len(k); {
		c := k[i]
		if c < utf8.RuneSelf {
			if 'A' <= c && c <= 'Z' {
				c += 'a' - 'A'
			}
			b = append(b, c)
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(k[i:])
		switch r {
		case '\u0130':
			b = append(b, 'i')
		case '\u017f':
			b = append(b, 's')
		case '\u212a':
			b = append(b, 'k')
		default:
			b = append(b, k[i:i+size]...)
		}
		i += size
	}
	return string(b)
}

// hasFoldPrefix reports whether s starts with prefix, ignoring ASCII case as
// well as U+0130 ("i"), U+017F ("s") and U+212A ("k").
func hasFoldPrefix(s, prefix string) bool {
	if len(s) < len(prefix) {
		return false
	}
	hasNonASCII := false
	for i := 0; i < len(prefix); i++ {
		if s[i] >= utf8.RuneSelf {
			hasNonASCII = true
			break
		}
	}
	if !hasNonASCII {
		return strings.EqualFold(s[:len(prefix)], prefix)
	}
	si := 0
	for pi := 0; pi < len(prefix); pi++ {
		if si >= len(s) {
			return false
		}
		c := s[si]
		var fc byte
		if c < utf8.RuneSelf {
			if 'A' <= c && c <= 'Z' {
				c += 'a' - 'A'
			}
			fc = c
			si++
		} else {
			r, size := utf8.DecodeRuneInString(s[si:])
			switch r {
			case '\u0130':
				fc = 'i'
			case '\u017f':
				fc = 's'
			case '\u212a':
				fc = 'k'
			default:
				return false
			}
			si += size
		}
		pc := prefix[pi]
		if 'A' <= pc && pc <= 'Z' {
			pc += 'a' - 'A'
		}
		if fc != pc {
			return false
		}
	}
	return true
}

// lowPrecisionKeySuffixes are the credential names that do not make a
// compound key sensitive when they end it, because they are ordinary words in
// a payload: a method_signature or a request_sig is not a credential. Neither
// do "credential" and "credentials": a key such as exchangedAuthCredential
// names a structure whose fields are redacted one by one, not a value to
// redact whole.
var lowPrecisionKeySuffixes = map[string]struct{}{
	"bearer":      {},
	"credential":  {},
	"credentials": {},
	"sig":         {},
	"signature":   {},
}

// isCompoundSensitiveKey reports whether k, a name made of ASCII letters,
// digits, "_", "-" and ".", ends in the words of a sensitive key name, read
// at its separators and camelCase boundaries (see splitCompoundKey):
// DB_PASSWORD, dbPassword and spring.datasource.password end in "password",
// and OPENAI_API_KEY and myAPIKey end in "api_key". A name that merely
// contains a credential name, such as token_count, prompt_tokens or
// secret_manager_project, is not sensitive, and neither is a pagination or
// sync cursor such as page_token, next_page_token, nextPageToken, sync_token
// or continuation_token (see isPageTokenKey).
func isCompoundSensitiveKey(k string) bool {
	if !hasCompoundWordBoundary(k) {
		return false
	}
	b0, b1, ok := last2FoldedKeyBytes(k)
	if !ok {
		return false
	}
	candidates := compoundKeySuffixesByLast2[uint16(b0)<<8|uint16(b1)]
	if len(candidates) == 0 {
		return false
	}
	compact, starts := splitCompoundKey(k)
	for _, name := range candidates {
		i := len(compact) - len(name)
		if i <= 0 || !starts[i] || compact[i:] != name {
			continue
		}
		if name == "token" && isPageTokenKey(compact[:i], starts) {
			continue
		}
		return true
	}
	return false
}

// splitCompoundKey returns k in lower case without its "_", "-" and "."
// separators, and, for each byte of that text, whether a word of k starts
// there: after a separator, and at an upper-case letter, which follows a
// separator, a lower-case letter or a digit (the "P" of dbPassword) or
// another upper-case letter, where an acronym ends (the "K" of APIKey) or,
// between two acronyms, a word starts (the "API" of OpenAIAPIKey). It returns
// "" when k holds other characters, such as the "/" of a path or the space of
// prose.
func splitCompoundKey(k string) (compact string, starts []bool) {
	b := make([]byte, 0, len(k))
	starts = make([]bool, 0, len(k))
	sep := true
	for i := 0; i < len(k); {
		c := k[i]
		if c < utf8.RuneSelf {
			switch {
			case c == '_' || c == '-' || c == '.':
				sep = true
				i++
				continue
			case 'A' <= c && c <= 'Z':
				starts = append(starts, true)
				c += 'a' - 'A'
			case 'a' <= c && c <= 'z', '0' <= c && c <= '9':
				starts = append(starts, sep)
			default:
				return "", nil
			}
			b = append(b, c)
			sep = false
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(k[i:])
		switch r {
		case '\u0130':
			b = append(b, 'i')
			starts = append(starts, true)
		case '\u017f':
			b = append(b, 's')
			starts = append(starts, sep)
		case '\u212a':
			b = append(b, 'k')
			starts = append(starts, true)
		default:
			return "", nil
		}
		sep = false
		i += size
	}
	return string(b), starts
}

// isPageTokenKey reports whether before, the words of a compound key before
// its last word "token" (see splitCompoundKey), make the key a pagination,
// sync or padding token rather than a credential: "next", "prev", "previous"
// or "pad" directly before "token" (as in next_token, NextToken, prev_token or
// pad_token), or "page", "sync" or "continuation" after no other word or after
// "next", "prev" or "previous" (as in page_token, next_page_token,
// nextPageToken, sync_token, syncToken, nextSyncToken, continuation_token or
// continuationToken). A key whose other words differ, such as fb_page_token,
// the access token of a Facebook Page, is not a cursor.
func isPageTokenKey(before string, starts []bool) bool {
	switch before {
	case "next", "prev", "previous", "pad":
		return true
	}
	for _, cursor := range []string{"page", "sync", "continuation"} {
		i := len(before) - len(cursor)
		if i < 0 || !starts[i] || before[i:] != cursor {
			continue
		}
		switch before[:i] {
		case "", "next", "prev", "previous":
			return true
		}
	}
	return false
}

func isHexByte(c byte) bool {
	return ('0' <= c && c <= '9') || ('a' <= c && c <= 'f') || ('A' <= c && c <= 'F')
}

func hasCanonicalizableEscape(s string) bool {
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '%':
			if i+2 < len(s) && isHexByte(s[i+1]) && isHexByte(s[i+2]) {
				return true
			}
		case '\\':
			if i+1 < len(s) {
				switch s[i+1] {
				case 'u', 'U', 'x', 'X', '"', '\'':
					return true
				}
			}
		}
	}
	return false
}

// canonicalizeCredentialMarker decodes nested ASCII \u, \x and percent
// escapes, and escaped quotes (\" and \'), in one pass. The result is used
// for detection only: callers keep the original text unless the canonical
// form exposes a credential construct.
func canonicalizeCredentialMarker(s string) string {
	if !hasCanonicalizableEscape(s) {
		return s
	}
	output := make([]byte, 0, len(s))
	changed := false
	for i := 0; i < len(s); i++ {
		output = append(output, s[i])
		for {
			n := len(output)
			if n >= 2 && output[n-2] == '\\' && (output[n-1] == '"' || output[n-1] == '\'') {
				output = append(output[:n-2], output[n-1])
				changed = true
				continue
			}
			escapeStart := -1
			var digits []byte
			if n >= 6 && output[n-6] == '\\' && (output[n-5] == 'u' || output[n-5] == 'U') &&
				isHexByte(output[n-4]) && isHexByte(output[n-3]) && isHexByte(output[n-2]) && isHexByte(output[n-1]) {
				escapeStart = n - 6
				digits = output[n-4:]
			} else if n >= 4 && output[n-4] == '\\' && (output[n-3] == 'x' || output[n-3] == 'X') &&
				isHexByte(output[n-2]) && isHexByte(output[n-1]) {
				escapeStart = n - 4
				digits = output[n-2:]
			} else if n >= 3 && output[n-3] == '%' &&
				isHexByte(output[n-2]) && isHexByte(output[n-1]) {
				escapeStart = n - 3
				digits = output[n-2:]
			}
			if escapeStart < 0 {
				break
			}
			decoded, err := strconv.ParseUint(string(digits), 16, 32)
			if err != nil || decoded > 0x7F {
				break
			}
			output = append(output[:escapeStart], byte(decoded))
			changed = true
		}
	}
	if !changed {
		return s
	}
	return string(output)
}

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
// sensitive key, starts with a key-value delimiter (":" or "=") after optional
// whitespace and is not immediately followed by an already-redacted
// placeholder value.
func followedByKeyDelimiter(rest string) bool {
	for i, r := range rest {
		if unicode.IsSpace(r) || unicode.In(r, unicode.Z) {
			continue
		}
		if r != ':' && r != '=' {
			return false
		}
		return !startsWithRedactedValue(rest[i+1:])
	}
	return false
}

// startsWithRedactedValue reports whether afterDelim, the text after a ":" or
// "=" delimiter, begins (after optional whitespace) with a complete "[REDACTED]"
// placeholder token rather than a secret that merely shares its prefix.
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
			return quotedValueEnds(val, len(lit))
		}
	}
	if strings.HasPrefix(val, redactedValue) {
		tail := val[len(redactedValue):]
		if tail == "" {
			return true
		}
		c := tail[0]
		return isTextValueStop(c) || c == '}' || c == ']' || c == ')' || c == '.' || c == ':'
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
	hasColonOrEq := strings.ContainsAny(s, ":=")
	if hasColonOrEq && strings.Contains(s, ":") {
		s = redactAuthHeaders(s)
	}
	if strings.ContainsAny(s, "bB") {
		s = redactBearerTokens(s)
		s = redactBasicCredentials(s)
	}
	if hasColonOrEq && strings.Contains(s, `"`) && strings.Contains(s, ":") {
		s = redactJSONKeyValues(s)
	}
	if strings.ContainsAny(s, "=:") {
		s = redactTextKeyValues(s)
	}
	return s
}

type quoteScanState struct {
	unclosedDoubleFrom  int
	unclosedDoubleUntil int
	unclosedSingleFrom  int
	unclosedSingleUntil int
	failedEscDoubleFrom int
	failedEscDoublePos  int
	failedEscSingleFrom int
	failedEscSinglePos  int
}

func newQuoteScanState() quoteScanState {
	return quoteScanState{
		unclosedDoubleUntil: -1,
		unclosedSingleUntil: -1,
		failedEscDoublePos:  -1,
		failedEscSinglePos:  -1,
	}
}

func isRegexpSpace(c byte) bool {
	return c == '\t' || c == '\n' || c == '\f' || c == '\r' || c == ' '
}

func isDelimSpace(c byte) bool {
	return c <= 0x20 || c == 0x7f
}

func isTextValueStop(c byte) bool {
	return isRegexpSpace(c) || c == ',' || c == ';' || c == '&'
}

func isJSONValueStop(c byte) bool {
	return isRegexpSpace(c) || c == ',' || c == '}'
}

func isBearerBareStop(c byte) bool {
	return isRegexpSpace(c) || c == ',' || c == ';' || c == ')' || c == '"' || c == '\'' || c == ']'
}

func isBearerBracketTailStop(c byte) bool {
	return isRegexpSpace(c) || c == ',' || c == ';' || c == ')' || c == '"' || c == '\''
}

func isTempKeyBodyByte(c byte) bool {
	switch c {
	case '=', ':', '"', '\'', ' ', '\t', '\r', '\n', '\f', ',', '}':
		return false
	default:
		return true
	}
}

func trimInlineSpaces(s string) string {
	return strings.TrimFunc(s, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.In(r, unicode.Z)
	})
}

func skipInlineSpacesFwd(s string, pos, limit int) int {
	for pos < limit {
		r, size := utf8.DecodeRuneInString(s[pos:limit])
		if size == 0 || (!unicode.IsSpace(r) && !unicode.In(r, unicode.Z)) {
			break
		}
		pos += size
	}
	return pos
}

// recordFailedQuoteSpan updates [from, until) with a failed quote scan from
// start to failEnd so forward scans inside [from, until) short-circuit in O(1)
// while backward re-scans starting before from are never falsely rejected.
func recordFailedQuoteSpan(from, until *int, start, failEnd int) {
	if *until < 0 || start > *until {
		*from, *until = start, failEnd
		return
	}
	if start >= *from && failEnd > *until {
		*until = failEnd
	}
}

func scanSimpleQuoted(s string, start int, q byte, st *quoteScanState) (int, bool) {
	if start >= len(s) || s[start] != q {
		return 0, false
	}
	if st != nil {
		if q == '"' && start >= st.unclosedDoubleFrom && start < st.unclosedDoubleUntil {
			return 0, false
		}
		if q == '\'' && start >= st.unclosedSingleFrom && start < st.unclosedSingleUntil {
			return 0, false
		}
	}
	for i := start + 1; i < len(s); i++ {
		c := s[i]
		if c == '\\' {
			if i+1 >= len(s) || s[i+1] == '\n' {
				if st != nil {
					if q == '"' {
						recordFailedQuoteSpan(&st.unclosedDoubleFrom, &st.unclosedDoubleUntil, start, i)
					} else {
						recordFailedQuoteSpan(&st.unclosedSingleFrom, &st.unclosedSingleUntil, start, i)
					}
				}
				return 0, false
			}
			i++
			continue
		}
		if c == q {
			return i + 1, true
		}
	}
	if st != nil {
		if q == '"' {
			recordFailedQuoteSpan(&st.unclosedDoubleFrom, &st.unclosedDoubleUntil, start, len(s))
		} else {
			recordFailedQuoteSpan(&st.unclosedSingleFrom, &st.unclosedSingleUntil, start, len(s))
		}
	}
	return 0, false
}

func scanEscapedQuoted(s string, start int, q byte, st *quoteScanState) (int, bool) {
	if start+1 >= len(s) || s[start] != '\\' || s[start+1] != q {
		return 0, false
	}
	if st != nil {
		if q == '"' && start >= st.failedEscDoubleFrom && start+2 <= st.failedEscDoublePos {
			return 0, false
		}
		if q == '\'' && start >= st.failedEscSingleFrom && start+2 <= st.failedEscSinglePos {
			return 0, false
		}
	}
	recordFail := func(failRunStart int) {
		if st == nil {
			return
		}
		if q == '"' {
			recordFailedQuoteSpan(&st.failedEscDoubleFrom, &st.failedEscDoublePos, start, failRunStart)
		} else {
			recordFailedQuoteSpan(&st.failedEscSingleFrom, &st.failedEscSinglePos, start, failRunStart)
		}
	}
	i := start + 2
	for i < len(s) {
		runStart := i
		m := 0
		for i < len(s) && s[i] == '\\' {
			m++
			i++
		}
		if i == len(s) {
			recordFail(len(s))
			return 0, false
		}
		c := s[i]
		i++
		if c == q {
			switch m & 3 {
			case 1:
				return i, true
			case 3:
				continue
			default:
				recordFail(runStart)
				return 0, false
			}
		}
		if m&3 == 3 {
			recordFail(runStart)
			return 0, false
		}
	}
	recordFail(len(s))
	return 0, false
}

func scanQuotedAt(s string, start int, st *quoteScanState) (int, bool) {
	if start >= len(s) {
		return 0, false
	}
	switch s[start] {
	case '"', '\'':
		return scanSimpleQuoted(s, start, s[start], st)
	case '\\':
		if start+1 < len(s) && (s[start+1] == '"' || s[start+1] == '\'') {
			return scanEscapedQuoted(s, start, s[start+1], st)
		}
	}
	return 0, false
}

func scanValueEnd(s string, valStart int, stop func(byte) bool, st *quoteScanState) int {
	if end, ok := scanQuotedAt(s, valStart, st); ok {
		return end
	}
	end := valStart + 1
	for end < len(s) && !stop(s[end]) {
		end++
	}
	return end
}

var authHeaderNames = [...]string{
	"proxy-authorization",
	"proxy_authorization",
	"authorization",
	"x-api-key",
	"x-api_key",
	"x_api-key",
	"x_api_key",
	"api-key",
	"api_key",
}

func matchSuffixFoldBack(s string, floor, hdrEnd int, target string) (int, bool) {
	si := hdrEnd
	for ti := len(target) - 1; ti >= 0; ti-- {
		if si <= floor {
			return 0, false
		}
		c := s[si-1]
		if c < utf8.RuneSelf {
			if 'A' <= c && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c != target[ti] {
				return 0, false
			}
			si--
			continue
		}
		r, size := utf8.DecodeLastRuneInString(s[floor:si])
		var fc byte
		switch r {
		case '\u017f':
			fc = 's'
		case '\u212a':
			fc = 'k'
		default:
			return 0, false
		}
		if fc != target[ti] {
			return 0, false
		}
		si -= size
	}
	if si > 0 && isWordByte(s[si-1]) {
		return 0, false
	}
	return si, true
}

func matchAuthHeaderAtEnd(s string, last, hdrEnd int) int {
	for _, name := range authHeaderNames {
		if start, ok := matchSuffixFoldBack(s, last, hdrEnd, name); ok {
			return start
		}
	}
	return -1
}

func redactAuthHeaders(s string) string {
	var b strings.Builder
	last, pos := 0, 0
	for pos < len(s) {
		rel := strings.IndexByte(s[pos:], ':')
		if rel < 0 {
			break
		}
		colonPos := pos + rel
		hdrEnd := colonPos
		for hdrEnd > last && (s[hdrEnd-1] == ' ' || s[hdrEnd-1] == '\t') {
			hdrEnd--
		}
		hdrStart := matchAuthHeaderAtEnd(s, last, hdrEnd)
		if hdrStart < 0 {
			pos = colonPos + 1
			continue
		}
		valEnd := colonPos + 1
		for valEnd < len(s) && s[valEnd] != '\r' && s[valEnd] != '\n' {
			valEnd++
		}
		valStart := colonPos + 1
		for valStart < valEnd && (s[valStart] == ' ' || s[valStart] == '\t') {
			valStart++
		}
		if valEnd == valStart+len(redactedValue) && s[valStart:valEnd] == redactedValue &&
			colonPos+2 == valStart && s[colonPos+1] == ' ' {
			pos = valEnd
			continue
		}
		b.WriteString(s[last:hdrStart])
		b.WriteString(s[hdrStart:hdrEnd])
		b.WriteString(": ")
		b.WriteString(redactedValue)
		last = valEnd
		pos = valEnd
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

func matchWordFoldASCII(s string, pos int, word string) bool {
	if pos+len(word) > len(s) {
		return false
	}
	if pos > 0 && isWordByte(s[pos-1]) {
		return false
	}
	return strings.EqualFold(s[pos:pos+len(word)], word)
}

func redactBearerTokens(s string) string {
	qState := newQuoteScanState()
	var b strings.Builder
	last, pos := 0, 0
	for pos+7 <= len(s) {
		rel := strings.IndexAny(s[pos:], "bB")
		if rel < 0 {
			break
		}
		i := pos + rel
		if !matchWordFoldASCII(s, i, "bearer") {
			pos = i + 1
			continue
		}
		afterWord := i + len("bearer")
		valStart := afterWord
		for valStart < len(s) && isRegexpSpace(s[valStart]) {
			valStart++
		}
		if valStart == afterWord || valStart >= len(s) {
			pos = afterWord
			continue
		}
		valEnd := scanBearerValueEnd(s, valStart, &qState)
		if valEnd <= valStart {
			pos = afterWord
			continue
		}
		if rest := s[valStart:]; len(rest) >= 2 && strings.EqualFold(rest[:2], "of") {
			if r, _ := utf8.DecodeRuneInString(rest[2:]); len(rest) == 2 || unicode.IsSpace(r) {
				pos = valEnd
				continue
			}
		}
		if inner, _ := unquoteValue(s[valStart:valEnd]); isRedactedValue(inner) {
			pos = valEnd
			continue
		}
		b.WriteString(s[last:i])
		b.WriteString(s[i:afterWord])
		b.WriteByte(' ')
		b.WriteString(redactedValue)
		last = valEnd
		pos = valEnd
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

func scanBearerValueEnd(s string, valStart int, st *quoteScanState) int {
	switch s[valStart] {
	case '"', '\'':
		if end, ok := scanSimpleQuoted(s, valStart, s[valStart], st); ok {
			return end
		}
		return 0
	case '[':
		j := valStart + 1
		for j < len(s) && s[j] != ']' && s[j] != '\r' && s[j] != '\n' && !isRegexpSpace(s[j]) {
			j++
		}
		if j > valStart+1 && j < len(s) && s[j] == ']' {
			j++
			for j < len(s) && !isBearerBracketTailStop(s[j]) {
				j++
			}
			return j
		}
	}
	end := valStart
	for end < len(s) && !isBearerBareStop(s[end]) {
		end++
	}
	return end
}

// maxBasicCredentialBytes is adk-python's _MAX_BASIC_AUTH_TOKEN_CHARS.
const maxBasicCredentialBytes = 16 * 1024

func matchBasicPrefix(s string, i int) (after int, ok bool) {
	if i > 0 && isWordByte(s[i-1]) {
		return 0, false
	}
	if i+5 <= len(s) && strings.EqualFold(s[i:i+5], "basic") {
		return i + 5, true
	}
	// Handle Unicode U+017F (ſ) case fold matching (?i)basic.
	if i+6 <= len(s) && (s[i] == 'b' || s[i] == 'B') && (s[i+1] == 'a' || s[i+1] == 'A') &&
		s[i+2] == '\xc5' && s[i+3] == '\xbf' &&
		(s[i+4] == 'i' || s[i+4] == 'I') && (s[i+5] == 'c' || s[i+5] == 'C') {
		return i + 6, true
	}
	return 0, false
}

func isBase64BodyByte(c byte) bool {
	return ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z') || ('0' <= c && c <= '9') || c == '+' || c == '/'
}

func redactBasicCredentials(s string) string {
	var b strings.Builder
	last, pos := 0, 0
	for pos+6 <= len(s) {
		rel := strings.IndexAny(s[pos:], "bB")
		if rel < 0 {
			break
		}
		i := pos + rel
		afterWord, ok := matchBasicPrefix(s, i)
		if !ok {
			pos = i + 1
			continue
		}
		valStart := afterWord
		for valStart < len(s) && isRegexpSpace(s[valStart]) {
			valStart++
		}
		if valStart == afterWord || valStart >= len(s) {
			pos = afterWord
			continue
		}
		end := valStart
		for end < len(s) && isBase64BodyByte(s[end]) {
			end++
		}
		if end == valStart {
			pos = afterWord
			continue
		}
		for pad := 0; pad < 2 && end < len(s) && s[end] == '='; pad++ {
			end++
		}
		if end < len(s) && isBase64Byte(s[end]) {
			pos = end
			continue
		}
		encoded := s[valStart:end]
		if len(encoded) <= maxBasicCredentialBytes {
			decoded, err := base64.StdEncoding.DecodeString(encoded)
			if err != nil || (!strings.Contains(encoded, "=") && !bytes.Contains(decoded, []byte(":"))) {
				pos = end
				continue
			}
		}
		b.WriteString(s[last:i])
		b.WriteString(s[i:afterWord])
		b.WriteByte(' ')
		b.WriteString(redactedValue)
		last = end
		pos = end
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

func isBase64Byte(c byte) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || '0' <= c && c <= '9' || c == '+' || c == '/' || c == '='
}

func isUnescapedQuote(s string, floor, quotePos int) bool {
	bs := 0
	for p := quotePos; p > floor && s[p-1] == '\\'; p-- {
		bs++
	}
	return bs%2 == 0
}

func isValidJSONPatternKey(key string) bool {
	if key == "" {
		return false
	}
	for {
		if hasFoldPrefix(key, session.KeyPrefixTemp) {
			rest := key[len(session.KeyPrefixTemp):]
			return rest != "" && !strings.ContainsAny(rest, "\"'\t\n\f\r ,}")
		}
		if !hasFoldPrefix(key, session.KeyPrefixUser) && !hasFoldPrefix(key, session.KeyPrefixApp) {
			break
		}
		colon := strings.IndexByte(key, ':')
		key = trimInlineSpaces(key[colon+1:])
	}
	return !strings.ContainsAny(key, "\"'\r\n,:{}") && isSensitiveKey(key)
}

func redactJSONKeyValues(s string) string {
	qState := newQuoteScanState()
	var brackets openBrackets
	var b strings.Builder
	last, pos := 0, 0
	for pos < len(s) {
		rel := strings.IndexByte(s[pos:], ':')
		if rel < 0 {
			break
		}
		colonPos := pos + rel
		p := colonPos
		for p > last && isRegexpSpace(s[p-1]) {
			p--
		}
		if p <= last || s[p-1] != '"' || !isUnescapedQuote(s, last, p-1) {
			pos = colonPos + 1
			continue
		}
		closeQuote := p - 1
		openQuote := -1
		for q := closeQuote - 1; q >= last; q-- {
			c := s[q]
			if c == '"' {
				if isUnescapedQuote(s, last, q) {
					openQuote = q
				}
				break
			}
			if c == '\'' || c == ',' || c == '}' {
				break
			}
		}
		if openQuote < 0 {
			pos = colonPos + 1
			continue
		}
		keyStart := skipInlineSpacesFwd(s, openQuote+1, closeQuote)
		keyEnd := skipInlineSpacesBack(s, keyStart, closeQuote)
		key := s[keyStart:keyEnd]
		if !isValidJSONPatternKey(key) {
			pos = colonPos + 1
			continue
		}
		valStart := colonPos + 1
		for valStart < len(s) && isRegexpSpace(s[valStart]) {
			valStart++
		}
		if valStart >= len(s) || isJSONValueStop(s[valStart]) {
			pos = colonPos + 1
			continue
		}
		end := scanValueEnd(s, valStart, isJSONValueStop, &qState)
		open := brackets.before(s, openQuote)
		switch kind, pair := nestsSensitiveKey(s, valStart, end, startsLine(s, colonPos+1, valStart), &qState); kind {
		case nestedKey:
			pos = valStart
			continue
		case valueEndingInKeyName:
			end = max(end, nestedPairEndWithState(s, pair, open, &qState))
			brackets.skip(end)
			b.WriteString(s[last:openQuote])
			b.WriteByte('"')
			b.WriteString(key)
			b.WriteString(`":"`)
			b.WriteString(redactedValue)
			b.WriteByte('"')
			last = end
			pos = end
			continue
		}
		end = sensitiveValueEnd(s, valStart, end)
		brackets.skip(end)
		if inner, _ := unquoteValue(s[valStart:end]); isRedactedValue(inner) {
			pos = end
			continue
		}
		b.WriteString(s[last:openQuote])
		b.WriteByte('"')
		b.WriteString(key)
		b.WriteString(`":"`)
		b.WriteString(redactedValue)
		b.WriteByte('"')
		last = end
		pos = end
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

func isCompoundKeyRuneBack(s string, floor, pos int) int {
	if pos <= floor {
		return 0
	}
	c := s[pos-1]
	if c < utf8.RuneSelf {
		if isCompoundKeyByte(c) {
			return 1
		}
		return 0
	}
	r, size := utf8.DecodeLastRuneInString(s[floor:pos])
	if r == '\u0130' || r == '\u017f' || r == '\u212a' {
		return size
	}
	return 0
}

func findTextKeyStart(s string, last, keyEnd int) int {
	// Shape A: temp:<non-delimiter/non-space> key preceded by a word boundary.
	tk := keyEnd
	for tk > last && isTempKeyBodyByte(s[tk-1]) {
		tk--
	}
	if tk < keyEnd && tk-len(session.KeyPrefixTemp) >= last &&
		strings.EqualFold(s[tk-len(session.KeyPrefixTemp):tk], session.KeyPrefixTemp) {
		tempStart := tk - len(session.KeyPrefixTemp)
		if tempStart == 0 || !isWordByte(s[tempStart-1]) {
			return tempStart
		}
	}
	// Shape B: compound or fixed sensitive key made of compound key runes.
	fullStart := keyEnd
	for fullStart > last {
		step := isCompoundKeyRuneBack(s, last, fullStart)
		if step == 0 {
			break
		}
		fullStart -= step
	}
	if fullStart >= keyEnd {
		return -1
	}
	return sensitiveKeyStart(s, fullStart, keyEnd)
}

func redactTextKeyValues(s string) string {
	qState := newQuoteScanState()
	var brackets openBrackets
	var b strings.Builder
	last, pos := 0, 0
	for pos < len(s) {
		rel := strings.IndexAny(s[pos:], "=:")
		if rel < 0 {
			break
		}
		delimPos := pos + rel
		valStart := delimPos + 1
		for valStart < len(s) && isDelimSpace(s[valStart]) {
			valStart++
		}
		if (valStart >= len(s) || isTextValueStop(s[valStart])) && valStart > delimPos+1 && !isTextValueStop(s[valStart-1]) {
			valStart--
		}
		if valStart >= len(s) || isTextValueStop(s[valStart]) {
			pos = delimPos + 1
			continue
		}
		p := delimPos
		for p > last && isDelimSpace(s[p-1]) {
			p--
		}
		quoteStart, keyEnd := p, p
		if p-2 >= last && s[p-2] == '\\' && (s[p-1] == '"' || s[p-1] == '\'') && !(p-3 >= last && s[p-3] == '\\') {
			quoteStart, keyEnd = p-2, p-2
		} else if p-1 >= last && (s[p-1] == '"' || s[p-1] == '\'') {
			quoteStart, keyEnd = p-1, p-1
		}
		keyStart := findTextKeyStart(s, last, keyEnd)
		if keyStart < 0 || (quoteStart == p && s[delimPos] == ':' && isPathPrefixedDottedKey(s, keyStart, keyEnd)) {
			pos = delimPos + 1
			continue
		}
		end := scanValueEnd(s, valStart, isTextValueStop, &qState)
		open := brackets.before(s, keyStart)
		switch kind, pair := nestsSensitiveKey(s, valStart, end, startsLine(s, delimPos+1, valStart), &qState); kind {
		case nestedKey:
			pos = valStart
			continue
		case valueEndingInKeyName:
			_, quote := unquoteValue(s[valStart:end])
			end = max(end, nestedPairEndWithState(s, pair, open, &qState))
			brackets.skip(end)
			b.WriteString(s[last:keyStart])
			b.WriteString(s[keyStart:keyEnd])
			b.WriteString(s[quoteStart:p])
			b.WriteByte(s[delimPos])
			b.WriteString(quote)
			b.WriteString(redactedValue)
			b.WriteString(quote)
			last = end
			pos = end
			continue
		}
		end = textValueEnd(s, valStart, end, open)
		brackets.skip(end)
		if end == valStart {
			// Only closing brackets follow the delimiter, as in {Password:}.
			pos = end
			continue
		}
		inner, quote := unquoteValue(s[valStart:end])
		if isRedactedValue(inner) {
			pos = end
			continue
		}
		b.WriteString(s[last:keyStart])
		b.WriteString(s[keyStart:keyEnd])
		b.WriteString(s[quoteStart:p])
		b.WriteByte(s[delimPos])
		b.WriteString(quote)
		b.WriteString(redactedValue)
		b.WriteString(quote)
		last = end
		pos = end
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

func skipInlineSpacesBack(s string, last, pos int) int {
	for pos > last {
		r, size := utf8.DecodeLastRuneInString(s[last:pos])
		if size == 0 || (!unicode.IsSpace(r) && !unicode.In(r, unicode.Z)) {
			break
		}
		pos -= size
	}
	return pos
}

func isCompoundKeyByte(c byte) bool {
	return isWordByte(c) || c == '.' || c == '-'
}

// maxSensitiveKeyLen is the length of the longest name in sensitiveKeys, and
// so of the longest key name other than a "temp:" one that
// isBareSensitiveKeyName accepts.
var maxSensitiveKeyLen = func() int {
	n := 0
	for k := range sensitiveKeys {
		n = max(n, len(k))
	}
	return n
}()

// sensitiveKeyStart returns where the sensitive key begins in the compound
// key s[start:end]: at start when the whole key is sensitive (see
// isSensitiveKey), such as DB_PASSWORD, and otherwise at the first word
// boundary in it after which a fixed sensitive key name ends the key, such as
// the "token" of next-page-token, or -1 when there is none, as in
// next_page_token or xtoken.
func sensitiveKeyStart(s string, start, end int) int {
	sub := s[start:end]
	if strings.IndexByte(sub, ':') < 0 {
		b0, b1, ok := last2FoldedKeyBytes(sub)
		if !ok || !validSensitiveLast2[b0][b1] {
			return -1
		}
	}
	if isBareSensitiveKeyName(sub) || isSensitiveKey(sub) {
		return start
	}
	scanFrom := max(start+1, end-maxSensitiveKeyLen*3)
	for q := scanFrom; q < end; q++ {
		if (!isWordByte(s[q-1]) || !isWordByte(s[q])) && isSensitiveStartByte(s, q) && isBareSensitiveKeyName(s[q:end]) {
			return q
		}
	}
	return -1
}

func isSensitiveStartByte(s string, q int) bool {
	if isWordByte(s[q]) {
		return true
	}
	return strings.HasPrefix(s[q:], "\u017f") || strings.HasPrefix(s[q:], "\u212a")
}

// isBareSensitiveKeyName reports whether name matches bareSensitiveKeyRegex.
func isBareSensitiveKeyName(name string) bool {
	if hasFoldPrefix(name, session.KeyPrefixTemp) {
		rest := name[len(session.KeyPrefixTemp):]
		return rest != "" && !strings.ContainsAny(rest, "=:\"' \t\r\n\f,}")
	}
	if len(name) > maxSensitiveKeyLen*3 {
		return false
	}
	b0, b1, ok := last2FoldedKeyBytes(name)
	if !ok || !validSensitiveLast2[b0][b1] {
		return false
	}
	_, ok = bareFixedSensitiveKeys[foldKeyASCII(name)]
	return ok
}

// isWordByte reports whether c is a word character of the \b assertion.
func isWordByte(c byte) bool {
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9' || c == '_'
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
func nestsSensitiveKey(s string, start, end int, lineStart bool, st *quoteScanState) (nestedKeyKind, int) {
	kind, pair := classifyNestedKey(s, start, end, lineStart, st)
	if kind == nestedKey && isSkippedPathDottedPair(s, pair) {
		return valueEndingInKeyName, pair
	}
	return kind, pair
}

func classifyNestedKey(s string, start, end int, lineStart bool, st *quoteScanState) (nestedKeyKind, int) {
	if end-start > maxNestedKeyLen || s[start] == '{' {
		return notNestedKey, 0
	}
	v := s[start:end]
	if s[start] == '[' {
		cLen := closedContainerLen(v)
		if cLen == 0 || cLen == len(v) {
			return notNestedKey, 0
		}
		if kind, pair := classifyNestedKey(s, start+cLen, end, false, st); kind != notNestedKey {
			return valueEndingInKeyName, pair
		}
		return notNestedKey, 0
	}
	if inner, quote := unquoteValue(v); quote != "" {
		if !valueFollowsDelimiter(s, end, st) {
			return notNestedKey, 0
		}
		if kind, pair := nestedKeyName(inner, start+len(quote), lineStart); kind != notNestedKey {
			return kind, pair
		}
		if isValidJSONPatternKey(trimInlineSpaces(inner)) {
			if lineStart && quote == `"` && isJSONColonAfter(s, end) {
				return nestedKey, start
			}
			return valueEndingInKeyName, start
		}
		return notNestedKey, 0
	}
	for bIdx := 1; bIdx < len(v); bIdx++ {
		if (v[bIdx] == '{' || v[bIdx] == '[') && (v[bIdx-1] == ':' || v[bIdx-1] == '=') &&
			end < len(s) && closedContainerLen(v[bIdx:]) == 0 {
			if kind, pair := nestedKeyName(v[:bIdx-1], start, lineStart); kind != notNestedKey {
				return kind, pair
			}
		}
	}
	if strings.HasSuffix(v, ":") || strings.HasSuffix(v, "=") {
		if !valueStartsAt(s, end, st) {
			return notNestedKey, 0
		}
		return nestedKeyName(v[:len(v)-1], start, lineStart)
	}
	if followedByDelimiter(s, end) {
		if !valueFollowsDelimiter(s, end, st) {
			return notNestedKey, 0
		}
		return nestedKeyName(v, start, lineStart)
	}
	// A name and a delimiter followed by a quoted value that holds
	// whitespace, ",", ";" or "&", as in credentials: password="a b", where
	// the value matched ends inside the quoted value; or a quoted sensitive key
	// containing whitespace (such as abc&" api_key\n": ...) whose closing quote
	// lies past end.
	for q := 0; q < len(v); q++ {
		if v[q] != '"' && v[q] != '\'' && v[q] != '\\' {
			continue
		}
		if q > 0 && (v[q-1] == ':' || v[q-1] == '=') {
			if qEnd, ok := scanQuotedAt(s, start+q, st); ok && qEnd > end && valueStartsAt(s, start+q, st) {
				if kind, pair := nestedKeyName(v[:q-1], start, lineStart); kind != notNestedKey {
					return kind, pair
				}
			}
			continue
		}
		if q == 0 || isJoinBeforeQuotedKey(v[q-1]) {
			if qEnd, ok := scanQuotedAt(s, start+q, st); ok && qEnd > end && qEnd-(start+q) <= maxNestedKeyLen &&
				valueFollowsDelimiter(s, qEnd, st) {
				inner, quote := unquoteValue(s[start+q : qEnd])
				if quote != "" && isValidJSONPatternKey(trimInlineSpaces(inner)) {
					if q == 0 && lineStart && quote == `"` && isJSONColonAfter(s, qEnd) {
						return nestedKey, start
					}
					return valueEndingInKeyName, start + q
				}
			}
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

// isPathPrefixedDottedKey reports whether the bare key s[keyStart:keyEnd] is a
// dotted filename segment in a file path (immediately preceded by "/" or "\",
// excluding URI "://"), such as "/etc/ssl/private.key" in a Go *os.PathError.
func isPathPrefixedDottedKey(s string, keyStart, keyEnd int) bool {
	if keyStart <= 0 || !strings.ContainsRune(s[keyStart:keyEnd], '.') {
		return false
	}
	prev := s[keyStart-1]
	if prev != '/' && prev != '\\' {
		return false
	}
	if prev == '/' && keyStart >= 3 && s[keyStart-3:keyStart] == "://" {
		return false
	}
	return true
}

// isSkippedPathDottedPair reports whether the pair starting at pair in s is an
// unquoted path-prefixed dotted key with a ":" delimiter that
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
	if end == pair || !isPathPrefixedDottedKey(s, pair, end) {
		return false
	}
	j := skipDelimSpaceFwd(s, end)
	return j < len(s) && s[j] == ':'
}

// isJSONColonAfter reports whether s[i:] begins with ":" after optional JSON
// whitespace, so redactJSONKeyValues will match a preceding double-quoted key.
func isJSONColonAfter(s string, i int) bool {
	for i < len(s) && isRegexpSpace(s[i]) {
		i++
	}
	return i < len(s) && s[i] == ':'
}

// closedContainerLen returns the byte length of the balanced "{" or "["
// container at the start of sub, or 0 if sub does not close within len(sub)
// bytes. Quotes (including escaped \" and \') inside sub are skipped.
func closedContainerLen(sub string) int {
	depth := 0
	var quote byte
	escaped := false
	for i := 0; i < len(sub); i++ {
		c := sub[i]
		switch {
		case escaped:
			if c != '\\' || i+1 == len(sub) {
				break
			}
			switch sub[i+1] {
			case quote:
				quote, escaped = 0, false
				i++
			case '\\':
				i += 2
				if i < len(sub) && sub[i] == '\\' {
					i++
				}
			default:
				i++
			}
		case c == '\\':
			if quote == 0 && i+1 < len(sub) && (sub[i+1] == '"' || sub[i+1] == '\'') {
				quote, escaped = sub[i+1], true
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
				return i + 1
			}
		}
	}
	return 0
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

// hasValidCompoundPrefix reports whether prefix matches compoundKeyNameRe's
// optional prefix ([^\s"'\\]*[^\w\s"'\\]): non-empty, free of whitespace,
// quotes and backslashes, and ending in a non-word byte.
func hasValidCompoundPrefix(prefix string) bool {
	if prefix == "" || isWordByte(prefix[len(prefix)-1]) {
		return false
	}
	for i := 0; i < len(prefix); i++ {
		c := prefix[i]
		if isRegexpSpace(c) || c == '"' || c == '\'' || c == '\\' {
			return false
		}
	}
	return true
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
	q := leadingOptionalQuoteLen(name)
	if m := compoundKeyNameRe.FindStringSubmatchIndex(name); m != nil {
		if lineStart && (q == 0 || m[2] == q) && !strings.ContainsAny(name[:m[2]], "&;|") {
			return nestedKey, at + m[2]
		}
		return valueEndingInKeyName, at + m[2]
	}
	if kind, pair := trailingQuotedSensitiveKey(name, at); kind != notNestedKey {
		return kind, pair
	}
	stateStart := skipStatePrefixes(name, q)
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
			if lineStart && off == stateStart {
				return nestedKey, at + kStart
			}
			return valueEndingInKeyName, at + kStart
		}
	}
	return notNestedKey, 0
}

// isJoinBeforeQuotedKey reports whether c is a non-word punctuation joiner
// (such as "&", ";", "|", "/" or "+") that separates an earlier value from a
// trailing quoted key, matching compoundKeyNameRe's [^\w\s"'\\] boundary.
func isJoinBeforeQuotedKey(c byte) bool {
	return !isWordByte(c) && !isRegexpSpace(c) && c != '"' && c != '\'' && c != '\\'
}

// trailingQuotedSensitiveKey checks whether name ends with a quoted sensitive
// key (", ', \" or \') after a non-word joiner (or at the start of name), such
// as abc&"token" or ;'client_secret', and if so returns valueEndingInKeyName
// and the offset of the opening quote in s.
func trailingQuotedSensitiveKey(name string, at int) (nestedKeyKind, int) {
	n := len(name)
	if n >= 4 && name[n-2] == '\\' && (name[n-1] == '"' || name[n-1] == '\'') && isUnescapedQuote(name, 0, n-2) {
		q := name[n-1]
		for openIdx := n - 4; openIdx >= 0; openIdx-- {
			if name[openIdx] == '\\' && name[openIdx+1] == q && isUnescapedQuote(name, 0, openIdx) {
				if (openIdx == 0 || isJoinBeforeQuotedKey(name[openIdx-1])) &&
					isValidJSONPatternKey(trimInlineSpaces(name[openIdx+2:n-2])) {
					return valueEndingInKeyName, at + openIdx
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
				if (openIdx == 0 || isJoinBeforeQuotedKey(name[openIdx-1])) &&
					isValidJSONPatternKey(trimInlineSpaces(name[openIdx+1:n-1])) {
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

// nestedPairEnd returns where the text to redact with a value that only ends
// in a key name ends (see nestsSensitiveKey), given pair, where the pair that
// the key name starts begins in s: at the end of the value of that pair, or,
// if that value is itself the name of a pair, as in "password=a-token: token:
// x", at the end of the value of that pair, and so on. open is the number of
// brackets opened before the key of the value.
func nestedPairEnd(s string, pair, open int) int {
	return nestedPairEndWithState(s, pair, open, nil)
}

func nestedPairEndWithState(s string, pair, open int, st *quoteScanState) int {
	for {
		delimAfter, start, end, ok := scanTextKeyValAt(s, pair, st)
		if !ok {
			return failClosedNestedPairEnd(s, pair, open, st)
		}
		kind, next := nestsSensitiveKey(s, start, end, startsLine(s, delimAfter, start), st)
		if kind == notNestedKey {
			return textValueEnd(s, start, end, open)
		}
		pair = next
	}
}

// failClosedNestedPairEnd extends past a nested key and its value when
// scanTextKeyValAt cannot re-parse pair directly (for example, a quoted key
// with surrounding padding), while returning pair unchanged if no delimiter
// follows pair at all.
func failClosedNestedPairEnd(s string, pair, open int, st *quoteScanState) int {
	if pair >= len(s) {
		return pair
	}
	i := pair
	if qEnd, qOK := scanQuotedAt(s, i, nil); qOK {
		i = qEnd
	}
	rel := strings.IndexAny(s[i:], "=:")
	if rel < 0 {
		return pair
	}
	delimAfter := i + rel + 1
	valStart := skipDelimSpaceFwd(s, delimAfter)
	if (valStart >= len(s) || isTextValueStop(s[valStart])) && valStart > delimAfter && !isTextValueStop(s[valStart-1]) {
		valStart--
	}
	if valStart >= len(s) || isTextValueStop(s[valStart]) {
		return valStart
	}
	return textValueEnd(s, valStart, scanValueEnd(s, valStart, isTextValueStop, st), open)
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

// valueFollowsDelimiter reports whether a key-value delimiter follows s[:i],
// possibly after whitespace, and a value starts after the delimiter (see
// valueStartsAt).
func valueFollowsDelimiter(s string, i int, st *quoteScanState) bool {
	j := skipDelimSpaceFwd(s, i)
	return j < len(s) && (s[j] == ':' || s[j] == '=') && valueStartsAt(s, j+1, st)
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
// i, possibly after whitespace. The value read is a quoted value that ends
// there (see quotedValueEnds), which runs on past any whitespace in it to its
// closing quote, or else the first run of bytes, up to the next whitespace,
// ",", ";" or "&". It must hold a byte other than whitespace and the closing
// brackets, quotes and punctuation that may end a value (see
// trimEnclosingClosers), unlike the "}" that closes the struct in the text
// {Password:abc-token: }, the unclosed quote in {Password:abc-token: "} or the
// empty quotes in {Password:abc-token: ""}.
func valueStartsAt(s string, i int, st *quoteScanState) bool {
	valStart := skipDelimSpaceFwd(s, i)
	if valStart >= len(s) {
		return false
	}
	rest := s[valStart:]
	if c := rest[0]; c == '"' || c == '\'' || c == '\\' {
		if qEnd, ok := scanQuotedAt(s, valStart, st); ok && quotedValueEnds(rest, qEnd-valStart) {
			return hasNonEmptyTextValue(s[valStart:qEnd], false)
		}
	}
	return hasNonEmptyTextValue(rest, true)
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
	// escaped quote closes it (see scanEscapedQuoted).
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
