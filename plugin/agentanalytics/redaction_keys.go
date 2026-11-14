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
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"google.golang.org/adk/v2/session"
)

// Sensitive key names and how text is matched against them: the key tables,
// isSensitiveKey and its compound-key rules, case folding, the pattern-word
// prefilter and the canonicalization of escaped credential markers.

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

// patternWordsByFirst holds patternWords by their first byte.
var patternWordsByFirst = func() (m [128][]string) {
	for _, w := range patternWords {
		m[w[0]] = append(m[w[0]], w)
	}
	return m
}()

// mayMatchTextPatterns reports whether s contains one of patternWords,
// ignoring case. When it does not, none of the credential text scanners can
// match s, so scrubTextPatterns and hasQuotedSensitiveKey need not scan it.
// It reports whether foldPatternText(s) contains one of patternWords, without
// building that text: a word of it starts at a byte of s that it does not
// remove. It steps over a whole character at a time, as no word starts
// inside one.
func mayMatchTextPatterns(s string) bool {
	for i, size := 0, 0; i < len(s); i += size {
		var c byte
		c, size = patternFoldAt(s, i)
		if c >= utf8.RuneSelf {
			continue
		}
		for _, w := range patternWordsByFirst[c] {
			if foldedHasPrefix(s, i, w) {
				return true
			}
		}
	}
	return false
}

// patternFoldAt returns the byte that foldPatternText writes for the
// character at s[i], or 0 for a separator it removes, and the length of that
// character; a non-ASCII character that it keeps returns a byte of
// utf8.RuneSelf or more.
func patternFoldAt(s string, i int) (byte, int) {
	c := s[i]
	if c < utf8.RuneSelf {
		switch {
		case 'A' <= c && c <= 'Z':
			return c + 'a' - 'A', 1
		case c == '-' || c == '_' || c == '.':
			return 0, 1
		}
		return c, 1
	}
	r, size := utf8.DecodeRuneInString(s[i:])
	switch r {
	case '\u0130':
		return 'i', size
	case '\u017f':
		return 's', size
	case '\u212a':
		return 'k', size
	}
	return utf8.RuneSelf, size
}

// foldedHasPrefix reports whether foldPatternText(s[i:]) starts with w.
func foldedHasPrefix(s string, i int, w string) bool {
	for k := 0; k < len(w); {
		if i >= len(s) {
			return false
		}
		c, size := patternFoldAt(s, i)
		i += size
		if c == 0 {
			continue
		}
		if c != w[k] {
			return false
		}
		k++
	}
	return true
}

// foldPatternText lowercases the ASCII letters of s and removes "-", "_" and ".",
// as the scanners accept key names with any of those separators or none
// (api_key, api-key, api.key, apiKey). It also folds the three non-ASCII
// letters that Go case conversion or RE2 (?i) maps onto ASCII letters: U+0130
// (dotted capital I -> "i"), U+017F (long s -> "s") and U+212A (Kelvin sign ->
// "k"). It is the text that mayMatchTextPatterns, which folds each character
// as it reads it (see patternFoldAt), searches for patternWords.
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

// nonCredentialSegments lists folded separator-free segments whose trailing
// "token" or "authorization" substring is part of a benign word or test dummy
// rather than a credential suffix, regardless of casing (such as XTOKEN,
// MY_XTOKEN, REAUTHORIZATION, or MY_XAUTHORIZATION).
var nonCredentialSegments = map[string]struct{}{
	"xtoken":           {},
	"reauthorization":  {},
	"preauthorization": {},
	"deauthorization":  {},
	"unauthorization":  {},
	"xauthorization":   {},
}

func lastCompoundSegment(k string) string {
	for i := len(k) - 1; i >= 0; i-- {
		if k[i] == '_' || k[i] == '-' || k[i] == '.' {
			return k[i+1:]
		}
	}
	return k
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
	if _, excluded := nonCredentialSegments[foldKeyASCII(lastCompoundSegment(k))]; excluded {
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
		n := strings.IndexAny(s[i:], `%\`)
		if n < 0 {
			return false
		}
		i += n
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

// fixedKeySuffixStart returns where the longest end of name that is a fixed
// sensitive key name, in any of its separator variants (see
// bareFixedSensitiveKeys), starts at a word boundary: at the start of name,
// after a byte other than an ASCII letter or digit, as the token of
// sk_v1_token or of dev-v1/token, or at a lower-case letter or digit
// to upper-case letter step, as the Token of secret123Token. It returns -1
// when name ends in no such key, as temp:v1 or DBPASSWORD do ("temp:" names
// are not fixed keys): the words of such a name before the key may be a
// value rather than part of the key.
func fixedKeySuffixStart(name string) int {
	for j := max(0, len(name)-maxSensitiveKeyLen*3); j < len(name); j++ {
		if j > 0 {
			prev, c := name[j-1], name[j]
			alnum := 'a' <= prev && prev <= 'z' || 'A' <= prev && prev <= 'Z' || '0' <= prev && prev <= '9'
			camel := ('a' <= prev && prev <= 'z' || '0' <= prev && prev <= '9') && 'A' <= c && c <= 'Z'
			if (alnum && !camel) || !utf8.RuneStart(c) {
				continue
			}
		}
		if _, ok := bareFixedSensitiveKeys[foldKeyASCII(name[j:])]; ok {
			return j
		}
	}
	return -1
}

// isWordByte reports whether c is a word character of the \b assertion.
func isWordByte(c byte) bool {
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9' || c == '_'
}
