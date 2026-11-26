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
	"unicode"
	"unicode/utf8"

	"google.golang.org/adk/v2/session"
)

// sensitiveKeys holds normalized (lowercase, "-" replaced by "_") keys whose
// values are always redacted, including whole credential envelopes.
//
// Alignment with adk-python: src/google/adk/plugins/bigquery_agent_analytics_plugin.py
// (lines 884-905 at d723cb87) defines a 20-key _SENSITIVE_KEYS set normalized
// via lowercase/"-"->"_", plus "temp:" session keys (lines 1737-1738, 2317-2318).
// Go keeps the broader v0.1.0 34-key case- and separator-insensitive set plus
// secret_key, signing_key, compound keys, and stacked user:/app:/temp: prefixes
// so upgrading from v0.1.0 never weakens redaction.
var sensitiveKeys = map[string]struct{}{
	"access_token": {}, "api_key": {}, "api_key_secret": {}, "api_key_string": {},
	"auth_code": {}, "authorization": {}, "aws_secret_access_key": {}, "aws_session_token": {},
	"bearer": {}, "client_secret": {}, "code_verifier": {}, "credential": {},
	"credentials": {}, "google_access_id": {}, "id_token": {}, "passphrase": {},
	"password": {}, "private_key": {}, "proxy_authorization": {}, "refresh_token": {},
	"secret": {}, "secret_access_key": {}, "secret_key": {}, "session_token": {},
	"sig": {}, "signature": {}, "signing_key": {}, "token": {},
	"x_amz_credential": {}, "x_amz_signature": {}, "x_api_key": {}, "x_auth_token": {},
	"x_goog_api_key": {}, "x_goog_credential": {}, "x_goog_security_token": {}, "x_goog_signature": {},
}

// lowPrecisionKeySuffixes are sensitive only as standalone keys, not as the
// trailing word of a compound name (for example, method_signature,
// request_sig, and exchangedAuthCredential stay readable).
var lowPrecisionKeySuffixes = map[string]struct{}{
	"bearer": {}, "credential": {}, "credentials": {}, "sig": {}, "signature": {},
}

// nonCredentialSegments are whole trailing segments that end in a credential
// word without being credentials (for example, my_xtoken, reauthorization).
var nonCredentialSegments = map[string]struct{}{
	"xtoken": {}, "reauthorization": {}, "preauthorization": {},
	"deauthorization": {}, "unauthorization": {}, "xauthorization": {},
}

// Derived lookup tables initialized once from sensitiveKeys:
//   - sensitiveKeyNames holds separator-free folded forms ("api_key" -> "apikey").
//   - compoundKeySuffixes excludes lowPrecisionKeySuffixes.
//   - patternWords is the minimal substring cover of sensitiveKeyNames plus
//     "basic" and "temp:", used by mayMatchTextPatterns to skip clean text.
var (
	sensitiveKeyNames   = map[string]struct{}{}
	compoundKeySuffixes []string
	patternWords        []string
	maxSensitiveKeyLen  int
)

func init() {
	for k := range sensitiveKeys {
		maxSensitiveKeyLen = max(maxSensitiveKeyLen, len(k))
		sensitiveKeyNames[compactKey(k)] = struct{}{}
	}
	for name := range sensitiveKeyNames {
		if _, low := lowPrecisionKeySuffixes[name]; !low {
			compoundKeySuffixes = append(compoundKeySuffixes, name)
		}
	}
	slices.Sort(compoundKeySuffixes)
	words := append([]string{"basic", session.KeyPrefixTemp}, slices.Collect(maps.Keys(sensitiveKeyNames))...)
	patternWords = slices.DeleteFunc(slices.Clone(words), func(w string) bool {
		return slices.ContainsFunc(words, func(other string) bool { return other != w && strings.Contains(w, other) })
	})
	slices.Sort(patternWords)
}

func foldForTextPatterns(s string) (string, bool) {
	folded := foldKeyText(s, true)
	for _, w := range patternWords {
		if strings.Contains(folded, w) {
			return folded, true
		}
	}
	return "", false
}

func mayMatchTextPatterns(s string) bool {
	_, ok := foldForTextPatterns(s)
	return ok
}

// foldRuneAt decodes s[i], lowercasing ASCII uppercase and folding U+0130
// ("i"), U+017F ("s"), and U+212A ("k").
func foldRuneAt(s string, i int) (folded byte, size int, upper, ok bool) {
	if c := s[i]; c < utf8.RuneSelf {
		if 'A' <= c && c <= 'Z' {
			return c + 'a' - 'A', 1, true, true
		}
		return c, 1, false, true
	}
	r, sz := utf8.DecodeRuneInString(s[i:])
	switch r {
	case '\u0130':
		return 'i', sz, true, true
	case '\u017f':
		return 's', sz, false, true
	case '\u212a':
		return 'k', sz, true, true
	}
	return 0, sz, false, false
}

func foldKeyText(s string, dropSeps bool) string {
	b := make([]byte, 0, len(s))
	changed := false
	for i := 0; i < len(s); {
		fc, sz, upper, ok := foldRuneAt(s, i)
		if !ok {
			b = append(b, s[i:i+sz]...)
		} else if dropSeps && (fc == '-' || fc == '_' || fc == '.') {
			changed = true
		} else {
			changed = changed || upper || sz > 1
			b = append(b, fc)
		}
		i += sz
	}
	if !changed {
		return s
	}
	return string(b)
}

func foldKeyASCII(k string) string { return foldKeyText(k, false) }
func compactKey(k string) string   { return foldKeyText(k, true) }

func fixedSensitiveKeyRegex() string {
	parts := make([]string, 0, len(sensitiveKeys))
	for _, k := range slices.Sorted(maps.Keys(sensitiveKeys)) {
		parts = append(parts, strings.ReplaceAll(strings.ReplaceAll(regexp.QuoteMeta(k), "_", "[._-]?"), "i", `[i\x{0130}]`))
	}
	return strings.Join(parts, "|")
}

func bareCandidateKeyRegex() string {
	sp := `(?:(?:` + regexp.QuoteMeta(session.KeyPrefixUser) + `|` + regexp.QuoteMeta(session.KeyPrefixApp) + `))*`
	return sp + `temp:[^=:"'\s\r\n,}]+|` + sp + `[A-Za-z0-9_.\x{0130}\x{017f}\x{212a}-]*(?:` + fixedSensitiveKeyRegex() + `)`
}

func normalizeKey(k string) string {
	return strings.ReplaceAll(foldKeyASCII(trimBOMAndSpace(k)), "-", "_")
}

func stripStatePrefixes(k string) (rest string, isTemp bool) {
	k = trimBOMAndSpace(k)
	for strings.IndexByte(k, ':') >= 0 {
		if hasFoldPrefix(k, session.KeyPrefixTemp) {
			return k, true
		}
		if !hasFoldPrefix(k, session.KeyPrefixUser) && !hasFoldPrefix(k, session.KeyPrefixApp) {
			break
		}
		k = trimBOMAndSpace(k[strings.IndexByte(k, ':')+1:])
	}
	return k, false
}

func hasStackedTempColon(k string) bool {
	rest, isTemp := stripStatePrefixes(k)
	return isTemp && strings.IndexByte(rest[len(session.KeyPrefixTemp):], ':') >= 0
}

func hasFoldPrefix(s, prefix string) bool {
	if len(s) < len(prefix) {
		return false
	}
	for si, i := 0, 0; i < len(prefix); i++ {
		if si >= len(s) {
			return false
		}
		fc, sz, _, ok := foldRuneAt(s, si)
		if !ok {
			return false
		}
		si += sz
		pc := prefix[i]
		if 'A' <= pc && pc <= 'Z' {
			pc += 'a' - 'A'
		}
		if fc != pc {
			return false
		}
	}
	return true
}

// isBaseSensitiveKey reports whether k belongs to the plugin/agentanalytics
// v0.1.0 sensitive key set: "temp:" keys and the 34 original separator-free
// names (excluding dotted names and post-v0.1.0 additions secret_key and
// signing_key), with at most one "user:" or "app:" prefix when allowStatePrefix
// is true. Callers use it to preserve v0.1.0 behavior where post-v0.1.0 keys
// differ (such as URI path chaining and whole-document JSON fallback).
func isBaseSensitiveKey(k string, allowStatePrefix bool) bool {
	s := strings.ToLower(strings.TrimSpace(k))
	if strings.HasPrefix(s, session.KeyPrefixTemp) {
		return true
	}
	if allowStatePrefix {
		if rest, ok := strings.CutPrefix(s, session.KeyPrefixUser); ok {
			s = strings.TrimSpace(rest)
		} else if rest, ok := strings.CutPrefix(s, session.KeyPrefixApp); ok {
			s = strings.TrimSpace(rest)
		}
	}
	if strings.ContainsRune(s, '.') {
		return false
	}
	stripped := compactKey(s)
	_, ok := sensitiveKeyNames[stripped]
	return ok && stripped != "secretkey" && stripped != "signingkey"
}

// isSensitiveKey reports whether values under k must be redacted: exact
// sensitiveKeys (ignoring case, "_", "-", ".", and stacked "user:"/"app:"
// prefixes), "temp:" session keys, and word-bounded compound keys.
func isSensitiveKey(k string) bool {
	rest, isTemp := stripStatePrefixes(k)
	if isTemp {
		return true
	}
	if rest == "" {
		return false
	}
	if _, ok := sensitiveKeyNames[compactKey(rest)]; ok {
		return true
	}
	return isCompoundSensitiveKey(rest)
}

func isCompoundSensitiveKey(k string) bool {
	lastSeg := k
	if idx := strings.LastIndexAny(k, "_-."); idx >= 0 {
		lastSeg = k[idx+1:]
	}
	if _, excluded := nonCredentialSegments[foldKeyASCII(lastSeg)]; excluded {
		return false
	}
	compact, starts, hasBoundary := splitCompoundKey(k)
	if !hasBoundary || compact == "" {
		return false
	}
	for _, name := range compoundKeySuffixes {
		i := len(compact) - len(name)
		if i > 0 && starts[i] && compact[i:] == name && (name != "token" || !isPageTokenKey(compact[:i], starts)) {
			return true
		}
	}
	return false
}

func splitCompoundKey(k string) (compact string, starts []bool, hasBoundary bool) {
	b, starts, sep := make([]byte, 0, len(k)), make([]bool, 0, len(k)), true
	for i := 0; i < len(k); {
		fc, sz, upper, ok := foldRuneAt(k, i)
		if !ok {
			return "", nil, false
		}
		if sz == 1 && (fc == '_' || fc == '-' || fc == '.') {
			sep, hasBoundary = true, true
			i++
			continue
		}
		if !('a' <= fc && fc <= 'z') && !('0' <= fc && fc <= '9') {
			return "", nil, false
		}
		hasBoundary = hasBoundary || upper
		starts, b, sep = append(starts, sep || upper), append(b, fc), false
		i += sz
	}
	return string(b), starts, hasBoundary
}

func isPageTokenKey(before string, starts []bool) bool {
	switch before {
	case "next", "prev", "previous", "pad", "eos", "eot", "bos", "unk", "cls", "sep", "mask":
		return true
	}
	for _, cursor := range []string{"page", "sync", "continuation", "resume"} {
		if i := len(before) - len(cursor); i >= 0 && starts[i] && before[i:] == cursor {
			switch before[:i] {
			case "", "next", "prev", "previous", "start", "newstart":
				return true
			}
		}
	}
	return false
}

func isWordByte(c byte) bool {
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9' || c == '_'
}

func isBareSensitiveKeyName(name string) bool {
	if hasFoldPrefix(name, session.KeyPrefixTemp) {
		rest := name[len(session.KeyPrefixTemp):]
		return rest != "" && !strings.ContainsAny(rest, "=:\"' \t\r\n\f,}")
	}
	if name == "" || len(name) > maxSensitiveKeyLen*3 {
		return false
	}
	prevSep := true
	for i := range len(name) {
		sep := name[i] == '_' || name[i] == '-' || name[i] == '.'
		if sep && prevSep {
			return false
		}
		prevSep = sep
	}
	if prevSep {
		return false
	}
	_, ok := sensitiveKeyNames[compactKey(name)]
	return ok
}

// sensitiveKeyStart returns the starting offset of the sensitive key within
// s[start:end], or -1 if none matches. It first checks the whole slice (for
// exact, snake_case, and camelCase compound keys), then checks shorter suffixes
// at non-word boundaries (such as "-" or "." in "next-page-token" or "x.sig").
func sensitiveKeyStart(s string, start, end int) int {
	sub := s[start:end]
	if isBareSensitiveKeyName(sub) || isSensitiveKey(sub) {
		return start
	}
	for q := max(start+1, end-maxSensitiveKeyLen*3); q < end; q++ {
		if (!isWordByte(s[q-1]) || !isWordByte(s[q])) && utf8.RuneStart(s[q]) && isBareSensitiveKeyName(s[q:end]) {
			return q
		}
	}
	return -1
}

func isValidJSONPatternKey(key string) bool {
	rest, isTemp := stripStatePrefixes(key)
	if isTemp {
		after := rest[len(session.KeyPrefixTemp):]
		return after != "" && !strings.ContainsAny(after, "\"'\t\n\f\r ,}")
	}
	return rest != "" && !strings.ContainsAny(rest, "\"'\r\n,:{}") && isSensitiveKey(rest)
}

func isHexByte(c byte) bool {
	return '0' <= c && c <= '9' || 'a' <= c && c <= 'f' || 'A' <= c && c <= 'F'
}

func canonicalizeCredentialMarker(s string) string {
	return canonicalizeCredentialMarkerWith(s, false)
}

// canonicalizeCredentialMarkerWith iteratively decodes \" / \', %XX, \xXX, and
// \uXXXX escapes in one pass to detect obfuscated credential markers. When
// asciiOnly is false, it also decodes U+0130, U+017F, U+212A, and \uXXXX space
// or format runes that can hide key boundaries.
func canonicalizeCredentialMarkerWith(s string, asciiOnly bool) string {
	if !strings.ContainsAny(s, `\%`) {
		return s
	}
	output, changed := make([]byte, 0, len(s)), false
	for i := range len(s) {
		output = append(output, s[i])
		for {
			n := len(output)
			if n >= 2 && output[n-2] == '\\' && (output[n-1] == '"' || output[n-1] == '\'') {
				output, changed = append(output[:n-2], output[n-1]), true
				continue
			}
			escapeStart := -1
			var digits []byte
			switch {
			case n >= 6 && output[n-6] == '\\' && (output[n-5] == 'u' || output[n-5] == 'U') &&
				isHexByte(output[n-4]) && isHexByte(output[n-3]) && isHexByte(output[n-2]) && isHexByte(output[n-1]):
				escapeStart, digits = n-6, output[n-4:]
			case n >= 4 && output[n-4] == '\\' && (output[n-3] == 'x' || output[n-3] == 'X') &&
				isHexByte(output[n-2]) && isHexByte(output[n-1]):
				escapeStart, digits = n-4, output[n-2:]
			case n >= 3 && output[n-3] == '%' && isHexByte(output[n-2]) && isHexByte(output[n-1]):
				escapeStart, digits = n-3, output[n-2:]
			}
			if escapeStart < 0 {
				break
			}
			decoded, err := strconv.ParseUint(string(digits), 16, 32)
			r := rune(decoded)
			allowed := r <= 0x7F || (!asciiOnly && (r == '\u0130' || r == '\u017f' || r == '\u212a' ||
				(escapeStart == n-6 && (unicode.IsSpace(r) || unicode.In(r, unicode.Z, unicode.Cf)))))
			if err != nil || !allowed {
				break
			}
			output, changed = utf8.AppendRune(output[:escapeStart], r), true
		}
	}
	if !changed {
		return s
	}
	return string(output)
}
