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
// names with "-" or "." for a separator, as in api-key or the dotted api.key
// of stripe.api.key, and without separators, as in the camelCase apiKey or
// the Go field name APIKey.
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

// maxSensitiveKeyWords is the greatest number of words in a name of
// sensitiveKeys, such as the four of aws_secret_access_key.
var maxSensitiveKeyWords = func() int {
	most := 1
	for k := range sensitiveKeys {
		most = max(most, strings.Count(k, "_")+1)
	}
	return most
}()

// sensitiveKeyNames holds the names in sensitiveKeys with "_" removed, the
// form in which isSensitiveKey compares keys.
var sensitiveKeyNames = func() map[string]struct{} {
	names := make(map[string]struct{}, len(sensitiveKeys))
	for k := range sensitiveKeys {
		names[removeKeySeparators(k)] = struct{}{}
	}
	return names
}()

// unicodeSpaceClass matches every Unicode White_Space character: ASCII
// whitespace (including vertical tab, which Go's \s excludes), any Unicode
// space/line/paragraph separator (\p{Z} covers \p{Zs}, \p{Zl} and \p{Zp}, so it
// includes U+2028 and U+2029 as well as U+00A0 and U+3000), and the next line
// character U+0085.
const unicodeSpaceClass = `[:space:]\p{Z}\x{0085}`

// unicodeSpaceRun matches one or more unicodeSpaceClass characters.
const unicodeSpaceRun = `[` + unicodeSpaceClass + `]+`

// delimSpaceClass matches every Unicode White_Space and ASCII control character
// around key-value delimiters.
const delimSpaceClass = unicodeSpaceClass + `[:cntrl:]`

var (
	uriTokenRe   = regexp.MustCompile(`[a-zA-Z][a-zA-Z0-9+.-]*://[^\s"'<>]+`)
	authHeaderRe = regexp.MustCompile(`(?im)\b((?:proxy[-_])?authorization|x[-_]api[-_]key|api[-_]key)[ \t]*:[ \t]*[^\r\n]*`)
	// bearerTokenRe matches a bearer token. A bearer token (RFC 6750) never
	// contains quotes, ")" or "]", so the bare value stops there and keeps
	// the surrounding JSON or prose intact.
	bearerTokenRe = regexp.MustCompile(`(?i)\b(bearer)` + unicodeSpaceRun + `("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\[[^\]\r\n` + unicodeSpaceClass + `]+\][^` + unicodeSpaceClass + `,;)"']*|[^` + unicodeSpaceClass + `,;)"'\]]+)`)
	// basicTokenRe matches a candidate basic credential, which
	// redactBasicCredential redacts only if it is valid base64.
	basicTokenRe      = regexp.MustCompile(`(?i)\b(basic)` + unicodeSpaceRun + `([A-Za-z0-9+/]+={0,2})`)
	jsonKeyValTokenRe = regexp.MustCompile(`(?i)"[\s\p{Z}]*(` + sensitiveKeyRegex() + `)[\s\p{Z}]*"\s*:\s*(` + sensitiveValueRegex(`\s,}`) + `)`)
	// textKeyValTokenAtRe matches a key=value or key: value pair at the start
	// of the text. As in adk-python, the key may be quoted (Python dict reprs
	// use single quotes); it may also be quoted with escaped quotes, as in
	// JSON text formatted with %q. Unlike adk-python's, a bare value does not
	// stop at "}", so a value such as "ab}rest" is redacted as a whole;
	// redactTextKeyValue leaves out the "}" and "]" that end the value and
	// close a bracket opened before the key, as in the %+v output
	// {APIKey:...}.
	textKeyValTokenAtRe = regexp.MustCompile(`(?i)^(` + textKeyNameRegex() + `)((?:\\?["'])?)[` + delimSpaceClass + `]*([=:])[` + delimSpaceClass + `]*(` + sensitiveValueRegex(`\s,;&`) + `)`)
	// textKeyNameAtRe matches the start of textKeyValTokenAtRe, up to and
	// including the delimiter, so that the name of a key can be read without
	// reading its value, which may be long (see allowedKeyAt).
	textKeyNameAtRe = regexp.MustCompile(`(?i)^(` + textKeyNameRegex() + `)((?:\\?["'])?)[` + delimSpaceClass + `]*([=:])`)
	// textKeyNameRe finds the key of a pair as textKeyNameAtRe reads it, up to
	// its delimiter, anywhere in the text: a credential name after a word
	// boundary or a "_" (see bareTextKeyNameRegex), whose compound prefix
	// keyValFinder reads back itself. Of the dotted OAuth names, it finds
	// those of two words only, such as oauth2Config.code; findCamelKeyName
	// finds the others.
	textKeyNameRe = regexp.MustCompile(`(?i)(?:\b|_)(` + bareTextKeyNameRegex() + `)((?:\\?["'])?)[` + delimSpaceClass + `]*([=:])`)
	// textKeyNameCamelAtRe matches the key of a pair at the start of the
	// text, like textKeyNameRe, but starts at the credential name without a
	// word boundary before it (see findCamelKeyName), and reads a dotted
	// OAuth name of any number of words (see anchoredBareTextKeyNameRegex).
	textKeyNameCamelAtRe = regexp.MustCompile(`(?i)^(` + anchoredBareTextKeyNameRegex() + `)((?:\\?["'])?)[` + delimSpaceClass + `]*([=:])`)
	// textKeyValBareAtRe matches a pair at the start of the text, like
	// textKeyValTokenAtRe, but from the credential name of its key, which
	// textKeyNameRe and textKeyNameCamelAtRe find, without the compound
	// prefix that keyValFinder reads back itself.
	textKeyValBareAtRe = regexp.MustCompile(`(?i)^(` + anchoredBareTextKeyNameRegex() + `)((?:\\?["'])?)[` + delimSpaceClass + `]*([=:])[` + delimSpaceClass + `]*(` + sensitiveValueRegex(`\s,;&`) + `)`)
	// textValueStartRe matches the start of the value that
	// textKeyValTokenAtRe reads after the delimiter: a quoted value starts
	// with a byte that a bare value may start with, so whitespace and one
	// such byte are enough.
	textValueStartRe = regexp.MustCompile(`^[` + delimSpaceClass + `]*[^\s,;&]`)
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
	// spring.datasource.password, the "stripe." of stripe.api.key or the "db/"
	// of db/password, but also the "dev-SECRET-" of dev-SECRET-password. The
	// submatch is the key name.
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
// bearerTokenRe, basicTokenRe, jsonKeyValTokenRe, textKeyNameRe and
// quotedSensitiveKeyRe contains at least one, once the text is folded by
// foldPatternText: the sensitive key names without separators (which include
// "authorization", "apikey" and "bearer"), the "temp:" prefix and the "basic"
// scheme. A name that contains another, such as "accesstoken", which contains
// "token", is left out, since the text then contains the shorter name too.
var patternWords = func() []string {
	words := append([]string{"basic", "oauth", session.KeyPrefixTemp}, slices.Collect(maps.Keys(sensitiveKeyNames))...)
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

// foldPatternText lowercases the ASCII letters of s and removes "-", "_" and
// ".", as the patterns accept key names with any of those separators or none
// (api_key, api-key, api.key, apiKey). It also replaces the two non-ASCII
// letters that (?i) matches against ASCII ones, U+017F (long s) and U+212A
// (Kelvin sign), with "s" and "k".
func foldPatternText(s string) string {
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

// textKeyNameRegex returns the pattern for the name of a key in key=value and
// key: value text: a sensitive key name (see bareSensitiveKeyRegex), possibly
// after a run of letters, digits, "_" and "-", so that an environment variable
// or a compound key that ends in one, such as DB_PASSWORD, OPENAI_API_KEY or
// dbPassword, is matched whole, and the oauth*.code key of an OAuth
// configuration (see oauthKeyNameRegex). redactTextKeyValue keeps only the
// compound names that isSensitiveKey accepts, so that a name such as
// next_page_token or mypassword is left alone. The prefix holds no dots: of a
// dotted name such as stripe.api.key, the key is the sensitive key name that
// its last dotted words make, api.key (see fixedSensitiveKeyRegex). It is
// matched at a given offset of the text only.
func textKeyNameRegex() string {
	return `[a-z0-9_-]*(?:` + anchoredBareTextKeyNameRegex() + `)`
}

// bareTextKeyNameRegex returns textKeyNameRegex without the prefix, and with
// the dotted OAuth names of two words only, as in oauth2Config.code. A search
// with textKeyNameRegex is markedly slower on long text than one that finds a
// bare name, so keyValFinder searches with this pattern, after a word
// boundary or a "_", and extends the key back over the prefix itself. A
// sensitive key name of several dotted words, such as the api.key of
// stripe.api.key, starts after the word boundary at a dot. A name that
// continues a word without a separator, as in dbPassword, has no such
// boundary; findCamelKeyName finds it in a name that is followed by "=" or
// ":" instead, at a camelCase boundary: a capital after a lower-case letter or
// a digit, or the last capital of an acronym, as in the "P" of DBPassword. It
// also finds the dotted OAuth names of more words, as in
// oauth2Config.extra.code: a search for those would read a run of dotted words
// that starts with "oauth" to its end from each of them that does, as in
// oauth.a.oauth.a..., in a time that grows with the square of the run.
func bareTextKeyNameRegex() string {
	return bareSensitiveKeyRegex() + `|oauth[a-z0-9_-]*[.]code`
}

// anchoredBareTextKeyNameRegex returns bareTextKeyNameRegex with the dotted
// OAuth names of any number of words (see oauthKeyNameRegex), for the
// expressions that are matched at a given offset of the text: at a name that
// keyValFinder has found, or after a dot or a colon of an oversized key (see
// oversizedNestedKey).
func anchoredBareTextKeyNameRegex() string {
	return bareSensitiveKeyRegex() + `|` + oauthKeyNameRegex()
}

// oauthKeyNameRegex returns the pattern for an OAuth name that
// isCompoundSensitiveKey accepts, from its word that starts with "oauth": that
// word and the dotted words (including empty segments) after it, up to the
// dotted word "code" that ends the name, as in oauth2Config.code,
// oauth2Config.extra.code or oauth..x.code. As isCompoundSensitiveKey wants,
// the words are made of ASCII letters, digits, "_" and "-" only. Case is
// ignored for ASCII letters only, so that the run of words that the pattern
// reads, which oauthRunLastName reads too, does not take the Kelvin sign and
// the long s, which (?i) matches against "k" and "s".
func oauthKeyNameRegex() string {
	return `(?-i:[oO][aA][uU][tT][hH][A-Za-z0-9_-]*(?:[.][A-Za-z0-9_-]*)*[.][cC][oO][dD][eE])`
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
// in which each "_" may be written as any run of "_", "-" and "." or left out,
// so that api_key, api-key, api.key, api__key, api..key, apiKey and APIKey all
// match, agreeing with removeKeySeparators and splitKeyWords.
func fixedSensitiveKeyRegex() string {
	parts := make([]string, 0, len(sensitiveKeys))
	for _, k := range slices.Sorted(maps.Keys(sensitiveKeys)) {
		parts = append(parts, strings.ReplaceAll(regexp.QuoteMeta(k), "_", "[_.-]*"))
	}
	return strings.Join(parts, "|")
}

func normalizeKey(k string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(k)), "-", "_")
}

// removeKeySeparators returns k without its "_", "-" and "." separators.
func removeKeySeparators(k string) string {
	return strings.Map(func(r rune) rune {
		if r == '_' || r == '-' || r == '.' {
			return -1
		}
		return r
	}, k)
}

// isSensitiveKey reports whether the value stored under k must be redacted:
// k names a credential, or it is a session "temp:" key, which holds
// invocation-scoped scratch data. Credential names are compared ignoring
// case, the "_", "-" and "." separators, and a "user:" or "app:" state
// prefix, so that api_key, api-key, api.key, apiKey, the Go field name APIKey
// and the state key user:api_key all match. A compound key that ends in a
// credential name, as an environment variable or a prefixed field does, such
// as DB_PASSWORD, GITHUB_TOKEN, OPENAI_API_KEY, dbPassword,
// x-amz-security-token or stripe.api.key, is sensitive too (see
// isCompoundSensitiveKey).
func isSensitiveKey(k string) bool {
	name, exact := exactSensitiveKey(k)
	return exact || isCompoundSensitiveKey(name)
}

// exactSensitiveKey reports whether k is sensitive by its whole name, as
// isSensitiveKey compares it without the compound names: a session "temp:"
// key, or a credential name, ignoring case, the "_", "-" and "." separators
// and a "user:" or "app:" state prefix. It also returns k without the
// whitespace around it and without that prefix.
func exactSensitiveKey(k string) (name string, ok bool) {
	orig := strings.TrimSpace(k)
	n := strings.ToLower(orig)
	if strings.HasPrefix(n, session.KeyPrefixTemp) {
		return orig, true
	}
	if rest, found := strings.CutPrefix(n, session.KeyPrefixUser); found {
		n, orig = rest, orig[len(session.KeyPrefixUser):]
	} else if rest, found := strings.CutPrefix(n, session.KeyPrefixApp); found {
		n, orig = rest, orig[len(session.KeyPrefixApp):]
	}
	_, ok = sensitiveKeyNames[removeKeySeparators(strings.TrimSpace(n))]
	return strings.TrimSpace(orig), ok
}

// lowPrecisionKeySuffixes are the credential names that are not taken as the
// last word of a compound key, because they are ordinary words in a payload:
// a "method_signature" or a "bearer" is not a credential.
//
// Neither are "credential" and "credentials" taken as the last word: a key such
// as exchangedAuthCredential names a structure whose fields are redacted one
// by one, not a value to redact whole.
var lowPrecisionKeySuffixes = map[string]struct{}{
	"bearer":      {},
	"credential":  {},
	"credentials": {},
	"sig":         {},
	"signature":   {},
}

// isPageTokenName reports whether the key k names a pagination cursor, which
// is not a credential although it ends in "token": its last two words (see
// splitKeyWords) are "page" and "token", after no other word or after
// "next", "prev" or "previous", as in page_token, next_page_token,
// next-page-token, nextPageToken, x.next_page_token or x.next.page.token. A
// key whose last words are other ones is not a cursor, such as
// homepage-token, fb-page-token or fb.page_token, the access token of a
// Facebook Page. k is the complete name, with the words that a dot precedes
// (see textKeyAllowedAt).
func isPageTokenName(k string) bool {
	words, _ := splitKeyWords(k)
	return isPageTokenWords(words)
}

// isPageTokenWords reports whether the words of a key, as splitKeyWords
// returns them, name a pagination cursor (see isPageTokenName).
func isPageTokenWords(words []string) bool {
	n := len(words)
	if n < 2 || words[n-2] != "page" || words[n-1] != "token" {
		return false
	}
	if n == 2 {
		return true
	}
	switch words[n-3] {
	case "next", "prev", "previous":
		return true
	}
	return false
}

// isCompoundSensitiveKey reports whether k, a name made of ASCII letters,
// digits, "_", "-" and ".", split into words at those separators and at
// camelCase boundaries, ends in the words of a sensitive key name:
// DB_PASSWORD, dbPassword and spring.datasource.password end in "password",
// and OPENAI_API_KEY and stripe.api.key end in "api_key". Two acronyms in a
// row have no boundary between them that splitKeyWords can see, so a name
// may also start inside an acronym that other words follow, after its first
// letter: the "aiapi" of OpenAIAPIKey ends in the "api" of "api_key". Names
// that merely contain a credential name, such as token_count, prompt_tokens
// or max_output_tokens, and the pagination tokens (see isPageTokenName), are
// not sensitive. As an OAuth configuration does, a key such as
// oauth2Config.code, whose last dotted segment is "code" after a segment
// that starts with "oauth", is sensitive too.
func isCompoundSensitiveKey(k string) bool {
	if k == "" || strings.ContainsFunc(k, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-' || r == '.')
	}) {
		// Not a key name: a URI path segment such as api_key=value, or prose.
		return false
	}
	lower := strings.ToLower(k)
	if i := strings.LastIndexByte(lower, '.'); i > 0 && lower[i+1:] == "code" {
		for _, seg := range strings.Split(lower[:i], ".") {
			if strings.HasPrefix(seg, "oauth") {
				return true
			}
		}
	}
	compact := removeKeySeparators(lower)
	// A cheap test before the allocation of the words: some credential
	// name must end the key.
	matched := false
	for name := range sensitiveKeyNames {
		if len(compact) > len(name) && strings.HasSuffix(compact, name) {
			matched = true
			break
		}
	}
	if !matched {
		return false
	}
	words, acronyms := splitKeyWords(k)
	if isPageTokenWords(words) {
		return false
	}
	// Only the last maxSensitiveKeyWords words can form a credential name,
	// so the suffixes are not rebuilt for a long prefix of other words.
	start := max(1, len(words)-maxSensitiveKeyWords)
	for i := start; i < len(words); i++ {
		if isCredentialSuffix(strings.Join(words[i:], "")) {
			return true
		}
	}
	// Of an acronym, only the letters that leave room for a name of
	// sensitiveKeys are tried, so a long one is not read to its start.
	for i := max(0, len(words)-maxSensitiveKeyWords); i < len(words)-1; i++ {
		if !acronyms[i] {
			continue
		}
		w, rest := words[i], strings.Join(words[i+1:], "")
		for j := max(1, len(w)+len(rest)-maxFixedKeyNameLen); j < len(w); j++ {
			if isCredentialSuffix(w[j:] + rest) {
				return true
			}
		}
	}
	return false
}

// isCredentialSuffix reports whether name, lower-case words joined without
// separators, is a credential name that may end a compound key.
func isCredentialSuffix(name string) bool {
	if _, low := lowPrecisionKeySuffixes[name]; low {
		return false
	}
	_, ok := sensitiveKeyNames[name]
	return ok
}

// splitKeyWords splits k into lowercase words at every character that is not a
// letter or a digit, and at camelCase boundaries: before an upper-case letter
// that follows a lower-case letter or a digit (dbPassword, oauth2Config), and
// before the last upper-case letter of a run that is followed by a lower-case
// letter (APIKey is "api" and "key"). It also reports for each word whether
// it is an acronym: a run of two or more upper-case letters, which may be
// two acronyms in a row, such as the "AIAPI" of OpenAIAPIKey.
func splitKeyWords(k string) (words []string, acronyms []bool) {
	var cur []rune
	upper := true
	flush := func() {
		if len(cur) > 0 {
			words = append(words, strings.ToLower(string(cur)))
			acronyms = append(acronyms, upper && len(cur) > 1)
			cur = cur[:0]
		}
		upper = true
	}
	runes := []rune(k)
	for i, r := range runes {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			flush()
			continue
		}
		if unicode.IsUpper(r) && len(cur) > 0 {
			prev := runes[i-1]
			nextLower := i+1 < len(runes) && unicode.IsLower(runes[i+1])
			if unicode.IsLower(prev) || unicode.IsDigit(prev) || (unicode.IsUpper(prev) && nextLower) {
				flush()
			}
		}
		upper = upper && unicode.IsUpper(r)
		cur = append(cur, r)
	}
	flush()
	return words, acronyms
}

// bareTextKeyRe matches a text key that is only a sensitive key name, with no
// prefix (see textKeyNameRegex).
var bareTextKeyRe = regexp.MustCompile(`(?i)^(?:` + bareSensitiveKeyRegex() + `)$`)

// textKeyNameWholeRe matches a text that is a whole key name as
// textKeyNameAtRe reads one, such as stripe_api.key or dbPassword (see
// textKeyNameRegex).
var textKeyNameWholeRe = regexp.MustCompile(`(?i)^(?:` + textKeyNameRegex() + `)$`)

// textKeyAllowed reports whether a key that keyValFinder found is to be
// redacted: a plain sensitive key name always is, and a compound name is
// if isSensitiveKey accepts it or a "-" in it precedes a plain name.
func textKeyAllowed(key string) bool {
	if bareTextKeyRe.MatchString(key) || isSensitiveKey(key) {
		return true
	}
	if isPageTokenName(key) {
		// next-page-token: its "token" must not match on its own below.
		return false
	}
	// A sensitive name after a "-" has always been matched on its own, as in
	// foo-signature=..., whose "-" is a word boundary.
	for i := strings.IndexByte(key, '-'); i >= 0; {
		if bareTextKeyRe.MatchString(key[i+1:]) {
			return true
		}
		next := strings.IndexByte(key[i+1:], '-')
		if next < 0 {
			break
		}
		i += 1 + next
	}
	return false
}

// textKeyAllowedAt reports whether the key that keyValFinder found at
// s[start:end] is to be redacted, as textKeyAllowed does for a key that no dot
// precedes. A key that a dot precedes ends a dotted name, such as the
// next.page.token of x.next.page.token=..., and the words of that name before
// its last dot, which the text patterns do not read as a part of the key (see
// textKeyNameRegex), are part of the name that a map key or a field name
// holds. So the complete dotted name decides first, as it does for
// isSensitiveKey: a name that ends in a pagination cursor (see
// isPageTokenName), as next.page.token and x.next_page_token do, is not
// redacted, although its "token" is a credential name on its own, and a name
// that ends in "page_token" after any other word, as fb.page_token and
// home.page_token do, is, although "page_token" alone is a cursor. A dotted
// name that is neither is read by its last words, as textKeyAllowed does, so
// that x.signature=... is redacted still.
func textKeyAllowedAt(s string, start, end int) bool {
	if start > 0 && s[start-1] == '.' {
		whole := s[dottedKeyStart(s, start):end]
		if isPageTokenName(whole) {
			return false
		}
		if isSensitiveKey(whole) {
			return true
		}
	}
	return textKeyAllowed(s[start:end])
}

// dottedKeyStart returns where the dotted name that ends at s[:i] starts: the
// start of the run of letters, digits, "_", "-" and "." that ends there (see
// isCamelTokenByte). A run longer than maxNestedKeyLen is cut near that
// length, so that a name is not read again from each of the names in a long
// run of them, but only at a separator and only once at least
// maxSensitiveKeyWords non-separator words before s[i:] have been included
// (skipping separator runs between them), so that names such as
// fb.page.....token agree with isSensitiveKey regardless of where
// maxNestedKeyLen falls. The names of two keys share no byte, as the delimiter
// after a key stops the run read back from the next key, so each byte is read
// once.
func dottedKeyStart(s string, i int) int {
	isSeparator := func(c byte) bool { return c == '.' || c == '_' || c == '-' }
	lo := max(0, i-maxNestedKeyLen)
	words := 0
	inWord := false
	for i > 0 && isCamelTokenByte(s[i-1]) {
		if i <= lo && words >= maxSensitiveKeyWords && isSeparator(s[i-1]) {
			break
		}
		if isSeparator(s[i-1]) {
			inWord = false
		} else if !inWord {
			words++
			inWord = true
		}
		i--
	}
	return i
}

// canonicalizeCredentialMarker decodes nested ASCII \u, \x and percent
// escapes, the \u and \x escapes of whitespace beyond ASCII, and escaped
// quotes (\" and \'), in one pass. Whitespace such as U+00A0, U+2028 or
// U+3000 separates a scheme such as Bearer from its credential as a space
// does, and JSON and Python reprs write it as \u00a0, \u2028 or \xa0; a
// percent escape beyond ASCII is one byte of UTF-8, not a character, and is
// left alone. The result is used for detection only: callers keep the
// original text unless the canonical form exposes a credential construct.
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
			percent := false
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
				percent = true
			} else if n >= 2 && output[n-2] == '\\' && (output[n-1] == '"' || output[n-1] == '\'') {
				output = append(output[:n-2], output[n-1])
				continue
			}
			if escapeStart < 0 {
				break
			}
			decoded, err := strconv.ParseUint(digits, 16, 32)
			if err != nil || decoded > 0x7F && (percent || !unicode.IsSpace(rune(decoded))) {
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
// segments whose whole name is a sensitive key name (see exactSensitiveKey)
// redact both that segment and the following value segment. Values of
// sensitive query parameters or fragment key-value pairs, and of the "key"
// parameter used for API keys, are replaced with "[REDACTED]". As in
// adk-python, every other path segment, query key and value, and the fragment
// are scrubbed with the text patterns, so credentials in a nested URL
// (?next=https://host/cb?access_token=...), a fragment
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
		// A segment by itself is read by its whole name only (see
		// exactSensitiveKey): the compound names of isSensitiveKey are for
		// map keys, struct fields and key=value text, and a segment that
		// merely ends in a credential name, as in /docs/reset-password/step-2
		// or gs://bucket/db-password/notes.txt, names a path.
		if _, exact := exactSensitiveKey(seg); exact {
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
	var bearerQuotes openQuotes
	s = replaceMatches(s, bearerTokenRe, func(s string, loc []int) (string, int) {
		return redactBearerToken(s, loc, &bearerQuotes)
	})
	s = replaceMatches(s, basicTokenRe, redactBasicCredential)
	var jsonQuotes openQuotes
	s = replaceMatches(s, jsonKeyValTokenRe, func(s string, loc []int) (string, int) {
		return redactJSONKeyValue(s, loc, &jsonQuotes)
	})
	var brackets openBrackets
	var quotes openQuotes
	finder := keyValFinder{whole: s}
	return replaceMatchesFunc(s, finder.find, func(s string, loc []int) (string, int) {
		return redactTextKeyValue(s, loc, &brackets, &quotes)
	})
}

// keyValFinder finds the key-value pairs with a sensitive key in whole, one
// after the other, for replaceMatchesFunc, which calls find with the text
// that is left. It reads the name of a key and the delimiter after it first,
// and the value, which may run on to the end of a long word, only for a key
// that textKeyAllowedAt accepts, so that a run of keys that it does not, as in
// next_page_token:next_page_token:..., is read once rather than to its end
// from each key. It keeps the next key that textKeyNameRe finds from one call
// to the next, so that the text between the keys is searched once, however
// many keys that textKeyNameRe does not find (see findCamelKeyName) the text
// holds.
type keyValFinder struct {
	whole string
	// next is the submatch indexes in whole of the first key, with a value
	// after it, that textKeyNameRe finds at or after the offset of the last
	// search (see nextKeyName), if searched and not nil. Once a search has
	// found nothing, none is run again.
	next     []int
	searched bool
}

// find returns the submatch indexes in rest, the end of f.whole, of the first
// key-value pair with a sensitive key in it, or nil (see pairAtIf).
func (f *keyValFinder) find(rest string) []int {
	pos := len(f.whole) - len(rest)
	if loc := textKeyNameAtRe.FindStringSubmatchIndex(rest); loc != nil && textValueStartRe.MatchString(rest[loc[1]:]) {
		return pairAtIf(textKeyAllowedAt(f.whole, pos+loc[2], pos+loc[3]), rest, loc, 0, textKeyValTokenAtRe)
	}
	if !f.searched || f.next != nil && f.next[0] < pos {
		f.searched = true
		f.next = nextKeyName(f.whole, pos)
	}
	var loc []int
	limit := len(rest)
	if f.next != nil {
		loc = make([]int, len(f.next))
		for i, v := range f.next {
			loc[i] = -1
			if v >= 0 {
				loc[i] = v - pos
			}
		}
		limit = loc[2]
	}
	// A key joined to its prefix at a camelCase boundary, as in dbPassword,
	// has no word boundary before its credential name; it starts before the
	// key at a word boundary, if there is one.
	if camel := findCamelKeyName(rest, limit); camel != nil {
		loc = camel
	}
	if loc == nil {
		return nil
	}
	// Extend the key back over the prefix of a compound name. The match
	// starts at the key.
	name := loc[2]
	start := name
	for start > 0 && isTextKeyByte(rest[start-1]) {
		start--
	}
	loc[0], loc[2] = start, start
	return pairAtIf(textKeyAllowedAt(f.whole, pos+start, pos+loc[3]), rest, loc, name, textKeyValBareAtRe)
}

// nextKeyName returns the submatch indexes in s of the first key and
// delimiter at or after s[pos:] that textKeyNameRe finds with a value after
// them (see textValueStartRe), or nil. After a "temp:" key without a value,
// the search goes on from after its "temp:" prefix: a "temp:" name runs on to
// the first "=" or ":", so it may end in "temp" and its delimiter may be the
// colon of a second "temp:" key that starts inside it, as in temp:x&temp:&y=v,
// where the "&" after that colon leaves the first key without a value and the
// second runs on to its own "=". Any non-"temp:" key without a value holds no
// delimiter inside it, so any key starting inside it ends at the same
// delimiter and has no value either; advancing past loc[1] scans a run of
// keys without values (such as oauth-oauth-...x.code=,) in linear time.
func nextKeyName(s string, pos int) []int {
	for {
		loc := textKeyNameRe.FindStringSubmatchIndex(s[pos:])
		if loc == nil {
			return nil
		}
		for i, v := range loc {
			if v >= 0 {
				loc[i] = v + pos
			}
		}
		if textValueStartRe.MatchString(s[loc[1]:]) {
			return loc
		}
		if len(s[loc[2]:loc[3]]) >= len(session.KeyPrefixTemp) &&
			strings.EqualFold(s[loc[2]:loc[2]+len(session.KeyPrefixTemp)], session.KeyPrefixTemp) {
			pos = loc[2] + len(session.KeyPrefixTemp)
		} else {
			pos = loc[1]
		}
	}
}

// pairAt returns the submatch indexes in s of the pair whose key and
// delimiter are at loc, starting at the key, if textKeyAllowed accepts the
// key (see pairAtIf).
func pairAt(s string, loc []int, at int, re *regexp.Regexp) []int {
	return pairAtIf(textKeyAllowed(s[loc[2]:loc[3]]), s, loc, at, re)
}

// pairAtIf returns the submatch indexes in s of the pair whose key and
// delimiter are at loc, starting at the key, if allowed, which tells whether
// the key is to be redacted (see textKeyAllowed and textKeyAllowedAt): those
// of re matched at s[at:], where re reads the key, with the key of loc. re
// reads the value, which ends the pair. Otherwise it returns loc and no
// value, which redactTextKeyValue leaves in place, as the value is not read. A
// match of re whose key does not end where that of loc does is not taken (see
// TestKeyNameRegexps_FindThePairsOfPairRegexps).
func pairAtIf(allowed bool, s string, loc []int, at int, re *regexp.Regexp) []int {
	if allowed {
		if pair := re.FindStringSubmatchIndex(s[at:]); pair != nil && at+pair[3] == loc[3] {
			for i, v := range pair {
				if v >= 0 {
					pair[i] = v + at
				}
			}
			pair[0], pair[2] = loc[0], loc[2]
			return pair
		}
	}
	return append(loc[:8:8], -1, -1)
}

// maxFixedKeyNameLen is the length of the longest name in sensitiveKeys.
var maxFixedKeyNameLen = func() int {
	most := 0
	for k := range sensitiveKeys {
		most = max(most, len(k))
	}
	return most
}()

// isCamelTokenByte reports whether c may be in a name that a camelCase
// boundary splits: a letter, a digit, "_", "-" and "." (see
// bareTextKeyNameRegex).
func isCamelTokenByte(c byte) bool {
	return isTextKeyByte(c) || c == '.'
}

// findCamelKeyName returns the submatch indexes in s of the first key and
// delimiter, with a value after them, whose credential name starts before
// s[:limit] ends, at a camelCase boundary inside a longer name, such as the
// "Password" of dbPassword, or at the first word of a dotted OAuth name
// (see oauthNameStart), such as the oauth2Config of
// status.oauth2Config.extra.code, or nil. A boundary is a capital after a
// lower-case letter or a digit, or the last capital of an acronym, as in the
// "P" of DBPassword. A name may also start inside an acronym whose last
// capital starts the word after it (see insideAcronym), as the "API" of
// OpenAIAPIKey does.
//
// A credential name ends the token (the run of name bytes) that it is in, as
// the delimiter of the pair follows it, so only a token that is followed by
// a delimiter, possibly after a quote and whitespace, is read at its
// boundaries. Of its boundaries, only those that leave room for a name of
// sensitiveKeys are tried, and of its words, only the one where a dotted
// OAuth name may start, so the work for a token is linear in its length.
func findCamelKeyName(s string, limit int) []int {
	for t := 0; t < limit; {
		if !isCamelTokenByte(s[t]) {
			t++
			continue
		}
		e := t + 1
		for e < len(s) && isCamelTokenByte(s[e]) {
			e++
		}
		if tokenEndsPair(s, e) {
			if w := oauthNameStart(s, t, e); w >= 0 && w < limit {
				if loc := keyNameAt(s, w); loc != nil {
					return loc
				}
			}
			for i := camelScanStart(s, t, e); i < min(e, limit); i++ {
				if !isCamelBoundary(s, i) && !insideAcronym(s, i) {
					continue
				}
				if loc := keyNameAt(s, i); loc != nil {
					return loc
				}
			}
		}
		t = e
	}
	return nil
}

// camelScanStart returns the earliest offset in s[t+1:e] at which a camelCase
// or acronym credential suffix of s[t:e] can begin: at most maxFixedKeyNameLen
// non-separator bytes before e, skipping "_", "-" and "." separator runs so
// names such as stripeApi..key or OpenAIAPI__Key are found regardless of
// separator length while testing at most maxFixedKeyNameLen boundaries.
func camelScanStart(s string, t, e int) int {
	nonSep := 0
	i := e
	for i > t+1 {
		c := s[i-1]
		if c != '.' && c != '_' && c != '-' {
			nonSep++
			if nonSep > maxFixedKeyNameLen {
				break
			}
		}
		i--
	}
	return i
}

// keyNameAt returns the submatch indexes in s of the key and delimiter that
// textKeyNameCamelAtRe matches at s[i:], if a value follows them, or nil.
func keyNameAt(s string, i int) []int {
	loc := textKeyNameCamelAtRe.FindStringSubmatchIndex(s[i:])
	if loc == nil || !textValueStartRe.MatchString(s[i+loc[1]:]) {
		return nil
	}
	for j, v := range loc {
		if v >= 0 {
			loc[j] = v + i
		}
	}
	return loc
}

// oauthNameStart returns where the dotted OAuth name that
// isCompoundSensitiveKey accepts in the token s[t:e] would start, or -1: the
// token must end in ".code", and the name starts at the first of the dotted
// words before it that starts with "oauth". The words are read once.
func oauthNameStart(s string, t, e int) int {
	const oauth, code = "oauth", ".code"
	if e-t < len(oauth)+len(code) || !strings.EqualFold(s[e-len(code):e], code) {
		return -1
	}
	start := -1
	words := e - len(code)
	for w := t; ; {
		n := strings.IndexByte(s[w:words], '.')
		end := words
		if n >= 0 {
			end = w + n
		}
		if start < 0 && end-w >= len(oauth) && strings.EqualFold(s[w:w+len(oauth)], oauth) {
			start = w
		}
		if n < 0 {
			return start
		}
		w = end + 1
	}
}

// tokenEndsPair reports whether a key-value delimiter follows the name that
// ends at s[:i], possibly after a quote, which may be escaped, and
// whitespace.
func tokenEndsPair(s string, i int) bool {
	if i < len(s) && s[i] == '\\' {
		i++
	}
	if i < len(s) && (s[i] == '"' || s[i] == '\'') {
		i++
	}
	return followedByDelimiter(s, i)
}

// isCamelBoundary reports whether s[i] starts a word of a camelCase name.
func isCamelBoundary(s string, i int) bool {
	c, prev := s[i], s[i-1]
	if c < 'A' || c > 'Z' {
		return false
	}
	switch {
	case 'a' <= prev && prev <= 'z' || '0' <= prev && prev <= '9':
		return true
	case 'A' <= prev && prev <= 'Z':
		return i+1 < len(s) && 'a' <= s[i+1] && s[i+1] <= 'z'
	}
	return false
}

// insideAcronym reports whether s[i] is a capital inside a run of capitals,
// after the first, that goes on past it to either a last capital followed by a
// lower-case letter (as the "A" of the "APIK" of OpenAIAPIKey is) or a key
// separator (as the "A" of the "API_" of OpenAIAPI__Key is): the run then ends
// in an acronym, such as "API", so the acronym may start at s[i].
func insideAcronym(s string, i int) bool {
	isUpper := func(c byte) bool { return 'A' <= c && c <= 'Z' }
	if !isUpper(s[i]) || !isUpper(s[i-1]) {
		return false
	}
	j := i + 1
	for j < len(s) && isUpper(s[j]) {
		j++
	}
	if j >= len(s) {
		return false
	}
	return j-1 > i && 'a' <= s[j] && s[j] <= 'z' ||
		s[j] == '_' || s[j] == '-' || s[j] == '.'
}

// isTextKeyByte reports whether c may be in the prefix of a compound text key
// (see textKeyNameRegex).
func isTextKeyByte(c byte) bool {
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9' || c == '_' || c == '-'
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
// A quoted match whose opening quote closes an enclosing string (unless it
// owns the value it quotes, see ownsQuotedValue) is left in place. One whose
// closing quote does not end a value (see quotedValueEnds) is a token that
// runs on (see textValueEnd), as in Bearer "abc"def or
// Bearer "abc_token=" expired, unless the text between its quotes is the text
// between two strings (see textBetweenStrings), as in
// Bearer " in config, api_key="x", where it is left in place with its opening
// quote skipped (see openQuotes.unpaired), so that the text after it is
// searched again.
func redactBearerToken(s string, loc []int, quotes *openQuotes) (string, int) {
	end := loc[5]
	quoted := quotes.before(s, loc[4])
	value := s[loc[4]:end]
	if rest := s[loc[4]:]; len(rest) >= 2 && strings.EqualFold(rest[:2], "of") {
		if r, _ := utf8.DecodeRuneInString(rest[2:]); len(rest) == 2 || unicode.IsSpace(r) {
			quotes.skip(s, end)
			return s[loc[0]:end], end
		}
	}
	if q := openingQuote(s, loc[4]); q != "" {
		closes := closesString(s, loc[4], q, quoted)
		if closes && !ownsQuotedValue(s, loc[4], end, q) {
			quotes.skip(s, loc[4])
			return s[loc[0]:loc[4]], loc[4]
		}
		if end > loc[4]+len(q) && !quotedValueEnds(s[loc[4]:], end-loc[4], quoted) {
			if quote := textBetweenStrings(s, loc[4], end, quoted); quote != "" {
				next := quotes.unpaired(s, loc[4], quote)
				return s[loc[0]:next], next
			}
			end = textValueEnd(s, loc[4], end, 0, quoted)
			value = s[loc[4]:end]
		}
		if closes {
			quotes.drop()
		}
	}
	if strings.HasPrefix(value, redactedValue) {
		if redEnd := loc[4] + len(redactedValue); redEnd < end {
			if kind, _ := afterClosedQuotePair(s, redEnd, quoted); kind == nestedKey || startsPlainPair(s, redEnd) {
				quotes.skip(s, redEnd)
				return s[loc[0]:redEnd], redEnd
			}
		}
	}
	quotes.skip(s, end)
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
// "key":"[REDACTED]". The value ends where the value of a text key ends (see
// textValueEnd), with no bracket open before it and quotes tracking any
// enclosing string: a value that starts with a quote that nothing closes, as
// in {"password": "a b, runs on to the end of s (unless that quote closes an
// enclosing string and does not own the value it quotes, see
// ownsQuotedValue), a closed quoted value whose closing quote does not end a
// value (see quotedValueEnds) runs on, as in {"password": "a"b}, and one that
// opens a quoted value, an object or an array after a delimiter, as in
// {"password": a= "b c"}, runs to the end of that value. A closed quoted value
// that is the text between two strings (see textBetweenStrings), as the text
// of " in config, api_key=" in "token": " in config, api_key="x" is, is left
// in place with its opening quote skipped (see openQuotes.unpaired).
func redactJSONKeyValue(s string, loc []int, quotes *openQuotes) (string, int) {
	quoted := quotes.before(s, loc[4])
	q := openingQuote(s, loc[4])
	closes := q != "" && closesString(s, loc[4], q, quoted)
	if closes && !ownsQuotedValue(s, loc[4], loc[5], q) {
		quotes.skip(s, loc[4])
		return s[loc[0]:loc[4]], loc[4]
	}
	if quote := textBetweenStrings(s, loc[4], loc[5], quoted); quote != "" {
		next := quotes.unpaired(s, loc[4], quote)
		return s[loc[0]:next], next
	}
	end := textValueEnd(s, loc[4], loc[5], 0, quoted)
	if closes && end > loc[4] {
		quotes.drop()
	}
	quotes.skip(s, end)
	if end == loc[4] {
		return s[loc[0]:end], end
	}
	if inner, _ := unquoteValue(s[loc[4]:end]); isRedactedValue(inner) || isRedactedValue(trimTrailingClosingQuote(s[loc[4]:end], s, end, quoted)) {
		return s[loc[0]:end], end
	}
	return `"` + s[loc[2]:loc[3]] + `":"` + redactedValue + `"`, end
}

// redactTextKeyValue replaces a pair from keyValFinder, keeping the key, its
// quote, the delimiter and the quotes around the value. A value that is itself
// a sensitive key, as in the YAML "credentials:\n  token: ...", is left in
// place, so that the pair it starts is matched and redacted on its own (see
// nestsSensitiveKey). A value that only ends in a key name, as in
// "password=dev-SECRET-password: dial tcp ...", is redacted together with the
// value of the pair it starts (see nestedPairEnd). A quoted value ends at its
// closing quote, unless that quote does not end a value (see textValueEnd).
// A value that opens a JSON object or array is extended as in
// sensitiveValueEnd. A bare value runs on past any "}" or "]" in it and is
// redacted as a whole, except for the "}" and "]" at its end that close a
// bracket opened before the key, and the punctuation after them (see
// trimEnclosingClosers), as in the %+v output {APIKey:...} or
// [{APIKey:...}]. brackets counts the brackets opened before the key, and
// quotes the quotes that leave a string open before the value.
func redactTextKeyValue(s string, loc []int, brackets *openBrackets, quotes *openQuotes) (string, int) {
	if loc[8] < 0 {
		// A compound name that does not end in a credential name, such as
		// next_page_token, which textKeyAllowedAt does not accept, and whose
		// value pairAtIf has not read. What follows the delimiter is scanned
		// again.
		return s[loc[0]:loc[7]], loc[7]
	}
	open := brackets.before(s, loc[0])
	start, end := loc[8], loc[9]
	quoted := quotes.before(s, start)
	if loc[4] < loc[5] && (quoted&quoteKind(openingQuote(s, start)) != 0 || s[loc[5]:loc[7]] != "=") {
		// The value of a quoted key that is the text between two strings, as
		// the value of "token": " in config, api_key="secret" is, is not its
		// value: the text after its opening quote is searched again (see
		// redactJSONKeyValue). The value of a key that "=" follows at once,
		// outside an enclosing string of the same quote kind, as in
		// export "DB_PASSWORD"=" a b"c, is not that text, as the quotes of a
		// shell command join words into one value.
		if quote := textBetweenStrings(s, start, end, quoted); quote != "" {
			next := quotes.unpaired(s, start, quote)
			brackets.skip(next)
			return s[loc[0]:next], next
		}
	}
	switch kind, pair := nestsSensitiveKey(s, start, end, startsLine(s, loc[7], start), true, quoted); kind {
	case nestedKey:
		return s[loc[0]:start], start
	case valueEndingInKeyName:
		_, quote := unquoteValue(s[start:end])
		end = nestedPairEnd(s, pair, end, open, quoted)
		brackets.skip(end)
		quotes.skip(s, end)
		return s[loc[2]:loc[3]] + s[loc[4]:loc[5]] + s[loc[6]:loc[7]] + quote + redactedValue + quote, end
	}
	end = textValueEnd(s, start, end, open, quoted)
	brackets.skip(end)
	if q := openingQuote(s, start); q != "" && end > start && closesString(s, start, q, quoted) {
		// The opening quote of the value, which would close the string open
		// before it, opens the value instead (see ownsQuotedValue).
		quotes.drop()
	}
	quotes.skip(s, end)
	if end == start {
		// Only closing brackets or an outer closing quote follow the delimiter,
		// as in {Password:} or failed: "password=" in config.
		return s[loc[0]:end], end
	}
	inner, quote := unquoteValue(s[start:end])
	if isRedactedValue(inner) || isRedactedValue(trimTrailingClosingQuote(s[start:end], s, end, quoted)) {
		return s[loc[0]:end], end
	}
	return s[loc[2]:loc[3]] + s[loc[4]:loc[5]] + s[loc[6]:loc[7]] + quote + redactedValue + quote, end
}

// textValueEnd returns the end of the value matched at s[start:end] after a
// key: a quoted value ends at its closing quote (unless its opening quote
// closes an outer string open before the value and the value does not own it,
// see ownsQuotedValue), a value that opens a JSON object or array is extended
// as in sensitiveValueEnd, and a bare value ends before the "}" and "]" at its
// end that close a bracket opened before the key, and the punctuation after
// them (see trimEnclosingClosers). open is the number of brackets opened
// before the key. A closed quoted value whose closing quote does not end a
// value (see quotedValueEnds) runs on, as in ""secret"" or "abc"def (see
// runOnValueEnd). A bare value that holds or is followed after horizontal
// whitespace by a delimiter followed by a quoted value, an object or an array
// that runs on past it, as in a:b="c d", a={"b": "c"}, a = "c d" or
// a : {"b": "c"}, is extended to the end of that value (see openedValueEnd
// and spacedValueEnd), so that no part of it is left after the redacted value
// without a key. A value that starts with a quote that is not closed, as in
// password="correct horse battery, is read to the end of s, as no closing
// quote tells where it ends, unless the quote does not open a value (see
// unclosedQuoteOpens): quoted holds the quotes that leave a string open before
// the value.
func textValueEnd(s string, start, end, open int, quoted quoteKinds) int {
	if q := openingQuote(s, start); q != "" {
		if closesString(s, start, q, quoted) && !ownsQuotedValue(s, start, end, q) {
			return start
		}
		if _, quote := unquoteValue(s[start:end]); quote != "" {
			if quotedValueEnds(s[start:], end-start, quoted) {
				return end
			}
			return runOnValueEnd(s, start, end, open, quoted)
		}
		if unclosedQuoteOpens(s, start, q, true, quoted) {
			return len(s)
		}
	}
	// rest is where the text after a JSON object or array value starts.
	rest := sensitiveValueEnd(s, start, start)
	if rest < end {
		if opened := openedValueEnd(s, rest, end, open, false, quoted); opened > end {
			return opened
		}
	}
	if rest == start {
		if spaced := spacedValueEnd(s, end, open, quoted); spaced > end {
			return spaced
		}
	}
	return rest + len(trimEnclosingClosers(s[rest:max(end, rest)], open))
}

// runOnValueEnd returns the end of the closed quoted value s[start:end] whose
// closing quote does not end a value (see quotedValueEnds), which is not
// where the value ends but where it runs on to, so that none of the text that
// the quote runs into is left in place after the redacted value: to the end
// of the value that the closing quote opens for a later key (see
// forwardValueEnd), as the closing quote of " in config, api_key=" does, or
// else to the end of the word that the quote runs into (see runOnWordEnd), as
// in ""secret"" or "abc"def, without the closing brackets and punctuation at
// its end that close brackets opened before the key (see trimEnclosingClosers,
// where open is their number). quoted holds the quotes that leave a string
// open before the value. A closing quote followed at once by a standalone
// sensitive key and its delimiter (even with whitespace before its value), or
// by an ordinary unspaced pair such as user="b", ends at the closing quote so
// that the pair is read on its own; a compound name ending in a sensitive key
// suffix on the same line, as in "abc"tail_token:8080, may be a credential
// tail and is redacted together with its pair through nestedPairEnd.
func runOnValueEnd(s string, start, end, open int, quoted quoteKinds) int {
	inner, _ := unquoteValue(s[start:end])
	if opensLaterValue(inner, s[end:]) {
		return forwardValueEnd(s, start, end, open, quoted)
	}
	switch kind, pair := afterClosedQuotePair(s, end, quoted); kind {
	case nestedKey:
		return end
	case valueEndingInKeyName:
		return nestedPairEnd(s, pair, end, open, quoted)
	}
	if startsPlainPair(s, end) {
		return end
	}
	word := runOnWordEnd(s, end, quoted)
	return end + len(trimEnclosingClosers(s[end:word], open))
}

// runOnWordEnd returns the end of the word that starts at s[i:]: the first
// whitespace, ",", ";" or "&" in it that is not inside a quoted value (see
// quotedValueRegex) that starts in it and closes on its line, or the end of
// the line if an unclosed joined quote opens a value (see unclosedQuoteOpens),
// so that a quoted string joined to a credential is read whole and its words
// are not left in place, as the 'b c' of the shell word 'a'"'"'b c' and the
// words of an SQL string that doubles its quote would be. A quote does not
// start a quoted value if it closes a string that is open before the
// credential in quoted (see endsString), or if it is an apostrophe inside a
// word, as in it's (see isMidWordApostrophe), unless a quoted value ends right
// before it, as the one of 'a'"b"c'd e' does.
func runOnWordEnd(s string, i int, quoted quoteKinds) int {
	joined := true
	for i < len(s) {
		if q := openingQuote(s, i); q != "" && !endsString(s, i, q, quoted) && (joined || q != "'" || !isMidWordApostrophe(s, i) || opensPrefixedString(s, i)) {
			m := quotedValueRe.FindString(s[i:])
			if m != "" && !strings.ContainsAny(m, "\r\n") {
				i += len(m)
				joined = true
				continue
			}
			lineEnd := len(s)
			if nl := strings.IndexAny(s[i:], "\r\n"); nl >= 0 {
				lineEnd = i + nl
			}
			if unclosedQuoteOpens(s[:lineEnd], i, q, true, quoted) {
				return lineEnd
			}
		}
		if strings.IndexByte(regexpSpace+",;&", s[i]) >= 0 {
			break
		}
		i++
		joined = false
	}
	return i
}

// afterClosedQuotePair classifies a sensitive key-value pair that begins
// immediately at s[end:] on the same line after a closed quoted value:
// nestedKey for a standalone sensitive key name (even when whitespace separates
// its delimiter from its value) or an unspaced compound sensitive assignment
// with an immediate value (as in DB_PASSWORD="a"GITHUB_TOKEN="b"),
// valueEndingInKeyName and the offset of the sensitive key name for a compound
// name ending in a credential suffix before ":port" or spaced delimiters, or
// notNestedKey when no sensitive pair with a value starts at s[end:].
func afterClosedQuotePair(s string, end int, quoted quoteKinds) (nestedKeyKind, int) {
	n := 0
	for end+n < len(s) && isCamelTokenByte(s[end+n]) {
		if n == maxNestedKeyLen {
			break
		}
		n++
	}
	if n > 0 && n < maxNestedKeyLen && isIdentifierKey(s[end:end+n]) {
		delim := skipHorizDelimSpace(s, end+n)
		if delim < len(s) && (s[delim] == ':' || s[delim] == '=') &&
			!(s[delim] == '=' && delim+1 < len(s) && s[delim+1] == '=') {
			name := s[end : end+n]
			if !isPageTokenName(name) {
				if kind, pair := nestedKeyName(name, end, false); kind != notNestedKey {
					if valueStartsAt(s, delim+1, true, quoted) {
						if kind == valueEndingInKeyName && delim == end+n && delim+1 < len(s) {
							if r, _ := utf8.DecodeRuneInString(s[delim+1:]); !isDelimSpace(r) && (s[delim] == '=' || openingQuote(s, delim+1) != "") {
								return nestedKey, pair
							}
						}
						return kind, pair
					}
					return notNestedKey, 0
				}
			}
		}
	}
	if startsPlainPair(s, end) {
		return notNestedKey, 0
	}
	if n == maxNestedKeyLen || (n > 0 && end+n < len(s) && s[end+n] == ':' && openingQuote(s, end+n+1) == "") {
		bareEnd := end
		for bareEnd < len(s) && strings.IndexByte(regexpSpace+",;&", s[bareEnd]) < 0 {
			bareEnd++
		}
		if bareEnd > end {
			return nestsSensitiveKey(s, end, bareEnd, false, true, quoted)
		}
	}
	return notNestedKey, 0
}

// startsPlainPair reports whether s[end:] starts with an ordinary unspaced
// pair: an identifier key no longer than maxNestedKeyLen, "=" (not "==") or
// ":" before a quoted value, and a value that starts at once (see
// valueStartsAt), as in user="bob" or label:"bob".
func startsPlainPair(s string, end int) bool {
	n := 0
	for end+n < len(s) && isCamelTokenByte(s[end+n]) {
		if n == maxNestedKeyLen {
			return false
		}
		n++
	}
	if n == 0 || end+n+1 >= len(s) || !isIdentifierKey(s[end:end+n]) || (s[end+n] != ':' && s[end+n] != '=') {
		return false
	}
	if r, _ := utf8.DecodeRuneInString(s[end+n+1:]); isDelimSpace(r) || strings.IndexByte(",;&)]}=", s[end+n+1]) >= 0 || !valueStartsAt(s, end+n+1, false, 0) {
		return false
	}
	return s[end+n] == '=' || openingQuote(s, end+n+1) != ""
}

// spacedValueEnd returns the end of a quoted value, JSON object or JSON array
// opened after horizontal whitespace and a ":" or "=" delimiter immediately
// following the bare value ending at end, as in password=a = "b c" or
// token=a : {"b": "c"}, or end when none follows. A quoted value whose
// closing quote does not end a value, as in password=a = "b c"d, runs on (see
// runOnValueEnd, where open is as for textValueEnd). Bare tokens after a
// spaced delimiter, as in failed: password: x : retry, are left alone.
func spacedValueEnd(s string, end, open int, quoted quoteKinds) int {
	at := skipHorizDelimSpace(s, end)
	if at == end || at >= len(s) || (s[at] != ':' && s[at] != '=') {
		return end
	}
	at = skipDelimSpace(s, at+1)
	if at >= len(s) {
		return end
	}
	if c := s[at]; c == '{' || c == '[' {
		return sensitiveValueEnd(s, at, at)
	}
	if quote := openingQuote(s, at); quote != "" {
		if closesString(s, at, quote, quoted) {
			return end
		}
		if q := quotedValueRe.FindString(s[at:]); q != "" {
			if !quotedValueEnds(s[at:], len(q), quoted) {
				return runOnValueEnd(s, at, at+len(q), open, quoted)
			}
			if valueStartsAt(s, at, true, quoted) {
				return at + len(q)
			}
			return end
		}
		if unclosedQuoteOpens(s, at, quote, true, quoted) {
			return len(s)
		}
	}
	return end
}

// openedValueEnd returns the end of the value that a delimiter, ":" or "=",
// in the bare value s[start:end] opens, possibly after whitespace, if it runs
// on past end: a quoted value (see quotedValueRegex), which runs on to the
// end of s if its quote is not closed (see unclosedQuoteOpens), or a JSON
// object or array, which runs on to the end of s if it is not closed (see
// sensitiveValueEnd). Otherwise it returns end. The delimiter may end the
// bare value, as the one of a: {"b": "c"} does, whose bare value is "a:". A
// value opened and closed inside s[start:end] is skipped, so that the text is
// read once. Unless trailingQuoteOpens is set, a quote that ends the bare
// value, so that what it quotes starts with the whitespace, "," or ";" that
// ended the bare value, or that follows the bare value, opens a value only if
// it does not close an outer string open before the value (see closesString)
// and, after the bare value, holds more than whitespace and punctuation (see
// valueStartsAt): otherwise it more likely closes a string that the bare
// value is in and opens the next, as the quotes around ", want " do in the
// text got "password=abc-token=", want "x". Inside the bare value, where a
// value may follow the quote in the same word, as in 'password=a:':{...} or
// "password=abc:{\"k\": \"v\"}", it is not treated as closing the outer string;
// but if its quoted value runs on past the bare value, it opens a value only
// if its closing quote does not open the value of a later key, sensitive or
// not (see opensLaterPair), unlike the apostrophe in token=abc:'s label=' x',
// and not if its quotes are the apostrophes of prose, one after a ":" that
// starts the ending of a contraction or a possessive and one inside a word
// that starts another (see isContraction), as in
// token=abc:'s scope missing; the user's request. A quoted value that opens a
// value but does not end where a value may end (see quotedValueEnds), as in
// password=a = "b"c, password=a="b"c or password=a="b"'c', runs on (see
// runOnValueEnd, where open is as for textValueEnd), so that the credential
// in it is not left in place after the redacted bare value.
func openedValueEnd(s string, start, end, open int, trailingQuoteOpens bool, quoted quoteKinds) int {
	for i := start; i < end; i++ {
		if s[i] != ':' && s[i] != '=' {
			continue
		}
		at := skipDelimSpace(s, i+1)
		if at == len(s) {
			continue
		}
		var e int
		if c := s[at]; c == '{' || c == '[' {
			e = sensitiveValueEnd(s, at, at)
		} else if quote := openingQuote(s, at); quote != "" {
			checkClose := !trailingQuoteOpens || at >= end
			if checkClose && at+len(quote) >= end && closesString(s, at, quote, quoted) {
				return end
			}
			q := quotedValueRe.FindString(s[at:])
			if q == "" {
				if n := contractionLen(s[at+len(quote):]); n > 0 && at == i+1 && at+len(quote)+n == end && s[i] == ':' && quote == "'" && proseFollowsContraction(s[end:]) {
					return end
				}
				if unclosedQuoteOpens(s, at, quote, at >= end, quoted) {
					return len(s)
				}
				return end
			}
			// A quote inside the bare value whose quoted value runs on past
			// it and ends where its closing quote opens a later pair, a
			// Bearer token or a quoted key, as in token=abc:'s label=' x',
			// token=abc:'s note Bearer ' x' or token=abc:'s note 'password': ' x',
			// opens no value.
			if at < end && at+len(q) > end {
				if inner, _ := unquoteValue(q); opensLaterConstruct(s, at, len(q), len(quote), inner) {
					return end
				}
			}
			// A quote that ends the bare value, or follows it, opens a quoted
			// value that does not end where a value may end, as in
			// password=a="b"c, even if trailingQuoteOpens is set.
			if at+len(quote) >= end && !quotedValueEnds(s[at:], len(q), quoted) {
				return runOnValueEnd(s, at, at+len(q), open, quoted)
			}
			// A quote inside the bare value whose quoted value runs on past
			// it but does not end there opens a value that runs on, as in
			// token=a:'s b 1'x or password=a='d b'c, unless an outer ";"
			// clause boundary separates two independent prose contractions,
			// as in token=abc:'s missing; the user's request.
			if at+len(quote) < end && at+len(q) > end && !quotedValueEnds(s[at:], len(q), quoted) {
				if closing := at + len(q) - len(quote); quote == "'" && isProseContractions(s, i, at, end, closing) {
					return end
				}
				return runOnValueEnd(s, at, at+len(q), open, quoted)
			}
			// A quote or value that follows a quoted value closed in or at the
			// end of the bare value, as in token=a:'b'' c d' or
			// password=a:'x_token=' "b c", is read with it (see runOnValueEnd).
			if at+len(q) <= end && !quotedValueEnds(s[at:], len(q), quoted) {
				if w := runOnValueEnd(s, at, at+len(q), open, quoted); w > end {
					return w
				}
			}
			if at >= end && !valueStartsAt(s, at, true, quoted) {
				return end
			}
			e = at + len(q)
		} else {
			continue
		}
		if e > end {
			return e
		}
		i = e - 1
	}
	return end
}

// openingQuote returns the quote, ", ', \" or \', that starts s[i:], or "".
func openingQuote(s string, i int) string {
	switch {
	case i < len(s) && (s[i] == '"' || s[i] == '\''):
		return s[i : i+1]
	case i+1 < len(s) && s[i] == '\\' && (s[i+1] == '"' || s[i+1] == '\''):
		return s[i : i+2]
	}
	return ""
}

// unclosedQuoteOpens reports whether the quote q at s[i:], which no quote of
// its kind closes (see quotedValueRegex), opens a value that runs on to the
// end of s, where only the end of s tells where the value ends. It does not
// if only whitespace and the closing brackets, quotes and punctuation that
// may end a value follow it (see closingPunctuation), as the quote in
// {Password:abc-token: "} does, nor, if mayClose is set, if it more likely
// closes a string that is open before the value (see closesString).
func unclosedQuoteOpens(s string, i int, q string, mayClose bool, quoted quoteKinds) bool {
	if !holdsValue(s[i:]) {
		return false
	}
	return !mayClose || !closesString(s, i, q, quoted)
}

// closesString reports whether the quote q at s[i:], closes a string that is
// open before the value rather than opens one: a quote of its kind leaves a
// string open there (quoted holds those), and the quote ends s or is followed
// by delimiter whitespace or by punctuation that may follow a string, as in
// parsing "password=abc-token: ": invalid syntax. A full stop, "!" or "?"
// after a quote that may open a value does not count, as a value may start
// with one, as "!Passw0rd" does in failed: "password="!Passw0rd"" (see
// endsString). Nor does a quote that opens a value nested at the end of the
// string (see opensNestedValue), as the second quote of
// failed: "password="   secret"" does.
func closesString(s string, i int, q string, quoted quoteKinds) bool {
	return quoteEndsString(s, i, q, quoted, ",;:)]}") && !opensNestedValue(s, i, q, quoted)
}

// opensNestedValue reports whether the quote q at s[i:], which is followed by
// delimiter whitespace and so may close a string open before it, opens a
// value nested at the end of that string instead: a quote of its kind closes
// the value, which holds a value (see holdsValue) but does not end in a
// sensitive key and its delimiter (see endsInSensitiveKeyDelimiter), whose
// value the closing quote more likely opens, and is followed, possibly after
// closing brackets and punctuation, by a quote of its kind that ends the
// string (see endsString), as in failed: "password="   secret"" or
// failed: "{"password": "   secret"}".
func opensNestedValue(s string, i int, q string, quoted quoteKinds) bool {
	if after := i + len(q); after == len(s) || skipDelimSpace(s, after) == after {
		return false
	}
	m := quotedValueRe.FindString(s[i:])
	inner, quote := unquoteValue(m)
	if quote != q || !holdsValue(inner) || endsInSensitiveKeyDelimiter(inner) {
		return false
	}
	at := i + len(m)
	for at < len(s) && strings.IndexByte(".,;!?)]}", s[at]) >= 0 {
		at++
	}
	return strings.HasPrefix(s[at:], q) && endsString(s, at, q, quoted)
}

// endsString reports whether the quote q at s[i:], which follows a value,
// closes a string that is open before the value, as closesString does, but
// also when a full stop, "!" or "?" follows it, as one does at the end of the
// sentence failed: "password=a: "x"".
func endsString(s string, i int, q string, quoted quoteKinds) bool {
	return quoteEndsString(s, i, q, quoted, ",;:.!?)]}")
}

// quoteEndsString reports whether a quote of the kind of the quote q at s[i:]
// leaves a string open before it (quoted holds those), and the quote ends s
// or is followed by delimiter whitespace or by a byte of punct.
func quoteEndsString(s string, i int, q string, quoted quoteKinds, punct string) bool {
	if quoted&quoteKind(q) == 0 {
		return false
	}
	after := i + len(q)
	return after == len(s) || skipDelimSpace(s, after) > after || strings.IndexByte(punct, s[after]) >= 0
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
func nestsSensitiveKey(s string, start, end int, lineStart, mayClose bool, quoted quoteKinds) (nestedKeyKind, int) {
	if s[start] == '{' || s[start] == '[' {
		return notNestedKey, 0
	}
	if end-start > maxNestedKeyLen {
		return oversizedNestedKey(s, start, end, lineStart, mayClose, quoted)
	}
	v := s[start:end]
	if inner, quote := unquoteValue(v); quote != "" {
		if !valueFollowsDelimiter(s, end, mayClose, quoted) {
			return notNestedKey, 0
		}
		return nestedKeyName(inner, start+len(quote), lineStart)
	}
	if strings.HasSuffix(v, ":") || strings.HasSuffix(v, "=") {
		if !valueStartsAt(s, end, mayClose, quoted) {
			return notNestedKey, 0
		}
		return nestedKeyName(v[:len(v)-1], start, lineStart)
	}
	if followedByDelimiter(s, end) {
		if !valueFollowsDelimiter(s, end, mayClose, quoted) {
			return notNestedKey, 0
		}
		return nestedKeyName(v, start, lineStart)
	}
	// A name and a delimiter followed by a quoted value that holds
	// whitespace, ",", ";" or "&", or is unclosed, as in
	// credentials: password="a b", where the value matched ends inside the
	// quoted value.
	if q := strings.IndexAny(v, `"'\`); q > 0 && (v[q-1] == ':' || v[q-1] == '=') {
		qv := quotedValueRe.FindString(s[start+q:])
		if (len(qv) > end-start-q || (qv == "" && openingQuote(s, start+q) != "")) &&
			valueStartsAt(s, start+q, mayClose, quoted) {
			if kind, pair := nestedKeyName(v[:q-1], start, lineStart); kind != notNestedKey {
				return kind, pair
			}
		}
	}
	// A pair that an earlier pass redacted, as in "token:[REDACTED]", or
	// token:"[REDACTED]"} in the %+v output of a struct.
	i := strings.LastIndexAny(v, ":=")
	if i <= 0 {
		return notNestedKey, 0
	}
	if inner, _ := unquoteValue(trimTrailingPunctuation(trimTrailingClosingQuote(v[i+1:], s, end, quoted))); !isRedactedValue(inner) {
		return notNestedKey, 0
	}
	return nestedKeyName(v[:i], start, lineStart)
}

// oversizedNestedKey classifies a value longer than maxNestedKeyLen that
// nestsSensitiveKey has read at s[start:end]. Its name is not read, but a
// sensitive key and a value that follow a delimiter are redacted together
// rather than the key only, which would leave the value in place: the value
// is a key when it ends with a delimiter or is followed by one, or is a
// quoted name followed by one, or a name and a delimiter and a redacted
// value or the start of a quoted value that runs on past it, and a pair with
// a sensitive key starts at its name, or after a dot or a colon in it. A key
// that starts a line is left in place, so that its pair is matched on its
// own (nestedKey); any other is redacted with the value of its pair
// (valueEndingInKeyName). Only the delimiter and the value that follow it
// are read besides the pair that the regular expressions match, which are
// matched once, as the pair is redacted whole.
func oversizedNestedKey(s string, start, end int, lineStart, mayClose bool, quoted quoteKinds) (nestedKeyKind, int) {
	name := start
	v := s[start:end]
	// keyText is the text of the key, which starts at s[name:].
	keyText := v
	if inner, quote := unquoteValue(v); quote != "" {
		if !valueFollowsDelimiter(s, end, mayClose, quoted) {
			return notNestedKey, 0
		}
		name += len(quote)
		keyText = inner
	} else if strings.HasSuffix(v, ":") || strings.HasSuffix(v, "=") {
		if !valueStartsAt(s, end, mayClose, quoted) {
			return notNestedKey, 0
		}
		keyText = v[:len(v)-1]
	} else if !valueFollowsDelimiter(s, end, mayClose, quoted) {
		// A name and a delimiter followed by a quoted value that holds
		// whitespace, ",", ";" or "&", or is unclosed, as in name="a b",
		// where the value matched ends inside the quoted value.
		q := strings.IndexAny(v, `"'\`)
		qv := ""
		if q > 0 && (v[q-1] == ':' || v[q-1] == '=') {
			qv = quotedValueRe.FindString(s[start+q:])
		}
		if q > 0 && (v[q-1] == ':' || v[q-1] == '=') &&
			(len(qv) > end-start-q || (qv == "" && openingQuote(s, start+q) != "")) &&
			valueStartsAt(s, start+q, mayClose, quoted) {
			// The pair that starts at the name is read below.
			keyText = v[:q-1]
		} else {
			// A pair that an earlier pass redacted, as in "name:[REDACTED]".
			i := strings.LastIndexAny(v, ":=")
			if i <= 0 {
				return notNestedKey, 0
			}
			if inner, _ := unquoteValue(trimTrailingPunctuation(trimTrailingClosingQuote(v[i+1:], s, end, quoted))); !isRedactedValue(inner) {
				return notNestedKey, 0
			}
			keyText = v[:i]
		}
	}
	if _, n := cutStatePrefix(s[name:]); n > 0 {
		name += n
		keyText = keyText[min(n, len(keyText)):]
	}
	if !allowedKeyAt(s, name) {
		// A name after a dot or a colon in the key text, as in x.y_PASSWORD
		// or a run of keys whose first is not a credential, as in
		// next_page_token:x_PASSWORD, is read from after it: the name is the
		// part that textKeyValTokenAtRe reads. Only the name after each dot
		// or colon is read until one is allowed (see allowedKeyAt), so that a
		// run of names that are not, as in
		// x.next_page_token:x.next_page_token:..., is read once rather than
		// to its end from each one. A dotted OAuth name runs on to the last
		// word of its run of dotted words, so a run whose words are not one,
		// as in oauth.oauth.oauth..., is skipped to its last word once a name
		// that holds "oauth" has been read in it (see oauthRunLastName).
		found := false
		next := oauthRunLastName(s, name)
		for d := strings.IndexAny(keyText, ".:"); d >= 0; {
			if at := name + d + 1; at >= next {
				if allowedKeyAt(s, at) {
					name = at
					found = true
					break
				}
				next = oauthRunLastName(s, at)
			}
			n := strings.IndexAny(keyText[d+1:], ".:")
			if n < 0 {
				break
			}
			d += 1 + n
		}
		if !found {
			return notNestedKey, 0
		}
	}
	if lineStart {
		return nestedKey, name
	}
	return valueEndingInKeyName, name
}

// oauthRunLastName returns where oversizedNestedKey reads on after
// allowedKeyAt(s, i) has not allowed a key at s[i:]: where the last word of
// the run of dotted words (skipping consecutive dots) that starts at s[i:]
// starts, if the first word holds "oauth", or else i. Within such a run, a
// word that a dot follows, in which textKeyNameAtRe reads no name of
// sensitiveKeys and no "temp:" name, can only start a dotted OAuth name (see
// oauthKeyNameRegex), and that runs on to the last word of the run, as the
// one read at the first word did: its words are among those of that one, so
// if that one is not allowed, neither is it. The words are made of ASCII
// letters, digits, "_" and "-", as those of oauthKeyNameRegex are.
func oauthRunLastName(s string, i int) int {
	j := i
	for j < len(s) && isTextKeyByte(s[j]) {
		j++
	}
	if !containsFoldASCII(s[i:j], "oauth") {
		return i
	}
	last := i
	for j < len(s) && s[j] == '.' {
		for j < len(s) && s[j] == '.' {
			j++
		}
		if j < len(s) && isTextKeyByte(s[j]) {
			last = j
			for j < len(s) && isTextKeyByte(s[j]) {
				j++
			}
		}
	}
	return last
}

// containsFoldASCII reports whether s contains sub, a lower-case ASCII word,
// ignoring the case of ASCII letters.
func containsFoldASCII(s, sub string) bool {
	for k := 0; k+len(sub) <= len(s); k++ {
		if strings.EqualFold(s[k:k+len(sub)], sub) {
			return true
		}
	}
	return false
}

// allowedKeyAt reports whether a pair whose key textKeyAllowedAt accepts starts
// at s[i:], as textKeyValTokenAtRe matches it, reading only the name of the
// key, the delimiter and the start of the value, which may run on to the end
// of s. textKeyValTokenAtRe matches where textKeyNameAtRe does and a value
// starts after the delimiter, and with the same name: none of the other names
// that it may read at s[i:] ends before a delimiter, since a name holds no
// delimiter but the colon of a "temp:" prefix, and no name ends in "temp".
func allowedKeyAt(s string, i int) bool {
	loc := textKeyNameAtRe.FindStringSubmatchIndex(s[i:])
	return loc != nil && textKeyAllowedAt(s, i+loc[2], i+loc[3]) && textValueStartRe.MatchString(s[i+loc[1]:])
}

// cutStatePrefix returns name without a "user:" or "app:" state prefix, as
// isSensitiveKey reads one, and the length of the prefix, or name and 0.
func cutStatePrefix(name string) (string, int) {
	for _, prefix := range []string{session.KeyPrefixUser, session.KeyPrefixApp} {
		if len(name) >= len(prefix) && strings.EqualFold(name[:len(prefix)], prefix) {
			return name[len(prefix):], len(prefix)
		}
	}
	return name, 0
}

// compoundKeyStart returns the offset within a compound key name (one
// accepted by isCompoundSensitiveKey) where textKeyValTokenAtRe matches it:
// at its first "oauth" word for a dotted OAuth ".code" key, or else at the
// first of its dotted words from which the rest of the name is a whole text
// key name (see textKeyNameWholeRe), as the "stripe_api.key" of
// my.stripe_api.key and the "dbPassword" of secretPrefix.dbPassword are, or,
// if none is, after its last ".".
func compoundKeyStart(name string) int {
	if o := oauthNameStart(name, 0, len(name)); o >= 0 {
		return o
	}
	for d := 0; ; {
		if textKeyNameWholeRe.MatchString(name[d:]) {
			return d
		}
		n := strings.IndexByte(name[d:], '.')
		if n < 0 {
			return d
		}
		d += n + 1
	}
}

// nestedKeyName classifies a value that may start a nested pair (see
// nestsSensitiveKey): name is the name of that pair, which begins at offset
// at of the text, and lineStart reports whether the value starts a line.
// Unless the value is notNestedKey, nestedKeyName also returns where the
// sensitive key name in name, and so the pair that it starts, begins in the
// text.
func nestedKeyName(name string, at int, lineStart bool) (nestedKeyKind, int) {
	// A child of a state key, such as user:DB_PASSWORD, is the key without
	// its prefix, which the pair is read from.
	if rest, n := cutStatePrefix(name); n > 0 {
		if kind, pair := nestedKeyName(rest, at+n, lineStart); kind != notNestedKey {
			return kind, pair
		}
	}
	if _, ok := sensitiveKeys[name]; ok {
		return nestedKey, at
	}
	if m := nestedKeyNameRe.FindStringSubmatchIndex(name); m != nil {
		return nestedKey, at + m[2]
	}
	m := compoundKeyNameRe.FindStringSubmatchIndex(name)
	switch {
	case m == nil:
		// A compound name such as DB_PASSWORD, dbPassword, my.stripe_api.key
		// or status.oauth2Config.extra.code is not matched by
		// compoundKeyNameRe, which wants a non-word character before the
		// credential name. It is a key all the same: a child pair such as
		// "DB_PASSWORD: secret" under "credentials:" must have its own value
		// redacted, not be consumed as the value of the parent.
		if isCompoundSensitiveKey(name) {
			pairAt := at + compoundKeyStart(name)
			if lineStart {
				return nestedKey, pairAt
			}
			return valueEndingInKeyName, pairAt
		}
		// So is the last name of a run of names in one word, as in
		// next_page_token:x_TOKEN, whose first is not a credential: the value
		// that follows the word is the value of its last name.
		if i := strings.LastIndexByte(name, ':'); i >= 0 && isCompoundSensitiveKey(name[i+1:]) {
			pairAt := at + i + 1 + compoundKeyStart(name[i+1:])
			if lineStart {
				return nestedKey, pairAt
			}
			return valueEndingInKeyName, pairAt
		}
		return notNestedKey, 0
	case lineStart:
		return nestedKey, at + m[2]
	default:
		return valueEndingInKeyName, at + m[2]
	}
}

// nestedPairEnd returns where the text to redact with a value that only ends
// in a key name ends (see nestsSensitiveKey), given pair, where the pair that
// the key name starts begins in s, and valueEnd, where the value that holds
// the key name ends: at the end of the value of that pair, or, if that value
// is itself the name of a pair, as in "password=a-token: token: x", at the
// end of the value of that pair, and so on. open is the number of brackets
// opened before the key of the value, and quoted the quotes that leave a
// string open before the value.
func nestedPairEnd(s string, pair, valueEnd, open int, quoted quoteKinds) int {
	// A value longer than maxNestedKeyLen is read as a key only while each
	// one ends further right than the one before it: keys apart from each
	// other, as in a: KEY: KEY: KEY="x y", are read once each, but a run of
	// them in one word, as in token:token:token:..., would be matched again
	// from each one of them, and is stopped at the second.
	prevEnd := -1
	for {
		loc := textKeyValTokenAtRe.FindStringSubmatchIndex(s[pair:])
		if loc == nil {
			// Not reached: nestsSensitiveKey has read a delimiter and a
			// value after a key name that textKeyValTokenAtRe matches. Were
			// the name not matched, the value that holds it would still be
			// redacted to its end, so that no part of it is left in place.
			return max(pair, valueEnd)
		}
		start, end := pair+loc[8], pair+loc[9]
		if end-start > maxNestedKeyLen && end <= prevEnd {
			return oversizedValueEnd(s, start, end, open, quoted)
		}
		prevEnd = end
		kind, next := nestsSensitiveKey(s, start, end, startsLine(s, pair+loc[7], start), true, quoted)
		if kind == notNestedKey {
			return textValueEnd(s, start, end, open, quoted)
		}
		pair, valueEnd = next, end
	}
}

// oversizedValueEnd returns the end of the value of the last key of a word
// longer than maxNestedKeyLen that holds a run of keys, s[start:end], which
// starts after the delimiter of a key: the object or array that the value
// opens, the quoted value, object or array that a delimiter in the word opens
// (see openedValueEnd), even if a quote ends the word, when it runs on past
// the word, or else the value that follows the word and a delimiter (adjacent
// to it or after whitespace): a quoted value (unless its opening quote closes
// an outer string open before the value, and a quoted value whose closing
// quote does not end a value runs on, see runOnValueEnd), an object or an
// array, which is read to its closing bracket or to the end of s as in
// sensitiveValueEnd, a quote that is not closed, which is read to the end of s
// unless it does not open a value (see unclosedQuoteOpens), or a bare value.
// Without a delimiter after it, the word is read as a bare value (see
// textValueEnd). open and quoted are as for textValueEnd.
func oversizedValueEnd(s string, start, end, open int, quoted quoteKinds) int {
	closed := sensitiveValueEnd(s, start, start)
	if closed > end {
		return closed
	}
	if opened := openedValueEnd(s, closed, end, open, true, quoted); opened > end {
		return opened
	}
	v := s[start:end]
	at := end
	if !strings.HasSuffix(v, ":") && !strings.HasSuffix(v, "=") {
		at = skipDelimSpace(s, end)
		if at == len(s) || (s[at] != ':' && s[at] != '=') {
			return textValueEnd(s, start, end, open, quoted)
		}
		at++
	}
	at = skipDelimSpace(s, at)
	rest := s[at:]
	if rest != "" && (rest[0] == '{' || rest[0] == '[') {
		return sensitiveValueEnd(s, at, at)
	}
	if q := openingQuote(s, at); q != "" {
		if closesString(s, at, q, quoted) {
			return textValueEnd(s, start, end, open, quoted)
		}
		if qv := quotedValueRe.FindString(rest); qv != "" {
			if !quotedValueEnds(rest, len(qv), quoted) {
				return runOnValueEnd(s, at, at+len(qv), open, quoted)
			}
			if valueStartsAt(s, at, true, quoted) {
				return at + len(qv)
			}
			return textValueEnd(s, start, end, open, quoted)
		}
		if unclosedQuoteOpens(s, at, q, true, quoted) {
			return len(s)
		}
		return textValueEnd(s, start, end, open, quoted)
	}
	if n := strings.IndexAny(rest, regexpSpace+",;&"); n >= 0 {
		return at + n
	}
	return len(s)
}

// regexpSpace holds the bytes that \s matches in a regular expression such
// as textKeyValTokenAtRe.
const regexpSpace = "\t\n\f\r "

// isDelimSpace reports whether r is a Unicode White_Space or ASCII control
// character matched by delimSpaceClass.
func isDelimSpace(r rune) bool {
	return r <= ' ' || r == '\x7f' || unicode.IsSpace(r)
}

// skipDelimSpace returns the index of the first rune in s[i:] that is not a
// delimiter whitespace or control character, or len(s).
func skipDelimSpace(s string, i int) int {
	for i < len(s) {
		c := s[i]
		if c < utf8.RuneSelf {
			if c > ' ' && c != '\x7f' {
				break
			}
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		if !unicode.IsSpace(r) {
			break
		}
		i += size
	}
	return i
}

// skipHorizDelimSpace skips horizontal delimiter whitespace in s[i:], stopping
// before line and paragraph breaks (\n, \r, U+0085, \p{Zl}, \p{Zp}).
func skipHorizDelimSpace(s string, i int) int {
	for i < len(s) {
		c := s[i]
		if c < utf8.RuneSelf {
			if (c > ' ' && c != '\x7f') || c == '\n' || c == '\r' {
				break
			}
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		if !unicode.IsSpace(r) || r == '\u0085' || unicode.Is(unicode.Zl, r) || unicode.Is(unicode.Zp, r) {
			break
		}
		i += size
	}
	return i
}

// holdsValue reports whether v contains a rune other than delimiter whitespace
// and closingPunctuation.
func holdsValue(v string) bool {
	for _, r := range v {
		if !isDelimSpace(r) && !strings.ContainsRune(closingPunctuation, r) {
			return true
		}
	}
	return false
}

// followedByDelimiter reports whether a key-value delimiter, ":" or "=",
// follows s[:i], possibly after delimiter whitespace.
func followedByDelimiter(s string, i int) bool {
	at := skipDelimSpace(s, i)
	return at < len(s) && (s[at] == ':' || s[at] == '=')
}

// valueFollowsDelimiter reports whether a key-value delimiter follows s[:i],
// possibly after delimiter whitespace, and a value starts after the delimiter
// (see valueStartsAt).
func valueFollowsDelimiter(s string, i int, mayClose bool, quoted quoteKinds) bool {
	at := skipDelimSpace(s, i)
	return at < len(s) && (s[at] == ':' || s[at] == '=') && valueStartsAt(s, at+1, mayClose, quoted)
}

// valueStartsAt reports whether a value that can be redacted starts in s at
// i, possibly after delimiter whitespace. The value read is a quoted value
// (see quotedValueRegex) that ends there (see quotedValueEnds), an unclosed
// quote that opens a value (see unclosedQuoteOpens), or else the first run of
// bytes, up to the next whitespace, ",", ";" or "&". It must hold a character
// other than delimiter whitespace and closingPunctuation, unlike {Password:abc-token: },
// {Password:abc-token: "} or {Password:abc-token: ""}, and must not close an
// enclosing string when mayClose is set.
func valueStartsAt(s string, i int, mayClose bool, quoted quoteKinds) bool {
	at := skipDelimSpace(s, i)
	if quote := openingQuote(s, at); quote != "" {
		if mayClose && closesString(s, at, quote, quoted) {
			return false
		}
		q := quotedValueRe.FindString(s[at:])
		if q != "" {
			if quotedValueEnds(s[at:], len(q), quoted) {
				inner, _ := unquoteValue(q)
				return holdsValue(inner)
			}
			return textBetweenStrings(s, at, at+len(q), quoted) == ""
		}
		return unclosedQuoteOpens(s, at, quote, mayClose, quoted)
	}
	rest := s[at:]
	if n := strings.IndexAny(rest, regexpSpace+",;&"); n >= 0 {
		rest = rest[:n]
	}
	return holdsValue(rest)
}

// quotedValueEnds reports whether the quoted value s[:n] at the start of s
// ends there: s ends after it, or its closing quote does not pair with a
// later quote (see pairsForward) and is followed either by a genuine value
// boundary (see followsValueBoundary) or by a quote, of any kind, that closes
// an enclosing string in quoted (see endsString) or starts a quoted key, as
// the "c" of the JSON text "a":"b""c":"d" does. Otherwise its quotes close one
// string and open the next rather than enclose a value, as do the quotes
// around " for user " in the text "password=abc-token: " for user "bob"", or
// the value runs into a joined string or punctuation-led continuation, as in
// 'my'"'"'tail' or "prefix"+tail.
func quotedValueEnds(s string, n int, quoted quoteKinds) bool {
	after := s[n:]
	if after == "" {
		return true
	}
	if pairsForward(s, n) {
		return false
	}
	quote := openingQuote(after, 0)
	if quote == "" {
		if _, valQuote := unquoteValue(s[:n]); len(valQuote) == 2 && len(after) >= 2 && after[0] == '\\' && strings.IndexByte("nrt", after[1]) >= 0 {
			return true
		}
		return followsValueBoundary(after, quoted)
	}
	return endsString(after, 0, quote, quoted) || startsQuotedKey(after)
}

// followsValueBoundary reports whether after, the non-empty text immediately
// following a closing quote when openingQuote(after, 0) == "", begins at a
// delimiter that terminates the quoted value: delimiter whitespace, a pair or
// clause separator (",", ";" or "&"), or closing brackets (")", "]" or "}")
// and at most one sentence/clause punctuation mark (".", "!", "?" or ":")
// followed by end-of-text, delimiter whitespace, ",", ";", "&" or an outer
// closing quote in quoted. Punctuation continuations such as "+secret",
// "-secret", "/secret", ".secret", "@secret", "%secret", "!secret", "=secret",
// ":secret" or "\secret" return false so runOnValueEnd consumes them.
func followsValueBoundary(after string, quoted quoteKinds) bool {
	if skipDelimSpace(after, 0) > 0 || after[0] == ',' || after[0] == ';' || after[0] == '&' {
		return true
	}
	i := 0
	for i < len(after) && (after[i] == ')' || after[i] == ']' || after[i] == '}') {
		i++
	}
	if i < len(after) && (after[i] == '.' || after[i] == '!' || after[i] == '?' || after[i] == ':') {
		i++
	}
	if i == 0 {
		return false
	}
	if i == len(after) || skipDelimSpace(after, i) > i || after[i] == ',' || after[i] == ';' || after[i] == '&' {
		return true
	}
	q := openingQuote(after, i)
	return q != "" && endsString(after, i, q, quoted)
}

// pairsForward reports whether the closing quote of the quoted value s[:n]
// at the start of s more likely pairs with a later quote than closes the
// value: it opens the value of a later key (see opensLaterValue), or s goes
// on with a letter, a digit, "_" or a redaction placeholder, which a value
// does not run into. Then the text between the quotes is not a value, as
// that of " for user " in got "token": " for user "bob" is not, unless it is
// not prose either (see readsAsProse): a credential that is run into
// something, as in Bearer "abc"def, or that ends in a sensitive key name and a
// delimiter, as in Bearer "abc_TOKEN=" expired, is still redacted, together
// with what it runs into (see runOnValueEnd). The callers that leave a quoted
// value in place when its closing quote pairs forward check both.
func pairsForward(s string, n int) bool {
	after := s[n:]
	if after == "" {
		return false
	}
	inner, _ := unquoteValue(s[:n])
	if opensLaterValue(inner, after) {
		return true
	}
	r, _ := utf8.DecodeRuneInString(after)
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) || strings.HasPrefix(after, redactedValue)
}

// readsAsProse reports whether inner, the text between the quotes of a quoted
// value, reads as the words between two strings rather than as a credential:
// it starts with delimiter whitespace and holds more than one word, as the
// text of " in config, api_key=" or " for user " does. A credential is one
// word, which a quote or whitespace may pad but the words of a sentence do
// not.
func readsAsProse(inner string) bool {
	words := strings.TrimLeftFunc(inner, isDelimSpace)
	if len(words) == len(inner) {
		return false
	}
	return strings.IndexFunc(strings.TrimRightFunc(words, isDelimSpace), isDelimSpace) >= 0
}

// textBetweenStrings returns the quote of the closed quoted value s[start:end]
// if it is the text between two strings rather than a credential, and "" if
// it is not, so that a caller leaves it in place and searches the text after
// its opening quote again, as for the text of " in config, api_key=" or of
// " for user " in got "token": " for user "bob". The text reads as prose (see
// readsAsProse), and its closing quote pairs with a later quote (see
// pairsForward) that opens a second string with something in it. The second
// string starts at the closing quote or, if the value of a key starts with a
// quote of its own, as in api_key=""x"", right after it. The text ends where
// a second string may begin: in delimiter whitespace, as " for user " does, or
// in a key name and its delimiter that whitespace precedes, as
// " in config, api_key=" does (see opensLaterPairAfterSpace). Where no closed
// second string follows the closing quote, as in " R8 R9"suffix, " R8 R9"" or
// " R8 R9_TOKEN=" expired, the text is a credential that the closing quote
// runs into (see runOnValueEnd).
func textBetweenStrings(s string, start, end int, quoted quoteKinds) string {
	inner, quote := unquoteValue(s[start:end])
	if quote == "" || !readsAsProse(inner) || !pairsForward(s[start:], end-start) {
		return ""
	}
	last, _ := utf8.DecodeLastRuneInString(inner)
	laterPair := !isDelimSpace(last) && opensLaterPairAfterSpace(inner, s[end:])
	if !isDelimSpace(last) && !laterPair {
		return ""
	}
	secStart := end - len(quote)
	secQuoted := quoted
	second := quotedValueRe.FindString(s[secStart:])
	if len(second) <= 2*len(quote) && laterPair {
		secStart = end
		secQuoted = quoted | quoteKind(quote)
		second = quotedValueRe.FindString(s[secStart:])
	}
	if len(second) <= 2*len(quote) {
		return ""
	}
	if !quotedValueEnds(s[secStart:], len(second), secQuoted) {
		if laterPair {
			if bStart := secStart + len(quote); bStart < len(s) && (s[bStart] == '{' || s[bStart] == '[') {
				if bEnd, closed := bracketValueEnd(s, bStart); closed &&
					strings.HasPrefix(s[bEnd:], quote) &&
					quotedValueEnds(s[secStart:], bEnd+len(quote)-secStart, secQuoted) {
					return quote
				}
			}
		}
		secInner, _ := unquoteValue(second)
		if !readsAsProse(secInner) || !opensLaterPairAfterSpace(secInner, s[secStart+len(second):]) {
			return ""
		}
	}
	return quote
}

// opensLaterValue reports whether the closing quote of a quoted value, whose
// text between the quotes is inner and which after follows, opens the value
// of a later key instead: inner ends in a sensitive key and its delimiter
// (see endsInSensitiveKeyDelimiter), and a value starts after the quote (see
// valueFollows). In got "token": " in config, api_key="   secret", the quotes
// around " in config, api_key=" enclose no value; the second opens the value
// of api_key, whatever it starts with.
func opensLaterValue(inner, after string) bool {
	return valueFollows(after) && endsInSensitiveKeyDelimiter(inner)
}

// valueFollows reports whether a value starts in after, possibly after
// delimiter whitespace: a rune other than delimiter whitespace and
// closingPunctuation comes before the next whitespace, ",", ";" or "&", and
// after does not go on with a ",", ";", "&" or closing bracket first.
func valueFollows(after string) bool {
	at := skipDelimSpace(after, 0)
	if at == len(after) || strings.IndexByte(",;&)]}", after[at]) >= 0 {
		return false
	}
	for _, r := range after[at:] {
		if strings.ContainsRune(regexpSpace+",;&", r) {
			return false
		}
		if !isDelimSpace(r) && !strings.ContainsRune(closingPunctuation, r) {
			return true
		}
	}
	return false
}

// endsInSensitiveKeyDelimiter reports whether v, the text between the quotes
// of a quoted value, ends in the name of a sensitive key (see textKeyAllowed)
// and a key-value delimiter, as the text of " in config, api_key=" does (see
// endsInKeyDelimiter).
func endsInSensitiveKeyDelimiter(v string) bool {
	return endsInKeyDelimiter(v, textKeyAllowed)
}

// endsInKeyDelimiter reports whether v, the text between the quotes of a
// quoted value, ends in a key name that allowed accepts, read as the run of
// name bytes (see isCamelTokenByte) before the delimiter, and a key-value
// delimiter, possibly with a quote, which may be escaped, between them and
// whitespace around the delimiter. A name longer than maxNestedKeyLen is not
// read.
func endsInKeyDelimiter(v string, allowed func(name string) bool) bool {
	return trailingKeyStart(v, allowed) >= 0
}

// trailingKeyStart returns where the key name that v ends in starts, as
// endsInKeyDelimiter reads it, or -1 if v does not end in a key name that
// allowed accepts and its delimiter.
func trailingKeyStart(v string, allowed func(name string) bool) int {
	v = strings.TrimRightFunc(v, isDelimSpace)
	if v == "" || (v[len(v)-1] != '=' && v[len(v)-1] != ':') {
		return -1
	}
	v = strings.TrimRightFunc(v[:len(v)-1], isDelimSpace)
	if strings.HasSuffix(v, `"`) || strings.HasSuffix(v, `'`) {
		v = strings.TrimSuffix(v[:len(v)-1], `\`)
	}
	i := len(v)
	for i > 0 && isCamelTokenByte(v[i-1]) {
		if len(v)-i == maxNestedKeyLen {
			return -1
		}
		i--
	}
	if i < len(v) && allowed(v[i:]) {
		return i
	}
	return -1
}

// isIdentifierKey reports whether name, a non-empty run of the bytes of a
// key name (see isCamelTokenByte), may name a key, sensitive or not: it
// starts with an ASCII letter or "_", as an identifier does.
func isIdentifierKey(name string) bool {
	c := name[0]
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || c == '_'
}

// opensLaterPair reports whether the closing quote of a quoted value, whose
// text between the quotes is inner and which after follows, opens the value
// of a later key, whatever its name: inner ends in a key name and its
// delimiter (see endsInKeyDelimiter and isIdentifierKey), and a value starts
// after the quote (see valueFollows). In token=abc:'s label=' x', the
// closing quote of 's label=' opens the value of label.
func opensLaterPair(inner, after string) bool {
	return valueFollows(after) && endsInKeyDelimiter(inner, isIdentifierKey)
}

// opensLaterPairAfterSpace reports whether the closing quote of a quoted
// value opens the value of a later key (see opensLaterPair) whose name starts
// a word of inner after prose words, as the api_key of " in config, api_key="
// does, rather than ends a credential word or compound secret token, as the
// TOKEN of " R8_TOKEN=" or " R8_1 R8_2_TOKEN=" does.
func opensLaterPairAfterSpace(inner, after string) bool {
	if !valueFollows(after) {
		return false
	}
	i := trailingKeyStart(inner, isIdentifierKey)
	if i < 0 {
		return false
	}
	if i > 0 {
		last, _ := utf8.DecodeLastRuneInString(inner[:i])
		if !isDelimSpace(last) || !hasLowerProseWord(inner[:i]) {
			return false
		}
	}
	n := 0
	for i+n < len(inner) && isCamelTokenByte(inner[i+n]) {
		n++
	}
	if n == 0 {
		return false
	}
	key := inner[i : i+n]
	kind, _ := nestedKeyName(key, 0, false)
	if kind == nestedKey {
		return true
	}
	if kind == valueEndingInKeyName {
		return false
	}
	return !strings.ContainsAny(key, "_-.") && strings.IndexFunc(key, unicode.IsDigit) < 0
}

// hasLowerProseWord reports whether s contains a whitespace-delimited token of
// lowercase ASCII letters (ignoring surrounding punctuation), as prose phrases
// such as " in config, " or " for user " do, unlike uppercase/digit/underscore
// credential words such as " R8_SECRET_1 ".
func hasLowerProseWord(s string) bool {
	for i := 0; i < len(s); {
		i = skipDelimSpace(s, i)
		if i >= len(s) {
			break
		}
		j := i
		for j < len(s) {
			r, size := utf8.DecodeRuneInString(s[j:])
			if isDelimSpace(r) {
				break
			}
			j += size
		}
		tok := strings.TrimLeft(s[i:j], "([")
		tok = strings.TrimRight(tok, ",;:.!?)]}")
		if tok != "" {
			allLower := true
			for k := 0; k < len(tok); k++ {
				if tok[k] < 'a' || tok[k] > 'z' {
					allLower = false
					break
				}
			}
			if allLower {
				return true
			}
		}
		i = j
	}
	return false
}

// opensLaterConstruct reports whether the closing quote of a nested quoted
// candidate at s[at:at+qLen] (with quote length quoteLen and unquoted body
// inner) actually opens a later key-value pair, a Bearer credential or a
// quoted key rather than closing a value opened at s[at].
func opensLaterConstruct(s string, at, qLen, quoteLen int, inner string) bool {
	after := s[at+qLen:]
	if opensLaterPair(inner, after) || opensLaterBearer(inner, after) {
		return true
	}
	closing := at + qLen - quoteLen
	last, _ := utf8.DecodeLastRuneInString(inner)
	return (inner == "" || isDelimSpace(last) || strings.ContainsRune(",;([", last)) && startsQuotedKeyWithValue(s[closing:])
}

// opensLaterBearer reports whether inner ends with the word "Bearer" followed
// by delimiter whitespace and a value starts in after, as in
// token=abc:'s note Bearer ' secret'.
func opensLaterBearer(inner, after string) bool {
	if !valueFollows(after) {
		return false
	}
	trimmed := strings.TrimRightFunc(inner, isDelimSpace)
	if len(trimmed) == len(inner) || len(trimmed) < len("bearer") {
		return false
	}
	bStart := len(trimmed) - len("bearer")
	if !strings.EqualFold(trimmed[bStart:], "bearer") {
		return false
	}
	if bStart > 0 {
		r, _ := utf8.DecodeLastRuneInString(trimmed[:bStart])
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			return false
		}
	}
	return true
}

// startsQuotedKeyWithValue reports whether s starts with a quoted identifier
// key followed by a ":" or "=" delimiter and a value, as in 'password': ' x'.
func startsQuotedKeyWithValue(s string) bool {
	q := quotedValueRe.FindString(s)
	if q == "" {
		return false
	}
	key, _ := unquoteValue(q)
	key = strings.TrimSpace(key)
	if key == "" || len(key) > maxNestedKeyLen || !isIdentifierKey(key) {
		return false
	}
	for i := 0; i < len(key); i++ {
		if !isCamelTokenByte(key[i]) && key[i] != ':' {
			return false
		}
	}
	delim := skipDelimSpace(s, len(q))
	if delim >= len(s) || (s[delim] != ':' && s[delim] != '=') || (s[delim] == '=' && delim+1 < len(s) && s[delim+1] == '=') {
		return false
	}
	return valueStartsAt(s, delim+1, false, 0)
}

// isProseContractions reports whether the apostrophe at s[at] immediately
// after ":" at s[colon] (ending the bare token s[:end]) and the apostrophe at
// s[closing] are independent prose contractions separated by an outer ";"
// clause boundary, as in token=abc:'s scope missing; the user's request,
// rather than a single-quoted or joined credential such as
// export DB_PASSWORD=a:'s correct horse battery's.
func isProseContractions(s string, colon, at, end, closing int) bool {
	n1 := contractionLen(s[at+1:])
	n2 := contractionLen(s[closing+1:])
	if s[colon] != ':' || at != colon+1 || n1 == 0 || at+1+n1 != end || !isMidWordApostrophe(s, closing) || n2 == 0 {
		return false
	}
	afterClosing := closing + 1 + n2
	if afterClosing < len(s) {
		r, _ := utf8.DecodeRuneInString(s[afterClosing:])
		if !isDelimSpace(r) && !strings.ContainsRune(",;:.!?)]}", r) {
			return false
		}
	}
	between := s[end:closing]
	semi := strings.IndexByte(between, ';')
	if semi < 0 || semi+1 >= len(between) || skipHorizDelimSpace(between, semi+1) == semi+1 {
		return false
	}
	if strings.ContainsAny(between, "\r\n=:,'\"\\") {
		return false
	}
	for _, field := range strings.FieldsFunc(strings.ReplaceAll(between, ";", " "), isDelimSpace) {
		for k := 0; k < len(field); k++ {
			if field[k] < 'a' || field[k] > 'z' {
				return false
			}
		}
	}
	return strings.TrimSpace(between[semi+1:]) != ""
}

// proseFollowsContraction reports whether after, the text immediately
// following an unclosed ':'<contraction>, ends the value at an outer delimiter
// or line break, or leads through lowercase prose words on the same line to an
// outer ";" clause boundary or an independent credential or key-value
// construct (Bearer token, quoted key or key=value pair), as in
// token=abc:'s note Bearer [REDACTED]. Bare words without such a boundary, as
// in password=a:'s correct horse battery, stay inside the unclosed credential
// span.
func proseFollowsContraction(after string) bool {
	if after == "" || strings.IndexByte("\r\n;,&", after[0]) >= 0 {
		return true
	}
	at := skipHorizDelimSpace(after, 0)
	if at == 0 {
		return false
	}
	line := after[at:]
	if nl := strings.IndexAny(line, "\r\n"); nl >= 0 {
		line = line[:nl]
	}
	line = strings.TrimRightFunc(line, isDelimSpace)
	if line == "" {
		return true
	}
	for pos := 0; pos < len(line); {
		rest := line[pos:]
		if startsLaterConstructAt(rest) {
			return true
		}
		end := pos
		for end < len(line) {
			r, size := utf8.DecodeRuneInString(line[end:])
			if isDelimSpace(r) {
				break
			}
			end += size
		}
		tok := line[pos:end]
		if word, hasSemi := strings.CutSuffix(tok, ";"); hasSemi {
			return isLowerASCIIWord(word) && isLowerProseClause(line[end:])
		}
		if !isLowerASCIIWord(tok) {
			return false
		}
		pos = skipHorizDelimSpace(line, end)
	}
	return false
}

// startsLaterConstructAt reports whether s begins with a Bearer credential, a
// quoted key-value pair, an unspaced key-value pair or a standalone sensitive
// key-value pair.
func startsLaterConstructAt(s string) bool {
	if len(s) > len("bearer") && strings.EqualFold(s[:len("bearer")], "bearer") {
		if sp := skipHorizDelimSpace(s, len("bearer")); sp > len("bearer") && valueStartsAt(s, sp, false, 0) {
			tail := s[sp:]
			if len(tail) < 2 || !strings.EqualFold(tail[:2], "of") || (len(tail) > 2 && skipDelimSpace(tail, 2) == 2) {
				return true
			}
		}
	}
	if startsQuotedKeyWithValue(s) || startsPlainPair(s, 0) {
		return true
	}
	kind, _ := afterClosedQuotePair(s, 0, 0)
	return kind == nestedKey
}

func isLowerASCIIWord(w string) bool {
	if w == "" {
		return false
	}
	for i := 0; i < len(w); i++ {
		if w[i] < 'a' || w[i] > 'z' {
			return false
		}
	}
	return true
}

func isLowerProseClause(s string) bool {
	at := skipHorizDelimSpace(s, 0)
	if at == 0 || at >= len(s) {
		return false
	}
	clause := strings.TrimRight(s[at:], ".!?")
	fields := strings.FieldsFunc(clause, isDelimSpace)
	if len(fields) == 0 {
		return false
	}
	for _, f := range fields {
		if !isLowerASCIIWord(f) {
			return false
		}
	}
	return true
}

// startsQuotedKey reports whether s starts with a quoted key: a quoted value
// (see quotedValueRegex) followed by a key-value delimiter, possibly after
// delimiter whitespace.
func startsQuotedKey(s string) bool {
	q := quotedValueRe.FindString(s)
	return q != "" && followedByDelimiter(s, len(q))
}

// ownsQuotedValue reports whether the quoted value s[start:end], whose
// opening quote q may close a string open before it (see closesString),
// opens a value of its own instead: a quote of its kind closes it, it holds a
// value (see holdsValue), it ends s or is followed by delimiter whitespace,
// possibly after closing brackets and punctuation, and its closing quote does
// not open the value of a later key (see opensLaterValue). In
// Bearer " in config, api_key="   secret", the quote after api_key= opens the
// value "   secret", though the quote after Bearer leaves a string open
// before it.
func ownsQuotedValue(s string, start, end int, q string) bool {
	inner, quote := unquoteValue(s[start:end])
	if quote != q || !holdsValue(inner) {
		return false
	}
	at := end
	for at < len(s) && strings.IndexByte(".,;!?)]}", s[at]) >= 0 {
		at++
	}
	if at < len(s) && skipDelimSpace(s, at) == at {
		return false
	}
	return !opensLaterValue(inner, s[end:])
}

// forwardValueEnd returns the end of the value that the quoted value
// s[start:end] after a key starts when its closing quote opens the value of a
// later key (see opensLaterValue): the value runs on through each quoted value
// whose closing quote opens the value of a later key, or across a quoted,
// bracketed or bare value separated by whitespace, to the end of the last, or
// to the end of s if a quote that opens a value is not closed, so that the
// pairs it holds are redacted with it without pairing a closing quote across
// whitespace with a following quoted value's opening quote.
func forwardValueEnd(s string, start, end, open int, quoted quoteKinds) int {
	for {
		inner, quote := unquoteValue(s[start:end])
		if quote == "" || !opensLaterValue(inner, s[end:]) {
			if quote != "" && !quotedValueEnds(s[start:], end-start, quoted) {
				return runOnValueEnd(s, start, end, open, quoted)
			}
			return end
		}
		after := skipDelimSpace(s, end)
		if after > end || openingQuote(s, after) != "" {
			if q2 := openingQuote(s, after); q2 != "" {
				m := quotedValueRe.FindString(s[after:])
				if m == "" {
					if unclosedQuoteOpens(s, after, q2, true, quoted) {
						return len(s)
					}
					return end
				}
				start, end = after, after+len(m)
				continue
			}
			if after < len(s) && (s[after] == '{' || s[after] == '[') {
				return sensitiveValueEnd(s, after, after)
			}
			if readsAsProse(inner) && opensLaterPairAfterSpace(inner, s[end:]) {
				qOpen := end - len(quote)
				if m := quotedValueRe.FindString(s[qOpen:]); m != "" && quotedValueEnds(s[qOpen:], len(m), quoted) {
					return qOpen + len(m)
				}
			}
			bareEnd := after
			for bareEnd < len(s) && strings.IndexByte(regexpSpace+",;&", s[bareEnd]) < 0 {
				bareEnd++
			}
			if kind, pair := nestsSensitiveKey(s, after, bareEnd, startsLine(s, end, after), true, quoted); kind != notNestedKey {
				return nestedPairEnd(s, pair, bareEnd, open, quoted)
			}
			return textValueEnd(s, after, bareEnd, open, quoted)
		}
		qOpen := end - len(quote)
		m := quotedValueRe.FindString(s[qOpen:])
		if m == "" {
			return len(s)
		}
		start, end = qOpen, qOpen+len(m)
	}
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

// quoteKinds is a set of the quotes ", ', \" and \'.
type quoteKinds uint8

// quoteKind returns the set that holds the quote q, ", ', \" or \'.
func quoteKind(q string) quoteKinds {
	switch q {
	case `"`:
		return 1
	case `'`:
		return 2
	case `\"`:
		return 4
	case `\'`:
		return 8
	}
	return 0
}

// openQuotes scans a text from its start in a single O(1)-per-byte pass and
// tracks the active outer string delimiter (", ', \" or \') at its cursor,
// ignoring inner quotes of other kinds while a string is open, apostrophes
// between letters (see isMidWordApostrophe) other than the quote of a
// prefixed string such as b'...' (see opensPrefixedString) and, outside
// strings, apostrophes that end a word (see isTrailingApostrophe), and
// resetting line-local quote state at line breaks.
type openQuotes struct {
	pos    int // s[:pos] has been read or skipped
	active quoteKinds
	bs     int // consecutive backslashes immediately preceding s[pos]
}

// isMidWordApostrophe reports whether the single quote at s[pos] is an
// apostrophe between two letters, as in don't or l'été, rather than a quote.
func isMidWordApostrophe(s string, pos int) bool {
	before, _ := utf8.DecodeLastRuneInString(s[:pos])
	after, _ := utf8.DecodeRuneInString(s[pos+1:])
	return unicode.IsLetter(before) && unicode.IsLetter(after)
}

// isTrailingApostrophe reports whether the single quote at s[pos] is an
// apostrophe that ends a word, as in the users' names: a letter precedes it,
// and it ends s or is followed by delimiter whitespace or by punctuation that
// may end a word. A quote that closes a string is such a quote too, so it is
// read as one only outside strings.
func isTrailingApostrophe(s string, pos int) bool {
	if before, _ := utf8.DecodeLastRuneInString(s[:pos]); !unicode.IsLetter(before) {
		return false
	}
	if pos+1 == len(s) || strings.IndexByte(",;.!?)]}", s[pos+1]) >= 0 {
		return true
	}
	after, _ := utf8.DecodeRuneInString(s[pos+1:])
	return isDelimSpace(after)
}

// isApostrophe reports whether the quote of kind at s[pos] is an apostrophe
// rather than a quote (see isMidWordApostrophe and isTrailingApostrophe),
// where active is the quote of the string open before it, or 0. A quote
// between letters that opens a string after a prefix (see
// opensPrefixedString), as in b'note=1' or curl -H'X-Custom: 1', is a quote.
func isApostrophe(s string, pos int, kind, active quoteKinds) bool {
	return kind == 2 && (isMidWordApostrophe(s, pos) && !opensPrefixedString(s, pos) ||
		active == 0 && (isTrailingApostrophe(s, pos) || isColonContractionApostrophe(s, pos)))
}

// isColonContractionApostrophe reports whether the single quote at s[pos],
// outside any open string, is a prose contraction or possessive apostrophe
// after a bare word and ":" (as in token=abc:'s note Bearer ' x'), consistent
// with openedValueEnd.
func isColonContractionApostrophe(s string, pos int) bool {
	if pos < 2 || s[pos-1] != ':' {
		return false
	}
	n := contractionLen(s[pos+1:])
	if n == 0 {
		return false
	}
	before, _ := utf8.DecodeLastRuneInString(s[:pos-1])
	if !unicode.IsLetter(before) && !unicode.IsDigit(before) {
		return false
	}
	cEnd := pos + 1 + n
	if cEnd < len(s) {
		r, _ := utf8.DecodeRuneInString(s[cEnd:])
		if !isDelimSpace(r) && !strings.ContainsRune(",;&", r) {
			return false
		}
	}
	q := quotedValueRe.FindString(s[pos:])
	if q == "" || strings.ContainsAny(q, "\r\n") {
		return proseFollowsContraction(s[cEnd:])
	}
	inner, _ := unquoteValue(q)
	if opensLaterConstruct(s, pos, len(q), 1, inner) {
		return true
	}
	return !quotedValueEnds(s[pos:], len(q), 0) && isProseContractions(s, pos-1, pos, cEnd, pos+len(q)-1)
}

// opensPrefixedString reports whether the single quote at s[pos], between two
// letters (see isMidWordApostrophe), opens a string after a prefix rather
// than being an apostrophe: the letters before it are a Python string prefix
// that starts a word (b, r, u, f, br, rb, fr or rf, in either case), as in
// b'...', r'...', f'...' or rb'...', or the letter of a short command-line
// option, a "-" and a letter at the start of s or after whitespace, as in
// curl -H'X-Custom: 1'. A quote before the ending of a contraction or a
// possessive, as in -I'm or the B's of plan B's, is an apostrophe all the
// same (see isContraction).
func opensPrefixedString(s string, pos int) bool {
	if isContraction(s[pos+1:]) {
		return false
	}
	isLetter := func(c byte) bool { return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' }
	w := pos
	for w > 0 && pos-w <= 2 && isLetter(s[w-1]) {
		w--
	}
	if w == pos || pos-w > 2 {
		return false
	}
	before, _ := utf8.DecodeLastRuneInString(s[:w])
	if w > 0 && (unicode.IsLetter(before) || unicode.IsDigit(before) || before == '_') {
		return false
	}
	switch strings.ToLower(s[w:pos]) {
	case "b", "r", "u", "f", "br", "rb", "fr", "rf":
		return true
	}
	if pos-w != 1 || before != '-' {
		return false
	}
	dash, _ := utf8.DecodeLastRuneInString(s[:w-1])
	return w == 1 || isDelimSpace(dash)
}

// contractionLen returns the byte length of the English contraction or
// possessive ending at the start of after (s, m, d, ll, re or ve, in either
// case) when followed by neither a letter, a digit, "_" nor "=", or 0.
func contractionLen(after string) int {
	for _, ending := range []string{"ll", "re", "ve", "s", "m", "d"} {
		if len(after) >= len(ending) && strings.EqualFold(after[:len(ending)], ending) {
			r, _ := utf8.DecodeRuneInString(after[len(ending):])
			if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' && r != '=' {
				return len(ending)
			}
		}
	}
	return 0
}

// isContraction reports whether after, the text after an apostrophe, starts
// with the ending of an English contraction or possessive, as the m of I'm
// and the s of B's do: s, m, d, ll, re or ve, in either case, followed by
// neither a letter, a digit, "_" nor "=", any of which would make it the
// start of a word or a key name instead, as in f'd={x}'.
func isContraction(after string) bool {
	return contractionLen(after) > 0
}

// before reads s up to i and returns the active quote kind open there, or 0.
func (q *openQuotes) before(s string, i int) quoteKinds {
	for ; q.pos < i; q.pos++ {
		c := s[q.pos]
		if c == '\n' || c == '\r' {
			q.active = 0
			q.bs = 0
			continue
		}
		if c == '\\' {
			q.bs++
			continue
		}
		bs := q.bs
		q.bs = 0
		if c != '"' && c != '\'' {
			continue
		}
		var kind quoteKinds
		switch {
		case bs%2 == 0:
			if c == '"' {
				kind = 1
			} else {
				kind = 2
			}
		case bs%4 == 1:
			if c == '"' {
				kind = 4
			} else {
				kind = 8
			}
		default: // bs%4 == 3: inner escaped quote inside %q-style string (\\\" or \\\')
			continue
		}
		if isApostrophe(s, q.pos, kind, q.active) {
			continue
		}
		switch q.active {
		case 0:
			q.active = kind
		case kind:
			q.active = 0
		}
	}
	return q.active
}

// skip moves past s[:i] for a redacted value: quotes inside a redacted value
// never open a new outer string, though a line break resets quote state and
// unescaped quotes of the already-open kind toggle it across s[q.pos:i]. A
// quoted value, JSON object or JSON array that starts the redacted value at
// s[q.pos] and closes inside it on its line, as the {"a":"don't"} of
// 'token={"a":"don't"}' does, is skipped whole: the quotes in it are its own
// and leave the outer string open.
func (q *openQuotes) skip(s string, i int) {
	if i <= q.pos {
		return
	}
	savedActive := q.active
	if savedActive == 0 {
		q.pos = i
		q.bs = 0
		return
	}
	if e := selfContainedValueEnd(s, q.pos, i); e > q.pos {
		q.pos, q.bs = e, 0
	}
	for ; q.pos < i; q.pos++ {
		c := s[q.pos]
		if c == '\n' || c == '\r' {
			q.active = 0
			q.pos = i
			q.bs = 0
			return
		}
		if c == '\\' {
			q.bs++
			continue
		}
		bs := q.bs
		q.bs = 0
		if c != '"' && c != '\'' {
			continue
		}
		var kind quoteKinds
		switch {
		case bs%2 == 0:
			if c == '"' {
				kind = 1
			} else {
				kind = 2
			}
		case bs%4 == 1:
			if c == '"' {
				kind = 4
			} else {
				kind = 8
			}
		default:
			continue
		}
		if isApostrophe(s, q.pos, kind, q.active) {
			continue
		}
		if kind == savedActive {
			if q.active == 0 {
				q.active = savedActive
			} else {
				q.active = 0
			}
		}
	}
	q.bs = 0
}

// unpaired moves past the opening quote at s[i:] of a quoted value whose text
// reads as prose and whose closing quote pairs with a later quote (see
// readsAsProse and pairsForward), and returns where to search on: at the
// quote, which is then read again, if it closes the string open before it, or
// else after it, without opening a string, as the quote that would close that
// string opens a later one.
func (q *openQuotes) unpaired(s string, i int, quote string) int {
	if q.active&quoteKind(quote) != 0 {
		q.skip(s, i)
		return i
	}
	q.pos, q.bs = max(q.pos, i+len(quote)), 0
	return i + len(quote)
}

// drop closes the string open before a value whose opening quote, which
// would close that string, opens the value instead (see ownsQuotedValue).
func (q *openQuotes) drop() {
	q.active = 0
}

// selfContainedValueEnd returns the end of the quoted value, JSON object or
// JSON array that starts at s[start] and closes inside s[:end] on the line
// that it starts on, or start if none does.
func selfContainedValueEnd(s string, start, end int) int {
	e := start
	if openingQuote(s, start) != "" {
		e = start + len(quotedValueRe.FindString(s[start:end]))
	} else if c := s[start]; c == '{' || c == '[' {
		if b, closed := bracketValueEnd(s[:end], start); closed {
			e = b
		}
	}
	if strings.ContainsAny(s[start:e], "\r\n") {
		return start
	}
	return e
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
	e, closed := bracketValueEnd(s, start)
	if !closed {
		return len(s)
	}
	return max(end, e)
}

// bracketValueEnd returns the end of the JSON object or array that opens at
// s[start], just past its closing bracket, and true, or len(s) and false if
// s ends before it closes. Brackets inside strings do not count, as for
// sensitiveValueEnd.
func bracketValueEnd(s string, start int) (int, bool) {
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
				return i + 1, true
			}
		}
	}
	return len(s), false
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

// trimTrailingClosingQuote strips a trailing quote (possibly followed by
// sentence or bracket punctuation) that closes an enclosing string in quoted
// (see endsString) at the end of v, where v ends at s[:end].
func trimTrailingClosingQuote(v, s string, end int, quoted quoteKinds) string {
	if quoted == 0 {
		return v
	}
	trimmed := strings.TrimRight(v, ".,;:!?)]}")
	qEnd := end - (len(v) - len(trimmed))
	for _, q := range []string{`\"`, `\'`, `"`, `'`} {
		if strings.HasSuffix(trimmed, q) && qEnd >= len(q) && endsString(s, qEnd-len(q), q, quoted) {
			return trimmed[:len(trimmed)-len(q)]
		}
	}
	return v
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
