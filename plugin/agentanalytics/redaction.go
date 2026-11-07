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
	"cmp"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/url"
	"reflect"
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
// "aws_session_token", "bearer", "code_verifier", "cookie", "credential",
// "credentials", "passphrase", "secret_access_key", "session_token",
// "set_cookie", "x_auth_token" and "x_goog_api_key". A value under
// "credential" or "credentials" in a map, in
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
	"cookie":                {},
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
	"set_cookie":            {},
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

// maxCompactSensitiveKeyLen is the length of the longest compact credential
// spelling in sensitiveKeyNames. Even if a separator splits every letter of a
// credential suffix (as in DB_P_A_S_S_W_O_R_D), at most this many words can
// belong to that suffix.
var maxCompactSensitiveKeyLen = func() int {
	most := 1
	for k := range sensitiveKeyNames {
		most = max(most, len(k))
	}
	return most
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
	uriTokenRe         = regexp.MustCompile(`[a-zA-Z][a-zA-Z0-9+.-]*://[^\s"'<>]+`)
	authHeaderRe       = regexp.MustCompile(`(?im)\b` + authHeaderNamePattern + `([^\r\n]*)`)
	authHeaderPrefixRe = regexp.MustCompile(`(?im)\b` + authHeaderNamePattern)
	// authHeaderAtRe and authHeaderPrefixAtRe match authHeaderRe and
	// authHeaderPrefixRe at the start of a text, without the word boundary,
	// which findAtKeys reads in the text around the match.
	authHeaderAtRe       = regexp.MustCompile(`(?i)^` + authHeaderNamePattern + `([^\r\n]*)`)
	authHeaderPrefixAtRe = regexp.MustCompile(`(?i)^` + authHeaderNamePattern)
	// bearerTokenRe matches a bearer token. A bearer token (RFC 6750) never
	// contains quotes, ")" or "]", so the bare value stops there and keeps
	// the surrounding JSON or prose intact.
	bearerTokenRe = regexp.MustCompile(`(?i)\b(bearer)` + unicodeSpaceRun + `("(?:[^"\\]|\\(?s:.))*"|'(?:[^'\\]|\\(?s:.))*'|'(?:"(?:[^"\\]|\\(?s:.))*"|[^` + unicodeSpaceClass + `,;)"'\\\]}][^` + unicodeSpaceClass + `,;)"'\]]*)|"(?:'(?:[^'\\]|\\(?s:.))*'|[^` + unicodeSpaceClass + `,;)"'\\\]}][^` + unicodeSpaceClass + `,;)"'\]]*)|\[[^\]\r\n` + unicodeSpaceClass + `]+\][^` + unicodeSpaceClass + `,;)"']*|[^` + unicodeSpaceClass + `,;)"'\]]+)`)
	// bearerKeywordRe finds the keyword of bearerTokenRe in any case.
	bearerKeywordRe = regexp.MustCompile(`(?i)bearer`)
	// bearerTokenAtRe matches bearerTokenRe at the start of a slice only.
	bearerTokenAtRe = regexp.MustCompile(`^(?:` + bearerTokenRe.String() + `)`)
	// basicTokenRe matches a candidate basic credential, which
	// redactBasicCredential redacts only if it is valid base64.
	basicTokenRe = regexp.MustCompile(`(?i)\b(basic)` + unicodeSpaceRun + `([A-Za-z0-9+/]+={0,2})`)
	// basicTokenAtRe matches basicTokenRe at the start of a slice only.
	basicTokenAtRe    = regexp.MustCompile(`^(?:` + basicTokenRe.String() + `)`)
	jsonKeyValTokenRe = regexp.MustCompile(`(?i)` + jsonKeyValTokenPattern())
	// jsonKeyValTokenAtRe matches jsonKeyValTokenRe at the start of a text.
	jsonKeyValTokenAtRe = regexp.MustCompile(`(?i)^` + jsonKeyValTokenPattern())
	// textValueAtRe matches delimiter whitespace and the value part of
	// textKeyValTokenAtRe at the start of a slice, for quoted keys whose name and
	// delimiter quotedKeyScanAt has already read.
	textValueAtRe = regexp.MustCompile(`^[` + delimSpaceClass + `]*(` + sensitiveValueRegex(`\s,;&`) + `)`)
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
	textKeyNameRe = regexp.MustCompile(`(?i)(?:\b|_)` + textKeyNamePattern())
	// textKeyNameAtBoundaryRe and textKeyNameAtUnderscoreRe match
	// textKeyNameRe at the start of a text, after the word boundary that
	// findTextKeyName reads in the text around the match, or from the "_"
	// that the match opens with.
	textKeyNameAtBoundaryRe   = regexp.MustCompile(`(?i)^` + textKeyNamePattern())
	textKeyNameAtUnderscoreRe = regexp.MustCompile(`(?i)^_` + textKeyNamePattern())
	// textKeyNameNoOAuthAtBoundaryRe and textKeyNameNoOAuthAtUnderscoreRe are
	// those two without the dotted OAuth names, for the starts at which
	// findTextKeyName knows that none matches, and oauthTailAtRe matches
	// what has to follow the run of words that such a name starts with: its
	// last word, the quote that may follow it and the delimiter.
	textKeyNameNoOAuthAtBoundaryRe   = regexp.MustCompile(`(?i)^` + textKeyNamePatternOf(bareSensitiveKeyRegex()))
	textKeyNameNoOAuthAtUnderscoreRe = regexp.MustCompile(`(?i)^_` + textKeyNamePatternOf(bareSensitiveKeyRegex()))
	oauthTailAtRe                    = regexp.MustCompile(`(?i)^[.]code((?:\\?["'])?)[` + delimSpaceClass + `]*([=:])`)
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
	nestedKeyNameRe = regexp.MustCompile(`(?i)^(?:\\?["'])?(?:` + regexp.QuoteMeta(session.KeyPrefixUser) + `|` + regexp.QuoteMeta(session.KeyPrefixApp) + `)?([_-]*(?:` + fixedSensitiveKeyRegex() + `))$`)
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
// (api_key, api-key, api.key, apiKey). It also strips backslash-escaped line
// breaks (shell line splices), so that a key name split across a line, as
// DB_PASS\<LF>WORD is, folds to the same text as DB_PASSWORD. It also replaces
// the two non-ASCII letters that (?i) matches against ASCII ones, U+017F (long
// s) and U+212A (Kelvin sign), with "s" and "k".
func foldPatternText(s string) string {
	b := make([]byte, 0, len(s))
	for i := 0; i < len(s); {
		c := s[i]
		if c < utf8.RuneSelf {
			// Skip backslash-escaped line breaks (shell line splices), so
			// that a key name split across a line, as DB_PASS\<LF>WORD is,
			// folds to the same text as DB_PASSWORD.
			if c == '\\' && i+1 < len(s) && (s[i+1] == '\n' || s[i+1] == '\r') {
				if s[i+1] == '\r' && i+2 < len(s) && s[i+2] == '\n' {
					i += 3
				} else {
					i += 2
				}
				continue
			}
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

// hasQuotedSensitiveKey reports whether s holds a quoted sensitive key
// followed by ":" or "=", including padded compound and camelCase keys.
func hasQuotedSensitiveKey(s string) bool {
	if !mayMatchTextPatterns(s) {
		return false
	}
	if quotedSensitiveKeyRe.MatchString(s) {
		return true
	}
	for i := 0; i < len(s); {
		quote := openingQuote(s, i)
		if quote == "" || (quote == "'" && isApostrophe(s, i, 2, 0)) {
			i++
			continue
		}
		_, closeQ, nextPos := scanQuotedKeyToken(s, i, quote)
		if closeQ >= 0 && (quotedKeyAllowed(s[i+len(quote):closeQ]) || isSensitiveKey(s[i+len(quote):closeQ])) && followedByDelimiter(s, closeQ+len(quote)) {
			return true
		}
		i = max(nextPos, i+len(quote))
	}
	return false
}

// authHeaderNamePattern is the pattern of the name of an Authorization,
// API-key, Cookie or Set-Cookie header, which is submatch 1, and of the colon
// after it.
const authHeaderNamePattern = `((?:proxy[-_])?authorization|(?:x[-_](?:goog[-_])?)?api[-_]key(?:[-_](?:secret|string))?|(?:set[-_])?cookie)[ \t]*:[ \t]*`

// jsonKeyValTokenPattern returns the pattern of jsonKeyValTokenRe: a quoted
// sensitive key, which may be padded with spaces inside its quotes, and the
// value after its colon.
func jsonKeyValTokenPattern() string {
	return `"([\s\p{Z}]*(?:` + sensitiveKeyRegex() + `)[\s\p{Z}]*)"\s*:\s*(` + sensitiveValueRegex(`\s,}`) + `)`
}

// textKeyNamePattern returns the pattern of textKeyNameRe after the word
// boundary or the "_" that a match opens with: a credential name, an
// optional quote, which may be escaped, and a delimiter.
func textKeyNamePattern() string {
	return textKeyNamePatternOf(bareTextKeyNameRegex())
}

// textKeyNamePatternOf returns textKeyNamePattern for the names of bare in
// place of those of bareTextKeyNameRegex.
func textKeyNamePatternOf(bare string) string {
	return `(` + bare + `)((?:\\?["'])?)[` + delimSpaceClass + `]*([=:])`
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
// A backslash escapes any byte, a line break too, so a closed part that
// holds a backslash-newline, as 'x \<LF>y' does, is matched whole, as one
// with a bare line break is.
func quotedValueRegex() string {
	return `"(?:[^"\\]|\\(?s:.))*"` +
		`|'(?:[^'\\]|\\(?s:.))*'` +
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
// keys read inside quotes, where a "temp:" key name may contain "=", ":", ","
// and braces, as the quotes around it tell where it ends, and may follow a
// "user:" or "app:" state prefix and whitespace, as isSensitiveKey reads it,
// as in "user:temp:a=b" or "app: temp:x".
func sensitiveKeyRegex() string {
	return `(?:(?:` + regexp.QuoteMeta(session.KeyPrefixUser) + `|` + regexp.QuoteMeta(session.KeyPrefixApp) + `)[\s\p{Z}]*)?` +
		`temp:[^"'\\\s\r\n]+|[_.-]*(?:` + fixedSensitiveKeyRegex() + `)`
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
// first "=", ":", "," or unmatched "}", so neither the assignment delimiter,
// an "=" or ":" inside the value, nor the closing brace of a template
// placeholder such as {temp:x}: hello can be read as part of the key, while
// "{" and balanced "{...}" segments inside the key (such as temp:a{b=... or
// temp:cache/{user}=...) are accepted.
func bareSensitiveKeyRegex() string {
	return `temp:(?:[^=:"'\s\r\n,}]|\{[^=:"'\s\r\n,{}]*\})+|` + fixedSensitiveKeyRegex()
}

// keyTrie factors shared ASCII byte prefixes across compact credential
// spellings in fixedSensitiveKeyRegex so Go's RE2 engine steps each shared
// prefix once.
type keyTrie struct {
	end      bool
	children [128]*keyTrie
}

func (n *keyTrie) tailRegex() string {
	var branches []string
	for i, ch := range n.children {
		if ch != nil {
			branches = append(branches, regexp.QuoteMeta(string(byte(i)))+ch.tailRegex())
		}
	}
	if len(branches) == 0 {
		return ""
	}
	var cont string
	if len(branches) == 1 {
		cont = `[_.-]*` + branches[0]
	} else {
		cont = `[_.-]*(?:` + strings.Join(branches, "|") + `)`
	}
	if n.end {
		// "?" is greedy in RE2, so longer suffixes (such as "credentials",
		// "signature" and "api_key_secret") match ahead of shorter prefixes.
		return `(?:` + cont + `)?`
	}
	return cont
}

// fixedSensitiveKeyRegex returns a prefix-factored expression of the compact
// credential spellings in sensitiveKeyNames with optional "_", "-" and "."
// separator runs allowed between any two letters and at the end, so that
// api_key, api-key, api.key, api__key, api..key, api_key_, api.k_ey, pass_word,
// apiKey and APIKey all match in text discovery just as removeKeySeparators and
// splitKeyWords classify them in maps and structs.
func fixedSensitiveKeyRegex() string {
	root := &keyTrie{}
	for k := range sensitiveKeyNames {
		cur := root
		for i := 0; i < len(k); i++ {
			b := k[i]
			if b >= 128 {
				continue
			}
			if cur.children[b] == nil {
				cur.children[b] = &keyTrie{}
			}
			cur = cur.children[b]
		}
		cur.end = true
	}
	var top []string
	for i, ch := range root.children {
		if ch != nil {
			top = append(top, regexp.QuoteMeta(string(byte(i)))+ch.tailRegex())
		}
	}
	return `(?:` + strings.Join(top, "|") + `)[_.-]*`
}

// normalizeKey returns k in lower case, without the whitespace around it,
// with "_" for "-", and without the shell line splices that break it, as the
// one of DB_PASS\<LF>WORD does, which a shell removes (see
// stripShellLineSplices).
func normalizeKey(k string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimFunc(stripShellLineSplices(k), isDelimSpace)), "-", "_")
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
// isCompoundSensitiveKey). A key is read without the shell line splices that
// break it (see stripShellLineSplices), as a shell joins DB_PASS\<LF>WORD
// into DB_PASSWORD.
func isSensitiveKey(k string) bool {
	name, exact := exactSensitiveKey(stripShellLineSplices(k))
	return exact || isCompoundSensitiveKey(name)
}

// exactSensitiveKey reports whether k is sensitive by its whole name, as
// isSensitiveKey compares it without the compound names: a session "temp:"
// key, or a credential name, ignoring case, the "_", "-" and "." separators
// and a "user:" or "app:" state prefix. A "temp:" key is one after a state
// prefix and the whitespace after it too, as in user:temp:scratch or
// " app: temp:x ", and whatever its name holds, including none, as ADK reads
// any key that starts with "temp:" as one. It also returns k without the
// whitespace around it and without that prefix.
func exactSensitiveKey(k string) (name string, ok bool) {
	orig := strings.TrimFunc(k, isDelimSpace)
	n := strings.ToLower(orig)
	if strings.HasPrefix(n, session.KeyPrefixTemp) {
		return orig, true
	}
	if rest, found := strings.CutPrefix(n, session.KeyPrefixUser); found {
		n, orig = rest, orig[len(session.KeyPrefixUser):]
	} else if rest, found := strings.CutPrefix(n, session.KeyPrefixApp); found {
		n, orig = rest, orig[len(session.KeyPrefixApp):]
	}
	n, orig = strings.TrimFunc(n, isDelimSpace), strings.TrimFunc(orig, isDelimSpace)
	if strings.HasPrefix(n, session.KeyPrefixTemp) {
		return orig, true
	}
	_, ok = sensitiveKeyNames[removeKeySeparators(n)]
	return orig, ok
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
// returns them, name a pagination cursor (see isPageTokenName), including when
// intra-word separators split "page", "token" or the "next", "prev" or
// "previous" before them, as in n_e_x_t_page_token or p_r_e_v.page.token.
func isPageTokenWords(words []string) bool {
	n := len(words)
	if n < 2 {
		return false
	}
	t := -1
	for i := n - 1; i >= max(1, n-len("token")); i-- {
		if joined := strings.Join(words[i:], ""); joined == "token" {
			t = i
			break
		} else if len(joined) >= len("token") {
			break
		}
	}
	if t < 1 {
		return false
	}
	p := -1
	for i := t - 1; i >= max(0, t-len("page")); i-- {
		if joined := strings.Join(words[i:t], ""); joined == "page" {
			p = i
			break
		} else if len(joined) >= len("page") {
			break
		}
	}
	if p < 0 {
		return false
	}
	if p == 0 {
		return true
	}
	// The word before "page" is rebuilt from at most len("previous") words,
	// as "page" and "token" are, so that a split direction word is read whole
	// and a word that only ends in one, as the "arn"+"ext" of arn_ext, is not.
	for i := p - 1; i >= max(0, p-len("previous")); i-- {
		joined := strings.Join(words[i:p], "")
		switch joined {
		case "next", "prev", "previous":
			return true
		}
		if len(joined) >= len("previous") {
			break
		}
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
	// Only the last maxCompactSensitiveKeyLen words can form a credential name,
	// so the suffixes are not rebuilt for a long prefix of other words.
	start := max(1, len(words)-maxCompactSensitiveKeyLen)
	for i := start; i < len(words); i++ {
		if isCredentialSuffix(strings.Join(words[i:], "")) {
			return true
		}
	}
	// Of an acronym, only the letters that leave room for a name of
	// sensitiveKeys are tried, so a long one is not read to its start.
	for i := max(0, len(words)-maxCompactSensitiveKeyLen); i < len(words)-1; i++ {
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

// isBareTextKey reports whether bareTextKeyRe matches key. Each key that it
// matches, a "temp:" key or a credential name, starts with the letters of a
// name of textKeyNames, so it is matched only where keyNameSet.mayStartAt
// finds them.
func isBareTextKey(key string) bool {
	return textKeyNames.mayStartAt(key, 0) && bareTextKeyRe.MatchString(key)
}

// isWholeTextKeyName reports whether textKeyNameWholeRe matches name. It is
// matched only where mayStartTextKeyName finds that a key name may start in
// the compound prefix that name starts with, as it finds in each name that
// textKeyNameWholeRe matches.
func isWholeTextKeyName(name string) bool {
	return mayStartTextKeyName(name) && textKeyNameWholeRe.MatchString(name)
}

// textKeyAllowed reports whether a key that keyValFinder found is to be
// redacted: a plain sensitive key name always is, and a compound name is
// if isSensitiveKey accepts it or a "-" in it precedes a plain name.
func textKeyAllowed(key string) bool {
	if isBareTextKey(key) || isSensitiveKey(key) {
		return true
	}
	if isPageTokenName(key) {
		// next-page-token: its "token" must not match on its own below.
		return false
	}
	// A sensitive name after a "-" has always been matched on its own, as in
	// foo-signature=..., whose "-" is a word boundary.
	for i := strings.IndexByte(key, '-'); i >= 0; {
		if isBareTextKey(key[i+1:]) {
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
		if i <= lo && words >= maxCompactSensitiveKeyLen && isSeparator(s[i-1]) {
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
	ansicClose := -1
	for i, r := range s {
		if i >= ansicClose {
			ansicClose = -1
		}
		if ansicClose < 0 && r == '$' && i+1 < len(s) && s[i+1] == '\'' {
			bs := 0
			for bs < i && s[i-1-bs] == '\\' {
				bs++
			}
			if bs%2 == 0 {
				if end := scanANSICQuotedClose(s, i+2, false); end >= 0 {
					ansicClose = end
				}
			}
		}
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
			} else if n >= 2 && output[n-2] == '\\' && (output[n-1] == '"' || (output[n-1] == '\'' && ansicClose < 0)) {
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
		scrubbed := scrubErrorTextWithBudget(raw, 0, budget)
		if scrubbed != raw {
			return scrubbed, true
		}
		return raw, false
	}
	if sanitized != unparseableJSONPlaceholder {
		scrubbed := scrubErrorTextWithBudget(sanitized, 0, budget)
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
		// A string leaf of a JSON value that a shell line splice breaks, as
		// the one of {"note": "DB_PASS\\\nWORD=x"} is, is scrubbed on its
		// own: a scrub of the encoded JSON text around it reads the splice
		// as the escapes \\ and \n rather than as a backslash and a line
		// break, and so finds no key there.
		if depth > 0 && (strings.Contains(s, "\\\n") || strings.Contains(s, "\\\r")) {
			if scrubbed := scrubErrorText(s); scrubbed != s {
				return scrubbed, true, false
			}
		}
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
			sensitiveKey := isSensitiveJSONKey(k)
			newK := scrubJSONMapKeyWithBudget(k, depth, budget)
			if newK != k {
				changed = true
			}
			if sensitiveKey {
				if budget != nil && !budget.take() {
					outMap[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
					return outMap, true, true
				}
				if str, ok := val.(string); !ok || str != redactedValue {
					changed = true
				}
				outMap[newK] = redactedValue
				continue
			}
			newVal, c, lost := redactJSONValue(val, depth+1, budget)
			if str, ok := newVal.(string); ok && str == sanitizeBudgetPlaceholder && budget != nil && budget.remaining <= 0 {
				outMap[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
				return outMap, true, true
			}
			if existing, exists := outMap[newK]; exists {
				if !reflect.DeepEqual(existing, newVal) {
					outMap[newK] = redactedValue
				}
				changed = true
			} else {
				outMap[newK] = newVal
			}
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
			scrubbed, strChanged := scrubErrorJSONStringsWithBudget(redacted, 1, budget)
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
		return scrubErrorTextWithBudget(msg, 0, budget)
	}
	if strings.HasPrefix(trimmed, `"`) {
		if redacted, ok, _ := redactJSONTextWithBudget(msg, 0, budget); ok {
			// As in sanitizeRawText, quote-led prose that only failed JSON
			// parsing is kept.
			if redacted == unparseableJSONPlaceholder && isPositivelyHarmlessProse(msg, trimmed) {
				return msg
			}
			return scrubErrorTextWithBudget(redacted, 0, budget)
		}
	}
	return scrubErrorTextWithBudget(msg, 0, budget)
}

func scrubErrorJSONStringsWithBudget(v any, depth int, budget *nodeBudget) (any, bool) {
	if depth >= maxSanitizeDepth {
		return v, false
	}
	if budget == nil {
		budget = &nodeBudget{remaining: maxSanitizeNodes}
	}
	switch x := v.(type) {
	case map[string]any:
		changed := false
		out := make(map[string]any, len(x))
		for k, val := range x {
			newK := scrubJSONMapKeyWithBudget(k, depth, budget)
			if newK != k {
				changed = true
			}
			newVal, c := scrubErrorJSONStringsWithBudget(val, depth+1, budget)
			if existing, exists := out[newK]; exists {
				if !reflect.DeepEqual(existing, newVal) {
					out[newK] = redactedValue
				}
				changed = true
			} else {
				out[newK] = newVal
			}
			changed = changed || c
		}
		return out, changed
	case []any:
		changed := false
		out := make([]any, len(x))
		for i, val := range x {
			newVal, c := scrubErrorJSONStringsWithBudget(val, depth+1, budget)
			out[i] = newVal
			changed = changed || c
		}
		return out, changed
	case string:
		if isKnownSentinel(x) {
			return x, false
		}
		scrubbed := scrubErrorTextWithBudget(x, depth, budget)
		return scrubbed, scrubbed != x
	default:
		return v, false
	}
}

func scrubErrorText(msg string) string {
	return scrubErrorTextWithBudget(msg, 0, &nodeBudget{remaining: maxSanitizeNodes})
}

func scrubErrorTextWithBudget(msg string, depth int, budget *nodeBudget) string {
	if depth >= maxSanitizeDepth {
		return redactedSensitiveTextPlaceholder
	}
	if budget == nil {
		budget = &nodeBudget{remaining: maxSanitizeNodes}
	}
	if cached, ok := budget.getErrorText(msg, depth); ok {
		return cached
	}
	res := scrubErrorTextWithBudgetUncached(msg, depth, budget)
	budget.putErrorText(msg, depth, res)
	return res
}

func scrubErrorTextWithBudgetUncached(msg string, depth int, budget *nodeBudget) string {
	scrubbed := scrubErrorTokens(msg)
	if exposes, ok := jsonExposesCredentialWithBudget(msg, scrubbed, depth, budget); ok {
		if !exposes {
			return scrubbed
		}
		if decoded, ok := decodeJSONDocumentWithBudget(trimBOMAndSpace(msg), budget); ok {
			redacted, valChanged, valLost := redactJSONValue(decoded, depth+1, budget)
			if !valLost {
				scrubbedJSON, strChanged := scrubErrorJSONStringsWithBudget(redacted, depth+1, budget)
				if valChanged || (strChanged && countFailClosedPlaceholders(scrubbedJSON) <= countFailClosedPlaceholders(redacted) && onlyNestedJSONContainerChanges(redacted, scrubbedJSON, budget)) {
					if b, err := json.Marshal(scrubbedJSON); err == nil {
						if exposes2, ok2 := jsonExposesCredentialWithBudget(msg, string(b), depth, budget); ok2 && !exposes2 {
							return string(b)
						}
					}
				}
			}
		}
		return redactedSensitiveTextPlaceholder
	}
	if hasEncodedCredentialMarker(scrubbed) || hasBackslashEscapedCredentialWithBudget(msg, scrubbed, depth, budget) {
		return redactedSensitiveTextPlaceholder
	}
	return scrubbed
}

func onlyNestedJSONContainerChanges(before, after any, budget *nodeBudget) bool {
	switch b := before.(type) {
	case map[string]any:
		a, ok := after.(map[string]any)
		if !ok || len(a) != len(b) {
			return false
		}
		for k, bv := range b {
			av, exists := a[k]
			if !exists || !onlyNestedJSONContainerChanges(bv, av, budget) {
				return false
			}
		}
		return true
	case []any:
		a, ok := after.([]any)
		if !ok || len(a) != len(b) {
			return false
		}
		for i := range b {
			if !onlyNestedJSONContainerChanges(b[i], a[i], budget) {
				return false
			}
		}
		return true
	case string:
		a, ok := after.(string)
		if !ok {
			return false
		}
		if a == b {
			return true
		}
		trimmed := trimBOMAndSpace(b)
		if trimmed == "" || (trimmed[0] != '{' && trimmed[0] != '[') {
			return false
		}
		_, valid := decodeJSONDocumentWithBudget(trimmed, budget)
		return valid
	default:
		return reflect.DeepEqual(before, after)
	}
}

func countFailClosedPlaceholders(v any) int {
	switch x := v.(type) {
	case map[string]any:
		n := 0
		for k, val := range x {
			n += countFailClosedPlaceholders(k) + countFailClosedPlaceholders(val)
		}
		return n
	case []any:
		n := 0
		for _, val := range x {
			n += countFailClosedPlaceholders(val)
		}
		return n
	case string:
		if x == redactedSensitiveTextPlaceholder || x == unparseableJSONPlaceholder || x == maxDepthPlaceholder || x == sanitizeBudgetPlaceholder {
			return 1
		}
	}
	return 0
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

// hasBackslashEscapedCredentialDepth reports whether unescaping standard backslash
// escapes (\b, \f, \n, \r, \t, \/, \", \' and \\), or canonicalizing percent,
// \x or \u escapes, exposes a credential that scrubErrorTokens or
// hasEncodedCredentialMarker catches, including when scrubErrorTokens on raw
// consumed an escape inside an earlier value and left a following secret in
// scrubbed. Each pass decodes a run of backslashes in one step (see
// unescapeBackslashControls), so passes are needed only where percent, \x or
// \u escapes and backslash escapes nest inside each other; up to
// 2*maxCanonicalUnescapePasses are run, failing closed if escapes still change
// past that depth. When raw and scrubbed are each one JSON document, the
// keys and strings of raw and scrubbed are decoded exactly and checked in
// pairs instead (see jsonExposesCredentialDepth).
func hasBackslashEscapedCredentialWithBudget(raw, scrubbed string, depth int, budget *nodeBudget) bool {
	if depth >= maxSanitizeDepth {
		return true
	}
	if budget == nil {
		budget = &nodeBudget{remaining: maxSanitizeNodes}
	}
	if !strings.ContainsAny(raw, `\%`) {
		return false
	}
	if exposes, ok := jsonExposesCredentialWithBudget(raw, scrubbed, depth, budget); ok {
		return exposes
	}
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

// jsonExposesCredentialDepth reports, when raw and scrubbed are each one JSON
// document (an object, an array or a string) of the same shape, whether
// scrubbed, decoded exactly, still holds a credential: a value under a
// sensitive key, or JSON text in a string that holds one (see
// redactJSONValue), or a key or a string that exposes a credential of the key
// or the string at the same place in raw (see jsonStringExposes). ok is
// false, and the caller unescapes the whole text instead, when they are not
// such documents or do not decode within the limits that redactJSONValue
// reads JSON with. Unescaping the whole of a JSON document would read the
// escapes of one string as bytes of the text around it: the \n of
// {"note":"The cookie: 2.5 ounces remained.\nKEEP"} becomes a line break that
// leaves the quote of the string open across it, and a scrub of that text
// then reads the prose in the string as a Cookie header's value.
func jsonExposesCredentialWithBudget(raw, scrubbed string, depth int, budget *nodeBudget) (exposes, ok bool) {
	if depth >= maxSanitizeDepth {
		return true, true
	}
	if budget == nil {
		budget = &nodeBudget{remaining: maxSanitizeNodes}
	}
	raw, scrubbed = trimBOMAndSpace(raw), trimBOMAndSpace(scrubbed)
	if len(raw) > maxInspectableJSONTextBytes || !startsJSONDocument(raw) || !startsJSONDocument(scrubbed) ||
		!utf8.ValidString(raw) || !utf8.ValidString(scrubbed) {
		return false, false
	}
	rawValue, ok := decodeJSONDocumentWithBudget(raw, budget)
	if !ok {
		return false, false
	}
	scrubbedValue, ok := decodeJSONDocumentWithBudget(scrubbed, budget)
	if !ok {
		return false, false
	}
	_, changed, lost := redactJSONValue(scrubbedValue, depth+1, budget)
	if lost {
		return false, false
	}
	if changed {
		return true, true
	}
	return jsonValuesExposeWithBudget(rawValue, scrubbedValue, depth+1, budget)
}

// startsJSONDocument reports whether s, with no BOM or space around it,
// starts a JSON object, array or string.
func startsJSONDocument(s string) bool {
	return s != "" && (s[0] == '{' || s[0] == '[' || s[0] == '"')
}

// decodeJSONDocument decodes s, one JSON document, as redactJSONValue reads
// JSON: within its node budget, and with no key twice in an object.
func decodeJSONDocumentWithBudget(s string, budget *nodeBudget) (any, bool) {
	if budget == nil {
		budget = &nodeBudget{remaining: maxSanitizeNodes}
	}
	v, sawDuplicateKey, exceeded, err := unmarshalJSONStrictWithBudget([]byte(s), budget)
	return v, err == nil && !exceeded && !sawDuplicateKey
}

// jsonValuesExpose reports whether scrubbed, the decoded value of a scrub of
// the JSON text of raw, keeps a credential of raw: whether a key or a string
// of scrubbed exposes one of the key or the string at the same place in raw
// (see jsonStringExposes). The keys of an object are read in sorted order.
// ok is false when raw and scrubbed differ in shape, so that their keys and
// strings cannot be paired. Past maxSanitizeDepth it reports true.
func jsonValuesExposeWithBudget(raw, scrubbed any, depth int, budget *nodeBudget) (exposes, ok bool) {
	if depth >= maxSanitizeDepth {
		return true, true
	}
	if budget == nil {
		budget = &nodeBudget{remaining: maxSanitizeNodes}
	}
	switch r := raw.(type) {
	case map[string]any:
		s, isMap := scrubbed.(map[string]any)
		if !isMap || len(s) != len(r) {
			return false, false
		}
		keys := make([]string, 0, len(r))
		scrubbedKeys := make(map[string]string, len(r))
		for k := range r {
			sk := k
			if _, found := s[k]; !found {
				sk = scrubJSONMapKeyWithBudget(k, depth, budget)
				if _, found := s[sk]; !found {
					return false, false
				}
			}
			scrubbedKeys[k] = sk
			keys = append(keys, k)
		}
		slices.Sort(keys)
		for _, k := range keys {
			sk := scrubbedKeys[k]
			if isSensitiveJSONKey(k) && s[sk] != redactedValue {
				return true, true
			}
			if !isKnownSentinel(sk) {
				if scrubJSONMapKeyWithBudget(sk, depth, budget) != sk {
					return true, true
				}
				if prefix, rawSuffix, ok := splitTempKeyPrefix(k); ok {
					if strings.HasPrefix(sk, prefix) {
						skSuffix := sk[len(prefix):]
						if !isKnownSentinel(skSuffix) && jsonStringExposesWithBudget(rawSuffix, skSuffix, depth, budget) {
							return true, true
						}
					} else if jsonStringExposesWithBudget(rawSuffix, sk, depth, budget) {
						return true, true
					}
				} else if jsonStringExposesWithBudget(k, sk, depth, budget) {
					return true, true
				}
			}
			if exposes, ok := jsonValuesExposeWithBudget(r[k], s[sk], depth+1, budget); exposes || !ok {
				return exposes, ok
			}
		}
		return false, true
	case []any:
		s, isSlice := scrubbed.([]any)
		if !isSlice || len(s) != len(r) {
			return false, false
		}
		for i := range r {
			if exposes, ok := jsonValuesExposeWithBudget(r[i], s[i], depth+1, budget); exposes || !ok {
				return exposes, ok
			}
		}
		return false, true
	case string:
		s, isString := scrubbed.(string)
		if !isString {
			return false, false
		}
		return jsonStringExposesWithBudget(r, s, depth, budget), true
	default:
		return false, raw == scrubbed
	}
}

// jsonStringExposes reports whether scrubbed, a key or a string of the
// decoded scrub of JSON text, exposes a credential of raw, the key or the
// string at the same place in the decoded text itself. When nothing of raw
// was redacted, it does if scrubErrorText redacts raw. Otherwise it does if
// scrubErrorText still redacts scrubbed, or redacts a part of raw that
// scrubbed keeps (see keptInOrder), as for a scrub of the escaped text that
// reads secret: a\npassword: SECRET as one value and keeps SECRET.
func jsonStringExposesWithBudget(raw, scrubbed string, depth int, budget *nodeBudget) bool {
	if depth >= maxSanitizeDepth {
		return true
	}
	if budget == nil {
		budget = &nodeBudget{remaining: maxSanitizeNodes}
	}
	if raw == scrubbed {
		return !isKnownSentinel(raw) && scrubErrorTextWithBudget(raw, depth, budget) != raw
	}
	if !isKnownSentinel(scrubbed) && scrubErrorTextWithBudget(scrubbed, depth, budget) != scrubbed {
		return true
	}
	return !keptInOrder(keptLiterals(scrubbed), scrubErrorTextWithBudget(raw, depth, budget))
}

func splitTempKeyPrefix(k string) (prefix, suffix string, ok bool) {
	i := 0
	for i < len(k) {
		r, w := utf8.DecodeRuneInString(k[i:])
		if !isDelimSpace(r) {
			break
		}
		i += w
	}
	lower := strings.ToLower(k[i:])
	if strings.HasPrefix(lower, session.KeyPrefixTemp) {
		cut := i + len(session.KeyPrefixTemp)
		return k[:cut], k[cut:], true
	}
	if strings.HasPrefix(lower, session.KeyPrefixUser) {
		i += len(session.KeyPrefixUser)
	} else if strings.HasPrefix(lower, session.KeyPrefixApp) {
		i += len(session.KeyPrefixApp)
	} else {
		return "", "", false
	}
	for i < len(k) {
		r, w := utf8.DecodeRuneInString(k[i:])
		if !isDelimSpace(r) {
			break
		}
		i += w
	}
	if strings.HasPrefix(strings.ToLower(k[i:]), session.KeyPrefixTemp) {
		cut := i + len(session.KeyPrefixTemp)
		return k[:cut], k[cut:], true
	}
	return "", "", false
}

func mayContainKeyCredential(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c <= ' ' || c >= utf8.RuneSelf {
			return true
		}
		switch c {
		case ':', '=', '%', '\\', '{', '[', '"', '\'':
			return true
		}
	}
	return false
}

func scrubJSONMapKey(k string, depth int) string {
	return scrubJSONMapKeyWithBudget(k, depth, &nodeBudget{remaining: maxSanitizeNodes})
}

func scrubJSONMapKeyWithBudget(k string, depth int, budget *nodeBudget) string {
	if depth >= maxSanitizeDepth {
		return redactedSensitiveTextPlaceholder
	}
	if budget == nil {
		budget = &nodeBudget{remaining: maxSanitizeNodes}
	}
	if isKnownSentinel(k) || !mayContainKeyCredential(k) {
		return k
	}
	if cached, ok := budget.getKeyScrub(k, depth); ok {
		return cached
	}
	var res string
	if prefix, suffix, ok := splitTempKeyPrefix(k); ok {
		if suffix == "" || isKnownSentinel(suffix) || !mayContainKeyCredential(suffix) {
			res = k
		} else {
			res = prefix + scrubErrorTextWithBudget(suffix, depth, budget)
		}
	} else {
		res = scrubErrorTextWithBudget(k, depth, budget)
	}
	budget.putKeyScrub(k, depth, res)
	return res
}

func isSensitiveJSONKey(k string) bool {
	if isSensitiveKey(k) {
		return true
	}
	if !strings.ContainsAny(k, `\%`) {
		return false
	}
	cur := k
	for range 2 * maxCanonicalUnescapePasses {
		if !strings.ContainsAny(cur, `\%`) {
			return false
		}
		next := unescapeCredentialStep(cur)
		if next == cur {
			return false
		}
		if isSensitiveKey(next) {
			return true
		}
		cur = next
	}
	return unescapeCredentialStep(cur) != cur
}

func hasUnquotedShellComment(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] == '#' && (i == 0 || s[i-1] == ' ' || s[i-1] == '\t') {
			return true
		}
	}
	return false
}

func (h *headerQuoteScanner) isShellBalancedFrom(start, lineEnd int, inJSON bool) bool {
	if h.balLineEnd != lineEnd || h.balInJSON != inJSON || h.balMemo == nil {
		h.balLineEnd = lineEnd
		h.balInJSON = inJSON
		h.balMemo = make(map[int]bool)
	}
	var visited []int
	stamp := func(res bool) bool {
		for _, pos := range visited {
			h.balMemo[pos] = res
		}
		return res
	}
	s := h.s
	p := start
	for p < lineEnd {
		if v, ok := h.balMemo[p]; ok {
			return stamp(v)
		}
		visited = append(visited, p)
		c := s[p]
		switch c {
		case '\\':
			if inJSON {
				bs := 0
				for p+bs < lineEnd && s[p+bs] == '\\' {
					bs++
				}
				if p+bs < lineEnd && s[p+bs] == '"' && bs%4 == 1 {
					end := h.doubleQuoteCloseJSONWithin(p+bs+1, lineEnd, true)
					if end < 0 {
						return stamp(false)
					}
					p = end
					continue
				}
				if p+bs < lineEnd && (s[p+bs] == '$' || s[p+bs] == '`') {
					if bs%4 == 2 {
						p += bs + 1
						continue
					}
					if bs%4 == 0 {
						p += bs
						continue
					}
				}
				if bs%2 == 1 && p+bs < lineEnd {
					p += bs + 1
				} else {
					p += bs
				}
				continue
			}
			p += 2
		case '\'':
			end := strings.IndexByte(s[p+1:lineEnd], '\'')
			if end < 0 {
				return stamp(false)
			}
			p += 1 + end + 1
		case '"':
			end := h.doubleQuoteCloseWithin(p+1, lineEnd)
			if end < 0 {
				return stamp(false)
			}
			p = end
		case '$':
			if p+1 < lineEnd && s[p+1] == '\'' {
				end := scanANSICQuotedClose(s[:lineEnd], p+2, inJSON)
				if end < 0 {
					return stamp(false)
				}
				p = end
				continue
			}
			if p+1 < lineEnd && s[p+1] == '(' {
				end := h.cmdSubstCloseWithin(p+2, lineEnd, inJSON)
				if end < 0 {
					return stamp(false)
				}
				p = end
				continue
			}
			if p+1 < lineEnd && s[p+1] == '{' {
				end := h.paramExpCloseWithin(p+2, false, lineEnd, inJSON)
				if end < 0 {
					return stamp(false)
				}
				p = end
				continue
			}
			p++
		case '`':
			end := h.backtickCloseWithin(p+1, lineEnd, false, inJSON)
			if end < 0 {
				return stamp(false)
			}
			p = end
		case '#':
			if p == 0 || s[p-1] == ' ' || s[p-1] == '\t' || strings.IndexByte(";&|()", s[p-1]) >= 0 {
				return stamp(true)
			}
			p++
		default:
			p++
		}
	}
	return stamp(true)
}

// isHeaderFlagGap reports whether gap between an apostrophe token and a
// subsequent single-quoted auth header contains only validated shell command
// separators, curl/wget flags/URLs/redirects, safe non-secret headers or
// key=value data args, and terminates with -H/--header.
func (h *headerQuoteScanner) isHeaderFlagGap(k int, gap string, lineEnd int, escJSON, trailingApostrophe bool) bool {
	if len(gap) == 0 {
		return false
	}
	if gap[0] == ')' {
		if h == nil || !h.hasOpenSubshellBefore(k, escJSON) {
			return false
		}
		gap = gap[1:]
		if len(gap) == 0 {
			return false
		}
	}
	if gap[0] != ' ' && gap[0] != '\t' && strings.IndexByte(";&|<>()", gap[0]) < 0 {
		return false
	}
	last := gap[len(gap)-1]
	var trimmed string
	switch last {
	case '=':
		if !strings.HasSuffix(gap, "--header=") && !strings.HasSuffix(gap, "-H=") {
			return false
		}
		trimmed = gap[:len(gap)-1]
	case ' ', '\t':
		trimmed = gap
	default:
		return false
	}
	var segs []string
	segStart := 0
	inDouble := false
	curTokEnd := -1
	curTokHasScheme := false
	isTokDelim := func(c byte) bool {
		switch c {
		case ' ', '\t', ';', '|', '(', ')', '<', '>', '"':
			return true
		default:
			return false
		}
	}
	for i := 0; i < len(trimmed); i++ {
		c := trimmed[i]
		if inDouble {
			if c == '\\' {
				if escJSON {
					bs := 0
					for i+bs < len(trimmed) && trimmed[i+bs] == '\\' {
						bs++
					}
					if i+bs < len(trimmed) && trimmed[i+bs] == '"' {
						if bs%4 == 1 {
							inDouble = false
						}
						i += bs
						continue
					}
					if bs%2 == 1 && i+bs < len(trimmed) {
						i += bs
						continue
					}
					i += bs - 1
					continue
				}
				i++
				continue
			}
			if c == '"' {
				if escJSON {
					return false
				}
				inDouble = false
			}
			continue
		}
		switch c {
		case '\\':
			if escJSON {
				bs := 0
				for i+bs < len(trimmed) && trimmed[i+bs] == '\\' {
					bs++
				}
				if i+bs < len(trimmed) && trimmed[i+bs] == '"' && bs%4 == 1 {
					inDouble = true
					i += bs
					continue
				}
				if bs%2 == 1 && i+bs < len(trimmed) {
					i += bs
					continue
				}
				i += bs - 1
				continue
			}
			i++
		case '"':
			if escJSON {
				return false
			}
			inDouble = true
		case ';':
			segs = append(segs, trimmed[segStart:i])
			segStart = i + 1
		case '|':
			segs = append(segs, trimmed[segStart:i])
			if i+1 < len(trimmed) && trimmed[i+1] == '|' {
				i++
			}
			segStart = i + 1
		case '&':
			if i+1 < len(trimmed) && trimmed[i+1] == '&' {
				segs = append(segs, trimmed[segStart:i])
				i++
				segStart = i + 1
				continue
			}
			if (i > 0 && trimmed[i-1] == '>') || (i+1 < len(trimmed) && trimmed[i+1] == '>') {
				continue
			}
			if i >= curTokEnd {
				tokStart := i
				for tokStart > 0 && !isTokDelim(trimmed[tokStart-1]) {
					tokStart--
				}
				curTokEnd = i + 1
				for curTokEnd < len(trimmed) && !isTokDelim(trimmed[curTokEnd]) {
					curTokEnd++
				}
				curTokHasScheme = strings.Contains(trimmed[tokStart:curTokEnd], "://")
			}
			if curTokHasScheme {
				continue
			}
			segs = append(segs, trimmed[segStart:i])
			segStart = i + 1
		}
	}
	if inDouble {
		return false
	}
	segs = append(segs, trimmed[segStart:])
	if strings.TrimSpace(segs[len(segs)-1]) == "" {
		return false
	}
	startsAfterUnquotedSep := len(segs) > 1 && strings.TrimSpace(segs[0]) == ""
	lastNonEmpty := -1
	for idx, seg := range segs {
		if strings.TrimSpace(seg) != "" {
			lastNonEmpty = idx
		}
	}
	if lastNonEmpty < 0 {
		return false
	}
	for idx, seg := range segs {
		segTokens, ok := splitHeaderGapTokens(seg, escJSON)
		if !ok {
			return false
		}
		if len(segTokens) == 0 {
			continue
		}
		allowCmd := idx > 0 || startsAfterUnquotedSep
		isFinal := idx == lastNonEmpty
		if !isValidHeaderFlagGapTokens(segTokens, allowCmd, isFinal, escJSON, trailingApostrophe) {
			return false
		}
	}
	return true
}

// splitHeaderGapTokens splits a single gap command segment on horizontal
// whitespace while keeping double-quoted (or JSON-escaped \"...\") arguments as
// single tokens.
func splitHeaderGapTokens(seg string, escJSON bool) ([]string, bool) {
	var toks []string
	i := 0
	for i < len(seg) {
		for i < len(seg) && (seg[i] == ' ' || seg[i] == '\t') {
			i++
		}
		if i >= len(seg) {
			break
		}
		tokStart := i
		inDouble := false
		for i < len(seg) {
			c := seg[i]
			if inDouble {
				if c == '\\' {
					if escJSON {
						bs := 0
						for i+bs < len(seg) && seg[i+bs] == '\\' {
							bs++
						}
						if i+bs < len(seg) && seg[i+bs] == '"' {
							if bs%4 == 1 {
								inDouble = false
							}
							i += bs + 1
							continue
						}
						if bs%2 == 1 && i+bs < len(seg) {
							i += bs + 1
							continue
						}
						i += bs
						continue
					}
					i += 2
					continue
				}
				if c == '"' {
					if escJSON {
						return nil, false
					}
					inDouble = false
				}
				i++
				continue
			}
			if c == ' ' || c == '\t' {
				break
			}
			if c == '\\' {
				if escJSON {
					bs := 0
					for i+bs < len(seg) && seg[i+bs] == '\\' {
						bs++
					}
					if i+bs < len(seg) && seg[i+bs] == '"' && bs%4 == 1 {
						inDouble = true
						i += bs + 1
						continue
					}
					if bs%2 == 1 && i+bs < len(seg) {
						i += bs + 1
						continue
					}
					i += bs
					continue
				}
				i += 2
				continue
			}
			if c == '"' {
				if escJSON {
					return nil, false
				}
				inDouble = true
			}
			i++
		}
		if inDouble {
			return nil, false
		}
		toks = append(toks, seg[tokStart:i])
	}
	return toks, true
}

func unquoteGapTok(tok string, escJSON bool) (string, bool) {
	if escJSON && len(tok) >= 4 && strings.HasPrefix(tok, `\"`) && strings.HasSuffix(tok, `\"`) {
		return tok[2 : len(tok)-2], true
	}
	if !escJSON && len(tok) >= 2 && tok[0] == '"' && tok[len(tok)-1] == '"' {
		return tok[1 : len(tok)-1], true
	}
	return tok, false
}

func isSafeGapShellRedirect(tok string) bool {
	switch tok {
	case "2>&1", "1>&2", "&>/dev/null", ">/dev/null", "2>/dev/null", "1>/dev/null":
		return true
	default:
		return false
	}
}

func isSafeGapHTTPMethod(m string) bool {
	switch strings.ToUpper(m) {
	case "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS":
		return true
	default:
		return false
	}
}

func isSafeGapKVList(arg string) bool {
	if len(arg) == 0 {
		return false
	}
	for _, part := range strings.Split(arg, "&") {
		eq := strings.IndexByte(part, '=')
		if eq <= 0 {
			return false
		}
		k := part[:eq]
		if strings.ContainsAny(k, " \t") || isSensitiveKey(k) {
			return false
		}
	}
	return true
}

func isSafeGapURL(u string) bool {
	scheme := strings.Index(u, "://")
	if scheme <= 0 {
		return false
	}
	p := strings.ToLower(u[:scheme])
	if p != "http" && p != "https" {
		return false
	}
	rest := u[scheme+3:]
	if len(rest) == 0 {
		return false
	}
	authEnd := strings.IndexAny(rest, "/?#")
	auth := rest
	if authEnd >= 0 {
		auth = rest[:authEnd]
	}
	if strings.Contains(auth, "@") {
		return false
	}
	if q := strings.IndexByte(rest, '?'); q >= 0 {
		query := rest[q+1:]
		if frag := strings.IndexByte(query, '#'); frag >= 0 {
			query = query[:frag]
		}
		if query != "" {
			for _, part := range strings.Split(query, "&") {
				if eq := strings.IndexByte(part, '='); eq > 0 {
					if isSensitiveKey(part[:eq]) {
						return false
					}
				}
			}
		}
	}
	return true
}

func isSafeNonSecretGapHeaderName(name string) bool {
	switch name {
	case "accept", "accept-encoding", "accept-language", "content-type",
		"content-length", "user-agent", "cache-control", "pragma",
		"host", "origin", "referer", "connection":
		return true
	default:
		return false
	}
}

func isSafeGapHeaderArg(hdr string) bool {
	colon := strings.IndexByte(hdr, ':')
	if colon <= 0 {
		return false
	}
	name := strings.ToLower(strings.TrimSpace(hdr[:colon]))
	if name == "" || isSensitiveKey(name) {
		return false
	}
	val := strings.TrimSpace(hdr[colon+1:])
	if isSafeNonSecretGapHeaderName(name) {
		for _, param := range strings.Split(val, ";") {
			param = strings.TrimSpace(param)
			if eq := strings.IndexByte(param, '='); eq > 0 {
				if isSensitiveKey(strings.TrimSpace(param[:eq])) {
					return false
				}
			}
		}
		return true
	}
	return !strings.ContainsAny(val, " \t") && isSafeGapKVList(val)
}

func isSafeGapBoolFlag(tok string) bool {
	if tok == "-qO-" {
		return true
	}
	if strings.HasPrefix(tok, "--") {
		if len(tok) <= 2 {
			return false
		}
		rest := tok[2:]
		if rest[0] < 'a' || rest[0] > 'z' {
			return false
		}
		for i := 1; i < len(rest); i++ {
			c := rest[i]
			if (c < 'a' || c > 'z') && (c < '0' || c > '9') && c != '-' {
				return false
			}
		}
		return true
	}
	if len(tok) < 2 || len(tok) > 4 || tok[0] != '-' {
		return false
	}
	for i := 1; i < len(tok); i++ {
		c := tok[i]
		if (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') {
			return false
		}
	}
	return true
}

// isValidHeaderFlagGapTokens reports whether segTokens holds only recognized
// shell wrappers/curl/wget command invocations, safe URLs/redirects/flags, and
// non-secret args, ending with -H/--header when isFinalSeg is true.
func isValidHeaderFlagGapTokens(segTokens []string, allowCmdPrefix, isFinalSeg, escJSON, trailingApostrophe bool) bool {
	pos := 0
	if allowCmdPrefix {
		for pos < len(segTokens) {
			tok := strings.TrimLeft(segTokens[pos], "(")
			switch tok {
			case "env":
				if trailingApostrophe {
					return false
				}
				pos++
				for pos < len(segTokens) {
					next := strings.TrimLeft(segTokens[pos], "(")
					if strings.HasPrefix(next, "-") {
						pos++
						continue
					}
					if eq := strings.IndexByte(next, '='); eq > 0 && next[0] != '-' && !isSensitiveKey(next[:eq]) {
						pos++
						continue
					}
					break
				}
			case "sudo", "command", "nohup", "time", "xargs", "nice":
				pos++
				for pos < len(segTokens) && strings.HasPrefix(strings.TrimLeft(segTokens[pos], "("), "-") {
					pos++
				}
			default:
				goto checkedWrapper
			}
		}
	checkedWrapper:
		if pos >= len(segTokens) {
			return false
		}
		cmd := strings.TrimLeft(segTokens[pos], "(")
		switch {
		case cmd == "curl" || cmd == "wget" || strings.HasSuffix(cmd, "/curl") || strings.HasSuffix(cmd, "/wget"):
			pos++
		case cmd == "-H" || cmd == "--header":
			// Only permitted as the trailing flag or an explicit safe intermediate -H.
		default:
			return false
		}
	}
	limit := len(segTokens)
	if isFinalSeg {
		if pos >= len(segTokens) {
			return false
		}
		lastTok := segTokens[len(segTokens)-1]
		if allowCmdPrefix && pos == len(segTokens)-1 {
			lastTok = strings.TrimLeft(lastTok, "(")
		}
		if lastTok != "-H" && lastTok != "--header" {
			return false
		}
		limit = len(segTokens) - 1
	}
	for pos < limit {
		tok := segTokens[pos]
		if isSafeGapShellRedirect(tok) {
			pos++
			continue
		}
		if unq, _ := unquoteGapTok(tok, escJSON); (!trailingApostrophe || !strings.ContainsAny(unq, "?#@=")) && isSafeGapURL(unq) {
			pos++
			continue
		}
		if tok == "-X" || tok == "--request" {
			if pos+1 >= limit {
				return false
			}
			method, _ := unquoteGapTok(segTokens[pos+1], escJSON)
			if !isSafeGapHTTPMethod(method) {
				return false
			}
			pos += 2
			continue
		}
		if strings.HasPrefix(tok, "-X=") && isSafeGapHTTPMethod(tok[3:]) {
			pos++
			continue
		}
		if strings.HasPrefix(tok, "--request=") && isSafeGapHTTPMethod(tok[len("--request="):]) {
			pos++
			continue
		}
		if tok == "-d" || tok == "--data" || tok == "--data-raw" || tok == "--data-binary" || tok == "-F" || tok == "--form" || tok == "-b" {
			if trailingApostrophe || pos+1 >= limit {
				return false
			}
			arg, _ := unquoteGapTok(segTokens[pos+1], escJSON)
			if !isSafeGapKVList(arg) {
				return false
			}
			pos += 2
			continue
		}
		if eq := strings.IndexByte(tok, '='); eq > 0 {
			flagPart, valPart := tok[:eq], tok[eq+1:]
			if flagPart == "--data" || flagPart == "--data-raw" || flagPart == "--data-binary" || flagPart == "--form" {
				if trailingApostrophe {
					return false
				}
				arg, _ := unquoteGapTok(valPart, escJSON)
				if !isSafeGapKVList(arg) {
					return false
				}
				pos++
				continue
			}
		}
		if tok == "-H" || tok == "--header" {
			if trailingApostrophe || pos+1 >= limit {
				return false
			}
			hdr, quoted := unquoteGapTok(segTokens[pos+1], escJSON)
			if !quoted || !isSafeGapHeaderArg(hdr) {
				return false
			}
			pos += 2
			continue
		}
		if isSafeGapBoolFlag(tok) {
			pos++
			continue
		}
		return false
	}
	return true
}

func isBenignCommandGap(gap string) bool {
	if len(gap) == 0 || strings.IndexByte(";&|", gap[0]) < 0 {
		return false
	}
	last := gap[len(gap)-1]
	if last != ' ' && last != '\t' {
		return false
	}
	cmd := strings.Trim(gap, ";&| \t")
	return cmd == "echo" || cmd == "printf"
}

// isWordApostropheAt reports whether s[k] is a word apostrophe (a contraction
// or possessive marker, mid-word apostrophe, or trailing possessive after a
// letter). Digit stems without a contraction suffix return false so bare tokens
// such as abc123' remain quoted until handled by header-opener checks.
func isWordApostropheAt(s string, k int) bool {
	if k <= 0 || k >= len(s) || s[k] != '\'' {
		return false
	}
	before, _ := utf8.DecodeLastRuneInString(s[:k])
	if !unicode.IsLetter(before) && !unicode.IsDigit(before) {
		return false
	}
	isWord := func(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }
	if unicode.IsDigit(before) {
		return startsContractionBefore(s, k, 1, isWord)
	}
	if startsContractionBefore(s, k, 1, isWord) || isMidWordApostrophe(s, k) {
		return true
	}
	if before != 's' && before != 'S' {
		return false
	}
	if _, w := utf8.DecodeLastRuneInString(s[:k]); w <= 0 || k-w <= 0 {
		return false
	} else if prev2, _ := utf8.DecodeLastRuneInString(s[:k-w]); !unicode.IsLetter(prev2) {
		return false
	}
	if k+1 == len(s) || strings.IndexByte(",;.!?)]}", s[k+1]) >= 0 {
		return true
	}
	after, _ := utf8.DecodeRuneInString(s[k+1:])
	return isDelimSpace(after)
}

// hasOpenSubshellBefore reports whether an unquoted command-position '('
// opened on the same line strictly before headerStart remains unclosed at
// headerStart. Scanning resumes incrementally across monotonic calls on the
// same line regardless of per-header outerEnd bounds to stay O(N).
func (h *headerQuoteScanner) hasOpenSubshellBefore(headerStart int, escJSON bool) bool {
	if h == nil || headerStart <= 0 {
		return false
	}
	s := h.s
	realLineEnd := func(ls int) int {
		if nl := strings.IndexAny(s[ls:], "\r\n"); nl >= 0 {
			return ls + nl
		}
		return len(s)
	}
	stateIdx := 0
	if escJSON {
		stateIdx = 1
	}
	st := &h.subStates[stateIdx]
	var ls int
	var scopeStart int
	var inInnerJSONStr bool
	if !st.ready || headerStart != st.pos {
		if !st.ready || headerStart < st.pos {
			if nl := strings.LastIndexAny(s[:headerStart], "\r\n"); nl >= 0 {
				ls = nl + 1
			}
		} else if headerStart > st.realLineEnd {
			ls = st.realLineEnd + 1
			if nl := strings.LastIndexAny(s[ls:headerStart], "\r\n"); nl >= 0 {
				ls += nl + 1
			}
		} else {
			ls = st.lineStart
		}
		scopeStart = ls
		if escJSON && h.hasCurLine && h.curLine.k <= headerStart && headerStart <= h.curLine.lineLim {
			if h.curLine.inner == "\\\"" && h.curInnerJSONKind != jsonContainerNone && h.curLine.innerOpenEnd > scopeStart && h.curLine.innerOpenEnd <= headerStart {
				scopeStart = h.curLine.innerOpenEnd
				inInnerJSONStr = true
			} else if h.curLine.outer == "\"" && h.curOuterJSONKind != jsonContainerNone && h.curLine.outerOpenEnd > scopeStart && h.curLine.outerOpenEnd <= headerStart {
				scopeStart = h.curLine.outerOpenEnd
			}
		}
	}
	if !st.ready || headerStart < st.pos {
		st.ready = true
		st.lineStart = ls
		st.realLineEnd = realLineEnd(ls)
		st.scopeStart = scopeStart
		st.inInnerJSONStr = inInnerJSONStr
		st.pos = scopeStart
		st.depth = 0
		st.inSingle = false
		st.inDouble = false
		st.cmdPos = true
	} else if headerStart > st.realLineEnd {
		st.lineStart = ls
		st.realLineEnd = realLineEnd(ls)
		st.scopeStart = scopeStart
		st.inInnerJSONStr = inInnerJSONStr
		st.pos = scopeStart
		st.depth = 0
		st.inSingle = false
		st.inDouble = false
		st.cmdPos = true
	} else if scopeStart > st.scopeStart {
		if scopeStart > st.pos {
			st.pos = scopeStart
		}
		st.scopeStart = scopeStart
		st.inInnerJSONStr = inInnerJSONStr
		st.depth = 0
		st.inSingle = false
		st.inDouble = false
		st.cmdPos = true
	}
	for st.pos < headerStart {
		p := st.pos
		c := s[p]
		if escJSON {
			if c == '"' {
				st.depth = 0
				st.inSingle = false
				st.inDouble = false
				st.cmdPos = true
				st.inInnerJSONStr = false
				st.pos++
				continue
			}
			if c == '\\' {
				bs := 0
				for p+bs < headerStart && s[p+bs] == '\\' {
					bs++
				}
				if p+bs < headerStart && s[p+bs] == '"' {
					if bs%2 == 0 {
						st.depth = 0
						st.inSingle = false
						st.inDouble = false
						st.cmdPos = true
						st.inInnerJSONStr = false
						st.pos = p + bs + 1
						continue
					}
					if st.inInnerJSONStr && bs%4 == 1 {
						st.depth = 0
						st.inSingle = false
						st.inDouble = false
						st.cmdPos = true
						st.inInnerJSONStr = false
						st.pos = p + bs + 1
						continue
					}
					if st.inSingle {
						st.pos = p + bs + 1
						continue
					}
					if (!st.inInnerJSONStr && bs%4 == 1) || (st.inInnerJSONStr && bs%8 == 3) {
						st.inDouble = !st.inDouble
					}
					st.cmdPos = false
					st.pos = p + bs + 1
					continue
				}
				if st.inSingle {
					st.pos = p + bs
					continue
				}
				if bs%2 == 1 && p+bs < headerStart && s[p+bs] == '\'' {
					if !st.inDouble {
						st.inSingle = true
					}
					st.cmdPos = false
					st.pos = p + bs + 1
					continue
				}
				st.cmdPos = false
				st.pos = p + bs
				continue
			}
			if st.inSingle {
				if c == '\'' {
					st.inSingle = false
				}
				st.pos++
				continue
			}
		} else {
			if st.inSingle {
				if c == '\'' {
					st.inSingle = false
				}
				st.pos++
				continue
			}
			if st.inDouble {
				if c == '\\' && p+1 < headerStart {
					st.pos += 2
					continue
				}
				if c == '"' {
					st.inDouble = false
				}
				st.pos++
				continue
			}
			if c == '\\' {
				st.cmdPos = false
				if p+1 < headerStart {
					st.pos += 2
				} else {
					st.pos++
				}
				continue
			}
			if c == '"' {
				st.inDouble = true
				st.cmdPos = false
				st.pos++
				continue
			}
		}
		if st.inDouble {
			st.pos++
			continue
		}
		switch c {
		case '\'':
			if !isWordApostropheAt(s, p) {
				st.inSingle = true
			}
			st.cmdPos = false
		case '(':
			st.depth++
		case ')':
			if st.depth > 0 {
				st.depth--
			}
			st.cmdPos = false
		case ';', '|', '&':
			st.cmdPos = true
		case ' ', '\t':
		case '$':
			if p+1 < len(s) && s[p+1] == '(' {
				st.cmdPos = true
			} else {
				st.cmdPos = false
			}
		default:
			st.cmdPos = false
		}
		st.pos++
	}
	return st.depth > 0
}

// parenClosesCleanlyAt reports whether h.s[parenPos] == ')' closes a genuine
// unquoted subshell opened before headerStart on the same line and is followed
// after optional horizontal whitespace by end-of-line, a shell delimiter, or
// a JSON terminator when unescaped in JSON.
func (h *headerQuoteScanner) parenClosesCleanlyAt(headerStart, parenPos, lineEnd int, inJSON, jsonEscaped bool) bool {
	escJSON := inJSON || jsonEscaped
	if !h.hasOpenSubshellBefore(headerStart, escJSON) {
		return false
	}
	return parenCloseSuffixOK(h.s, parenPos, lineEnd, inJSON, jsonEscaped)
}

func parenCloseSuffixOK(s string, parenPos, lineEnd int, inJSON, jsonEscaped bool) bool {
	escJSON := inJSON || jsonEscaped
	p := skipHorizDelimSpace(s, parenPos+1)
	for p < lineEnd && s[p] == ')' {
		p = skipHorizDelimSpace(s, p+1)
	}
	if p >= lineEnd {
		return true
	}
	switch s[p] {
	case ';', '|', '#':
		return true
	case '&':
		return p+1 < lineEnd && s[p+1] == '&'
	case '"':
		return escJSON && jsonCloseBoundaryOK(s, p+1, lineEnd, false, true)
	case '\\':
		return jsonEscaped && p+1 < lineEnd && s[p+1] == '"' && jsonCloseBoundaryOK(s, p+2, lineEnd, true, true)
	case ',', '}', ']':
		return inJSON && !jsonEscaped && jsonCloseBoundaryOK(s, p, lineEnd, false, true)
	default:
		return false
	}
}

func substClosesAtParen(substTop *intListNode, parenPos int) bool {
	for node := substTop; node != nil; node = node.prev {
		if node.val == parenPos+1 {
			return true
		}
	}
	return false
}

// apostropheTailStopsOnLine reports whether the remainder of the header line
// after an unclosed word apostrophe at h.s[k] terminates on a safe boundary
// (end of line, shell terminator, comment, closed subshell, balanced
// -H/--header flag opening a recognized auth header, auth header, or URL)
// rather than running into bare token text that may carry a credential.
func (h *headerQuoteScanner) apostropheTailStopsOnLine(headerStart, k, lineEnd int, inJSON, jsonEscaped bool) bool {
	s := h.s
	escJSON := inJSON || jsonEscaped
	afterEnd := k + 1
	for afterEnd < lineEnd {
		r, w := utf8.DecodeRuneInString(s[afterEnd:lineEnd])
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			break
		}
		afterEnd += w
	}
	if afterEnd < lineEnd {
		c := s[afterEnd]
		if c == ')' {
			return h.parenClosesCleanlyAt(headerStart, afterEnd, lineEnd, inJSON, jsonEscaped)
		}
		if strings.IndexByte(";&|<>", c) >= 0 || (inJSON && (c == ',' || c == '}' || c == ']')) {
			return true
		}
	}
	if afterEnd < lineEnd {
		r, _ := utf8.DecodeRuneInString(s[afterEnd:lineEnd])
		if !isDelimSpace(r) {
			return false
		}
	}
	rest := s[afterEnd:lineEnd]
	for len(rest) > 0 {
		r, w := utf8.DecodeRuneInString(rest)
		if !isDelimSpace(r) {
			break
		}
		rest = rest[w:]
	}
	if len(rest) == 0 {
		return true
	}
	restPos := lineEnd - len(rest)
	if rest[0] == ')' {
		return h.parenClosesCleanlyAt(headerStart, restPos, lineEnd, inJSON, jsonEscaped)
	}
	if strings.IndexByte(";|&<>#", rest[0]) >= 0 || (inJSON && (rest[0] == ',' || rest[0] == '}' || rest[0] == ']')) {
		return true
	}
	if rest[0] == '-' {
		if h == nil {
			return false
		}
		var hdrStart int
		switch {
		case strings.HasPrefix(rest, "-H="):
			hdrStart = len("-H=")
		case strings.HasPrefix(rest, "--header="):
			hdrStart = len("--header=")
		case strings.HasPrefix(rest, "-H"):
			hdrStart = skipHorizDelimSpace(rest, len("-H"))
			if hdrStart == len("-H") {
				return false
			}
		case strings.HasPrefix(rest, "--header"):
			hdrStart = skipHorizDelimSpace(rest, len("--header"))
			if hdrStart == len("--header") {
				return false
			}
		default:
			return false
		}
		gapEnd := restPos + hdrStart
		authOff := hdrStart
		if escJSON && strings.HasPrefix(rest[authOff:], `\"`) {
			authOff += 2
		} else if authOff < len(rest) && rest[authOff] == '"' {
			authOff++
		}
		if authOff >= len(rest) || !startsAuthHeaderPrefix(rest[authOff:]) {
			return false
		}
		return h.isShellBalancedFrom(gapEnd, lineEnd, escJSON)
	}
	if authHeaderKeys.mayStartAt(rest, 0) {
		if loc := authHeaderPrefixAtRe.FindStringSubmatchIndex(rest); loc != nil {
			return !isCookieHeaderKey(rest[loc[2]:loc[3]]) || !cookieValueIsProse(rest, loc[1])
		}
	}
	if strings.HasPrefix(rest, "http://") || strings.HasPrefix(rest, "https://") ||
		strings.HasPrefix(rest, `"http://`) || strings.HasPrefix(rest, `"https://`) ||
		(escJSON && (strings.HasPrefix(rest, `\"http://`) || strings.HasPrefix(rest, `\"https://`))) {
		tokLen := 0
		for tokLen < len(rest) {
			r, w := utf8.DecodeRuneInString(rest[tokLen:])
			if isDelimSpace(r) {
				break
			}
			tokLen += w
		}
		urlTok := rest[:tokLen]
		if rest[0] == 'h' || rest[0] == 'H' {
			cutChars := `";|&<>`
			if inJSON {
				cutChars = `";|&<>,}]`
			}
			if cut := strings.IndexAny(urlTok, cutChars); cut > 0 {
				urlTok = urlTok[:cut]
				tokLen = cut
			}
		}
		unq, _ := unquoteGapTok(urlTok, escJSON)
		if strings.ContainsAny(unq, "?#@=") || !isSafeGapURL(unq) {
			return false
		}
		return h.isShellBalancedFrom(restPos+tokLen, lineEnd, escJSON)
	}
	return len(rest) >= 3 && rest[0] >= '0' && rest[0] <= '9' && (rest[1] == '>' || rest[1] == '<')
}

// isShortLowerProseTail reports whether s[k+1:lineEnd] after a word-apostrophe
// (such as "Bob's demo key") ends the line in a short sequence of at least two
// plain lowercase ASCII words separated by horizontal spaces. Keeping the bound
// constant preserves O(N) scanning while distinguishing benign prose tails from
// single opaque or credential-shaped tokens.
func isShortLowerProseTail(s string, k, lineEnd int) bool {
	p := k + 1
	for p < lineEnd && s[p] >= 'a' && s[p] <= 'z' {
		p++
	}
	if p >= lineEnd || lineEnd-p > 64 {
		return false
	}
	words := 0
	for p < lineEnd {
		sp := skipHorizDelimSpace(s[:lineEnd], p)
		if sp == p || sp >= lineEnd {
			return false
		}
		wStart := sp
		for sp < lineEnd && s[sp] >= 'a' && s[sp] <= 'z' {
			sp++
		}
		if sp-wStart < 2 {
			return false
		}
		words++
		if words > 6 {
			return false
		}
		p = sp
	}
	return words >= 2
}

// isTrailingDigitApostropheAt reports whether s[k] is a single quote directly
// after a digit at the end of a token before horizontal space or punctuation.
func isTrailingDigitApostropheAt(s string, k int) bool {
	if k <= 0 || k >= len(s) || s[k] != '\'' {
		return false
	}
	before, _ := utf8.DecodeLastRuneInString(s[:k])
	if !unicode.IsDigit(before) {
		return false
	}
	if k+1 == len(s) || strings.IndexByte(",;.!?)]}", s[k+1]) >= 0 {
		return true
	}
	after, _ := utf8.DecodeRuneInString(s[k+1:])
	return isDelimSpace(after)
}

// isAuthHeaderOpenerGap reports whether gap sits between a preceding token and
// an immediately following auth header opener (optionally preceded by shell
// separators and a -H/--header flag).
func isAuthHeaderOpenerGap(gap string) bool {
	if len(gap) == 0 {
		return false
	}
	r, _ := utf8.DecodeRuneInString(gap)
	if !isDelimSpace(r) && strings.IndexByte(";&|<>()", gap[0]) < 0 {
		return false
	}
	rem := gap[skipHorizDelimSpace(gap, 0):]
	for len(rem) > 0 && strings.IndexByte(";&|()", rem[0]) >= 0 {
		rem = rem[1:]
		rem = rem[skipHorizDelimSpace(rem, 0):]
	}
	for _, flag := range []string{"--header", "-H"} {
		if strings.HasPrefix(rem, flag) && len(rem) > len(flag) {
			after := rem[len(flag):]
			if n := skipHorizDelimSpace(after, 0); n > 0 {
				rem = after[n:]
				break
			} else if after[0] == '=' {
				rem = after[1:]
				break
			}
		}
	}
	if !authHeaderKeys.mayStartAt(rem, 0) {
		return false
	}
	loc := authHeaderPrefixAtRe.FindStringIndex(rem)
	return loc != nil && loc[0] == 0 && loc[1] == len(rem)
}

// isJSONEscapedSpaceBefore reports whether s[:pos] ends with a backslash-escaped
// whitespace sequence (\n, \r, or \t, single- or double-escaped in JSON text).
func isJSONEscapedSpaceBefore(s string, pos int) bool {
	if pos < 2 {
		return false
	}
	switch s[pos-1] {
	case 'n', 'r', 't':
	default:
		return false
	}
	bs := 0
	for p := pos - 2; p >= 0 && s[p] == '\\'; p-- {
		bs++
	}
	return bs >= 1
}

// spanEndsWithAuthHeaderPrefix reports whether s[spanStart:spanEnd] terminates
// with a recognized auth header name and colon delimiter on a clean boundary.
func spanEndsWithAuthHeaderPrefix(s string, spanStart, spanEnd int) bool {
	if spanStart < 0 || spanEnd > len(s) || spanStart >= spanEnd {
		return false
	}
	span := s[spanStart:spanEnd]
	j := len(span) - 1
	for j >= 0 && (span[j] == ' ' || span[j] == '\t') {
		j--
	}
	if j < 0 || span[j] != ':' {
		return false
	}
	nameStart := j - 1
	for nameStart >= 0 && (span[nameStart] == ' ' || span[nameStart] == '\t') {
		nameStart--
	}
	for nameStart >= 0 {
		c := span[nameStart]
		if c != '-' && c != '_' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9') {
			break
		}
		if isJSONEscapedSpaceBefore(s, spanStart+nameStart+1) {
			break
		}
		nameStart--
	}
	idx := nameStart + 1
	if idx >= len(span) || !authHeaderKeys.mayStartAt(span, idx) {
		return false
	}
	loc := authHeaderPrefixAtRe.FindStringIndex(span[idx:])
	if loc == nil || loc[0] != 0 || idx+loc[1] != len(span) {
		return false
	}
	return wordBoundaryAt(s, spanStart+idx) || isJSONEscapedSpaceBefore(s, spanStart+idx)
}

// spanEndsWithNestedHeaderPrefix reports whether s[spanStart:spanEnd] ends
// with a recognized auth header prefix or a sensitive key-colon prefix on a
// clean boundary.
func spanEndsWithNestedHeaderPrefix(s string, spanStart, spanEnd int) bool {
	if spanEndsWithAuthHeaderPrefix(s, spanStart, spanEnd) {
		return true
	}
	if spanStart < 0 || spanEnd > len(s) || spanStart >= spanEnd {
		return false
	}
	span := s[spanStart:spanEnd]
	j := len(span) - 1
	for j >= 0 && (span[j] == ' ' || span[j] == '\t') {
		j--
	}
	if j < 0 || span[j] != ':' {
		return false
	}
	nameEnd := j - 1
	for nameEnd >= 0 && (span[nameEnd] == ' ' || span[nameEnd] == '\t') {
		nameEnd--
	}
	if nameEnd < 0 {
		return false
	}
	nameStart := nameEnd
	for nameStart >= 0 {
		c := span[nameStart]
		if c != '-' && c != '_' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9') {
			break
		}
		if isJSONEscapedSpaceBefore(s, spanStart+nameStart+1) {
			break
		}
		nameStart--
	}
	idx := nameStart + 1
	if idx > nameEnd {
		return false
	}
	if !wordBoundaryAt(s, spanStart+idx) && !isJSONEscapedSpaceBefore(s, spanStart+idx) {
		return false
	}
	return isSensitiveKey(span[idx : nameEnd+1])
}

func startsAuthHeaderPrefix(sub string) bool {
	return authHeaderKeys.mayStartAt(sub, 0) && authHeaderPrefixAtRe.FindStringIndex(sub) != nil
}

func (h *headerQuoteScanner) isCandidateHeaderApostrophe(k, closePos, lineEnd int, inJSON bool) bool {
	s := h.s
	wordApostrophe := isWordApostropheAt(s, k)
	if !wordApostrophe && !isTrailingDigitApostropheAt(s, k) {
		return false
	}
	tokEnd := k + 1
	for tokEnd < closePos-1 {
		r, w := utf8.DecodeRuneInString(s[tokEnd : closePos-1])
		if isDelimSpace(r) || strings.IndexByte(";&|<>()", s[tokEnd]) >= 0 {
			break
		}
		tokEnd += w
	}
	if tokEnd >= closePos-1 {
		return false
	}
	gap := s[tokEnd : closePos-1]
	if hasUnquotedShellComment(gap) {
		return false
	}
	if closePos < 2 || (s[closePos-2] != ' ' && s[closePos-2] != '\t' && strings.IndexByte("=:;&|(", s[closePos-2]) < 0 && !opensPrefixedString(s, closePos-1)) {
		return false
	}
	if startsAuthHeaderPrefix(s[closePos:lineEnd]) {
		return wordApostrophe && h.isHeaderFlagGap(k, gap, lineEnd, inJSON, isTrailingApostrophe(s, k))
	}
	authGap := isAuthHeaderOpenerGap(gap)
	if !wordApostrophe && !authGap {
		return false
	}
	if !authGap && !isBenignCommandGap(gap) {
		return false
	}
	if closePos < lineEnd && s[closePos] == '#' {
		return false
	}
	nextClose := quoteCloseScan(s[:lineEnd], closePos, "'", "")
	if nextClose < 0 {
		return false
	}
	inner := s[closePos : nextClose-1]
	if authGap {
		if hasUnquotedShellComment(inner) {
			return false
		}
		return h.isShellBalancedFrom(tokEnd, lineEnd, inJSON)
	}
	if len(inner) == 0 || hasUnquotedShellComment(inner) || strings.ContainsAny(inner, "\"\\") {
		return false
	}
	firstRune, _ := utf8.DecodeRuneInString(inner)
	if isDelimSpace(firstRune) || strings.IndexByte(";&|<>()", inner[0]) >= 0 {
		return false
	}
	if nextClose < lineEnd {
		r, _ := utf8.DecodeRuneInString(s[nextClose:lineEnd])
		if !isDelimSpace(r) && strings.IndexByte(";&|<>(),}]", s[nextClose]) < 0 {
			return false
		}
	}
	return h.isShellBalancedFrom(tokEnd, lineEnd, inJSON)
}

// unescapeBackslashControls decodes the standard backslash escapes in s. A
// run of backslashes, which each level of nested escaping doubles, is decoded
// in one step as the escape its last backslash starts, so that the depth of
// the nesting does not set the number of passes.
func unescapeBackslashControls(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	ansicEnd := -1
	noMoreANSIC := false
	for i := 0; i < len(s); i++ {
		if i == ansicEnd {
			ansicEnd = -1
		}
		if s[i] != '\\' {
			if ansicEnd < 0 && !noMoreANSIC && s[i] == '$' && i+1 < len(s) && s[i+1] == '\'' && (i == 0 || s[i-1] != '\\') {
				if end := scanANSICQuotedClose(s, i+2, false); end >= 0 {
					ansicEnd = end
				} else {
					noMoreANSIC = true
				}
			}
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
		case '/', '"':
			b.WriteByte(c)
		case '\'':
			if ansicEnd >= 0 && j < ansicEnd {
				if (j-i)%2 == 1 {
					b.WriteString(`\'`)
				} else {
					b.WriteString(`\\'`)
				}
			} else {
				b.WriteByte(c)
			}
		default:
			if (c == '$' || c == '`') && (j-i)%2 == 0 {
				b.WriteString(`\\`)
				b.WriteByte(c)
			} else {
				if ansicEnd < 0 && !noMoreANSIC && c == '$' && j+1 < len(s) && s[j+1] == '\'' && (j-i)%2 == 0 {
					if end := scanANSICQuotedClose(s, j+2, false); end >= 0 {
						ansicEnd = end
					} else {
						noMoreANSIC = true
					}
				}
				b.WriteByte('\\')
				b.WriteByte(c)
			}
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

// scrubTextPatterns redacts Authorization, API-key, Cookie and Set-Cookie
// headers, bearer and basic credentials, and the values of sensitive
// key-value pairs in s.
//
// Its passes read quotes, shell assignments, line splices and prose more
// closely than the baseline pass does (see baselineCuts), so that they redact
// more, but such a reading may also take a value to end elsewhere than the
// baseline pass does. Whatever they read, every byte of s that the baseline
// pass would replace with a placeholder is left out of the result: the
// passes record which bytes of the text they write are copies of s (see
// passRecorder), and any copy of a byte that the baseline pass cuts is
// redacted after them (see keepBaselineCuts).
//
// The passes are written so that a scrub of their own result changes
// nothing. A placeholder that keepBaselineCuts writes between bytes that they
// kept can make them read the result differently from s, though, so when it
// writes one the scrub is repeated on its own result, at most maxScrubRounds
// times in all, until it changes nothing. Every round only adds redaction,
// and a fixed number of rounds keeps the work linear.
func scrubTextPatterns(s string) string {
	out, floored := scrubTextRound(s)
	for round := 1; round < maxScrubRounds && floored; round++ {
		prev := out
		if out, floored = scrubTextRound(prev); out == prev {
			break
		}
	}
	return out
}

// maxScrubRounds bounds the rounds of scrubTextPatterns.
const maxScrubRounds = 3

// scrubTextRound runs the passes of scrubTextPatterns once over s and
// redacts what the baseline pass cuts in s. It also reports whether that
// redacted anything that the passes kept.
func scrubTextRound(s string) (string, bool) {
	if !mayMatchTextPatterns(s) {
		return s, false
	}
	var rec [5]passRecorder
	out := redactAuthHeadersRecorded(s, &rec[0])
	var bearerQuotes openQuotes
	var bearerPairs sensitivePairSpans
	out = replaceMatchesFunc(out, findBearerToken, func(s string, loc []int) (string, int) {
		return redactBearerToken(s, loc, &bearerQuotes, &bearerPairs)
	}, &rec[1])
	out = replaceMatchesFunc(out, findBasicToken, redactBasicCredential, &rec[2])
	var jsonQuotes openQuotes
	out = replaceMatchesFunc(out, findJSONKeyValToken, func(s string, loc []int) (string, int) {
		return redactJSONKeyValue(s, loc, &jsonQuotes)
	}, &rec[3])
	var brackets openBrackets
	var quotes openQuotes
	finder := keyValFinder{whole: out}
	out = replaceMatchesFunc(out, finder.find, func(s string, loc []int) (string, int) {
		return redactTextKeyValue(s, loc, &brackets, &quotes)
	}, &rec[4])
	return keepBaselineCuts(out, len(s), rec[:], baselineCuts(s))
}

// copySpan records that the n bytes that a pass wrote at offset out of its
// result are a copy of the n bytes at offset in of the text it read.
type copySpan struct{ out, in, n int }

// textRange is the range [start, end) of a text.
type textRange struct{ start, end int }

// passRecorder records, for a pass that copies the text it reads and replaces
// parts of it, which bytes of its result are copies of that text (kept, in
// order), and which ranges of that text it replaced (cut). Methods on a nil
// *passRecorder do nothing.
type passRecorder struct {
	kept    []copySpan
	cut     []textRange
	written int // the length of the result so far
}

// reset forgets what r has recorded.
func (r *passRecorder) reset() {
	if r != nil {
		*r = passRecorder{kept: r.kept[:0], cut: r.cut[:0]}
	}
}

// copied records that the pass wrote a copy of the n bytes at offset at of
// the text it read.
func (r *passRecorder) copied(at, n int) {
	if r == nil || n <= 0 {
		return
	}
	if k := len(r.kept) - 1; k >= 0 && r.kept[k].out+r.kept[k].n == r.written && r.kept[k].in+r.kept[k].n == at {
		r.kept[k].n += n
	} else {
		r.kept = append(r.kept, copySpan{r.written, at, n})
	}
	r.written += n
}

// replaced records that the pass wrote repl in place of in[start:end]. The
// bytes that repl shares with in[start:end] at its start and at its end (see
// sharedEnds) are copies; the bytes of in[start:end] between them are cut.
func (r *passRecorder) replaced(in string, start, end int, repl string) {
	if r == nil {
		return
	}
	if end < start {
		end = start
	}
	pre, suf := sharedEnds(in[start:end], repl)
	r.copied(start, pre)
	r.written += len(repl) - pre - suf
	if start+pre < end-suf {
		r.cut = append(r.cut, textRange{start + pre, end - suf})
	}
	r.copied(end-suf, suf)
}

// placeholderStart is the text that every redaction placeholder starts with.
const placeholderStart = "[REDACTED"

// sharedEnds returns the lengths of the longest prefix and the longest suffix
// that m and repl share, which hold whole runes, do not overlap, and leave out
// any redaction placeholder in repl, so that the placeholder is never read as
// a copy of the text it replaced.
func sharedEnds(m, repl string) (pre, suf int) {
	if m == repl {
		return len(m), 0
	}
	limit := min(len(m), len(repl))
	if i := strings.Index(repl, placeholderStart); i >= 0 {
		limit = min(limit, i)
	}
	for pre < limit && m[pre] == repl[pre] {
		pre++
	}
	for pre > 0 && (pre < len(m) && !utf8.RuneStart(m[pre]) || pre < len(repl) && !utf8.RuneStart(repl[pre])) {
		pre--
	}
	limit = min(len(m), len(repl)) - pre
	if i := strings.LastIndex(repl, placeholderStart); i >= 0 {
		closing := strings.IndexByte(repl[i:], ']')
		if closing < 0 {
			closing = len(repl) - i - 1
		}
		limit = min(limit, len(repl)-(i+closing+1))
	}
	for suf < limit && m[len(m)-1-suf] == repl[len(repl)-1-suf] {
		suf++
	}
	for suf > 0 && (!utf8.RuneStart(m[len(m)-suf]) || !utf8.RuneStart(repl[len(repl)-suf])) {
		suf--
	}
	return pre, suf
}

// composeSpans returns which bytes of the result of a second pass are copies
// of the text that a first pass read, given the copies that each pass made of
// the text it read (first, then second, whose input is the result of the
// first pass).
func composeSpans(first, second []copySpan) []copySpan {
	var out []copySpan
	for _, c := range second {
		// The first span of first that ends after c starts.
		j, _ := slices.BinarySearchFunc(first, c.in, func(f copySpan, at int) int {
			return cmp.Compare(f.out+f.n, at+1)
		})
		for ; j < len(first) && first[j].out < c.in+c.n; j++ {
			f := first[j]
			lo, hi := max(c.in, f.out), min(c.in+c.n, f.out+f.n)
			if lo < hi {
				out = append(out, copySpan{c.out + lo - c.in, f.in + lo - f.out, hi - lo})
			}
		}
	}
	return out
}

// mapRanges returns the ranges of the text that a pass read of which the
// bytes in ranges, ranges of its result, are copies (see copySpan).
func mapRanges(ranges []textRange, kept []copySpan) []textRange {
	var out []textRange
	for _, r := range ranges {
		j, _ := slices.BinarySearchFunc(kept, r.start, func(k copySpan, at int) int {
			return cmp.Compare(k.out+k.n, at+1)
		})
		for ; j < len(kept) && kept[j].out < r.end; j++ {
			k := kept[j]
			lo, hi := max(r.start, k.out), min(r.end, k.out+k.n)
			if lo < hi {
				out = append(out, textRange{k.in + lo - k.out, k.in + hi - k.out})
			}
		}
	}
	return out
}

// isCutPadding reports whether c, at an end of a range that a pass cut, is
// whitespace, a quote, a bracket or punctuation around a value rather than a
// byte of the value. The baseline pass writes its own delimiters, quotes and
// spaces around a placeholder, so these bytes at the ends of what it cut are
// not what it redacted.
func isCutPadding(c byte) bool {
	return c <= ' ' || c == 0x7f || strings.IndexByte(`"'\:=,;.{}[]()`, c) >= 0
}

// trimCut returns r without the padding bytes of s (see isCutPadding) at its
// ends, leaving any placeholder in cutPlaceholders whole.
func trimCut(s string, r textRange) textRange {
	for r.start < r.end && isCutPadding(s[r.start]) {
		if _, ok := placeholderStartingAt(s, r.start); ok {
			break
		}
		r.start++
	}
	for r.end > r.start && isCutPadding(s[r.end-1]) {
		if _, ok := placeholderEndingAt(s, r.end); ok {
			break
		}
		r.end--
	}
	return r
}

// cutPlaceholders holds the placeholders that a scrub of text may leave in it.
var cutPlaceholders = []string{redactedValue, redactedSensitiveTextPlaceholder, redactedURIPlaceholder, url.QueryEscape(redactedValue)}

// refineCuts returns the ranges, in order, of the bytes in cuts, ranges of s,
// without the padding at their ends (see trimCut) and without the
// placeholders that s already holds.
//
// A cut that starts with a placeholder that ends the value that starts there
// (see placeholderEnds) is left out whole: the value that the baseline pass
// read there was redacted by an earlier scrub, which ended it at the
// placeholder and kept the pairs that follow it, so that a scrub of its own
// result changes nothing. Of any other cut that holds a placeholder, only the
// bytes beside the placeholders are returned (see appendWithoutPlaceholders):
// text can hold a placeholder without any scrub having written it, as in
// "password": " [REDACTED] x", where it is only a part of a credential that
// it does not cover whole, and the rest of the credential is redacted with it
// and is not kept for the placeholder.
func refineCuts(s string, cuts []textRange) []textRange {
	slices.SortFunc(cuts, func(a, b textRange) int { return cmp.Compare(a.start, b.start) })
	var out []textRange
	var quotes openQuotes
	for _, c := range cuts {
		t := trimCut(s, c)
		if t.start >= t.end {
			continue
		}
		quoted := quotes.before(s, t.start)
		if strings.HasPrefix(s[t.start:t.end], redactedValue) &&
			(placeholderEnds(s, t.start+len(redactedValue), quoted, quotes.innerAt(s, t.start), quotes.subSingleAt(s, t.start), quotes.subEscDoubleAt(s, t.start)) || closesOuterQuote(s, t.start+len(redactedValue), c.end, 0, quoted)) {
			continue
		}
		out = appendWithoutPlaceholders(out, s, t)
	}
	return out
}

// appendWithoutPlaceholders appends to out the ranges of t, a range of s, that
// the placeholders that s holds in t leave, without the padding at their ends
// (see trimCut). A range that only spaces or tabs separate from a placeholder
// takes in the placeholder, so that the placeholder and the bytes of the
// credential beside it are replaced as one.
func appendWithoutPlaceholders(out []textRange, s string, t textRange) []textRange {
	holes := placeholderRanges(s, t)
	if len(holes) == 0 {
		return append(out, t)
	}
	at := t.start
	for i := 0; i <= len(holes); i++ {
		end := t.end
		if i < len(holes) {
			end = holes[i].start
		}
		if r := trimCut(s, textRange{at, end}); r.start < r.end {
			if i > 0 && isBlank(s[holes[i-1].end:r.start]) {
				r.start = holes[i-1].start
			}
			if i < len(holes) && isBlank(s[r.end:holes[i].start]) {
				r.end = holes[i].end
			}
			if last := len(out) - 1; last >= 0 && out[last].end >= r.start {
				out[last].end = max(out[last].end, r.end)
			} else {
				out = append(out, r)
			}
		}
		if i < len(holes) {
			at = holes[i].end
		}
	}
	return out
}

// placeholderRanges returns the ranges of the placeholders of cutPlaceholders
// that s holds in r, in order and cut off at the ends of r, with those that
// overlap or touch merged.
func placeholderRanges(s string, r textRange) []textRange {
	var holes []textRange
	for _, p := range cutPlaceholders {
		hi := min(len(s), r.end+len(p)-1)
		// Any placeholder that starts in this window overlaps r.
		for i := max(0, r.start-len(p)+1); i < hi; {
			j := strings.Index(s[i:hi], p)
			if j < 0 {
				break
			}
			h := textRange{i + j, i + j + len(p)}
			holes = append(holes, textRange{max(h.start, r.start), min(h.end, r.end)})
			i = h.start + 1
		}
	}
	slices.SortFunc(holes, func(a, b textRange) int { return cmp.Compare(a.start, b.start) })
	merged := holes[:0]
	for _, h := range holes {
		if last := len(merged) - 1; last >= 0 && merged[last].end >= h.start {
			merged[last].end = max(merged[last].end, h.end)
		} else {
			merged = append(merged, h)
		}
	}
	return merged
}

// isBlank reports whether s holds only spaces and tabs.
func isBlank(s string) bool {
	return strings.Trim(s, " \t") == ""
}

// keepBaselineCuts returns out, the result of the passes of scrubTextPatterns
// on a text of n bytes, with every copy that it holds of a byte in cuts (see
// baselineCuts), ranges of that text, replaced with the redaction
// placeholder. recs are what the passes recorded, in the order in which they
// ran. A run of such copies is replaced as a whole, without the padding at
// its ends (see isCutPadding). It also reports whether it replaced any.
func keepBaselineCuts(out string, n int, recs []passRecorder, cuts []textRange) (string, bool) {
	if len(cuts) == 0 {
		return out, false
	}
	kept := []copySpan{{0, 0, n}}
	for i := range recs {
		kept = composeSpans(kept, recs[i].kept)
	}
	var redact []textRange
	for _, k := range kept {
		j, _ := slices.BinarySearchFunc(cuts, k.in, func(c textRange, at int) int {
			return cmp.Compare(c.end, at+1)
		})
		for ; j < len(cuts) && cuts[j].start < k.in+k.n; j++ {
			lo, hi := max(k.in, cuts[j].start), min(k.in+k.n, cuts[j].end)
			if lo >= hi {
				continue
			}
			r := textRange{k.out + lo - k.in, k.out + hi - k.in}
			if last := len(redact) - 1; last >= 0 && redact[last].end == r.start {
				redact[last].end = r.end
			} else {
				redact = append(redact, r)
			}
		}
	}
	var b strings.Builder
	at := 0
	for _, r := range redact {
		t := trimCut(out, r)
		if t.start >= t.end {
			continue
		}
		// A placeholder that the copies run up to, past padding that the
		// baseline pass cut too, is replaced together with them, so that a
		// value that both cut is written as one placeholder.
		if p, ok := placeholderEndingAt(out, r.start); ok {
			t.start = p.start
		}
		if p, ok := placeholderStartingAt(out, r.end); ok {
			t.end = p.end
		}
		if at > 0 && t.start <= at {
			// The placeholder written last already covers the start.
			at = max(at, t.end)
			continue
		}
		b.WriteString(out[at:t.start])
		b.WriteString(redactedValue)
		at = t.end
	}
	if at == 0 {
		return out, false
	}
	b.WriteString(out[at:])
	return b.String(), true
}

// placeholderEndingAt returns the range of the placeholder in s that ends at
// offset i, if there is one.
func placeholderEndingAt(s string, i int) (textRange, bool) {
	for _, p := range cutPlaceholders {
		if strings.HasSuffix(s[:i], p) {
			return textRange{i - len(p), i}, true
		}
	}
	return textRange{}, false
}

// placeholderStartingAt returns the range of the placeholder in s that starts
// at offset i, if there is one.
func placeholderStartingAt(s string, i int) (textRange, bool) {
	for _, p := range cutPlaceholders {
		if strings.HasPrefix(s[i:], p) {
			return textRange{i, i + len(p)}, true
		}
	}
	return textRange{}, false
}

// The baseline pass is the plain scrub of key-value text that the passes of
// scrubTextPatterns refine. It reads quotes only as its regular expressions
// do and reads no shell syntax or prose, and its keys are those of
// baselineKeyNames. scrubTextPatterns redacts at least what it redacts.

// baselineKeyNames holds the key names that the baseline pass reads as
// sensitive, in order.
var baselineKeyNames = []string{
	"access_token", "api_key", "api_key_secret", "api_key_string", "auth_code",
	"authorization", "aws_secret_access_key", "aws_session_token", "bearer",
	"client_secret", "code_verifier", "credential", "credentials",
	"google_access_id", "id_token", "passphrase", "password", "private_key",
	"proxy_authorization", "refresh_token", "secret", "secret_access_key",
	"session_token", "sig", "signature", "token", "x_amz_credential",
	"x_amz_signature", "x_api_key", "x_auth_token", "x_goog_api_key",
	"x_goog_credential", "x_goog_security_token", "x_goog_signature",
}

// baselineKeySet holds the names in baselineKeyNames.
var baselineKeySet = func() map[string]struct{} {
	set := make(map[string]struct{}, len(baselineKeyNames))
	for _, k := range baselineKeyNames {
		set[k] = struct{}{}
	}
	return set
}()

// baselineFixedKeyRegex returns an alternation of baselineKeyNames, in which
// each "_" may also be written as "-" or left out.
func baselineFixedKeyRegex() string {
	parts := make([]string, 0, len(baselineKeyNames))
	for _, k := range baselineKeyNames {
		parts = append(parts, strings.ReplaceAll(regexp.QuoteMeta(k), "_", "[_-]?"))
	}
	return strings.Join(parts, "|")
}

// baselineKeyRegex returns an alternation of the baseline key names and a
// "temp:" key name, which runs on up to a byte in tempStop.
func baselineKeyRegex(tempStop string) string {
	return `temp:[^` + tempStop + `]+|` + baselineFixedKeyRegex()
}

// baselineQuotedValueRegex returns the pattern for a quoted value that the
// baseline pass reads: a double- or single-quoted string, in which a
// backslash escapes any byte but a line break, or a string quoted with
// escaped quotes (see escapedQuotedRegex).
func baselineQuotedValueRegex() string {
	return `"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|` + escapedQuotedRegex(`"`) + `|` + escapedQuotedRegex(`'`)
}

// baselineValueRegex returns the pattern for the value after a key that the
// baseline pass reads: a quoted value or a run of bytes other than those in
// bareStop.
func baselineValueRegex(bareStop string) string {
	return baselineQuotedValueRegex() + `|[^` + bareStop + `]+`
}

// The patterns of the baseline pass. It reads each of them, case-insensitively,
// where a word boundary (\b) is, but the JSON pattern, which opens with a
// quote, at any offset. They are compiled anchored to the start of the text
// that they are matched against, and findAtKeys and findAtQuotedKeys try
// them only where the key names that they start with may start (see
// keyNameSet): a regular expression search for them reads every byte of the
// text with each of their many alternatives.
var (
	baselineAuthHeaderPrefixPattern = `((?:proxy[-_])?authorization|x[-_]api[-_]key|api[-_]key)[ \t]*:[ \t]*`
	baselineAuthHeaderPattern       = baselineAuthHeaderPrefixPattern + `[^\r\n]*`
	baselineBearerTokenPattern      = `(bearer)\s+("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\[[^\]\r\n\s]+\][^\s,;)"']*|[^\s,;)"'\]]+)`
	baselineBasicTokenPattern       = `(basic)\s+([A-Za-z0-9+/]+={0,2})`
	baselineJSONKeyValTokenPattern  = `"[\s\p{Z}]*(` + baselineKeyRegex(`"'\s\r\n,}`) + `)[\s\p{Z}]*"\s*:\s*(` + baselineValueRegex(`\s,}`) + `)`
	baselineTextKeyValTokenPattern  = `(` + baselineKeyRegex(`=:"'\s\r\n,}`) + `)((?:\\?["'])?)[\s[:cntrl:]]*([=:])[\s[:cntrl:]]*(` + baselineValueRegex(`\s,;&`) + `)`
)

// The key names that the patterns of the baseline pass start with, after
// the opening quote and spaces of the JSON pattern, and those that the
// patterns of the passes of scrubTextPatterns start with: authHeaderRe,
// textKeyNameRe, after its word boundary or "_", textKeyNameCamelAtRe, and
// jsonKeyValTokenRe, after its opening quote, spaces and separators. The
// bearer and basic patterns of both passes start with bearerKeys and
// basicKeys.
var (
	baselineAuthHeaderKeys = newKeyNameSet("proxy_authorization", "authorization", "x_api_key", "api_key")
	bearerKeys             = newKeyNameSet("bearer")
	basicKeys              = newKeyNameSet("basic")
	baselineKeys           = newKeyNameSet(append([]string{session.KeyPrefixTemp}, baselineKeyNames...)...)
	authHeaderKeys         = newKeyNameSet("proxy_authorization", "authorization", "x_api_key", "x_goog_api_key", "api_key", "set_cookie", "cookie")
	textKeyNames           = newKeyNameSet(append([]string{session.KeyPrefixTemp, "oauth"}, slices.Collect(maps.Keys(sensitiveKeys))...)...)
	jsonKeyNames           = newKeyNameSet(append([]string{session.KeyPrefixUser, session.KeyPrefixApp, session.KeyPrefixTemp}, slices.Collect(maps.Keys(sensitiveKeys))...)...)
)

var (
	baselineAuthHeaderPrefixAtRe = regexp.MustCompile(`(?i)^` + baselineAuthHeaderPrefixPattern)
	baselineBearerTokenAtRe      = regexp.MustCompile(`(?i)^` + baselineBearerTokenPattern)
	baselineBasicTokenAtRe       = regexp.MustCompile(`(?i)^` + baselineBasicTokenPattern)
	baselineJSONKeyValTokenAtRe  = regexp.MustCompile(`(?i)^` + baselineJSONKeyValTokenPattern)
	baselineTextKeyValTokenAtRe  = regexp.MustCompile(`(?i)^` + baselineTextKeyValTokenPattern)
	baselineNestedKeyNameRe      = regexp.MustCompile(`(?i)^(?:\\?["'])?(?:` + regexp.QuoteMeta(session.KeyPrefixUser) + `|` + regexp.QuoteMeta(session.KeyPrefixApp) + `)?(` + baselineFixedKeyRegex() + `)$`)
	baselineCompoundKeyNameRe    = regexp.MustCompile(`(?i)^(?:\\?["'])?(?:[^\s"'\\]*[^\w\s"'\\])?(` + baselineKeyRegex(`=:"'\s\r\n,}`) + `)$`)
	baselineQuotedValueRe        = regexp.MustCompile(`^(?:` + baselineQuotedValueRegex() + `)`)
)

// baselineDelimSpace holds the bytes that [\s[:cntrl:]] matches around the
// delimiters of baselineTextKeyValTokenRe.
const baselineDelimSpace = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\v\f\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f \x7f"

// baselineCuts returns the ranges of s, in order, whose bytes the baseline
// pass replaces with a placeholder, without the padding at their ends (see
// isCutPadding) and without any placeholder already in s. Like the passes of
// scrubTextPatterns, the baseline pass redacts Authorization and API-key
// headers, then bearer and basic credentials, then the values of JSON pairs
// and then those of key-value pairs, each in the text that the one before it
// wrote.
func baselineCuts(s string) []textRange {
	_, cuts := baselineScrub(s)
	if len(cuts) == 0 {
		return nil
	}
	return refineCuts(s, cuts)
}

// baselineScrub returns the text that the baseline pass writes for s, and the
// ranges of s whose bytes it replaced with a placeholder.
func baselineScrub(s string) (string, []textRange) {
	var cuts []textRange
	// kept maps the text that the last step wrote back to s.
	kept := []copySpan{{0, 0, len(s)}}
	cur := s
	step := func(scrub func(string, *passRecorder) string) {
		var rec passRecorder
		next := scrub(cur, &rec)
		cuts = append(cuts, mapRanges(rec.cut, kept)...)
		kept = composeSpans(kept, rec.kept)
		cur = next
	}
	step(baselineAuthHeaders)
	step(func(s string, rec *passRecorder) string {
		var quotes openQuotes
		return replaceMatchesFunc(s, func(s string) []int {
			return findAtKeys(s, 0, bearerKeys, baselineBearerTokenAtRe)
		}, func(s string, loc []int) (string, int) {
			return baselineRedactBearerToken(s, loc, &quotes)
		}, rec)
	})
	step(func(s string, rec *passRecorder) string {
		return replaceMatchesFunc(s, func(s string) []int {
			return findAtKeys(s, 0, basicKeys, baselineBasicTokenAtRe)
		}, redactBasicCredential, rec)
	})
	step(baselineJSONKeyValues)
	step(baselineTextKeyValues)
	return cur, cuts
}

// baselineAuthHeaders replaces the value of each Authorization or API-key
// header that baselineAuthHeaderPrefixAtRe matches at a word boundary, to the
// end of its line, with the redaction placeholder.
func baselineAuthHeaders(s string, rec *passRecorder) string {
	var b strings.Builder
	var quotes *headerQuoteScanner
	last := 0
	curLineEnd := -1
	for pos := 0; ; {
		loc := findAtKeys(s, pos, baselineAuthHeaderKeys, baselineAuthHeaderPrefixAtRe)
		if loc == nil {
			break
		}
		if quotes == nil {
			quotes = newHeaderQuoteScanner(s)
		}
		if loc[1] > curLineEnd {
			curLineEnd = len(s)
			if nl := strings.IndexAny(s[loc[1]:], "\r\n"); nl >= 0 {
				curLineEnd = loc[1] + nl
			}
		}
		repl := s[loc[2]:loc[3]] + ": " + redactedValue
		end := curLineEnd
		// A header in a string that is open before it, as the one of
		// curl -H 'Authorization: Bearer abc' https://x is, ends at the
		// quote on its line that closes that string, so that
		// keepBaselineCuts keeps that quote and the text after it. Only
		// the header's line is read for that quote, so that the pass
		// stays linear however many headers the string holds, and a
		// single quote between two letters, as the one of Bob's is, reads
		// as an apostrophe and leaves the cut at the end of the line.
		if outer, inner := quotes.openBefore(loc[0], loc[3]); outer != "" || inner != "" {
			inSingleInJSON := outer == `"` && hasSingleQuoteBeforeHeader(s, loc[1]) &&
				((inner == "" && quotes.inJSON()) || (inner == `\"` && quotes.curInnerJSONKind != jsonContainerNone))
			inEscDQInInnerJSON := outer == `"` && inner == `\"` && quotes.curInnerJSONKind != jsonContainerNone && hasEscDQBeforeHeader(s, loc[1])
			if outer == `"` && inner == `\"` {
				if endPos, closedQuotes, ok := quotes.jsonWrappedInnerHeaderEnd(loc[1], end, inSingleInJSON, inEscDQInInnerJSON); ok {
					if closedQuotes == "" {
						end = endPos
					} else {
						cutInner := inner
						cutOuter := ""
						if len(closedQuotes) > len(inner) && !strings.HasSuffix(closedQuotes, inner) {
							cutOuter = outer
						}
						cutEnd := closingQuotesStart(s, loc[3], endPos, cutOuter, cutInner)
						if closedQuotes != "" && cutEnd == endPos {
							repl += closedQuotes
						}
						end = cutEnd
					}
				}
			} else if outer == "" && inner != "" {
				singleHandled := false
				if inner == `\"` && (quotes.curInnerSubSQ || hasSingleQuoteBeforeHeader(s, loc[1])) {
					for k := loc[1]; k < end; {
						if s[k] == '\\' {
							bs := 0
							for k+bs < end && s[k+bs] == '\\' {
								bs++
							}
							if k+bs < end && s[k+bs] == '"' && bs%4 == 1 {
								break
							}
							k += bs
							continue
						}
						if s[k] == '\'' && (!isWordApostropheAt(s, k) || isTrailingApostrophe(s, k)) {
							if endPos, unclosed := quotes.joinedHeaderArgEnd(k+1, end, "", false, false); unclosed == "" && endPos == k+1 {
								end = k
								singleHandled = true
								break
							}
						}
						k++
					}
				}
				if !singleHandled {
					if k := quoteCloseScan(s[:end], loc[3], outer, inner); k > loc[3] {
						k, unclosed := quotes.joinedHeaderArgEnd(k, end, outer, quotes.inJSON(), false)
						if unclosed == "" && k < end && (s[k] == ' ' || s[k] == '\t' || s[k] == ';' || s[k] == '&' || s[k] == '|') {
							cutEnd := closingQuotesStart(s, loc[3], k, outer, inner)
							if cutEnd == k {
								repl += inner
							}
							end = cutEnd
						}
					}
				}
			} else {
				singleHandled := false
				if inSingleInJSON {
					for k := loc[1]; k < end; {
						if s[k] == '\\' {
							if k+1 < end {
								k += 2
							} else {
								k++
							}
							continue
						}
						if s[k] == '"' {
							break
						}
						if s[k] == '\'' {
							if endPos, closedQuotes, ok := quotes.jsonWrappedInnerHeaderEndAt(k+1, end, outer, "'"); ok {
								if (endPos > k+1 && s[endPos-1] == '"' && s[endPos-2] == '\'') || (endPos == k+1 && closedQuotes != "") {
									end = k
								} else {
									cutEnd := closingQuotesStart(s, loc[3], endPos, outer, "")
									if closedQuotes != "" && cutEnd == endPos {
										repl += closedQuotes
									}
									end = cutEnd
								}
								singleHandled = true
								break
							}
						}
						k++
					}
				}
				if !singleHandled {
					if k := quoteCloseScan(s[:end], loc[3], outer, inner); k > loc[3] {
						var unclosed string
						k, unclosed = quotes.joinedHeaderArgEnd(k, end, outer, quotes.inJSON(), false)
						if unclosed == "" {
							cutEnd := closingQuotesStart(s, loc[3], k, outer, inner)
							if cutEnd == k && !(k < end && outer != "" && s[k] == outer[0]) {
								repl += inner + outer
							}
							end = cutEnd
						} else {
							end = k
						}
					}
				}
			}
		} else if !isCookieHeaderKey(s[loc[2]:loc[3]]) {
			rest := strings.TrimLeft(s[loc[1]:end], " \t")
			if strings.HasPrefix(rest, redactedValue) {
				after := strings.TrimLeft(strings.TrimPrefix(rest, redactedValue), " \t")
				if after != "" && (after[0] == ';' || after[0] == '|' || strings.HasPrefix(after, "&&")) {
					lead := len(s[loc[1]:end]) - len(rest)
					end = loc[1] + lead + len(redactedValue)
				}
			} else if k0 := loc[1] + len(s[loc[1]:end]) - len(rest); k0 < end {
				if m := quotedValueAt(quotes.shellScanner(), s, k0); m != "" && k0+len(m) < end {
					after := strings.TrimLeft(s[k0+len(m):end], " \t")
					if after != "" && (after[0] == ';' || after[0] == '|' || strings.HasPrefix(after, "&&")) {
						end = k0 + len(m)
					}
				}
			}
		}
		b.WriteString(s[last:loc[0]])
		b.WriteString(repl)
		rec.copied(last, loc[0]-last)
		rec.replaced(s, loc[0], end, repl)
		last = end
		pos = max(end, loc[0]+1)
	}
	if last == 0 {
		rec.copied(0, len(s))
		return s
	}
	b.WriteString(s[last:])
	rec.copied(last, len(s)-last)
	return b.String()
}

// baselineRedactBearerToken replaces a baselineBearerTokenAtRe match. "bearer
// of" followed by a space is prose ("the bearer of bad news"), not a token.
// When a quote opened before the keyword closes at loc[4], it ends the
// enclosing string rather than opening a quoted token; if another quote
// follows at once, as in echo 'Bearer '"secret"; echo 'done', that quote
// opens the token.
func baselineRedactBearerToken(s string, loc []int, quotes *openQuotes) (string, int) {
	end := loc[5]
	quoted := quotes.before(s, loc[4])
	if q := openingQuote(s, loc[4]); q != "" && quoted&quoteKind(q) != 0 {
		if q2 := openingQuote(s, loc[4]+len(q)); q2 != "" {
			q2Start := loc[4] + len(q)
			if m := baselineQuotedValueRe.FindString(s[q2Start:]); m != "" && q2Start+len(m) < end && !startsQuotedKeyWithValue(s[q2Start:], 0) {
				end = q2Start + len(m)
			}
		}
	}
	value := s[loc[4]:end]
	if rest := s[loc[4]:]; len(rest) >= 2 && strings.EqualFold(rest[:2], "of") {
		if r, _ := utf8.DecodeRuneInString(rest[2:]); len(rest) == 2 || unicode.IsSpace(r) {
			quotes.skip(s, end)
			return s[loc[0]:end], end
		}
	}
	quotes.skip(s, end)
	if inner, _ := unquoteValue(value); isRedactedValue(inner) {
		return s[loc[0]:end], end
	}
	return s[loc[2]:loc[3]] + " " + redactedValue, end
}

// baselineJSONKeyValues replaces each baselineJSONKeyValTokenAtRe match in s as
// baselineRedactJSONKeyValue does, and records in rec which bytes of the
// result are copies of s (see passRecorder). The key that a replacement
// writes is recorded as a copy of the key in s, and the whitespace around
// that key, which the replacement leaves out, as neither copied nor cut, so
// that only the value reads as cut: the baseline pass rewrites the key
// without that whitespace, but does not redact it.
func baselineJSONKeyValues(s string, rec *passRecorder) string {
	var b strings.Builder
	last, pos := 0, 0
	for pos < len(s) {
		loc := baselineFindJSONKeyValToken(s, pos)
		if loc == nil {
			break
		}
		repl, end := baselineRedactJSONKeyValue(s, loc)
		b.WriteString(s[last:loc[0]])
		b.WriteString(repl)
		rec.copied(last, loc[0]-last)
		if key := s[loc[2]:loc[3]]; repl != s[loc[0]:end] && strings.HasPrefix(repl, `"`+key+`"`) {
			// The match opens with the quote before the key, and the first
			// quote after the key closes it.
			keyClose := loc[3] + strings.IndexByte(s[loc[3]:end], '"')
			rec.copied(loc[0], 1)
			rec.copied(loc[2], len(key))
			rec.replaced(s, keyClose, end, repl[1+len(key):])
		} else {
			rec.replaced(s, loc[0], end, repl)
		}
		last = end
		pos = max(end, loc[0]+1)
	}
	if last == 0 {
		rec.reset()
		rec.copied(0, len(s))
		return s
	}
	b.WriteString(s[last:])
	rec.copied(last, len(s)-last)
	return b.String()
}

// baselineRedactJSONKeyValue replaces a baselineJSONKeyValTokenAtRe match with
// "key":"[REDACTED]".
func baselineRedactJSONKeyValue(s string, loc []int) (string, int) {
	if pEnd := placeholderCut(s, loc[4]); pEnd > 0 {
		if pEnd == len(s) || quotesEndLine(s, pEnd) || ((s[pEnd] == '"' || s[pEnd] == '\'') && (pEnd+1 == len(s) || strings.IndexByte(" \t\r\n,;)]}", s[pEnd+1]) >= 0)) || strings.IndexByte(" \t\r\n,;)]}", s[pEnd]) >= 0 {
			return s[loc[0]:pEnd], pEnd
		}
		if q := openingQuote(s, pEnd); len(q) == 2 && pEnd+2 < len(s) && s[pEnd+2] == q[1] &&
			(pEnd+3 >= len(s) || strings.IndexByte(" \t\r\n,;)]}", s[pEnd+3]) >= 0) {
			return s[loc[0]:pEnd], pEnd
		}
	}
	end := baselineSensitiveValueEnd(s, loc[4], loc[5])
	if inner, _ := unquoteValue(s[loc[4]:end]); isRedactedValue(inner) {
		return s[loc[0]:end], end
	}
	return `"` + s[loc[2]:loc[3]] + `":"` + redactedValue + `"`, end
}

// baselineFindJSONKeyValToken returns the submatch indexes in s of the first
// match of baselineJSONKeyValTokenAtRe at or after s[from:], or nil. A match
// opens with a quote, which spaces may follow, and then a key name of
// baselineKeys.
func baselineFindJSONKeyValToken(s string, from int) []int {
	return findAtQuotedKeys(s, from, baselineKeys, false, baselineJSONKeyValTokenAtRe)
}

// findJSONKeyValToken returns the submatch indexes in s of the first match of
// jsonKeyValTokenRe in s, or nil. A match opens with a quote, which spaces
// may follow, and then a key name of jsonKeyNames, which a run of "_", "-"
// and "." may come before (see sensitiveKeyRegex).
func findJSONKeyValToken(s string) []int {
	return findAtQuotedKeys(s, 0, jsonKeyNames, true, jsonKeyValTokenAtRe)
}

// findAtQuotedKeys returns the submatch indexes in s of the first match of at
// at or after s[from:], or nil: at is anchored with ^, and each of its
// matches opens with a quote, which spaces that [\s\p{Z}] matches may follow,
// and then a key name of keys, or, if separated, a run of "_", "-" and "."
// and then one, so at is tried only where one of those names may start
// (see keyNameSet), as findAtKeys tries the patterns that open with a key.
func findAtQuotedKeys(s string, from int, keys *keyNameSet, separated bool, at *regexp.Regexp) []int {
	for i := from; i < len(s); i++ {
		q := strings.IndexByte(s[i:], '"')
		if q < 0 {
			return nil
		}
		i += q
		k := i + 1
		for k < len(s) {
			r, n := utf8.DecodeRuneInString(s[k:])
			if !isJSONKeySpace(r) {
				break
			}
			k += n
		}
		for separated && k < len(s) && (s[k] == '_' || s[k] == '-' || s[k] == '.') {
			k++
		}
		if keys.mayStartAt(s, k) {
			if loc := at.FindStringSubmatchIndex(s[i:]); loc != nil {
				return offsetLoc(loc, i)
			}
		}
	}
	return nil
}

// isJSONKeySpace reports whether r is a space that [\s\p{Z}] matches.
func isJSONKeySpace(r rune) bool {
	switch r {
	case '\t', '\n', '\f', '\r', ' ':
		return true
	}
	return r >= utf8.RuneSelf && unicode.Is(unicode.Z, r)
}

// baselineFindKeyValToken returns the submatch indexes in s of the first
// match of baselineTextKeyValTokenAtRe at a word boundary in s, or at its
// start without one, or nil.
func baselineFindKeyValToken(s string) []int {
	if s == "" {
		return nil
	}
	if baselineKeys.mayStartAt(s, 0) {
		if loc := baselineTextKeyValTokenAtRe.FindStringSubmatchIndex(s); loc != nil {
			return loc
		}
	}
	return findAtKeys(s, 1, baselineKeys, baselineTextKeyValTokenAtRe)
}

// findAtKeys returns the submatch indexes in s of the first match of at at
// or after s[from:] that starts at a word boundary, or nil: at is anchored
// with ^, and is tried only where a key name of keys, which each of its
// matches starts with, may start. That finds the match that the pattern of at
// with \b in place of ^ would find in s from offset from, and in the same
// time as a single search for that pattern takes when the text holds a few
// of those key names, rather than after reading each byte of the text with
// each alternative of the pattern. The boundary at from is read as \b reads
// it in s, with the byte before from, as FindAllStringSubmatchIndex and the
// other searches from an offset of a text do; a search in a text that starts
// at from reads none.
func findAtKeys(s string, from int, keys *keyNameSet, at *regexp.Regexp) []int {
	for i := from; i < len(s); i++ {
		if !wordBoundaryAt(s, i) || !keys.mayStartAt(s, i) {
			continue
		}
		if loc := at.FindStringSubmatchIndex(s[i:]); loc != nil {
			return offsetLoc(loc, i)
		}
	}
	return nil
}

// findAllAtKeys returns the submatch indexes in s of the matches of at that
// findAtKeys finds, in order, each searched for from the end of the one
// before it, as FindAllStringSubmatchIndex searches for the pattern of at
// with \b in place of ^. A match holds a key name, so it is not empty.
func findAllAtKeys(s string, keys *keyNameSet, at *regexp.Regexp) [][]int {
	var locs [][]int
	for pos := 0; ; {
		loc := findAtKeys(s, pos, keys, at)
		if loc == nil {
			return locs
		}
		locs = append(locs, loc)
		pos = loc[1]
	}
}

// findBearerToken returns the submatch indexes in s of the first match of
// bearerTokenRe in s, or nil.
func findBearerToken(s string) []int {
	return findAtKeys(s, 0, bearerKeys, bearerTokenAtRe)
}

// findBasicToken returns the submatch indexes in s of the first match of
// basicTokenRe in s, or nil.
func findBasicToken(s string) []int {
	return findAtKeys(s, 0, basicKeys, basicTokenAtRe)
}

// findTextKeyName returns the submatch indexes in s of the first match of
// textKeyNameRe in s, or nil. A match opens with a word boundary or a "_" and
// then a name of bareTextKeyNameRegex, which starts with a letter, so it can
// start only where a key name of textKeyNames may start (see keyNameSet)
// after a word boundary, as \b reads it in s, or after a "_" that the match
// holds; it is searched for there only, as findAtKeys searches.
//
// The dotted OAuth name oauth[a-z0-9_-]*[.]code reads the run of bytes that
// [a-z0-9_-] matches that it starts in to its end, so the pattern that holds
// it is tried at the start of an "oauth" only if that run is followed by the
// rest of such a name (see oauthTailAtRe): whether it is, is the same for every
// "oauth" in the run, and is read once, as oauth-oauth-oauth-... holds one
// run, which a pattern tried at each of its words would read again and again.
func findTextKeyName(s string) []int {
	// oauthEnd is the end of the run that the last "oauth" read was in, and
	// oauthTail reports whether the rest of a dotted OAuth name follows it.
	oauthEnd, oauthTail := 0, false
	for i := 0; i < len(s); i++ {
		start := i
		underscore := s[i] == '_'
		if underscore {
			if !textKeyNames.mayStartAt(s, i+1) {
				continue
			}
			start = i + 1
		} else if !wordBoundaryAt(s, i) || !textKeyNames.mayStartAt(s, i) {
			continue
		}
		withOAuth := false
		if len(s)-start >= len("oauth") && strings.EqualFold(s[start:start+len("oauth")], "oauth") {
			if start >= oauthEnd {
				oauthEnd = textKeyRunEnd(s, start)
				oauthTail = oauthTailAtRe.MatchString(s[oauthEnd:])
			}
			withOAuth = oauthTail
		}
		var at *regexp.Regexp
		switch {
		case underscore && withOAuth:
			at = textKeyNameAtUnderscoreRe
		case underscore:
			at = textKeyNameNoOAuthAtUnderscoreRe
		case withOAuth:
			at = textKeyNameAtBoundaryRe
		default:
			at = textKeyNameNoOAuthAtBoundaryRe
		}
		if loc := at.FindStringSubmatchIndex(s[i:]); loc != nil {
			return offsetLoc(loc, i)
		}
	}
	return nil
}

// textKeyRunEnd returns the offset in s at which the run of the runes that
// [a-z0-9_-] matches in a case-insensitive pattern, which include the Kelvin
// sign and the long s, that starts at s[i:] ends.
func textKeyRunEnd(s string, i int) int {
	for i < len(s) {
		switch c, size := foldKeyLetter(s, i); {
		case isTextKeyByte(s[i]):
			i++
		case c == 'k' || c == 's':
			i += size
		default:
			return i
		}
	}
	return i
}

// mayStartTextKeyName reports whether textKeyNameAtRe may match at the start
// of s: whether a key name of textKeyNames may start in the run of bytes that
// [a-z0-9_-] matches in a case-insensitive pattern, as the compound prefix of
// a key does (see textKeyNameRegex), that s starts with. The credential name
// of the key starts with a letter, so it starts in that run.
func mayStartTextKeyName(s string) bool {
	for j := 0; j < len(s); {
		if textKeyNames.mayStartAt(s, j) {
			return true
		}
		switch c, size := foldKeyLetter(s, j); {
		case isTextKeyByte(s[j]):
			j++
		case c == 'k' || c == 's':
			// The Kelvin sign or the long s.
			j += size
		default:
			return false
		}
	}
	return false
}

// offsetLoc adds offset to each submatch index in loc that is not -1, and
// returns loc.
func offsetLoc(loc []int, offset int) []int {
	for k, v := range loc {
		if v >= 0 {
			loc[k] = v + offset
		}
	}
	return loc
}

// wordBoundaryAt reports whether \b matches at s[i:]: whether exactly one of
// the runes before and at s[i] is an ASCII word character, as \b reads them.
// Every byte of a rune outside ASCII, and every byte of invalid UTF-8, is at
// or after a byte that is not an ASCII word character itself.
func wordBoundaryAt(s string, i int) bool {
	before := i > 0 && isASCIIWordByte(s[i-1])
	return before != (i < len(s) && isASCIIWordByte(s[i]))
}

// isASCIIWordByte reports whether c is an ASCII word character, [0-9A-Za-z_].
func isASCIIWordByte(c byte) bool {
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9' || c == '_'
}

// keyNameSet holds a set of key names, as a trie of their letters, for
// telling where one of them may start in a text without reading the text
// with a regular expression at each of its bytes: findAtKeys and the other
// searches of the patterns that open with one of the names try the pattern
// only where mayStartAt finds the letters of a name.
type keyNameSet struct {
	// next holds the nodes for the letters "a" to "z", and for ":", that
	// may follow the letters that lead to this node in a name.
	next [27]*keyNameSet
	// end reports whether the letters that lead to this node are a name.
	end bool
}

// newKeyNameSet returns the keyNameSet of names, which are written in lower
// case, with ASCII letters, "_" and ":" only, and start with a letter. A "_"
// is not a letter of a name, as any run of separators may be read in its
// place (see mayStartAt).
func newKeyNameSet(names ...string) *keyNameSet {
	root := &keyNameSet{}
	for _, name := range names {
		n := root
		for i := 0; i < len(name); i++ {
			if name[i] == '_' {
				continue
			}
			k := keyNameIndex(name[i])
			if n.next[k] == nil {
				n.next[k] = &keyNameSet{}
			}
			n = n.next[k]
		}
		n.end = true
	}
	return root
}

// keyNameIndex returns the index in keyNameSet.next of c, a lower-case ASCII
// letter or ":".
func keyNameIndex(c byte) int {
	if c == ':' {
		return 26
	}
	return int(c - 'a')
}

// mayStartAt reports whether a key name of p may start at s[i:]: whether the
// letters of one of the names follow, in order, as a case-insensitive
// pattern reads them, with any run of "_", "-" and "." between any two of
// them, as fixedSensitiveKeyRegex reads separators. A pattern that reads
// fewer separators in a name, or none, as the patterns of the baseline pass
// and bearerTokenRe do, finds no name at s[i:] when mayStartAt finds none.
func (p *keyNameSet) mayStartAt(s string, i int) bool {
	n := p
	for i < len(s) {
		c, size := foldKeyLetter(s, i)
		switch {
		case 'a' <= c && c <= 'z' || c == ':':
			if n = n.next[keyNameIndex(c)]; n == nil {
				return false
			}
			if n.end {
				return true
			}
		case n == p || c != '_' && c != '-' && c != '.':
			return false
		}
		i += size
	}
	return false
}

// foldKeyLetter returns the lower-case ASCII letter, or the "_", "-", "." or
// ":", that the rune at s[i:] matches in a case-insensitive pattern, or 0 if
// it matches none, and the length of the rune if it matches one. Case
// folding matches two runes outside ASCII with ASCII letters: the Kelvin
// sign with "k" and the long s with "s".
func foldKeyLetter(s string, i int) (byte, int) {
	switch c := s[i]; {
	case 'a' <= c && c <= 'z', c == '_', c == '-', c == '.', c == ':':
		return c, 1
	case 'A' <= c && c <= 'Z':
		return c + 'a' - 'A', 1
	case strings.HasPrefix(s[i:], "\u017f"):
		return 's', len("\u017f")
	case strings.HasPrefix(s[i:], "\u212a"):
		return 'k', len("\u212a")
	}
	return 0, 1
}

// baselineTextKeyValues replaces each baselineTextKeyValTokenAtRe match in s as
// baselineRedactTextKeyValueWithQuotes does, skipping dotted or hyphenated pagination
// cursor suffixes ("token" preceded by "." or "-" where isPageTokenName holds)
// before running baselineTextKeyValTokenAtRe so long runs of dotted cursor
// names stay linear while preserving baselineFindKeyValToken's exact regexp
// behavior.
func baselineTextKeyValues(s string, rec *passRecorder) string {
	var brackets openBrackets
	var quotes openQuotes
	var sh shellAssignScanner
	var b strings.Builder
	last, pos := 0, 0
	for i := 0; i < len(s); i++ {
		if i > pos && !wordBoundaryAt(s, i) {
			continue
		}
		if !baselineKeys.mayStartAt(s, i) {
			continue
		}
		if i > 0 && (s[i-1] == '.' || s[i-1] == '-') && len(s)-i >= len("token") &&
			strings.EqualFold(s[i:i+len("token")], "token") &&
			(len(s)-i == len("token") || !isTextKeyByte(s[i+len("token")])) &&
			isPageTokenName(s[dottedKeyStart(s, i):i+len("token")]) {
			i += len("token") - 1
			continue
		}
		loc := baselineTextKeyValTokenAtRe.FindStringSubmatchIndex(s[i:])
		if loc == nil {
			continue
		}
		loc = offsetLoc(loc, i)
		quoted := quotes.before(s, loc[0])
		innerQuoted := quotes.innerAt(s, loc[0])
		subSQ := quotes.subSingleAt(s, loc[0])
		subEscDQ := quotes.subEscDoubleAt(s, loc[0])
		repl, end := baselineRedactTextKeyValueWithQuotes(&sh, s, loc, &brackets, &quotes, quoted, innerQuoted, subSQ, subEscDQ)
		quotes.skip(s, max(end, loc[0]+1))
		b.WriteString(s[last:loc[0]])
		b.WriteString(repl)
		rec.copied(last, loc[0]-last)
		rec.replaced(s, loc[0], end, repl)
		last = end
		pos = max(end, loc[0]+1)
		i = pos - 1
	}
	if last == 0 {
		rec.reset()
		rec.copied(0, len(s))
		return s
	}
	b.WriteString(s[last:])
	rec.copied(last, len(s)-last)
	return b.String()
}

// baselineRedactTextKeyValueWithQuotes replaces a baselineTextKeyValTokenRe match,
// keeping the key, its quote, the delimiter and the quotes around the value.
// A value that is itself a sensitive key is left in place, so that the pair
// it starts is matched on its own, and one that only ends in a key name is
// redacted together with the value of the pair it starts (see
// baselineNestsSensitiveKey). brackets counts the brackets opened before the
// key.
func baselineRedactTextKeyValueWithQuotes(sh *shellAssignScanner, s string, loc []int, brackets *openBrackets, quotes *openQuotes, quoted, innerQuoted quoteKinds, subSQ, subEscDQ bool) (string, int) {
	if loc[2] > 0 && (s[loc[2]-1] == '.' || s[loc[2]-1] == '-') {
		if isPageTokenName(s[dottedKeyStart(s, loc[2]):loc[3]]) {
			return s[loc[0]:loc[7]], loc[7]
		}
	}
	if loc[4] == loc[5] && strings.HasPrefix(strings.ToLower(s[loc[2]:loc[3]]), session.KeyPrefixTemp) {
		if qPair := enclosingQuotedTempPair(s, loc[0], loc[3]); qPair != nil {
			loc = qPair
		}
	}
	open := brackets.before(s, loc[0])
	start, end := loc[8], loc[9]
	if pEnd := placeholderCut(s, start); pEnd > 0 {
		if pEnd == len(s) || quotesEndLine(s, pEnd) || len(openingQuote(s, pEnd)) == 2 || ((s[pEnd] == '"' || s[pEnd] == '\'') && (pEnd+1 == len(s) || strings.IndexByte(" \t\r\n,;)]}", s[pEnd+1]) >= 0)) || strings.IndexByte(" \t\r\n,;)]}", s[pEnd]) >= 0 || placeholderEndsQuote(s, pEnd, quoted, innerQuoted, subSQ, subEscDQ) {
			brackets.skip(pEnd)
			return s[loc[0]:pEnd], pEnd
		}
		if q := openingQuote(s, pEnd); len(q) == 2 && pEnd+2 < len(s) && s[pEnd+2] == q[1] {
			k := pEnd + 3
			for k < len(s) && (s[k] == '"' || s[k] == '\'') {
				k++
			}
			if k >= len(s) || strings.IndexByte(" \t\r\n,;)]}", s[k]) >= 0 {
				brackets.skip(pEnd)
				return s[loc[0]:pEnd], pEnd
			}
		}
	}
	switch kind, pair := baselineNestsSensitiveKey(s, start, end, startsLine(s, loc[7], start)); kind {
	case nestedKey:
		return s[loc[0]:start], start
	case valueEndingInKeyName:
		_, quote := unquoteValue(s[start:end])
		end = baselineNestedPairEnd(s, pair, open)
		brackets.skip(end)
		return s[loc[2]:loc[3]] + s[loc[4]:loc[5]] + s[loc[6]:loc[7]] + quote + redactedValue + quote, end
	}
	end = baselineTextValueEndWithQuotes(sh, quotes, s, start, end, open, quoted, innerQuoted, subSQ, subEscDQ)
	brackets.skip(end)
	if end == start {
		return s[loc[0]:end], end
	}
	inner, quote := unquoteValue(s[start:end])
	if isRedactedValue(inner) {
		return s[loc[0]:end], end
	}
	return s[loc[2]:loc[3]] + s[loc[4]:loc[5]] + s[loc[6]:loc[7]] + quote + redactedValue + quote, end
}

// enclosingQuotedTempPair checks whether a "temp:" match at s[keyStart:keyEnd]
// stands inside a single-line quoted key that contains a ":" or "=" delimiter
// (such as 'temp:a=b': 'x' or "user:temp:a:b": "x"), and if so returns the
// outer quoted pair's submatch slice starting at keyStart.
func enclosingQuotedTempPair(s string, keyStart, keyEnd int) []int {
	p := keyStart
	for p > 0 && s[p-1] != '\n' && s[p-1] != '\r' && isDelimSpace(rune(s[p-1])) {
		p--
	}
	for _, pfx := range []string{session.KeyPrefixUser, session.KeyPrefixApp} {
		if p >= len(pfx) && strings.EqualFold(s[p-len(pfx):p], pfx) {
			p -= len(pfx)
			for p > 0 && s[p-1] != '\n' && s[p-1] != '\r' && isDelimSpace(rune(s[p-1])) {
				p--
			}
			break
		}
	}
	qOpen := -1
	if p >= 2 && s[p-2] == '\\' && (s[p-1] == '"' || s[p-1] == '\'') {
		qOpen = p - 2
	} else if p >= 1 && (s[p-1] == '"' || s[p-1] == '\'') {
		qOpen = p - 1
	}
	if qOpen < 0 {
		return nil
	}
	qPair := quotedPairAtStart(s[qOpen:])
	if qPair == nil || qPair[8] < 0 || qOpen+qPair[3] <= keyEnd {
		return nil
	}
	offsetLoc(qPair, qOpen)
	qPair[0], qPair[2] = keyStart, keyStart
	return qPair
}

// baselineTextValueEnd returns the end of the value matched at s[start:end]
// after a key: a quoted value ends at its closing quote, a value that opens a
// JSON object or array is extended as in baselineSensitiveValueEnd, and a
// bare value ends before the "}" and "]" at its end that close a bracket
// opened before the key (see trimEnclosingClosers). open is the number of
// brackets opened before the key.
func baselineTextValueEnd(s string, start, end, open int) int {
	return baselineTextValueEndWithQuotes(nil, nil, s, start, end, open, 0, 0, false, false)
}

func baselineTextValueEndWithQuotes(sh *shellAssignScanner, quotes *openQuotes, s string, start, end, open int, quoted, innerQuoted quoteKinds, subSQ, subEscDQ bool) int {
	if inner, quote := unquoteValue(s[start:end]); quote != "" && isRedactedValue(inner) {
		return end
	}
	if open > 0 || innerQuoted&(4|8) != 0 {
		end = textValueEndWithEnclosing(sh, s, start, end, open, quoted, innerQuoted, subSQ, subEscDQ, quotes)
		if end > start && openingQuote(s, start) == "" &&
			(quoted != 0 || innerQuoted != 0 || subSQ || subEscDQ) &&
			strings.IndexByte(s[start:end], '|') >= 0 {
			end = baselineStopBeforeClosingPipe(s, start, end, quoted, innerQuoted, subSQ, subEscDQ)
		}
		return trimTrailingSubshellOrJSONQuotes(s, start, end, quoted, innerQuoted, subSQ, subEscDQ)
	}
	if _, quote := unquoteValue(s[start:end]); quote != "" {
		return end
	}
	rest := baselineSensitiveValueEnd(s, start, start)
	end = rest + len(trimEnclosingClosers(s[rest:max(end, rest)], open))
	if rest == start && openingQuote(s, start) == "" {
		end = extendSubshellQuotedValueEnd(quotes, s, start, end, quoted, innerQuoted, subSQ, subEscDQ)
	}
	if end > start && openingQuote(s, start) == "" &&
		(quoted != 0 || innerQuoted != 0 || subSQ || subEscDQ) &&
		strings.IndexByte(s[start:end], '|') >= 0 {
		end = baselineStopBeforeClosingPipe(s, start, end, quoted, innerQuoted, subSQ, subEscDQ)
	}
	return trimTrailingSubshellOrJSONQuotes(s, start, end, quoted, innerQuoted, subSQ, subEscDQ)
}

func baselineStopBeforeClosingPipe(s string, start, end int, quoted, innerQuoted quoteKinds, subSQ, subEscDQ bool) int {
	curInner, curSubSQ, curSubEscDQ := innerQuoted, subSQ, subEscDQ
	for i := start; i < end; i++ {
		if i <= start {
			continue
		}
		if s[i] == '\\' {
			k := i
			for k < end && s[k] == '\\' {
				k++
			}
			nBS := k - i
			if k < end && (s[k] == '"' || s[k] == '\'') && nBS%2 == 1 {
				if s[k] == '"' && curSubEscDQ && !curSubSQ &&
					((curInner&4 == 0 && nBS%4 == 3) || (curInner&4 != 0 && (nBS%8 == 7 || nBS%4 == 3))) {
					qLen := 4
					if curInner&4 != 0 && nBS%8 == 7 {
						qLen = 8
					}
					qStart := k + 1 - qLen
					if qStart >= start && closedQuoteBeforePipe(s, k+1, len(s)) {
						return qStart
					}
					curSubEscDQ = false
					i = k
					continue
				}
				if s[k] == '"' && nBS%4 == 1 && (curInner|quoted)&4 != 0 && !curSubSQ && !curSubEscDQ {
					if k-1 >= start && closedQuoteBeforePipe(s, k+1, len(s)) {
						return k - 1
					}
					if curInner&4 != 0 {
						curInner &^= 4
					}
					i = k
					continue
				}
				if s[k] == '\'' && nBS%4 == 1 && (curInner|quoted)&8 != 0 {
					if k-1 >= start && closedQuoteBeforePipe(s, k+1, len(s)) {
						return k - 1
					}
					if curInner&8 != 0 {
						curInner &^= 8
					}
					i = k
					continue
				}
				i = k
				continue
			} else {
				i = k - 1
				continue
			}
		} else if curSubSQ && s[i] == '\'' {
			if closedQuoteBeforePipe(s, i+1, len(s)) {
				return i
			}
			if isMidWordApostrophe(s, i) && !opensPrefixedString(s, i) {
				if pipeEnd := sqCloseBeforePipe(s, i); pipeEnd >= 0 {
					return pipeEnd
				}
				continue
			}
			curSubSQ = false
		} else if !curSubSQ && !curSubEscDQ && ((quoted&2 != 0 && s[i] == '\'') || (quoted&1 != 0 && s[i] == '"')) {
			if closedQuoteBeforePipe(s, i+1, len(s)) {
				return i
			}
			if s[i] == '\'' && quoted&2 != 0 && isMidWordApostrophe(s, i) && !opensPrefixedString(s, i) {
				if pipeEnd := sqCloseBeforePipe(s, i); pipeEnd >= 0 {
					return pipeEnd
				}
			}
		}
	}
	return end
}

// baselineNestsSensitiveKey tells whether the value matched at s[start:end]
// is a sensitive key rather than a value, as nestsSensitiveKey does for the
// passes of scrubTextPatterns, but as the baseline pass reads values (see
// baselineValueStartsAt).
func baselineNestsSensitiveKey(s string, start, end int, lineStart bool) (nestedKeyKind, int) {
	if end-start > maxNestedKeyLen || s[start] == '{' || s[start] == '[' {
		return notNestedKey, 0
	}
	v := s[start:end]
	if inner, quote := unquoteValue(v); quote != "" {
		if !baselineValueFollowsDelimiter(s, end) {
			return notNestedKey, 0
		}
		return baselineNestedKeyName(inner, start+len(quote), lineStart)
	}
	if strings.HasSuffix(v, ":") || strings.HasSuffix(v, "=") {
		if !baselineValueStartsAt(s, end) {
			return notNestedKey, 0
		}
		return baselineNestedKeyName(v[:len(v)-1], start, lineStart)
	}
	if baselineFollowedByDelimiter(s, end) {
		if !baselineValueFollowsDelimiter(s, end) {
			return notNestedKey, 0
		}
		return baselineNestedKeyName(v, start, lineStart)
	}
	if q := strings.IndexAny(v, `"'\`); q > 0 && (v[q-1] == ':' || v[q-1] == '=') &&
		len(baselineQuotedValueRe.FindString(s[start+q:])) > end-start-q && baselineValueStartsAt(s, start+q) {
		if kind, pair := baselineNestedKeyName(v[:q-1], start, lineStart); kind != notNestedKey {
			return kind, pair
		}
	}
	i := strings.LastIndexAny(v, ":=")
	if i <= 0 {
		return notNestedKey, 0
	}
	rest := trimTrailingPunctuation(v[i+1:])
	inner, _ := unquoteValue(rest)
	if !isRedactedValue(inner) {
		for _, q := range []string{`\"`, `\'`, `"`, `'`} {
			if strings.HasSuffix(rest, q) {
				if inner2, q2 := unquoteValue(rest[:len(rest)-len(q)]); q2 != "" && isRedactedValue(inner2) {
					inner = inner2
					break
				}
			}
		}
		if !isRedactedValue(inner) {
			return notNestedKey, 0
		}
	}
	return baselineNestedKeyName(v[:i], start, lineStart)
}

// baselineNestedKeyName classifies a value that may start a nested pair (see
// baselineNestsSensitiveKey): name is the name of that pair, which begins at
// offset at of the text, and lineStart reports whether the value starts a
// line.
func baselineNestedKeyName(name string, at int, lineStart bool) (nestedKeyKind, int) {
	if _, ok := baselineKeySet[name]; ok {
		return nestedKey, at
	}
	if m := baselineNestedKeyNameRe.FindStringSubmatchIndex(name); m != nil {
		return nestedKey, at + m[2]
	}
	m := baselineCompoundKeyNameRe.FindStringSubmatchIndex(name)
	switch {
	case m == nil:
		return notNestedKey, 0
	case lineStart:
		return nestedKey, at + m[2]
	default:
		return valueEndingInKeyName, at + m[2]
	}
}

// baselineNestedPairEnd returns where the text to redact with a value that
// only ends in a key name ends, given pair, where the pair that the key name
// starts begins in s. open is the number of brackets opened before the key of
// the value.
func baselineNestedPairEnd(s string, pair, open int) int {
	for {
		loc := baselineTextKeyValTokenAtRe.FindStringSubmatchIndex(s[pair:])
		if loc == nil {
			return pair
		}
		start, end := pair+loc[8], pair+loc[9]
		kind, next := baselineNestsSensitiveKey(s, start, end, startsLine(s, pair+loc[7], start))
		if kind == notNestedKey {
			return baselineTextValueEnd(s, start, end, open)
		}
		pair = next
	}
}

// baselineFollowedByDelimiter reports whether a key-value delimiter, ":" or
// "=", follows s[:i], possibly after whitespace.
func baselineFollowedByDelimiter(s string, i int) bool {
	rest := strings.TrimLeft(s[i:], baselineDelimSpace)
	return rest != "" && (rest[0] == ':' || rest[0] == '=')
}

// baselineValueFollowsDelimiter reports whether a key-value delimiter follows
// s[:i], possibly after whitespace, and a value starts after it (see
// baselineValueStartsAt).
func baselineValueFollowsDelimiter(s string, i int) bool {
	rest := strings.TrimLeft(s[i:], baselineDelimSpace)
	return rest != "" && (rest[0] == ':' || rest[0] == '=') && baselineValueStartsAt(rest, 1)
}

// baselineValueStartsAt reports whether a value that can be redacted starts
// in s at i, possibly after whitespace: a quoted value that ends there (see
// baselineQuotedValueEnds), or else the first run of bytes up to the next
// whitespace, ",", ";" or "&", which holds a byte other than whitespace and
// closing punctuation.
func baselineValueStartsAt(s string, i int) bool {
	rest := strings.TrimLeft(s[i:], baselineDelimSpace)
	if rest != "" && (rest[0] == '"' || rest[0] == '\'' || rest[0] == '\\') {
		if q := baselineQuotedValueRe.FindString(rest); q != "" && baselineQuotedValueEnds(rest, len(q)) {
			rest, _ = unquoteValue(q)
			return strings.Trim(rest, baselineDelimSpace+closingPunctuation) != ""
		}
	}
	if n := strings.IndexAny(rest, regexpSpace+",;&"); n >= 0 {
		rest = rest[:n]
	}
	return strings.Trim(rest, baselineDelimSpace+closingPunctuation) != ""
}

// baselineQuotedValueEnds reports whether the quoted value s[:n] at the start
// of s ends there: s ends after it, or goes on with a character that is not a
// letter, a digit, "_" or the quote that the value starts with.
func baselineQuotedValueEnds(s string, n int) bool {
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

// baselineSensitiveValueEnd returns the end of the value matched at
// s[start:end]. A value that opens a JSON object or array is extended to its
// closing bracket, or to the end of s if it is not closed. Brackets inside
// strings do not count, including strings delimited by escaped quotes.
func baselineSensitiveValueEnd(s string, start, end int) int {
	if c := s[start]; c != '{' && c != '[' {
		return end
	}
	depth := 0
	// quote is the quote of the string being skipped. escaped reports that
	// the string was opened with an escaped quote (\" or \'); only the same
	// escaped quote closes it.
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

// redactAuthHeaders replaces the value of each Authorization, API-key,
// Cookie or Set-Cookie header that authHeaderRe matches with the redaction
// placeholder. The value runs to the end of its line, and on over later lines
// where a quote in it is not closed on its line (see authHeaderValueEnd), so
// that the opening quote is not redacted while the rest of the credential is
// left on the lines after it, as in Authorization: abc'x<LF>SECRET'. A Cookie
// or Set-Cookie name and a colon whose value does not open with a quote,
// starts with a word and holds no cookie pair (see cookieValueIsProse) on its
// one line, outside any quote, as in The cookie: Monster was here., are prose
// and are left as they are. A value in a string that was open before the
// header keeps the quote that closes that string (see closingQuotesStart).
func redactAuthHeaders(s string) string {
	return redactAuthHeadersRecorded(s, nil)
}

// redactAuthHeadersRecorded is redactAuthHeaders that also records in rec,
// if it is not nil, which bytes of the result are copies of s (see
// passRecorder).
func redactAuthHeadersRecorded(s string, rec *passRecorder) string {
	var b strings.Builder
	var quotes *headerQuoteScanner
	last := 0
	curLineStart, curLineEnd, curIndentLen := -1, -1, 0
	for pos := 0; ; {
		loc := findAtKeys(s, pos, authHeaderKeys, authHeaderPrefixAtRe)
		if loc == nil {
			break
		}
		if quotes == nil {
			quotes = newHeaderQuoteScanner(s)
		}
		outer, inner := quotes.openBefore(loc[0], loc[1])
		if loc[1] > curLineEnd {
			curLineStart = strings.LastIndexAny(s[:loc[0]], "\r\n") + 1
			curLineEnd = len(s)
			if nl := strings.IndexAny(s[loc[1]:], "\r\n"); nl >= 0 {
				curLineEnd = loc[1] + nl
			}
			curIndentLen = 0
			for curLineStart+curIndentLen < loc[0] && (s[curLineStart+curIndentLen] == ' ' || s[curLineStart+curIndentLen] == '\t') {
				curIndentLen++
			}
		}
		lineStart, lineEnd, indentLen := curLineStart, curLineEnd, curIndentLen
		canFold := outer == "" && inner == "" && lineStart+indentLen == loc[0]
		if isCookieHeaderKey(s[loc[2]:loc[3]]) && cookieLineProseAllowed(s, loc[1], lineEnd, outer, inner, quotes) && !hasFoldedHeaderContinuation(s, lineEnd, canFold, indentLen) {
			if sub := findAtKeys(s[loc[1]:lineEnd], 0, authHeaderKeys, authHeaderPrefixAtRe); sub != nil {
				pos = loc[1] + sub[0]
			} else {
				pos = lineEnd
			}
			continue
		}
		end, closedQuotes := authHeaderValueEnd(s, loc[1], lineEnd, outer, inner, canFold, indentLen, quotes)
		repl := s[loc[2]:loc[3]] + ": " + redactedValue
		if outer != "" || inner != "" {
			cutEnd := closingQuotesStart(s, loc[1], end, outer, inner)
			if closedQuotes != "" && cutEnd == end {
				repl += closedQuotes
			}
			end = cutEnd
		}
		b.WriteString(s[last:loc[0]])
		b.WriteString(repl)
		if rec != nil {
			rec.copied(last, loc[0]-last)
			rec.replaced(s, loc[0], end, repl)
		}
		last = end
		pos = max(end, loc[0]+1)
	}
	if last == 0 {
		// Every header read as prose: s is returned as it is, not as a
		// copy, so that the passes after this one read the same bytes.
		rec.reset()
		rec.copied(0, len(s))
		return s
	}
	b.WriteString(s[last:])
	rec.copied(last, len(s)-last)
	return b.String()
}

// closingQuotesStart returns where the quotes that close the strings outer
// and inner (see headerQuoteScanner.openBefore), which were open before the
// header value at s[from:end], start at the end of that value, after any
// spaces or tabs there, or end if it does not end with them. A header in a
// string that was open before it, as the one in
// curl -H 'Authorization: Bearer abc' is, ends with the quote that closes that
// string; that quote is left out of the redaction, so that a scrub of the
// result reads the string closed where s closed it, rather than reading it
// open up to the next quote on a later line and redacting the lines up to
// there. The quote that closes outer comes last and the escaped quote that
// closes inner before it, as in "{\"h\": \"Authorization: abc\""; a quote
// after a backslash is escaped, and so closes inner rather than outer, unless
// outer is a single quote, in which (as in a POSIX shell) a backslash is a
// byte of the string. Any other quote at the end of the value, such as one
// that closes a string that the value itself opened, as the last quote of
// Authorization: 'abc' does, is a byte of the value, and is redacted with it.
func closingQuotesStart(s string, from, end int, outer, inner string) int {
	v := end
	for v > from && (s[v-1] == ' ' || s[v-1] == '\t') {
		v--
	}
	start := end
	if outer != "" && v-1 >= from && s[v-1] == outer[0] && (outer == "'" || v-2 < from || s[v-2] != '\\') {
		v--
		start = v
	}
	if inner != "" && v-len(inner) >= from && s[v-len(inner):v] == inner {
		start = v - len(inner)
	}
	return start
}

// isCookieHeaderKey reports whether name is Cookie or Set-Cookie, ignoring
// case, the whitespace around it, and the "_", "-" and "." separators.
func isCookieHeaderKey(name string) bool {
	n := removeKeySeparators(strings.ToLower(strings.TrimFunc(name, isDelimSpace)))
	return n == "cookie" || n == "setcookie"
}

// isAuthHeaderKey reports whether name is an Authorization, API-key, Cookie, or
// Set-Cookie header name matched by authHeaderNamePattern, ignoring case, the
// whitespace around it, and the "_", "-" and "." separators.
func isAuthHeaderKey(name string) bool {
	switch removeKeySeparators(strings.ToLower(strings.TrimFunc(name, isDelimSpace))) {
	case "cookie", "setcookie", "authorization", "proxyauthorization",
		"apikey", "apikeysecret", "apikeystring",
		"xapikey", "xapikeysecret", "xapikeystring",
		"xgoogapikey", "xgoogapikeysecret", "xgoogapikeystring":
		return true
	default:
		return false
	}
}

// endsWithAuthHeaderKey reports whether name is an auth or cookie header key
// matched by isAuthHeaderKey or ends with "cookie" (such as my-cookie,
// x-set-cookie or foo.cookie).
func endsWithAuthHeaderKey(name string) bool {
	if len(name) >= len("cookie") && strings.EqualFold(name[len(name)-len("cookie"):], "cookie") {
		return true
	}
	return isAuthHeaderKey(name)
}

// cookieValueIsProse reports whether the text starting at s[start:] after a
// Cookie or Set-Cookie name and its colon reads as prose rather than as a
// cookie list (RFC 6265). Its first token on its line must be a word of
// letters, with at most punctuation after them, as Monster in The cookie:
// Monster was here. is, rather than an opaque value, as 5f2b9c0e7d1a is, or a
// value with a colon in it, as app:sid=x is; neither it nor a token that a
// semicolon or a comma joins to it (a cookie attribute, as HttpOnly in
// Set-Cookie: x; HttpOnly is, or the next item of a list, as sid=x in
// Set-Cookie: Secure, sid=x is) may be a cookie pair with "=", nor may the
// rest of the value hold "=", other than after a plain credential name that
// the key-value pass redacts on its own (see restHoldsCookiePair); and the
// text must not start with the redaction placeholder that a scrub of a header
// value leaves. A token that holds a colon ends the tokens read, so that the
// tokens read for one header are never read again for a header that the
// colon may start later on the line.
func cookieValueIsProse(s string, start int) bool {
	at := start
	for at < len(s) && s[at] != '\r' && s[at] != '\n' {
		r, width := utf8.DecodeRuneInString(s[at:])
		if !isDelimSpace(r) {
			break
		}
		at += width
	}
	if strings.HasPrefix(s[at:], redactedValue) {
		return false
	}
	// Handle a quoted first word: if the value starts with an opening quote,
	// validate the inner word as prose and advance past it into the
	// standard token loop, which checks that at least one more prose word
	// follows.
	first := true
	numFirst := true
	hasAlpha := false
	if q := openingQuote(s, at); q != "" {
		qcOuter, qcInner := q, ""
		if len(q) == 2 {
			qcOuter, qcInner = "", q
		}
		closeIdx := quoteCloseScan(s[:min(len(s), at+256)], at+len(q), qcOuter, qcInner)
		if closeIdx < 0 || closeIdx >= len(s) || s[closeIdx] == '\r' || s[closeIdx] == '\n' {
			return false
		}
		innerWord := s[at+len(q) : closeIdx-len(q)]
		if !isProseCookieWord(innerWord) {
			return false
		}
		afterQ := closeIdx
		for afterQ < len(s) && strings.IndexByte(".,;:!?)", s[afterQ]) >= 0 {
			afterQ++
		}
		sp := skipHorizDelimSpace(s, afterQ)
		if sp <= afterQ || sp >= len(s) || s[sp] == '\r' || s[sp] == '\n' {
			return false
		}
		numFirst = false
		at = sp
	}
	for at < len(s) && s[at] != '\r' && s[at] != '\n' {
		if s[at] == '"' || s[at] == '\'' || (s[at] == '$' && at+1 < len(s) && s[at+1] == '(') || s[at] == '`' || (s[at] == '\\' && at+1 < len(s) && (s[at+1] == '"' || s[at+1] == '\'' || s[at+1] == 'n' || s[at+1] == 'r' || s[at+1] == 't' || s[at+1] == '\\')) {
			if restHoldsCookiePair(s, at, false) {
				return false
			}
			break
		}
		k := at
		colon := false
		for k < len(s) {
			c := s[k]
			if c == '=' {
				return false
			}
			if c == ',' && k > at && s[k-1] >= '0' && s[k-1] <= '9' && k+1 < len(s) && s[k+1] >= '0' && s[k+1] <= '9' {
				k++
				continue
			}
			if c == '\'' && isMidWordApostrophe(s, k) && startsContraction(s, k, 1) {
				k++
				continue
			}
			if c == '\\' && k+1 < len(s) && (s[k+1] == '"' || s[k+1] == '\'' || s[k+1] == 'n' || s[k+1] == 'r' || s[k+1] == 't' || s[k+1] == '\\') {
				break
			}
			if c <= ' ' || c == ';' || c == ',' || c == '"' || c == '\'' || (c == '$' && k+1 < len(s) && s[k+1] == '(') || c == '`' {
				break
			}
			r, width := utf8.DecodeRuneInString(s[k:])
			if isDelimSpace(r) {
				break
			}
			if c == ':' {
				colon = true
			}
			k += width
		}
		if first && !isProseCookieWord(s[at:k]) {
			if numFirst && isProseQuantity(s[at:k]) {
				if sp := skipHorizDelimSpace(s, k); sp > k && sp < len(s) && s[sp] != '\r' && s[sp] != '\n' {
					numFirst = false
					at = sp
					continue
				}
			}
			return false
		}
		if isProseCookieWord(s[at:k]) {
			hasAlpha = true
		}
		first = false
		if colon {
			return hasAlpha && !restHoldsCookiePair(s, at, true)
		}
		sp := k
		if k > at {
			sp = skipHorizDelimSpace(s, k)
		}
		if sp < len(s) && s[sp] == '=' {
			return false
		}
		if sp < len(s) && (s[sp] == ';' || s[sp] == ',') {
			at = skipHorizDelimSpace(s, sp+1)
			continue
		}
		if restHoldsCookiePair(s, sp, true) {
			return false
		}
		break
	}
	return hasAlpha
}

// restHoldsCookiePair reports whether the rest of a Cookie value that
// cookieValueIsProse reads as prose, s[at:] up to the end of its line, the
// start of another Cookie or Authorization header on the line (including
// hyphen- or dot-prefixed cookie key names and delimiter whitespace before the
// colon), or the standard net/http ErrNoCookie phrase
// ("http: named cookie not present"), holds "=", as a cookie pair does: the
// sid=x of Cookie: "word" Monster sid=x, the "sid=x" of
// curl -H 'Cookie: chocolate, "sid=x"', the sid=x of
// Cookie: Monster at 10:30 sid=x, and the password=x of the JSON string
// "The cookie: chocolate\npassword=x", whose escaped line break leaves it on
// the line, all hold one. Any other "=" on the same line fails closed as a
// cookie pair, except when keyed is true and "=" stands right after a plain
// credential name (see isBareTextKey) that starts at s[at] or after a space or
// a tab, as the password= of cookie: Monster password=x does: the key-value
// pass redacts the value after it on its own, so the words before it may stay.
// After a JSON line-break or tab escape, keyed is cleared so that a credential
// pair or colon-delimited key on the escaped continuation line is recognized
// too.
func restHoldsCookiePair(s string, at int, keyed bool) bool {
	// name is where the run of key name bytes that ends at s[i] starts, if
	// s[at], a space or a tab starts it, or -1.
	name := at
	nestedEscSQ := false
	for i := at; i < len(s); i++ {
		c := s[i]
		if nestedEscSQ {
			switch c {
			case '=':
				return true
			case '\r', '\n':
				return true
			case '\\':
				if i+1 < len(s) && s[i+1] == '\'' {
					if !cookieProseCloseBoundaryOK(s, i+2, len(s), jsonContainerNone, true) {
						return true
					}
					nestedEscSQ = false
					i++
					name = -1
				}
			}
			continue
		}
		switch {
		case c == '=':
			if !keyed || name < 0 || name == i || !isBareTextKey(s[name:i]) {
				return true
			}
			name = -1
		case c == '\r' || c == '\n':
			return false
		case (c == '$' && i+1 < len(s) && s[i+1] == '(') || c == '`':
			return true
		case c == ':':
			const errNoCookieColon = ": named cookie not present"
			if len(s)-i >= len(errNoCookieColon) && strings.EqualFold(s[i:i+len(errNoCookieColon)], errNoCookieColon) {
				return false
			}
			end := i
			for end > at {
				r, width := utf8.DecodeLastRuneInString(s[at:end])
				if !isDelimSpace(r) && !(width == 1 && (s[end-1] <= ' ' || s[end-1] == 0x7f)) {
					break
				}
				end -= width
			}
			start := end
			for start > at && isTextKeyByte(s[start-1]) {
				start--
			}
			if start < end && endsWithAuthHeaderKey(s[start:end]) {
				return false
			}
			if !keyed && name >= 0 && name < i && isBareTextKey(s[name:i]) {
				return true
			}
			name = -1
		case c == ' ' || c == '\t':
			name = i + 1
		case c == '\\' && i+1 < len(s):
			switch s[i+1] {
			case '\'':
				if i > at && isTextKeyByte(s[i-1]) && startsContraction(s, i+1, 1) {
					i++
					name = -1
					continue
				}
				if i > at && !canOpenNestedEscDQInProse(s, i) {
					return true
				}
				nestedEscSQ = true
				i++
				name = -1
			case 'n', 'r', 't', '\\':
				keyed = false
				i++
				name = i + 1
			default:
				if name >= 0 && !isTextKeyByte(c) {
					name = -1
				}
			}
		case name >= 0 && !isTextKeyByte(c):
			name = -1
		}
	}
	return nestedEscSQ
}

// isProseCookieWord reports whether w, the first token of the text after a
// Cookie or Set-Cookie name and its colon, is a word that prose may hold
// there: one or more letters (optionally after an opening parenthesis), or
// hyphenated letter segments such as chocolate-chip, followed by nothing but
// a closing parenthesis or punctuation that may end a word in a sentence
// (".", "!", "?" or ":"), as Monster, (chocolate) and Monster. are and
// 5f2b9c0e and app:sid are not.
func isProseCookieWord(w string) bool {
	// Trim leading opening parenthesis and typographic open quotes.
	w = strings.TrimLeft(w, "(“”‘’")
	i := 0
	for {
		seg := 0
		for i+seg < len(w) {
			r, width := utf8.DecodeRuneInString(w[i+seg:])
			if !unicode.IsLetter(r) {
				break
			}
			seg += width
		}
		if seg == 0 {
			return false
		}
		i += seg
		if i < len(w) && (w[i] == '-' || w[i] == '\'') {
			i++
			continue
		}
		break
	}
	return strings.Trim(w[i:], ").,;:!?\u201d\u2019") == ""
}

// isUnsignedDecimal reports whether w is a non-negative numeric token:
// an integer ("2"), decimal ("2.5"), fraction ("1/2"), comma-grouped count
// ("2,000"), or a Unicode vulgar fraction ("½"). It is the numeric
// core of isProseQuantity without the sign, parenthesis, or unit handling.
func isUnsignedDecimal(w string) bool {
	if w == "" {
		return false
	}
	prevNum := false
	for i := 0; i < len(w); {
		c := w[i]
		if c == '.' || c == ',' || c == '/' {
			if !prevNum || i+1 >= len(w) {
				return false
			}
			prevNum = false
			i++
			continue
		}
		r, width := utf8.DecodeRuneInString(w[i:])
		if !unicode.IsNumber(r) {
			return false
		}
		prevNum = true
		i += width
	}
	return prevNum
}

// isLowerAlphaWord reports whether w is a word of two or more lowercase
// ASCII letters, as in "ounce" and "pack", so a hyphenated token like
// "2.5-ounce" reads as a quantity-unit compound rather than a value.
func isLowerAlphaWord(w string) bool {
	if len(w) < 2 {
		return false
	}
	for i := 0; i < len(w); i++ {
		if w[i] < 'a' || w[i] > 'z' {
			return false
		}
	}
	return true
}

// isASCIIDigits reports whether w is one or more ASCII digits.
func isASCIIDigits(w string) bool {
	if w == "" {
		return false
	}
	for i := 0; i < len(w); i++ {
		if w[i] < '0' || w[i] > '9' {
			return false
		}
	}
	return true
}

// isProseQuantity reports whether w is a quantity that may precede a prose
// noun after "cookie:": an unsigned number (see isUnsignedDecimal), as 2,
// 2.5, 1/2, 2,000 and ½ are; an ordinal, as 2nd is; a time of day, as 2:30,
// 2:30pm and 14:30:05 are; a range or a quantity-unit compound, as 2-3 and
// 2.5-ounce are; a multiplier, as x2 and 2x are; or a letter grade, as A+
// is. Each may have a leading parenthesis, sign, currency symbol or '#', as
// in $2 and #2, and trailing punctuation or '%'.
func isProseQuantity(w string) bool {
	w = strings.TrimLeft(w, "(-+$#\u20ac\u00a3\u00a5")
	w = strings.TrimRight(w, ").,;:!?%")
	n := len(w)
	if n == 0 {
		return false
	}
	if n == 2 && w[0]|0x20 >= 'a' && w[0]|0x20 <= 'z' && (w[1] == '+' || w[1] == '-') {
		return true
	}
	if (w[0] == 'x' || w[0] == 'X') && isUnsignedDecimal(w[1:]) {
		return true
	}
	if (w[n-1] == 'x' || w[n-1] == 'X') && isUnsignedDecimal(w[:n-1]) {
		return true
	}
	if n > 2 {
		switch strings.ToLower(w[n-2:]) {
		case "st", "nd", "rd", "th":
			if isASCIIDigits(w[:n-2]) {
				return true
			}
		}
	}
	if hh, rest, ok := strings.Cut(w, ":"); ok {
		if len(hh) > 2 || !isASCIIDigits(hh) {
			return false
		}
		if r := len(rest); r > 2 {
			if suffix := strings.ToLower(rest[r-2:]); suffix == "am" || suffix == "pm" {
				rest = rest[:r-2]
			}
		}
		mm, ss, hasSeconds := strings.Cut(rest, ":")
		return len(mm) == 2 && isASCIIDigits(mm) && (!hasSeconds || len(ss) == 2 && isASCIIDigits(ss))
	}
	if dash := strings.IndexByte(w, '-'); dash > 0 {
		numPart, unitPart := w[:dash], w[dash+1:]
		if isUnsignedDecimal(numPart) && (isUnsignedDecimal(unitPart) || isLowerAlphaWord(unitPart)) {
			return true
		}
	}
	return isUnsignedDecimal(w)
}

// cookieLineProseAllowed reports whether the text that starts at s[start:]
// after a Cookie or Set-Cookie name and its colon, on the line ending at or
// before lineEnd, reads as prose rather than as a cookie list (see
// cookieValueIsProse), whether or not the value is inside an open quote.
// When outer or inner is open (as in a JSON string), it finds where they
// close on the line, slices the value up to the closing quote, and reads
// that slice as prose; when neither is open, it reads the value directly.
// A value whose open quotes do not close on its line is not prose. quotes
// is the scanner of s that found outer and inner; it finds where they close
// (see headerQuoteScanner.closeAt), so that the checks for all the headers
// inside one long quoted string read that string once, not once each.
// cookieProseCloseBoundaryOK reports whether s[after:lineEnd] after a closing
// quote of a candidate Cookie or Set-Cookie prose segment forms a clean
// non-concatenated boundary. Whitespace, line breaks, shell command separators
// (;, &, |), closing grouping brackets followed by a clean boundary, sentence
// punctuation followed by whitespace/EOF/separator, and (when allowJSONDelim is
// true) JSON comma delimiters before valid JSON tokens are accepted; adjacent
// word/token/quote/escape/substitution characters or punctuation glued directly
// to non-whitespace tokens indicate shell concatenation and fail closed.
type jsonContainerKind uint8

const (
	jsonContainerNone jsonContainerKind = iota
	jsonContainerObject
	jsonContainerArray
)

type jsonContainerEntry struct {
	openChar byte
	pos      int
	isJSON   bool
}

type jsonContainerNode struct {
	entry jsonContainerEntry
	prev  *jsonContainerNode
}

type intListNode struct {
	val  int
	prev *intListNode
}

func jsonContainerMatches(openChar, closeChar byte) bool {
	return (openChar == '{' && closeChar == '}') || (openChar == '[' && closeChar == ']')
}

func isJSONContainerWordOrDollar(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '$'
}

func isJSONObjectOpenerHead(s string, pos int) bool {
	p := pos + 1
	for p < len(s) && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r' || s[p] == '\n') {
		p++
	}
	if p >= len(s) {
		return false
	}
	if s[p] == '}' {
		return true
	}
	if s[p] == '"' {
		lim := min(len(s), p+128)
		bs := 0
		for i := p + 1; i < lim; i++ {
			c := s[i]
			if c == '\r' || c == '\n' {
				return false
			}
			if c == '\\' {
				bs++
				continue
			}
			if c == '"' && bs%2 == 0 {
				after := i + 1
				afterLim := min(len(s), after+64)
				for after < afterLim && (s[after] == ' ' || s[after] == '\t' || s[after] == '\r' || s[after] == '\n') {
					after++
				}
				return after < len(s) && s[after] == ':'
			}
			bs = 0
		}
		return false
	}
	if s[p] == '\\' {
		openBS := 0
		for openBS < 8 && p+openBS < len(s) && s[p+openBS] == '\\' {
			openBS++
		}
		if openBS%2 == 1 && p+openBS < len(s) && s[p+openBS] == '"' {
			wantMod := openBS % 4
			lim := min(len(s), p+openBS+1+128)
			bs := 0
			for i := p + openBS + 1; i < lim; i++ {
				c := s[i]
				if c == '\r' || c == '\n' {
					return false
				}
				if c == '\\' {
					bs++
					continue
				}
				if c == '"' {
					if bs%2 == 1 && bs%4 == wantMod {
						after := i + 1
						afterLim := min(len(s), after+64)
						for after < afterLim && (s[after] == ' ' || s[after] == '\t' || s[after] == '\r' || s[after] == '\n') {
							after++
						}
						return after < len(s) && s[after] == ':'
					}
					return false
				}
				bs = 0
			}
		}
	}
	return false
}

func isJSONContainerOpener(s string, pos int) bool {
	if pos < 0 || pos >= len(s) {
		return false
	}
	if pos > 0 && isJSONContainerWordOrDollar(s[pos-1]) {
		return false
	}
	if s[pos] == '{' {
		return isJSONObjectOpenerHead(s, pos)
	}
	if s[pos] != '[' {
		return false
	}
	p := pos + 1
	for p < len(s) && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r' || s[p] == '\n') {
		p++
	}
	if p >= len(s) {
		return false
	}
	switch s[p] {
	case ']', '"', '[', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		return true
	case '{':
		return isJSONObjectOpenerHead(s, p)
	case '\\':
		bs := 0
		for bs < 8 && p+bs < len(s) && s[p+bs] == '\\' {
			bs++
		}
		return bs%2 == 1 && p+bs < len(s) && s[p+bs] == '"'
	case 't', 'f', 'n':
		for _, kw := range []string{"true", "false", "null"} {
			if strings.HasPrefix(s[p:min(len(s), p+len(kw))], kw) {
				after := p + len(kw)
				if after >= len(s) {
					return true
				}
				c := s[after]
				return c == ',' || c == ']' || c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '"'
			}
		}
	}
	return false
}

func classifyQuoteJSONContainer(s string, quotePos, boundStart int, escaped bool, jsonTop *jsonContainerNode) jsonContainerKind {
	if jsonTop == nil || !jsonTop.entry.isJSON {
		return jsonContainerNone
	}
	top := jsonTop.entry
	if top.pos < boundStart || quotePos <= top.pos {
		return jsonContainerNone
	}
	p := quotePos
	if escaped {
		for p > boundStart && s[p-1] == '\\' {
			p--
		}
	}
	for p > top.pos+1 {
		c := s[p-1]
		if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
			p--
			continue
		}
		break
	}
	if p == top.pos+1 {
		if top.openChar == '[' {
			return jsonContainerArray
		}
		return jsonContainerObject
	}
	if p <= top.pos {
		return jsonContainerNone
	}
	delim := s[p-1]
	if delim != ':' && delim != ',' {
		return jsonContainerNone
	}
	if top.openChar == '[' {
		return jsonContainerArray
	}
	return jsonContainerObject
}

func atLineEnd(s string, p, lineEnd int) bool {
	return p >= lineEnd || s[p] == '\r' || s[p] == '\n'
}

func cookieProseCloseBoundaryOK(s string, after, lineEnd int, jsonKind jsonContainerKind, innerEscaped bool) bool {
	p := after
	for !atLineEnd(s, p, lineEnd) && (s[p] == ')' || s[p] == ']' || s[p] == '}') {
		p++
	}
	if atLineEnd(s, p, lineEnd) {
		return true
	}
	c := s[p]
	if r, _ := utf8.DecodeRuneInString(s[p:]); isDelimSpace(r) {
		return true
	}
	if c == ';' || c == '&' || c == '|' {
		return true
	}
	if c == ',' || c == '.' || c == ':' || c == '!' || c == '?' {
		if c == ',' && jsonKind != jsonContainerNone {
			after := skipHorizDelimSpace(s[:min(lineEnd, p+64)], p+1)
			if atLineEnd(s, after, lineEnd) {
				return true
			}
			return jsonDelimAfterComma(s, after, lineEnd, jsonKind, innerEscaped)
		}
		q := p + 1
		for !atLineEnd(s, q, lineEnd) && (s[q] == ')' || s[q] == ']' || s[q] == '}') {
			q++
		}
		if atLineEnd(s, q, lineEnd) {
			return true
		}
		if r, _ := utf8.DecodeRuneInString(s[q:]); isDelimSpace(r) {
			return true
		}
		return s[q] == ';' || s[q] == '&' || s[q] == '|'
	}
	return false
}

func looksLikeJSONValueAt(s string, p, lineEnd int, escaped bool) (int, bool) {
	if atLineEnd(s, p, lineEnd) {
		return 0, false
	}
	lim := min(lineEnd, p+256)
	c := s[p]
	if escaped && c == '\\' {
		if p+1 >= lim || s[p+1] != '"' {
			return 0, false
		}
		bs := 0
		for i := p + 2; i < lim; i++ {
			ch := s[i]
			if ch == '\r' || ch == '\n' {
				return 0, false
			}
			if ch == '\\' {
				bs++
				continue
			}
			if ch == '"' {
				if bs%2 == 1 {
					if bs%4 == 1 {
						return i + 1, true
					}
					bs = 0
					continue
				}
				return 0, false
			}
			bs = 0
		}
		if p+256 <= lineEnd && lim < lineEnd {
			return lim, true
		}
		return 0, false
	}
	if !escaped && c == '"' {
		bs := 0
		for i := p + 1; i < lim; i++ {
			ch := s[i]
			if ch == '\r' || ch == '\n' {
				return 0, false
			}
			if ch == '\\' {
				bs++
				continue
			}
			if ch == '"' && bs%2 == 0 {
				return i + 1, true
			}
			bs = 0
		}
		if p+256 <= lineEnd && lim < lineEnd {
			return lim, true
		}
		return 0, false
	}
	if c == '{' {
		if !isJSONObjectOpenerHead(s, p) {
			return 0, false
		}
		if escaped {
			after := skipHorizDelimSpace(s[:min(lineEnd, p+64)], p+1)
			if !atLineEnd(s, after, lineEnd) && s[after] == '}' {
				return after + 1, true
			}
		}
		depth := 0
		inStr := false
		bs := 0
		for i := p; i < lim; i++ {
			ch := s[i]
			if ch == '\r' || ch == '\n' {
				return lim, true
			}
			if !inStr {
				if ch == '=' || ch == ';' {
					return 0, false
				}
				switch ch {
				case '{', '[':
					depth++
				case '}', ']':
					depth--
					if depth == 0 {
						return i + 1, true
					}
				case '"':
					if escaped {
						return 0, false
					}
					inStr = true
					bs = 0
				case '\\':
					if escaped && i+1 < lim && s[i+1] == '"' {
						inStr = true
						i++
						bs = 0
					}
				}
			} else {
				if ch == '\\' {
					bs++
					continue
				}
				if ch == '"' {
					if !escaped && bs%2 == 0 {
						inStr = false
					} else if escaped && bs%2 == 1 && bs%4 == 1 {
						inStr = false
					}
				}
				bs = 0
			}
		}
		return lim, true
	}
	if c == '[' {
		if !isJSONContainerOpener(s, p) {
			return 0, false
		}
		depth := 0
		inStr := false
		bs := 0
		for i := p; i < lim; i++ {
			ch := s[i]
			if ch == '\r' || ch == '\n' {
				return lim, true
			}
			if !inStr {
				if ch == '=' || ch == ';' {
					return 0, false
				}
				switch ch {
				case '[', '{':
					depth++
				case ']', '}':
					depth--
					if depth == 0 {
						return i + 1, true
					}
				case '"':
					if escaped {
						return 0, false
					}
					inStr = true
					bs = 0
				case '\\':
					if escaped && i+1 < lim && s[i+1] == '"' {
						inStr = true
						i++
						bs = 0
					}
				}
			} else {
				if ch == '\\' {
					bs++
					continue
				}
				if ch == '"' {
					if !escaped && bs%2 == 0 {
						inStr = false
					} else if escaped && bs%2 == 1 && bs%4 == 1 {
						inStr = false
					}
				}
				bs = 0
			}
		}
		return lim, true
	}
	if c == '-' || (c >= '0' && c <= '9') {
		i := p
		if c == '-' {
			i++
			if atLineEnd(s, i, lineEnd) || s[i] < '0' || s[i] > '9' {
				return 0, false
			}
		}
		for !atLineEnd(s, i, lineEnd) && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-') {
			i++
		}
		return i, true
	}
	for _, kw := range []string{"true", "false", "null"} {
		if strings.HasPrefix(s[p:min(lineEnd, p+len(kw))], kw) {
			return p + len(kw), true
		}
	}
	return 0, false
}

func jsonArrayChainOK(s string, p, lineEnd int, escaped bool) bool {
	for step := 0; step < 8; step++ {
		if step > 0 {
			p = skipHorizDelimSpace(s[:min(lineEnd, p+64)], p)
		}
		if atLineEnd(s, p, lineEnd) {
			return step > 0
		}
		next, ok := looksLikeJSONValueAt(s, p, lineEnd, escaped)
		if !ok {
			return false
		}
		if p+256 <= lineEnd && next == p+256 && next < lineEnd {
			return true
		}
		next = skipHorizDelimSpace(s[:min(lineEnd, next+64)], next)
		if atLineEnd(s, next, lineEnd) {
			return true
		}
		ch := s[next]
		if escaped && ch == '"' {
			return jsonCloseBoundaryOK(s, next+1, lineEnd, false, true)
		}
		switch ch {
		case ',':
			p = next + 1
		case ']':
			return jsonCloseBoundaryOK(s, next+1, lineEnd, escaped, true)
		default:
			return false
		}
	}
	return true
}

func jsonCloseBoundaryOK(s string, after, lineEnd int, escaped, allowComma bool) bool {
	for {
		if atLineEnd(s, after, lineEnd) {
			return true
		}
		if escaped && s[after] == '"' {
			escaped = false
			allowComma = true
			after++
			continue
		}
		if r, _ := utf8.DecodeRuneInString(s[after:]); isDelimSpace(r) {
			return true
		}
		ac := s[after]
		if ac == ';' || ac == '&' || ac == '|' {
			return true
		}
		if ac == '}' || ac == ']' {
			after++
			for after < lineEnd && (s[after] == '}' || s[after] == ']') {
				after++
			}
			continue
		}
		if ac == '.' || ac == '!' || ac == '?' {
			if atLineEnd(s, after+1, lineEnd) {
				return true
			}
			r, _ := utf8.DecodeRuneInString(s[after+1:])
			return isDelimSpace(r)
		}
		if ac == ',' {
			afterC := skipHorizDelimSpace(s[:min(lineEnd, after+64)], after+1)
			if atLineEnd(s, afterC, lineEnd) {
				return true
			}
			if !allowComma {
				_, valOK := looksLikeJSONValueAt(s, afterC, lineEnd, escaped)
				return valOK
			}
			if jsonArrayChainOK(s, afterC, lineEnd, escaped) {
				return true
			}
			return looksLikeJSONKeyAfterCommaDepth(s, afterC, lineEnd, escaped, false)
		}
		return false
	}
}

// jsonDelimAfterComma reports whether s[p:lineEnd] after a comma following a
// closing quote starts a valid compact JSON token rather than a shell-joined
// continuation such as ",sid=...".
func jsonDelimAfterComma(s string, p, lineEnd int, jsonKind jsonContainerKind, escaped bool) bool {
	if atLineEnd(s, p, lineEnd) {
		return false
	}
	if jsonKind == jsonContainerObject {
		return looksLikeJSONKeyAfterComma(s, p, lineEnd, escaped)
	}
	if jsonKind == jsonContainerArray {
		return jsonArrayChainOK(s, p, lineEnd, escaped)
	}
	return false
}

func jsonCommaBreakOK(s string, after, lineEnd int, outerJSONKind jsonContainerKind, escaped bool) bool {
	if atLineEnd(s, after, lineEnd) {
		return true
	}
	if outerJSONKind == jsonContainerArray {
		return jsonArrayChainOK(s, after, lineEnd, escaped)
	}
	return looksLikeJSONKeyAfterComma(s, after, lineEnd, escaped)
}

// looksLikeJSONKeyAfterComma reports whether a quoted token at s[p:lineEnd]
// after a JSON comma cleanly closes as valid JSON object field(s) within
// bounded lookahead.
func looksLikeJSONKeyAfterComma(s string, p, lineEnd int, escaped bool) bool {
	return looksLikeJSONKeyAfterCommaDepth(s, p, lineEnd, escaped, true)
}

func looksLikeJSONKeyAfterCommaDepth(s string, p, lineEnd int, escaped, allowCommaClose bool) bool {
	for step := 0; step < 8; step++ {
		if step > 0 {
			p = skipHorizDelimSpace(s[:min(lineEnd, p+64)], p)
			if atLineEnd(s, p, lineEnd) {
				return true
			}
		}
		limit := min(lineEnd, p+256)
		valPos := -1
		if escaped {
			if p+1 >= limit || s[p] != '\\' || s[p+1] != '"' {
				return false
			}
			bs := 0
			for i := p + 2; i < limit; i++ {
				c := s[i]
				if c == '\r' || c == '\n' || c == '=' {
					return false
				}
				if c == '\\' {
					bs++
					continue
				}
				if c == '"' {
					if bs%2 == 1 {
						if bs%4 == 1 {
							after := skipHorizDelimSpace(s[:min(lineEnd, i+64)], i+1)
							if atLineEnd(s, after, lineEnd) || s[after] != ':' {
								return false
							}
							valPos = skipHorizDelimSpace(s[:min(lineEnd, after+65)], after+1)
							break
						}
						bs = 0
						continue
					}
					return false
				}
				bs = 0
			}
		} else {
			if p >= limit || s[p] != '"' {
				return false
			}
			bs := 0
			for i := p + 1; i < limit; i++ {
				c := s[i]
				if c == '\r' || c == '\n' || c == '=' {
					return false
				}
				if c == '\\' {
					bs++
					continue
				}
				if c == '"' && bs%2 == 0 {
					after := skipHorizDelimSpace(s[:min(lineEnd, i+64)], i+1)
					if atLineEnd(s, after, lineEnd) || s[after] != ':' {
						return false
					}
					valPos = skipHorizDelimSpace(s[:min(lineEnd, after+65)], after+1)
					break
				}
				bs = 0
			}
		}
		if valPos < 0 {
			return false
		}
		if atLineEnd(s, valPos, lineEnd) {
			return true
		}
		valEnd, valOK := looksLikeJSONValueAt(s, valPos, lineEnd, escaped)
		if !valOK {
			return false
		}
		if valPos+256 <= lineEnd && valEnd == valPos+256 && valEnd < lineEnd {
			return true
		}
		next := skipHorizDelimSpace(s[:min(lineEnd, valEnd+64)], valEnd)
		if atLineEnd(s, next, lineEnd) {
			return true
		}
		if escaped && s[next] == '"' {
			return true
		}
		switch s[next] {
		case ',':
			p = next + 1
		case '}':
			return jsonCloseBoundaryOK(s, next+1, lineEnd, escaped, allowCommaClose)
		default:
			return false
		}
	}
	return true
}

func checkCookieSubBounds(s string, start, lineEnd int, contentEnd *int, subSQ bool, findSubSQClose func(int, int) int, substTop *intListNode) bool {
	for node := substTop; node != nil; node = node.prev {
		endPos := node.val
		if endPos <= *contentEnd {
			if !cookieProseCloseBoundaryOK(s, endPos, *contentEnd, jsonContainerNone, false) {
				return false
			}
			if endPos > start && endPos-1 < *contentEnd {
				*contentEnd = endPos - 1
			}
		} else if endPos <= lineEnd {
			if !cookieProseCloseBoundaryOK(s, endPos, lineEnd, jsonContainerNone, false) {
				return false
			}
		}
	}
	if subSQ {
		sqAt := findSubSQClose(start, *contentEnd)
		if sqAt < 0 {
			return false
		}
		if !cookieProseCloseBoundaryOK(s, sqAt+1, *contentEnd, jsonContainerNone, false) {
			return false
		}
		*contentEnd = sqAt
	}
	return true
}

type escDQCache struct {
	s             string
	from          int
	end           int
	escDQSlashPos []int
	badOutside    []bool
}

func canOpenNestedEscDQInProse(s string, pos int) bool {
	if pos <= 0 {
		return false
	}
	prev := s[pos-1]
	if prev == ' ' || prev == '\t' || prev == '\n' || prev == '\r' || prev == '(' || prev == '[' || prev == '{' || prev == ':' || prev == ';' {
		return true
	}
	r, _ := utf8.DecodeLastRuneInString(s[:pos])
	return isDelimSpace(r)
}

func hasConcatOrUnclosedEscDQ(c *escDQCache, s string, start, end int) bool {
	if c != nil && len(c.s) == len(s) && c.s == s && c.end == end && c.from <= start {
		if len(c.escDQSlashPos) == 0 {
			return false
		}
		idx, _ := slices.BinarySearch(c.escDQSlashPos, start)
		return c.badOutside[idx]
	}
	var slashPos []int
	var openOK []bool
	var closeOK []bool
	bs := 0
	for i := start; i < end; i++ {
		ch := s[i]
		if ch == '\\' {
			bs++
			continue
		}
		if ch == '"' && bs%4 == 1 {
			slashPos = append(slashPos, i-1)
			openOK = append(openOK, canOpenNestedEscDQInProse(s, i-1))
			closeOK = append(closeOK, cookieProseCloseBoundaryOK(s, i+1, end, jsonContainerNone, true))
		}
		bs = 0
	}
	var badOutside []bool
	if len(slashPos) > 0 {
		badOutside = make([]bool, len(slashPos)+1)
		for j := len(slashPos) - 1; j >= 0; j-- {
			if !openOK[j] || j+1 >= len(slashPos) || !closeOK[j+1] || badOutside[j+2] {
				badOutside[j] = true
			}
		}
	}
	if c != nil {
		*c = escDQCache{
			s:             s,
			from:          start,
			end:           end,
			escDQSlashPos: slashPos,
			badOutside:    badOutside,
		}
	}
	if len(slashPos) == 0 {
		return false
	}
	return badOutside[0]
}

func cookieLineProseAllowed(s string, start, lineEnd int, outer, inner string, quotes *headerQuoteScanner) bool {
	if outer == "" && inner == "" {
		contentEnd := len(s)
		if quotes != nil {
			for node := quotes.curSubstTop; node != nil; node = node.prev {
				endPos := node.val
				if endPos > start && endPos <= lineEnd {
					if !cookieProseCloseBoundaryOK(s, endPos, lineEnd, jsonContainerNone, false) {
						return false
					}
					if endPos-1 < contentEnd {
						contentEnd = endPos - 1
					}
				}
			}
		}
		return cookieValueIsProse(s[:contentEnd], start)
	}
	k := quotes.closeAt(start, outer, inner)
	if k < 0 || k > lineEnd {
		return false
	}
	if k < lineEnd && openingQuote(s, k) != "" {
		return false
	}
	contentEnd := closingQuotesStart(s, start, k, outer, inner)
	outerBoundJSON := jsonContainerNone
	var innerBoundJSON jsonContainerKind
	var outerSubSQ, innerSubSQ bool
	var outerSubstTop, innerSubstTop *intListNode
	if quotes != nil {
		switch inner {
		case "":
			if outer == `"` {
				outerBoundJSON = quotes.curOuterJSONKind
			}
		case `\"`:
			innerBoundJSON = quotes.curInnerJSONKind
		}
		outerSubSQ = quotes.curOuterSubSQ
		innerSubSQ = quotes.curInnerSubSQ
		outerSubstTop = quotes.curSubstTop
		innerSubstTop = quotes.curInnerSubstTop
	}
	if !cookieProseCloseBoundaryOK(s, k, lineEnd, outerBoundJSON, outer == "" && inner != "") {
		return false
	}
	if outer != "" && inner != "" {
		innerK := quotes.closeAt(start, "", inner)
		if innerK > start && innerK <= contentEnd {
			innerContentEnd := closingQuotesStart(s, start, innerK, "", inner)
			afterInner := innerK
			for afterInner < k && (s[afterInner] == ')' || s[afterInner] == ']' || s[afterInner] == '}') {
				afterInner++
			}
			afterInner = skipHorizDelimSpace(s[:k], afterInner)
			if afterInner+len(outer) != k || s[afterInner:k] != outer {
				if !cookieProseCloseBoundaryOK(s, innerK, lineEnd, innerBoundJSON, true) {
					return false
				}
			}
			contentEnd = innerContentEnd
		}
	}
	if inner != "" {
		if inner == `\"` && !checkCookieSubBounds(s, start, lineEnd, &contentEnd, innerSubSQ, func(from, limit int) int {
			return quotes.subSQCloseAt(1, from, limit)
		}, innerSubstTop) {
			return false
		}
	} else if !checkCookieSubBounds(s, start, lineEnd, &contentEnd, outerSubSQ, func(from, limit int) int {
		return quotes.subSQCloseAt(0, from, limit)
	}, outerSubstTop) {
		return false
	}
	if outer == `"` && inner == "" {
		var escCache *escDQCache
		if quotes != nil {
			escCache = &quotes.escDQ
		}
		if hasConcatOrUnclosedEscDQ(escCache, s, start, contentEnd) {
			return false
		}
	}
	return cookieValueIsProse(s[:contentEnd], start)
}

// isCookieProse reports whether an unquoted key name, the delimiter delim
// after it, and the value that starts at s[start:] read as the word cookie
// and a colon in prose, as in The cookie: Monster was here., rather than as a
// Cookie or Set-Cookie header: name is Cookie or Set-Cookie (see
// isCookieHeaderKey), delim is a colon, any quote that opens the value or
// encloses s[start] closes on the same line with a non-concatenated boundary
// (mirroring cookieLineProseAllowed), and the value slice up to any enclosing
// closing quote reads as prose (see cookieValueIsProse).
// Structured Cookie and Set-Cookie fields, whose keys are quoted or whose
// values are read by key, are redacted whatever their values hold.
func isCookieProse(s, name, delim string, start int, inBracketContainer bool, quotes *openQuotes) bool {
	if strings.TrimFunc(delim, isDelimSpace) != ":" || !isCookieHeaderKey(name) {
		return false
	}
	quoted := quotes.before(s, start)
	inner := quotes.innerAt(s, start)
	lineEnd := len(s)
	// If the value opens with a quote of a different kind than what is already
	// open, check that it closes on the same line; if it does not, the value
	// is a header inside a quoted string, not prose.
	if q := openingQuote(s, start); q != "" && (quoted|inner)&quoteKind(q) == 0 {
		qcOuter, qcInner := q, ""
		if len(q) == 2 {
			qcOuter, qcInner = "", q
		}
		closeIdx := quoteCloseScan(s[:min(len(s), start+256)], start+len(q), qcOuter, qcInner)
		if closeIdx < 0 || closeIdx >= len(s) || s[closeIdx] == '\r' || s[closeIdx] == '\n' {
			return false
		}
	}
	sliceEnd := len(s)
	if quoted != 0 || inner != 0 {
		outerJSONKind := jsonContainerNone
		if inBracketContainer && quotes != nil && quoted == 1 {
			outerJSONKind = quotes.openJSONKind
		}
		innerJSONKind := jsonContainerNone
		if quotes != nil && inner == 4 {
			innerJSONKind = quotes.inner.jsonKind
		}
		var outerAt, outerAfter int
		if quoted != 0 {
			outerAt, outerAfter = quotes.closeOnLine(s, start, quoted)
			if outerAt < 0 {
				return false
			}
			outerBoundJSON := outerJSONKind
			if inner != 0 {
				outerBoundJSON = jsonContainerNone
			}
			if !cookieProseCloseBoundaryOK(s, outerAfter, lineEnd, outerBoundJSON, quoted == 4 || quoted == 8) {
				return false
			}
			sliceEnd = outerAt
		}
		if inner != 0 {
			innerAt, innerAfter := quotes.closeOnLine(s, start, inner)
			if innerAt >= 0 && (quoted == 0 || innerAt < outerAt) {
				afterInner := innerAfter
				if quoted != 0 {
					for afterInner < outerAt && (s[afterInner] == ')' || s[afterInner] == ']' || s[afterInner] == '}') {
						afterInner++
					}
					afterInner = skipHorizDelimSpace(s[:outerAt], afterInner)
				}
				if quoted == 0 || afterInner != outerAt {
					if !cookieProseCloseBoundaryOK(s, innerAfter, lineEnd, innerJSONKind, true) {
						return false
					}
				}
				sliceEnd = innerAt
			} else if quoted == 0 {
				return false
			}
		}
		if inner != 0 {
			if inner == 4 && quotes != nil && !checkCookieSubBounds(s, start, lineEnd, &sliceEnd, quotes.inner.subSQ, func(from, limit int) int {
				return quotes.subSQCloseOnLine(s, 1, from, limit)
			}, quotes.inner.substTop) {
				return false
			}
		} else if quotes != nil && !checkCookieSubBounds(s, start, lineEnd, &sliceEnd, quotes.subSQ, func(from, limit int) int {
			return quotes.subSQCloseOnLine(s, 0, from, limit)
		}, quotes.substTop) {
			return false
		}
	} else if quotes != nil {
		for node := quotes.substTop; node != nil; node = node.prev {
			endPos := node.val
			if endPos > start && endPos <= lineEnd {
				if !cookieProseCloseBoundaryOK(s, endPos, lineEnd, jsonContainerNone, false) {
					return false
				}
				if endPos-1 < sliceEnd {
					sliceEnd = endPos - 1
				}
			}
		}
	}
	if quoted == 1 && inner == 0 {
		var escCache *escDQCache
		if quotes != nil {
			escCache = &quotes.escDQ
		}
		if hasConcatOrUnclosedEscDQ(escCache, s, start, sliceEnd) {
			return false
		}
	}
	return cookieValueIsProse(s[:sliceEnd], start)
}

// hasFoldedHeaderContinuation reports whether canFold is true and the line
// ending at s[lineEnd] is followed immediately by an RFC 9112 whitespace-folded
// continuation line (see isFoldedHeaderLine).
func hasFoldedHeaderContinuation(s string, lineEnd int, canFold bool, indentLen int) bool {
	if !canFold || lineEnd >= len(s) {
		return false
	}
	next := lineEnd + 1
	if s[lineEnd] == '\r' && next < len(s) && s[next] == '\n' {
		next++
	}
	return next < len(s) && isFoldedHeaderLine(s, next, indentLen)
}

// headerQuoteScanner finds the quotes that are open before each header that
// redactAuthHeaders reads in s, in order (see openBefore). It reads the lines
// of s once, front to back, and carries the state of the quotes that are open
// at the end of the lines before one header on to the next, rather than
// reading back over those lines again for each header, so that a quote that
// opens any number of lines and bytes before a header and stays open into
// it, as the first quote of 'X-Pad: <5000 bytes><LF>Authorization: Bearer abc<LF>SECRET'
// does, is found, and finding the quotes for all the headers in s takes time
// linear in its length.
type headerQuoteScanner struct {
	s string
	// lines is the state of the scan of the lines of s, up to the line break
	// before the line of the last header that openBefore read the lines
	// before for.
	lines lineQuoteState
	// curLine is the state of the scan on the line of the last header that
	// openBefore read, so that subsequent headers on the same line advance
	// from curLine.k rather than rescanning from lineStart.
	curLine    lineQuoteState
	hasCurLine bool
	balLineEnd int
	balInJSON  bool
	balMemo    map[int]bool
	// closes caches where the quotes that stay open into a header from a line
	// before it close, by kind of quote (see quoteCloses).
	closes [4]quoteCloseMemo
	// proseCloses caches where the quotes that are open before a Cookie
	// header's value close, by kind of quote (see closeAt).
	proseCloses [4]quoteCloseMemo
	// joinedCloses caches where multiline joined header quotes close on later
	// lines without mixing joined-quote lookahead into proseCloses.
	joinedCloses [4]quoteCloseMemo
	// chainCloses caches whether a joined-quote close position is followed by
	// balanced shell quoting through the remainder of its multiline chain.
	chainCloses map[int]bool
	// shell answers inShellAssignment for the quotes of s. It belongs to
	// this scan alone, so that its cursors are never evicted by another
	// scrub and nothing keeps s once the scan is done.
	shell *shellAssignScanner
	// joinedFrom, joinedLineEnd, joinedEnd, joinedEntryOuter, and
	// joinedUnclosed memoize the most recent joinedHeaderArgEnd scan on a
	// line so repeated headers on the same line never rescan the same
	// suffix.
	joinedFrom, joinedLineEnd, joinedEnd int
	joinedEntryOuter, joinedUnclosed     string
	joinedInJSON, joinedEscJSON          bool
	curOuterJSON                         bool
	curOuterJSONKind, curInnerJSONKind   jsonContainerKind
	curOuterSubSQ, curInnerSubSQ         bool
	curSubstTop, curInnerSubstTop        *intListNode
	jsonOuterFrom, jsonOuterLineEnd      int
	jsonOuterEnd                         int
	plainPrefixReady                     bool
	plainPrefixBits                      []uint64
	cookieOpenReady                      bool
	cookieOpenBits                       []uint64
	proseCloseMemo                       map[int]bool
	closeTailStops                       map[int]bool
	subStates                            [2]subshellScanState
	shellMemos                           shellCloseMemo
	subSQEnds                            [2]quoteCloseCache
	escDQ                                escDQCache
	nextLineUnclosedMemo                 map[int]bool
}

type subshellScanState struct {
	ready          bool
	lineStart      int
	realLineEnd    int
	scopeStart     int
	inInnerJSONStr bool
	pos            int
	depth          int
	inSingle       bool
	inDouble       bool
	cmdPos         bool
}

// newHeaderQuoteScanner returns a headerQuoteScanner for s.
func newHeaderQuoteScanner(s string) *headerQuoteScanner {
	return &headerQuoteScanner{s: s, lines: newLineQuoteState(s, 0, len(s)), shell: newShellAssignScanner(s)}
}

// inJSON reports whether the outer quote open before the most recent
// openBefore call opened inside a JSON container.
func (h *headerQuoteScanner) inJSON() bool {
	return h != nil && h.curOuterJSON
}

func (h *headerQuoteScanner) subSQCloseAt(slot, from, limit int) int {
	if h == nil || from >= limit {
		return -1
	}
	c := &h.subSQEnds[slot]
	if c.from <= from && c.lineEnd == limit && (c.at >= from || c.at < 0) {
		return c.at
	}
	at := -1
	if rel := strings.IndexByte(h.s[from:limit], '\''); rel >= 0 {
		at = from + rel
	}
	*c = quoteCloseCache{from: from, at: at, after: at + 1, lineEnd: limit}
	return at
}

// shellScanner returns the scanner that answers inShellAssignment for the
// text that h reads, or nil if h is nil.
func (h *headerQuoteScanner) shellScanner() *shellAssignScanner {
	if h == nil {
		return nil
	}
	return h.shell
}

func endsAtOuterWithClosers(s string, pos, outerEnd int) bool {
	for pos < outerEnd {
		switch s[pos] {
		case ' ', '\t', ')', ']', '}', '\'':
			pos++
		case '\\':
			if pos+1 < outerEnd && (s[pos+1] == '"' || s[pos+1] == '\'') {
				pos += 2
			} else {
				return false
			}
		default:
			return false
		}
	}
	return true
}

// jsonWrappedInnerHeaderEndAt resolves where an inner escaped-quoted header
// (such as -H \"Authorization: ...\" inside an outer JSON string) ends on
// s[k:lineEnd] once inner has closed at k while outer (") remains open.
func (h *headerQuoteScanner) jsonWrappedInnerHeaderEndAt(k, lineEnd int, entryOuter, entryInner string) (int, string, bool) {
	if h == nil || k > lineEnd {
		return 0, "", false
	}
	s := h.s
	outerEnd := -1
	if h.jsonOuterLineEnd == lineEnd && k >= h.jsonOuterFrom && k <= h.jsonOuterEnd {
		outerEnd = h.jsonOuterEnd
	} else {
		for p := k; p < lineEnd; {
			if s[p] == '\\' {
				bs := 0
				for p+bs < lineEnd && s[p+bs] == '\\' {
					bs++
				}
				if p+bs < lineEnd && (s[p+bs] == '"' || s[p+bs] == '\'') {
					if bs%2 == 1 {
						p += bs + 1
					} else {
						p += bs
					}
				} else {
					p += bs
				}
				continue
			}
			if s[p] == '"' {
				outerEnd = p
				break
			}
			p++
		}
		if outerEnd >= 0 {
			h.jsonOuterFrom, h.jsonOuterLineEnd, h.jsonOuterEnd = k, lineEnd, outerEnd
		}
	}
	if outerEnd < 0 {
		return 0, "", false
	}
	inJSONInner := entryInner == "'"
	escJSONInner := !inJSONInner
	outerBoundaryOK := outerEnd >= lineEnd || jsonCloseBoundaryOK(s, outerEnd+1, lineEnd, false, true)
	endPos, unclosed := h.joinedHeaderArgEnd(k, outerEnd, entryOuter, inJSONInner, escJSONInner)
	if !outerBoundaryOK && (unclosed != "" || endsAtOuterWithClosers(s, endPos, outerEnd)) {
		tailEnd, _ := h.joinedHeaderArgEnd(outerEnd+1, lineEnd, "", false, false)
		return tailEnd, "", true
	}
	if unclosed == "" && endPos < outerEnd {
		return endPos, entryInner, true
	}
	if outerEnd < lineEnd {
		return outerEnd + 1, entryInner + entryOuter, true
	}
	return outerEnd, entryInner, true
}

func hasSingleQuoteBeforeHeader(s string, from int) bool {
	ks := from
	for ks > 0 && (s[ks-1] == ' ' || s[ks-1] == '\t') {
		ks--
	}
	if ks == 0 || s[ks-1] != ':' {
		return false
	}
	ks--
	for ks > 0 && (s[ks-1] == ' ' || s[ks-1] == '\t') {
		ks--
	}
	for ks > 0 && ((s[ks-1] >= 'a' && s[ks-1] <= 'z') || (s[ks-1] >= 'A' && s[ks-1] <= 'Z') || (s[ks-1] >= '0' && s[ks-1] <= '9') || s[ks-1] == '-' || s[ks-1] == '_' || s[ks-1] == '.') {
		ks--
	}
	if ks == 0 || s[ks-1] != '\'' {
		return false
	}
	if ks == 1 {
		return true
	}
	p := s[ks-2]
	return p == ' ' || p == '\t' || p == '=' || p == ',' || p == '[' || p == '(' || p == '"' || p == ';'
}

func hasEscDQBeforeHeader(s string, from int) bool {
	ks := from
	for ks > 0 && (s[ks-1] == ' ' || s[ks-1] == '\t') {
		ks--
	}
	if ks == 0 || s[ks-1] != ':' {
		return false
	}
	ks--
	for ks > 0 && (s[ks-1] == ' ' || s[ks-1] == '\t') {
		ks--
	}
	for ks > 0 && ((s[ks-1] >= 'a' && s[ks-1] <= 'z') || (s[ks-1] >= 'A' && s[ks-1] <= 'Z') || (s[ks-1] >= '0' && s[ks-1] <= '9') || s[ks-1] == '-' || s[ks-1] == '_' || s[ks-1] == '.') {
		ks--
	}
	if ks < 4 || s[ks-4:ks] != `\\\"` {
		return false
	}
	if ks == 4 {
		return true
	}
	p := s[ks-5]
	return p == ' ' || p == '\t' || p == '=' || p == ',' || p == '[' || p == '(' || p == '"' || p == ';'
}

func (h *headerQuoteScanner) singleQuoteClosesInEscapedJSON(k, lineEnd int) bool {
	if h == nil || k >= lineEnd {
		return false
	}
	s := h.s
	if innerEnd := h.doubleQuoteCloseJSONWithin(k+1, lineEnd, true); innerEnd >= k+3 && s[innerEnd-2:innerEnd] == `\"` {
		if jsonCloseBoundaryOK(s, innerEnd, lineEnd, true, true) {
			if endPos, unclosed := h.joinedHeaderArgEnd(k+1, innerEnd-2, "", false, true); unclosed == "" && endPos == k+1 {
				return true
			}
		}
	}
	return false
}

func (h *headerQuoteScanner) escDQCloseTargetInEscapedJSON(after, lineEnd int) (int, bool) {
	if h == nil || after > lineEnd {
		return 0, false
	}
	s := h.s
	if innerEnd := h.doubleQuoteCloseJSONWithin(after, lineEnd, true); innerEnd >= after+2 && s[innerEnd-2:innerEnd] == `\"` {
		if jsonCloseBoundaryOK(s, innerEnd, lineEnd, true, true) {
			if endPos, unclosed := h.joinedHeaderArgEnd(after, innerEnd-2, "", false, true); unclosed == "" {
				return endPos, true
			}
		}
	}
	return 0, false
}

// jsonWrappedInnerHeaderEnd scans s[from:lineEnd] for an inner escaped-quoted
// header closing inside an outer JSON string and returns its end offset and
// closed quote suffix.
func (h *headerQuoteScanner) jsonWrappedInnerHeaderEnd(from, lineEnd int, inSingleInJSON, inEscDQInInnerJSON bool) (int, string, bool) {
	if h == nil || from >= lineEnd {
		return 0, "", false
	}
	s := h.s
	inner := `\"`
	for k := from; k < lineEnd; {
		if inner == "" {
			return h.jsonWrappedInnerHeaderEndAt(k, lineEnd, `"`, `\"`)
		}
		c := s[k]
		if c == '\\' {
			bs := 0
			for k+bs < lineEnd && s[k+bs] == '\\' {
				bs++
			}
			if k+bs < lineEnd && (s[k+bs] == '"' || s[k+bs] == '\'') {
				if bs%2 == 1 {
					if bs%4 == 1 && s[k+bs] == '"' {
						inner = ""
					} else if inEscDQInInnerJSON && bs%4 == 3 && s[k+bs] == '"' {
						if endPos, ok := h.escDQCloseTargetInEscapedJSON(k+bs+1, lineEnd); ok {
							if endPos == k+bs+1 {
								return k, "", true
							}
							return endPos, s[k : k+bs+1], true
						}
					}
					k += bs + 1
				} else {
					k += bs
				}
			} else {
				k += bs
			}
			continue
		}
		if inSingleInJSON && c == '\'' && h.singleQuoteClosesInEscapedJSON(k, lineEnd) {
			return k, "", true
		}
		if c == '"' {
			return 0, "", false
		}
		k++
	}
	if inner == "" {
		return h.jsonWrappedInnerHeaderEndAt(lineEnd, lineEnd, `"`, `\"`)
	}
	return 0, "", false
}

// joinedHeaderArgEnd advances k across any bare or quoted shell-word fragments
// on the header's line (up to lineEnd) that immediately continue a quoted
// header argument after its apparent closing quote, and returns the end offset
// of the joined argument along with any quote that remained unclosed at
// lineEnd.
func (h *headerQuoteScanner) joinedHeaderArgEnd(k, lineEnd int, entryOuter string, inJSON, jsonEscaped bool) (int, string) {
	if h == nil || k >= lineEnd {
		return k, ""
	}
	s := h.s
	stopInnerJSON := jsonEscaped && entryOuter != "" && h.curInnerJSONKind != jsonContainerNone
	stopJSON := (inJSON && !jsonEscaped) || stopInnerJSON
	substTop := h.curSubstTop
	jsonKind := h.curOuterJSONKind
	substEscJSON := jsonEscaped
	if jsonEscaped && entryOuter == "" {
		substTop = h.curInnerSubstTop
		jsonKind = jsonContainerNone
		substEscJSON = false
	}
	if h.curSubstTop == nil && h.curInnerSubstTop == nil && h.curInnerJSONKind == jsonContainerNone && h.joinedLineEnd == lineEnd && h.joinedEntryOuter == entryOuter && h.joinedInJSON == inJSON && h.joinedEscJSON == jsonEscaped && k >= h.joinedFrom && k <= h.joinedEnd {
		return h.joinedEnd, h.joinedUnclosed
	}
	escJSON := inJSON || jsonEscaped
	start := k
	if substTop != nil {
		substNodes := make([]int, 0, 4)
		for node := substTop; node != nil; node = node.prev {
			substNodes = append(substNodes, node.val)
		}
		for idx := len(substNodes) - 1; idx >= 0; idx-- {
			closeAt := substNodes[idx]
			if closeAt >= k && closeAt <= lineEnd && !cookieProseCloseBoundaryOK(s, closeAt, lineEnd, jsonKind, substEscJSON) {
				k = closeAt
				break
			}
		}
		if k >= lineEnd {
			return lineEnd, ""
		}
	}
	cookieOpen := h.isHeaderValueOpenAt(start, escJSON)
	hasOpenSubshell := h.hasOpenSubshellBefore(start, escJSON)
	if start > 0 && s[start-1] == '\'' && (isWordApostropheAt(s, start-1) || isTrailingDigitApostropheAt(s, start-1)) {
		if parenPos := skipHorizDelimSpace(s, k); parenPos < lineEnd && s[parenPos] == ')' {
			if substClosesAtParen(substTop, parenPos) {
				if !parenCloseSuffixOK(s, parenPos, lineEnd, inJSON, jsonEscaped) {
					return lineEnd, ""
				}
			} else if !h.parenClosesCleanlyAt(start, parenPos, lineEnd, inJSON, jsonEscaped) {
				return lineEnd, ""
			}
		}
	}
	if (s[k] == '.' || s[k] == '!' || s[k] == '?') && (k+1 >= lineEnd || func() bool { r, _ := utf8.DecodeRuneInString(s[k+1:]); return isDelimSpace(r) }()) {
		return k, ""
	}
	parenDepth := 0
	jsonSpanActive := false
	unclosedParenNoCloseUntil := -1
	var matchedParenSet map[int]bool
	matchedParenUntil := -1
	quoteCloseWithin := func(from, limit int, outer, inner string) int {
		switch {
		case outer == `"` && inner == "":
			return h.doubleQuoteCloseWithin(from, limit)
		case outer == "" && inner == `\"`:
			return h.doubleQuoteCloseJSONWithin(from, limit, false)
		default:
			return quoteCloseScan(s[:limit], from, outer, inner)
		}
	}
	hasBalancedCloseParen := func(openPos int) bool {
		if openPos < unclosedParenNoCloseUntil {
			return false
		}
		if openPos < matchedParenUntil {
			return matchedParenSet[openPos]
		}
		depth := 1
		closesSeen := 0
		segEnd := lineEnd
		for p := openPos + 1; p < lineEnd; {
			if q := openingQuote(s, p); q != "" {
				if q == "'" && isWordApostropheAt(s, p) {
					p++
					continue
				}
				var qOuter, qInner string
				if len(q) == 1 {
					qOuter = q
				} else {
					qInner = q
				}
				closePos := quoteCloseWithin(p+len(q), lineEnd, qOuter, qInner)
				if closePos < 0 {
					segEnd = p
					break
				}
				p = closePos
				continue
			}
			c := s[p]
			if c == '\\' {
				if p+1 < lineEnd {
					p += 2
				} else {
					p++
				}
				continue
			}
			if strings.IndexByte(";&|<>", c) >= 0 || (inJSON && !jsonEscaped && (c == ',' || c == '}' || c == ']')) {
				segEnd = p
				break
			}
			switch c {
			case '(':
				depth++
			case ')':
				closesSeen++
				depth--
				if depth == 0 {
					return true
				}
			}
			p++
		}
		if closesSeen == 0 {
			unclosedParenNoCloseUntil = segEnd
			return false
		}
		matched := make(map[int]bool, closesSeen)
		var stack []int
		for p := openPos; p < segEnd; {
			if q := openingQuote(s, p); q != "" {
				if q == "'" && isWordApostropheAt(s, p) {
					p++
					continue
				}
				var qOuter, qInner string
				if len(q) == 1 {
					qOuter = q
				} else {
					qInner = q
				}
				closePos := quoteCloseWithin(p+len(q), segEnd, qOuter, qInner)
				if closePos < 0 {
					break
				}
				p = closePos
				continue
			}
			c := s[p]
			if c == '\\' {
				if p+1 < segEnd {
					p += 2
				} else {
					p++
				}
				continue
			}
			switch {
			case c == '(':
				stack = append(stack, p)
			case c == ')' && len(stack) > 0:
				top := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				matched[top] = true
			}
			p++
		}
		matchedParenSet = matched
		matchedParenUntil = segEnd
		return matched[openPos]
	}
	for k < lineEnd {
		if wordBoundaryAt(s, k) && startsAuthHeaderPrefix(s[k:lineEnd]) {
			break
		}
		if s[k] == '\\' && k+2 < lineEnd && s[k+1] == '\'' && s[k+2] == '\'' && entryOuter != "" && !escJSON {
			k += 2
			continue
		}
		if q := openingQuote(s, k); q != "" {
			if inJSON && !jsonEscaped && q == `"` && k+1 < lineEnd {
				c1 := s[k+1]
				if (c1 == ',' && jsonCommaBreakOK(s, skipHorizDelimSpace(s[:lineEnd], k+2), lineEnd, h.curOuterJSONKind, false)) ||
					((c1 == '}' || c1 == ']') && jsonCloseBoundaryOK(s, k+2, lineEnd, false, true)) {
					break
				}
			}
			var qOuter, qInner string
			if len(q) == 1 {
				qOuter = q
			} else {
				qInner = q
			}
			closePos := quoteCloseWithin(k+len(q), lineEnd, qOuter, qInner)
			if closePos < 0 {
				if q == "'" && isWordApostropheAt(s, k) && !h.singleQuoteClosesAfterLine(lineEnd) {
					if isTrailingApostrophe(s, k) {
						if h.apostropheTailStopsOnLine(start, k, lineEnd, inJSON, jsonEscaped) && (inJSON || jsonEscaped || !h.hasUnclosedQuoteOnNextLine(lineEnd)) {
							k++
							continue
						}
						if h.joinedQuoteCloseAt(lineEnd, "'", "") >= 0 && !h.apostropheTailStopsOnLine(start, k, lineEnd, inJSON, jsonEscaped) && !h.hasUnclosedQuoteOnNextLine(lineEnd) {
							return lineEnd, ""
						}
					} else {
						if h.apostropheTailStopsOnLine(start, k, lineEnd, inJSON, jsonEscaped) || isShortLowerProseTail(s, k, lineEnd) {
							k++
							continue
						}
						if inJSON && !jsonEscaped {
							if after := k + len(q); after < lineEnd && (((s[after] == '}' || s[after] == ']') && jsonCloseBoundaryOK(s, after, lineEnd, false, true)) || (s[after] == ',' && (after+1 >= lineEnd || s[after+1] == ' ' || s[after+1] == '\t'))) {
								break
							}
						}
						return lineEnd, ""
					}
				}
				if inJSON && !jsonEscaped {
					if after := k + len(q); after < lineEnd && (((s[after] == '}' || s[after] == ']') && jsonCloseBoundaryOK(s, after, lineEnd, false, true)) || (s[after] == ',' && (after+1 >= lineEnd || s[after+1] == ' ' || s[after+1] == '\t'))) {
						break
					}
				}
				h.joinedFrom, h.joinedLineEnd, h.joinedEnd, h.joinedEntryOuter, h.joinedUnclosed, h.joinedInJSON, h.joinedEscJSON = start, lineEnd, lineEnd, entryOuter, q, inJSON, jsonEscaped
				return lineEnd, q
			}
			if q == "'" && h.isCandidateHeaderApostrophe(k, closePos, lineEnd, escJSON) {
				k++
				continue
			}
			if q == "'" && (isWordApostropheAt(s, k) || isTrailingDigitApostropheAt(s, k)) {
				if startsAuthHeaderPrefix(s[closePos:lineEnd]) {
					return lineEnd, ""
				}
				if isTrailingApostrophe(s, k) {
					if parenPos := skipHorizDelimSpace(s, k+1); parenPos < closePos-1 && s[parenPos] == ')' {
						if h.parenClosesCleanlyAt(start, parenPos, lineEnd, inJSON, jsonEscaped) {
							k++
							continue
						}
						return lineEnd, ""
					}
				}
			}
			if spanEndsWithAuthHeaderPrefix(s, k+len(q), closePos-len(q)) {
				nextClose := quoteCloseWithin(closePos, lineEnd, qOuter, qInner)
				if nextClose >= 0 {
					if h.isShellBalancedFrom(nextClose, lineEnd, escJSON) {
						k = nextClose
						continue
					}
					return lineEnd, ""
				} else if lineEnd < len(s) && (s[lineEnd] == '\n' || s[lineEnd] == '\r') && h.joinedQuoteCloses(lineEnd, qOuter, qInner) {
					h.joinedFrom, h.joinedLineEnd, h.joinedEnd, h.joinedEntryOuter, h.joinedUnclosed, h.joinedInJSON, h.joinedEscJSON = start, lineEnd, lineEnd, entryOuter, q, inJSON, jsonEscaped
					return lineEnd, q
				} else {
					return lineEnd, ""
				}
			}
			if startsAuthHeaderPrefix(s[closePos:lineEnd]) {
				k = closePos
				break
			}
			k = closePos
			continue
		}
		if s[k] == '$' && k+1 < lineEnd && (s[k+1] == '(' || s[k+1] == '{' || s[k+1] == '\'') {
			var nextK int
			switch s[k+1] {
			case '(':
				nextK = h.cmdSubstCloseWithin(k+2, lineEnd, escJSON)
			case '{':
				nextK = h.paramExpCloseWithin(k+2, false, lineEnd, escJSON)
			default:
				nextK = scanANSICQuotedClose(s[:lineEnd], k+2, escJSON)
			}
			if nextK >= 0 {
				k = nextK
				continue
			}
		}
		if s[k] == '`' {
			if nextK := h.backtickCloseWithin(k+1, lineEnd, false, escJSON); nextK >= 0 {
				k = nextK
				continue
			}
		}
		if s[k] == '\\' {
			if k+1 < lineEnd {
				k += 2
			} else {
				k++
			}
			continue
		}
		if stopJSON && (s[k] == '}' || s[k] == ']') {
			if jsonCloseBoundaryOK(s, k+1, lineEnd, stopInnerJSON, true) {
				break
			}
		}
		if k == start && inJSON && !jsonEscaped && s[k] == ':' {
			break
		}
		if stopJSON && s[k] == ',' {
			after := skipHorizDelimSpace(s[:lineEnd], k+1)
			commaKind := h.curOuterJSONKind
			if stopInnerJSON {
				commaKind = h.curInnerJSONKind
			}
			if jsonCommaBreakOK(s, after, lineEnd, commaKind, stopInnerJSON) {
				break
			}
			jsonSpanActive = true
			if after < lineEnd {
				k = after
				continue
			}
		}
		if s[k] == '(' {
			if parenDepth > 0 || hasBalancedCloseParen(k) {
				parenDepth++
			}
			k++
			continue
		}
		if s[k] == ')' {
			if parenDepth > 0 {
				parenDepth--
				k++
				continue
			}
			hasSub := substClosesAtParen(substTop, k) || hasOpenSubshell
			if hasSub && parenCloseSuffixOK(s, k, lineEnd, inJSON, jsonEscaped) {
				break
			}
			if !hasSub || (k+1 < lineEnd && (s[k+1] == ' ' || s[k+1] == '\t')) {
				return lineEnd, ""
			}
			k++
			continue
		}
		r, w := utf8.DecodeRuneInString(s[k:])
		if parenDepth == 0 {
			if strings.IndexByte("&|<>", s[k]) >= 0 {
				break
			}
			if s[k] == ';' {
				if !cookieOpen || !cookieSemiContinues(s, k+1, lineEnd, inJSON && !jsonEscaped) {
					break
				}
				k = skipHorizDelimSpace(s[:lineEnd], k+1)
				continue
			}
			if isDelimSpace(r) {
				if jsonSpanActive {
					k = skipHorizDelimSpace(s[:lineEnd], k+w)
					continue
				}
				if cookieOpen {
					sp := skipHorizDelimSpace(s[:lineEnd], k+w)
					if sp < lineEnd && s[sp] == ';' && cookieSemiContinues(s, sp+1, lineEnd, inJSON && !jsonEscaped) {
						k = sp
						continue
					}
				}
				break
			}
		}
		k += w
	}
	h.joinedFrom, h.joinedLineEnd, h.joinedEnd, h.joinedEntryOuter, h.joinedUnclosed, h.joinedInJSON, h.joinedEscJSON = start, lineEnd, k, entryOuter, "", inJSON, jsonEscaped
	return k, ""
}

// cookieSemiContinues reports whether text after a ';' at s[from:lineEnd]
// continues an active Cookie / Set-Cookie header with another cookie pair
// (token=...) or standard Set-Cookie attribute flag rather than starting a
// separate shell command such as "; echo ...".
func cookieSemiContinues(s string, from, lineEnd int, jsonDelim bool) bool {
	p := skipHorizDelimSpace(s[:lineEnd], from)
	if p >= lineEnd {
		return false
	}
	start := p
	for p < lineEnd {
		c := s[p]
		if c == '=' {
			return p > start
		}
		if c == ';' || strings.IndexByte("&|<>#", c) >= 0 || (jsonDelim && (c == ',' || c == '}' || c == ']' || c == '"')) {
			if tok := s[start:p]; strings.EqualFold(tok, "httponly") || strings.EqualFold(tok, "secure") || strings.EqualFold(tok, "partitioned") {
				return true
			}
			return false
		}
		r, w := utf8.DecodeRuneInString(s[p:])
		if isDelimSpace(r) {
			if tok := s[start:p]; strings.EqualFold(tok, "httponly") || strings.EqualFold(tok, "secure") || strings.EqualFold(tok, "partitioned") {
				after := skipHorizDelimSpace(s[:lineEnd], p+w)
				return after >= lineEnd || s[after] == ';' || strings.IndexByte("&|<>#", s[after]) >= 0
			}
			return false
		}
		p += w
	}
	tok := s[start:p]
	return strings.EqualFold(tok, "httponly") || strings.EqualFold(tok, "secure") || strings.EqualFold(tok, "partitioned")
}

// lineQuoteState is the state of a scan of the lines of s for the quotes that
// are open before a header (see headerQuoteScanner.openBefore): the next byte
// to read, k, the start and the end of the line of k, the quotes outer and
// inner that are open there, where each opened and where the part of the line
// that it holds starts, whether each stays open from an earlier line, and the
// explicit outer quote that the last line break closed, if any.
type lineQuoteState struct {
	k, curLineStart, lineLim       int
	outer, inner                   string
	outerOpenPos, outerOpenEnd     int
	innerOpenPos, innerOpenEnd     int
	outerPriorLine, innerPriorLine bool
	closingExplicitOuter           byte
	braceDepth                     int
	outerJSON                      bool
	outerJSONTop                   *jsonContainerNode
	innerJSONTop                   *jsonContainerNode
	outerJSONKind                  jsonContainerKind
	innerJSONKind                  jsonContainerKind
	outerSubSQ                     bool
	innerSubSQ                     bool
	substTop                       *intListNode
	innerSubstTop                  *intListNode
}

// newLineQuoteState returns the state of a scan of s from the start of a line
// at s[start:], with no quote open, that reads s up to bound.
func newLineQuoteState(s string, start, bound int) lineQuoteState {
	st := lineQuoteState{k: start, curLineStart: start, lineLim: bound, outerOpenPos: -1, outerOpenEnd: -1, innerOpenPos: -1, innerOpenEnd: -1}
	if nl := strings.IndexAny(s[start:bound], "\r\n"); nl >= 0 {
		st.lineLim = start + nl
	}
	return st
}

// openBefore returns the quotes of the strings that the part of the line of
// s before i (or an earlier line in s when the quote stays open across the
// line break into the header at i and closes at or after valStart) opens and
// leaves open: outer, a double or a single quote, as the " before
// Authorization in "Authorization: x" is, and inner, an escaped quote (\" or
// \') that opens a string nested in outer, or in no string, as the \" before
// Authorization in {\"h\": \"Authorization: x\"} does. Inside a single-quoted
// string every byte but the closing quote is literal, as a shell reads it;
// elsewhere a backslash escapes the byte after it, and inside an escaped
// string a quote is a byte of it unless it closes outer. The lines before the
// header's line are read only if the part of that line before the header is
// whitespace or starts another header, and then from the start of s, however
// far back, with the scan of them carried over from the header before; i
// must lie after the line of the header of the call before.
func (h *headerQuoteScanner) openBefore(i, valStart int) (outer, inner string) {
	s := h.s
	var st lineQuoteState
	var lineStart int
	if h.hasCurLine && i >= h.curLine.k && i <= h.curLine.lineLim {
		lineStart = h.curLine.curLineStart
		h.curLine.scan(h.shell, h.shellMemoPtr(), s, i, h.curLine.lineLim, lineStart)
		st = h.curLine
	} else {
		lineStart = strings.LastIndexAny(s[:i], "\r\n") + 1
		lineEnd := len(s)
		if nl := strings.IndexAny(s[lineStart:], "\r\n"); nl >= 0 {
			lineEnd = lineStart + nl
		}
		if lineStart > 0 {
			// Advance the shared line state up to the preceding line break so
			// braceDepth is preserved across multiline JSON containers even
			// when the current line does not carry over an open quote.
			brk := lineStart - 1
			if brk > 0 && s[brk] == '\n' && s[brk-1] == '\r' {
				brk--
			}
			h.lines.scan(h.shell, h.shellMemoPtr(), s, brk, len(s), len(s)+1)
			if quoteRestIsHeaderOrSpace(s, lineStart, i) {
				st = h.lines
				st.lineLim = lineEnd
			} else {
				st = newLineQuoteState(s, lineStart, len(s))
				st.lineLim = lineEnd
				st.braceDepth = h.lines.braceDepth
				st.outerJSONTop = h.lines.outerJSONTop
				st.substTop = h.lines.substTop
			}
		} else {
			st = newLineQuoteState(s, lineStart, len(s))
			st.lineLim = lineEnd
		}
		st.scan(h.shell, h.shellMemoPtr(), s, i, lineEnd, lineStart)
	}
	h.curLine = st
	h.hasCurLine = true
	h.curOuterJSON = st.outerJSON
	h.curOuterJSONKind = st.outerJSONKind
	h.curInnerJSONKind = st.innerJSONKind
	h.curOuterSubSQ = st.outerSubSQ
	h.curInnerSubSQ = st.innerSubSQ
	h.curSubstTop = st.substTop
	h.curInnerSubstTop = st.innerSubstTop
	outer, inner = st.outer, st.inner
	outerPriorLine, innerPriorLine := st.outerPriorLine, st.innerPriorLine
	if inner != "" && innerPriorLine && !quoteRestIsHeaderOrSpace(s, st.innerOpenEnd, i) {
		inner = ""
		innerPriorLine = false
		h.curInnerJSONKind = jsonContainerNone
		h.curInnerSubSQ = false
		h.curInnerSubstTop = nil
	}
	if outer != "" && outerPriorLine && inner == "" && !quoteRestIsHeaderOrSpace(s, st.outerOpenEnd, i) {
		outer = ""
		outerPriorLine = false
		h.curOuterJSON = false
		h.curOuterJSONKind = jsonContainerNone
		h.curOuterSubSQ = false
	}
	if (outerPriorLine || innerPriorLine) && !h.quoteCloses(valStart, outer, inner) {
		// A quote from a line before that never closes holds no string, so
		// the header's line is read again from its start with no quote open,
		// as the quote before Authorization in
		// '<LF>Cookie: x; curl -H "Authorization: y<LF>z" is read. The state
		// read again replaces the one carried into the line, so each line is
		// read again at most once.
		lineLim := st.lineLim
		st = newLineQuoteState(s, lineStart, len(s))
		st.lineLim = lineLim
		st.braceDepth = h.lines.braceDepth
		st.outerJSONTop = h.lines.outerJSONTop
		st.substTop = h.lines.substTop
		st.scan(h.shell, h.shellMemoPtr(), s, i, lineLim, lineStart)
		h.curLine = st
		h.curOuterJSON = st.outerJSON
		h.curOuterJSONKind = st.outerJSONKind
		h.curInnerJSONKind = st.innerJSONKind
		h.curOuterSubSQ = st.outerSubSQ
		h.curInnerSubSQ = st.innerSubSQ
		h.curSubstTop = st.substTop
		h.curInnerSubstTop = st.innerSubstTop
		outer, inner = st.outer, st.inner
	}
	return outer, inner
}

func isJSONStringOpener(s string, lineStart, k int) bool {
	if k >= len(s) || s[k] != '"' {
		return false
	}
	p := k - 1
	for p >= lineStart && (s[p] == ' ' || s[p] == '\t') {
		p--
	}
	if p < lineStart {
		return true
	}
	return s[p] == ':' || s[p] == '[' || s[p] == ',' || s[p] == '{'
}

// scan advances st over s up to to, reading s up to bound, for the header
// whose line starts at hdrLineStart, or, if hdrLineStart is past the end of
// s, over lines before the line of a header only.
func (st *lineQuoteState) scan(sh *shellAssignScanner, memos *shellCloseMemo, s string, to, bound, hdrLineStart int) {
	k, curLineStart, lineLim := st.k, st.curLineStart, st.lineLim
	outer, inner := st.outer, st.inner
	outerOpenPos, outerOpenEnd := st.outerOpenPos, st.outerOpenEnd
	innerOpenPos, innerOpenEnd := st.innerOpenPos, st.innerOpenEnd
	outerPriorLine, innerPriorLine := st.outerPriorLine, st.innerPriorLine
	closingExplicitOuter := st.closingExplicitOuter
	braceDepth, outerJSON := st.braceDepth, st.outerJSON
	outerJSONTop, innerJSONTop := st.outerJSONTop, st.innerJSONTop
	outerJSONKind, innerJSONKind := st.outerJSONKind, st.innerJSONKind
	outerSubSQ, innerSubSQ := st.outerSubSQ, st.innerSubSQ
	substTop, innerSubstTop := st.substTop, st.innerSubstTop
	for k < to {
		for substTop != nil && k >= substTop.val {
			substTop = substTop.prev
		}
		for innerSubstTop != nil && k >= innerSubstTop.val {
			innerSubstTop = innerSubstTop.prev
		}
		c := s[k]
		if c == '\r' || c == '\n' {
			if c == '\r' && k+1 < bound && s[k+1] == '\n' {
				k += 2
			} else {
				k++
			}
			prevLineStart := curLineStart
			curLineStart = k
			lineLim = bound
			if nl := strings.IndexAny(s[curLineStart:bound], "\r\n"); nl >= 0 {
				lineLim = curLineStart + nl
			}
			validInner := inner != "" && innerOpenEnd >= 0 &&
				(innerPriorLine || isExplicitLineQuoteOpener(sh, s, prevLineStart, innerOpenPos))
			if validInner {
				innerOpenEnd = k
				innerPriorLine = true
			} else {
				inner = ""
				innerOpenPos, innerOpenEnd = -1, -1
				innerPriorLine = false
				innerJSONTop = nil
				innerJSONKind = jsonContainerNone
				innerSubSQ = false
				innerSubstTop = nil
			}
			explicitOuter := outer != "" && outerOpenEnd >= 0 &&
				(outerPriorLine || isExplicitLineQuoteOpener(sh, s, prevLineStart, outerOpenPos))
			if explicitOuter {
				if !validInner {
					outerOpenEnd = k
				}
				outerPriorLine = true
				closingExplicitOuter = 0
			} else {
				outer = ""
				outerOpenPos, outerOpenEnd = -1, -1
				outerPriorLine = false
				outerJSON = false
				outerJSONKind = jsonContainerNone
				outerSubSQ = false
			}
			continue
		}
		switch {
		case outer == "'":
			if c == '\'' {
				outer = ""
				outerOpenPos, outerOpenEnd = -1, -1
				outerPriorLine = false
				outerJSON = false
				outerJSONKind = jsonContainerNone
				outerSubSQ = false
			}
			k++
		case c == '\\':
			bs := 0
			for k+bs < bound && s[k+bs] == '\\' {
				bs++
			}
			if outer == `"` && inner == "" && outerSubSQ && bs%2 == 1 && k+bs < bound && s[k+bs] == '\'' {
				k += bs
				continue
			}
			if k+bs < bound && (s[k+bs] == '"' || s[k+bs] == '\'') {
				if bs%2 == 1 {
					if bs%4 == 1 {
						q := s[k+bs-1 : k+bs+1]
						switch inner {
						case q:
							inner = ""
							innerOpenPos, innerOpenEnd = -1, -1
							innerPriorLine = false
							innerJSONKind = jsonContainerNone
							innerSubSQ = false
							innerSubstTop = nil
						case "":
							if outer == "" && !canOpenPriorLineQuote(sh, s, curLineStart, k, lineLim, k < hdrLineStart) && !inShellAssignment(sh, s, k) {
								break
							}
							inner = q
							innerOpenPos = k
							innerOpenEnd = k + len(q)
							innerPriorLine = false
							if q == `\"` && outer == `"` && !outerSubSQ && outerOpenEnd >= 0 {
								innerJSONKind = classifyQuoteJSONContainer(s, k, outerOpenEnd, true, innerJSONTop)
							} else {
								innerJSONKind = jsonContainerNone
							}
							innerSubSQ = false
							innerSubstTop = nil
						}
					}
					k += bs + 1
				} else {
					k += bs
				}
			} else if bs%2 == 1 && k+bs < bound && s[k+bs] != '\r' && s[k+bs] != '\n' {
				k += bs + 1
			} else {
				k += bs
			}
		case inner != "":
			if outer != "" && c == outer[0] {
				outer, inner = "", ""
				outerOpenPos, outerOpenEnd = -1, -1
				innerOpenPos, innerOpenEnd = -1, -1
				outerPriorLine, innerPriorLine = false, false
				outerJSON = false
				outerJSONKind = jsonContainerNone
				innerJSONKind = jsonContainerNone
				outerSubSQ = false
				innerSubSQ = false
				innerJSONTop = nil
				innerSubstTop = nil
			} else if inner == `\"` {
				switch c {
				case '\'':
					if innerSubSQ {
						innerSubSQ = false
					} else if (!isMidWordApostrophe(s, k) || opensPrefixedString(s, k)) && !isTrailingApostrophe(s, k) || inShellAssignment(sh, s, k) {
						innerSubSQ = true
					}
				case '$':
					if k+1 < bound && s[k+1] == '(' {
						if endPos := shellScanClose(memos, s, shellFrameCS, k+2, false, true, outer == `"`); endPos >= 0 {
							innerSubstTop = &intListNode{val: endPos, prev: innerSubstTop}
						}
					}
				case '`':
					if endPos := shellScanClose(memos, s, shellFrameBT, k+1, true, true, outer == `"`); endPos >= 0 {
						innerSubstTop = &intListNode{val: endPos, prev: innerSubstTop}
					}
				}
			}
			k++
		case c == '"' || c == '\'':
			if outer == "" {
				if c == '\'' && (isMidWordApostrophe(s, k) && !opensPrefixedString(s, k) || isTrailingApostrophe(s, k)) && !inShellAssignment(sh, s, k) {
					k++
					continue
				}
				if c == closingExplicitOuter && k < hdrLineStart {
					closingExplicitOuter = 0
					if skipHorizDelimSpace(s, curLineStart) == k && skipHorizDelimSpace(s, k+1) == lineLim {
						k++
						continue
					}
				}
				if !canOpenPriorLineQuote(sh, s, curLineStart, k, lineLim, k < hdrLineStart) && !inShellAssignment(sh, s, k) {
					k++
					continue
				}
				closingExplicitOuter = 0
				outer = s[k : k+1]
				outerOpenPos = k
				outerOpenEnd = k + 1
				outerPriorLine = false
				outerSubSQ = false
				innerJSONTop = nil
				innerJSONKind = jsonContainerNone
				innerSubSQ = false
				innerSubstTop = nil
				if outer == `"` {
					outerJSONKind = classifyQuoteJSONContainer(s, k, 0, false, outerJSONTop)
				} else {
					outerJSONKind = jsonContainerNone
				}
				outerJSON = braceDepth > 0 && isJSONStringOpener(s, curLineStart, k)
			} else if c == outer[0] {
				outer = ""
				outerOpenPos, outerOpenEnd = -1, -1
				outerPriorLine = false
				outerJSON = false
				outerJSONKind = jsonContainerNone
				outerSubSQ = false
				innerJSONTop = nil
				innerJSONKind = jsonContainerNone
				innerSubSQ = false
				innerSubstTop = nil
			} else if outer == `"` && c == '\'' {
				if outerSubSQ {
					outerSubSQ = false
				} else if (!isMidWordApostrophe(s, k) || opensPrefixedString(s, k)) && !isTrailingApostrophe(s, k) || inShellAssignment(sh, s, k) {
					outerSubSQ = true
				}
			}
			k++
		default:
			if outer == "" && inner == "" {
				switch c {
				case '{', '[':
					braceDepth++
					outerJSONTop = &jsonContainerNode{
						entry: jsonContainerEntry{openChar: c, pos: k, isJSON: isJSONContainerOpener(s, k)},
						prev:  outerJSONTop,
					}
				case '}', ']':
					if braceDepth > 0 {
						braceDepth--
					}
					if outerJSONTop != nil && jsonContainerMatches(outerJSONTop.entry.openChar, c) {
						outerJSONTop = outerJSONTop.prev
					}
				case '$':
					if k+1 < bound && s[k+1] == '(' {
						if endPos := shellScanClose(memos, s, shellFrameCS, k+2, false, false, false); endPos >= 0 {
							substTop = &intListNode{val: endPos, prev: substTop}
						}
					}
				case '`':
					if endPos := shellScanClose(memos, s, shellFrameBT, k+1, false, false, false); endPos >= 0 {
						substTop = &intListNode{val: endPos, prev: substTop}
					}
				}
			} else if outer == `"` && inner == "" {
				if !outerSubSQ {
					if c == '{' || c == '[' {
						innerJSONTop = &jsonContainerNode{
							entry: jsonContainerEntry{openChar: c, pos: k, isJSON: isJSONContainerOpener(s, k)},
							prev:  innerJSONTop,
						}
					} else if (c == '}' || c == ']') && innerJSONTop != nil && jsonContainerMatches(innerJSONTop.entry.openChar, c) {
						innerJSONTop = innerJSONTop.prev
					}
				}
				if c == '$' && k+1 < bound && s[k+1] == '(' {
					escJSON := outerJSONKind != jsonContainerNone
					if endPos := shellScanClose(memos, s, shellFrameCS, k+2, false, escJSON, false); endPos >= 0 {
						substTop = &intListNode{val: endPos, prev: substTop}
					}
				} else if c == '`' {
					escJSON := outerJSONKind != jsonContainerNone
					if endPos := shellScanClose(memos, s, shellFrameBT, k+1, true, escJSON, false); endPos >= 0 {
						substTop = &intListNode{val: endPos, prev: substTop}
					}
				}
			}
			k++
		}
	}
	for substTop != nil && k >= substTop.val {
		substTop = substTop.prev
	}
	for innerSubstTop != nil && k >= innerSubstTop.val {
		innerSubstTop = innerSubstTop.prev
	}
	*st = lineQuoteState{
		k: k, curLineStart: curLineStart, lineLim: lineLim,
		outer: outer, inner: inner,
		outerOpenPos: outerOpenPos, outerOpenEnd: outerOpenEnd,
		innerOpenPos: innerOpenPos, innerOpenEnd: innerOpenEnd,
		outerPriorLine: outerPriorLine, innerPriorLine: innerPriorLine,
		closingExplicitOuter: closingExplicitOuter,
		braceDepth:           braceDepth,
		outerJSON:            outerJSON,
		outerJSONTop:         outerJSONTop,
		innerJSONTop:         innerJSONTop,
		outerJSONKind:        outerJSONKind,
		innerJSONKind:        innerJSONKind,
		outerSubSQ:           outerSubSQ,
		innerSubSQ:           innerSubSQ,
		substTop:             substTop,
		innerSubstTop:        innerSubstTop,
	}
}

// quoteCloseMemo holds a scan of quoteCloseScan from from for one kind of
// quote (see headerQuoteScanner.quoteCloses): where the quote closes, or -1,
// the start of the last line in s[from+1:closePos+1] that starts another
// header, or -1 (see crossesHeaderLine), and whether the line on which the
// quote closes rejects the close if the quote crosses such a line (see
// closeLineRejects).
type quoteCloseMemo struct {
	ok                          bool
	from, closePos, lastHdrLine int
	rejects                     bool
}

// quoteCloses reports what outerQuoteCloses reports for s[from:] and the
// quotes outer and inner that stay open into a header from a line before it
// (see openBefore), where from is the start of a header value, after its
// colon and whitespace, so that no backslash before it escapes it. Where such
// a quote closes depends on from and its kind alone: a single quote closes at
// the next single quote, a double quote at the next one that a backslash does
// not escape, whatever escaped quote is open inside it, and an escaped quote
// open in no other string at the next escaped quote of its kind. The value of
// each later header starts after the one of the header before, so the scan
// that found the close for a header holds for every later header whose value
// starts before that close, and is reused for them. The scans for a kind of
// quote thus cover s at most once, however many headers a quote stays open
// into, as the first quote of '<LF>Authorization: a<LF>Authorization: b<LF>...
// does.
func (h *headerQuoteScanner) quoteCloses(from int, outer, inner string) bool {
	m := closeSlot(&h.closes, outer, inner)
	if m == nil {
		return outerQuoteCloses(h.shell, h.s, from, outer, inner)
	}
	s := h.s
	if !m.ok || from < m.from || m.closePos >= 0 && from >= m.closePos {
		closePos := quoteCloseScan(s, from, outer, inner)
		*m = quoteCloseMemo{ok: true, from: from, closePos: closePos, lastHdrLine: -1}
		if closePos >= 0 {
			for p := closePos - 1; p >= from; p-- {
				if (s[p] == '\n' || s[p] == '\r' && (p+1 >= len(s) || s[p+1] != '\n')) && startsHeaderLine(s, p+1) {
					m.lastHdrLine = p + 1
					break
				}
			}
			checkQuote := outer
			if checkQuote == "" {
				checkQuote = inner
			}
			if checkQuote != "" {
				m.rejects = closeLineRejects(h.shell, s, closePos, checkQuote)
			}
		}
	}
	return m.closePos >= 0 && !(m.lastHdrLine > from && m.rejects)
}

// placeholderQuoteCloses reports whether outer and inner close after a header
// line whose first-line value is already redactedValue, without pairing with
// an opening quote on a later line (see closeLineRejects). A single quote
// closes there even after a backslash, as the last quote of
// curl -H 'Authorization: [REDACTED]<LF>SECRET\'<LF>echo ok does, since a
// backslash in a single-quoted shell string is a byte of the string.
func (h *headerQuoteScanner) placeholderQuoteCloses(from int, outer, inner string) bool {
	if inner != "" {
		outer = ""
	}
	if !h.quoteCloses(from, outer, inner) {
		return false
	}
	if m := closeSlot(&h.closes, outer, inner); m != nil {
		return !m.rejects
	}
	return false
}

// joinedQuoteCloseAt returns quoteCloseScan(h.s, from, outer, inner) cached
// in joinedCloses so multiline joined-quote lookahead scales linearly without
// mutating proseCloses.
func (h *headerQuoteScanner) joinedQuoteCloseAt(from int, outer, inner string) int {
	if outer == `"` && inner == "" {
		return h.doubleQuoteClosePos(from)
	}
	if outer == "" && inner == `\"` {
		return h.doubleQuoteCloseJSONPos(from)
	}
	m := closeSlot(&h.joinedCloses, outer, inner)
	if m == nil {
		return quoteCloseScan(h.s, from, outer, inner)
	}
	if !m.ok || from < m.from || (m.closePos >= 0 && from >= m.closePos) {
		*m = quoteCloseMemo{ok: true, from: from, closePos: quoteCloseScan(h.s, from, outer, inner), lastHdrLine: -1}
	}
	return m.closePos
}

// chainMode selects how chainClosesFrom handles unquoted apostrophes on a
// closing line when checking whether the remainder of a quote chain is
// shell-balanced.
type chainMode int

const (
	chainStrict chainMode = iota
	chainSkipContractions
	chainSkipPossessives
)

// isProsePluralPossessive reports whether line[k] is a trailing plural
// possessive apostrophe after a prose word of at least three letters ending in
// 's' or 'S' (such as "users'" or "James'") that starts after whitespace or an
// opening/clause delimiter inside line.
func isProsePluralPossessive(line string, k int) bool {
	if k < 3 || (line[k-1] != 's' && line[k-1] != 'S') || !isTrailingApostrophe(line, k) {
		return false
	}
	start, letters := k, 0
	for start > 0 {
		r, size := utf8.DecodeLastRuneInString(line[:start])
		if !unicode.IsLetter(r) {
			break
		}
		start -= size
		letters++
	}
	if start == 0 || letters < 3 {
		return false
	}
	r, _ := utf8.DecodeLastRuneInString(line[:start])
	return isDelimSpace(r) || strings.ContainsRune("([{,;", r)
}

// closesProseLineWord reports whether h.s[pos] is a trailing apostrophe closing
// a letter-only word on a line whose preceding text contains only plain
// alphanumeric tokens and horizontal whitespace (no key-value delimiters or
// quotes). Precomputed line prefixes and memoization keep repeated checks
// linear across the scan.
func (h *headerQuoteScanner) closesProseLineWord(pos int) bool {
	if h == nil || pos <= 0 || pos >= len(h.s) {
		return false
	}
	if v, ok := h.proseCloseMemo[pos]; ok {
		return v
	}
	stamp := func(v bool) bool {
		if h.proseCloseMemo == nil {
			h.proseCloseMemo = make(map[int]bool)
		}
		h.proseCloseMemo[pos] = v
		return v
	}
	s := h.s
	if !isTrailingApostrophe(s, pos) {
		return stamp(false)
	}
	start := pos
	for start > 0 {
		r, size := utf8.DecodeLastRuneInString(s[:start])
		if !unicode.IsLetter(r) {
			break
		}
		start -= size
	}
	if start > 0 && s[start-1] != '\n' && s[start-1] != '\r' && s[start-1] != ' ' && s[start-1] != '\t' {
		return stamp(false)
	}
	if !h.plainPrefixReady {
		h.plainPrefixBits = make([]uint64, (len(s)+64)/64)
		h.plainPrefixBits[0] |= 1
		lineOK := true
		for i := 0; i < len(s); {
			r, w := utf8.DecodeRuneInString(s[i:])
			end := i + w
			if r == '\r' || r == '\n' {
				lineOK = true
			} else if lineOK && !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' && r != '-' && r != ' ' && r != '\t' {
				lineOK = false
			}
			if lineOK {
				h.plainPrefixBits[end>>6] |= 1 << (end & 63)
			}
			i = end
		}
		h.plainPrefixReady = true
	}
	return stamp((h.plainPrefixBits[start>>6]>>(start&63))&1 != 0)
}

// isHeaderValueOpenOnLine reports whether a closing quote at closePos falls
// inside an active non-prose Cookie / Set-Cookie header value on the same line.
func isHeaderValueOpenOnLine(s string, lineStart, closePos int, escaped bool) bool {
	if closePos <= lineStart || closePos >= len(s) {
		return false
	}
	// If the character right after the closing quote is already whitespace or a
	// shell command separator, the quote closed a value rather than opening one.
	if s[closePos] == ' ' || s[closePos] == '\t' || s[closePos] == ';' || s[closePos] == '&' || s[closePos] == '|' {
		return false
	}
	qPos := closePos - 1
	if escaped && qPos > lineStart && s[qPos-1] == '\\' {
		qPos--
	}
	if qPos <= lineStart {
		return false
	}
	// Restrict mid-line semicolon tolerance to active non-prose Cookie and
	// Set-Cookie headers on the line, resetting if an unquoted & or |, or a
	// non-Cookie ';' continuation, starts a later shell command before qPos.
	lineLim := len(s)
	if nl := strings.IndexAny(s[qPos:], "\r\n"); nl >= 0 {
		lineLim = qPos + nl
	}
	cookieOpen := false
	for p := lineStart; p < qPos; {
		c := s[p]
		if c == '&' || c == '|' || (c == ';' && cookieOpen && !cookieSemiContinues(s, p+1, lineLim, false)) {
			cookieOpen = false
			p++
			continue
		}
		if wordBoundaryAt(s, p) && authHeaderKeys.mayStartAt(s[p:qPos], 0) {
			if loc := authHeaderPrefixAtRe.FindStringSubmatchIndex(s[p:qPos]); loc != nil {
				if isCookieHeaderKey(s[p+loc[2]:p+loc[3]]) && !cookieValueIsProse(s, p+loc[1]) {
					cookieOpen = true
				}
				p += loc[1]
				continue
			}
		}
		p++
	}
	return cookieOpen
}

func (h *headerQuoteScanner) initCookieOpenBits() {
	s := h.s
	h.cookieOpenBits = make([]uint64, (len(s)+64)/64)
	cookieOpen := false
	lineLim := -1
	for p := 0; p < len(s); {
		if p >= lineLim {
			lineLim = len(s)
			if nl := strings.IndexAny(s[p:], "\r\n"); nl >= 0 {
				lineLim = p + nl
			}
			cookieOpen = false
		}
		c := s[p]
		if c == '&' || c == '|' || c == '\r' || c == '\n' || (c == ';' && cookieOpen && !cookieSemiContinues(s, p+1, lineLim, false)) {
			cookieOpen = false
			p++
			continue
		}
		if cookieOpen {
			h.cookieOpenBits[p>>6] |= 1 << (p & 63)
		}
		if (c == 'c' || c == 'C' || c == 's' || c == 'S') && wordBoundaryAt(s, p) && authHeaderKeys.mayStartAt(s[p:lineLim], 0) {
			if loc := authHeaderPrefixAtRe.FindStringSubmatchIndex(s[p:lineLim]); loc != nil {
				wasOpen := cookieOpen
				if isCookieHeaderKey(s[p+loc[2]:p+loc[3]]) && !cookieValueIsProse(s, p+loc[1]) {
					cookieOpen = true
				}
				end := p + loc[1]
				if wasOpen {
					for k := p + 1; k < end; k++ {
						h.cookieOpenBits[k>>6] |= 1 << (k & 63)
					}
				}
				p = end
				continue
			}
		}
		p++
	}
	h.cookieOpenReady = true
}

func (h *headerQuoteScanner) isHeaderValueOpenAt(closePos int, escaped bool) bool {
	if h == nil || closePos <= 0 || closePos >= len(h.s) {
		return false
	}
	s := h.s
	if s[closePos] == ' ' || s[closePos] == '\t' || s[closePos] == ';' || s[closePos] == '&' || s[closePos] == '|' {
		return false
	}
	qPos := closePos - 1
	if escaped && qPos > 0 && s[qPos-1] == '\\' && s[qPos-1] != '\n' && s[qPos-1] != '\r' {
		qPos--
	}
	if qPos <= 0 || s[qPos-1] == '\n' || s[qPos-1] == '\r' {
		return false
	}
	if !h.cookieOpenReady {
		h.initCookieOpenBits()
	}
	return (h.cookieOpenBits[qPos>>6]>>(qPos&63))&1 != 0
}

// chainClosesFrom reports whether a joined-quote close at closePos has
// balanced shell quoting through the remainder of its chain under mode.
func (h *headerQuoteScanner) chainClosesFrom(closePos int, escaped bool, mode chainMode) bool {
	if h == nil {
		return false
	}
	s := h.s
	allowMidSemi := !h.isHeaderValueOpenAt(closePos, escaped)
	key := closePos<<5 | int(mode)<<3
	if escaped {
		key |= 1
	}
	if h.curOuterJSON {
		key |= 2
	}
	if !allowMidSemi {
		key |= 4
	}
	if v, ok := h.chainCloses[key]; ok {
		return v
	}
	var visited []int
	lastStamp := closePos
	chainFlags := key & 31
	stamp := func(res bool) bool {
		if h.chainCloses == nil {
			h.chainCloses = make(map[int]bool)
		}
		h.chainCloses[key] = res
		if allowMidSemi {
			for _, vPos := range visited {
				h.chainCloses[(vPos<<5)|chainFlags] = res
			}
		}
		return res
	}
	lineLim := len(s)
	if nl := strings.IndexAny(s[closePos:], "\r\n"); nl >= 0 {
		lineLim = closePos + nl
	}
	sep := closePos
	for sep < lineLim && (s[sep] == ' ' || s[sep] == '\t') {
		sep++
	}
	if sep < lineLim && (s[sep] == '&' || s[sep] == '|' || (s[sep] == ';' && (allowMidSemi || !cookieSemiContinues(s, sep+1, lineLim, h.curOuterJSON && !escaped)))) {
		return stamp(true)
	}
	hasChainMemo := len(h.chainCloses) > 0
	for k := closePos; k < lineLim; {
		if allowMidSemi && k > closePos && (s[k-1] == ' ' || s[k-1] == '\t') {
			if hasChainMemo {
				if v, ok := h.chainCloses[(k<<5)|chainFlags]; ok {
					return stamp(v)
				}
			}
			if k-lastStamp >= 32 {
				visited = append(visited, k)
				lastStamp = k
			}
		}
		c := s[k]
		if c == '&' || c == '|' || (c == ';' && (allowMidSemi || !cookieSemiContinues(s, k+1, lineLim, h.curOuterJSON && !escaped))) {
			return stamp(true)
		}
		if c == '#' && (k == 0 || s[k-1] == ' ' || s[k-1] == '\t' || strings.IndexByte(";&|()\r\n", s[k-1]) >= 0) {
			return stamp(true)
		}
		if wordBoundaryAt(s, k) && authHeaderKeys.mayStartAt(s[k:lineLim], 0) {
			if loc := authHeaderPrefixAtRe.FindStringSubmatchIndex(s[k:lineLim]); loc != nil {
				if !isCookieHeaderKey(s[k+loc[2]:k+loc[3]]) || !cookieValueIsProse(s, k+loc[1]) {
					return stamp(true)
				}
			}
		}
		if q := openingQuote(s[:lineLim], k); q != "" {
			if h.curOuterJSON && !escaped && q == `"` && k+1 < lineLim && (s[k+1] == ',' || s[k+1] == '}' || s[k+1] == ']') {
				return stamp(true)
			}
			if mode != chainStrict && q == "'" {
				line, pos := s[closePos:lineLim], k-closePos
				if isProseContraction(line, pos) || mode == chainSkipPossessives && isProsePluralPossessive(line, pos) {
					k += len(q)
					continue
				}
			}
			var qOuter, qInner string
			var qEsc bool
			if len(q) == 1 {
				qOuter = q
			} else {
				qInner = q
				qEsc = true
			}
			var nextClose int
			switch {
			case qOuter == `"` && qInner == "":
				nextClose = h.doubleQuoteClosePos(k + len(q))
			case qOuter == "" && qInner == `\"`:
				nextClose = h.doubleQuoteCloseJSONPos(k + len(q))
			default:
				if cPos := quoteCloseScan(s[:lineLim], k+len(q), qOuter, qInner); cPos >= 0 {
					k = cPos
					continue
				}
				nextClose = quoteCloseScan(s, lineLim, qOuter, qInner)
			}
			if nextClose < 0 {
				return stamp(false)
			}
			if nextClose <= lineLim {
				k = nextClose
				continue
			}
			nextMode := mode
			if nextMode == chainSkipPossessives {
				nextMode = chainSkipContractions
			}
			return stamp(h.chainClosesFrom(nextClose, qEsc, nextMode))
		}
		if c == '\\' {
			if !escaped {
				if k+1 < lineLim {
					k += 2
				} else {
					k++
				}
				continue
			}
			bs := 0
			for k+bs < lineLim && s[k+bs] == '\\' {
				bs++
			}
			if k+bs < lineLim && (s[k+bs] == '$' || s[k+bs] == '`') {
				if bs%4 == 2 {
					k += bs + 1
					continue
				}
				if bs%4 == 0 {
					k += bs
					continue
				}
			}
			if bs%2 == 1 && k+bs < lineLim {
				k += bs + 1
			} else {
				k += bs
			}
			continue
		}
		if c == '$' && k+1 < lineLim && (s[k+1] == '(' || s[k+1] == '{' || s[k+1] == '\'') {
			var nextClose int
			switch s[k+1] {
			case '(':
				nextClose = h.cmdSubstClosePos(k+2, escaped)
			case '{':
				nextClose = h.paramExpClosePos(k+2, false, escaped)
			default:
				nextClose = scanANSICQuotedClose(s, k+2, escaped)
			}
			if nextClose < 0 {
				return stamp(false)
			}
			if nextClose <= lineLim {
				k = nextClose
				continue
			}
			nextMode := mode
			if nextMode == chainSkipPossessives {
				nextMode = chainSkipContractions
			}
			return stamp(h.chainClosesFrom(nextClose, escaped, nextMode))
		}
		if c == '`' {
			nextClose := h.backtickClosePos(k+1, false, escaped)
			if nextClose < 0 {
				return stamp(false)
			}
			if nextClose <= lineLim {
				k = nextClose
				continue
			}
			nextMode := mode
			if nextMode == chainSkipPossessives {
				nextMode = chainSkipContractions
			}
			return stamp(h.chainClosesFrom(nextClose, escaped, nextMode))
		}
		k++
	}
	return stamp(true)
}

type shellFrameKind uint8

const (
	shellFrameDQ shellFrameKind = iota
	shellFrameCS
	shellFramePE
	shellFrameBT
)

type casePhase uint8

const (
	caseExpectWord casePhase = iota
	caseExpectIn
	casePattern
	caseBody
)

type caseState struct {
	parenDepth    int
	phase         casePhase
	sawWord       bool
	hasOptParen   bool
	allowOptParen bool
	justAfterSemi bool
}

type shellFrame struct {
	kind       shellFrameKind
	openPos    int
	inDouble   bool
	parenDepth int
	braceDepth int
	cmdPos     bool
	cases      []caseState
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

func isPOSIXShellWordBreak(c byte) bool {
	switch c {
	case ' ', '\t', '\r', '\n', '#', '\\', '\'', '"', '`', '$', ';', '&', '|', '(', ')', '<', '>':
		return true
	default:
		return false
	}
}

func isShellCommentBoundaryByte(c byte) bool {
	switch c {
	case ' ', '\t', '\r', '\n', ';', '&', '|', '(', ')', '{', '}':
		return true
	default:
		return false
	}
}

type shellCloseMemo struct {
	dq       map[int]int
	cmdSubst map[int]int
	paramExp map[int]int
	backtick map[int]int
}

func (m *shellCloseMemo) lookup(kind shellFrameKind, from int, inDouble, jsonEscaped, stopAtOuterJSONQuote bool) (int, bool) {
	if m == nil {
		return 0, false
	}
	key := (from << 2) | (b2i(jsonEscaped) << 1) | b2i(stopAtOuterJSONQuote)
	switch kind {
	case shellFrameDQ:
		if m.dq != nil {
			v, ok := m.dq[key]
			return v, ok
		}
	case shellFrameCS:
		if m.cmdSubst != nil {
			v, ok := m.cmdSubst[key]
			return v, ok
		}
	case shellFramePE:
		if m.paramExp != nil {
			peKey := (from << 3) | (b2i(inDouble) << 2) | (b2i(jsonEscaped) << 1) | b2i(stopAtOuterJSONQuote)
			v, ok := m.paramExp[peKey]
			return v, ok
		}
	case shellFrameBT:
		if m.backtick != nil {
			btKey := (from << 3) | (b2i(inDouble) << 2) | (b2i(jsonEscaped) << 1) | b2i(stopAtOuterJSONQuote)
			v, ok := m.backtick[btKey]
			return v, ok
		}
	}
	return 0, false
}

func (m *shellCloseMemo) store(kind shellFrameKind, from int, inDouble, jsonEscaped, stopAtOuterJSONQuote bool, val int) {
	if m == nil {
		return
	}
	key := (from << 2) | (b2i(jsonEscaped) << 1) | b2i(stopAtOuterJSONQuote)
	switch kind {
	case shellFrameDQ:
		if m.dq == nil {
			m.dq = make(map[int]int)
		}
		m.dq[key] = val
	case shellFrameCS:
		if m.cmdSubst == nil {
			m.cmdSubst = make(map[int]int)
		}
		m.cmdSubst[key] = val
	case shellFramePE:
		if m.paramExp == nil {
			m.paramExp = make(map[int]int)
		}
		peKey := (from << 3) | (b2i(inDouble) << 2) | (b2i(jsonEscaped) << 1) | b2i(stopAtOuterJSONQuote)
		m.paramExp[peKey] = val
	case shellFrameBT:
		if m.backtick == nil {
			m.backtick = make(map[int]int)
		}
		btKey := (from << 3) | (b2i(inDouble) << 2) | (b2i(jsonEscaped) << 1) | b2i(stopAtOuterJSONQuote)
		m.backtick[btKey] = val
	}
}

func (h *headerQuoteScanner) shellMemoPtr() *shellCloseMemo {
	if h == nil {
		return nil
	}
	return &h.shellMemos
}

func (h *headerQuoteScanner) doubleQuoteClosePos(from int) int {
	if h != nil && from >= 0 && from < len(h.s) {
		s := h.s
		end := min(from+64, len(s))
		for k := from; k < end; k++ {
			switch s[k] {
			case '\\', '$', '`':
				goto full
			case '"':
				return k + 1
			}
		}
	}
full:
	return shellScanClose(h.shellMemoPtr(), h.s, shellFrameDQ, from, false, false, false)
}

func (h *headerQuoteScanner) doubleQuoteCloseWithin(from, limit int) int {
	cPos := h.doubleQuoteClosePos(from)
	if cPos < 0 || cPos > limit {
		return -1
	}
	return cPos
}

func (h *headerQuoteScanner) doubleQuoteCloseJSONPos(from int) int {
	return shellScanClose(h.shellMemoPtr(), h.s, shellFrameDQ, from, false, true, false)
}

func (h *headerQuoteScanner) doubleQuoteCloseJSONWithin(from, limit int, stopAtOuterJSONQuote bool) int {
	cPos := shellScanClose(h.shellMemoPtr(), h.s, shellFrameDQ, from, false, true, stopAtOuterJSONQuote)
	if cPos < 0 || cPos > limit {
		return -1
	}
	return cPos
}

func (h *headerQuoteScanner) cmdSubstClosePos(from int, jsonEscaped bool) int {
	return shellScanClose(h.shellMemoPtr(), h.s, shellFrameCS, from, false, jsonEscaped, false)
}

func (h *headerQuoteScanner) cmdSubstCloseWithin(from, limit int, jsonEscaped bool) int {
	cPos := shellScanClose(h.shellMemoPtr(), h.s, shellFrameCS, from, false, jsonEscaped, false)
	if cPos < 0 || cPos > limit {
		return -1
	}
	return cPos
}

func (h *headerQuoteScanner) paramExpClosePos(from int, inDouble, jsonEscaped bool) int {
	return shellScanClose(h.shellMemoPtr(), h.s, shellFramePE, from, inDouble, jsonEscaped, false)
}

func (h *headerQuoteScanner) paramExpCloseWithin(from int, inDouble bool, limit int, jsonEscaped bool) int {
	cPos := shellScanClose(h.shellMemoPtr(), h.s, shellFramePE, from, inDouble, jsonEscaped, false)
	if cPos < 0 || cPos > limit {
		return -1
	}
	return cPos
}

func (h *headerQuoteScanner) backtickClosePos(from int, inDouble, jsonEscaped bool) int {
	return shellScanClose(h.shellMemoPtr(), h.s, shellFrameBT, from, inDouble, jsonEscaped, false)
}

func (h *headerQuoteScanner) backtickCloseWithin(from, limit int, inDouble, jsonEscaped bool) int {
	cPos := shellScanClose(h.shellMemoPtr(), h.s, shellFrameBT, from, inDouble, jsonEscaped, false)
	if cPos < 0 || cPos > limit {
		return -1
	}
	return cPos
}

func noteCSWordPart(f *shellFrame) {
	f.cmdPos = false
	if n := len(f.cases); n > 0 {
		top := &f.cases[n-1]
		switch top.phase {
		case caseExpectWord:
			top.sawWord = true
		case casePattern:
			top.justAfterSemi = false
			top.allowOptParen = false
		}
	}
}

func scanSingleQuotedClose(s string, from int, jsonEscaped bool) int {
	if !jsonEscaped {
		if rel := strings.IndexByte(s[from:], '\''); rel >= 0 {
			return from + rel + 1
		}
		return -1
	}
	for j := from; j < len(s); {
		c := s[j]
		if c == '\'' {
			return j + 1
		}
		if c == '"' {
			return -1
		}
		if c == '\\' {
			bs := 0
			for j+bs < len(s) && s[j+bs] == '\\' {
				bs++
			}
			if bs%2 == 1 && j+bs < len(s) {
				j += bs + 1
			} else {
				j += bs
			}
			continue
		}
		j++
	}
	return -1
}

// scanANSICQuotedClose scans past the closing single quote of an ANSI-C $'...'
// string starting immediately after $', handling backslash escapes (and doubled
// backslashes when jsonEscaped is true).
func scanANSICQuotedClose(s string, from int, jsonEscaped bool) int {
	if !jsonEscaped {
		for j := from; j < len(s); j++ {
			if s[j] == '\'' {
				return j + 1
			}
			if s[j] == '\\' {
				j++
			}
		}
		return -1
	}
	for j := from; j < len(s); {
		c := s[j]
		if c == '\'' {
			return j + 1
		}
		if c == '"' {
			return -1
		}
		if c == '\\' {
			bs := 0
			for j+bs < len(s) && s[j+bs] == '\\' {
				bs++
			}
			if j+bs < len(s) {
				next := s[j+bs]
				if next == '"' {
					if bs%2 == 0 {
						return -1
					}
					j += bs + 1
					continue
				}
				if next == '\'' {
					if bs%2 == 1 || bs%4 == 2 {
						j += bs + 1
						continue
					}
					j += bs
					continue
				}
			}
			if bs%2 == 1 && j+bs < len(s) {
				j += bs + 1
			} else {
				j += bs
			}
			continue
		}
		j++
	}
	return -1
}

func makeShellFrame(kind shellFrameKind, openPos int, inDouble bool) shellFrame {
	f := shellFrame{kind: kind, openPos: openPos, inDouble: inDouble}
	switch kind {
	case shellFrameCS:
		f.parenDepth = 1
		f.cmdPos = true
	case shellFramePE:
		f.braceDepth = 1
	}
	return f
}

// shellScanClose returns the byte offset just after the closing delimiter of a
// POSIX shell construct (double quote, $(...), ${...}, or `...`) starting at
// s[from:], or -1 if unclosed. Memoized on h by construct kind and start
// offset so scanning across any number of headers remains strictly O(N).
func shellScanClose(memos *shellCloseMemo, s string, startKind shellFrameKind, from int, inDouble, jsonEscaped, stopAtOuterJSONQuote bool) int {
	if from < 0 || from > len(s) {
		return -1
	}
	if v, ok := memos.lookup(startKind, from, inDouble, jsonEscaped, stopAtOuterJSONQuote); ok {
		return v
	}
	var stackBuf [16]shellFrame
	stack := stackBuf[:0]
	stack = append(stack, makeShellFrame(startKind, from, inDouble))

	failAll := func() int {
		for idx := range stack {
			memos.store(stack[idx].kind, stack[idx].openPos, stack[idx].inDouble, jsonEscaped, stopAtOuterJSONQuote, -1)
		}
		return -1
	}

	k := from
	pushOrJump := func(kind shellFrameKind, childFrom int, childInDouble bool) (int, bool, bool) {
		if v, ok := memos.lookup(kind, childFrom, childInDouble, jsonEscaped, stopAtOuterJSONQuote); ok {
			if v < 0 {
				return -1, false, false
			}
			return v, true, false
		}
		stack = append(stack, makeShellFrame(kind, childFrom, childInDouble))
		return childFrom, false, true
	}

	closeTop := func(endPos int) (int, bool) {
		top := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		memos.store(top.kind, top.openPos, top.inDouble, jsonEscaped, stopAtOuterJSONQuote, endPos)
		if len(stack) == 0 {
			return endPos, true
		}
		parent := &stack[len(stack)-1]
		if parent.kind == shellFrameCS {
			noteCSWordPart(parent)
		}
		return endPos, false
	}

	for len(stack) > 0 {
		if k >= len(s) {
			return failAll()
		}
		f := &stack[len(stack)-1]
		c := s[k]

		switch f.kind {
		case shellFrameDQ:
			switch c {
			case '\\':
				if !jsonEscaped {
					if k+1 < len(s) {
						k += 2
					} else {
						k++
					}
					continue
				}
				bs := 0
				for k+bs < len(s) && s[k+bs] == '\\' {
					bs++
				}
				if k+bs < len(s) && s[k+bs] == '"' {
					if bs%4 == 1 {
						endPos, done := closeTop(k + bs + 1)
						if done {
							return endPos
						}
						k = endPos
						continue
					}
					if bs%4 == 3 {
						k += bs + 1
						continue
					}
					k += bs
					continue
				}
				if k+bs < len(s) && (s[k+bs] == '$' || s[k+bs] == '`') {
					if bs%4 == 2 {
						k += bs + 1
						continue
					}
					if bs%4 == 0 {
						k += bs
						continue
					}
				}
				if bs%2 == 1 && k+bs < len(s) {
					k += bs + 1
				} else {
					k += bs
				}
			case '"':
				if !jsonEscaped || stopAtOuterJSONQuote {
					endPos, done := closeTop(k + 1)
					if done {
						return endPos
					}
					k = endPos
					continue
				}
				return failAll()
			case '$':
				if k+1 < len(s) && (s[k+1] == '(' || s[k+1] == '{') {
					kind := shellFrameCS
					if s[k+1] == '{' {
						kind = shellFramePE
					}
					nextK, _, _ := pushOrJump(kind, k+2, kind == shellFramePE)
					if nextK < 0 {
						return failAll()
					}
					k = nextK
					continue
				}
				k++
			case '`':
				nextK, _, _ := pushOrJump(shellFrameBT, k+1, true)
				if nextK < 0 {
					return failAll()
				}
				k = nextK
			default:
				k++
			}

		case shellFrameBT:
			switch c {
			case '\\':
				if !jsonEscaped {
					if k+1 < len(s) && (s[k+1] == '\\' || s[k+1] == '`' || s[k+1] == '$' || s[k+1] == '\n' || (f.inDouble && s[k+1] == '"')) {
						k += 2
					} else {
						k++
					}
					continue
				}
				bs := 0
				for k+bs < len(s) && s[k+bs] == '\\' {
					bs++
				}
				if k+bs < len(s) && (s[k+bs] == '`' || s[k+bs] == '$') {
					if bs%4 == 2 {
						k += bs + 1
						continue
					}
					if bs%4 == 0 {
						k += bs
						continue
					}
				}
				if bs%2 == 1 && k+bs < len(s) {
					k += bs + 1
				} else {
					k += bs
				}
			case '`':
				endPos, done := closeTop(k + 1)
				if done {
					return endPos
				}
				k = endPos
			case '"':
				if jsonEscaped {
					return failAll()
				}
				k++
			default:
				k++
			}

		case shellFramePE:
			switch c {
			case '\\':
				if !jsonEscaped {
					if k+1 < len(s) {
						k += 2
					} else {
						k++
					}
					continue
				}
				bs := 0
				for k+bs < len(s) && s[k+bs] == '\\' {
					bs++
				}
				if k+bs < len(s) && s[k+bs] == '"' {
					if bs%4 == 1 {
						nextK, _, _ := pushOrJump(shellFrameDQ, k+bs+1, false)
						if nextK < 0 {
							return failAll()
						}
						k = nextK
						continue
					}
					if bs%4 == 3 {
						k += bs + 1
						continue
					}
					k += bs
					continue
				}
				if k+bs < len(s) && (s[k+bs] == '$' || s[k+bs] == '`') {
					if bs%4 == 2 {
						k += bs + 1
						continue
					}
					if bs%4 == 0 {
						k += bs
						continue
					}
				}
				if bs%2 == 1 && k+bs < len(s) {
					k += bs + 1
				} else {
					k += bs
				}
			case '\'':
				if f.inDouble {
					k++
					continue
				}
				sqEnd := scanSingleQuotedClose(s, k+1, jsonEscaped)
				if sqEnd < 0 {
					return failAll()
				}
				k = sqEnd
			case '"':
				if jsonEscaped {
					return failAll()
				}
				nextK, _, _ := pushOrJump(shellFrameDQ, k+1, false)
				if nextK < 0 {
					return failAll()
				}
				k = nextK
			case '$':
				if !f.inDouble && k+1 < len(s) && s[k+1] == '\'' {
					acEnd := scanANSICQuotedClose(s, k+2, jsonEscaped)
					if acEnd < 0 {
						return failAll()
					}
					k = acEnd
					continue
				}
				if k+1 < len(s) && (s[k+1] == '(' || s[k+1] == '{') {
					kind := shellFrameCS
					if s[k+1] == '{' {
						kind = shellFramePE
					}
					nextK, _, _ := pushOrJump(kind, k+2, kind == shellFramePE && f.inDouble)
					if nextK < 0 {
						return failAll()
					}
					k = nextK
					continue
				}
				k++
			case '`':
				nextK, _, _ := pushOrJump(shellFrameBT, k+1, f.inDouble)
				if nextK < 0 {
					return failAll()
				}
				k = nextK
			case '{':
				f.braceDepth++
				k++
			case '}':
				f.braceDepth--
				if f.braceDepth == 0 {
					endPos, done := closeTop(k + 1)
					if done {
						return endPos
					}
					k = endPos
					continue
				}
				k++
			default:
				k++
			}

		case shellFrameCS:
			switch c {
			case ' ', '\t', '\r':
				if n := len(f.cases); n > 0 && f.cases[n-1].phase == caseExpectWord && f.cases[n-1].sawWord {
					f.cases[n-1].phase = caseExpectIn
					f.cases[n-1].sawWord = false
				}
				k++
			case '\n':
				if n := len(f.cases); n > 0 && f.cases[n-1].phase == caseExpectWord && f.cases[n-1].sawWord {
					f.cases[n-1].phase = caseExpectIn
					f.cases[n-1].sawWord = false
				}
				f.cmdPos = true
				k++
			case '#':
				if k == f.openPos || isShellCommentBoundaryByte(s[k-1]) {
					for k < len(s) && s[k] != '\n' {
						if jsonEscaped && s[k] == '"' {
							break
						}
						k++
					}
					continue
				}
				noteCSWordPart(f)
				k++
			case '\\':
				if !jsonEscaped {
					if k+1 < len(s) && s[k+1] == '\n' {
						k += 2
						continue
					}
					noteCSWordPart(f)
					if k+1 < len(s) {
						k += 2
					} else {
						k++
					}
					continue
				}
				bs := 0
				for k+bs < len(s) && s[k+bs] == '\\' {
					bs++
				}
				if k+bs < len(s) && s[k+bs] == '"' {
					if bs%4 == 1 {
						noteCSWordPart(f)
						nextK, _, _ := pushOrJump(shellFrameDQ, k+bs+1, false)
						if nextK < 0 {
							return failAll()
						}
						k = nextK
						continue
					}
					if bs%4 == 3 {
						noteCSWordPart(f)
						k += bs + 1
						continue
					}
					k += bs
					continue
				}
				if k+bs < len(s) && (s[k+bs] == '$' || s[k+bs] == '`') {
					if bs%4 == 2 {
						noteCSWordPart(f)
						k += bs + 1
						continue
					}
					if bs%4 == 0 {
						k += bs
						continue
					}
				}
				noteCSWordPart(f)
				if bs%2 == 1 && k+bs < len(s) {
					k += bs + 1
				} else {
					k += bs
				}
			case '\'':
				noteCSWordPart(f)
				sqEnd := scanSingleQuotedClose(s, k+1, jsonEscaped)
				if sqEnd < 0 {
					return failAll()
				}
				k = sqEnd
			case '"':
				if jsonEscaped {
					return failAll()
				}
				noteCSWordPart(f)
				nextK, _, _ := pushOrJump(shellFrameDQ, k+1, false)
				if nextK < 0 {
					return failAll()
				}
				k = nextK
			case '$':
				if k+1 < len(s) && s[k+1] == '\'' {
					noteCSWordPart(f)
					acEnd := scanANSICQuotedClose(s, k+2, jsonEscaped)
					if acEnd < 0 {
						return failAll()
					}
					k = acEnd
					continue
				}
				if k+1 < len(s) && (s[k+1] == '(' || s[k+1] == '{') {
					noteCSWordPart(f)
					kind := shellFrameCS
					if s[k+1] == '{' {
						kind = shellFramePE
					}
					nextK, _, _ := pushOrJump(kind, k+2, false)
					if nextK < 0 {
						return failAll()
					}
					k = nextK
					continue
				}
				noteCSWordPart(f)
				k++
			case '`':
				noteCSWordPart(f)
				nextK, _, _ := pushOrJump(shellFrameBT, k+1, false)
				if nextK < 0 {
					return failAll()
				}
				k = nextK
			case ';':
				if n := len(f.cases); n > 0 {
					top := &f.cases[n-1]
					if f.parenDepth == top.parenDepth && top.phase == caseBody {
						if strings.HasPrefix(s[k:], ";;&") {
							top.phase = casePattern
							top.justAfterSemi = true
							top.hasOptParen = false
							top.allowOptParen = true
							f.cmdPos = false
							k += 3
							continue
						}
						if strings.HasPrefix(s[k:], ";;") || strings.HasPrefix(s[k:], ";&") {
							top.phase = casePattern
							top.justAfterSemi = true
							top.hasOptParen = false
							top.allowOptParen = true
							f.cmdPos = false
							k += 2
							continue
						}
					}
				}
				f.cmdPos = true
				k++
			case '&', '|':
				if n := len(f.cases); n > 0 {
					top := &f.cases[n-1]
					if c == '|' && f.parenDepth == top.parenDepth && top.phase == casePattern {
						top.justAfterSemi = false
						top.allowOptParen = false
						f.cmdPos = false
						k++
						continue
					}
				}
				if k+1 < len(s) && (s[k+1] == c || (c == '|' && s[k+1] == '&')) {
					k += 2
				} else {
					k++
				}
				f.cmdPos = true
			case '<', '>':
				k++
			case '(':
				if n := len(f.cases); n > 0 {
					top := &f.cases[n-1]
					if top.phase == casePattern && f.parenDepth == top.parenDepth && !top.hasOptParen && top.allowOptParen {
						top.hasOptParen = true
						top.allowOptParen = false
						top.justAfterSemi = false
						f.parenDepth++
						f.cmdPos = false
						k++
						continue
					}
				}
				f.parenDepth++
				f.cmdPos = true
				k++
			case ')':
				if n := len(f.cases); n > 0 {
					top := &f.cases[n-1]
					if top.phase == casePattern && f.parenDepth == top.parenDepth {
						top.phase = caseBody
						top.justAfterSemi = false
						top.allowOptParen = false
						f.cmdPos = true
						k++
						continue
					}
					if top.phase == casePattern && top.hasOptParen && f.parenDepth == top.parenDepth+1 {
						f.parenDepth--
						top.hasOptParen = false
						top.phase = caseBody
						top.justAfterSemi = false
						top.allowOptParen = false
						f.cmdPos = true
						k++
						continue
					}
				}
				f.parenDepth--
				if f.parenDepth == 0 {
					endPos, done := closeTop(k + 1)
					if done {
						return endPos
					}
					k = endPos
					continue
				}
				f.cmdPos = false
				k++
			default:
				wStart := k
				for k < len(s) && !isPOSIXShellWordBreak(s[k]) {
					k++
				}
				word := s[wStart:k]
				isWordConcat := k < len(s) && (s[k] == '\'' || s[k] == '"' || s[k] == '$' || s[k] == '`' || s[k] == '\\')
				n := len(f.cases)
				switch {
				case n > 0 && f.cases[n-1].phase == caseExpectWord:
					f.cases[n-1].sawWord = true
					f.cmdPos = false
				case !isWordConcat && n > 0 && f.cases[n-1].phase == caseExpectIn:
					if word == "in" {
						f.cases[n-1].phase = casePattern
						f.cases[n-1].justAfterSemi = true
						f.cases[n-1].allowOptParen = true
					}
					f.cmdPos = false
				case n > 0 && f.cases[n-1].phase == casePattern:
					top := &f.cases[n-1]
					if !isWordConcat && top.justAfterSemi && word == "esac" {
						f.cases = f.cases[:n-1]
					} else {
						top.justAfterSemi = false
						top.allowOptParen = false
					}
					f.cmdPos = false
				default:
					wasCmdPos := f.cmdPos
					if !isWordConcat && wasCmdPos {
						switch word {
						case "esac":
							if n > 0 && f.parenDepth == f.cases[n-1].parenDepth && f.cases[n-1].phase == caseBody {
								f.cases = f.cases[:n-1]
							}
							f.cmdPos = false
						case "case":
							f.cases = append(f.cases, caseState{parenDepth: f.parenDepth, phase: caseExpectWord})
							f.cmdPos = false
						case "{", "if", "then", "else", "elif", "while", "until", "do", "!", "time":
							f.cmdPos = true
						default:
							f.cmdPos = false
						}
					} else {
						f.cmdPos = false
					}
				}
			}
		}
	}
	return -1
}

// hasUnquotedSingleQuoteInShellLine reports whether s[next:nextEnd], read in
// unquoted POSIX shell mode (skipping double-quoted strings, $(...), ${...},
// backticks, backslash escapes, and # comments), contains a single quote.
func (h *headerQuoteScanner) hasUnquotedSingleQuoteInShellLine(next, nextEnd int) bool {
	s := h.s
	for k := next; k < nextEnd; {
		switch s[k] {
		case '\\':
			if k+1 < nextEnd {
				k += 2
			} else {
				k++
			}
		case '\'':
			return true
		case '"':
			cPos := h.doubleQuoteCloseWithin(k+1, nextEnd)
			if cPos < 0 {
				return false
			}
			k = cPos
		case '$':
			if k+1 < nextEnd && s[k+1] == '(' {
				cPos := h.cmdSubstCloseWithin(k+2, nextEnd, false)
				if cPos < 0 {
					return false
				}
				k = cPos
				continue
			}
			if k+1 < nextEnd && s[k+1] == '{' {
				cPos := h.paramExpCloseWithin(k+2, false, nextEnd, false)
				if cPos < 0 {
					return false
				}
				k = cPos
				continue
			}
			if k+1 < nextEnd && s[k+1] == '\'' {
				cPos := scanANSICQuotedClose(s[:nextEnd], k+2, false)
				if cPos < 0 {
					return false
				}
				k = cPos
				continue
			}
			k++
		case '`':
			cPos := h.backtickCloseWithin(k+1, nextEnd, false, false)
			if cPos < 0 {
				return false
			}
			k = cPos
		case '#':
			if k == next || s[k-1] == ' ' || s[k-1] == '\t' || strings.IndexByte(";&|()", s[k-1]) >= 0 {
				return false
			}
			k++
		default:
			k++
		}
	}
	return false
}

// hasUnclosedQuoteOnNextLine reports whether the first non-empty line after
// lineEnd does not open a new header and contains an unquoted single quote.
func (h *headerQuoteScanner) hasUnclosedQuoteOnNextLine(lineEnd int) bool {
	if h == nil || lineEnd >= len(h.s) || (h.s[lineEnd] != '\n' && h.s[lineEnd] != '\r') {
		return false
	}
	if h.nextLineUnclosedMemo != nil {
		if v, ok := h.nextLineUnclosedMemo[lineEnd]; ok {
			return v
		}
	}
	stamp := func(v bool) bool {
		if h.nextLineUnclosedMemo == nil {
			h.nextLineUnclosedMemo = make(map[int]bool)
		}
		h.nextLineUnclosedMemo[lineEnd] = v
		return v
	}
	s := h.s
	next := lineEnd + 1
	if s[lineEnd] == '\r' && next < len(s) && s[next] == '\n' {
		next++
	}
	nextEnd := len(s)
	if nl := strings.IndexAny(s[next:], "\r\n"); nl >= 0 {
		nextEnd = next + nl
	}
	for next == nextEnd && nextEnd < len(s) {
		next = nextEnd + 1
		if s[nextEnd] == '\r' && next < len(s) && s[next] == '\n' {
			next++
		}
		nextEnd = len(s)
		if nl := strings.IndexAny(s[next:], "\r\n"); nl >= 0 {
			nextEnd = next + nl
		}
	}
	if next >= nextEnd || startsHeaderLine(s, next) {
		return stamp(false)
	}
	return stamp(h.hasUnquotedSingleQuoteInShellLine(next, nextEnd))
}

// joinedQuoteCloses reports whether outer and inner close at or after from and
// leave balanced shell quoting under strict lexical mode, prose-contraction
// mode, or (when the closing quote terminates a plain prose word on its line)
// plural-possessive mode.
func (h *headerQuoteScanner) joinedQuoteCloses(from int, outer, inner string) bool {
	if h == nil || from >= len(h.s) {
		return false
	}
	closePos := h.joinedQuoteCloseAt(from, outer, inner)
	if closePos < 0 {
		return false
	}
	if outer == "'" && inner == "" && closePos > 0 {
		if isWordApostropheAt(h.s, closePos-1) && isTrailingApostrophe(h.s, closePos-1) {
			stops, ok := h.closeTailStops[closePos]
			if !ok {
				closeLineStart := strings.LastIndexAny(h.s[:closePos-1], "\r\n") + 1
				closeLineEnd := len(h.s)
				if nl := strings.IndexAny(h.s[closePos:], "\r\n"); nl >= 0 {
					closeLineEnd = closePos + nl
				}
				stops = h.apostropheTailStopsOnLine(closeLineStart, closePos-1, closeLineEnd, false, false)
				if h.closeTailStops == nil {
					h.closeTailStops = make(map[int]bool)
				}
				h.closeTailStops[closePos] = stops
			}
			if !stops {
				return false
			}
		}
	}
	esc := inner != ""
	if h.chainClosesFrom(closePos, esc, chainStrict) || h.chainClosesFrom(closePos, esc, chainSkipContractions) {
		return true
	}
	return closePos > 0 && h.closesProseLineWord(closePos-1) && h.chainClosesFrom(closePos, esc, chainSkipPossessives)
}

// singleQuoteClosesAfterLine reports whether a single quote that stays open at
// lineEnd closes on a subsequent line and leaves a balanced shell quote chain.
func (h *headerQuoteScanner) singleQuoteClosesAfterLine(lineEnd int) bool {
	if h == nil || lineEnd >= len(h.s) || (h.s[lineEnd] != '\n' && h.s[lineEnd] != '\r') {
		return false
	}
	return h.joinedQuoteCloses(lineEnd, "'", "")
}

// closeSlot returns the entry of memos that caches the close of the quotes
// outer and inner (see quoteCloses), or nil if their close is not cached.
func closeSlot(memos *[4]quoteCloseMemo, outer, inner string) *quoteCloseMemo {
	switch {
	case outer == `'` && inner == "":
		return &memos[0]
	case outer == `"`:
		return &memos[1]
	case outer == "" && inner == `\'`:
		return &memos[2]
	case outer == "" && inner == `\"`:
		return &memos[3]
	default:
		return nil
	}
}

// closeAt returns quoteCloseScan(h.s, from, outer, inner): where the quotes
// outer and inner that are open before a Cookie header's value close, where
// from is the start of that value, or the end of a string that the value
// opens with a quote of their kind, and is no earlier than the from of the
// call before for those quotes, as it is for the headers of s in order. As
// with quoteCloses, where such quotes close depends on from and their kind
// alone, so the scan that found the close for one header holds for every
// later header whose value starts before that close, and is reused for it.
// The checks for all the headers inside one long quoted string, as the ones
// in message="The cookie: a was b. The cookie: a was b. ..." are, thus read
// that string once, rather than once for each header.
func (h *headerQuoteScanner) closeAt(from int, outer, inner string) int {
	m := closeSlot(&h.proseCloses, outer, inner)
	if m == nil {
		return quoteCloseScan(h.s, from, outer, inner)
	}
	if !m.ok || from < m.from || m.closePos >= 0 && from >= m.closePos {
		*m = quoteCloseMemo{ok: true, from: from, closePos: quoteCloseScan(h.s, from, outer, inner), lastHdrLine: -1}
	}
	return m.closePos
}

// isExplicitLineQuoteOpener reports whether the quote at s[k] on the line
// starting at lineStart stands at an explicit quote-opening position (at the
// start of the line, in a shell assignment or prefixed string, after a
// delimiter such as =, :, {, [, (, or comma, or after a CLI flag token starting
// with '-') rather than after plain prose words as in `world "`.
func isExplicitLineQuoteOpener(sh *shellAssignScanner, s string, lineStart, k int) bool {
	if k < lineStart {
		return false
	}
	if skipHorizDelimSpace(s, lineStart) == k || inShellAssignment(sh, s, k) || opensPrefixedString(s, k) {
		return true
	}
	p := k - 1
	for p >= lineStart && (s[p] == ' ' || s[p] == '\t') {
		p--
	}
	if p < lineStart {
		return true
	}
	if strings.IndexByte("=:([{,;&|<>+-*/~^", s[p]) >= 0 {
		return true
	}
	if p < k-1 {
		tokStart := p
		for tokStart > lineStart && s[tokStart-1] != ' ' && s[tokStart-1] != '\t' {
			tokStart--
		}
		if s[tokStart] == '-' {
			return true
		}
	}
	return false
}

// canOpenPriorLineQuote reports whether the quote at s[k], on the line
// s[lineStart:lineLim], stands where a quoted string may open rather than at
// the closing quote of an earlier word or sentence, as in world!", world2' or
// last_word".
func canOpenPriorLineQuote(sh *shellAssignScanner, s string, lineStart, k, lineLim int, isPriorLine bool) bool {
	if k == 0 {
		return true
	}
	qLen := 1
	if s[k] == '\\' && k+1 < len(s) {
		qLen = 2
	}
	if isPriorLine && quoteRestIsHeaderOrSpace(s, min(k+qLen, lineLim), lineLim) {
		return isExplicitLineQuoteOpener(sh, s, lineStart, k)
	}
	r, _ := utf8.DecodeLastRuneInString(s[:k])
	if isDelimSpace(r) || opensPrefixedString(s, k) {
		return true
	}
	c := s[k-1]
	if strings.IndexByte("=:([{,;&|<>+-*/~^", c) >= 0 {
		return true
	}
	if c == '!' || c == '?' {
		if k == 1 {
			return true
		}
		prev, _ := utf8.DecodeLastRuneInString(s[:k-1])
		return !unicode.IsLetter(prev) && !unicode.IsDigit(prev)
	}
	return false
}

// quoteRestIsHeaderOrSpace reports whether s[qEnd:lineEnd], the remainder of
// the line after an opening quote before a line break, consists solely of
// horizontal whitespace or starts with a header line.
func quoteRestIsHeaderOrSpace(s string, qEnd, lineEnd int) bool {
	if qEnd > lineEnd {
		return false
	}
	j := skipHorizDelimSpace(s, qEnd)
	return j >= lineEnd || startsHeaderLine(s, j)
}

// outerQuoteCloses reports whether the open outer and inner quotes at s[from]
// close before the end of s without pairing an unclosed quote across a
// subsequent header line with the opening quote of a paired string on a later
// line, as in req="Authorization: Bearer x<LF>Content-Type: application/json<LF><LF>{"status": "ok"}.
func outerQuoteCloses(sh *shellAssignScanner, s string, from int, outer, inner string) bool {
	closePos := quoteCloseScan(s, from, outer, inner)
	if closePos < 0 {
		return false
	}
	checkQuote := outer
	if checkQuote == "" {
		checkQuote = inner
	}
	return checkQuote == "" || !crossesHeaderLine(s, from, closePos) || !closeLineRejects(sh, s, closePos, checkQuote)
}

// quoteCloseScan returns where the open outer and inner quotes at s[from]
// close: the offset just after the quote that closes the last of them, or -1
// if they do not close before the end of s. It reads s with the lexical rules
// of headerQuoteScanner.openBefore.
func quoteCloseScan(s string, from int, outer, inner string) int {
	for k := from; k < len(s); {
		if outer == "" && inner == "" {
			return k
		}
		c := s[k]
		switch {
		case outer == "'":
			if c == '\'' {
				outer = ""
			}
			k++
		case c == '\\':
			bs := 0
			for k+bs < len(s) && s[k+bs] == '\\' {
				bs++
			}
			if k+bs < len(s) && (s[k+bs] == '"' || s[k+bs] == '\'') {
				if bs%2 == 1 {
					if bs%4 == 1 {
						q := s[k+bs-1 : k+bs+1]
						switch inner {
						case q:
							inner = ""
						case "":
							inner = q
						}
					}
					k += bs + 1
				} else {
					k += bs
				}
			} else if bs > 1 {
				k += bs
			} else if k+1 < len(s) && s[k+1] != '\r' && s[k+1] != '\n' {
				k += 2
			} else {
				k++
			}
		case inner != "":
			if outer != "" && c == outer[0] {
				outer, inner = "", ""
			}
			k++
		default:
			if outer != "" && c == outer[0] {
				outer = ""
			}
			k++
		}
	}
	if outer == "" && inner == "" {
		return len(s)
	}
	return -1
}

// closeLineRejects reports whether the line on which a quote checkQuote that
// stays open over a later header line closes, just before closePos, shows
// that the quote there opens a string of its own on that line rather than
// closing the one opened before the header line: the rest of that line
// leaves a quote of the kind open, or the quote ends the line and follows a
// key and its colon or a shell assignment, as the last quote of
// X-Note: 'hi<LF> does.
func closeLineRejects(sh *shellAssignScanner, s string, closePos int, checkQuote string) bool {
	rest := s[closePos:min(len(s), closePos+maxShellLineScan)]
	if nl := strings.IndexAny(rest, "\r\n"); nl >= 0 {
		rest = rest[:nl]
	}
	if hasUnpairedQuoteOnLine(rest, checkQuote) {
		return true
	}
	qPos := closePos - len(checkQuote)
	closeLineStart := strings.LastIndexAny(s[:qPos], "\r\n") + 1
	hdrStart := skipHorizDelimSpace(s, closeLineStart)
	if skipHorizDelimSpace(s, closePos) == closePos+len(rest) && hdrStart < qPos {
		p := qPos - 1
		for p >= hdrStart && (s[p] == ' ' || s[p] == '\t') {
			p--
		}
		if p >= hdrStart && strings.IndexByte(s[hdrStart:p], ':') < 0 && strings.IndexByte(s[hdrStart:p], '=') < 0 {
			if s[p] == ':' || inShellAssignment(sh, s, qPos) || (s[p] == '=' && p > hdrStart && (s[p-1] == ' ' || s[p-1] == '\t')) {
				return true
			}
		}
	}
	return false
}

// isFoldedHeaderLine reports whether the line starting at s[next:] is an
// RFC 9112 whitespace-folded continuation of a header whose line started with
// indentLen leading spaces or tabs: it is indented further than the header's
// line. A line of spaces or tabs alone is such a continuation too, as the
// second line of Authorization: Bearer abc<CRLF> <CRLF>\tSECRET is, so that
// the folded lines after it stay in the value; an empty line, or a line that
// starts at the header's indentation or before it, as another header does, is
// not.
func isFoldedHeaderLine(s string, next, indentLen int) bool {
	nextIndent := 0
	for next+nextIndent < len(s) && (s[next+nextIndent] == ' ' || s[next+nextIndent] == '\t') {
		nextIndent++
	}
	if indentLen == 0 {
		return nextIndent > 0
	}
	return nextIndent > indentLen
}

// authHeaderValueEnd returns where the header value that starts at s[from:]
// and whose line ends at end ends. A quote in the value that a quote on a
// later line closes (see quotedValueRegex) carries the value on to the end of
// the line that the closing quote is on, as in api-key: 'abc<LF>SECRET' x and
// Authorization: "abc<LF>part2: SECRET", unless it is a single quote that
// reads as an apostrophe, between two letters or at the end of a word, whose
// closing quote stands on or after a later header line (see
// crossesHeaderLine) and leaves a single quote open on its line, as the
// apostrophe of Authorization: Bob's key<LF>X-Note: 'hi' does. A quote that
// nothing closes carries it on over the lines after it up to a line that
// starts another header (see startsHeaderLine), as in
// x-api-key: "abc<LF>SECRET<LF>Content-Type: text/plain, or to the end of s,
// unless it is a single quote that reads as the apostrophe of a prose word
// (see unclosedContractionIsProse). An escaped quote (\" or \') opens and
// closes a value as a quote does (see escapedQuotedRegex), a backslash
// escapes any other byte after it, and the quotes outer and inner of the
// strings that are open before the header (see headerQuoteScanner.openBefore)
// advance through the header value with the same lexical rules (across line
// breaks up to a line that starts another header when outer/inner is unclosed,
// or across colon-led lines when outer/inner is demonstrably closed, or to the
// end of the closing line of a nested multiline quoted value, see
// nestedQuotedValueEnd) until those strings close, as the last quote of
// "Authorization: x" or curl -H 'Authorization: Bearer abc<LF>SECRET' does,
// rather than opening a new string. When canFold is true, unquoted RFC 9112
// whitespace-folded continuation lines after end are also included (see
// isFoldedHeaderLine), even after a first line that holds only the redaction
// placeholder, as the folded line of Authorization: [REDACTED]<CRLF>\tSECRET
// does, since one redacted line does not show that the whole value is.
func authHeaderValueEnd(s string, from, end int, outer, inner string, canFold bool, indentLen int, quotes *headerQuoteScanner) (int, string) {
	sh := quotes.shellScanner()
	var closesChecked, closes bool
	if outer == "" && inner == "" && !headerBeforeIsCookie(s, from) {
		rest := strings.TrimLeft(s[from:end], " \t")
		if strings.HasPrefix(rest, redactedValue) {
			after := strings.TrimLeft(strings.TrimPrefix(rest, redactedValue), " \t")
			if after != "" && (after[0] == ';' || after[0] == '|' || strings.HasPrefix(after, "&&")) {
				lead := len(s[from:end]) - len(rest)
				return from + lead + len(redactedValue), ""
			}
		}
	}
	if !canFold && strings.TrimSpace(s[from:end]) == redactedValue {
		if outer == "" && inner == "" || quotes == nil || !quotes.placeholderQuoteCloses(end, outer, inner) {
			return end, ""
		}
		closesChecked = true
		closes = true
	}
	nestedEnd := 0
	lineEnd := end
	curLineStart := from
	entryOuter, entryInner := outer, inner
	joinedMultiline := false
	inSingleInJSON := outer == `"` && quotes != nil && hasSingleQuoteBeforeHeader(s, from) &&
		((inner == "" && quotes.inJSON()) || (inner == `\"` && quotes.curInnerJSONKind != jsonContainerNone))
	inSingleInEscapedText := outer == "" && inner == `\"` && quotes != nil && (quotes.curInnerSubSQ || hasSingleQuoteBeforeHeader(s, from))
	inEscDQInInnerJSON := outer == `"` && inner == `\"` && quotes != nil && quotes.curInnerJSONKind != jsonContainerNone && hasEscDQBeforeHeader(s, from)
	for k := from; ; {
		if k < end {
			if outer != "" || inner != "" {
				prevOuter, prevInner := outer, inner
				mLen, vEnd := nestedQuotedValueEnd(s, from, k, end, outer, inner)
				c := s[k]
				switch {
				case vEnd > end:
					nestedEnd = vEnd
					k += mLen
					end = vEnd
					if k < end && outer != "" && inner == "" && strings.TrimSpace(s[from:k-mLen]) == "" && quoteCloseScan(s, k, outer, "") < 0 {
						if endPos, unclosed := quotes.joinedHeaderArgEnd(k, end, "", quotes.inJSON(), false); unclosed == "" && endPos < end {
							return endPos, ""
						}
					}
				case outer == "'":
					if c == '\'' {
						if !joinedMultiline || closes {
							outer = ""
						} else if k > curLineStart {
							prevRune, _ := utf8.DecodeLastRuneInString(s[curLineStart:k])
							if (unicode.IsLetter(prevRune) || unicode.IsDigit(prevRune)) && quotes != nil && quotes.apostropheTailStopsOnLine(curLineStart, k, end, quotes.inJSON(), false) {
								outer = ""
							}
						}
					}
					k++
				case c == '\\':
					bs := 0
					for k+bs < end && s[k+bs] == '\\' {
						bs++
					}
					if k+bs < end && (s[k+bs] == '"' || s[k+bs] == '\'') {
						if bs%2 == 1 {
							if bs%4 == 1 {
								q := s[k+bs-1 : k+bs+1]
								switch inner {
								case q:
									inner = ""
								case "":
									inner = q
								}
							} else if inEscDQInInnerJSON && inner == `\"` && bs%4 == 3 && s[k+bs] == '"' && k < lineEnd && quotes != nil {
								if endPos, ok := quotes.escDQCloseTargetInEscapedJSON(k+bs+1, lineEnd); ok {
									if endPos == k+bs+1 {
										return k, ""
									}
									return endPos, s[k : k+bs+1]
								}
							}
							k += bs + 1
						} else {
							k += bs
						}
					} else if bs > 1 {
						k += bs
					} else if k+1 < end {
						k += 2
					} else {
						k++
					}
				case inner != "":
					if inner == "'" && c == '\'' {
						inner = ""
					} else if outer != "" && c == outer[0] {
						outer, inner = "", ""
					} else if inner == `\"` && c == '\'' && k < lineEnd && quotes != nil {
						if inSingleInJSON && quotes.singleQuoteClosesInEscapedJSON(k, lineEnd) {
							return k, ""
						}
						if inSingleInEscapedText && (!isWordApostropheAt(s, k) || isTrailingApostrophe(s, k)) {
							if endPos, unclosed := quotes.joinedHeaderArgEnd(k+1, lineEnd, "", false, false); unclosed == "" && endPos == k+1 {
								return k, ""
							}
						}
					}
					k++
				default:
					if outer == `"` && inner == "" && quotes != nil {
						if c == '$' && k+1 < end && (s[k+1] == '(' || s[k+1] == '{') {
							var nextK int
							if s[k+1] == '(' {
								nextK = quotes.cmdSubstClosePos(k+2, quotes.inJSON())
							} else {
								nextK = quotes.paramExpClosePos(k+2, true, quotes.inJSON())
							}
							if nextK >= 0 {
								k = nextK
								if k > end {
									end = len(s)
									if nl := strings.IndexAny(s[k:], "\r\n"); nl >= 0 {
										end = k + nl
									}
									curLineStart = strings.LastIndexAny(s[:k], "\r\n") + 1
									closesChecked = false
								}
								continue
							}
						}
						if c == '`' {
							if nextK := quotes.backtickClosePos(k+1, true, quotes.inJSON()); nextK >= 0 {
								k = nextK
								if k > end {
									end = len(s)
									if nl := strings.IndexAny(s[k:], "\r\n"); nl >= 0 {
										end = k + nl
									}
									curLineStart = strings.LastIndexAny(s[:k], "\r\n") + 1
									closesChecked = false
								}
								continue
							}
						}
					}
					if inSingleInJSON && c == '\'' && outer == `"` && inner == "" && k < lineEnd {
						if endPos, closedQuotes, ok := quotes.jsonWrappedInnerHeaderEndAt(k+1, lineEnd, entryOuter, "'"); ok {
							if (endPos > k+1 && s[endPos-1] == '"' && s[endPos-2] == '\'') || (endPos == k+1 && closedQuotes != "") {
								return k, ""
							}
							return endPos, closedQuotes
						}
					}
					if c == outer[0] {
						outer = ""
					}
					k++
				}
				if outer == `"` && (entryInner == `\"` || (entryInner == "'" && quotes != nil && quotes.inJSON())) && inner == "" && k <= lineEnd {
					if endPos, closedQuotes, ok := quotes.jsonWrappedInnerHeaderEndAt(k, lineEnd, entryOuter, entryInner); ok {
						return endPos, closedQuotes
					}
				}
				if outer == "" && inner == "" {
					nestedEnd = 0
					if joinedMultiline && (entryOuter != "" || entryInner != "") {
						if qLen := len(prevOuter) + len(prevInner); qLen > 0 && k-qLen > from && spanEndsWithNestedHeaderPrefix(s, from, k-qLen) {
							var nextClose int
							switch {
							case prevOuter == `"` && prevInner == "" && quotes != nil:
								nextClose = quotes.doubleQuoteCloseWithin(k, end)
							case prevOuter == "" && prevInner == `\"` && quotes != nil:
								nextClose = quotes.doubleQuoteCloseJSONWithin(k, end, false)
							default:
								nextClose = quoteCloseScan(s[:end], k, prevOuter, prevInner)
							}
							if nextClose >= 0 {
								if quotes == nil || quotes.isShellBalancedFrom(nextClose, end, quotes.inJSON()) {
									k = nextClose
								} else {
									k = end
								}
							} else if (quotes != nil && quotes.joinedQuoteCloses(end, prevOuter, prevInner)) || k == end || end < len(s) {
								outer, inner = prevOuter, prevInner
								joinedMultiline = true
								closesChecked = false
								k = end
								continue
							} else {
								if quotes != nil && quotes.inJSON() {
									for p := k; p < end; p++ {
										c := s[p]
										if c == '\\' {
											p++
											continue
										}
										if (c == '"' && jsonCloseBoundaryOK(s, p+1, end, false, true)) || ((c == '}' || c == ']') && jsonCloseBoundaryOK(s, p, end, false, true)) {
											return p, entryInner + entryOuter
										}
									}
								}
								return end, entryInner + entryOuter
							}
						}
					}
					if entryOuter != "" && (k <= lineEnd || joinedMultiline) {
						endPos, unclosed := quotes.joinedHeaderArgEnd(k, end, entryOuter, quotes.inJSON(), false)
						if unclosed == "" || endPos == len(s) {
							return endPos, entryInner + entryOuter
						}
						if len(unclosed) == 1 {
							outer = unclosed
						} else {
							inner = unclosed
						}
						joinedMultiline = true
						closesChecked = false
						k = endPos
						continue
					}
					if entryOuter == "" && entryInner != "" && (k <= lineEnd || joinedMultiline) {
						endPos, unclosed := quotes.joinedHeaderArgEnd(k, end, entryOuter, false, false)
						if unclosed == "" && ((endPos < end && (s[endPos] == ' ' || s[endPos] == '\t' || s[endPos] == ';' || s[endPos] == '&' || s[endPos] == '|')) || (joinedMultiline && endPos == end)) {
							return endPos, entryInner
						}
						if unclosed != "" && endPos < len(s) {
							if len(unclosed) == 1 {
								outer = unclosed
							} else {
								inner = unclosed
							}
							joinedMultiline = true
							closesChecked = false
							k = endPos
							continue
						}
					}
				}
				continue
			}
			q := openingQuote(s, k)
			switch {
			case q == "" && s[k] == '\\':
				if k+1 < end {
					k += 2
				} else {
					k++
				}
				continue
			case q == "":
				k++
				continue
			}
			m := quotedValueAt(sh, s, k)
			if m != "" && k+len(m) > end && q == "'" && (isMidWordApostrophe(s, k) || isTrailingApostrophe(s, k)) && crossesHeaderLine(s, end, k+len(m)) {
				rest := s[k+len(m):]
				if nl := strings.IndexAny(rest, "\r\n"); nl >= 0 {
					rest = rest[:nl]
				}
				if hasUnpairedQuoteOnLine(rest, "'") {
					m = ""
				}
			}
			if m == "" {
				if q == "'" && startsContraction(s, k, 1) && unclosedContractionIsProse(sh, s, k, 1) {
					k++
					continue
				}
				for end < len(s) {
					next := end + 1
					if s[end] == '\r' && next < len(s) && s[next] == '\n' {
						next++
					}
					if startsHeaderLine(s, next) {
						break
					}
					end = len(s)
					if nl := strings.IndexAny(s[next:], "\r\n"); nl >= 0 {
						end = next + nl
					}
				}
				return end, ""
			}
			origK := k
			k += len(m)
			if k > end {
				end = len(s)
				if nl := strings.IndexAny(s[k:], "\r\n"); nl >= 0 {
					end = k + nl
				}
			}
			if k < end && strings.TrimSpace(s[from:origK]) == "" && !headerBeforeIsCookie(s, from) {
				rest := k
				for rest < end && (s[rest] == ' ' || s[rest] == '\t') {
					rest++
				}
				if rest < end && (s[rest] == ';' || s[rest] == '|' || (rest+1 < end && s[rest] == '&' && s[rest+1] == '&')) {
					return k, ""
				}
			}
			continue
		}
		if (outer != "" || inner != "") && end < len(s) {
			if !closesChecked {
				closesChecked = true
				switch {
				case joinedMultiline:
					closes = quotes != nil && quotes.joinedQuoteCloses(end, outer, inner)
				case quotes != nil:
					closes = quotes.quoteCloses(end, outer, inner)
				default:
					closes = outerQuoteCloses(sh, s, end, outer, inner)
				}
			}
			next := end + 1
			if s[end] == '\r' && next < len(s) && s[next] == '\n' {
				next++
			}
			if !closes && startsHeaderLine(s, next) {
				break
			}
			end = len(s)
			if nl := strings.IndexAny(s[next:], "\r\n"); nl >= 0 {
				end = next + nl
			}
			k = next
			for k == end && end < len(s) && !(s[end-1] == '\r' && s[end] == '\n') {
				if s[end] == '\r' && end+1 < len(s) && s[end+1] == '\n' {
					end++
				}
				next = end + 1
				if !closes && startsHeaderLine(s, next) {
					k = end
					break
				}
				end = len(s)
				if nl := strings.IndexAny(s[next:], "\r\n"); nl >= 0 {
					end = next + nl
				}
				k = next
			}
			if k < end {
				curLineStart = k
				continue
			}
			break
		}
		if outer == "" && inner == "" && canFold && end < len(s) {
			next := end + 1
			if s[end] == '\r' && next < len(s) && s[next] == '\n' {
				next++
			}
			if isFoldedHeaderLine(s, next, indentLen) {
				end = len(s)
				if nl := strings.IndexAny(s[next:], "\r\n"); nl >= 0 {
					end = next + nl
				}
				k = next
				curLineStart = k
				continue
			}
		}
		break
	}
	closedQ := ""
	if joinedMultiline {
		closedQ = entryInner + entryOuter
	}
	if (outer != "" || inner != "") && nestedEnd > 0 {
		return nestedEnd, closedQ
	}
	return end, closedQ
}

// nestedQuotedValueEnd checks whether s[k:], at the start of a token in the
// header value at s[from:] inside an open outer or inner quote on a line
// ending at end, opens a closed multiline quoted value of another quote kind
// whose interior does not close outer or inner, as the inner quotes of
// curl -H 'Authorization: "abc<LF><LF>SECRET"',
// msg="Authorization: 'abc<LF>part2: SECRET'", and
// curl -H 'Authorization: \'abc<LF>SECRET\” do, and returns the byte length
// of that quoted value and the end of the line on which it closes, or 0, 0.
func nestedQuotedValueEnd(s string, from, k, end int, outer, inner string) (int, int) {
	if k > from {
		prev, _ := utf8.DecodeLastRuneInString(s[:k])
		if !isDelimSpace(prev) {
			return 0, 0
		}
	}
	q := openingQuote(s, k)
	if q == "" || q == outer || q == inner {
		return 0, 0
	}
	if q == "'" && (isMidWordApostrophe(s, k) || isTrailingApostrophe(s, k)) {
		return 0, 0
	}
	m := findQuotedValue(s[k:])
	if m == "" || k+len(m) <= end {
		return 0, 0
	}
	if spanEndsWithNestedHeaderPrefix(s, from, k+len(m)-len(q)) {
		return 0, 0
	}
	body := m[len(q) : len(m)-len(q)]
	if inner != "" && strings.Contains(body, inner) {
		return 0, 0
	}
	vEnd := len(s)
	if nl := strings.IndexAny(s[k+len(m):], "\r\n"); nl >= 0 {
		vEnd = k + len(m) + nl
	}
	switch outer {
	case "'":
		if strings.IndexByte(body, '\'') >= 0 {
			return 0, 0
		}
		if q == `\'` && strings.IndexByte(s[k+len(m):vEnd], '\'') < 0 {
			return 0, 0
		}
	case `"`:
		if closedQuoteIn(`"`+body) > 0 {
			return 0, 0
		}
	}
	return len(m), vEnd
}

// headerBeforeIsCookie reports whether the header name directly preceding
// the separator before s[from:] is Cookie, Set-Cookie, or Cookie2.
func headerBeforeIsCookie(s string, from int) bool {
	p := from
	for p > 0 && (s[p-1] == ' ' || s[p-1] == '\t') {
		p--
	}
	if p == 0 || s[p-1] != ':' {
		return false
	}
	p--
	for p > 0 && (s[p-1] == ' ' || s[p-1] == '\t') {
		p--
	}
	end := p
	for p > 0 {
		c := s[p-1]
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_') {
			break
		}
		p--
	}
	return isCookieHeaderKey(s[p:end])
}

// crossesHeaderLine reports whether a line that starts in s[from:to], after a
// line break there, starts another header (see startsHeaderLine), so that a
// quote that closes on or after that line, as the one of
// X-Note: 'hi' closes the apostrophe of Authorization: Bob's, may not carry
// a header value over it.
func crossesHeaderLine(s string, from, to int) bool {
	for i := from; i < to; i++ {
		if (s[i] == '\n' || s[i] == '\r' && (i+1 >= len(s) || s[i+1] != '\n')) && startsHeaderLine(s, i+1) {
			return true
		}
	}
	return false
}

// startsHeaderLine reports whether s[i:], the start of a line, starts with a
// header name and its colon, as Content-Type: does. A line that starts with
// whitespace continues the value on the line before it (an obsolete line
// folding of RFC 9112), or is a line of the text in a quoted value, as the
// indented lines of x-api-key: "{<LF>  sub_field: x<LF>}" are, and does not
// start a header.
func startsHeaderLine(s string, i int) bool {
	n := 0
	for i+n < len(s) && (isTextKeyByte(s[i+n]) || n > 0 && s[i+n] == '.') {
		n++
	}
	if n == 0 || !('a' <= s[i]|0x20 && s[i]|0x20 <= 'z') {
		return false
	}
	j := skipHorizDelimSpace(s, i+n)
	return j < len(s) && s[j] == ':'
}

// keyValFinder finds the key-value pairs with a sensitive key in whole, one
// after the other, for replaceMatchesFunc, which calls find with the text
// that is left. It reads the name of a key and the delimiter after it first,
// and the value, which may run on to the end of a long word, only for a key
// that textKeyAllowedAt accepts, so that a run of keys that it does not, as in
// next_page_token:next_page_token:..., is read once rather than to its end
// from each key. A value that starts with a redaction placeholder is read
// only up to the end of the placeholder if the value may end there (see
// placeholderCut), so that a run of redacted pairs in one word, as in
// password=[REDACTED]token=[REDACTED]..., is read once too. It keeps the next
// key that textKeyNameRe finds from one call to the next, so that the text
// between the keys is searched once, however many keys that textKeyNameRe
// does not find (see findCamelKeyName) the text holds.
type keyValFinder struct {
	whole string
	// next is the submatch indexes in whole of the first key, with a value
	// after it, that textKeyNameRe finds at or after the offset of the last
	// search (see nextKeyName), if searched and not nil. Once a search has
	// found nothing, none is run again.
	next     []int
	searched bool
	// qNext is the submatch indexes in whole of the first sensitive quoted
	// key that the text patterns do not read whole (see scannedQuotedKey),
	// such as " user: DB_PASSWORD " or 'temp:a=b', opening at whole[qOpen:],
	// with a delimiter and value start after its closing quote, at or after
	// the offset of the last quoted-key search. Other quoted keys are found
	// by nextKeyName and findCamelKeyName.
	qNext     []int
	qOpen     int
	qSearched bool
	// splicedNext is the submatch indexes in whole of the first key, with a
	// value after it, whose name contains a backslash-escaped line break
	// (see nextSplicedKeyName), if splicedSearched and not nil.
	splicedNext     []int
	splicedSearched bool
}

// find returns the submatch indexes in rest, the end of f.whole, of the first
// key-value pair with a sensitive key in it, or nil (see pairAtIf).
func (f *keyValFinder) find(rest string) []int {
	pos := len(f.whole) - len(rest)
	if !f.qSearched || f.qNext != nil && f.qOpen < pos {
		f.qSearched = true
		f.qNext, f.qOpen = nextQuotedKeyName(f.whole, pos)
	}
	if f.qNext != nil && f.qOpen == pos {
		qLoc := make([]int, len(f.qNext))
		for i, v := range f.qNext {
			qLoc[i] = -1
			if v >= 0 {
				qLoc[i] = v - pos
			}
		}
		return cutQuotedPairAt(rest, qLoc)
	}
	if rest != "" && (isTextKeyByte(rest[0]) || rest[0] >= utf8.RuneSelf) && mayStartTextKeyName(rest) {
		if loc := textKeyNameAtRe.FindStringSubmatchIndex(rest); loc != nil && hasTextValueStart(rest[loc[1]:]) {
			return cutPairAtIf(textKeyAllowedAt(f.whole, pos+loc[2], pos+loc[3]), rest, loc, 0, textKeyValTokenAtRe)
		}
	}
	if !f.searched || f.next != nil && f.next[0] < pos {
		f.searched = true
		f.next = nextKeyName(f.whole, pos)
	}
	if !f.splicedSearched || f.splicedNext != nil && f.splicedNext[0] < pos {
		f.splicedSearched = true
		f.splicedNext = nextSplicedKeyName(f.whole, pos)
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
	if f.qNext != nil {
		if qLim := f.qNext[0] - pos; qLim < limit {
			limit = qLim
		}
	}
	if f.splicedNext != nil {
		if sLim := f.splicedNext[0] - pos; sLim < limit {
			limit = sLim
		}
	}
	// A key joined to its prefix at a camelCase boundary, as in dbPassword,
	// has no word boundary before its credential name; it starts before the
	// key at a word boundary, if there is one.
	if camel := findCamelKeyName(rest, limit); camel != nil {
		loc = camel
	}
	var name, start int
	if loc != nil {
		// Extend the key back over the prefix of a compound name. The match
		// starts at the key.
		name = loc[2]
		start = name
		for start > 0 && isTextKeyByte(rest[start-1]) {
			start--
		}
	}
	// A quoted key that the text patterns do not read whole is read from its
	// opening quote, also when a key that they read starts at its first byte,
	// as temp:a does in "temp:a=b c": "x", where they would read "b" as the
	// value and leave "x" in place.
	if f.qNext != nil && (loc == nil || f.qNext[0]-pos <= start) &&
		(f.splicedNext == nil || f.qNext[0] <= f.splicedNext[0]) {
		qLoc := make([]int, len(f.qNext))
		for i, v := range f.qNext {
			qLoc[i] = -1
			if v >= 0 {
				qLoc[i] = v - pos
			}
		}
		return cutQuotedPairAt(rest, qLoc)
	}
	if f.splicedNext != nil && (loc == nil || f.splicedNext[0]-pos <= start) {
		sLoc := make([]int, len(f.splicedNext))
		for i, v := range f.splicedNext {
			sLoc[i] = -1
			if v >= 0 {
				sLoc[i] = v - pos
			}
		}
		return sLoc
	}
	if loc == nil {
		return nil
	}
	loc[0], loc[2] = start, start
	return cutPairAtIf(textKeyAllowedAt(f.whole, pos+start, pos+loc[3]), rest, loc, name, textKeyValBareAtRe)
}

// cutQuotedPairAt returns what quotedPairAtIf(true, s, loc) does, but with a
// value that a redaction placeholder starts read only up to the end of the
// placeholder if the value may end there (see placeholderCut). As the value
// is read after the delimiter whitespace, the match of all of s reads the
// same value start, and the bare value that runs on from it.
func cutQuotedPairAt(s string, loc []int) []int {
	vs := skipDelimSpace(s, loc[7])
	if e := placeholderCut(s, vs); e >= 0 {
		if pair := quotedPairAtIf(true, s[:e], loc); pair[8] == vs && pair[9] == e {
			return pair
		}
	}
	return quotedPairAtIf(true, s, loc)
}

// cutPairAtIf returns what pairAtIf does, but with a value that a redaction
// placeholder starts read only up to the end of the placeholder if the value
// may end there (see placeholderCut). The match of s cut there is taken only
// if it reads the key of loc and the value from where the placeholder
// starts. The match of all of s then reads them too, and the bare value that
// runs on from there, as none of the keys that re may read at s[at:] runs
// into the placeholder: the name of a key holds no "[" but in a "temp:" name
// (see bareSensitiveKeyRegex), which ends at the first "=" or ":", at or
// before the delimiter of loc.
func cutPairAtIf(allowed bool, s string, loc []int, at int, re *regexp.Regexp) []int {
	if allowed {
		vs := skipDelimSpace(s, loc[7])
		if e := placeholderCut(s, vs); e >= 0 {
			if pair := pairAtIf(true, s[:e], loc, at, re); pair[8] == vs && pair[9] == e {
				return pair
			}
		}
	}
	return pairAtIf(allowed, s, loc, at, re)
}

// quotedKeyAllowed reports whether raw, the complete unquoted text of a
// single-line quoted key name, names a sensitive key once interior delimiter
// whitespace padding around it is trimmed. But for the keys that hold a quote,
// "\" or a line break, it accepts what isSensitiveKey accepts, so that a
// quoted key is read as a map key or a struct field name is: a "temp:" key
// after an optional state prefix and whitespace, including one whose name is
// empty or holds "=", ":", ",", "{", "}" or whitespace, as in temp:,
// " temp:a=b ", user:temp:a:b, 'temp:a,b', 'temp:cache/{user}', 'temp:a}b' or
// " user: temp:a b ", and a credential name after a state prefix and
// whitespace, as in " user: DB_PASSWORD ". The quotes around the key, which
// the caller has found, tell where it ends, so a comma or brace in it, as in
// {'label': 'keep', 'temp:a,b': 'x'} or {'temp:cache/{user}': 'x'}, is a byte
// of its name (whereas unquoted text patterns stop before an unmatched "}").
// It accepts no key that holds a quote, "\" or a line break, whatever
// isSensitiveKey says of it: single-line quoted-key scanning (see
// scanQuotedKeyToken) reads those as the structure of the text around a key.
func quotedKeyAllowed(raw string) bool {
	trimmed := strings.TrimFunc(raw, isDelimSpace)
	if trimmed == "" || strings.ContainsAny(trimmed, `"'`+"\\\r\n") {
		return false
	}
	if isSensitiveKey(trimmed) {
		return true
	}
	if strings.ContainsFunc(trimmed, isDelimSpace) {
		// Prose, or a name that only holds a credential name, as
		// "my password" does.
		return false
	}
	rest, _ := cutStatePrefix(trimmed)
	return !isPageTokenName(trimmed) && isBareTextKey(rest)
}

// scanQuotedKeyToken scans from the opening quote at s[i:] (of length
// len(quote)) to a matching closing quote on the same line, stopping early if
// a line break or an inner quote appears inside. A comma or brace does not
// stop it: the closing quote of one string and the opening quote of the next,
// as in {'a': 'b', 'c': 'd'}, quote only the text between two strings, which
// names no key. It returns (keyStart, closeQuoteStart, nextPos) where
// closeQuoteStart is -1 when no single-line quoted key token closes cleanly,
// and nextPos > i is the next offset at which another quote can begin, keeping
// full-text scans O(n).
func scanQuotedKeyToken(s string, i int, quote string) (int, int, int) {
	keyStart := i + len(quote)
	p := keyStart
	for p < len(s) {
		c := s[p]
		if c == '\r' || c == '\n' {
			return keyStart, -1, p + 1
		}
		if strings.HasPrefix(s[p:], quote) {
			return keyStart, p, p
		}
		if c == '\\' {
			if p+1 < len(s) && (s[p+1] == '"' || s[p+1] == '\'') {
				return keyStart, -1, p
			}
			if p+1 < len(s) && s[p+1] == '\\' {
				return keyStart, -1, p + 2
			}
			return keyStart, -1, p + 1
		}
		if c == '"' || c == '\'' {
			return keyStart, -1, p
		}
		p++
	}
	return keyStart, -1, max(p, keyStart)
}

// quotedKeyScanAt inspects s at an opening quote quote at s[i:]. When s[i:] is
// a single-line quoted key token followed outside its closing quote by ":" or
// "=" and the start of a value, it returns the 8-index submatch slice matching
// textKeyNameAtRe's layout ([keyStart, delimEnd, keyStart, closeQ, closeQ,
// closeQEnd, delim, delimEnd]) in s, along with nextPos. The key may hold
// whitespace, as " user: DB_PASSWORD " does: whether it is sensitive is for
// quotedKeyAllowed to tell.
func quotedKeyScanAt(s string, i int, quote string) ([]int, int) {
	keyStart, closeQ, nextPos := scanQuotedKeyToken(s, i, quote)
	if closeQ < 0 {
		return nil, nextPos
	}
	if strings.TrimFunc(s[keyStart:closeQ], isDelimSpace) == "" {
		return nil, nextPos
	}
	qEnd := closeQ + len(quote)
	delim := skipDelimSpace(s, qEnd)
	if delim >= len(s) || (s[delim] != ':' && s[delim] != '=') || !hasTextValueStart(s[delim+1:]) {
		return nil, nextPos
	}
	return []int{keyStart, delim + 1, keyStart, closeQ, closeQ, qEnd, delim, delim + 1}, nextPos
}

// quotedKeyAtStart matches a single-line quoted key and its delimiter at s[0:],
// returning the 8-index submatch layout of textKeyNameAtRe, or nil.
func quotedKeyAtStart(s string) []int {
	quote := openingQuote(s, 0)
	if quote == "" || (quote == "'" && isApostrophe(s, 0, 2, 0)) {
		return nil
	}
	loc, _ := quotedKeyScanAt(s, 0, quote)
	return loc
}

// quotedPairAtStart matches a single-line quoted key, its delimiter, and (when
// quotedKeyAllowed accepts the key) its value at s[0:], agreeing with
// quotedKeyAtStart on submatch boundaries [0..7].
func quotedPairAtStart(s string) []int {
	loc := quotedKeyAtStart(s)
	if loc == nil {
		return nil
	}
	return quotedPairAtIf(quotedKeyAllowed(s[loc[2]:loc[3]]), s, loc)
}

// quotedPairUncheckedAtStart matches a single-line quoted key, delimiter, and
// value at s[0:] when the caller (nestsSensitiveKey) has already classified
// the key as sensitive.
func quotedPairUncheckedAtStart(s string) []int {
	loc := quotedKeyAtStart(s)
	if loc == nil {
		return nil
	}
	return quotedPairAtIf(true, s, loc)
}

// quotedPairAtIf extends an 8-index quoted-key match loc in s with the value
// submatch [valStart, valEnd] when allowed is true, or [-1, -1] otherwise.
func quotedPairAtIf(allowed bool, s string, loc []int) []int {
	if allowed {
		if vLoc := textValueAtRe.FindStringSubmatchIndex(s[loc[7]:]); vLoc != nil {
			return append(loc[:8:8], loc[7]+vLoc[2], loc[7]+vLoc[3])
		}
	}
	return append(loc[:8:8], -1, -1)
}

// scannedQuotedKey reports whether raw, the unquoted text of a quoted key,
// is one that only nextQuotedKeyName reads whole: one padded with delimiter
// whitespace, one that holds whitespace, as " user: DB_PASSWORD " and
// "temp:a b" do, or a "temp:" key, after an optional state prefix, whose name
// is empty or holds "=", ":", ",", "{" or "}", as temp:, temp:a=b, temp:a:b,
// user:temp:a=b, temp:a,b and temp:cache/{user} do. The text patterns read a
// "temp:" key as a name up to its first "=", ":", ",", "{" or "}", that
// delimiter and a value: of such a quoted key, they would read part of the key
// as the value and leave the value after its closing quote in place, or redact
// it with the closing quote and so break the key, or, at a comma or brace,
// read no pair at all. A "temp:" key whose name holds none of them, as
// temp:scratch, is left to them: they read it up to its closing quote. Whether
// the key is sensitive is for quotedKeyAllowed to tell; a quoted key with no
// delimiter after it, as the whole of 'temp:cache=secret' is, is left to the
// text patterns.
func scannedQuotedKey(raw string) bool {
	if strings.ContainsFunc(raw, isDelimSpace) {
		return true
	}
	rest, _ := cutStatePrefix(raw)
	if len(rest) < len(session.KeyPrefixTemp) || !strings.EqualFold(rest[:len(session.KeyPrefixTemp)], session.KeyPrefixTemp) {
		return false
	}
	name := rest[len(session.KeyPrefixTemp):]
	return name == "" || strings.ContainsAny(name, "=:,{}")
}

// nextQuotedKeyName returns the 8-index submatch slice in s and opening
// quote index of the first sensitive quoted key that the text patterns do not
// read whole (see scannedQuotedKey), with a delimiter and value start after
// its closing quote, at or after s[pos:], scanning each inter-quote span at
// most once.
func nextQuotedKeyName(s string, pos int) ([]int, int) {
	for i := pos; i < len(s); {
		quote := openingQuote(s, i)
		if quote == "" || (quote == "'" && isApostrophe(s, i, 2, 0)) {
			i++
			continue
		}
		keyStart, closeQ, nextPos := scanQuotedKeyToken(s, i, quote)
		if closeQ > keyStart && scannedQuotedKey(s[keyStart:closeQ]) {
			qEnd := closeQ + len(quote)
			delim := skipDelimSpace(s, qEnd)
			if delim < len(s) && (s[delim] == ':' || s[delim] == '=') &&
				quotedKeyAllowed(s[keyStart:closeQ]) && hasTextValueStart(s[delim+1:]) {
				return []int{keyStart, delim + 1, keyStart, closeQ, closeQ, qEnd, delim, delim + 1}, i
			}
		}
		i = max(nextPos, i+len(quote))
	}
	return nil, -1
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
		loc := findTextKeyName(s[pos:])
		if loc == nil {
			return nil
		}
		for i, v := range loc {
			if v >= 0 {
				loc[i] = v + pos
			}
		}
		if hasTextValueStart(s[loc[1]:]) {
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

// nextSplicedKeyName returns the submatch indexes in s of the first key whose
// name contains a backslash-escaped line break (as in DB_PASS\<LF>WORD=SECRET
// or DB_PASSWORD\<LF>=SECRET), at or after s[pos:], or nil. The regex
// textKeyNameRe cannot match across line breaks, so keys with line splices in
// their names are found here. It scans for \ followed by \n or \r\n, then
// tries to read an identifier key around the splice by stripping the splice
// (see stripShellLineSplices) and checking that the stripped name is a
// sensitive key (see textKeyAllowed) followed by = or : and a value start
// (see hasTextValueStart). The returned loc is the same 10-element format as
// nextKeyName: [matchStart, matchEnd, keyStart, keyEnd, quoteStart, quoteEnd,
// delimStart, delimEnd, valStart, valEnd], where quoteStart == quoteEnd
// (no quote) and matchStart == keyStart (key at the start of the match).
func nextSplicedKeyName(s string, pos int) []int {
	for i := pos; i < len(s); {
		bs := strings.IndexByte(s[i:], '\\')
		if bs < 0 {
			return nil
		}
		i += bs
		if i+1 >= len(s) {
			return nil
		}
		// Must be a line splice: \ followed by \n or \r\n.
		if s[i+1] != '\n' && s[i+1] != '\r' {
			i += 2
			continue
		}
		spliceLen := 2
		if s[i+1] == '\r' && i+2 < len(s) && s[i+2] == '\n' {
			spliceLen = 3
		}
		// Try to read an identifier key that contains this splice. Scan
		// backward from i for the start of the key name: text key bytes
		// (letters, digits, _, -) and line splices.
		keyStart := i
		for keyStart > pos && isTextKeyByte(s[keyStart-1]) {
			keyStart--
		}
		// Scan forward from i+spliceLen for the end of the key name: text
		// key bytes and more line splices.
		keyEnd := i + spliceLen
		for keyEnd < len(s) {
			c := s[keyEnd]
			if isTextKeyByte(c) {
				keyEnd++
				continue
			}
			if c == '\\' && keyEnd+1 < len(s) && (s[keyEnd+1] == '\n' || s[keyEnd+1] == '\r') {
				s2 := 2
				if s[keyEnd+1] == '\r' && keyEnd+2 < len(s) && s[keyEnd+2] == '\n' {
					s2 = 3
				}
				keyEnd += s2
				continue
			}
			break
		}
		// Strip line splices from the key name and check if it is sensitive.
		rawKey := s[keyStart:keyEnd]
		stripped := stripShellLineSplices(rawKey)
		if stripped == "" || !textKeyAllowed(stripped) {
			i = keyEnd
			continue
		}
		// Check for a delimiter (= or :) after the key, with optional
		// horizontal whitespace and a quote between the key and delimiter.
		quoteStart := keyEnd
		quoteEnd := keyEnd
		d := skipHorizDelimSpace(s, keyEnd)
		if d < len(s) && (s[d] == '=' || s[d] == ':') {
			// Check for a value start after the delimiter.
			afterDelim := skipDelimSpace(s, d+1)
			if hasTextValueStart(s[afterDelim:]) {
				valEnd := splicedValueEnd(s, afterDelim)
				return []int{
					keyStart, valEnd, keyStart, keyEnd,
					quoteStart, quoteEnd, d, d + 1, afterDelim, valEnd,
				}
			}
		}
		// Also check for a quote right after the key before the delimiter.
		if d < len(s) && (s[d] == '"' || s[d] == '\'') {
			quoteEnd = d + 1
			d2 := skipHorizDelimSpace(s, quoteEnd)
			if d2 < len(s) && (s[d2] == '=' || s[d2] == ':') {
				afterDelim := skipDelimSpace(s, d2+1)
				if hasTextValueStart(s[afterDelim:]) {
					valEnd := splicedValueEnd(s, afterDelim)
					return []int{
						keyStart, valEnd, keyStart, keyEnd,
						quoteStart, quoteEnd, d2, d2 + 1, afterDelim, valEnd,
					}
				}
			}
		}
		i = keyEnd
	}
	return nil
}

// splicedValueEnd returns where the value that nextSplicedKeyName reads at
// s[start:] ends: a quoted value at its closing quote, and a bare value as
// bareValueEnd reads it, except that a value that starts with a redaction
// placeholder is read only up to the end of the placeholder if the value may
// end there (see placeholderCut), as keyValFinder reads the values of the
// keys that textKeyNameRe finds. bareValueEnd runs on past line splices, so
// a run of redacted spliced pairs in one word would otherwise be read from
// each key to the end of the word.
func splicedValueEnd(s string, start int) int {
	if cut := placeholderCut(s, start); cut >= 0 {
		return cut
	}
	if q := findQuotedValue(s[start:]); q != "" {
		return start + len(q)
	}
	return bareValueEnd(s, start)
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
			endNonSep := e
			for endNonSep > t && (s[endNonSep-1] == '.' || s[endNonSep-1] == '_' || s[endNonSep-1] == '-') {
				endNonSep--
			}
			for i := camelScanStart(s, t, endNonSep); i < min(endNonSep, limit); i++ {
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
// textKeyNameCamelAtRe matches at s[i:], if a value follows them, or nil. A
// match starts with a key name of textKeyNames (see keyNameSet).
func keyNameAt(s string, i int) []int {
	if !textKeyNames.mayStartAt(s, i) {
		return nil
	}
	loc := textKeyNameCamelAtRe.FindStringSubmatchIndex(s[i:])
	if loc == nil || !hasTextValueStart(s[i+loc[1]:]) {
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

// replaceMatchesFunc replaces each match that find finds in s with the text
// returned by repl, which receives the submatch indexes of the match and also
// returns where the replaced text ends; it may extend past the match. find
// returns the submatch indexes of the first match in the text that it is
// given, as FindStringSubmatchIndex does, or nil. The search resumes where
// the replaced text ends, so the replaced spans never overlap. rec, if not
// nil, records which bytes of the result are copies of s (see passRecorder).
func replaceMatchesFunc(s string, find func(string) []int, repl func(s string, loc []int) (string, int), rec *passRecorder) string {
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
		rec.copied(last, loc[0]-last)
		rec.replaced(s, loc[0], end, replacement)
		last = end
		pos = max(end, loc[0]+1)
	}
	if last == 0 {
		rec.reset()
		rec.copied(0, len(s))
		return s
	}
	b.WriteString(s[last:])
	rec.copied(last, len(s)-last)
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
func redactBearerToken(s string, loc []int, quotes *openQuotes, pairs *sensitivePairSpans) (string, int) {
	// sh reads s; the prefixes of s that the token is read in below share
	// it (see shellAssignScanner.use).
	sh := quotes.shellFor(s)
	end := loc[5]
	quoted := quotes.before(s, loc[4])
	innerQuoted := quotes.innerAt(s, loc[4])
	// A backslash that ends the token and escapes a quote that closes the
	// string that the token is in, or one nested in it (see innerAt), is a
	// byte of that quote, as the one before the closing \" of
	// {\"auth\": \"Bearer abc\"} is, so it is left in place with the quote.
	if end < len(s) && (quoted|innerQuoted)&quoteKind(s[end-1:end+1]) != 0 {
		n := 0
		for n < end-loc[4] && s[end-1-n] == '\\' {
			n++
		}
		if n%4 == 1 && end-1 > loc[4] {
			end--
		}
	}
	if end < len(s) && (s[end] == '"' || s[end] == '\'') && end-1 > loc[4] && s[end-1] == '\\' {
		n := 0
		for n < end-loc[4] && s[end-1-n] == '\\' {
			n++
		}
		if n%4 == 1 && startsQuotedKey(s[end-1:], quoted) {
			end--
		}
	}
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
		if quoted&quoteKind(q) != 0 && !closes && pairs.covers(s, quotes.openPos, loc[4]) {
			quotes.skip(s, loc[4])
			return s[loc[0]:loc[4]], loc[4]
		}
		if quoted&quoteKind(q) != 0 {
			afterClose0 := quoted &^ quoteKind(q)
			if q2 := openingQuote(s, loc[4]+len(q)); q2 != "" && q2 != q && !startsRedactedPair(s, loc[4]+len(q), afterClose0) {
				afterClose := afterClose0
				q2Start := loc[4] + len(q)
				if m := quotedValueRe.FindString(s[q2Start:]); m != "" {
					end = q2Start + len(m)
					if !quotedValueEnds(s[q2Start:], len(m), afterClose) || opensDelimitedValue(sh, s, end, afterClose) {
						end = runOnValueEnd(sh, s, q2Start, end, 0, afterClose)
					}
				} else {
					end = len(s)
				}
				// Extend past a non-sensitive quoted key whose closing quote is
				// followed by a delimiter and a bare value, as "label": ZQXSEKRETN
				// is after Bearer abc. When quotedValueEnds returns true (the
				// closing quote followed by : reads as a sentence boundary),
				// runOnValueEnd is not called, so check the delimiter here.
				if end > q2Start && end < len(s) {
					delim := skipHorizDelimSpace(s, end)
					if delim < len(s) && (s[delim] == ':' || s[delim] == '=') {
						end = delim + 1
					}
				}
				if end > q2Start && end < len(s) && (s[end-1] == ':' || s[end-1] == '=') {
					if sp := skipHorizDelimSpace(s, end); sp > end && sp < len(s) {
						if bEnd := bareValueEnd(s, sp); bEnd > sp {
							end = sp + len(trimUnopenedClosers(s[sp:bEnd]))
						}
					}
				}
				end = laterBearerCut(s, loc[5], end)
				quotes.pos = q2Start
				quotes.active = afterClose
				quotes.bs = 0
				closeOuter := ""
				if quotes.before(s, end)&quoteKind(q) == 0 {
					closeOuter = q
				}
				return s[loc[2]:loc[3]] + " " + redactedValue + closeOuter, end
			}
			if _, valQuote := unquoteValue(value); valQuote == "" {
				if endsString(s, loc[4], q, quoted) {
					quotes.skip(s, loc[4])
					return s[loc[0]:loc[4]], loc[4]
				}
				if rest := s[loc[4]+len(q):]; len(rest) >= 2 && strings.EqualFold(rest[:2], "of") {
					if r, _ := utf8.DecodeRuneInString(rest[2:]); len(rest) == 2 || unicode.IsSpace(r) {
						quotes.skip(s, end)
						return s[loc[0]:end], end
					}
				}
				quotes.drop()
				quotes.skip(s, end)
				return s[loc[2]:loc[3]] + " " + redactedValue, end
			}
		}
		if end > loc[4]+len(q) && !quotedValueEnds(s[loc[4]:], end-loc[4], quoted) {
			if quote := textBetweenStrings(sh, s, loc[4], end, quoted); quote != "" {
				next := quotes.unpaired(s, loc[4], quote)
				return s[loc[0]:next], next
			}
			end = textValueEnd(sh, s, loc[4], end, 0, quoted)
			value = s[loc[4]:end]
		}
		if closes {
			quotes.drop()
		}
	}
	if strings.HasPrefix(value, redactedValue) && (startsRedactedPair(s, loc[4]+len(redactedValue), quoted) || startsBearerKeyword(s, loc[4]+len(redactedValue))) {
		redEnd := loc[4] + len(redactedValue)
		quotes.skip(s, redEnd)
		return s[loc[0]:redEnd], redEnd
	}
	closeOuter := ""
	if openingQuote(s, loc[4]) == "" {
		if q := openingQuote(s, end); q != "" {
			afterClose2 := quoted &^ quoteKind(q)
			if q2 := openingQuote(s, end+len(q)); quoted&quoteKind(q) != 0 && q2 != "" && !startsRedactedPair(s, end+len(q), afterClose2) {
				afterClose := afterClose2
				q2Start := end + len(q)
				if m := quotedValueRe.FindString(s[q2Start:]); m != "" {
					end = q2Start + len(m)
					if !quotedValueEnds(s[q2Start:], len(m), afterClose) || opensDelimitedValue(sh, s, end, afterClose) {
						end = runOnValueEnd(sh, s, q2Start, end, 0, afterClose)
					}
				} else {
					end = len(s)
				}
				// Extend past a non-sensitive quoted key whose closing quote is
				// followed by a delimiter and a bare value, as above.
				if end > q2Start && end < len(s) {
					delim := skipHorizDelimSpace(s, end)
					if delim < len(s) && (s[delim] == ':' || s[delim] == '=') {
						end = delim + 1
					}
				}
				if end > q2Start && end < len(s) && (s[end-1] == ':' || s[end-1] == '=') {
					if sp := skipHorizDelimSpace(s, end); sp > end && sp < len(s) {
						if bEnd := bareValueEnd(s, sp); bEnd > sp {
							end = sp + len(trimUnopenedClosers(s[sp:bEnd]))
						}
					}
				}
				end = laterBearerCut(s, loc[5], end)
				quotes.pos = q2Start
				quotes.active = afterClose
				quotes.bs = 0
				if quotes.before(s, end)&quoteKind(q) == 0 {
					closeOuter = q
				}
				value = s[loc[4]:end]
			} else if quoted&quoteKind(q) != 0 && (strings.HasSuffix(s[:loc[2]], q) || !startsContraction(s, end, len(q))) {
				// The quote closes the string that the token is in. A word
				// joined to it is read as gluedWordEnd reads it.
				end, closeOuter = gluedWordEnd(sh, s, loc, end, q, quoted, quotes)
				value = s[loc[4]:end]
			} else if !endsString(s, end, q, quoted) && !startsQuotedKey(s[end:], quoted) && (innerQuoted&quoteKind(q) == 0 || !closersEndValue(s, end+len(q), quoted)) {
				// A quote that closes a string nested in the one that the
				// token is in (see innerAt) and that closing brackets that
				// end the value follow (see closersEndValue), as after the
				// token of msg="{\"auth\": \"Bearer a\"}{\"b\": 1}", ends the
				// value without this branch: the quote is left in place
				// with the brackets, and the object after them is read on
				// its own.
				//
				// The value is read only up to the quote that ends the
				// string that the token is in, or a string nested in it,
				// on its line (see endOnLine), as the closing " of
				// {"auth": "Bearer a'b"} does, so that a quote of the value
				// that nothing closes in that string, as the ' is, does
				// not pair with a quote of a later string, as the ' of the
				// next object that the text glues on would, and read the
				// value on through the rest of the text for each token.
				// The value of a token in a string that does not end on
				// its line is read on as before, and clamped at the quote
				// that ends that string if a later line has one.
				allQuoted := quoted | innerQuoted
				t := s
				at := -1
				if allQuoted != 0 {
					if at = quotes.endOnLine(s, end+len(q), allQuoted); at >= 0 {
						t = s[:at]
					}
				}
				bEnd := bareValueEnd(t, end)
				vEnd, next := joinedQuoteEnd(sh, t, end, bEnd, 0, allQuoted)
				if vEnd > bEnd {
					// When the string does not end on the token's line, a
					// quote on a later line that ends it also ends a value
					// that runs on past the bare value after a ' that no
					// quote closes, which would otherwise run the value on
					// to the end of the text.
					if sEnd := stringEndIn(s, end+len(q), vEnd, allQuoted); sEnd < vEnd {
						vEnd = sEnd
						bEnd = min(bEnd, sEnd)
					}
				}
				if vEnd > bEnd {
					end = textValueEnd(sh, t, loc[4], vEnd, 0, allQuoted)
					value = s[loc[4]:end]
				} else if next > end {
					if oq := openingQuote(s, next); oq != "" && endsString(s, next, oq, allQuoted) {
						end = next
					} else {
						end = runOnValueEnd(sh, t, end, next, 0, allQuoted)
					}
					value = s[loc[4]:end]
				} else if end+len(q) < bEnd && s[end-1] != ':' && s[end-1] != '=' && s[end-1] != '\\' && !startsContraction(s, end, len(q)) {
					origEnd := end
					// A quote that ends the string that the token is in ends
					// the bare value and is left in place, as the closing "
					// of {"auth": "Bearer a'b"} is.
					bEnd = stringEndIn(s, origEnd+len(q), bEnd, allQuoted)
					tEnd := origEnd + len(trimUnopenedClosers(s[origEnd:bEnd]))
					if tEnd > origEnd+len(q) {
						quotes.before(s, origEnd)
						quotes.pos = origEnd + len(q)
						quotes.bs = 0
						end = tEnd
						value = s[loc[4]:end]
					}
				}
				// A value that runs up to the quote that ends the string,
				// as the a'b of {"auth": "Bearer a'b"}SECRET does, ends as
				// a token that the quote follows does: a word joined to the
				// quote, as the }SECRET is, is read as gluedWordEnd reads
				// it.
				if at >= 0 && end == at {
					if cq := openingQuote(s, at); allQuoted&quoteKind(cq) != 0 {
						end, closeOuter = gluedWordEnd(sh, s, loc, end, cq, quoted, quotes)
						value = s[loc[4]:end]
					}
				}
			}
		} else if tEnd := textValueEnd(sh, s, loc[4], end, 0, quoted); tEnd != end {
			end = tEnd
			value = s[loc[4]:end]
		}
	}
	if cut := laterBearerCut(s, loc[5], end); cut < end {
		end = cut
		value = s[loc[4]:end]
	}
	quotes.skip(s, end)
	if inner, _ := unquoteValue(value); isRedactedValue(inner) {
		return s[loc[0]:end], end
	}
	return s[loc[2]:loc[3]] + " " + redactedValue + closeOuter, end
}

// gluedWordEnd returns where the redaction of a bearer token ends whose value
// ends at the quote q at s[end:], which closes the string that the token is
// in or a string nested in it (quoted holds the strings open before the
// token, see openQuotes.before), and the quote to write after the redacted
// value. A pair whose value the text patterns redact on its own, at the quote
// or after brackets joined to it (see startsPairAfterBrackets), or a later
// bearer credential that is joined to the quote is left in place after it, as
// the DB_PASSWORD='b c' of x='Bearer a'DB_PASSWORD='b c' and the
// password: c of x='Bearer a'}{password: c} are, so that it is matched and
// redacted on its own; any other word joined to the quote is the rest of the
// token as a shell reads it, and is redacted with it, quoted parts and all
// (see runOnWordEnd), as the mid' b' of x='Bearer a'mid' b' is, but only up to
// the end of a string open before the token. Closing brackets that end the
// value after the quote (see followsValueBoundary), as the } before the next
// object of {"auth": "Bearer a"}{"user": "b"} does, are not such a word, and
// are left in place with the quote. When the word is redacted, quotes is set
// to read on from its end.
func gluedWordEnd(sh *shellAssignScanner, s string, loc []int, end int, q string, quoted quoteKinds, quotes *openQuotes) (int, string) {
	afterClose := quoted &^ quoteKind(q)
	tailStart := end + len(q)
	if closersEndValue(s, tailStart, afterClose) || startsPairAfterBrackets(s, tailStart, afterClose) || startsBearerKeyword(s, tailStart) {
		return end, ""
	}
	wordEnd, closes := runOnWord(sh, s, tailStart, afterClose)
	end = stringEndIn(s, tailStart, wordEnd, afterClose)
	// Every quoted part of a word that ends on its line closes in it, as the
	// shell reads the word (see runOnWord), so the string that the token is
	// in stays closed after the word even where a quote of the word would
	// read as an apostrophe on its own, as the one after mid in
	// x='Bearer a'mid' b9' would.
	closed := closes && end == wordEnd && !strings.ContainsAny(s[tailStart:end], "\r\n")
	end = tailStart + len(trimUnopenedClosers(s[tailStart:end]))
	if cut := laterBearerCut(s, loc[5], end); cut < end {
		end = cut
		closed = false
	}
	quotes.pos = tailStart
	quotes.active = afterClose
	quotes.bs = 0
	if closed {
		quotes.pos = end
		return end, q
	}
	if quotes.before(s, end)&quoteKind(q) == 0 {
		return end, q
	}
	return end, ""
}

// closersEndValue reports whether s[i:], the text after the quote that closes
// a value, starts with closing brackets that end the value (see
// followsValueBoundary; quoted holds the strings open after the quote), as
// the } before the next object of {"auth": "Bearer a"}{"user": "b"} does.
func closersEndValue(s string, i int, quoted quoteKinds) bool {
	return i < len(s) && strings.IndexByte(")]}", s[i]) >= 0 && followsValueBoundary(s[i:], quoted)
}

// startsPairAfterBrackets reports whether a key-value pair that the text
// patterns redact on its own starts at s[i:], the text right after the quote
// that closes a value, or after a run of brackets (")]}{[(") there, as the
// password: c of x='Bearer a'}{password: c} does: a pair that
// startsRedactedPair reads (quoted holds the strings open there), or a
// sensitive key, a delimiter and a value that keyValFinder reads at that
// offset (see textKeyValTokenAtRe), as with whitespace before the value. The
// key is then not read as part of a word joined to the value, which would
// leave the value of the pair in place.
func startsPairAfterBrackets(s string, i int, quoted quoteKinds) bool {
	k := i
	for k < len(s) && strings.IndexByte(")]}{[(", s[k]) >= 0 {
		k++
	}
	if startsRedactedPair(s, k, quoted) {
		return true
	}
	rest := s[k:]
	if rest == "" || !isTextKeyByte(rest[0]) && rest[0] < utf8.RuneSelf || !mayStartTextKeyName(rest) {
		return false
	}
	loc := textKeyNameAtRe.FindStringSubmatchIndex(rest)
	if loc == nil || !hasTextValueStart(rest[loc[1]:]) {
		return false
	}
	return cutPairAtIf(textKeyAllowedAt(s, k+loc[2], k+loc[3]), rest, loc, 0, textKeyValTokenAtRe)[8] >= 0
}

// laterBearerCut returns where the keyword of a later bearer credential
// starts if it starts in s[from:end], the part of a value that is read on
// past the end of a bearer token at from, but its token runs on past end, as
// the second Bearer of Bearer a';'Bearer b'c does when the value runs on
// over the joined ';' to the next whitespace; otherwise it returns end. The
// value then ends before that keyword, so that the later credential is
// matched and redacted on its own rather than redacted without its token,
// which would leave the token in place. Only a value that holds the keyword
// is searched for a later credential, so the cost is linear in the length of
// the value unless it holds one. A keyword that an escaped line break or tab
// comes before, as the second Bearer of "{\"a\": \"Bearer x\"}\nBearer y"
// does in JSON text quoted into a string, starts a credential too, although
// no word boundary comes before it.
func laterBearerCut(s string, from, end int) int {
	if end <= from || !bearerKeywordRe.MatchString(s[from:end]) {
		return end
	}
	cut := end
	for at := from; at < end; {
		m := findBearerToken(s[at:])
		if m == nil || at+m[2] >= end {
			break
		}
		if at+m[5] > end {
			cut = at + m[2]
			break
		}
		at += m[1]
	}
	for _, k := range bearerKeywordRe.FindAllStringIndex(s[from:cut], -1) {
		p := from + k[0]
		if p >= 2 && s[p-2] == '\\' && strings.IndexByte("nrt", s[p-1]) >= 0 {
			if m := bearerTokenAtRe.FindStringSubmatchIndex(s[p:]); m != nil && p+m[5] > end {
				return p
			}
		}
	}
	return cut
}

// startsBearerKeyword reports whether s[i:] starts with the keyword of a
// bearer credential, "bearer" in any case followed by whitespace, as the
// value that a bearer token runs on into does where laterBearerCut ends it
// before a later credential, so that the redaction placeholder that such a
// value leaves before the later credential, as in
// Bearer [REDACTED]Bearer [REDACTED], ends there when the text is scrubbed
// again, rather than being read with the keyword as one value.
func startsBearerKeyword(s string, i int) bool {
	const keyword = "bearer"
	if len(s)-i <= len(keyword) || !strings.EqualFold(s[i:i+len(keyword)], keyword) {
		return false
	}
	r, _ := utf8.DecodeRuneInString(s[i+len(keyword):])
	return unicode.IsSpace(r)
}

// stringEndIn returns the index of the first quote in s[from:to] that ends a
// string open before it (quoted holds those, see endsString), as the closing
// " of {"auth": "Bearer a'b"} does, or to if none does, so that a bare value
// read on past a bearer token stops at the end of the string that the token
// is in rather than redacting the quote that closes it. The quotes are read
// after the backslashes before them (see quoteAfterBackslashes): an escaped
// quote is read as one quote, which ends only a string that an escaped quote
// opened. from follows a quote, so no backslash before it escapes a quote.
func stringEndIn(s string, from, to int, quoted quoteKinds) int {
	if quoted == 0 {
		return to
	}
	bs := 0
	for k := from; k <= to && k < len(s); k++ {
		switch s[k] {
		case '\\':
			bs++
			continue
		case '"', '\'':
			if q, at := quoteAfterBackslashes(s, k, bs); at < to && quoted&quoteKind(q) != 0 && endsString(s, at, q, quoted) {
				return at
			}
		}
		bs = 0
	}
	return to
}

// quoteAfterBackslashes returns the quote that the quote byte at s[k], after a
// run of bs backslashes, stands for, and its offset, as stringEndIn and
// endOnLine read quotes. After one backslash, or five, it is the escaped quote
// \" or \' at k-1, and after three it is none, as openQuotes.before reads
// them: the quote of a string nested in a nested string. After none it is the
// quote at k. After an even run it is the quote at k only if the end of the
// line or a byte of ",;:.!?)]}" follows it at once, as one does in
// {"auth": "Bearer a'b\\", "user": "x"}, where the run is a backslash escaped
// in the string, and none otherwise, as in "Bearer a'b\\" S", where the run
// may as well escape the quote.
func quoteAfterBackslashes(s string, k, bs int) (string, int) {
	switch {
	case bs%4 == 1:
		return s[k-1 : k+1], k - 1
	case bs%4 == 3:
		return "", k
	case bs > 0 && k+1 < len(s) && s[k+1] != '\n' && s[k+1] != '\r' && strings.IndexByte(",;:.!?)]}", s[k+1]) < 0:
		return "", k
	}
	return s[k : k+1], k
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
	sh := quotes.shellFor(s)
	quoted := quotes.before(s, loc[4])
	q := openingQuote(s, loc[4])
	closes := q != "" && closesString(s, loc[4], q, quoted)
	if closes && !ownsQuotedValue(s, loc[4], loc[5], q) {
		quotes.skip(s, loc[4])
		return s[loc[0]:loc[4]], loc[4]
	}
	if quote := textBetweenStrings(sh, s, loc[4], loc[5], quoted); quote != "" {
		next := quotes.unpaired(s, loc[4], quote)
		return s[loc[0]:next], next
	}
	colon := loc[3] + 1 + max(0, strings.IndexByte(s[loc[3]+1:loc[4]], ':'))
	keyName := scrubJSONMapKey(s[loc[2]:loc[3]], 0)
	switch kind, pair := nestsSensitiveKey(s, loc[4], loc[5], startsLine(s, colon+1, loc[4]), true, quoted); kind {
	case nestedKey:
		return s[loc[0]:loc[4]], loc[4]
	case valueEndingInKeyName:
		end := nestedPairEndWith(sh, s, pair, loc[5], 0, quoted)
		if closes && end > loc[4] {
			quotes.drop()
		}
		quotes.skip(s, end)
		return `"` + keyName + `":"` + redactedValue + `"`, end
	}
	innerQuoted := quotes.innerAt(s, loc[4])
	subSQ := quotes.subSingleAt(s, loc[4])
	subEscDQ := quotes.subEscDoubleAt(s, loc[4])
	if pEnd := placeholderCut(s, loc[4]); pEnd > 0 && placeholderEnds(s, pEnd, quoted, innerQuoted, subSQ, subEscDQ) {
		if closes && pEnd > loc[4] {
			quotes.drop()
		}
		quotes.skip(s, pEnd)
		return s[loc[0]:loc[2]] + keyName + s[loc[3]:pEnd], pEnd
	}
	end := textValueEndWithEnclosing(sh, s, loc[4], loc[5], 0, quoted, innerQuoted, subSQ, subEscDQ, quotes)
	if closes && end > loc[4] {
		quotes.drop()
	}
	quotes.skip(s, end)
	if end == loc[4] {
		return s[loc[0]:end], end
	}
	if inner, _ := unquoteValue(s[loc[4]:end]); isRedactedValue(inner) || isRedactedValue(trimTrailingClosingQuote(s[loc[4]:end], s, end, quoted)) {
		return s[loc[0]:loc[2]] + keyName + s[loc[3]:end], end
	}
	return `"` + keyName + `":"` + redactedValue + `"`, end
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
	sh := quotes.shellFor(s)
	if loc[8] < 0 {
		// A compound name that does not end in a credential name, such as
		// next_page_token, which textKeyAllowedAt does not accept, and whose
		// value pairAtIf has not read. What follows the delimiter is scanned
		// again.
		return s[loc[0]:loc[7]], loc[7]
	}
	open := brackets.before(s, loc[0])
	if loc[4] == loc[5] && isCookieProse(s, s[loc[2]:loc[3]], s[loc[6]:loc[7]], loc[8], open > 0, quotes) {
		// A Cookie or Set-Cookie name and a colon in prose, as in
		// The cookie: Monster was here., which redactAuthHeaders leaves as
		// it is too. What follows the delimiter is scanned again.
		return s[loc[0]:loc[7]], loc[7]
	}
	start, end := loc[8], loc[9]
	quoted := quotes.before(s, start)
	innerQuoted := quotes.innerAt(s, start)
	subSQ := quotes.subSingleAt(s, start)
	subEscDQ := quotes.subEscDoubleAt(s, start)
	keyName := scrubJSONMapKey(s[loc[2]:loc[3]], 0)
	if pEnd := placeholderCut(s, start); pEnd > 0 {
		if loc[4] == loc[5] && loc[2] > 0 && s[loc[2]-1] == '"' && pEnd < len(s) && s[pEnd] == '"' && placeholderDelimBoundary(s, pEnd, quoted, false) {
			brackets.skip(loc[7])
			quotes.skip(s, loc[7])
			return s[loc[0]:loc[7]], loc[7]
		}
		if pEnd <= end && (s[start] == '"' || s[start] == '\'') && pEnd < len(s) && s[pEnd] == s[start] && placeholderDelimBoundary(s, pEnd+1, quoted, false) {
			end = pEnd + 1
			brackets.skip(end)
			quotes.skip(s, end)
			return s[loc[0]:loc[2]] + keyName + s[loc[3]:end], end
		}
		// keyValFinder may read past the redaction placeholder that starts
		// the value when non-whitespace characters follow it. If the
		// placeholder ends there, keep it up to pEnd; otherwise read the
		// full value to its end.
		if placeholderEnds(s, pEnd, quoted, innerQuoted, subSQ, subEscDQ) || authHeaderPlaceholderEnds(s, loc, pEnd, quoted, innerQuoted) {
			brackets.skip(pEnd)
			quotes.skip(s, pEnd)
			return s[loc[0]:loc[2]] + keyName + s[loc[3]:pEnd], pEnd
		}
		end = bareValueEnd(s, pEnd)
	}
	if loc[4] < loc[5] && (quoted&quoteKind(openingQuote(s, start)) != 0 || s[loc[5]:loc[7]] != "=") {
		// The value of a quoted key that is the text between two strings, as
		// the value of "token": " in config, api_key="secret" is, is not its
		// value: the text after its opening quote is searched again (see
		// redactJSONKeyValue). The value of a key that "=" follows at once,
		// outside an enclosing string of the same quote kind, as in
		// export "DB_PASSWORD"=" a b"c, is not that text, as the quotes of a
		// shell command join words into one value.
		if quote := textBetweenStrings(sh, s, start, end, quoted); quote != "" {
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
		end = nestedPairEndWithEnclosing(sh, s, pair, end, open, quoted, innerQuoted)
		brackets.skip(end)
		quotes.skip(s, end)
		return keyName + s[loc[4]:loc[5]] + s[loc[6]:loc[7]] + quote + redactedValue + quote, end
	}
	if redEnd := redactedPrefixEnd(s, start, end, quoted, innerQuoted, subSQ, subEscDQ); redEnd >= 0 {
		brackets.skip(redEnd)
		quotes.skip(s, redEnd)
		return s[loc[0]:loc[2]] + keyName + s[loc[3]:redEnd], redEnd
	}
	end = textValueEndWithEnclosing(sh, s, start, end, open, quoted, innerQuoted, subSQ, subEscDQ, quotes)
	end = trimTrailingSubshellOrJSONQuotes(s, start, end, quoted, innerQuoted, subSQ, subEscDQ)
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
		return s[loc[0]:loc[2]] + keyName + s[loc[3]:end], end
	}
	return keyName + s[loc[4]:loc[5]] + s[loc[6]:loc[7]] + quote + redactedValue + quote, end
}

// redactedPrefixEnd returns where the redaction placeholder that starts the
// value s[start:end] ends if the value ends with it (see placeholderEnds),
// and else -1. Redacting DB_PASSWORD=pre"a b"token=x gives
// DB_PASSWORD=[REDACTED]token=[REDACTED] and redacting
// DB_PASSWORD=pre"a b""token": "x" gives
// DB_PASSWORD=[REDACTED]"token":"[REDACTED]" (see joinedQuoteEnd), and the
// value that the placeholder stands for ends where it does, so that
// scrubbing the text again leaves it as it is, as redactBearerToken does for
// a Bearer token. A pair or a quoted key whose key is not sensitive ends no
// value, so password=[REDACTED]x=SECRET is redacted whole, as
// password=pre"a b"x=SECRET is.
func redactedPrefixEnd(s string, start, end int, quoted, innerQuoted quoteKinds, subSQ, subEscDQ bool) int {
	redEnd := start + len(redactedValue)
	if redEnd >= end || !strings.HasPrefix(s[start:end], redactedValue) || !placeholderEnds(s, redEnd, quoted, innerQuoted, subSQ, subEscDQ) {
		return -1
	}
	return redEnd
}

// placeholderEnds reports whether a value whose leading redaction
// placeholder ends at s[end:] ends with it: whether a pair whose value the
// text patterns redact on its own follows it at once (see
// startsRedactedPair), as in password=[REDACTED]token=[REDACTED], which a
// scrub writes for password=x"a b"token=y and which scrubbing again leaves as
// it is. Text can hold the placeholder without any scrub having written it,
// so a pair whose key is not sensitive, as in password=[REDACTED]x=SECRET, or
// a quoted key that is not, as in password=[REDACTED]"x": "SECRET", does not
// end the value there: it is the rest of a credential that only starts with
// the placeholder, which a shell reads as one word, and it is redacted with
// it.
//
// The placeholder also ends a value that only quotes follow up to the end of
// its line, as the quote that redactAuthHeaders leaves after the placeholder
// of curl -H 'Authorization: Bearer abc' closes the string that holds the
// header (see closingQuotesStart).
func placeholderEnds(s string, end int, quoted, innerQuoted quoteKinds, subSQ, subEscDQ bool) bool {
	if startsRedactedPair(s, end, quoted) || startsBearerKeyword(s, end) || quotesEndLine(s, end) {
		return true
	}
	return placeholderEndsQuote(s, end, quoted, innerQuoted, subSQ, subEscDQ)
}

func placeholderEndsQuote(s string, end int, quoted, innerQuoted quoteKinds, subSQ, subEscDQ bool) bool {
	if subSQ && end < len(s) && s[end] == '\'' {
		if placeholderDelimBoundary(s, end+1, quoted|innerQuoted, true) {
			return true
		}
	}
	if subEscDQ {
		if innerQuoted&4 != 0 && end+8 <= len(s) && s[end:end+8] == `\\\\\\\"` {
			if placeholderDelimBoundary(s, end+8, quoted|innerQuoted, true) {
				return true
			}
		} else if end+4 <= len(s) && s[end:end+4] == `\\\"` {
			if placeholderDelimBoundary(s, end+4, quoted|innerQuoted, true) {
				return true
			}
		}
	}
	// A closing quote after [REDACTED] that matches the enclosing quote
	// kind (outer or nested), followed by whitespace, a structural delimiter,
	// or the end of the string, ends the placeholder. This prevents
	// redactTextKeyValue from eating the closing quote and the rest of the
	// line (as in 'Authorization: [REDACTED]' user=bob or JSON-wrapped
	// shell/nested-JSON values). The quote must match an active enclosing
	// quote kind so that password=[REDACTED]" secret" (where no quote is
	// open before [REDACTED]) still redacts " secret".
	enclosing := quoted | innerQuoted
	if enclosing != 0 {
		if q := openingQuote(s, end); q != "" {
			k := end + len(q)
			qk := quoteKind(q)
			matched := enclosing&qk != 0
			if matched {
				enclosing &^= qk
			} else if len(q) == 2 && enclosing&quoteKind(q[1:2]) != 0 {
				if k < len(s) && s[k] == q[1] {
					k++
				}
				enclosing &^= quoteKind(q[1:2])
				matched = true
			}
			if matched && len(q) == 2 && k < len(s) && (s[k] == '"' || s[k] == '\'') && enclosing&quoteKind(s[k:k+1]) != 0 {
				enclosing &^= quoteKind(s[k : k+1])
				k++
			}
			if matched && placeholderDelimBoundary(s, k, enclosing, true) {
				return true
			}
		}
	}
	return false
}

func hasUnnestedEscapedDoubleQuote(s string) bool {
	for i := 0; i < len(s); {
		if s[i] != '\\' {
			i++
			continue
		}
		j := i
		for j < len(s) && s[j] == '\\' {
			j++
		}
		if j < len(s) && s[j] == '"' && (j-i)%4 == 1 {
			return true
		}
		i = j
	}
	return false
}

func hasUnescapedDoubleQuote(s string) bool {
	bs := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' {
			bs++
			continue
		}
		if s[i] == '"' && bs%2 == 0 {
			return true
		}
		bs = 0
	}
	return false
}

func skipJSONWhitespace(s string, k int) int {
	for k < len(s) && (s[k] == ' ' || s[k] == '\t' || s[k] == '\r' || s[k] == '\n') {
		k++
	}
	return k
}

func trimTrailingJSONEscapedWSAndClosers(s string, start, end int) (cand int, trimmedWS bool) {
	cand = end
	for cand > start {
		c := s[cand-1]
		if c == ')' || c == ']' || c == '}' {
			cand--
			continue
		}
		if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
			cand--
			trimmedWS = true
			continue
		}
		if cand >= start+2 && (c == 'n' || c == 'r' || c == 't') && s[cand-2] == '\\' {
			k := cand - 2
			for k > start && s[k-1] == '\\' {
				k--
			}
			if (cand-1-k)%2 == 1 {
				cand -= 2
				trimmedWS = true
				continue
			}
		}
		break
	}
	return cand, trimmedWS
}

func enclosingSQClosedAt(s string, start, cand int, quoted quoteKinds, subSQ bool) bool {
	if cand <= start || s[cand-1] != '\'' {
		return false
	}
	bs := 0
	if subSQ {
		cur := true
		for i := start; i < cand; i++ {
			if s[i] == '\\' {
				bs++
				continue
			}
			b := bs
			bs = 0
			if s[i] != '\'' || b%2 == 1 {
				continue
			}
			if cur {
				if isMidWordApostrophe(s, i) && !opensPrefixedString(s, i) {
					continue
				}
				cur = false
			} else if (!isMidWordApostrophe(s, i) || opensPrefixedString(s, i)) && !isTrailingApostrophe(s, i) {
				cur = true
			}
		}
		return !cur
	}
	if quoted&2 == 0 {
		return false
	}
	cur := quoteKinds(2)
	for i := start; i < cand; i++ {
		if s[i] == '\\' {
			bs++
			continue
		}
		b := bs
		bs = 0
		if s[i] != '\'' || b%2 == 1 {
			continue
		}
		if isApostrophe(s, i, 2, cur) {
			continue
		}
		switch cur {
		case 0:
			cur = 2
		case 2:
			cur = 0
		}
	}
	return cur == 0
}

func hasJSONValueCloserAfter(s string, after int, allowEscWS bool) bool {
	ws := skipJSONWhitespace(s, after)
	for allowEscWS && ws+1 < len(s) && s[ws] == '\\' && (s[ws+1] == 'n' || s[ws+1] == 'r' || s[ws+1] == 't') {
		ws = skipJSONWhitespace(s, ws+2)
	}
	return ws < len(s) && (s[ws] == ',' || s[ws] == '}' || s[ws] == ']')
}

func trimUnclosedOuterClosingQuote(s string, start, end int, q string, quoted, innerQuoted quoteKinds) int {
	enclosing := (quoted | innerQuoted) &^ quoteKind(q)
	if enclosing == 0 {
		return end
	}
	if s[end-1] == '"' || s[end-1] == '\'' {
		bs := 0
		for k := end - 2; k >= start && s[k] == '\\'; k-- {
			bs++
		}
		qk := quoteKinds(1)
		if s[end-1] == '\'' {
			qk = 2
		}
		switch {
		case bs%2 == 0 && enclosing&qk != 0:
			if qk == 2 && isApostrophe(s, end-1, 2, 2) {
				return end
			}
			if endsString(s, end-1, s[end-1:end], enclosing) && hasJSONValueCloserAfter(s, end, innerQuoted&4 != 0) {
				return end - 1
			}
		case bs%4 == 1 && end >= start+2 && enclosing&(qk<<2) != 0:
			if endsString(s, end-2, s[end-2:end], enclosing) && hasJSONValueCloserAfter(s, end, true) {
				return end - 2
			}
		}
	}
	return end
}

func startsJSONObjectCommaMember(s string, commaIdx int) bool {
	if commaIdx < 0 || commaIdx >= len(s) || s[commaIdx] != ',' {
		return false
	}
	j := skipJSONWhitespace(s, commaIdx+1)
	if j >= len(s) || s[j] != '"' {
		return false
	}
	j++
	limit := min(len(s), j+256)
	for j < limit {
		if s[j] == '\\' {
			if j+1 >= limit {
				return false
			}
			j += 2
			continue
		}
		if s[j] == '"' {
			k := skipJSONWhitespace(s, j+1)
			return k < len(s) && s[k] == ':'
		}
		if s[j] == '\n' || s[j] == '\r' {
			return false
		}
		j++
	}
	return false
}

func trimTrailingSubshellOrJSONQuotes(s string, start, end int, quoted, innerQuoted quoteKinds, subSQ, subEscDQ bool) int {
	if end <= start {
		return end
	}
	if q := openingQuote(s, start); q != "" {
		if _, quote := unquoteValue(s[start:end]); quote == "" {
			return trimUnclosedOuterClosingQuote(s, start, end, q, quoted, innerQuoted)
		}
		return end
	}
	if quoted&1 == 0 {
		if !subSQ && innerQuoted == 0 && !subEscDQ {
			cand, trimmedWS := trimTrailingJSONEscapedWSAndClosers(s, start, end)
			bracketCand := end
			for bracketCand > start && (s[bracketCand-1] == ')' || s[bracketCand-1] == ']' || s[bracketCand-1] == '}') {
				bracketCand--
			}
			delimOK := bracketCand < end || (end < len(s) && strings.IndexByte(";|&)]}", s[end]) >= 0)
			switch {
			case quoted&2 != 0:
				ws := skipHorizDelimSpace(s, end)
				shellAfterWS := ws > end && ws < len(s) && (strings.IndexByte(";|&)]}(`", s[ws]) >= 0 || strings.HasPrefix(s[ws:], "$("))
				hasInteriorSQ := bracketCand > start+1 && strings.ContainsRune(s[start:bracketCand-1], '\'')
				if bracketCand > start+1 && s[bracketCand-1] == '\'' && enclosingSQClosedAt(s, start, bracketCand, quoted, false) &&
					(delimOK || shellAfterWS || (hasInteriorSQ && placeholderDelimBoundary(s, end, quoted&^2, true) && !startsRedactedPair(s, ws, quoted&^2))) {
					return bracketCand - 1
				}
			case quoted&4 != 0:
				if cand >= start+2 && s[cand-1] == '"' {
					bs := 0
					for k := cand - 2; k >= 0 && s[k] == '\\'; k-- {
						bs++
					}
					wsOK := trimmedWS && placeholderDelimBoundary(s, end, quoted, true)
					if bs%4 == 1 && !hasUnnestedEscapedDoubleQuote(s[start:cand-2]) && (delimOK || wsOK) {
						return cand - 2
					}
				}
			case quoted&8 != 0:
				if bracketCand >= start+2 && s[bracketCand-1] == '\'' {
					bs := 0
					for k := bracketCand - 2; k >= 0 && s[k] == '\\'; k-- {
						bs++
					}
					if bs%4 == 1 && !strings.Contains(s[start:bracketCand-2], `\'`) && delimOK {
						return bracketCand - 2
					}
				}
			}
		}
		return end
	}
	if !subSQ && innerQuoted&4 == 0 && !subEscDQ {
		cand := end
		for cand > start && (s[cand-1] == ')' || s[cand-1] == ']' || s[cand-1] == '}') {
			cand--
		}
		if cand > start+1 && s[cand-1] == '"' {
			bs := 0
			for k := cand - 2; k >= start && s[k] == '\\'; k-- {
				bs++
			}
			noUnescDQ := !hasUnescapedDoubleQuote(s[start : cand-1])
			escQuoteOnly := strings.ContainsRune(s[start:cand-1], '"') && noUnescDQ
			ws := skipJSONWhitespace(s, end)
			jsonClose := cand < end || (ws < len(s) && (s[ws] == ',' || s[ws] == '}' || s[ws] == ']') && placeholderDelimBoundary(s, ws, quoted&^1, false))
			innerQuoteOrSub := escQuoteOnly || ((subSQ || subEscDQ || openingQuote(s, start) != "") && noUnescDQ)
			prettyOrStructClose := (ws > end || (cand >= start+2 && s[cand-2] == ')') || startsJSONObjectCommaMember(s, ws)) && noUnescDQ
			if bs%2 == 0 && ((!strings.ContainsRune(s[start:cand-1], '"') && (cand < end || (end < len(s) && s[end] == ')'))) || ((innerQuoteOrSub || prettyOrStructClose) && jsonClose)) {
				return cand - 1
			}
		}
		return end
	}
	p := end
	remClose := quoted
	boundaryOK := false
	if remClose&1 != 0 && p > start && s[p-1] == '"' {
		bs := 0
		for k := p - 2; k >= start && s[k] == '\\'; k-- {
			bs++
		}
		if bs%2 == 0 && placeholderDelimBoundary(s, p, remClose&^1, false) {
			p--
			remClose &^= 1
			boundaryOK = true
		}
	}
	if innerQuoted&4 != 0 && p >= start+2 {
		cand, _ := trimTrailingJSONEscapedWSAndClosers(s, start, p)
		if cand >= start+2 && s[cand-2:cand] == `\"` {
			bs := 1
			for k := cand - 3; k >= 0 && s[k] == '\\'; k-- {
				bs++
			}
			if bs%4 == 1 && !hasUnnestedEscapedDoubleQuote(s[start:cand-2]) && (boundaryOK || placeholderDelimBoundary(s, p, remClose, true)) {
				p = cand - 2
				boundaryOK = true
			}
		}
	}
	if subEscDQ {
		cand := p
		for cand > start && (s[cand-1] == ')' || s[cand-1] == ']' || s[cand-1] == '}') {
			cand--
		}
		if cand >= start+4 && s[cand-1] == '"' {
			bs := 0
			for k := cand - 2; k >= 0 && s[k] == '\\'; k-- {
				bs++
			}
			qLen := 0
			if innerQuoted&4 != 0 && bs%8 == 7 && cand >= start+8 {
				qLen = 8
			} else if bs%4 == 3 {
				qLen = 4
			}
			if qLen > 0 && !strings.Contains(s[start:cand-qLen], `\"`) && (boundaryOK || placeholderDelimBoundary(s, p, remClose, true)) {
				p = cand - qLen
				boundaryOK = true
			}
		}
	}
	if subSQ && p > start {
		cand := p
		for cand > start && (s[cand-1] == ')' || s[cand-1] == ']' || s[cand-1] == '}') {
			cand--
		}
		if cand > start && s[cand-1] == '\'' && enclosingSQClosedAt(s, start, cand, quoted, true) && (boundaryOK || placeholderDelimBoundary(s, p, remClose, true)) {
			p = cand - 1
		}
	}
	if p == start {
		return end
	}
	return p
}

// authHeaderPlaceholderEnds preserves an outer wrapper quote right after a
// redaction placeholder (e.g. 'X-Api-Key: [REDACTED]';) when quoted == 0,
// preventing subsequent passes from losing track of the closing quote and
// swallowing trailing text.
func authHeaderPlaceholderEnds(s string, loc []int, end int, quoted, innerQuoted quoteKinds) bool {
	if len(loc) < 8 || loc[4] != loc[5] || loc[2] <= 0 || end >= len(s) {
		return false
	}
	if !strings.Contains(s[loc[5]:loc[7]], ":") {
		return false
	}
	hdr := s[loc[2]:loc[7]]
	if !authHeaderKeys.mayStartAt(hdr, 0) || authHeaderPrefixAtRe.FindStringIndex(hdr) == nil {
		return false
	}
	qStart := loc[2]
	for qStart > 0 && (s[qStart-1] == ' ' || s[qStart-1] == '\t') {
		qStart--
	}
	if qStart <= 0 {
		return false
	}
	prev := s[qStart-1]
	if prev != '\'' && prev != '"' {
		return false
	}
	bs := 0
	for k := qStart - 2; k >= 0 && s[k] == '\\'; k-- {
		bs++
	}
	q := s[qStart-1 : qStart]
	nestedEscDQ := prev == '"' && bs%4 == 3 && qStart >= 4 && s[qStart-4:qStart] == `\\\"` && (quoted|innerQuoted)&quoteKind(`\"`) != 0
	if nestedEscDQ {
		q = s[qStart-4 : qStart]
	} else if bs%2 == 1 {
		q = s[qStart-2 : qStart]
	}
	if !nestedEscDQ && quoted&quoteKind(q) != 0 {
		return false
	}
	if !strings.HasPrefix(s[end:], q) {
		return false
	}
	after := end + len(q)
	enclosing := quoted | innerQuoted
	if !nestedEscDQ {
		enclosing = quoted | (innerQuoted &^ quoteKind(q))
	}
	return placeholderDelimBoundary(s, after, enclosing, true)
}

func placeholderDelimBoundary(s string, k int, closingQuotes quoteKinds, allowAmpPipe bool) bool {
	startK := k
	sawBracket := false
	for {
		if k >= len(s) {
			return true
		}
		switch s[k] {
		case ' ', '\t', '\r', '\n', ',', ';':
			return true
		case '|':
			return allowAmpPipe
		case '&':
			return allowAmpPipe && (k+1 >= len(s) || s[k+1] == '&' || s[k+1] == ' ' || s[k+1] == '\t')
		case ')', ']', '}':
			sawBracket = true
			k++
			continue
		default:
			if s[k] == '\\' && k+1 < len(s) && (s[k+1] == 'n' || s[k+1] == 'r' || s[k+1] == 't') {
				k += 2
				continue
			}
			if aq := openingQuote(s, k); aq != "" && closingQuotes&quoteKind(aq) != 0 {
				closingQuotes &^= quoteKind(aq)
				k += len(aq)
				continue
			}
			if !sawBracket && k > startK && (startsRedactedPair(s, k, 0) || startsBearerKeyword(s, k)) {
				return true
			}
			return false
		}
	}
}

// quotesEndLine reports whether s[i:] starts with quotes, each of which may
// follow a backslash, that only spaces or tabs follow up to the end of the
// line or of s.
func quotesEndLine(s string, i int) bool {
	k := i
	for k < len(s) {
		if s[k] == '\\' && k+1 < len(s) && (s[k+1] == '\'' || s[k+1] == '"') {
			k += 2
		} else if s[k] == '\'' || s[k] == '"' {
			k++
		} else {
			break
		}
	}
	if k == i {
		return false
	}
	for k < len(s) && (s[k] == ' ' || s[k] == '\t') {
		k++
	}
	return k == len(s) || s[k] == '\n' || s[k] == '\r'
}

// startsRedactedPair reports whether a pair that ends a preceding value starts
// at s[end:], right after a value: a key that afterClosedQuotePair reads as a
// sensitive key with a value, whose name textKeyAllowed accepts, as token=x
// and GITHUB_TOKEN="y" do, a known non-sensitive pagination cursor or integer
// token counter pair (see startsPreservedTokenPair), as next_page_token=x,
// next.page.token="y" and token_count=7 do, or a quoted key that
// quotedKeyAllowed, isPreservedPageTokenName or isTokenCounterName accepts,
// with a delimiter and valid value after it, as "token": "x" does (see
// startsQuotedKey). Of an ordinary pair whose key is not sensitive, as label=x
// or "x": "y", the text patterns redact no value, and of a key that only ends
// in a credential name that textKeyAllowed does not accept, as in
// Q1password=x, foo.signature=x, token_hint=x or access_tokens=x, they read no
// pair, so neither ends a value: joined to the value at once, it may be more
// of a credential that a shell joins into one word with it, as in
// DB_PASSWORD="[REDACTED]"label=SECRET, so the value runs on over it.
func startsRedactedPair(s string, end int, quoted quoteKinds) bool {
	if kind, _ := afterClosedQuotePair(nil, s, end, quoted); kind == nestedKey {
		n := 0
		for end+n < len(s) && n < maxNestedKeyLen && isCamelTokenByte(s[end+n]) {
			n++
		}
		return n > 0 && textKeyAllowed(s[end:end+n])
	}
	return startsPreservedTokenPair(s, end, quoted) || startsQuotedKey(s[end:], quoted)
}

// tokenCounterNames holds the separator-stripped lowercase names of known
// non-sensitive token-usage counter fields.
var tokenCounterNames = map[string]struct{}{
	"tokencount":              {},
	"prompttokens":            {},
	"completiontokens":        {},
	"totaltokens":             {},
	"maxoutputtokens":         {},
	"thinkingtokens":          {},
	"cachedcontenttokencount": {},
	"prompttokencount":        {},
	"candidatestokencount":    {},
	"totaltokencount":         {},
	"thoughtstokencount":      {},
	"inputtokens":             {},
	"outputtokens":            {},
	"reasoningtokens":         {},
	"cachedtokens":            {},
	"tooluseprompttokencount": {},
	"completiontokencount":    {},
}

// isPreservedKeyShape reports whether name is a non-empty identifier key
// (without digits or trailing separators) of at most maxNestedKeyLen
// camel-token bytes that textKeyAllowed does not classify as sensitive.
func isPreservedKeyShape(name string) bool {
	if name == "" || len(name) > maxNestedKeyLen || !isIdentifierKey(name) || strings.IndexFunc(name, unicode.IsDigit) >= 0 {
		return false
	}
	last := name[len(name)-1]
	if last == '_' || last == '-' || last == '.' {
		return false
	}
	for i := 0; i < len(name); i++ {
		if !isCamelTokenByte(name[i]) {
			return false
		}
	}
	return !textKeyAllowed(name)
}

// isPreservedPageTokenName reports whether name is a preserved pagination
// cursor key name (see isPageTokenName).
func isPreservedPageTokenName(name string) bool {
	return isPreservedKeyShape(name) && isPageTokenName(name)
}

// isTokenCounterName reports whether name is a known non-sensitive token
// counter key name in tokenCounterNames.
func isTokenCounterName(name string) bool {
	if !isPreservedKeyShape(name) || strings.Contains(name, "..") {
		return false
	}
	_, ok := tokenCounterNames[removeKeySeparators(strings.ToLower(name))]
	return ok
}

// startsDecimalCounterValue reports whether s[i:] starts (after optional
// delimiter whitespace) with an unsigned decimal integer token count, bare or
// between a matching pair of quotes (", ', \" or \'), as the "128" of
// input_tokens="128" and the '10' of prompt_tokens='10' are, followed by
// end-of-text or a value boundary. A sign, any other byte, an empty value,
// and a quoted count that more text is glued to, as in "10"SECRET, are not a
// count.
func startsDecimalCounterValue(s string, i int, quoted quoteKinds) bool {
	at := skipDelimSpace(s, i)
	q := openingQuote(s, at)
	at += len(q)
	if at >= len(s) || s[at] < '0' || s[at] > '9' {
		return false
	}
	end := at + 1
	for end < len(s) && s[end] >= '0' && s[end] <= '9' {
		end++
	}
	if q != "" {
		if !strings.HasPrefix(s[end:], q) {
			return false
		}
		end += len(q)
	}
	return end == len(s) || followsValueBoundary(s[end:], quoted)
}

// startsPreservedTokenPair reports whether s[end:] starts with a preserved
// pagination cursor key and value or a preserved token counter key and
// unsigned decimal integer value.
func startsPreservedTokenPair(s string, end int, quoted quoteKinds) bool {
	if end >= len(s) || !isIdentifierKey(s[end:end+1]) {
		return false
	}
	n := 1
	for end+n < len(s) && n <= maxNestedKeyLen && isCamelTokenByte(s[end+n]) {
		n++
	}
	if n > maxNestedKeyLen {
		return false
	}
	name := s[end : end+n]
	page := isPreservedPageTokenName(name)
	counter := !page && isTokenCounterName(name)
	if !page && !counter {
		return false
	}
	delim := skipHorizDelimSpace(s, end+n)
	if delim >= len(s) || (s[delim] != ':' && s[delim] != '=') || (s[delim] == '=' && delim+1 < len(s) && s[delim+1] == '=') {
		return false
	}
	if counter {
		return startsDecimalCounterValue(s, delim+1, quoted)
	}
	return valueStartsAt(s, delim+1, true, quoted)
}

// placeholderCut returns the end of the redaction placeholder that starts
// the value at s[start:] if an ASCII letter or digit, "_", "-", ".", a quote
// or "\" follows it at once, and else -1: only after such a byte, which may
// start a pair or a quoted key (see placeholderEnds), may the value end with
// the placeholder, and as no such byte ends a bare value, the bare value then
// runs on past the placeholder, which may be to the end of a long word.
// keyValFinder reads such a value only up to the end of the placeholder, and
// redactTextKeyValue reads the rest of it only if the value does not end
// there, so that a run of redacted pairs in one word, as in
// password=[REDACTED]token=[REDACTED]..., is read once rather than from each
// key to the end of the word.
func placeholderCut(s string, start int) int {
	end := start + len(redactedValue)
	if end >= len(s) || !strings.HasPrefix(s[start:], redactedValue) {
		return -1
	}
	if c := s[end]; isCamelTokenByte(c) || c == '"' || c == '\'' || c == '\\' {
		return end
	}
	return -1
}

// bareValueEnd returns where the bare value that runs on through s[i:] ends,
// as sensitiveValueRegex reads it: at the first whitespace, ",", ";" or "&",
// or at the end of s. A backslash before a line break (\<LF> or \<CR><LF>) is a
// shell line continuation, so the bare value runs on past it.
func bareValueEnd(s string, i int) int {
	for {
		n := strings.IndexAny(s[i:], regexpSpace+",;&")
		if n < 0 {
			return len(s)
		}
		end := i + n
		// Check for a backslash-newline or backslash-CRLF line splice.
		if (s[end] == '\n' || s[end] == '\r') && end > i && s[end-1] == '\\' {
			// Count preceding backslashes: an odd count means the line break
			// is escaped (a line continuation), so skip past it.
			bs := 0
			for k := end - 1; k >= i && s[k] == '\\'; k-- {
				bs++
			}
			if bs%2 == 1 {
				if s[end] == '\r' && end+1 < len(s) && s[end+1] == '\n' {
					i = end + 2
				} else {
					i = end + 1
				}
				continue
			}
		}
		return end
	}
}

// quoteKindString returns the opening quote string for the quote kinds in
// quoted, or "" if quoted is 0 or ambiguous. It returns the first matching
// quote kind: ", ', ", or '.
func quoteKindString(quoted quoteKinds) string {
	switch {
	case quoted&1 != 0:
		return `"`
	case quoted&2 != 0:
		return `'`
	case quoted&4 != 0:
		return `\"`
	case quoted&8 != 0:
		return `\'`
	}
	return ""
}

// interiorApostropheEnd returns the end of the value that starts at s[start]
// with opening quote q, when the closing quote at s[end-len(q):end] is an
// interior word apostrophe (a letter immediately precedes it, as the one in
// 'it's), by scanning forward on the same line for the first later quote of
// the same kind that ends at a value boundary. Quotes between two letters on
// the way, as the one of friend's, are apostrophes too and are passed over, as
// are pairs inside the single-quoted value, as in 'it's token=prefix secret',
// and times, as in 10:30, which may all be words of a passphrase. It returns
// end unchanged when the closing quote is not an interior apostrophe, when the
// line holds no such later quote, or when a later quote opens the value of a
// sensitive key or a quoted sensitive key (see endsInSensitiveKeyDelimiter and
// opensQuotedPair), so that the value never takes in that key and leaves its
// value after it. A value whose opening quote is joined to the words on both
// sides of it, as the one of DB_PASSWORD=pre'it's correct SECRET is, instead
// runs to the end of the line when the line holds no such later quote. In a
// value that its quote opens explicitly, the quote found may be followed on
// to a later one on its line (see laterValueQuoteEnd).
func interiorApostropheEnd(sh *shellAssignScanner, s string, start, end int, q string, quoted quoteKinds) int {
	closing := end - len(q)
	if closing <= start+len(q) {
		return end
	}
	// If the opening quote at start is itself a contraction (as in
	// password=it's Bob's), do not extend: the quote is prose.
	if start > 0 && isCamelTokenByte(s[start-1]) && startsContraction(s, start, len(q)) {
		return end
	}
	// A letter must precede the closing quote for it to be an interior apostrophe.
	prev, _ := utf8.DecodeLastRuneInString(s[start+len(q) : closing])
	if !unicode.IsLetter(prev) {
		return end
	}
	// A letter must follow the closing quote, and after advancing over contiguous
	// letters, horizontal whitespace must follow.
	if end >= len(s) {
		return end
	}
	r, _ := utf8.DecodeRuneInString(s[end:])
	if !unicode.IsLetter(r) {
		return end
	}
	k := end
	for k < len(s) {
		rr, sz := utf8.DecodeRuneInString(s[k:])
		if !unicode.IsLetter(rr) {
			break
		}
		k += sz
	}
	if k >= len(s) {
		return end
	}
	prevStart, _ := utf8.DecodeLastRuneInString(s[:start])
	explicitOpen := start == 0 || (!isCamelTokenByte(s[start-1]) && !unicode.IsLetter(prevStart) && !unicode.IsNumber(prevStart))
	// The closing quote starts the ending of a contraction or a possessive
	// (see startsContraction), as the second quote of 'it's... SECRET' does,
	// rather than closing the part before it, as the second quote of
	// 'benign'api_key='x' does.
	contraction := startsContraction(s, closing, len(q))
	// A quote that is not explicitly opened but is joined both to the word
	// before it and to the byte after it, and whose closing quote starts a
	// contraction, as the first quote of DB_PASSWORD=pre'it's correct SECRET
	// is, opens no string of its own. When no later quote on the line ends
	// the value, the value runs to the end of the line, rather than ending
	// at the apostrophe of it's and leaving the rest of the value after it.
	// A quote with a space after it, as the one of users' wasn't, is a
	// trailing apostrophe and is left as it is, and a part that a quote
	// closes before a word, as the one of pre'a b'c is, stays a part.
	nextRune, _ := utf8.DecodeRuneInString(s[start+len(q):])
	joinedOpen := !explicitOpen && contraction && !isDelimSpace(nextRune)
	// In a value that its quote opens explicitly, the ending of a
	// contraction that a digit, "_" or "=" follows, as in 'it's_x S',
	// 'it's=x S' and 'it's1 S', starts the ending of a contraction too when
	// a later quote on the line closes it before any key-value delimiter:
	// startsContraction leaves those out so that a quote before a word or a
	// key name, as in f'd={x}' or 'abc'd1 user=x, opens a part, while a
	// quoted passphrase with a glued suffix reads on to the quote that ends it.
	gluedContraction := explicitOpen && !contraction && startsContractionBefore(s, closing, len(q), unicode.IsLetter)
	contraction = contraction || gluedContraction
	if joinedOpen && strings.HasPrefix(s[k:], q) && !strings.HasPrefix(s[k+len(q):], q) {
		return end
	}
	// After the ending of a contraction in a value that its quote opens
	// explicitly or joins to the words on both sides of it, any punctuation
	// continues the value, as the ... of 'it's... SECRET', the ) of
	// 'it's) SECRET' and the , of DB_PASSWORD=pre'it's,SECRET do. Elsewhere
	// only one mark of punctuation that whitespace or the end of s follows
	// does.
	if s[k] != ' ' && s[k] != '\t' && s[k] != '\n' && s[k] != '\r' && !(contraction && (explicitOpen || joinedOpen)) {
		if strings.IndexByte(".,;:!?", s[k]) < 0 {
			return end
		}
		// Punctuation after the contraction suffix is allowed only if
		// followed by whitespace or end of string.
		if k+1 < len(s) && s[k+1] != ' ' && s[k+1] != '\t' && s[k+1] != '\n' && s[k+1] != '\r' {
			return end
		}
	}
	bound := sh.boundTo(s) && (len(q) == 1 || len(q) == 2)
	qIdx := len(q) - 1
	if explicitOpen {
		qIdx += 2
	}
	lineEnd := sh.stringEndOnLine(s, k, quoted)
	if bound && sh.noApostrophe[qIdx][2] == int(quoted) && sh.noApostrophe[qIdx][0] <= k && k < sh.noApostrophe[qIdx][1] {
		if explicitOpen && !gluedContraction || joinedOpen && sh.noApostrophe[qIdx][1] >= lineEnd {
			return sh.noApostrophe[qIdx][1]
		}
		return end
	}
	scanEnd := lineEnd
	if explicitOpen {
		if quoted == 0 {
			scanEnd = len(s)
		} else if lineEnd == lineBreakFrom(sh, s, k) && lineEnd < len(s) {
			// The outer string did not close on the first line; bound
			// scanEnd to the outer string's closing quote.
			outerQ := quoteKindString(quoted)
			if outerQ != "" {
				qcOuter, qcInner := outerQ, ""
				if len(outerQ) == 2 {
					qcOuter, qcInner = "", outerQ
				}
				closePos := quoteCloseScan(s, lineEnd, qcOuter, qcInner)
				if closePos > lineEnd {
					scanEnd = closePos - len(outerQ)
				} else {
					scanEnd = len(s)
				}
			}
		}
	}
	stop := scanEnd
	for i := k; i < scanEnd; {
		idx := strings.Index(s[i:scanEnd], q)
		if idx < 0 {
			break
		}
		c := i + idx
		if c > lineEnd {
			lineStart := i
			if nl := strings.LastIndexAny(s[i:c], "\r\n"); nl >= 0 {
				lineStart = i + nl + 1
			}
			if endsInKeyDelimiter(s[lineStart:c], isIdentifierKey) || opensQuotedPair(s, c, len(q), s[lineStart:c]) {
				stop = c
				break
			}
		}
		if gluedContraction && (endsInKeyDelimiter(s[i:c], isIdentifierKey) || opensQuotedPair(s, c, len(q), s[i:c])) {
			stop = c
			break
		}
		if endsInSensitiveKeyDelimiter(s[i:c]) || opensQuotedPair(s, c, len(q), s[i:c]) {
			if !explicitOpen || c > lineEnd {
				stop = c
				break
			}
			qv := quotedValueAt(sh, s, c)
			if len(qv) >= 2*len(q) && c+len(qv) <= scanEnd && !spansLine(sh, s, c, len(qv)) && !isMidWordApostrophe(s, c+len(qv)-len(q)) {
				i = c + len(qv)
				continue
			}
			i = c + len(q)
			continue
		}
		i = c + len(q)
		if isMidWordApostrophe(s, c) {
			continue
		}
		if i >= len(s) || s[i] == '\n' || s[i] == '\r' || quotedValueEnds(s[start:], i-start, quoted) || openingQuote(s, i) != "" {
			if explicitOpen && contraction && c < lineEnd {
				return laterValueQuoteEnd(s, start, i, lineEnd, q, quoted)
			}
			return i
		}
		if c > lineEnd {
			stop = c
			break
		}
	}
	// For an explicitly-opened unquoted value, find the first continuation
	// line after lineEnd that starts a new key assignment, and stop there,
	// so the value does not extend past a line like user=bob in
	// password='it's pad\nuser=bob.
	if explicitOpen && quoted == 0 && stop > lineEnd {
		searchFrom := lineEnd + 1
		if s[lineEnd] == '\r' && searchFrom < stop && s[searchFrom] == '\n' {
			searchFrom++
		}
		for contStart := searchFrom; contStart < stop; {
			lineEnd2 := stop
			if nl := strings.IndexAny(s[contStart:stop], "\r\n"); nl >= 0 {
				lineEnd2 = contStart + nl
			}
			contLine := s[contStart:lineEnd2]
			if eq := strings.IndexByte(contLine, '='); eq > 0 && eq <= maxNestedKeyLen && eq+1 < len(contLine) && contLine[eq+1] != '=' && skipHorizDelimSpace(contLine, eq+1) < len(contLine) {
				keyName := strings.TrimRightFunc(contLine[:eq], isDelimSpace)
				validKey := keyName != "" && isIdentifierKey(keyName)
				for ki := 0; validKey && ki < len(keyName); ki++ {
					if !isCamelTokenByte(keyName[ki]) {
						validKey = false
					}
				}
				if validKey {
					if contStart > lineEnd+1 && s[contStart-1] == '\n' && contStart-2 >= 0 && s[contStart-2] == '\r' {
						stop = contStart - 2
					} else if contStart > lineEnd && s[contStart-1] == '\n' {
						stop = contStart - 1
					} else {
						stop = contStart
					}
					break
				}
			}
			if lineEnd2 >= stop {
				break
			}
			contStart = lineEnd2 + 1
			if contStart < stop && s[contStart-1] == '\r' && s[contStart] == '\n' {
				contStart++
			}
		}
	}
	if bound && (!gluedContraction || stop >= lineEnd) {
		sh.noApostrophe[qIdx] = [3]int{k, stop, int(quoted)}
	}
	// A joined quote's scan stops before lineEnd only at a quote that opens
	// the value of a sensitive key, as the one after token= in
	// password=pre'it's token='x' does; the value then ends at the
	// apostrophe, so that the key and its value are read on their own.
	if explicitOpen && !gluedContraction || joinedOpen && stop >= lineEnd {
		return stop
	}
	return end
}

// laterValueQuoteEnd returns where a value ends that a quote opens explicitly
// at s[start], that holds the ending of a contraction (see
// interiorApostropheEnd), and whose quote q ending at s[from] would close it
// at a value boundary: at the end of the next quote q before lineEnd that
// ends at a value boundary too (see quotedValueEnds), as the last quote of
// password='it's' QQSA' user=x does, a shell joining the parts between them
// into the value, as it would for a value such as it's' QQSA written into
// the quotes as it is; at lineEnd when only horizontal whitespace separates
// s[from] from an unclosed quote q on the line, as in token='don't_'  'S; or
// at from, when no such quote follows, when prose words precede a quote that
// opens a new quoted word, or when a key and a key-value delimiter (see
// trailingKeyStart) or a quote that opens a quoted pair (see opensQuotedPair)
// comes first, as in password='it's' user=x and password='it's' note='x'.
// Apostrophes between two letters on the way, as the one of Bob's, are passed
// over.
func laterValueQuoteEnd(s string, start, from, lineEnd int, q string, quoted quoteKinds) int {
	keysFrom := from
	for i := from; i < lineEnd; {
		idx := strings.Index(s[i:lineEnd], q)
		if idx < 0 {
			break
		}
		c := i + idx
		for p := keysFrom; p < c; p++ {
			if (s[p] == '=' || s[p] == ':') && trailingKeyStart(s[from:p+1], isIdentifierKey) >= 0 {
				return from
			}
		}
		keysFrom = c
		i = c + len(q)
		if isMidWordApostrophe(s, c) {
			continue
		}
		if opensQuotedPair(s, c, len(q), s[from:c]) {
			return from
		}
		if i >= len(s) || s[i] == '\n' || s[i] == '\r' || quotedValueEnds(s[start:], i-start, quoted) {
			return i
		}
		if skipHorizDelimSpace(s, from) < c {
			return from
		}
		if !strings.Contains(s[i:lineEnd], q) && unclosedQuoteOpens(s, c, q, true, quoted) {
			return lineEnd
		}
	}
	return from
}

// gluedQuotedKeyStart returns the offset of the first quote in the bare value
// s[start+1:end] that is joined to a letter, a digit, "_", "-", "." or the "]"
// that ends a placeholder before it and that starts a quoted key whose value
// the text patterns redact on its own, with its delimiter and value (see
// startsQuotedKey), as the quote after abc does in token=abc'password': x and
// token=abc"password" = x; or -1. A shell would join such a key to the value,
// but read on its own the key keeps its value from being left in place after
// a redacted value that took the key. Only keys that close within
// maxNestedKeyLen bytes and inside the bare value are read, and a value is
// looked for after one key at most, so that the scan reads each byte of the
// bare value a bounded number of times.
func gluedQuotedKeyStart(s string, start, end int, quoted quoteKinds) int {
	for i := start + 1; i < end; i++ {
		q := openingQuote(s, i)
		if q == "" {
			continue
		}
		if p := s[i-1]; !isCamelTokenByte(p) && p != ']' {
			i += len(q) - 1
			continue
		}
		lim := min(end, i+len(q)+maxNestedKeyLen+len(q))
		if i+len(q) >= lim || !strings.Contains(s[i+len(q):lim], q) {
			i += len(q) - 1
			continue
		}
		loc := quotedKeyAtStart(s[i:])
		if loc == nil || s[i+loc[6]] == '=' && i+loc[7] < len(s) && s[i+loc[7]] == '=' {
			i += len(q) - 1
			continue
		}
		raw := s[i+loc[2] : i+loc[3]]
		trimmed := strings.TrimFunc(raw, isDelimSpace)
		if !quotedKeyAllowed(raw) && !isPreservedPageTokenName(trimmed) && !isTokenCounterName(trimmed) {
			i += len(q) - 1
			continue
		}
		if startsQuotedKey(s[i:], quoted) {
			return i
		}
		return -1
	}
	return -1
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
func textValueEnd(sh *shellAssignScanner, s string, start, end, open int, quoted quoteKinds) int {
	return textValueEndWithEnclosing(sh, s, start, end, open, quoted, 0, false, false, nil)
}

func textValueEndWithEnclosing(sh *shellAssignScanner, s string, start, end, open int, quoted, innerQuoted quoteKinds, subSQ, subEscDQ bool, quotes *openQuotes) int {
	if q := openingQuote(s, start); q != "" {
		if closesString(s, start, q, quoted) && !ownsQuotedValue(s, start, end, q) {
			return start
		}
		if _, quote := unquoteValue(s[start:end]); quote != "" {
			if quotedValueEnds(s[start:], end-start, quoted) {
				return end
			}
			// Check for an interior apostrophe in a single-quoted value, as in
			// password='it's SECRET': the closing quote is an interior word
			// apostrophe (a letter precedes it), and a later same-line quote of
			// the same kind closes the value. Extend end to that later closing
			// quote so the full passphrase is redacted.
			if strings.HasSuffix(q, "'") && end > start+len(q) {
				if nextEnd := interiorApostropheEnd(sh, s, start, end, q, quoted|innerQuoted); nextEnd > end {
					if strings.HasSuffix(s[start:nextEnd], q) && (!quotedValueEnds(s[start:], nextEnd-start, quoted|innerQuoted) || opensDelimitedValueWithEnclosing(sh, s, nextEnd, open, quoted, innerQuoted)) {
						return runOnValueEndWithEnclosing(sh, s, start, nextEnd, open, quoted, innerQuoted, subSQ)
					}
					return nextEnd
				}
			}
			return runOnValueEndWithEnclosing(sh, s, start, end, open, quoted, innerQuoted, subSQ)
		}
		if unclosedQuoteOpens(s, start, q, true, quoted|innerQuoted) {
			return unclosedEnclosingEnd(s, start+len(q), open, q, quoted, innerQuoted)
		}
	}
	// A bare value that ends at a backslash-escaped line break, as in
	// DB_PASSWORD=pre\<LF>SECRET, continues past the line break, as
	// bareValueEnd reads. The regex match stops at the line break, so
	// extend end to where bareValueEnd reads the value.
	if openingQuote(s, start) == "" && end > start && end < len(s) &&
		(s[end] == '\n' || s[end] == '\r') && s[end-1] == '\\' {
		end = bareValueEnd(s, start)
	}
	// A quote joined to the bare value that starts a quoted key whose value
	// the text patterns redact on its own (see gluedQuotedKeyStart), as the
	// one after abc does in token=abc'password': x, ends the value there, so
	// that the key is read with its value rather than redacted with this
	// value while its own value is left in place.
	if openingQuote(s, start) == "" {
		if g := gluedQuotedKeyStart(s, start, end, quoted); g > start {
			return g
		}
		end = extendSubshellQuotedValueEnd(quotes, s, start, end, quoted, innerQuoted, subSQ, subEscDQ)
	}
	// rest is where the text after a JSON object or array value starts.
	rest := sensitiveValueEndWith(sh, s, start, start)
	if rest < end {
		// The closing bracket of an object or array that starts the value
		// is a byte of the value, so a quote joined to it opens a quoted
		// part of the value (see joinedQuoteEnd), as the one after the
		// placeholder does in password=[REDACTED]"x": "y", rather than
		// closes one, unless it ends a string open before the value (see
		// endsString), as the one after the placeholder does in
		// failed: "password=[REDACTED]". retry.
		from := rest
		if rest > start && !endsString(s, rest, openingQuote(s, rest), quoted) {
			from = rest - 1
		}
		if opened := openedValueEndWithEnclosing(sh, s, from, end, open, false, quoted, innerQuoted, subSQ, subEscDQ); opened != end {
			if opened < end && opened >= start {
				return rest + len(trimEnclosingClosers(s[rest:max(opened, rest)], open))
			}
			return opened
		}
	}
	if rest == start {
		if spaced := spacedValueEndWithEnclosing(sh, s, end, open, quoted, innerQuoted); spaced > end {
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
// open before the value. A closing quote followed at once by a pair whose
// value the text patterns redact on its own (see startsRedactedPair), as in
// DB_PASSWORD="a"GITHUB_TOKEN="b", ends at the closing quote so that the pair
// is read on its own; a compound name ending in a sensitive key suffix on the
// same line, as in "abc"tail_token:8080, may be a credential tail and is
// redacted together with its pair through nestedPairEnd; and a pair whose key
// is not sensitive, as the user="b" of password="a"user="b", is a part of the
// word that a shell joins to the value, which runs on over it. A word that
// ends in a delimiter, as the "x": of password="a""x": "b" does, runs on over
// the quoted value, object or array that whitespace and the delimiter open
// (see delimitedValueEnd), so that no value is left after it without its key.
func unclosedEnclosingEnd(s string, afterOpen, open int, q string, quoted, innerQuoted quoteKinds) int {
	enclosing := (quoted | innerQuoted) &^ quoteKind(q)
	if enclosing == 0 || (open == 0 && innerQuoted&(4|8) == 0) {
		return len(s)
	}
	for k := afterOpen; k < len(s); {
		c := s[k]
		if c == '\n' || c == '\r' {
			return len(s)
		}
		if c == '\\' {
			j := k
			for j < len(s) && s[j] == '\\' {
				j++
			}
			bs := j - k
			if j < len(s) && (s[j] == '"' || s[j] == '\'') {
				if bs%4 == 1 {
					qk := quoteKinds(4)
					if s[j] == '\'' {
						qk = 8
					}
					if enclosing&qk != 0 && endsString(s, j-1, s[j-1:j+1], enclosing) && hasJSONValueCloserAfter(s, j+1, true) {
						return j - 1
					}
				} else if bs%2 == 0 {
					qk := quoteKinds(1)
					if s[j] == '\'' {
						qk = 2
					}
					if enclosing&qk != 0 && (qk != 2 || !isApostrophe(s, j, 2, 2)) && endsString(s, j, s[j:j+1], enclosing) && hasJSONValueCloserAfter(s, j+1, innerQuoted&4 != 0) {
						return j
					}
				}
				k = j + 1
				continue
			}
			if bs%2 == 0 {
				k = j
				continue
			}
			if j < len(s) && s[j] != '\n' && s[j] != '\r' {
				k = j + 1
				continue
			}
			k = j
			continue
		}
		if c == '"' && enclosing&1 != 0 && endsString(s, k, `"`, enclosing) && hasJSONValueCloserAfter(s, k+1, innerQuoted&4 != 0) {
			return k
		}
		if c == '\'' && enclosing&2 != 0 && !isApostrophe(s, k, 2, 2) && endsString(s, k, `'`, enclosing) && hasJSONValueCloserAfter(s, k+1, innerQuoted&4 != 0) {
			return k
		}
		k++
	}
	return len(s)
}

func runOnValueEnd(sh *shellAssignScanner, s string, start, end, open int, quoted quoteKinds) int {
	return runOnValueEndWithEnclosing(sh, s, start, end, open, quoted, 0, false)
}

func runOnValueEndWithEnclosing(sh *shellAssignScanner, s string, start, end, open int, quoted, innerQuoted quoteKinds, subSQ bool) int {
	inner, _ := unquoteValue(s[start:end])
	if opensLaterValue(inner, s[end:]) {
		return forwardValueEndWithEnclosing(sh, s, start, end, open, quoted, innerQuoted)
	}
	if kind, pair := afterClosedQuotePair(sh, s, end, quoted); kind == valueEndingInKeyName {
		return nestedPairEndWithEnclosing(sh, s, pair, end, open, quoted, innerQuoted)
	}
	if startsRedactedPair(s, end, quoted) {
		return end
	}
	word := runOnWordEndWithEnclosing(sh, s, end, open, quoted, innerQuoted, subSQ)
	for word > end && (s[word-1] == ':' || s[word-1] == '=') {
		e, runsOn := delimitedValueEndWithEnclosing(sh, s, word, open, quoted, innerQuoted)
		if e < 0 {
			break
		}
		if !runsOn {
			return e
		}
		end, word = e, runOnWordEndWithEnclosing(sh, s, e, open, quoted, innerQuoted, subSQ)
	}
	return end + len(trimEnclosingClosers(s[end:word], open))
}

// delimitedValueEnd returns the end of the quoted value, JSON object or JSON
// array that horizontal whitespace and the delimiter that ends the word s[:i]
// open, as the "b" of x: "b" or the {"c": "d"} of x= {"c": "d"} does, and
// whether the closing quote of that quoted value runs into more of a word
// (see quotedValueEnds); or -1 if none opens there. A quote that closes a
// string open before the value opens nothing (see closesString), and one that
// is not closed opens a value that runs on to the end of s (see
// unclosedQuoteOpens). sh, if bound to s, keeps what it works out (see
// delimitedChainEnd).
func delimitedValueEndWithEnclosing(sh *shellAssignScanner, s string, i, open int, quoted, innerQuoted quoteKinds) (int, bool) {
	d := delimitedChainEndWithEnclosing(sh, s, i, open, quoted, innerQuoted)
	return d.end, d.runsOn
}

// opensDelimitedValue reports whether s[i:] starts with zero or more closing
// brackets (")", "]" or "}") and ":" followed by horizontal whitespace and a
// quoted value, JSON object or JSON array (see delimitedValueEnd).
func opensDelimitedValue(sh *shellAssignScanner, s string, i int, quoted quoteKinds) bool {
	return opensDelimitedValueWithEnclosing(sh, s, i, 0, quoted, 0)
}

func opensDelimitedValueWithEnclosing(sh *shellAssignScanner, s string, i, open int, quoted, innerQuoted quoteKinds) bool {
	if j := delimiterAfterClosers(s, i); j >= 0 {
		return delimitedChainEndWithEnclosing(sh, s, j+1, open, quoted, innerQuoted).end >= 0
	}
	return false
}

// delimiterAfterClosers returns the offset of the ":" that zero or more
// closing brackets (")", "]" or "}") at s[i:] lead to, or -1.
func delimiterAfterClosers(s string, i int) int {
	for i < len(s) && (s[i] == ')' || s[i] == ']' || s[i] == '}') {
		i++
	}
	if i >= len(s) || s[i] != ':' {
		return -1
	}
	return i
}

// delimitedEnd is what delimitedValueEnd returns for an offset: the end of
// the value, or -1, and whether the value runs on.
type delimitedEnd struct {
	end    int
	runsOn bool
}

// delimitedKey returns the key of shellAssignScanner.delimited for what
// delimitedValueEnd returns at s[i:] when the quotes in quoted are open.
func delimitedKey(i, open int, quoted, innerQuoted quoteKinds) int {
	openBit := 0
	if open > 0 {
		openBit = 1
	}
	return i<<9 | openBit<<8 | int(innerQuoted&0xf)<<4 | int(quoted&0xf)
}

// delimitedChainEnd returns what delimitedValueEnd returns for s[i:]. What
// it returns for a quoted value whose closing quote ends a value and a ":"
// follows (see opensDelimitedValue) depends on whether that ":" opens
// another value, which may in turn depend on the value after it, and so on,
// as along "x": "x": "x": ...: it reads such a chain forwards once and then
// works out each value from the one after it, rather than reading the rest
// of the chain again from each value, and sh, if bound to s, keeps what it
// worked out for each, so that a caller that asks about each value of the
// chain in turn, as runOnValueEnd does, reads the chain once.
func delimitedChainEndWithEnclosing(sh *shellAssignScanner, s string, i, open int, quoted, innerQuoted quoteKinds) delimitedEnd {
	bound := sh.boundTo(s)
	enclosing := quoted | innerQuoted
	// chain holds, for each value whose result waits on the next one, the
	// offset it was asked about, where its quote opens and where it ends.
	type link struct{ from, at, end int }
	var chain []link
	var d delimitedEnd
	for {
		if bound {
			if memo, ok := sh.delimited[delimitedKey(i, open, quoted, innerQuoted)]; ok {
				d = memo
				break
			}
		}
		step, at, next := delimitedStepWithEnclosing(sh, s, i, open, quoted, innerQuoted)
		if next < 0 {
			d = step
			if bound {
				sh.keepDelimited(i, open, quoted, innerQuoted, d)
			}
			break
		}
		chain = append(chain, link{from: i, at: at, end: step.end})
		i = next
	}
	for k := len(chain) - 1; k >= 0; k-- {
		l := chain[k]
		switch {
		case d.end >= 0:
			d = delimitedEnd{end: l.end, runsOn: true}
		case valueStartsAt(s, l.at, true, enclosing):
			d = delimitedEnd{end: l.end}
		default:
			d = delimitedEnd{end: -1}
		}
		if bound {
			sh.keepDelimited(l.from, open, quoted, innerQuoted, d)
		}
	}
	return d
}

// keepDelimited records d as what delimitedValueEnd returns at s[i:] when
// the quotes in quoted are open.
func (sc *shellAssignScanner) keepDelimited(i, open int, quoted, innerQuoted quoteKinds, d delimitedEnd) {
	if sc.delimited == nil {
		sc.delimited = make(map[int]delimitedEnd)
	}
	sc.delimited[delimitedKey(i, open, quoted, innerQuoted)] = d
}

// delimitedStep works out what delimitedValueEnd returns for s[i:] where
// that does not depend on a value after it, and returns it with next < 0.
// Where the quoted value at s[i:] ends a value and a ":" follows (see
// delimiterAfterClosers), whether the value runs on depends on whether that
// ":" opens another value: it returns the end of the value in d.end, where
// its quote opens in at, and next, the offset after the ":", where the
// caller reads on.
func delimitedStepWithEnclosing(sh *shellAssignScanner, s string, i, open int, quoted, innerQuoted quoteKinds) (d delimitedEnd, at, next int) {
	none := delimitedEnd{end: -1}
	enclosing := quoted | innerQuoted
	at = skipHorizDelimSpace(s, i)
	if at == i || at >= len(s) {
		return none, at, -1
	}
	if c := s[at]; c == '{' || c == '[' {
		return delimitedEnd{end: sensitiveValueEndWith(sh, s, at, at)}, at, -1
	}
	quote := openingQuote(s, at)
	if quote == "" || closesString(s, at, quote, enclosing) {
		return none, at, -1
	}
	q := quotedValueAt(sh, s, at)
	if q == "" {
		if unclosedQuoteOpens(s, at, quote, true, enclosing) {
			return delimitedEnd{end: unclosedEnclosingEnd(s, at+len(quote), open, quote, quoted, innerQuoted)}, at, -1
		}
		return none, at, -1
	}
	end := at + len(q)
	if !quotedValueEnds(s[at:], len(q), enclosing) {
		return delimitedEnd{end: end, runsOn: true}, at, -1
	}
	if j := delimiterAfterClosers(s, end); j >= 0 {
		return delimitedEnd{end: end}, at, j + 1
	}
	if !valueStartsAt(s, at, true, enclosing) {
		return none, at, -1
	}
	return delimitedEnd{end: end}, at, -1
}

// runOnWordEnd returns the end of the word that starts at s[i:]: the first
// whitespace, ",", ";" or "&" in it that is not inside a quoted value (see
// quotedValueRegex) that starts in it and closes on its line, or the end of
// the text if a joined quote that no quote closes opens a value (see
// unclosedQuoteOpens), so that a quoted string joined to a credential is read
// whole and its words are not left in place, as the 'b c' of the shell word
// 'a'"'"'b c' and the words of an SQL string that doubles its quote would be.
// A quote does not start a quoted value if it closes a string that is open
// before the credential in quoted (see endsString), or if it is a single
// quote after bare word bytes that does not close at a value boundary or
// before another quote and either starts a contraction while what it quotes
// ends in whitespace or does not join the text after it (see
// closingQuoteJoins), so that the word stops at the it of it's ' x while
// shell-spliced segments such as DB_PASSWORD="a"pre'correct horse'suffix are
// consumed. The word starts after a closed quoted part, which makes it a
// shell word, so a quote that closes a later part is not read as the
// apostrophe of a prose word, as the last quote of
// DB_PASSWORD="a"pre' violetfern's is not. A single quote that
// quotedValueRegex finds no closing quote for is closed by the quote of a
// later \' on its line, as a shell reads it (see shellQuotedSegment).
func closesJSONContainerValue(s string, i int, q string, enclosing, innerQuoted quoteKinds, open int) bool {
	if open == 0 && innerQuoted&quoteKind(q) == 0 {
		return false
	}
	if !endsString(s, i, q, enclosing) {
		return false
	}
	return hasJSONValueCloserAfter(s, i+len(q), innerQuoted&4 != 0)
}

// runOnWordEndWithEnclosing returns the end of the word starting at s[i:], stopping before
// a closing quote that terminates an enclosing JSON container string.
func runOnWordEndWithEnclosing(sh *shellAssignScanner, s string, i, open int, quoted, innerQuoted quoteKinds, subSQ bool) int {
	end, _ := runOnWordWithEnclosing(sh, s, i, open, quoted, innerQuoted, subSQ)
	return end
}

// runOnWord returns the end of the word that starts at s[i:] (see
// runOnWordEndWithEnclosing) and whether every quoted part that the word joins closes in
// it, as the shell reads the word: not if a joined quote that no quote closes
// runs the word on to the end of the text (see unclosedQuoteOpens), which
// leaves the string that the quote opens open, nor if the word ends after a
// quote that closes a string open before the word (see startsQuotedKey).
func runOnWord(sh *shellAssignScanner, s string, i int, quoted quoteKinds) (int, bool) {
	return runOnWordWithEnclosing(sh, s, i, 0, quoted, 0, false)
}

func runOnWordWithEnclosing(sh *shellAssignScanner, s string, i, open int, quoted, innerQuoted quoteKinds, subSQ bool) (int, bool) {
	joined := true
	enclosing := quoted | innerQuoted
	allQuoted := enclosing
	if subSQ {
		allQuoted |= 2
	}
	for i < len(s) {
		if s[i] == '\\' {
			j := i
			for j < len(s) && s[j] == '\\' {
				j++
			}
			nBS := j - i
			if j < len(s) && (s[j] == '"' || s[j] == '\'') && nBS%2 == 1 {
				if nBS%4 == 1 {
					if closesJSONContainerValue(s, j-1, s[j-1:j+1], enclosing, innerQuoted, open) {
						return j - 1, false
					}
					if endsString(s, j-1, s[j-1:j+1], quoted) {
						i = j + 1
						joined = false
						continue
					}
					i = j - 1
				} else {
					i = j + 1
					joined = false
					continue
				}
			} else if nBS%2 == 0 {
				i = j
				joined = false
				if i >= len(s) {
					break
				}
			}
		}
		if q := openingQuote(s, i); q != "" {
			if closesJSONContainerValue(s, i, q, enclosing, innerQuoted, open) {
				return i, false
			}
			if !endsString(s, i, q, allQuoted) {
				if !joined && startsQuotedKey(s[i+len(q):], quoted) {
					// A quote after bare word bytes that a sensitive or
					// preserved quoted key and its value follow (see
					// startsQuotedKey) closes the string that the word ends,
					// as the quote after QQSA in "token":" a b "QQSA""token":
					// does, rather than opening an empty string that joins
					// the key to the word: the word ends after the quote, and
					// the pair after it is matched on its own.
					return i + len(q), false
				}
				m := findQuotedOrShellSegment(sh, s, i)
				if m != "" {
					inner, _ := unquoteValue(m)
					last, _ := utf8.DecodeLastRuneInString(inner)
					hasNL := spansLine(sh, s, i, len(m))
					if !hasNL && (joined || !strings.HasSuffix(q, "'")) ||
						!multilineContractionOpensPair(sh, s, i, m, enclosing) &&
							(quotedValueEnds(s[i:], len(m), enclosing) ||
								openingQuote(s, i+len(m)) != "" ||
								(!startsContraction(s, i, len(q)) || !isDelimSpace(last)) && (!hasNL || !isDelimSpace(last)) && closingQuoteJoins(sh, s, i, m, true)) {
						i += len(m)
						joined = true
						continue
					}
				} else {
					// A quote that no quote closes opens a part that runs on to
					// the end of the text, past its line too, as a shell reads
					// it: DB_PASSWORD="a"pre'correct<LF>SECRET is one value. The
					// closed part that the word starts with makes it a shell
					// word, so a single quote that would read as the apostrophe
					// of a contraction or a word in prose opens a part too, as
					// those of DB_PASSWORD="a"pre'd secret and
					// DB_PASSWORD="a"pre'<LF>SECRET do.
					if unclosedQuoteOpens(s, i, q, true, allQuoted) {
						return unclosedEnclosingEnd(s, i+len(q), open, q, quoted, innerQuoted), false
					}
				}
			}
		}
		if strings.IndexByte(regexpSpace+",;&", s[i]) >= 0 {
			break
		}
		i++
		joined = false
	}
	return i, true
}

// afterClosedQuotePair classifies a sensitive key-value pair that begins
// immediately at s[end:] on the same line after a closed quoted value:
// nestedKey for a standalone sensitive key name (even when whitespace separates
// its delimiter from its value) or an unspaced compound sensitive assignment
// with an immediate value (as in DB_PASSWORD="a"GITHUB_TOKEN="b"),
// valueEndingInKeyName and the offset of the sensitive key name for a compound
// name ending in a credential suffix before ":port" or spaced delimiters, or
// notNestedKey when no sensitive pair with a value starts at s[end:].
func afterClosedQuotePair(sh *shellAssignScanner, s string, end int, quoted quoteKinds) (nestedKeyKind, int) {
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
		if bareEnd := bareWordEnd(sh, s, end); bareEnd > end {
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
func spacedValueEndWithEnclosing(sh *shellAssignScanner, s string, end, open int, quoted, innerQuoted quoteKinds) int {
	at := skipHorizDelimSpace(s, end)
	if at == end || at >= len(s) || (s[at] != ':' && s[at] != '=') {
		return end
	}
	at = skipDelimSpace(s, at+1)
	if at >= len(s) {
		return end
	}
	if c := s[at]; c == '{' || c == '[' {
		return sensitiveValueEndWith(sh, s, at, at)
	}
	enclosing := quoted | innerQuoted
	if quote := openingQuote(s, at); quote != "" {
		if closesString(s, at, quote, enclosing) {
			return end
		}
		if q := findQuotedValue(s[at:]); q != "" {
			if !quotedValueEnds(s[at:], len(q), enclosing) || opensDelimitedValueWithEnclosing(sh, s, at+len(q), open, quoted, innerQuoted) {
				return runOnValueEndWithEnclosing(sh, s, at, at+len(q), open, quoted, innerQuoted, false)
			}
			if valueStartsAt(s, at, true, enclosing) {
				return at + len(q)
			}
			return end
		}
		if unclosedQuoteOpens(s, at, quote, true, enclosing) {
			return unclosedEnclosingEnd(s, at+len(quote), open, quote, quoted, innerQuoted)
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
// and not if an enclosing outer quote closes the bare value right after a
// contraction ending (see closesOuterQuote), as in
// failed: "token=abc:'s"; the user's request. A quoted value that opens a
// value but does not end where a value may end (see quotedValueEnds), as in
// password=a = "b"c, password=a="b"c or password=a="b"'c', runs on (see
// runOnValueEnd, where open is as for textValueEnd), so that the credential
// in it is not left in place after the redacted bare value. A quote joined to
// the bytes of the bare value before it, with no delimiter between them, as
// in DB_PASSWORD=pre"correct horse staple", opens a quoted part of the same
// value too (see joinedQuoteEnd).
func closedQuoteBeforePipe(s string, afterQuote, end int) bool {
	k := afterQuote
	for k < end && (s[k] == ')' || s[k] == ']' || s[k] == '}') {
		k++
	}
	return k < end && s[k] == '|'
}

func subSQClosedBefore(s string, start, end int) bool {
	cur := true
	bs := 0
	for i := start; i < end; i++ {
		if s[i] == '\\' {
			bs++
			continue
		}
		b := bs
		bs = 0
		if s[i] != '\'' || b%2 == 1 {
			continue
		}
		if cur {
			if isMidWordApostrophe(s, i) && !opensPrefixedString(s, i) {
				continue
			}
			cur = false
		} else if (!isMidWordApostrophe(s, i) || opensPrefixedString(s, i)) && !isTrailingApostrophe(s, i) {
			cur = true
		}
	}
	return !cur
}

func sqCloseBeforePipe(s string, i int) int {
	for k := i; k < len(s); k++ {
		if strings.IndexByte(" \t\r\n,;|&", s[k]) >= 0 {
			return -1
		}
		if s[k] == '\'' {
			if closedQuoteBeforePipe(s, k+1, len(s)) {
				return k
			}
			if isMidWordApostrophe(s, k) && !opensPrefixedString(s, k) {
				continue
			}
			return -1
		}
	}
	return -1
}

func extendSubshellQuotedValueEnd(quotes *openQuotes, s string, start, end int, quoted, innerQuoted quoteKinds, subSQ, subEscDQ bool) int {
	if end >= len(s) || (s[end] != ';' && s[end] != '&' && s[end] != ',') {
		return end
	}
	subParen := quotes != nil && quotes.active != 0 && quotes.openInParen
	var mode byte
	switch {
	case subEscDQ && innerQuoted&4 != 0:
		mode = 3
	case subEscDQ:
		mode = 2
	case (subSQ || (quoted&2 != 0 && innerQuoted == 0)) && !subSQClosedBefore(s, start, end):
		mode = 1
	case subParen && quoted&1 != 0 && innerQuoted == 0:
		mode = 4
	case subParen && quoted&4 != 0 && innerQuoted == 0:
		mode = 5
	case subParen && quoted&8 != 0 && innerQuoted == 0:
		mode = 6
	default:
		return end
	}
	if quotes != nil && quotes.noSubCloseMode == mode && end >= quotes.noSubCloseFrom && end < quotes.noSubCloseUntil {
		return end
	}
	recordFail := func(stop int) int {
		if quotes != nil {
			quotes.noSubCloseMode = mode
			quotes.noSubCloseFrom = end
			quotes.noSubCloseUntil = stop
		}
		return end
	}
	for k := end; k < len(s); {
		c := s[k]
		if c == '\n' || c == '\r' || c == ' ' || c == '\t' {
			return recordFail(k)
		}
		if r, _ := utf8.DecodeRuneInString(s[k:]); isDelimSpace(r) {
			return recordFail(k)
		}
		if (c == ';' || c == '&' || c == ',') && k+1 < len(s) && (startsRedactedPair(s, k+1, quoted) || startsBearerKeyword(s, k+1)) {
			return recordFail(k)
		}
		qStart, qEnd := -1, -1
		if c == '\\' {
			j := k
			for j < len(s) && s[j] == '\\' {
				j++
			}
			nBS := j - k
			if j < len(s) {
				switch s[j] {
				case '"':
					if nBS%2 == 0 && quoted&1 != 0 && mode != 4 {
						return recordFail(j)
					}
					if nBS%4 == 1 && (innerQuoted|quoted)&4 != 0 && mode != 5 {
						return recordFail(j - 1)
					}
				case '\'':
					if nBS%2 == 0 && quoted&2 != 0 && mode != 1 {
						return recordFail(j)
					}
					if nBS%4 == 1 && (innerQuoted|quoted)&8 != 0 && mode != 6 {
						return recordFail(j - 1)
					}
				}
			}
			switch mode {
			case 1:
				if j < len(s) && s[j] == '\'' {
					qStart, qEnd = j, j+1
				}
			case 4:
				if j < len(s) && s[j] == '"' && nBS%2 == 0 {
					qStart, qEnd = j, j+1
				}
			case 2, 3:
				if j < len(s) && s[j] == '"' {
					if mode == 3 && nBS%8 == 7 {
						qStart, qEnd = j-7, j+1
					} else if nBS%4 == 3 {
						qStart, qEnd = j-3, j+1
					}
				}
			case 5, 6:
				wantQuote := byte('"')
				if mode == 6 {
					wantQuote = '\''
				}
				if j < len(s) && s[j] == wantQuote && nBS%4 == 1 {
					qStart, qEnd = j-1, j+1
				}
			}
			if qStart < 0 {
				if j == len(s) {
					k = j
				} else if nBS%2 == 1 && (s[j] == '"' || s[j] == '\'') {
					k = j + 1
				} else {
					k = j
				}
				continue
			}
		} else {
			switch c {
			case '"':
				if mode == 4 {
					qStart, qEnd = k, k+1
				} else if quoted&1 != 0 {
					return recordFail(k)
				}
			case '\'':
				if mode == 1 {
					if isMidWordApostrophe(s, k) && !opensPrefixedString(s, k) {
						k++
						continue
					}
					qStart, qEnd = k, k+1
				} else if quoted&2 != 0 {
					return recordFail(k)
				}
			}
			if qStart < 0 {
				k++
				continue
			}
		}
		if qStart >= 0 {
			if placeholderDelimBoundary(s, qEnd, quoted|innerQuoted, true) {
				return qStart
			}
			return bareValueEnd(s, qEnd)
		}
	}
	return recordFail(len(s))
}

func openedValueEndWithEnclosing(sh *shellAssignScanner, s string, start, end, open int, trailingQuoteOpens bool, quoted, innerQuoted quoteKinds, subSQ, subEscDQ bool) int {
	curInner, curSubSQ, curSubEscDQ := innerQuoted, subSQ, subEscDQ
	for i := start; i < end; i++ {
		if s[i] != ':' && s[i] != '=' {
			if i > start {
				if s[i] == '\\' {
					k := i
					for k < end && s[k] == '\\' {
						k++
					}
					nBS := k - i
					if k < end && (s[k] == '"' || s[k] == '\'') && nBS%2 == 1 {
						if s[k] == '"' && curSubEscDQ && !curSubSQ &&
							((curInner&4 == 0 && nBS%4 == 3) || (curInner&4 != 0 && (nBS%8 == 7 || nBS%4 == 3))) {
							qLen := 4
							if curInner&4 != 0 && nBS%8 == 7 {
								qLen = 8
							}
							qStart := k + 1 - qLen
							if qStart >= start && closedQuoteBeforePipe(s, k+1, len(s)) {
								return qStart
							}
							curSubEscDQ = false
							i = k
							continue
						}
						if s[k] == '"' && nBS%4 == 1 && (curInner|quoted)&4 != 0 && !curSubSQ && !curSubEscDQ {
							if k-1 >= start && closedQuoteBeforePipe(s, k+1, len(s)) {
								return k - 1
							}
							if curInner&4 != 0 {
								curInner &^= 4
							}
							i = k
							continue
						}
						if s[k] == '\'' && nBS%4 == 1 && (curInner|quoted)&8 != 0 {
							if k-1 >= start && closedQuoteBeforePipe(s, k+1, len(s)) {
								return k - 1
							}
							if curInner&8 != 0 {
								curInner &^= 8
							}
							i = k
							continue
						}
						if k-1 > i {
							i = k
							continue
						}
					} else {
						i = k - 1
						continue
					}
				} else if curSubSQ && s[i] == '\'' {
					if closedQuoteBeforePipe(s, i+1, len(s)) {
						return i
					}
					if isMidWordApostrophe(s, i) && !opensPrefixedString(s, i) {
						if pipeEnd := sqCloseBeforePipe(s, i); pipeEnd >= 0 {
							return pipeEnd
						}
					} else {
						curSubSQ = false
						continue
					}
				} else if !curSubSQ && !curSubEscDQ && ((quoted&2 != 0 && s[i] == '\'') || (quoted&1 != 0 && s[i] == '"')) {
					if closedQuoteBeforePipe(s, i+1, len(s)) {
						return i
					}
					if s[i] == '\'' && quoted&2 != 0 && isMidWordApostrophe(s, i) && !opensPrefixedString(s, i) {
						if pipeEnd := sqCloseBeforePipe(s, i); pipeEnd >= 0 {
							return pipeEnd
						}
					}
				}
				e, next := joinedQuoteEndWithEnclosing(sh, s, i, end, open, quoted, curInner, curSubSQ)
				if e > end {
					return e
				}
				if s[i] == '\\' && next <= i+1 {
					i = i + 1
					continue
				}
				i = max(i, next-1)
			}
			continue
		}
		at := skipDelimSpace(s, i+1)
		if at == len(s) {
			continue
		}
		var e int
		if c := s[at]; c == '{' || c == '[' {
			e = sensitiveValueEndWith(sh, s, at, at)
		} else if quote := openingQuote(s, at); quote != "" {
			if quote == "'" && curSubSQ {
				curSubSQ = false
				i = at
				continue
			}
			if quote == `\"` && curInner&4 != 0 && !curSubSQ && !curSubEscDQ {
				curInner &^= 4
				i = at + 1
				continue
			}
			if quote == `\'` && curInner&8 != 0 {
				curInner &^= 8
				i = at + 1
				continue
			}
			allQuoted := quoted | curInner
			if curSubSQ {
				allQuoted |= 2
			}
			checkClose := !trailingQuoteOpens || at >= end
			if checkClose && at+len(quote) >= end && closesString(s, at, quote, allQuoted) {
				return end
			}
			q := findQuotedValue(s[at:])
			if q == "" {
				if n := contractionLen(s[at+len(quote):]); n > 0 && at == i+1 && s[i] == ':' && quote == "'" {
					cEnd := at + len(quote) + n
					if closesOuterQuote(s, cEnd, end, open, quoted) || (cEnd == end && laterConstructFollowsContraction(s[end:], quoted)) {
						return end
					}
				}
				if unclosedQuoteOpens(s, at, quote, at >= end, allQuoted) {
					return unclosedEnclosingEnd(s, at+len(quote), open, quote, quoted, curInner)
				}
				return end
			}
			// A quote inside the bare value whose quoted value runs on past
			// it opens no value when an enclosing outer quote closes the
			// bare value right after a contraction ending, or when its
			// closing quote opens a later pair, a Bearer token or a quoted
			// key, as in token=abc:'s label=' x',
			// token=abc:'s note Bearer ' x' or token=abc:'s note 'password': ' x'.
			if at < end && at+len(q) > end {
				if n := contractionLen(s[at+len(quote):]); n > 0 && at == i+1 && s[i] == ':' && quote == "'" &&
					closesOuterQuote(s, at+len(quote)+n, end, open, quoted) {
					return end
				}
				if inner, _ := unquoteValue(q); opensLaterConstruct(s, at, len(q), len(quote), inner, quoted) {
					return end
				}
			}
			// A quote that ends the bare value, or follows it, opens a quoted
			// value that does not end where a value may end, as in
			// password=a="b"c, even if trailingQuoteOpens is set.
			runsOn := !quotedValueEnds(s[at:], len(q), allQuoted) || opensDelimitedValueWithEnclosing(sh, s, at+len(q), open, quoted, curInner)
			if at+len(quote) >= end && runsOn {
				return runOnValueEndWithEnclosing(sh, s, at, at+len(q), open, quoted, curInner, curSubSQ)
			}
			// A quote inside the bare value whose quoted value runs on past
			// it but does not end there opens a value that runs on, as in
			// token=a:'s b 1'x, password=a='d b'c or
			// export DB_PASSWORD=a:'s correct staple horse; the user's.
			if at+len(quote) < end && at+len(q) > end && runsOn {
				return runOnValueEndWithEnclosing(sh, s, at, at+len(q), open, quoted, curInner, curSubSQ)
			}
			// A quote or value that follows a quoted value closed in or at the
			// end of the bare value, as in token=a:'b'' c d' or
			// password=a:'x_token=' "b c", is read with it (see runOnValueEnd).
			if at+len(q) <= end && runsOn {
				if w := runOnValueEndWithEnclosing(sh, s, at, at+len(q), open, quoted, curInner, curSubSQ); w > end {
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

// joinedQuoteEnd returns the end of the value that the quote at s[i:], joined
// to the byte of the bare value s[:end] before it, which is not a delimiter
// (see openedValueEnd), opens if it runs on past end, and else end; and, as
// next, the end of what the quote quotes if that is closed inside the bare
// value, so that the caller reads it once, or else i. As a shell joins them,
// the quoted parts of DB_PASSWORD=pre"correct horse staple" and
// DB_PASSWORD=pre'correct horse'"staple" are parts of one value, and a value
// whose quote is not closed, as DB_PASSWORD=pre"correct horse is, runs on to
// the end of s (see unclosedQuoteOpens). A quote that closes a string open
// before the value (see closesString) opens nothing, as the one after abc does
// in got "password=abc" want "x". Whether a single quote that more of the
// bare value follows opens a quoted part is decided at that quote: it does
// unless it starts the ending of a contraction (see startsContraction), as
// the apostrophes of password=abc's value, password=it's Bob's and
// password=doesn't work do, and is not closed on its line at a value boundary,
// before another quote, or joining text without trailing delimiter whitespace
// (as DB_PASSWORD=pre's secret'; echo ok is). The quote that closes the part
// then closes it wherever it stands, also between two letters or before more
// of the word, as the second quote of password=pre'a b'c does, whose c runs on
// with the value (see runOnValueEnd), and a part that is not closed runs on to
// the end of s, as one of a double quote does. A quote that ends the bare
// value, so that what it quotes starts with the whitespace, "," or ";" that
// ended it (including a newline, as in DB_PASSWORD=pre'\nsecret'; echo ok),
// opens a quoted part as a quote inside the bare value does if what it quotes
// starts with horizontal whitespace and does not run past its line, as in
// DB_PASSWORD=pre" correct horse", DB_PASSWORD=pre' correct horse',
// DB_PASSWORD=pre" correct horse"staple or DB_PASSWORD=pre" correct horse;
// otherwise only if what it quotes is closed by a quote that ends a value, is
// followed by another quote, or (for a single quote) joins text after it, and
// not in got password=abc", want "x", where the closing quote is joined to x.
// A single quote that ends the bare value opens a quoted part only if what it
// quotes is closed by a quote that ends a value (see quotedValueEnds), that
// another quote follows, or that joins the text after it to the part (see
// closingQuoteJoins), as the closing quote of DB_PASSWORD=pre' correct
// horse'staple does, as such a quote is more often a trailing apostrophe, as
// in the password=users' wasn't changed, or the start of an SQL string, as in
// password=abc' OR '1'='1, than a shell quote. A quote that ends the bare
// value, that nothing closes and that a line break follows in a shell
// assignment (see inShellAssignment), as the one of
// DB_PASSWORD=pre'<LF>SECRET does, opens a part that runs on to the end of s.
// A quoted part that runs on past end is not joined if its
// closing quote opens a later construct (see opensLaterConstruct) or a quoted
// pair (see opensQuotedPair), as the second quote of
// password=a"b x "token=SECRETP-token: " in config does. A quote right after
// an opening bracket opens an argument or element, as in
// f("SECRET123 token=SECRET456"), whose quoted part is joined whole, up to
// and including its closing quote, whatever pairs it holds: a redacted value
// never takes an opening quote while it leaves the quote that closes it in
// the text, where that quote would open a string that the next quote
// closes, so that the text between them, as the credential in
// password=f("a token=b"); DB_PASSWORD=pre" c d", would be read as being
// outside of any quoted value. A single quote that quotedValueRegex finds no
// closing quote for is closed by the quote of a later \' on its line, as in
// DB_PASSWORD=pre' SECRET\'; echo ok (see shellQuotedSegment).
func joinedQuoteEnd(sh *shellAssignScanner, s string, i, end, open int, quoted quoteKinds) (valueEnd, next int) {
	return joinedQuoteEndWithEnclosing(sh, s, i, end, open, quoted, 0, false)
}

func joinedQuoteEndWithEnclosing(sh *shellAssignScanner, s string, i, end, open int, quoted, innerQuoted quoteKinds, subSQ bool) (valueEnd, next int) {
	allQuoted := quoted | innerQuoted
	if subSQ {
		allQuoted |= 2
	}
	quote := openingQuote(s, i)
	if quote == "" || s[i-1] == ':' || s[i-1] == '=' || s[i-1] == '\\' || closesString(s, i, quote, allQuoted) {
		return end, i
	}
	q := findQuotedOrShellSegment(sh, s, i)
	quote2 := openingQuote(s, i)
	if q != "" && strings.HasSuffix(quote2, "'") && !spansLine(sh, s, i, len(q)) && !quotedValueEnds(s[i:], len(q), allQuoted) {
		if nextEnd := interiorApostropheEnd(sh, s, i, i+len(q), quote2, allQuoted); nextEnd > i+len(q) {
			if strings.HasSuffix(s[i:nextEnd], quote2) {
				q = s[i:nextEnd]
			} else if prev, _ := utf8.DecodeLastRuneInString(s[:i]); isCamelTokenByte(s[i-1]) || unicode.IsLetter(prev) || unicode.IsNumber(prev) {
				// The quote is joined to the words on both sides of
				// it, as the first one of
				// DB_PASSWORD=pre'it's correct SECRET is, and no later
				// quote closes it (see interiorApostropheEnd): the
				// value runs to nextEnd and is redacted whole.
				return nextEnd, nextEnd
			}
		}
	}
	inner, _ := unquoteValue(q)
	last, _ := utf8.DecodeLastRuneInString(inner)
	joins := q != "" && !isDelimSpace(last) && closingQuoteJoins(sh, s, i, q, false)
	// A quote that nothing closes and that a line break follows in a shell
	// assignment (see inShellAssignment), as the one of
	// DB_PASSWORD=pre'<LF>SECRET does, opens a part that runs on over the line
	// break, as a shell reads it, rather than reads as a trailing apostrophe,
	// as the one of password=it'<LF>next does.
	breaksLine := q == "" && i+len(quote) < len(s) && (s[i+len(quote)] == '\n' || s[i+len(quote)] == '\r') && inShellAssignment(sh, s, i)
	if strings.HasSuffix(quote, "'") {
		if i+len(quote) < end {
			if startsContraction(s, i, len(quote)) &&
				(q == "" && unclosedContractionIsProse(sh, s, i, len(quote)) ||
					q != "" && !quotedValueEnds(s[i:], len(q), allQuoted) && openingQuote(s, i+len(q)) == "" && (!joins || multilineContractionOpensPair(sh, s, i, q, allQuoted))) {
				return end, i
			}
		} else if q == "" && !breaksLine || q != "" && !closingQuoteJoins(sh, s, i, q, false) && !quotedValueEnds(s[i:], len(q), allQuoted) && openingQuote(s, i+len(q)) == "" {
			return end, i
		}
	}
	if i+len(quote) >= end && !breaksLine && (skipHorizDelimSpace(s, i+len(quote)) == i+len(quote) || spansLine(sh, s, i, len(q))) &&
		(q == "" || !quotedValueEnds(s[i:], len(q), allQuoted) && openingQuote(s, i+len(q)) == "" && (!strings.HasSuffix(quote, "'") && !spansLine(sh, s, i, len(q)) || !joins)) {
		return end, i
	}
	if q == "" {
		// An apostrophe that nothing closes and that starts the ending of a
		// contraction or a possessive (see startsContraction), as the one of
		// abc's in password=abc's prompt_tokens=10 next_page_token=c1 does,
		// ends with its word when the next word on the line starts a pair
		// that ends a value (see startsRedactedPair), rather than opening a
		// part that runs on over that pair: the value is redacted up to the
		// pair, and the pair is read on its own.
		if strings.HasSuffix(quote, "'") && i+len(quote) < end && !breaksLine && startsContraction(s, i, len(quote)) {
			if k := skipHorizDelimSpace(s, end); k > end && startsRedactedPair(s, k, quoted) {
				return end, i
			}
		}
		if unclosedQuoteOpens(s, i, quote, true, allQuoted) {
			if clamped := unclosedEnclosingEnd(s, i+len(quote), open, quote, quoted, innerQuoted); clamped < len(s) {
				return clamped, i
			}
			if quote == `\"` && quoted&1 != 0 {
				bs := 0
				for k := i + len(quote); k < len(s); k++ {
					if s[k] == '\\' {
						bs++
						continue
					}
					if s[k] == '"' && bs%2 == 0 {
						return max(end, k+1), i
					}
					bs = 0
				}
			}
			return len(s), i
		}
		return end, i
	}
	if i+len(q) <= end {
		return end, i + len(q)
	}
	closing := i + len(q) - len(quote)
	if inner, _ := unquoteValue(q); opensLaterConstruct(s, i, len(q), len(quote), inner, quoted) || opensQuotedPair(s, closing, len(quote), inner) {
		return end, i
	}
	if !quotedValueEnds(s[i:], len(q), allQuoted) || opensDelimitedValueWithEnclosing(sh, s, i+len(q), open, quoted, innerQuoted) {
		return runOnValueEndWithEnclosing(sh, s, i, i+len(q), open, quoted, innerQuoted, subSQ), i
	}
	return i + len(q), i
}

// multilineContractionOpensPair reports whether the closed multiline
// single-quoted segment q at s[i:], whose opening quote starts a contraction
// (see startsContraction), spans a single line break into a new line whose
// text before the closing quote ends at a valid boundary in an identifier key
// and its delimiter and whose closing quote opens a single-line quoted value
// on that line (without stranding an unpaired quote of q's kind on the rest of
// that line), as in "password=it's\nuser='bob'". In a shell assignment (see
// inShellAssignment) the segment is a quoted part of the value, as a shell
// reads it, as in export DB_PASSWORD=pre's\nfoo='bar' violetfern's.
func multilineContractionOpensPair(sh *shellAssignScanner, s string, i int, q string, quoted quoteKinds) bool {
	quote := openingQuote(s, i)
	quoteLen := len(quote)
	if quoteLen == 0 || len(q) < 2*quoteLen || !spansLine(sh, s, i, len(q)) || !startsContraction(s, i, quoteLen) || inShellAssignment(sh, s, i) {
		return false
	}
	inner := q[quoteLen : len(q)-quoteLen]
	lastNL := strings.LastIndexAny(inner, "\r\n")
	if lastNL < 0 {
		return false
	}
	beforeNL := strings.TrimSuffix(inner[:lastNL], "\r")
	if strings.ContainsAny(beforeNL, "\r\n") {
		return false
	}
	cLen := contractionLen(inner)
	if cLen == 0 {
		cLen = 1
	}
	if cLen > len(beforeNL) || strings.TrimLeft(strings.TrimRightFunc(beforeNL[cLen:], isDelimSpace), ".,;:!?)]}") != "" {
		return false
	}
	line := inner[lastNL+1:]
	k := trailingKeyStart(line, isIdentifierKey)
	if k < 0 {
		return false
	}
	closing := i + len(q) - quoteLen
	prefix := line[:k]
	if strings.HasSuffix(prefix, `"`) || strings.HasSuffix(prefix, `'`) {
		prefix = strings.TrimSuffix(prefix[:len(prefix)-1], `\`)
	}
	if strings.TrimLeft(strings.TrimFunc(prefix, isDelimSpace), ",;{([") != "" &&
		!opensLaterPairAfterSpace(line, s[closing+quoteLen:]) {
		return false
	}
	nextQ := findQuotedValue(s[closing:])
	if nextQ == "" || strings.ContainsAny(nextQ, "\r\n") || !quotedValueEnds(s[closing:], len(nextQ), quoted) {
		return false
	}
	remStart := closing + len(nextQ)
	lineEnd := len(s)
	if nl := strings.IndexAny(s[remStart:], "\r\n"); nl >= 0 {
		lineEnd = remStart + nl
	}
	return !hasUnpairedQuoteOnLine(s[remStart:lineEnd], quote)
}

// hasUnpairedQuoteOnLine reports whether line, the rest of a line after a
// quoted value, leaves a quote of quote's kind open when it is read as shell
// text that starts outside any string. A quote of that kind opens or closes a
// string, unless, outside one, it is the apostrophe of a contraction or a
// possessive in a prose word (see isProseContraction), as the ' of it's or
// don't is. Outside such a string, a string of the other kind that closes on
// the line is passed over whole, so that the ' of "it's" is not counted, and
// a backslash escapes the byte after it; inside one, every other byte is
// literal, as the " of '"' is. Two readings that are left in doubt are
// reported as an unpaired quote too: a quote of the other kind that does not
// close on the line, as the last " of SECRET'"1'2" is, and, for "'", a \'
// outside a string, which a shell reads as an escaped quote there but as the
// end of a single-quoted string inside one.
func hasUnpairedQuoteOnLine(line, quote string) bool {
	open := false
	for k := 0; k < len(line); {
		if open && line[k] == '\\' && k+1 < len(line) {
			if quote == `"` || (quote == `\"` && line[k+1] == '\\') {
				k += 2
				continue
			}
		}
		q := openingQuote(line, k)
		if q == quote {
			if !open && quote == "'" && isProseContraction(line, k) {
				k += len(q)
				continue
			}
			open = !open
			k += len(q)
			continue
		}
		if !open {
			switch {
			case quote == "'" && q == `\'`:
				return true
			case q != "" && len(q) == len(quote):
				m := findQuotedValue(line[k:])
				if m == "" {
					return true
				}
				k += len(m)
				continue
			case line[k] == '\\' && k+1 < len(line):
				k += 2
				continue
			}
		}
		k++
	}
	return open
}

// isProseContraction reports whether the single quote at line[k] is the
// apostrophe of a contraction or a possessive in a prose word, as the ' of
// it's, don't, I'm or Bob's is, rather than a shell quote: it stands between
// two letters (see isMidWordApostrophe) and starts the ending of a
// contraction (see startsContraction), no quote follows that ending, as one
// does in a's' SECRET', nor a shell operator (";", "&" or "|", after optional
// whitespace), as the ; of DB_PASSWORD=pre' violetfern's; echo ok does, which
// ends a shell word rather than a prose one, and the word before it, back to
// the start of line,
// delimiter whitespace or one of "([{,;", is made of letters only, unlike
// the SECRET_it of SECRET_it's or the key=val of key=val's, and is not a
// name of more than one letter in upper case, as the SECRET of SECRET's is.
func isProseContraction(line string, k int) bool {
	if k == 0 || !isMidWordApostrophe(line, k) || !startsContraction(line, k, 1) {
		return false
	}
	after := k + 1 + max(contractionLen(line[k+1:]), 1)
	if openingQuote(line, after) != "" {
		return false
	}
	if j := skipHorizDelimSpace(line, after); j < len(line) && strings.IndexByte(";&|", line[j]) >= 0 {
		return false
	}
	start, upper := k, true
	for start > 0 {
		r, size := utf8.DecodeLastRuneInString(line[:start])
		if !unicode.IsLetter(r) {
			if !isDelimSpace(r) && !strings.ContainsRune("([{,;", r) {
				return false
			}
			break
		}
		upper = upper && unicode.IsUpper(r)
		start -= size
	}
	return !upper || utf8.RuneCountInString(line[start:k]) == 1
}

// maxShellLineScan is how much of the rest of a line findQuotedOrShellSegment
// reads to tell a shell's reading of a single-quoted part from
// quotedValueRegex's, so that telling them apart costs at most that much for
// each part, however long the line is.
const maxShellLineScan = 4096

// findQuotedOrShellSegment returns the quoted part that the quote at s[i:]
// opens in a word: the quoted value that quotedValueRegex reads, or, for a
// single quote, the part that a shell reads (see shellQuotedSegment), which
// ends at the "'" of a \' that quotedValueRegex reads as an escaped quote.
// The shell's part is taken if quotedValueRegex finds no closing quote, and,
// if it finds one later on the same line, when the rest of the line leaves
// no quote open after the shell's part but leaves one open after
// quotedValueRegex's value (see hasUnpairedQuoteOnLine), as a valid shell
// line does, where pairing the escaped quote with a later one would leave
// the quotes after it paired wrongly, as those around SECRET2 in
// DB_PASSWORD=pre' SECRET\'; API_KEY=pre2' SECRET2' would be. A value whose
// escaped quote the rest of the line reads as a Python or JavaScript string
// does, as the b'secret\'b' of token=b'secret\'b'; echo 'ok' does, keeps
// quotedValueRegex's reading, and so does one whose line runs on for more
// than maxShellLineScan bytes after the shell's part, or that runs past its
// line. sh, if bound to s, keeps the scans of quotedValueAt and the line
// breaks of s (see lineBreakFrom), so that the quotes of a word that one
// scan reads, which the word's reader asks about in turn, are not each read
// again to the end of the value.
func findQuotedOrShellSegment(sh *shellAssignScanner, s string, i int) string {
	m := quotedValueAt(sh, s, i)
	seg := shellQuotedSegment(s, i)
	switch {
	case m == "":
		return seg
	case seg == "" || len(seg) >= len(m) || spansLine(sh, s, i, len(m)):
		return m
	}
	from := i + len(seg)
	end := min(len(s), from+maxShellLineScan)
	if n := strings.IndexAny(s[from:end], "\r\n"); n >= 0 {
		end = from + n
	} else if end < len(s) && s[end] != '\n' && s[end] != '\r' {
		return m
	}
	if !hasUnpairedQuoteOnLine(s[from:end], "'") && hasUnpairedQuoteOnLine(s[i+len(m):end], "'") {
		return seg
	}
	return m
}

// shellQuotedSegment returns the single-quoted part that the "'" at s[i:]
// opens when a shell reads it, if quotedValueRegex, which reads \' as an
// escaped quote, finds no closing quote for it: in a shell, a backslash
// between single quotes is a literal byte, so the "'" of \' closes the part,
// as in DB_PASSWORD=pre' SECRET\'; echo ok. It returns "" unless the first
// "'" after s[i] on its line is the "'" of a \'.
func shellQuotedSegment(s string, i int) string {
	if i >= len(s) || s[i] != '\'' {
		return ""
	}
	j := strings.IndexAny(s[i+1:], "'\r\n")
	if j <= 0 || s[i+1+j] != '\'' || s[i+j] != '\\' {
		return ""
	}
	return s[i : i+2+j]
}

// opensQuotedPair reports whether the quote at s[closing:], of length
// quoteLen, which would close a quoted part whose text between the quotes is
// inner, more likely opens a quoted pair, as the second quote of
// password=a"b x "token=SECRETP-token: " in config does: whitespace ends
// inner, so that the quote stands where a string opens, a pair whose key is
// sensitive starts right after it (see allowedKeyAt), and a quote of its kind
// closes what it opens on its line.
func opensQuotedPair(s string, closing, quoteLen int, inner string) bool {
	last, _ := utf8.DecodeLastRuneInString(inner)
	if inner == "" || !isDelimSpace(last) || !allowedKeyAt(s, closing+quoteLen) {
		return false
	}
	m := findQuotedValue(s[closing:])
	return m != "" && !strings.ContainsAny(m, "\r\n")
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
	m := findQuotedValue(s[i:])
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
		leadTrim := strings.TrimLeftFunc(inner, isDelimSpace)
		lead := len(inner) - len(leadTrim)
		trimmed := strings.TrimRightFunc(leadTrim, isDelimSpace)
		kind, pair := nestedKeyName(trimmed, start+len(quote)+lead, lineStart)
		if kind != notNestedKey && len(trimmed) != len(inner) {
			return kind, start
		}
		if kind == notNestedKey && quotedKeyAllowed(inner) {
			// A quoted key that nestedKeyName does not read, as a "temp:"
			// key that holds "=", ":" or whitespace or has no name, or a
			// credential name after a state prefix and whitespace, as in
			// ' user: DB_PASSWORD ', is read from its opening quote (see
			// quotedPairUncheckedAtStart), as a compound name is.
			if lineStart {
				return nestedKey, start
			}
			return valueEndingInKeyName, start
		}
		return kind, pair
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
		// A padded key, or another that the text patterns do not read whole
		// (see scannedQuotedKey), is read from its opening quote.
		allowed := scannedQuotedKey(inner) && quotedKeyAllowed(inner)
		if allowed || len(strings.TrimFunc(inner, isDelimSpace)) != len(inner) {
			if !allowed {
				return notNestedKey, 0
			}
			if lineStart {
				return nestedKey, start
			}
			return valueEndingInKeyName, start
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
	if qLoc := quotedKeyAtStart(s[i:]); qLoc != nil {
		return quotedKeyAllowed(s[i+qLoc[2] : i+qLoc[3]])
	}
	if !mayStartTextKeyName(s[i:]) {
		return false
	}
	loc := textKeyNameAtRe.FindStringSubmatchIndex(s[i:])
	return loc != nil && textKeyAllowedAt(s, i+loc[2], i+loc[3]) && hasTextValueStart(s[i+loc[1]:])
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
		if isWholeTextKeyName(name[d:]) {
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
	return nestedPairEndWithEnclosing(nil, s, pair, valueEnd, open, quoted, 0)
}

// nestedPairEndWith is nestedPairEnd, with sh answering inShellAssignment
// for s (a fresh scanner is used if sh is nil).
func nestedPairEndWith(sh *shellAssignScanner, s string, pair, valueEnd, open int, quoted quoteKinds) int {
	return nestedPairEndWithEnclosing(sh, s, pair, valueEnd, open, quoted, 0)
}

func nestedPairEndWithEnclosing(sh *shellAssignScanner, s string, pair, valueEnd, open int, quoted, innerQuoted quoteKinds) int {
	// A value longer than maxNestedKeyLen is read as a key only while each
	// one ends further right than the one before it: keys apart from each
	// other, as in a: KEY: KEY: KEY="x y", are read once each, but a run of
	// them in one word, as in token:token:token:..., would be matched again
	// from each one of them, and is stopped at the second.
	prevEnd := -1
	for {
		loc := quotedPairUncheckedAtStart(s[pair:])
		if loc == nil {
			loc = textKeyValTokenAtRe.FindStringSubmatchIndex(s[pair:])
		}
		if loc == nil {
			// Not reached: nestsSensitiveKey has read a delimiter and a
			// value after a key name that textKeyValTokenAtRe matches. Were
			// the name not matched, the value that holds it would still be
			// redacted to its end, so that no part of it is left in place.
			return max(pair, valueEnd)
		}
		start, end := pair+loc[8], pair+loc[9]
		if end-start > maxNestedKeyLen && end <= prevEnd {
			return oversizedValueEndWithEnclosing(sh, s, start, end, open, quoted, innerQuoted)
		}
		prevEnd = end
		kind, next := nestsSensitiveKey(s, start, end, startsLine(s, pair+loc[7], start), true, quoted)
		if kind == notNestedKey {
			return textValueEndWithEnclosing(sh, s, start, end, open, quoted, innerQuoted, false, false, nil)
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
func oversizedValueEndWithEnclosing(sh *shellAssignScanner, s string, start, end, open int, quoted, innerQuoted quoteKinds) int {
	closed := sensitiveValueEndWith(sh, s, start, start)
	if closed > end {
		return closed
	}
	if opened := openedValueEndWithEnclosing(sh, s, closed, end, open, true, quoted, innerQuoted, false, false); opened > end {
		return opened
	}
	v := s[start:end]
	at := end
	if !strings.HasSuffix(v, ":") && !strings.HasSuffix(v, "=") {
		at = skipDelimSpace(s, end)
		if at == len(s) || (s[at] != ':' && s[at] != '=') {
			return textValueEndWithEnclosing(sh, s, start, end, open, quoted, innerQuoted, false, false, nil)
		}
		at++
	}
	at = skipDelimSpace(s, at)
	rest := s[at:]
	if rest != "" && (rest[0] == '{' || rest[0] == '[') {
		return sensitiveValueEndWith(sh, s, at, at)
	}
	enclosing := quoted | innerQuoted
	if q := openingQuote(s, at); q != "" {
		if closesString(s, at, q, enclosing) {
			return textValueEndWithEnclosing(sh, s, start, end, open, quoted, innerQuoted, false, false, nil)
		}
		if qv := quotedValueRe.FindString(rest); qv != "" {
			if !quotedValueEnds(rest, len(qv), enclosing) || opensDelimitedValueWithEnclosing(sh, s, at+len(qv), open, quoted, innerQuoted) {
				return runOnValueEndWithEnclosing(sh, s, at, at+len(qv), open, quoted, innerQuoted, false)
			}
			if valueStartsAt(s, at, true, enclosing) {
				return at + len(qv)
			}
			return textValueEndWithEnclosing(sh, s, start, end, open, quoted, innerQuoted, false, false, nil)
		}
		if unclosedQuoteOpens(s, at, q, true, enclosing) {
			return unclosedEnclosingEnd(s, at+len(q), open, q, quoted, innerQuoted)
		}
		return textValueEndWithEnclosing(sh, s, start, end, open, quoted, innerQuoted, false, false, nil)
	}
	limit := unclosedEnclosingEnd(s, at, open, "", quoted, innerQuoted)
	if n := strings.IndexAny(rest, regexpSpace+",;&"); n >= 0 {
		return min(at+n, limit)
	}
	return limit
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

// hasTextValueStart reports whether s, the text after a key-value delimiter,
// starts with a non-whitespace, non-delimiter byte, as textValueStartRe does.
// It reads at most a few bytes rather than compiling a regex match.
func hasTextValueStart(s string) bool {
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '\t', '\n', '\f', '\r', ' ':
			continue
		case ',', ';', '&':
			return false
		default:
			return true
		}
	}
	return false
}

// findQuotedValue returns the quoted-value match that quotedValueRe.FindString
// returns for s, read by quotedValueScan without the regexp.
func findQuotedValue(s string) string {
	if end, _ := quotedValueScan(s, 0); end > 0 {
		return s[:end]
	}
	return ""
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
		q := findQuotedValue(s[at:])
		if q != "" {
			inner, _ := unquoteValue(q)
			if holdsValue(inner) {
				return textBetweenStrings(nil, s, at, at+len(q), quoted) == ""
			}
			after := s[at+len(q):]
			return after != "" && !followsValueBoundary(after, quoted) && openingQuote(after, 0) == ""
		}
		return unclosedQuoteOpens(s, at, quote, mayClose, quoted)
	}
	rest := s[at:]
	// The bare value runs on to the first of regexpSpace, ",", ";" and "&";
	// it holds a value if a rune before that is not whitespace or closing
	// punctuation (see holdsValue). The runes are read up to the first that
	// tells, not to the end of the value, which may be the rest of a long
	// word, so that a check of each pair in a run of pairs, as in
	// password=[REDACTED]token=[REDACTED]..., reads a few bytes of it.
	for _, r := range rest {
		if r < utf8.RuneSelf && strings.IndexByte(regexpSpace+",;&", byte(r)) >= 0 {
			return false
		}
		if !isDelimSpace(r) && !strings.ContainsRune(closingPunctuation, r) {
			return true
		}
	}
	return false
}

// quotedValueEnds reports whether the quoted value s[:n] at the start of s
// ends there: s ends after it, or its closing quote does not pair with a
// later quote (see pairsForward) and is followed either by a genuine value
// boundary (see followsValueBoundary) or by a quote, of any kind, that closes
// an enclosing string in quoted (see endsString) or starts a sensitive quoted
// key (see startsQuotedKey), as the "token" of the JSON text
// "a":"b""token":"d" does. Otherwise its quotes close one string and open the
// next rather than enclose a value, as do the quotes around " for user " in
// the text "password=abc-token: " for user "bob"", or the value runs into a
// joined string, a quoted key that is not sensitive or a punctuation-led
// continuation, as in 'my'"'"'tail', "a""x": "b" or "prefix"+tail.
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
	return endsString(after, 0, quote, quoted) || startsQuotedKey(after, quoted)
}

// followsValueBoundary reports whether after, the non-empty text immediately
// following a closing quote when openingQuote(after, 0) == "", begins at a
// delimiter that terminates the quoted value: delimiter whitespace, a pair or
// clause separator (",", ";" or "&"), closing brackets (")", "]" or "}")
// whose last "}" or "]" an opening "{" or "[" of a glued JSON object or array
// follows at once (see opensGluedStructure), as between the glued objects
// {"a": "b"}{"c": "d"} or arrays ["a"]["b"], closing brackets followed by
// escaped line breaks or tabs (\n, \r or \t) and then end-of-text, delimiter
// whitespace or such a glued object or array, or
// closing brackets and at most one sentence/clause punctuation mark (".",
// "!", "?" or ":") followed by end-of-text, delimiter whitespace, ",", ";",
// "&" or an outer closing quote in quoted. Punctuation continuations such as
// "+secret", "-secret", "/secret", ".secret", "@secret", "%secret",
// "!secret", "=secret", ":secret" or "\secret" return false so runOnValueEnd
// consumes them, and so do brackets glued to any other word, as the
// "}{secret" of x='Bearer abc'}{secret is.
func followsValueBoundary(after string, quoted quoteKinds) bool {
	if skipDelimSpace(after, 0) > 0 || after[0] == ',' || after[0] == ';' || after[0] == '&' {
		return true
	}
	i := 0
	for i < len(after) && (after[i] == ')' || after[i] == ']' || after[i] == '}') {
		i++
	}
	if i > 0 && i < len(after) && (after[i] == '{' || after[i] == '[') && (after[i-1] == '}' || after[i-1] == ']') && opensGluedStructure(after, i) {
		return true
	}
	// Escaped line breaks and tabs after the brackets, as the \n between the
	// objects of the JSON lines "{\"a\": \"b\"}\n{\"c\": \"d\"}" quoted into a
	// string is, stand for the whitespace that they escape: they end the
	// value before the end of the text, whitespace, or a glued JSON object or
	// array as above.
	if i > 0 {
		j := i
		for j+1 < len(after) && after[j] == '\\' && strings.IndexByte("nrt", after[j+1]) >= 0 {
			j += 2
		}
		if j > i && (j == len(after) || skipDelimSpace(after, j) > j || (after[j] == '{' || after[j] == '[') && (after[i-1] == '}' || after[i-1] == ']') && opensGluedStructure(after, j)) {
			return true
		}
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

// maxGluedKeyScan is how much of the text after the "{" or "[" of a glued
// JSON object or array opensGluedStructure reads for its first key or
// element, so that the check costs at most that much for each pair of glued
// brackets, however long the text after them is.
const maxGluedKeyScan = 256

// opensGluedStructure reports whether the "{" or "[" at s[i], which follows a
// closing "}" or "]" at once, opens a JSON object or array glued to the one
// that closed, as the second object of {"a": "b"}{"c": "d"} or array of
// ["a"]["b"] does: after delimiter whitespace, and after any further opening
// brackets in an array, a quoted key and ":" for an object, or a quoted
// element and "," or "]" for an array, all in the maxGluedKeyScan bytes after
// s[i] and with the quoted text on one line. Any other text glued to the
// brackets, as the rest of the word x='Bearer abc'}{rest or
// x='Bearer abc'}{'rest' is, does not open one, and is read as the rest of
// the value.
func opensGluedStructure(s string, i int) bool {
	t := s[:min(len(s), i+1+maxGluedKeyScan)]
	open := t[i]
	j := skipDelimSpace(t, i+1)
	for open == '[' && j < len(t) && (t[j] == '[' || t[j] == '{') {
		open = t[j]
		j = skipDelimSpace(t, j+1)
	}
	q := openingQuote(t, j)
	if q == "" {
		return false
	}
	k := j + len(q)
	for k < len(t) && !strings.HasPrefix(t[k:], q) {
		if t[k] == '\n' || t[k] == '\r' {
			return false
		}
		if len(q) == 1 && t[k] == '\\' {
			k++
		}
		k++
	}
	if k >= len(t) {
		return false
	}
	k = skipDelimSpace(t, k+len(q))
	if k >= len(t) {
		return false
	}
	if open == '{' {
		return t[k] == ':'
	}
	return t[k] == ',' || t[k] == ']'
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
// runs into (see runOnValueEnd). sh, if bound to s, keeps the scans of
// quotedValueAt and the brackets of s (see bracketEnd), so that a second
// string or an object that runs on to the end of s is not read to its end
// again for each value that it follows.
func textBetweenStrings(sh *shellAssignScanner, s string, start, end int, quoted quoteKinds) string {
	inner, quote := unquoteValue(s[start:end])
	if quote == "" || !readsAsProse(inner) || !pairsForward(s[start:], end-start) {
		return ""
	}
	if quoted&quoteKind(quote) == 0 && start > 0 && (s[start-1] == ':' || s[start-1] == '=') {
		return ""
	}
	last, _ := utf8.DecodeLastRuneInString(inner)
	laterPair := !isDelimSpace(last) && opensLaterPairAfterSpace(inner, s[end:])
	if !isDelimSpace(last) && !laterPair {
		return ""
	}
	secStart := end - len(quote)
	secQuoted := quoted
	second := quotedValueAt(sh, s, secStart)
	if len(second) <= 2*len(quote) && laterPair {
		secStart = end
		secQuoted = quoted | quoteKind(quote)
		second = quotedValueAt(sh, s, secStart)
	}
	if len(second) <= 2*len(quote) {
		return ""
	}
	if secInner, _ := unquoteValue(second); !laterPair && !hasLowerProseWord(secInner) {
		return ""
	}
	if !quotedSecondEnds(s[secStart:], len(second), secQuoted) {
		if laterPair {
			if bStart := secStart + len(quote); bStart < len(s) && (s[bStart] == '{' || s[bStart] == '[') {
				if bEnd, closed := bracketEnd(sh, s, bStart); closed &&
					strings.HasPrefix(s[bEnd:], quote) &&
					quotedSecondEnds(s[secStart:], bEnd+len(quote)-secStart, secQuoted) {
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

// quotedSecondEnds reports whether the second quoted string s[:n] in
// textBetweenStrings ends at s[n:] without recursing into startsQuotedKey.
func quotedSecondEnds(s string, n int, quoted quoteKinds) bool {
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
	return endsString(after, 0, quote, quoted)
}

// opensLaterValue reports whether the closing quote of a quoted value, whose
// text between the quotes is inner and which after follows, opens the value
// of a later key instead: inner ends in a sensitive key and its delimiter
// (see endsInSensitiveKeyDelimiter), and a value starts after the quote (see
// valueFollows). In got "token": " in config, api_key="   secret", the quotes
// around " in config, api_key=" enclose no value; the second opens the value
// of api_key, whatever it starts with.
func opensLaterValue(inner, after string) bool {
	if !valueFollows(after) || !endsInSensitiveKeyDelimiter(inner) {
		return false
	}
	if skipDelimSpace(after, 0) == 0 && openingQuote(after, 0) == "" && strings.IndexFunc(inner, isDelimSpace) >= 0 && !readsAsProse(inner) {
		return false
	}
	return true
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
		v = strings.TrimRightFunc(v, isDelimSpace)
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
// of a later key: inner ends in a key name and its delimiter (see
// endsInKeyDelimiter and isIdentifierKey), and a value starts after delimiter
// whitespace in after (see valueFollows). In token=abc:'s label=' x', the
// closing quote of 's label=' opens the value of label, as delimiter
// whitespace follows it. Text joined to the closing quote at once, with no
// whitespace between them, is a further part of the same word instead, as a
// shell joins 's note label=' and x in DB_PASSWORD=a:'s note label='x or
// DB_PASSWORD=pre'correct secret token='suffix into one value, so the closing
// quote does not open a later pair then.
func opensLaterPair(inner, after string) bool {
	return skipDelimSpace(after, 0) > 0 && valueFollows(after) && endsInKeyDelimiter(inner, isIdentifierKey)
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
func opensLaterConstruct(s string, at, qLen, quoteLen int, inner string, quoted quoteKinds) bool {
	after := s[at+qLen:]
	if opensLaterPair(inner, after) || opensLaterBearer(inner, after) {
		return true
	}
	closing := at + qLen - quoteLen
	if at > 0 && s[at-1] == ':' && quoteLen == 1 && s[at] == '\'' {
		if n := contractionLen(inner); n > 0 && opensLaterPairAfterSpace(inner[n:], after) && endsInSensitiveKeyDelimiter(inner) {
			if nextQ := findQuotedValue(s[closing:]); nextQ != "" && !strings.ContainsAny(nextQ, "\r\n") && quotedValueEnds(s[closing:], len(nextQ), quoted) {
				return true
			}
		}
	}
	last, _ := utf8.DecodeLastRuneInString(inner)
	if strings.IndexFunc(inner, isDelimSpace) >= 0 && (isDelimSpace(last) || endsInKeyDelimiter(inner, isIdentifierKey)) && startsQuotedBearerAt(s[closing:], quoteLen, quoted) {
		return true
	}
	return (inner == "" || isDelimSpace(last) || strings.ContainsRune(",;([", last)) && startsQuotedKeyWithValue(s[closing:], quoted)
}

// startsQuotedBearerAt reports whether s (starting at a closing quote of length
// quoteLen) is a single-line quoted Bearer credential such as 'Bearer secret'.
func startsQuotedBearerAt(s string, quoteLen int, quoted quoteKinds) bool {
	if len(s) <= quoteLen+len("bearer") {
		return false
	}
	after := s[quoteLen:]
	if !strings.EqualFold(after[:len("bearer")], "bearer") {
		return false
	}
	sp := skipHorizDelimSpace(after, len("bearer"))
	if sp == len("bearer") || !valueStartsAt(after, sp, true, quoted) {
		return false
	}
	q := quotedValueRe.FindString(s)
	return q != "" && !strings.ContainsAny(q, "\r\n")
}

// opensLaterBearer reports whether inner ends with the word "Bearer" followed
// by delimiter whitespace and a separated value starts after delimiter
// whitespace in after, as in token=abc:'s note Bearer ' secret'. An
// immediately joined fragment after the closing quote (without leading
// delimiter whitespace in after), as in
// export DB_PASSWORD=a:'s correct; horse Bearer '"secret", stays part of the
// credential rather than opening an independent Bearer token.
func opensLaterBearer(inner, after string) bool {
	if skipDelimSpace(after, 0) == 0 || !valueFollows(after) {
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
// key followed by a ":" or "=" delimiter and a value, as in 'password': ' x',
// or with a quoted key that quotedKeyAllowed accepts, as nextQuotedKeyName
// reads it, as in ' user: DB_PASSWORD ': ' x' or 'temp:a=b': ' x'.
func startsQuotedKeyWithValue(s string, quoted quoteKinds) bool {
	q := quotedValueRe.FindString(s)
	if q == "" {
		return false
	}
	raw, _ := unquoteValue(q)
	key := strings.TrimFunc(raw, isDelimSpace)
	if key == "" || len(key) > maxNestedKeyLen {
		return false
	}
	if !quotedKeyAllowed(raw) {
		if !isIdentifierKey(key) {
			return false
		}
		for i := 0; i < len(key); i++ {
			if !isCamelTokenByte(key[i]) && key[i] != ':' {
				return false
			}
		}
	}
	delim := skipDelimSpace(s, len(q))
	if delim >= len(s) || (s[delim] != ':' && s[delim] != '=') || (s[delim] == '=' && delim+1 < len(s) && s[delim+1] == '=') {
		return false
	}
	return valueStartsAt(s, delim+1, true, quoted)
}

// closesOuterQuote reports whether s[cEnd:end] begins with a quote closing an
// active enclosing outer string in quoted (see endsString), optionally
// followed by closing brackets for brackets opened before the pair (up to
// open) and sentence punctuation. Requiring an enclosing outer string to close
// at cEnd (as in failed: "token=abc:'s"; the user's request) ensures that a
// "}" or "]" inside a candidate quoted credential (as in
// command: { export DB_PASSWORD=a:'s} secret'; }) is never mistaken for an
// outer bracket terminator before quote ownership is established.
func closesOuterQuote(s string, cEnd, end, open int, quoted quoteKinds) bool {
	if cEnd >= end || quoted == 0 {
		return false
	}
	q := openingQuote(s, cEnd)
	if q == "" || !endsString(s, cEnd, q, quoted) {
		return false
	}
	pos := cEnd + len(q)
	remOpen := open
	for pos < end && remOpen > 0 && (s[pos] == '}' || s[pos] == ']') {
		pos++
		remOpen--
	}
	for pos < end && strings.IndexByte(".,;:!?)", s[pos]) >= 0 {
		pos++
	}
	return pos == end
}

// laterConstructFollowsContraction reports whether after, the text immediately
// following an unclosed ':'<contraction>, leads on the same line (through
// optional ",", ";" delimiters and lowercase prose words) to an independent
// sensitive credential construct (a Bearer credential, a quoted key-value
// pair, or a sensitive key-value pair), as in
// token=abc:'s note Bearer [REDACTED] or token=abc:'s "password":"[REDACTED]".
// Plain words or semicolons without a following sensitive construct, as in
// password=a:'s correct horse battery or password=a:'s correct; horse battery,
// stay inside the unclosed credential span.
func laterConstructFollowsContraction(after string, quoted quoteKinds) bool {
	i := 0
	for i < len(after) && (after[i] == ';' || after[i] == ',') {
		i++
	}
	at := skipHorizDelimSpace(after, i)
	if at == 0 {
		return false
	}
	line := after[at:]
	if nl := strings.IndexAny(line, "\r\n"); nl >= 0 {
		line = line[:nl]
	}
	line = strings.TrimRightFunc(line, isDelimSpace)
	for pos := 0; pos < len(line); {
		rest := line[pos:]
		if startsLaterConstructAt(rest, quoted) {
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
		tok := strings.TrimRight(line[pos:end], ";,")
		if !isLowerASCIIWord(tok) {
			return false
		}
		pos = skipHorizDelimSpace(line, end)
	}
	return false
}

// startsLaterConstructAt reports whether s begins with a Bearer credential, a
// quoted key-value pair, or a sensitive key-value pair.
func startsLaterConstructAt(s string, quoted quoteKinds) bool {
	if len(s) > len("bearer") && strings.EqualFold(s[:len("bearer")], "bearer") {
		if sp := skipHorizDelimSpace(s, len("bearer")); sp > len("bearer") && valueStartsAt(s, sp, true, quoted) {
			tail := s[sp:]
			if len(tail) < 2 || !strings.EqualFold(tail[:2], "of") || (len(tail) > 2 && skipDelimSpace(tail, 2) == 2) {
				return true
			}
		}
	}
	if startsQuotedKeyWithValue(s, quoted) {
		return true
	}
	kind, _ := afterClosedQuotePair(nil, s, 0, quoted)
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

// startsQuotedKey reports whether s starts with a sensitive or preserved
// quoted key and its value: a single-line quoted key token (see
// quotedKeyAtStart) that quotedKeyAllowed, isPreservedPageTokenName or
// isTokenCounterName accepts, followed by ":" or "=" (not "==") and a valid
// value that starts at once (see valueStartsAt and startsDecimalCounterValue).
// A quoted key that is neither sensitive nor a preserved token counter/cursor,
// as the "x" of "a""x": "b", or one with no value after its delimiter, as in
// 'temp:x':; echo ok, does not end a value that it is joined to (see
// quotedValueEnds and startsRedactedPair).
func startsQuotedKey(s string, quoted quoteKinds) bool {
	loc := quotedKeyAtStart(s)
	if loc == nil || (s[loc[6]] == '=' && loc[7] < len(s) && s[loc[7]] == '=') {
		return false
	}
	raw := s[loc[2]:loc[3]]
	if quotedKeyAllowed(raw) {
		return valueStartsAt(s, loc[7], true, quoted)
	}
	trimmed := strings.TrimFunc(raw, isDelimSpace)
	if isPreservedPageTokenName(trimmed) {
		return valueStartsAt(s, loc[7], true, quoted)
	}
	if isTokenCounterName(trimmed) {
		return startsDecimalCounterValue(s, loc[7], quoted)
	}
	return false
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
func forwardValueEndWithEnclosing(sh *shellAssignScanner, s string, start, end, open int, quoted, innerQuoted quoteKinds) int {
	enclosing := quoted | innerQuoted
	for {
		inner, quote := unquoteValue(s[start:end])
		if quote == "" || !opensLaterValue(inner, s[end:]) {
			if quote != "" && (!quotedValueEnds(s[start:], end-start, enclosing) || opensDelimitedValueWithEnclosing(sh, s, end, open, quoted, innerQuoted)) {
				return runOnValueEndWithEnclosing(sh, s, start, end, open, quoted, innerQuoted, false)
			}
			return end
		}
		after := skipDelimSpace(s, end)
		if after > end || openingQuote(s, after) != "" {
			if q2 := openingQuote(s, after); q2 != "" {
				m := quotedValueRe.FindString(s[after:])
				if m == "" {
					if unclosedQuoteOpens(s, after, q2, true, enclosing) {
						return unclosedEnclosingEnd(s, after+len(q2), open, q2, quoted, innerQuoted)
					}
					return end
				}
				start, end = after, after+len(m)
				continue
			}
			if after < len(s) && (s[after] == '{' || s[after] == '[') {
				return sensitiveValueEndWith(sh, s, after, after)
			}
			if readsAsProse(inner) && opensLaterPairAfterSpace(inner, s[end:]) {
				qOpen := end - len(quote)
				if m := quotedValueRe.FindString(s[qOpen:]); m != "" && quotedValueEnds(s[qOpen:], len(m), enclosing) {
					return qOpen + len(m)
				}
			}
			bareEnd := after
			for bareEnd < len(s) && strings.IndexByte(regexpSpace+",;&", s[bareEnd]) < 0 {
				bareEnd++
			}
			if kind, pair := nestsSensitiveKey(s, after, bareEnd, startsLine(s, end, after), true, quoted); kind != notNestedKey {
				return nestedPairEndWithEnclosing(sh, s, pair, bareEnd, open, quoted, innerQuoted)
			}
			return textValueEndWithEnclosing(sh, s, after, bareEnd, open, quoted, innerQuoted, false, false, nil)
		}
		qOpen := end - len(quote)
		m := quotedValueRe.FindString(s[qOpen:])
		if m == "" {
			return unclosedEnclosingEnd(s, qOpen+len(quote), open, quote, quoted, innerQuoted)
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
	pos     int // s[:pos] has been read or skipped
	active  quoteKinds
	openPos int // byte offset where active became non-zero
	bs      int // consecutive backslashes immediately preceding s[pos]
	// inner caches the reading of innerAt, and ends the searches of
	// endOnLine, one for each set of quotes that it is asked for (see
	// endScanSlot). They let redactBearerToken read a string once however
	// many bearer tokens it holds, and a line once however many strings
	// that do not end on it hold bearer tokens.
	inner  innerQuoteScan
	ends   [6]stringEndScan
	closes [6]quoteCloseCache
	// shell answers inShellAssignment for the text of the pass that owns
	// these quotes (see shellFor).
	shell           *shellAssignScanner
	jsonTop         *jsonContainerNode
	openJSONKind    jsonContainerKind
	subSQ           bool
	subEscDQ        bool
	beforeInnerDQ   bool
	parenDepth      int
	openInParen     bool
	noSubCloseMode  byte
	noSubCloseFrom  int
	noSubCloseUntil int
	substTop        *intListNode
	shellMemos      shellCloseMemo
	memoStr         string
	subSQEnds       [2]quoteCloseCache
	escDQ           escDQCache
}

// shellFor returns the scanner that answers inShellAssignment for s, the
// text of the pass that owns q, or a prefix of it. It is made on first use
// and belongs to that pass alone.
func (q *openQuotes) shellFor(s string) *shellAssignScanner {
	if q.shell == nil {
		q.shell = newShellAssignScanner(s)
	}
	q.shell.use(s)
	return q.shell
}

func (q *openQuotes) shellMemosFor(s string) *shellCloseMemo {
	if len(q.memoStr) != len(s) || q.memoStr != s {
		q.shellMemos = shellCloseMemo{}
		q.memoStr = s
	}
	return &q.shellMemos
}

// innerQuoteScan is where innerAt has read the string that opens at the
// offset open-1, and the escaped quotes that are open there.
type innerQuoteScan struct {
	open     int // 1 + the offset of the opening quote of the string, or 0
	pos      int // s[:pos] has been read
	bs       int // consecutive backslashes immediately preceding s[pos]
	kinds    quoteKinds
	jsonTop  *jsonContainerNode
	jsonKind jsonContainerKind
	subSQ    bool
	subEscDQ bool
	substTop *intListNode
}

// stringEndScan is the result of a search of endOnLine for the quotes in
// kinds, or the zero value: at, the first quote in s[from:lineEnd] that ends
// a string in kinds, or -1, where lineEnd is the end of the line if at is
// -1.
type stringEndScan struct {
	kinds   quoteKinds
	from    int
	at      int
	lineEnd int
}

// quoteCloseCache records the first quote-close pair (at, after) for
// openQuotes.closeOnLine in s[from:lineEnd], so repeated scans along a line
// take O(N) total time.
type quoteCloseCache struct {
	kinds   quoteKinds
	from    int
	at      int
	after   int
	lineEnd int
}

// innerAt returns the escaped quotes of the kind of the string open at i
// (see before), \" in a string that " opens or \' in one that ' opens, that
// open a string nested in it and are still open at i, as the \" before
// Bearer is in "{\"auth\": \"Bearer abc\"}", or 0. As in before, a quote
// after one backslash, or after five, is escaped, and one after three is a
// quote of a string nested in the nested one, which is not read. The string
// is read once from its opening quote, however many times innerAt is asked
// for offsets in it.
func (q *openQuotes) innerAt(s string, i int) quoteKinds {
	var c byte
	var kind quoteKinds
	switch q.active {
	case 1:
		c, kind = '"', 4
	case 2:
		c, kind = '\'', 8
	default:
		return 0
	}
	in := &q.inner
	if in.open != q.openPos+1 || in.pos > i {
		*in = innerQuoteScan{open: q.openPos + 1, pos: q.openPos + 1}
	}
	for ; in.pos < i; in.pos++ {
		for in.substTop != nil && in.pos >= in.substTop.val {
			in.substTop = in.substTop.prev
		}
		b := s[in.pos]
		if b == '\n' || b == '\r' {
			in.kinds, in.bs = 0, 0
			in.jsonTop = nil
			in.jsonKind = jsonContainerNone
			in.subSQ = false
			in.subEscDQ = false
			in.substTop = nil
			continue
		}
		if b == '\\' {
			in.bs++
			continue
		}
		bs := in.bs
		in.bs = 0
		if q.active == 1 && bs%2 == 0 {
			switch in.kinds {
			case 0:
				switch b {
				case '{', '[':
					in.jsonTop = &jsonContainerNode{
						entry: jsonContainerEntry{openChar: b, pos: in.pos, isJSON: isJSONContainerOpener(s, in.pos)},
						prev:  in.jsonTop,
					}
				case '}', ']':
					if in.jsonTop != nil && jsonContainerMatches(in.jsonTop.entry.openChar, b) {
						in.jsonTop = in.jsonTop.prev
					}
				}
			case 4:
				switch b {
				case '\'':
					if (in.jsonKind != jsonContainerNone && bs%8 == 4) || (q.openJSONKind != jsonContainerNone && bs%4 == 2) {
						continue
					}
					if in.subSQ {
						if isMidWordApostrophe(s, in.pos) && !opensPrefixedString(s, in.pos) {
							continue
						}
						in.subSQ = false
					} else if (!isMidWordApostrophe(s, in.pos) || opensPrefixedString(s, in.pos)) && !isTrailingApostrophe(s, in.pos) {
						in.subSQ = true
					}
				case '$':
					if in.pos+1 < len(s) && s[in.pos+1] == '(' {
						if endPos := shellScanClose(q.shellMemosFor(s), s, shellFrameCS, in.pos+2, false, true, true); endPos >= 0 {
							in.substTop = &intListNode{val: endPos, prev: in.substTop}
						}
					}
				case '`':
					if endPos := shellScanClose(q.shellMemosFor(s), s, shellFrameBT, in.pos+1, true, true, true); endPos >= 0 {
						in.substTop = &intListNode{val: endPos, prev: in.substTop}
					}
				}
			}
		}
		if b == c && bs%4 == 1 {
			if in.kinds == 0 {
				in.kinds = kind
				if kind == 4 && in.pos > q.openPos+1 {
					in.jsonKind = classifyQuoteJSONContainer(s, in.pos-1, q.openPos+1, true, in.jsonTop)
				} else {
					in.jsonKind = jsonContainerNone
				}
				in.subSQ = false
				in.subEscDQ = false
				in.substTop = nil
			} else {
				in.kinds = 0
				in.jsonKind = jsonContainerNone
				in.subSQ = false
				in.subEscDQ = false
				in.substTop = nil
			}
		} else if q.active == 1 && in.kinds == 4 && b == '"' && bs%4 == 3 && !in.subSQ {
			in.subEscDQ = !in.subEscDQ
		}
	}
	for in.substTop != nil && in.pos >= in.substTop.val {
		in.substTop = in.substTop.prev
	}
	return in.kinds
}

func (q *openQuotes) subSingleAt(s string, i int) bool {
	if q == nil || q.active != 1 {
		return false
	}
	if q.innerAt(s, i) == 4 {
		return q.inner.subSQ
	}
	return q.subSQ
}

func (q *openQuotes) subEscDoubleAt(s string, i int) bool {
	if q == nil || q.active != 1 {
		return false
	}
	if q.innerAt(s, i) == 4 {
		return q.inner.subEscDQ
	}
	return q.subEscDQ
}

// endOnLine returns the offset of the first quote in s[from:], before the
// end of its line, that ends a string in kinds (see endsString), which holds
// the quote of the string open at from (see before) and those of the strings
// nested in it that are open there (see innerAt), or -1 if none does. It
// reads s only up to that quote or the end of the line. Which quote that is
// depends on kinds and from only, not on the string, so a search is reused
// by a later call for the same kinds that starts no earlier and no later
// than the quote that it found, or than the end of the line if it found
// none, so that a line is searched once however many bearer tokens its
// strings hold. A call never starts within an escaped quote, as from
// follows a quote, so it reads the quotes after from as that search did,
// after the backslashes before them (see quoteAfterBackslashes).
func (q *openQuotes) endOnLine(s string, from int, kinds quoteKinds) int {
	c := &q.ends[endScanSlot(kinds)]
	if c.kinds == kinds && c.from <= from && (c.at >= from || c.at < 0 && from <= c.lineEnd) {
		return c.at
	}
	*c = stringEndScan{kinds: kinds, from: from, at: -1, lineEnd: len(s)}
	bs := 0
	for k := from; k < len(s); k++ {
		switch s[k] {
		case '\n', '\r':
			c.lineEnd = k
			return c.at
		case '\\':
			bs++
			continue
		case '"', '\'':
			if quote, at := quoteAfterBackslashes(s, k, bs); kinds&quoteKind(quote) != 0 && endsString(s, at, quote, kinds) {
				c.at = at
				return c.at
			}
		}
		bs = 0
	}
	return c.at
}

func (q *openQuotes) subSQCloseOnLine(s string, slot, from, limit int) int {
	if from >= limit {
		return -1
	}
	c := &q.subSQEnds[slot]
	if c.from <= from && c.lineEnd == limit && (c.at >= from || c.at < 0) {
		return c.at
	}
	at := -1
	if rel := strings.IndexByte(s[from:limit], '\''); rel >= 0 {
		at = from + rel
	}
	*c = quoteCloseCache{from: from, at: at, after: at + 1, lineEnd: limit}
	return at
}

// closeOnLine returns (quoteStart, quoteEnd) for the first quote in s[from:]
// before the end of its line that closes a string in kinds, or (-1, -1) if
// none does. Unlike endOnLine, it returns the raw closing quote without
// requiring endsString punctuation so callers can inspect the exact
// post-closing-quote boundary, reading each line at most once per kind mask.
func (q *openQuotes) closeOnLine(s string, from int, kinds quoteKinds) (int, int) {
	if kinds == 0 {
		return -1, -1
	}
	c := &q.closes[endScanSlot(kinds)]
	if c.kinds == kinds && c.from <= from && (c.at >= from || c.at < 0 && from <= c.lineEnd) {
		if c.at < 0 {
			return -1, -1
		}
		return c.at, c.after
	}
	*c = quoteCloseCache{kinds: kinds, from: from, at: -1, after: -1, lineEnd: len(s)}
	bs := 0
	for k := from; k < len(s); k++ {
		switch s[k] {
		case '\n', '\r':
			c.lineEnd = k
			return -1, -1
		case '\\':
			bs++
			continue
		case '"':
			if bs%2 == 0 {
				if kinds&1 != 0 {
					c.at, c.after = k, k+1
					return c.at, c.after
				}
			} else if bs%4 == 1 {
				if kinds&4 != 0 {
					c.at, c.after = k-1, k+1
					return c.at, c.after
				}
			}
		case '\'':
			if kinds == 2 {
				c.at, c.after = k, k+1
				return c.at, c.after
			}
			if bs%2 == 0 {
				if kinds&2 != 0 {
					c.at, c.after = k, k+1
					return c.at, c.after
				}
			} else if bs%4 == 1 {
				if kinds&8 != 0 {
					c.at, c.after = k-1, k+1
					return c.at, c.after
				}
			}
		}
		bs = 0
	}
	return -1, -1
}

// endScanSlot returns the index in openQuotes.ends of the search for kinds,
// the quote of an open string, ", ', \" or \', alone or, for " or ', with
// the escaped one of a string nested in it (see innerAt).
func endScanSlot(kinds quoteKinds) int {
	switch kinds {
	case 1:
		return 0
	case 2:
		return 1
	case 4:
		return 2
	case 8:
		return 3
	case 1 | 4:
		return 4
	}
	return 5
}

type pairSpan struct {
	keyStart int
	valStart int
	valEnd   int
}

// sensitivePairSpans lazily records the non-overlapping value spans of
// sensitive key-value pairs in s in a single O(N) forward pass (mirroring
// redactTextKeyValue's bracket and quote tracking) and answers coverage queries
// in amortized O(1) as monotonically increasing Bearer match positions arrive.
type sensitivePairSpans struct {
	built  bool
	spans  []pairSpan
	cursor int
}

// covers reports whether an earlier sensitive key-value pair starts before
// openPos and has a value span covering [openPos, target+1), so
// redactBearerToken yields to keyValFinder rather than splitting a joined
// credential containing "Bearer".
func (p *sensitivePairSpans) covers(s string, openPos, target int) bool {
	if openPos < 0 || openPos >= target || target > len(s) {
		return false
	}
	if !p.built {
		p.built = true
		p.build(s)
	}
	for p.cursor < len(p.spans) && p.spans[p.cursor].valEnd <= target {
		p.cursor++
	}
	if p.cursor >= len(p.spans) {
		return false
	}
	sp := p.spans[p.cursor]
	return sp.keyStart < openPos && sp.valStart <= openPos && sp.valEnd > target
}

func (p *sensitivePairSpans) build(s string) {
	finder := keyValFinder{whole: s}
	var brackets openBrackets
	var quotes openQuotes
	sh := quotes.shellFor(s)
	for pos := 0; pos < len(s); {
		loc := finder.find(s[pos:])
		if loc == nil {
			break
		}
		for i, v := range loc {
			if v >= 0 {
				loc[i] = v + pos
			}
		}
		if loc[8] < 0 {
			pos = max(loc[7], loc[0]+1)
			continue
		}
		open := brackets.before(s, loc[0])
		if loc[4] == loc[5] && isCookieProse(s, s[loc[2]:loc[3]], s[loc[6]:loc[7]], loc[8], open > 0, &quotes) {
			pos = max(loc[7], loc[0]+1)
			continue
		}
		start, end := loc[8], loc[9]
		quoted := quotes.before(s, start)
		innerQuoted := quotes.innerAt(s, start)
		subSQ := quotes.subSingleAt(s, start)
		subEscDQ := quotes.subEscDoubleAt(s, start)
		if placeholderCut(s, start) == end {
			// As in redactTextKeyValue.
			if placeholderEnds(s, end, quoted, innerQuoted, subSQ, subEscDQ) || authHeaderPlaceholderEnds(s, loc, end, quoted, innerQuoted) {
				brackets.skip(end)
				quotes.skip(s, end)
				p.spans = append(p.spans, pairSpan{keyStart: loc[0], valStart: start, valEnd: end})
				pos = max(end, loc[0]+1)
				continue
			}
			end = bareValueEnd(s, end)
		}
		if loc[4] < loc[5] && (quoted&quoteKind(openingQuote(s, start)) != 0 || s[loc[5]:loc[7]] != "=") {
			if quote := textBetweenStrings(sh, s, start, end, quoted); quote != "" {
				next := quotes.unpaired(s, start, quote)
				brackets.skip(next)
				pos = max(next, loc[0]+1)
				continue
			}
		}
		switch kind, pair := nestsSensitiveKey(s, start, end, startsLine(s, loc[7], start), true, quoted); kind {
		case nestedKey:
			pos = max(start, loc[0]+1)
			continue
		case valueEndingInKeyName:
			end = nestedPairEndWith(sh, s, pair, end, open, quoted)
			brackets.skip(end)
			quotes.skip(s, end)
			p.spans = append(p.spans, pairSpan{keyStart: loc[0], valStart: start, valEnd: end})
			pos = max(end, loc[0]+1)
			continue
		}
		if redEnd := redactedPrefixEnd(s, start, end, quoted, innerQuoted, subSQ, subEscDQ); redEnd >= 0 {
			brackets.skip(redEnd)
			quotes.skip(s, redEnd)
			p.spans = append(p.spans, pairSpan{keyStart: loc[0], valStart: start, valEnd: redEnd})
			pos = max(redEnd, loc[0]+1)
			continue
		}
		end = textValueEndWithEnclosing(sh, s, start, end, open, quoted, innerQuoted, subSQ, subEscDQ, &quotes)
		end = trimTrailingSubshellOrJSONQuotes(s, start, end, quoted, innerQuoted, subSQ, subEscDQ)
		brackets.skip(end)
		if q := openingQuote(s, start); q != "" && end > start && closesString(s, start, q, quoted) {
			quotes.drop()
		}
		quotes.skip(s, end)
		if end > start {
			p.spans = append(p.spans, pairSpan{keyStart: loc[0], valStart: start, valEnd: end})
		}
		pos = max(end, loc[0]+1)
	}
}

// isMidWordApostrophe reports whether the single quote at s[pos] is an
// apostrophe between two letters, as in don't or l'été, rather than a quote.
func isMidWordApostrophe(s string, pos int) bool {
	before, _ := utf8.DecodeLastRuneInString(s[:pos])
	after, _ := utf8.DecodeRuneInString(s[pos+1:])
	return unicode.IsLetter(before) && unicode.IsLetter(after)
}

// unclosedContractionIsProse reports whether the single quote at s[i:], of
// length quoteLen, which starts the ending of a contraction (see
// startsContraction) and which no quote closes, reads as the apostrophe of a
// prose word rather than a quote that opens a part of a value: the word
// before it is one that the ending makes a contraction of (see
// contractionStemIsProse), it does not stand in a shell assignment (see
// inShellAssignment), and the ending ends the text, is followed by a byte
// other than whitespace, or is followed by horizontal whitespace and a
// lower-case letter, as in password=doesn't work and Bearer user's request.
// Followed by a line break, as in DB_PASSWORD=pre's<LF>SECRET, or by a word
// that does not start with a lower-case letter, as in
// DB_PASSWORD=pre'd SECRET_1, after a word that the ending does not make a
// contraction of, as in x="Bearer abc'd secret", or in a shell assignment, as
// in export DB_PASSWORD=pre'd secret, the quote is left open over the rest of
// the value, as a shell leaves it.
func unclosedContractionIsProse(sh *shellAssignScanner, s string, i, quoteLen int) bool {
	n := contractionLen(s[i+quoteLen:])
	after := i + quoteLen + max(n, 1)
	if after < len(s) && (s[after] == '\n' || s[after] == '\r') {
		return false
	}
	if !contractionStemIsProse(s, i, quoteLen, n) || inShellAssignment(sh, s, i) {
		return false
	}
	if after >= len(s) {
		return true
	}
	j := skipHorizDelimSpace(s, after)
	if j == after || j >= len(s) {
		return true
	}
	r, _ := utf8.DecodeRuneInString(s[j:])
	return unicode.IsLower(r)
}

// maxContractionStem is the length of the longest word in the contraction
// stem lists that contractionStemIsProse reads.
const maxContractionStem = 8

// sContractionStems are the words that 's makes a contraction of (of is,
// has or us), as in it's, that's and let's.
var sContractionStems = []string{"he", "here", "how", "it", "let", "one", "she", "that", "there", "this", "what", "when", "where", "who", "why"}

// contractionStemIsProse reports whether the word before the quote at s[i:],
// of length quoteLen, whose ending of n bytes (see contractionLen), or the t
// of n't if n is 0, follows the quote, is one that the ending makes a
// contraction of in prose: any word before 's, which may be a possessive, as
// in Bearer user's request, and before the other endings only the pronouns
// and verbs that take them, as in I'd, we'll, they're, you've, I'm and
// doesn't, unlike the pre of pre'd. A word that an "=" joins to a key before
// it, as pre is in password=pre's x's, is the start of the key's value rather
// than a possessive, so before 's it reads as prose only if 's makes a
// contraction of it (see sContractionStems), as in the password=it's Bob's;
// a word of letters too long to read back over does not.
func contractionStemIsProse(s string, i, quoteLen, n int) bool {
	ending := strings.ToLower(s[min(len(s), i+quoteLen):min(len(s), i+quoteLen+max(n, 1))])
	if ending == "s" {
		start := i
		for start > 0 {
			r, size := utf8.DecodeLastRuneInString(s[:start])
			if !unicode.IsLetter(r) {
				break
			}
			if i-start >= maxShellWordScan {
				return false
			}
			start -= size
		}
		if start == 0 {
			return true
		}
		if s[start-1] == '=' {
			return slices.Contains(sContractionStems, strings.ToLower(s[start:i]))
		}
		tokStart := start
		for tokStart > 0 {
			r, size := utf8.DecodeLastRuneInString(s[:tokStart])
			if isDelimSpace(r) || strings.ContainsRune(",;([{\"'=", r) {
				break
			}
			if i-tokStart >= maxShellWordScan {
				return false
			}
			tokStart -= size
		}
		return tokStart == 0 || s[tokStart-1] != '='
	}
	start := i
	for start > 0 && i-start <= maxContractionStem+1 {
		c := s[start-1]
		if !('a' <= c|0x20 && c|0x20 <= 'z') {
			break
		}
		start--
	}
	if i-start > maxContractionStem+1 {
		return false
	}
	stem := strings.ToLower(s[start:i])
	var stems []string
	switch ending {
	case "t":
		if !strings.HasSuffix(stem, "n") {
			return false
		}
		stem = stem[:len(stem)-1]
		stems = []string{"ai", "are", "ca", "could", "dare", "did", "do", "does", "had", "has", "have", "is", "may", "might", "must", "need", "ought", "sha", "should", "was", "were", "wo", "would"}
	case "d", "ll":
		stems = []string{"he", "here", "how", "i", "it", "she", "that", "there", "they", "this", "we", "what", "when", "where", "who", "why", "you"}
	case "re":
		stems = []string{"here", "how", "there", "these", "they", "those", "we", "what", "when", "where", "who", "why", "you"}
	case "ve":
		stems = []string{"could", "i", "might", "must", "should", "they", "we", "what", "where", "who", "would", "you"}
	case "m":
		stems = []string{"i"}
	}
	return slices.Contains(stems, stem)
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
	for cEnd < len(s) && (s[cEnd] == '}' || s[cEnd] == ']') {
		cEnd++
	}
	if cEnd < len(s) {
		r, _ := utf8.DecodeRuneInString(s[cEnd:])
		if !isDelimSpace(r) && !strings.ContainsRune(",;&", r) {
			return false
		}
	}
	q := quotedValueRe.FindString(s[pos:])
	if q == "" || strings.ContainsAny(q, "\r\n") {
		return laterConstructFollowsContraction(s[cEnd:], 0)
	}
	inner, _ := unquoteValue(q)
	return opensLaterConstruct(s, pos, len(q), 1, inner, 0)
}

// opensPrefixedString reports whether the quote at s[pos], after one or two
// letters, opens a string after a prefix rather than being an apostrophe or a
// word-trailing closing quote: the letters before it are a Python string
// prefix that starts a word (b, r, u, f, br, rb, fr or rf, in either case), as
// in b'...', r"...", f'...' or rb'...', or the letter of a short command-line
// option, a "-" and a letter at the start of s or after whitespace, as in
// curl -H'X-Custom: 1' or curl -H"X-Custom: 1". A single quote before the
// ending of a contraction or a possessive, as in -I'm or the B's of plan B's,
// is an apostrophe all the same (see isContraction).
func opensPrefixedString(s string, pos int) bool {
	if pos < len(s) && s[pos] == '\'' && isContraction(s[pos+1:]) {
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
	return contractionEndingLen(after, continuesWordOrKey)
}

// contractionEndingLen returns the byte length of the English contraction or
// possessive ending at the start of after (s, m, d, ll, re or ve, in either
// case) when the rune after it is not one that continues reports, or 0.
func contractionEndingLen(after string, continues func(rune) bool) int {
	for _, ending := range []string{"ll", "re", "ve", "s", "m", "d"} {
		if len(after) >= len(ending) && strings.EqualFold(after[:len(ending)], ending) {
			r, _ := utf8.DecodeRuneInString(after[len(ending):])
			if !continues(r) {
				return len(ending)
			}
		}
	}
	return 0
}

// continuesWordOrKey reports whether r, after the ending of a contraction,
// makes that ending the start of a word or a key name instead: a letter, a
// digit, "_" or "=".
func continuesWordOrKey(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '='
}

// isContraction reports whether after, the text after an apostrophe, starts
// with the ending of an English contraction or possessive, as the m of I'm
// and the s of B's do: s, m, d, ll, re or ve, in either case, followed by
// neither a letter, a digit, "_" nor "=", any of which would make it the
// start of a word or a key name instead, as in f'd={x}'.
func isContraction(after string) bool {
	return contractionLen(after) > 0
}

// startsContraction reports whether the single quote at s[i:], of length
// quoteLen, starts the ending of a contraction or a possessive, as the quotes
// of it's, we'll, I'm and don't do (see contractionLen), rather than opens a
// quoted part.
func startsContraction(s string, i, quoteLen int) bool {
	return startsContractionBefore(s, i, quoteLen, continuesWordOrKey)
}

// startsContractionBefore reports whether the single quote at s[i:], of
// length quoteLen, starts the ending of a contraction or a possessive, as
// startsContraction reads it, when the rune after that ending is not one that
// continues reports.
func startsContractionBefore(s string, i, quoteLen int, continues func(rune) bool) bool {
	after := s[i+quoteLen:]
	if contractionEndingLen(after, continues) > 0 {
		return true
	}
	if i == 0 || (s[i-1] != 'n' && s[i-1] != 'N') || after == "" || (after[0] != 't' && after[0] != 'T') {
		return false
	}
	r, _ := utf8.DecodeRuneInString(after[1:])
	return !continues(r)
}

// closingQuoteJoins reports whether the closing quote of the quoted part q at
// s[i:] joins the text after it to the part, as a shell joins them, rather
// than stands for an apostrophe or opens the next string: a single quote is
// not read as the apostrophe of a contraction or a possessive in a prose word
// (see closingQuoteIsApostrophe), as the one of wasn't is in the
// password=users' wasn't changed and the ones of violetfern's, SECRET_it's
// and SECRET_1's in DB_PASSWORD=pre' violetfern's,
// DB_PASSWORD=pre' SECRET_it's and DB_PASSWORD=pre' SECRET_1's are not, and
// an escaped quote does not start the ending of a contraction (see
// startsContraction); and if the text between the quotes ends in whitespace,
// which leaves the quote where a string may open, as the third quote of
// password=abc' OR '1'='1 stands, no quote on the same clause of its line
// closes a string opened there, as in DB_PASSWORD=pre' correct horse 'suffix;
// echo ok. In a word that a closed quoted part has already joined (compound,
// see runOnWord), as the one of DB_PASSWORD="a"pre' violetfern's, a single
// quote that closes a part is never read as an apostrophe.
func closingQuoteJoins(sh *shellAssignScanner, s string, i int, q string, compound bool) bool {
	quoteLen := len(openingQuote(s, i))
	if len(q) < 2*quoteLen {
		return false
	}
	inner := q[quoteLen : len(q)-quoteLen]
	closing := i + len(q) - quoteLen
	if inner == "" || startsContraction(s, closing, quoteLen) && (quoteLen != 1 || !compound && closingQuoteIsApostrophe(sh, s, i, closing)) {
		return false
	}
	if last, _ := utf8.DecodeLastRuneInString(inner); isDelimSpace(last) {
		m := findQuotedValue(s[closing:])
		return m == "" || strings.ContainsFunc(m, isDelimSpace) || strings.ContainsAny(m, "\r\n;,&|") || closing+len(m) >= len(s) || s[closing+len(m)] != '='
	}
	return true
}

// closingQuoteIsApostrophe reports whether the single quote at s[closing],
// which would close the part that the single quote at s[i] opens, reads as
// the apostrophe of a contraction or a possessive in a prose word (see
// isProseContraction) rather than as the quote that closes the part. The
// words around it do not prove that on their own, as violetfern's in
// DB_PASSWORD=pre' violetfern's shows, where a shell closes the part there:
// the part does not stand in a shell assignment (see inShellAssignment), and
// either its opening quote starts the ending of a contraction too, of a word
// that the ending makes one of (see contractionStemIsProse), as in
// password=it's Bob's and the password=it's administrator's but not
// password=pre's x's, or it is the apostrophe of a plural possessive, as the
// one of users' is, and prose follows the ending on its line (see
// proseFollowsEnding), as in the password=users' wasn't changed. How long the
// word before the closing quote is does not count, so that a long name, as in
// the password=it's Bartholomewsmith's, reads as prose as a short one does.
func closingQuoteIsApostrophe(sh *shellAssignScanner, s string, i, closing int) bool {
	if !isProseContraction(s, closing) || inShellAssignment(sh, s, i) {
		return false
	}
	if startsContraction(s, i, 1) {
		return contractionStemIsProse(s, i, 1, contractionLen(s[i+1:]))
	}
	return i > 0 && (s[i-1] == 's' || s[i-1] == 'S') && isTrailingApostrophe(s, i) && proseFollowsEnding(s, closing)
}

// proseFollowsEnding reports whether prose follows, on its line, the ending
// of the contraction or possessive whose apostrophe is at s[k]: a full stop,
// a comma, "!", "?" or ":" right after it, or horizontal whitespace and a
// lower-case letter, as the changed after wasn't does.
func proseFollowsEnding(s string, k int) bool {
	after := k + 1 + max(contractionLen(s[k+1:]), 1)
	if after >= len(s) {
		return false
	}
	if strings.IndexByte(".,!?:", s[after]) >= 0 {
		return true
	}
	j := skipHorizDelimSpace(s, after)
	if j == after || j >= len(s) {
		return false
	}
	r, _ := utf8.DecodeRuneInString(s[j:])
	return unicode.IsLower(r)
}

// maxShellWordScan bounds how much contractionStemIsProse reads back from a
// quote.
const maxShellWordScan = 256

// inShellAssignment reports whether the quote at s[i] stands in the value of
// a shell variable assignment, as those of DB_PASSWORD=pre' violetfern's,
// db_password=pre's x, export db_password=pre'd x, export DB_PASSWORD="a b"pre's
// and export "DB_PASSWORD"=pre's do, where a shell reads a single quote as a
// quote, whatever the words around it look like. The shell word that holds
// the quote (with closed quoted segments and backslash escapes grouped inside
// it, back to unquoted whitespace or one of ";&|(`") holds a variable name
// (which may be quoted, as in "DB_PASSWORD"=) and an "=", and either a name
// before an "=" in it is an environment or snake_case variable name (see
// isEnvVarName), or the word starts with its name and follows export, env,
// declare, local, readonly or typeset on its logical line, after options and
// other assignments, as in export A=1 password=x. sh reads s, or a text that
// s is a prefix of, and carries its scan from one call to the next (see
// shellAssignScanner.use); a fresh scanner is used if sh is nil.
func inShellAssignment(sh *shellAssignScanner, s string, i int) bool {
	if i <= 0 || i > len(s) {
		return false
	}
	p := i - 1
	for p >= 0 {
		c := s[p]
		if c == '=' || c == '\'' || c == '"' || c == '\\' {
			break
		}
		if isShellWordBreak(c) && (p == 0 || s[p-1] != '\\') && !(c == '\n' && p >= 2 && s[p-1] == '\r' && s[p-2] == '\\') {
			return false
		}
		p--
	}
	if p < 0 {
		return false
	}
	if sh == nil {
		sh = newShellAssignScanner(s)
	}
	sh.use(s)
	return sh.at(i)
}

// newShellAssignScanner returns a scanner that reads s for inShellAssignment.
// Each pass of a scrub owns the scanners it uses: a scrub asks about many
// quotes of its text, each in a call of its own, and the scanner goes on from
// the cursors and quote records of the calls before, as reading each line
// again from its start for every quote would make the scrub quadratic in the
// length of a line. No scanner is shared between scrubs, so concurrent scrubs
// neither contend for nor evict each other's scanners, and none of them keeps
// the text once the pass that owns it returns.
func newShellAssignScanner(s string) *shellAssignScanner {
	sc := &shellAssignScanner{}
	sc.reset(s)
	return sc
}

// use points sc at s, unless sc already reads s or a text that s is a prefix
// of, as when redactBearerToken reads a token in t = s[:at]: such a prefix
// shares the bytes of the text, so the comparison returns at once, and the
// scan goes on as the one of the whole text.
func (sc *shellAssignScanner) use(s string) {
	if len(s) <= len(sc.s) && sc.s[:len(s)] == s {
		return
	}
	sc.reset(s)
}

type shellQuoteMemo struct {
	openPos, limit, closeEnd int
	// noCloseFrom, noCloseStop and noCloseLimit record that a scan from the
	// quote at noCloseFrom, under a limit of noCloseLimit, found no closing
	// quote before noCloseStop. A scan from an opening quote of the same kind
	// at or after noCloseFrom and before noCloseStop finds none either under
	// a limit up to noCloseLimit, because it reads the same bytes, or fewer.
	// Under a wider limit it can read on to a closing quote, so it scans
	// again.
	noCloseFrom, noCloseStop, noCloseLimit int
}

type shellScanLevel struct {
	pos, limit, quoteEnd int
	wordStart            int
	wordSawEq            bool
	wordLeading          bool
	wordHasEnv           bool
	wordFirstByte        byte
	runValid             bool
	runAtWordStart       bool
	runFirstDigit        bool
	runLetter            bool
	runLower             bool
	runUnderOrDigit      bool
	inDecl               bool
}

func (lv *shellScanLevel) resetWord() {
	lv.wordStart = -1
	lv.wordSawEq = false
	lv.wordLeading = false
	lv.wordHasEnv = false
	lv.wordFirstByte = 0
	lv.runValid = false
}

func (lv *shellScanLevel) finishWord(s string) {
	if lv.wordStart < 0 {
		return
	}
	if !lv.wordLeading && lv.wordFirstByte != '-' {
		if !lv.wordSawEq && (lv.pos-lv.wordStart <= 64 || strings.IndexByte(s[lv.wordStart:lv.pos], '\\') >= 0) {
			w := stripShellLineSplices(s[lv.wordStart:lv.pos])
			switch {
			case w == "" || w[0] == '-':
			case isShellDeclKeyword(w):
				lv.inDecl = true
			default:
				lv.inDecl = false
			}
		} else {
			lv.inDecl = false
		}
	}
	lv.resetWord()
}

type shellCursor struct {
	curPos int
	depth  int
	levels [8]shellScanLevel
}

// shellAssignScanner reads one text, s, for the pass that owns it. Besides
// the scan of inShellAssignment, it keeps what the pass has read of s for
// other questions that the pass asks again and again about places along s,
// each of which could otherwise read on to the end of s: so that a pass
// stays linear in the length of s, each answer is read once and then looked
// up. Each of these records holds only for s itself, not for a text that s
// is a prefix of, so a helper uses one only when sh.s is s (see
// boundTo).
type shellAssignScanner struct {
	s       string
	cursors [8]shellCursor
	quotes  [4]shellQuoteMemo
	// word is the run of bytes other than whitespace, ",", ";" and "&"
	// that bareWordEnd read last in s, s[word[0]:word[1]], which ends at
	// such a byte or at the end of s.
	word [2]int
	// values holds the last scan of quotedValueAt for each kind of quote
	// (see quoteKindIndex).
	values [4]quotedValueMemo
	// breaks holds the offsets of the line breaks of s, in order, once
	// breaksRead is set (see lineBreakFrom).
	breaks     []int
	breaksRead bool
	// brackets is the table of bracketSkips for s, or nil until bracketEnd
	// first needs it.
	brackets []int32
	// delimited holds what delimitedChainEnd worked out for each offset and
	// set of open quotes (see delimitedKey).
	delimited map[int]delimitedEnd
	// noApostrophe records, for "'" and "\'" (slots 0..1 when !explicitOpen,
	// slots 2..3 when explicitOpen), the last range [from, to, quoted) in
	// which interiorApostropheEnd found no closing quote.
	noApostrophe [4][3]int
	// ends memoizes stringEndIn per quoteKinds slot across calls on the same
	// line (see stringEndOnLine).
	ends [6]stringEndScan
}

func (sc *shellAssignScanner) reset(s string) {
	sc.s = s
	for idx := range sc.cursors {
		sc.cursors[idx] = shellCursor{
			curPos: 0,
			depth:  0,
			levels: [8]shellScanLevel{{pos: 0, limit: len(s), quoteEnd: len(s), wordStart: -1}},
		}
	}
	sc.quotes = [4]shellQuoteMemo{}
	sc.word = [2]int{-1, -1}
	sc.values = [4]quotedValueMemo{}
	sc.breaks, sc.breaksRead = sc.breaks[:0], false
	sc.brackets = nil
	sc.delimited = nil
	sc.noApostrophe = [4][3]int{{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}}
	sc.ends = [6]stringEndScan{}
}

// boundTo reports whether sc is not nil and reads s itself, so that what it
// keeps of its text holds for s.
func (sc *shellAssignScanner) boundTo(s string) bool {
	return sc != nil && len(sc.s) == len(s) && sc.s == s
}

// stringEndOnLine returns stringEndIn(s, from, lineBreakFrom(sc, s, from), quoted),
// memoizing the scan per quoteKinds slot on sc so repeated calls on the same
// line do not rescan to the end of the line.
func (sc *shellAssignScanner) stringEndOnLine(s string, from int, quoted quoteKinds) int {
	lineEnd := lineBreakFrom(sc, s, from)
	if quoted == 0 {
		return lineEnd
	}
	if !sc.boundTo(s) {
		return stringEndIn(s, from, lineEnd, quoted)
	}
	c := &sc.ends[endScanSlot(quoted)]
	if c.kinds == quoted && c.from <= from && (c.at >= from || c.at < 0 && from <= c.lineEnd) {
		if c.at >= 0 {
			return c.at
		}
		return c.lineEnd
	}
	end := stringEndIn(s, from, lineEnd, quoted)
	at := end
	if end == lineEnd {
		at = -1
	}
	*c = stringEndScan{kinds: quoted, from: from, lineEnd: lineEnd, at: at}
	return end
}

// bareWordEnd returns where the run of bytes other than whitespace, ",", ";"
// and "&" that runs on through s[i:] ends: at the first such byte, or at the
// end of s. sh, if not nil and bound to s, keeps the run it read last, so
// that the pairs of one long word, which nestedPairEndWith reads in turn as
// in 'temp:a='temp:a=..., read the word once rather than each to its end.
func bareWordEnd(sh *shellAssignScanner, s string, i int) int {
	bound := sh.boundTo(s)
	if bound && sh.word[0] <= i && i <= sh.word[1] {
		return sh.word[1]
	}
	end := len(s)
	if n := strings.IndexAny(s[i:], regexpSpace+",;&"); n >= 0 {
		end = i + n
	}
	if bound {
		sh.word = [2]int{i, end}
	}
	return end
}

// quotedValueMemo records a scan of quotedValueScan from the quote at open:
// it read s[open:stop] as a part of the value and found the value
// s[open:end], or none if end is negative.
type quotedValueMemo struct {
	open, end, stop int
}

// quoteKindIndex returns the index of the quote q, ", ', \" or \', in
// shellAssignScanner.values.
func quoteKindIndex(q string) int {
	idx := 0
	if q[len(q)-1] == '\'' {
		idx = 1
	}
	if len(q) == 2 {
		idx += 2
	}
	return idx
}

// quotedValueScan reads the quoted value that the quote at s[i:] opens as
// quotedValueRe reads it, without the regexp: it returns end, just past the
// closing quote of the value, or -1 if quotedValueRe matches nothing at
// s[i:], and stop, the offset at which the scan ended: the closing quote, or
// the byte (or the end of s) at which the value can go on no further. Every
// byte of s[i:stop] is read as the opening quote, a byte of the value or a
// part of an escape. quotedValueRegex reads each byte one way only, so the
// scan never goes back.
func quotedValueScan(s string, i int) (end, stop int) {
	q := openingQuote(s, i)
	switch len(q) {
	case 1:
		c := q[0]
		for j := i + 1; j < len(s); j++ {
			switch s[j] {
			case c:
				return j + 1, j
			case '\\':
				// A backslash escapes the byte after it, a line break
				// too.
				if j+1 == len(s) {
					return -1, j
				}
				j++
			}
		}
		return -1, len(s)
	case 2:
		c := q[1]
		j := i + 2
		for j < len(s) {
			if b := s[j]; b == c {
				// Only \q closes the value; a bare quote ends it unclosed.
				return -1, j
			} else if b != '\\' {
				j++
				continue
			}
			if j+1 == len(s) {
				return -1, j
			}
			switch s[j+1] {
			case c:
				return j + 2, j
			case '\\':
			default:
				j += 2
				continue
			}
			// A doubled backslash starts an escape of the inner text:
			// \\\\, \\\q, or \\ and a byte other than a backslash and q
			// (see escapedQuotedRegex).
			if j+2 == len(s) || s[j+2] == c {
				return -1, j
			}
			if s[j+2] != '\\' {
				j += 3
				continue
			}
			if j+3 == len(s) || (s[j+3] != '\\' && s[j+3] != c) {
				return -1, j
			}
			j += 4
		}
		return -1, len(s)
	}
	return -1, i
}

// quotedValueAt returns the quoted value that the quote at s[i:] opens, as
// quotedValueRe reads it (see quotedValueScan), or "". sh, if bound to s
// (see shellAssignScanner.boundTo), keeps the last scan for each kind of
// quote, so that a quote of the same kind that the scan read as a byte of
// the value, as each \' of a word that repeats x\\\' is, is not read again
// to where the value ends: such a quote is the escaped byte of an escape
// (one that stood where the value could close would have closed it), so a
// scan from it reads on in step with the scan before, from the byte after
// it, and ends where that scan did.
func quotedValueAt(sh *shellAssignScanner, s string, i int) string {
	q := openingQuote(s, i)
	if q == "" {
		return ""
	}
	var memo *quotedValueMemo
	if sh.boundTo(s) {
		memo = &sh.values[quoteKindIndex(q)]
		if memo.open <= i && i+len(q) <= memo.stop {
			if memo.end < 0 {
				return ""
			}
			return s[i:memo.end]
		}
	}
	end, stop := quotedValueScan(s, i)
	if memo != nil {
		*memo = quotedValueMemo{open: i, end: end, stop: stop}
	}
	if end < 0 {
		return ""
	}
	return s[i:end]
}

// lineBreakFrom returns the offset of the first line break ("\r" or "\n") in
// s at or after i, or len(s) if there is none. sh, if bound to s (see
// shellAssignScanner.boundTo), reads the line breaks of s once, so that
// asking whether each of many long quoted parts spans a line does not read
// each of them again.
func lineBreakFrom(sh *shellAssignScanner, s string, i int) int {
	if !sh.boundTo(s) {
		if n := strings.IndexAny(s[i:], "\r\n"); n >= 0 {
			return i + n
		}
		return len(s)
	}
	if !sh.breaksRead {
		sh.breaksRead = true
		for j := 0; j < len(s); {
			n := strings.IndexAny(s[j:], "\r\n")
			if n < 0 {
				break
			}
			sh.breaks = append(sh.breaks, j+n)
			j += n + 1
		}
	}
	if k, _ := slices.BinarySearch(sh.breaks, i); k < len(sh.breaks) {
		return sh.breaks[k]
	}
	return len(s)
}

// spansLine reports whether the part s[i:i+n] holds a line break (see
// lineBreakFrom).
func spansLine(sh *shellAssignScanner, s string, i, n int) bool {
	return lineBreakFrom(sh, s, i) < i+n
}

// bracketEnd returns what bracketValueEnd returns for the "{" or "[" at
// s[start]. sh, if bound to s (see shellAssignScanner.boundTo), reads the
// brackets and strings of s once (see bracketSkips), so that the objects
// and arrays that a pass asks about, each of which runs on to the end of s
// if nothing closes it, are not each read to their end.
func bracketEnd(sh *shellAssignScanner, s string, start int) (int, bool) {
	if c := s[start]; !sh.boundTo(s) || len(s) >= 1<<31-1 || c != '{' && c != '[' {
		return bracketValueEnd(s, start)
	}
	if sh.brackets == nil {
		sh.brackets = bracketSkips(s)
	}
	if m := sh.brackets[start+1]; m >= 0 {
		return int(m) + 1, true
	}
	return len(s), false
}

// bracketSkips returns a table, for each offset i of s and for len(s), of
// the offset of the first "}" or "]" that bracketValueEnd meets if it reads
// s from i outside of a string at a depth of 0, past the strings and
// the objects and arrays that open and close at or after i, or -1 if it
// meets none before the end of s. bracketValueEnd(s, i) for a "{" or "[" at
// s[i] thus ends just past the offset at i+1. It reads s once, from its end,
// and works out the offset at i from the ones after it: a "}" or "]" is its
// own; a "{" or "[" skips to the offset after the bracket that closes it;
// and a string, an escape or another byte skips to the offset after it. A
// string that a quote opens ends at the first quote of its kind that an
// even run of backslashes (none, too) precedes, as a backslash in it
// escapes the byte after it; one that an escaped quote, \" or \', opens
// ends at the first such quote that a run of 4n+1 backslashes precedes, the
// last of which escapes it, as the earlier ones pair into escapes of the
// inner text (see bracketValueEnd).
func bracketSkips(s string) []int32 {
	skips := make([]int32, len(s)+1)
	skips[len(s)] = -1
	quoteIdx := func(c byte) int {
		if c == '\'' {
			return 1
		}
		return 0
	}
	// plain holds the first quote of each kind after i that closes a string
	// that a quote opens; escapedAfter holds the first that closes one
	// that an escaped quote opens, at or after i+1, and escaped the same at
	// or after i+2, where such a string opened at i starts.
	plain := [2]int32{-1, -1}
	escaped, escapedAfter := plain, plain
	for i := len(s) - 1; i >= 0; i-- {
		switch c := s[i]; c {
		case '}', ']':
			skips[i] = int32(i)
		case '{', '[':
			skips[i] = -1
			if m := skips[i+1]; m >= 0 {
				skips[i] = skips[m+1]
			}
		default:
			// last is the offset of the last byte of the string, escape
			// or byte that starts at i, or -1 if s ends before it does.
			var last int32
			switch {
			case c == '"' || c == '\'':
				last = plain[quoteIdx(c)]
			case c == '\\' && i+1 < len(s) && (s[i+1] == '"' || s[i+1] == '\''):
				last = escaped[quoteIdx(s[i+1])]
			case c == '\\' && i+1 < len(s):
				last = int32(i + 1)
			default:
				last = int32(i)
			}
			skips[i] = -1
			if last >= 0 {
				skips[i] = skips[last+1]
			}
		}
		escaped = escapedAfter
		if c := s[i]; c == '"' || c == '\'' {
			run := 0
			for run < i && s[i-1-run] == '\\' {
				run++
			}
			if run%2 == 0 {
				plain[quoteIdx(c)] = int32(i)
			}
			if run%4 == 1 {
				escapedAfter[quoteIdx(c)] = int32(i)
			}
		}
	}
	return skips
}

func (sc *shellAssignScanner) closedQuoteLen(pos, limit int) int {
	s := sc.s
	if pos+1 >= limit {
		return 0
	}
	q := s[pos]
	memoIdx := 0
	if q == '"' {
		memoIdx = 1
	}
	m := &sc.quotes[memoIdx]
	if m.closeEnd > 0 && m.openPos == pos && m.limit == limit {
		return m.closeEnd - pos
	}
	if pos >= m.noCloseFrom && pos < m.noCloseStop && limit <= m.noCloseLimit {
		return 0
	}
	switch q {
	case '\'':
		for k := pos + 1; k < limit; k++ {
			c := s[k]
			if c == '\n' || c == '\r' {
				m.noCloseFrom, m.noCloseStop, m.noCloseLimit = pos, k, limit
				return 0
			}
			if c == '\'' {
				*m = shellQuoteMemo{openPos: pos, limit: limit, closeEnd: k + 1}
				return k + 1 - pos
			}
		}
		m.noCloseFrom, m.noCloseStop, m.noCloseLimit = pos, limit, limit
	case '"':
		for k := pos + 1; k < limit; k++ {
			c := s[k]
			if c == '\n' || c == '\r' {
				m.noCloseFrom, m.noCloseStop, m.noCloseLimit = pos, k, limit
				return 0
			}
			if c == '\\' {
				if k+1 >= limit {
					m.noCloseFrom, m.noCloseStop, m.noCloseLimit = pos, limit, limit
					return 0
				}
				if s[k+1] == '\r' {
					if k+2 < limit && s[k+2] == '\n' {
						k += 2
						continue
					}
					m.noCloseFrom, m.noCloseStop, m.noCloseLimit = pos, k+1, limit
					return 0
				}
				k++
				continue
			}
			if c == '"' {
				*m = shellQuoteMemo{openPos: pos, limit: limit, closeEnd: k + 1}
				return k + 1 - pos
			}
		}
		m.noCloseFrom, m.noCloseStop, m.noCloseLimit = pos, limit, limit
	}
	return 0
}

func (sc *shellAssignScanner) closedEscapedQuoteLen(pos, limit int) int {
	s := sc.s
	if pos+3 >= limit || s[pos] != '\\' {
		return 0
	}
	q := s[pos+1]
	memoIdx := 2
	if q == '"' {
		memoIdx = 3
	}
	m := &sc.quotes[memoIdx]
	if m.closeEnd > 0 && m.openPos == pos && m.limit == limit {
		return m.closeEnd - pos
	}
	if pos >= m.noCloseFrom && pos < m.noCloseStop && limit <= m.noCloseLimit {
		return 0
	}
	for k := pos + 2; k < limit; k++ {
		c := s[k]
		if c == '\n' || c == '\r' || c == '"' || (q == '\'' && c == '\'') {
			m.noCloseFrom, m.noCloseStop, m.noCloseLimit = pos, k, limit
			return 0
		}
		if c == '\\' {
			if k+1 >= limit {
				m.noCloseFrom, m.noCloseStop, m.noCloseLimit = pos, limit, limit
				return 0
			}
			if s[k+1] == q {
				*m = shellQuoteMemo{openPos: pos, limit: limit, closeEnd: k + 2}
				return k + 2 - pos
			}
			if s[k+1] == '\r' {
				if k+2 < limit && s[k+2] == '\n' {
					k += 2
					continue
				}
				m.noCloseFrom, m.noCloseStop, m.noCloseLimit = pos, k+1, limit
				return 0
			}
			if s[k+1] == '\\' && k+2 < limit && s[k+2] == q {
				k += 2
				continue
			}
			k++
		}
	}
	m.noCloseFrom, m.noCloseStop, m.noCloseLimit = pos, limit, limit
	return 0
}

func (sc *shellAssignScanner) selectCursor(i int) *shellCursor {
	var best *shellCursor
	for idx := range sc.cursors {
		c := &sc.cursors[idx]
		if c.curPos <= i {
			if best == nil || c.curPos >= best.curPos {
				best = c
			}
		}
	}
	if best != nil {
		return best
	}
	// All cursors are past i; reset the one farthest ahead.
	victim := &sc.cursors[0]
	for idx := 1; idx < len(sc.cursors); idx++ {
		if sc.cursors[idx].curPos > victim.curPos {
			victim = &sc.cursors[idx]
		}
	}
	*victim = shellCursor{
		curPos: 0,
		depth:  0,
		levels: [8]shellScanLevel{{pos: 0, limit: len(sc.s), quoteEnd: len(sc.s), wordStart: -1}},
	}
	return victim
}

func (sc *shellAssignScanner) at(i int) bool {
	s := sc.s
	c := sc.selectCursor(i)
	if nl := lastUnescapedLineBreak(s, c.curPos, i); nl > c.curPos {
		*c = shellCursor{
			curPos: nl,
			depth:  0,
			levels: [8]shellScanLevel{{pos: nl, limit: len(sc.s), quoteEnd: len(sc.s), wordStart: -1}},
		}
	}
	for c.depth > 0 && i >= c.levels[c.depth].quoteEnd {
		qEnd := c.levels[c.depth].quoteEnd
		c.depth--
		lv := &c.levels[c.depth]
		if lv.wordStart < 0 {
			lv.wordStart = lv.pos
			lv.wordFirstByte = s[lv.pos]
		}
		lv.runValid = false
		lv.pos = qEnd
	}
	lv := &c.levels[c.depth]
	for lv.pos < i {
		ch := s[lv.pos]
		if ch == '\'' || ch == '"' {
			n := sc.closedQuoteLen(lv.pos, lv.limit)
			if n > 0 && lv.pos+n <= i {
				if lv.wordStart < 0 {
					lv.wordStart = lv.pos
					lv.wordFirstByte = ch
				}
				lv.runValid = false
				lv.pos += n
				continue
			}
			if n > 0 && c.depth+1 < len(c.levels) {
				c.depth++
				c.levels[c.depth] = shellScanLevel{
					pos:       lv.pos + 1,
					limit:     lv.pos + n - 1,
					quoteEnd:  lv.pos + n,
					wordStart: -1,
				}
				lv = &c.levels[c.depth]
				continue
			}
			lv.resetWord()
			lv.inDecl = false
			lv.pos++
			continue
		}
		if ch == '\\' {
			if lv.pos+1 >= i {
				if lv.wordStart < 0 {
					lv.wordStart = lv.pos
					lv.wordFirstByte = '\\'
				}
				lv.runValid = false
				break
			}
			next := s[lv.pos+1]
			if next == '\n' || (next == '\r' && lv.pos+2 < lv.limit && s[lv.pos+2] == '\n') {
				spliceLen := 2
				if next == '\r' {
					spliceLen = 3
				}
				if lv.pos+spliceLen > i {
					break
				}
				lv.pos += spliceLen
				continue
			}
			if next == '\'' || next == '"' {
				n := sc.closedEscapedQuoteLen(lv.pos, lv.limit)
				if n > 0 && lv.pos+n <= i {
					if lv.wordStart < 0 {
						lv.wordStart = lv.pos
						lv.wordFirstByte = '\\'
					}
					lv.runValid = false
					lv.pos += n
					continue
				}
				if n > 0 && c.depth+1 < len(c.levels) {
					c.depth++
					c.levels[c.depth] = shellScanLevel{
						pos:       lv.pos + 2,
						limit:     lv.pos + n - 2,
						quoteEnd:  lv.pos + n,
						wordStart: -1,
					}
					lv = &c.levels[c.depth]
					continue
				}
				lv.resetWord()
				lv.inDecl = false
				lv.pos += 2
				continue
			}
			if lv.wordStart < 0 {
				lv.wordStart = lv.pos
				lv.wordFirstByte = '\\'
			}
			lv.runValid = false
			lv.pos += 2
			continue
		}
		if isShellWordBreak(ch) {
			lv.finishWord(s)
			if ch != ' ' && ch != '\t' {
				lv.inDecl = false
			}
			lv.pos++
			continue
		}
		if ch == '=' {
			if lv.wordStart < 0 {
				lv.wordStart = lv.pos
				lv.wordFirstByte = '='
				lv.wordSawEq = true
				lv.runValid = false
				lv.pos++
				continue
			}
			if !lv.wordSawEq {
				lv.wordSawEq = true
				if lv.runValid && !lv.runFirstDigit {
					if lv.runAtWordStart {
						lv.wordLeading = true
					}
					if lv.runLetter && (!lv.runLower || lv.runUnderOrDigit) {
						lv.wordHasEnv = true
					}
				} else if lv.pos-lv.wordStart <= 128 {
					if name := stripShellNameQuotes(stripShellLineSplices(s[lv.wordStart:lv.pos])); isShellName(name) {
						lv.wordLeading = true
						if isEnvVarName(name) {
							lv.wordHasEnv = true
						}
					}
				}
			} else if lv.runValid && !lv.runFirstDigit {
				if lv.runAtWordStart {
					lv.wordLeading = true
				}
				if lv.runLetter && (!lv.runLower || lv.runUnderOrDigit) {
					lv.wordHasEnv = true
				}
			}
			lv.runValid = false
			lv.pos++
			continue
		}
		atWordStart := false
		if lv.wordStart < 0 {
			lv.wordStart = lv.pos
			lv.wordFirstByte = ch
			atWordStart = true
		}
		if isShellNameByte(ch) {
			if !lv.runValid {
				lv.runValid = true
				lv.runAtWordStart = atWordStart
				lv.runFirstDigit = '0' <= ch && ch <= '9'
				lv.runLetter = ('a' <= ch && ch <= 'z') || ('A' <= ch && ch <= 'Z')
				lv.runLower = 'a' <= ch && ch <= 'z'
				lv.runUnderOrDigit = ch == '_' || ('0' <= ch && ch <= '9')
			} else {
				if ('a' <= ch && ch <= 'z') || ('A' <= ch && ch <= 'Z') {
					lv.runLetter = true
				}
				if 'a' <= ch && ch <= 'z' {
					lv.runLower = true
				}
				if ch == '_' || ('0' <= ch && ch <= '9') {
					lv.runUnderOrDigit = true
				}
			}
		} else {
			lv.runValid = false
		}
		lv.pos++
	}
	c.curPos = i
	return lv.wordStart >= 0 && (lv.wordHasEnv || (lv.wordLeading && lv.inDecl))
}

// stripShellLineSplices removes unquoted backslash-newline line continuations
// (\<LF> and \<CR><LF>) from a shell word fragment, as a POSIX shell removes
// them before lexical token recognition.
func stripShellLineSplices(w string) string {
	if !strings.Contains(w, `\`) {
		return w
	}
	s := strings.ReplaceAll(w, "\\\r\n", "")
	return strings.ReplaceAll(s, "\\\n", "")
}

// lastUnescapedLineBreak returns the byte offset immediately after the last
// unescaped line break in s[from:to], ignoring \<LF> and \<CR><LF> shell line
// continuations, or -1 if none is present.
func lastUnescapedLineBreak(s string, from, to int) int {
	for k := to - 1; k >= from; k-- {
		if s[k] != '\n' && s[k] != '\r' {
			continue
		}
		lbStart := k
		if s[k] == '\n' && k > 0 && s[k-1] == '\r' {
			lbStart = k - 1
		}
		bs := 0
		for lbStart-1-bs >= 0 && s[lbStart-1-bs] == '\\' {
			bs++
		}
		if s[k] == '\n' && bs%2 == 1 {
			k = lbStart
			continue
		}
		return k + 1
	}
	return -1
}

// isShellDeclKeyword reports whether w is a shell declaration or environment
// builtin that precedes variable assignments.
func isShellDeclKeyword(w string) bool {
	return w == "export" || w == "env" || w == "declare" || w == "local" || w == "readonly" || w == "typeset"
}

// closedQuoteIn returns the byte length of the closed single- or double-quoted
// segment starting at s[0], or 0 if s does not start with a quote that closes
// within s.
func closedQuoteIn(s string) int {
	if len(s) < 2 {
		return 0
	}
	switch s[0] {
	case '\'':
		if i := strings.IndexByte(s[1:], '\''); i >= 0 {
			return i + 2
		}
	case '"':
		for i := 1; i < len(s); i++ {
			if s[i] == '\\' {
				i++
				continue
			}
			if s[i] == '"' {
				return i + 1
			}
		}
	}
	return 0
}

// stripShellNameQuotes returns s with closed single- and double-quoted
// segments unquoted, or "" if s contains an unclosed quote or backslash escape.
func stripShellNameQuotes(s string) string {
	if !strings.ContainsAny(s, `'"\`) {
		return s
	}
	var b strings.Builder
	for i := 0; i < len(s); {
		if s[i] == '\'' || s[i] == '"' {
			n := closedQuoteIn(s[i:])
			if n == 0 || strings.ContainsRune(s[i+1:i+n-1], '\\') {
				return ""
			}
			b.WriteString(s[i+1 : i+n-1])
			i += n
			continue
		}
		if s[i] == '\\' && i+1 < len(s) && (s[i+1] == '\'' || s[i+1] == '"') {
			q := s[i : i+2]
			end := strings.Index(s[i+2:], q)
			if end < 0 || strings.ContainsRune(s[i+2:i+2+end], '\\') {
				return ""
			}
			b.WriteString(s[i+2 : i+2+end])
			i += 4 + end
			continue
		}
		if s[i] == '\\' {
			return ""
		}
		b.WriteByte(s[i])
		i++
	}
	return b.String()
}

// isShellWordBreak reports whether the byte c ends a shell word that
// inShellAssignment reads: whitespace, or one of ";&|(`".
func isShellWordBreak(c byte) bool {
	switch c {
	case ' ', '\t', '\n', '\r', '\v', '\f', ';', '&', '|', '(', '`':
		return true
	default:
		return false
	}
}

// isShellNameByte reports whether c may stand in a shell variable name.
func isShellNameByte(c byte) bool {
	return c == '_' || 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9'
}

// isShellName reports whether name is a shell variable name: a letter or
// "_", then letters, digits and "_".
func isShellName(name string) bool {
	if name == "" || '0' <= name[0] && name[0] <= '9' {
		return false
	}
	for k := 0; k < len(name); k++ {
		if !isShellNameByte(name[k]) {
			return false
		}
	}
	return true
}

// isEnvVarName reports whether the shell variable name holds a letter and
// either holds no lower-case letter, as DB_PASSWORD and X do, or holds "_" or
// a digit, as snake_case or numbered variable names such as db_password,
// api_key, client_secret and oauth2_token do.
func isEnvVarName(name string) bool {
	letter := false
	lower := false
	underscoreOrDigit := false
	for k := 0; k < len(name); k++ {
		switch c := name[k]; {
		case 'a' <= c && c <= 'z':
			letter = true
			lower = true
		case 'A' <= c && c <= 'Z':
			letter = true
		case c == '_' || '0' <= c && c <= '9':
			underscoreOrDigit = true
		}
	}
	return letter && (!lower || underscoreOrDigit)
}

// before reads s up to i and returns the active quote kind open there, or 0.
func (q *openQuotes) before(s string, i int) quoteKinds {
	return q.advance(s, i, 0)
}

func (q *openQuotes) advance(s string, i int, reopenOnly quoteKinds) quoteKinds {
	skipMode := reopenOnly != 0
	canOpen := reopenOnly
	for ; q.pos < i; q.pos++ {
		for q.substTop != nil && q.pos >= q.substTop.val {
			q.substTop = q.substTop.prev
		}
		c := s[q.pos]
		if c == '\n' || c == '\r' {
			q.active = 0
			q.bs = 0
			q.openJSONKind = jsonContainerNone
			q.subSQ = false
			q.subEscDQ = false
			q.beforeInnerDQ = false
			q.parenDepth = 0
			q.openInParen = false
			if skipMode {
				canOpen = 0
			}
			continue
		}
		if c == '\\' {
			q.bs++
			continue
		}
		bs := q.bs
		q.bs = 0
		if bs%2 == 0 {
			switch q.active {
			case 0:
				switch c {
				case '(':
					q.parenDepth++
				case ')':
					if q.parenDepth > 0 {
						q.parenDepth--
					}
				case '{', '[':
					q.jsonTop = &jsonContainerNode{
						entry: jsonContainerEntry{openChar: c, pos: q.pos, isJSON: isJSONContainerOpener(s, q.pos)},
						prev:  q.jsonTop,
					}
				case '}', ']':
					if q.jsonTop != nil && jsonContainerMatches(q.jsonTop.entry.openChar, c) {
						q.jsonTop = q.jsonTop.prev
					}
				case '$':
					if q.pos+1 < len(s) && s[q.pos+1] == '(' {
						if endPos := shellScanClose(q.shellMemosFor(s), s, shellFrameCS, q.pos+2, false, false, false); endPos >= 0 {
							q.substTop = &intListNode{val: endPos, prev: q.substTop}
						}
					}
				case '`':
					if endPos := shellScanClose(q.shellMemosFor(s), s, shellFrameBT, q.pos+1, false, false, false); endPos >= 0 {
						q.substTop = &intListNode{val: endPos, prev: q.substTop}
					}
				}
			case 1:
				if !q.beforeInnerDQ && ((c == '$' && q.pos+1 < len(s) && s[q.pos+1] == '(') || c == '`') {
					escJSON := q.openJSONKind != jsonContainerNone
					var endPos int
					if c == '$' {
						endPos = shellScanClose(q.shellMemosFor(s), s, shellFrameCS, q.pos+2, false, escJSON, false)
					} else {
						endPos = shellScanClose(q.shellMemosFor(s), s, shellFrameBT, q.pos+1, true, escJSON, false)
					}
					if endPos >= 0 {
						q.substTop = &intListNode{val: endPos, prev: q.substTop}
					}
				}
			}
		}
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
			if q.active == 1 && !q.beforeInnerDQ && c == '"' && !q.subSQ {
				q.subEscDQ = !q.subEscDQ
			}
			continue
		}
		if q.active == 1 && !q.beforeInnerDQ && kind == 2 {
			if q.openJSONKind != jsonContainerNone && bs%4 == 2 {
				continue
			}
			if q.subSQ {
				if isMidWordApostrophe(s, q.pos) && !opensPrefixedString(s, q.pos) {
					continue
				}
				q.subSQ = false
				continue
			}
			if (!isMidWordApostrophe(s, q.pos) || opensPrefixedString(s, q.pos)) && !isTrailingApostrophe(s, q.pos) {
				q.subSQ = true
			}
			continue
		}
		if q.active == 1 && kind == 4 {
			q.beforeInnerDQ = !q.beforeInnerDQ
			if q.beforeInnerDQ {
				q.subEscDQ = false
			}
		}
		if isApostrophe(s, q.pos, kind, q.active) {
			continue
		}
		switch q.active {
		case 0:
			if skipMode && kind != canOpen {
				continue
			}
			q.active = kind
			q.openPos = q.pos
			q.subSQ = false
			q.subEscDQ = false
			q.beforeInnerDQ = false
			q.openInParen = q.parenDepth > 0 || q.substTop != nil
			if kind == 1 {
				q.openJSONKind = classifyQuoteJSONContainer(s, q.pos, 0, false, q.jsonTop)
			} else {
				q.openJSONKind = jsonContainerNone
			}
		case kind:
			q.active = 0
			q.openJSONKind = jsonContainerNone
			q.subSQ = false
			q.subEscDQ = false
			q.beforeInnerDQ = false
			q.openInParen = false
		}
	}
	for q.substTop != nil && q.pos >= q.substTop.val {
		q.substTop = q.substTop.prev
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
		for q.substTop != nil && q.pos >= q.substTop.val {
			q.substTop = q.substTop.prev
		}
		return
	}
	if e := selfContainedValueEnd(s, q.pos, i); e > q.pos {
		q.pos, q.bs = e, 0
	}
	q.advance(s, i, savedActive)
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
	q.openJSONKind = jsonContainerNone
	q.subSQ = false
	q.subEscDQ = false
	q.beforeInnerDQ = false
	q.openInParen = false
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

// trimUnopenedClosers returns v, a bare suffix glued to a credential, without
// the ")", "]" and "}" at its end that close no bracket opened in v and
// without the sentence punctuation (".,;:!?") among them, as in the ")." of
// (echo 'Bearer abc'x). A closer that closes a bracket opened in v stays in
// v, together with everything before it, as the "()" of x)() and the "(b)" of
// a)(b) do. One forward pass counts the openers of each kind that are still
// open and records where the last closer that closes one of them ends, so
// the cost is linear in len(v) however many closers v ends with.
func trimUnopenedClosers(v string) string {
	keep := 0
	var open [3]int
	for i := 0; i < len(v); i++ {
		switch v[i] {
		case '(':
			open[0]++
		case '[':
			open[1]++
		case '{':
			open[2]++
		case ')', ']', '}':
			if k := strings.IndexByte(")]}", v[i]); open[k] > 0 {
				open[k]--
				keep = i + 1
			}
		}
	}
	end := len(v)
	for end > keep && strings.IndexByte(".,;:!?)]}", v[end-1]) >= 0 {
		end--
	}
	return v[:end]
}

// sensitiveValueEndWith returns the end of the value matched at
// s[start:end]. A value that opens a JSON object or array, such as the one in
// {"token": {"value": "..."}}, is extended to its closing bracket, or to the
// end of s if it is not closed, so that the fields nested in it are redacted
// together with it. Brackets inside strings do not count, including strings
// delimited by escaped quotes, as in JSON text formatted with %q. sh, if
// bound to s, reads the brackets of s once (see bracketEnd).
func sensitiveValueEndWith(sh *shellAssignScanner, s string, start, end int) int {
	if c := s[start]; c != '{' && c != '[' {
		return end
	}
	e, closed := bracketEnd(sh, s, start)
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
