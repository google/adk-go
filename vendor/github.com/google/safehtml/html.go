// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package safehtml

import (
	"bytes"
	"html"
	"unicode"

	"golang.org/x/text/unicode/rangetable"
)

// An HTML is an immutable string-like type that is safe to use in HTML
// contexts in DOM APIs and HTML documents.
//
// HTML guarantees that its value as a string will not cause untrusted script
// execution when evaluated as HTML in a browser.
//
// Values of this type are guaranteed to be safe to use in HTML contexts,
// such as assignment to the innerHTML DOM property, or interpolation into an
// HTML template in HTML PC_DATA context, in the sense that the use will not
// result in a Cross-site Scripting (XSS) vulnerability.
type HTML struct {
	// We declare an HTML not as a string but as a struct wrapping a string
	// to prevent construction of HTML values through string conversion.
	str string
}

// HTMLer is implemented by any value that has an HTML method, which defines the
// safe HTML format for that value.
type HTMLer interface {
	HTML() HTML
}

// HTMLEscaped returns an HTML whose value is text, with the characters [&<>"'] escaped.
//
// text is coerced to interchange valid, so the resulting HTML contains only
// valid UTF-8 characters which are legal in HTML and XML.
//
func HTMLEscaped(text string) HTML {
	return HTML{escapeAndCoerceToInterchangeValid(text)}
}

// HTMLConcat returns an HTML which contains, in order, the string representations
// of the given htmls.
func HTMLConcat(htmls ...HTML) HTML {
	var b bytes.Buffer
	for _, html := range htmls {
		b.WriteString(html.String())
	}
	return HTML{b.String()}
}

// String returns the string form of the HTML.
func (h HTML) String() string {
	return h.str
}

// escapeAndCoerceToInterchangeValid coerces the string to interchange-valid
// UTF-8 and then HTML-escapes it.
func escapeAndCoerceToInterchangeValid(str string) string {
	return html.EscapeString(coerceToUTF8InterchangeValid(str))
}

// coerceToUTF8InterchangeValid coerces a string to interchange-valid UTF-8.
// Illegal UTF-8 bytes are replaced with the Unicode replacement character
// ('\uFFFD'). C0 and C1 control codes (other than CR LF HT FF) and
// non-characters are also replaced with the Unicode replacement character.
func coerceToUTF8InterchangeValid(s string) string {
	// TODO: Replace this entire function with stdlib function if https://golang.org/issue/25805 gets addressed.
	runes := make([]rune, 0, len(s))
	// If s contains any invalid UTF-8 byte sequences, range will have rune
	// contain the Unicode replacement character and there's no need to call
	// utf8.ValidRune. I.e. iteration over the string implements
	// CoerceToStructurallyValid() from C++/Java.
	// See https://blog.golang.org/strings.
	for _, rune := range s {
		if unicode.Is(controlAndNonCharacter, rune) {
			runes = append(runes, unicode.ReplacementChar)
		} else {
			runes = append(runes, rune)
		}
	}
	return string(runes)
}

// controlAndNonCharacters contains the non-interchange-valid codepoints.
//
// See http://www.w3.org/TR/html5/syntax.html#preprocessing-the-input-stream
//
// safehtml functions do a lot of lookups on these tables, so merging them is probably
// worth it to avoid comparing against both tables each time.
var controlAndNonCharacter = rangetable.Merge(unicode.Noncharacter_Code_Point, controlChar)

// controlChar contains Unicode control characters disallowed in interchange
// valid UTF-8. This table is slightly different from unicode.Cc:
// - Disallows null.
// - Allows LF, CR, HT, and FF.
//
// unicode.C is mentioned in unicode.IsControl; it contains "special" characters
// which includes at least control characters, surrogate code points, and
// formatting codepoints (e.g. word joiner). We don't need to exclude all of
// those. In particular, surrogates are handled by the for loop converting
// invalid UTF-8 byte sequences to the Unicode replacement character.
var controlChar = &unicode.RangeTable{
	R16: []unicode.Range16{
		{0x0000, 0x0008, 1},
		{0x000B, 0x000B, 1},
		{0x000E, 0x001F, 1},
		{0x007F, 0x009F, 1},
	},
	LatinOffset: 4,
}
