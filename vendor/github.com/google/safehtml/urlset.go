// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package safehtml

import (
	"bytes"
	"strconv"
)

// https://infra.spec.whatwg.org/#ascii-whitespace
// ASCII whitespace is U+0009 TAB, U+000A LF, U+000C FF, U+000D CR, or U+0020 SPACE.
var asciiWhitespace [256]bool

// Metacharacters that affect parsing of srcset values.
var srcsetMetachars [256]bool

func init() {
	asciiWhitespace['\t'] = true
	asciiWhitespace[' '] = true
	asciiWhitespace['\n'] = true
	asciiWhitespace['\f'] = true
	asciiWhitespace['\r'] = true

	srcsetMetachars['\t'] = true
	srcsetMetachars[' '] = true
	srcsetMetachars['\n'] = true
	srcsetMetachars['\f'] = true
	srcsetMetachars['\r'] = true
	srcsetMetachars[','] = true
}

// URLSetSanitized returns a safe srcset by individually vetting each
// substring that specifies a URL.
//
// https://html.spec.whatwg.org/multipage/images.html#srcset-attributes
func URLSetSanitized(str string) URLSet {
	var buffer bytes.Buffer

	for len(str) != 0 {
		// Consume one image candidate
		var url, metadata string
		_, str = consumeIn(str, asciiWhitespace)
		url, str = consumeNotIn(str, asciiWhitespace)
		_, str = consumeIn(str, asciiWhitespace)
		metadata, str = consumeNotIn(str, srcsetMetachars)
		_, str = consumeIn(str, asciiWhitespace)

		// Append sanitized content onto buffer.
		if len(url) != 0 && isSafeURL(url) && isOptionalSrcMetadataWellFormed(metadata) {
			if buffer.Len() != 0 {
				// The space before the comma is necessary because
				// a comma adjacent to a URL will attach to it.
				buffer.WriteString(" , ")
			}
			// URL may contain commas.  Disambiguate.
			appendURLToSet(url, &buffer)
			if len(metadata) != 0 {
				buffer.WriteByte(' ')
				buffer.WriteString(metadata)
			}
		}

		// Consume any trailing comma
		if len(str) == 0 || str[0] != ',' {
			break
		}
		str = str[1:]
	}

	if buffer.Len() == 0 {
		return URLSet{InnocuousURL}
	}

	return URLSet{buffer.String()}
}

// appendURLToSet appends a URL so that it does not start or end with a comma
//
// https://html.spec.whatwg.org/multipage/images.html#srcset-attributes
// parsing step 2 which says:
// """
// A valid non-empty URL that does not start or end with a U+002C COMMA character (,),
// referencing a non-interactive, optionally animated, image resource that is neither
// paged nor scripted
// """
//
// Simply replacing all commas would break data:image/png;base64,IMAGECONTENT
// Note: This breaks data URLs with empty content since they end with a comma.
// We could handle that case by appending a '#'.
func appendURLToSet(url string, buffer *bytes.Buffer) {
	n := len(url)
	left, right := 0, n
	if url[left] == ',' {
		buffer.WriteString("%2c")
		left++
	}
	commaAtEnd := false
	if left < right && url[right-1] == ',' {
		commaAtEnd = true
		right--
	}
	buffer.WriteString(url[left:right])
	if commaAtEnd {
		buffer.WriteString("%2c")
	}
}

// consumeNotIn uses bytes in str as bit indices in mask to find
// the least index >= left whose byte corresponds to a zero bit.
func consumeNotIn(str string, mask [256]bool) (consumed, rest string) {
	i, n := 0, len(str)
	for ; i < n; i++ {
		if mask[str[i]] {
			return str[0:i], str[i:n]
		}
	}
	return str, ""
}

// consumeIn is like consumeNotIn but treats mask as inverted.
func consumeIn(str string, mask [256]bool) (consumed, rest string) {
	for i, n := 0, len(str); i < n; i++ {
		if !mask[str[i]] {
			return str[0:i], str[i:n]
		}
	}
	return str, ""
}

// isOptionalSrcMetadataWellFormed is true when its input is empty and
// when it is a floating point number optionally followed by an ASCII letter.
func isOptionalSrcMetadataWellFormed(metadata string) bool {
	// srcset for both image candidates (<img srcset>) and
	// the proposal for script allow a number and an optional letter
	// afterwards.
	n := len(metadata)
	if n == 0 {
		// Metadata is optional
		return true
	}
	metadataPrefix := metadata
	if last := metadata[n-1] | 32; 'a' <= last && last <= 'z' {
		metadataPrefix = metadata[0 : n-1]
	}
	// This overmatches
	// html.spec.whatwg.org/multipage/common-microsyntaxes.html#valid-floating-point-number
	// but is sufficient.
	_, err := strconv.ParseFloat(metadataPrefix, 64)
	return err == nil
}

// URLSet corresponds to the value of a srcset attribute outside a
// TrustedResourceURL context.
type URLSet struct {
	// We declare a URLSet not as a string but as a struct wrapping a string
	// to prevent construction of URL values through string conversion.
	str string
}

// String returns the string content of a URLSet
func (s URLSet) String() string {
	return s.str
}
