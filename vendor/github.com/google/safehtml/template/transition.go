// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"strings"
)

// transitionFunc is the array of context transition functions for text nodes.
// A transition function takes a context and template text input, and returns
// the updated context and the number of bytes consumed from the front of the
// input.
var transitionFunc = [...]func(context, []byte) (context, int){
	stateText:               tText,
	stateSpecialElementBody: tSpecialTagEnd,
	stateTag:                tTag,
	stateAttrName:           tAttrName,
	stateAfterName:          tAfterName,
	stateBeforeValue:        tBeforeValue,
	stateHTMLCmt:            tHTMLCmt,
	stateAttr:               tAttr,
	stateError:              tError,
}

var commentStart = []byte("<!--")
var commentEnd = []byte("-->")

// tText is the context transition function for the text state.
func tText(c context, s []byte) (context, int) {
	k := 0
	for {
		i := k + bytes.IndexByte(s[k:], '<')
		if i < k || i+1 == len(s) {
			return c, len(s)
		} else if i+4 <= len(s) && bytes.Equal(commentStart, s[i:i+4]) {
			return context{state: stateHTMLCmt}, i + 4
		}
		i++
		end := false
		if s[i] == '/' {
			if i+1 == len(s) {
				return c, len(s)
			}
			end, i = true, i+1
		}
		j, e := eatTagName(s, i)
		if j != i {
			// We've found an HTML tag.
			ret := context{state: stateTag}
			// Element name not needed if we are at the end of the element.
			if !end {
				ret.element = e
			}
			return ret, j
		}
		k = j
	}
}

// specialElements contains the names of elements whose bodies are treated
// differently by the parser and escaper from stateText.
var specialElements = map[string]bool{
	"script":   true,
	"style":    true,
	"textarea": true,
	"title":    true,
}

// voidElements contains the names of all void elements.
// https://www.w3.org/TR/html5/syntax.html#void-elements
var voidElements = map[string]bool{
	"area":   true,
	"base":   true,
	"br":     true,
	"col":    true,
	"embed":  true,
	"hr":     true,
	"img":    true,
	"input":  true,
	"keygen": true,
	"link":   true,
	"meta":   true,
	"param":  true,
	"source": true,
	"track":  true,
	"wbr":    true,
}

// tTag is the context transition function for the tag state.
func tTag(c context, s []byte) (context, int) {
	// Find the attribute name.
	i := eatWhiteSpace(s, 0)
	if i == len(s) {
		return c, len(s)
	}
	if s[i] == '>' {
		ret := context{
			state:      stateText,
			element:    c.element,
			scriptType: c.scriptType,
			linkRel:    c.linkRel,
		}
		if specialElements[c.element.name] {
			ret.state = stateSpecialElementBody
		}
		if c.element.name != "" && voidElements[c.element.name] {
			// Special case: end of start tag of a void element.
			// Discard unnecessary state, since this element have no content.
			ret.element = element{}
			ret.scriptType = ""
			ret.linkRel = ""
		}
		return ret, i + 1
	}
	j, err := eatAttrName(s, i)
	if err != nil {
		return context{state: stateError, err: err}, len(s)
	}
	state := stateTag
	if i == j {
		return context{
			state: stateError,
			err:   errorf(ErrBadHTML, nil, 0, "expected space, attr name, or end of tag, but got %q", s[i:]),
		}, len(s)
	}

	if j == len(s) {
		state = stateAttrName
	} else {
		state = stateAfterName
	}
	return context{
		state:   state,
		element: c.element,
		attr:    attr{name: strings.ToLower(string(s[i:j]))},
		linkRel: c.linkRel,
	}, j
}

// tAttrName is the context transition function for stateAttrName.
func tAttrName(c context, s []byte) (context, int) {
	i, err := eatAttrName(s, 0)
	if err != nil {
		return context{state: stateError, err: err}, len(s)
	} else if i != len(s) {
		c.state = stateAfterName
	}
	return c, i
}

// tAfterName is the context transition function for stateAfterName.
func tAfterName(c context, s []byte) (context, int) {
	// Look for the start of the value.
	i := eatWhiteSpace(s, 0)
	if i == len(s) {
		return c, len(s)
	} else if s[i] != '=' {
		// Occurs due to tag ending '>', and valueless attribute.
		c.state = stateTag
		return c, i
	}
	c.state = stateBeforeValue
	// Consume the "=".
	return c, i + 1
}

// tBeforeValue is the context transition function for stateBeforeValue.
func tBeforeValue(c context, s []byte) (context, int) {
	i := eatWhiteSpace(s, 0)
	if i == len(s) {
		return c, len(s)
	}
	// Find the attribute delimiter.
	// TODO: consider disallowing single-quoted or unquoted attribute values completely, even in hardcoded template text.
	delim := delimSpaceOrTagEnd
	switch s[i] {
	case '\'':
		delim, i = delimSingleQuote, i+1
	case '"':
		delim, i = delimDoubleQuote, i+1
	}
	c.state, c.delim = stateAttr, delim
	return c, i
}

// tHTMLCmt is the context transition function for stateHTMLCmt.
func tHTMLCmt(c context, s []byte) (context, int) {
	if i := bytes.Index(s, commentEnd); i != -1 {
		return context{}, i + 3
	}
	return c, len(s)
}

var (
	specialTagEndPrefix = []byte("</")
	tagEndSeparators    = []byte("> \t\n\f/")
)

// tSpecialTagEnd is the context transition function for raw text, RCDATA
// script data, and stylesheet element states.
func tSpecialTagEnd(c context, s []byte) (context, int) {
	if specialElements[c.element.name] {
		if i := indexTagEnd(s, []byte(c.element.name)); i != -1 {
			return context{}, i
		}
	}
	return c, len(s)
}

// indexTagEnd finds the index of a special tag end in a case insensitive way, or returns -1
func indexTagEnd(s []byte, tag []byte) int {
	res := 0
	plen := len(specialTagEndPrefix)
	for len(s) > 0 {
		// Try to find the tag end prefix first
		i := bytes.Index(s, specialTagEndPrefix)
		if i == -1 {
			return i
		}
		s = s[i+plen:]
		// Try to match the actual tag if there is still space for it
		if len(tag) <= len(s) && bytes.EqualFold(tag, s[:len(tag)]) {
			s = s[len(tag):]
			// Check the tag is followed by a proper separator
			if len(s) > 0 && bytes.IndexByte(tagEndSeparators, s[0]) != -1 {
				return res + i
			}
			res += len(tag)
		}
		res += i + plen
	}
	return -1
}

// tAttr is the context transition function for the attribute state.
func tAttr(c context, s []byte) (context, int) {
	return c, len(s)
}

// tError is the context transition function for the error state.
func tError(c context, s []byte) (context, int) {
	return c, len(s)
}

// eatAttrName returns the largest j such that s[i:j] is an attribute name.
// It returns an error if s[i:] does not look like it begins with an
// attribute name, such as encountering a quote mark without a preceding
// equals sign.
func eatAttrName(s []byte, i int) (int, *Error) {
	for j := i; j < len(s); j++ {
		switch s[j] {
		case ' ', '\t', '\n', '\f', '\r', '=', '>':
			return j, nil
		case '\'', '"', '<':
			// These result in a parse warning in HTML5 and are
			// indicative of serious problems if seen in an attr
			// name in a template.
			return -1, errorf(ErrBadHTML, nil, 0, "%q in attribute name: %.32q", s[j:j+1], s)
		default:
			// No-op.
		}
	}
	return len(s), nil
}

// asciiAlpha reports whether c is an ASCII letter.
func asciiAlpha(c byte) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z'
}

// asciiAlphaNum reports whether c is an ASCII letter or digit.
func asciiAlphaNum(c byte) bool {
	return asciiAlpha(c) || '0' <= c && c <= '9'
}

// eatTagName returns the largest j such that s[i:j] is a tag name and the tag name.
func eatTagName(s []byte, i int) (int, element) {
	if i == len(s) || !asciiAlpha(s[i]) {
		return i, element{}
	}
	j := i + 1
	for j < len(s) {
		x := s[j]
		if asciiAlphaNum(x) {
			j++
			continue
		}
		// Allow "x-y" or "x:y" but not "x-", "-y", or "x--y".
		if (x == ':' || x == '-') && j+1 < len(s) && asciiAlphaNum(s[j+1]) {
			j += 2
			continue
		}
		break
	}
	return j, element{name: strings.ToLower(string(s[i:j]))}
}

// eatWhiteSpace returns the largest j such that s[i:j] is white space.
func eatWhiteSpace(s []byte, i int) int {
	for j := i; j < len(s); j++ {
		switch s[j] {
		case ' ', '\t', '\n', '\f', '\r':
			// No-op.
		default:
			return j
		}
	}
	return len(s)
}
