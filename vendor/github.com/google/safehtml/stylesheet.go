// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package safehtml

import (
	"container/list"
	"fmt"
	"regexp"
	"strings"
)

// A StyleSheet is an immutable string-like type which represents a CSS
// style sheet and guarantees that its value, as a string, will not cause
// untrusted script execution (cross-site scripting) when evaluated as CSS
// in a browser.
//
// StyleSheet's string representation can safely be interpolated as the
// content of a style element within HTML. The StyleSheet string should
// not be escaped before interpolation.
type StyleSheet struct {
	// We declare a StyleSheet not as a string but as a struct wrapping a string
	// to prevent construction of StyleSheet values through string conversion.
	str string
}

// StyleSheetFromConstant constructs a StyleSheet with the
// underlying stylesheet set to the given styleSheet, which must be an untyped string
// constant.
//
// No runtime validation or sanitization is performed on script; being under
// application control, it is simply assumed to comply with the StyleSheet
// contract.
func StyleSheetFromConstant(styleSheet stringConstant) StyleSheet {
	return StyleSheet{string(styleSheet)}
}

// CSSRule constructs a StyleSheet containng a CSS rule of the form:
//   selector{style}
// It returns an error if selector contains disallowed characters or unbalanced
// brackets.
//
// The constructed StyleSheet value is guaranteed to fulfill its type contract,
// but is not guaranteed to be semantically valid CSS.
func CSSRule(selector string, style Style) (StyleSheet, error) {
	if strings.ContainsRune(selector, '<') {
		return StyleSheet{}, fmt.Errorf("selector %q contains '<'", selector)
	}
	selectorWithoutStrings := cssStringPattern.ReplaceAllString(selector, "")
	if matches := invalidCSSSelectorRune.FindStringSubmatch(selectorWithoutStrings); matches != nil {
		return StyleSheet{}, fmt.Errorf("selector %q contains %q, which is disallowed outside of CSS strings", selector, matches[0])
	}
	if !hasBalancedBrackets(selectorWithoutStrings) {
		return StyleSheet{}, fmt.Errorf("selector %q contains unbalanced () or [] brackets", selector)
	}
	return StyleSheet{fmt.Sprintf("%s{%s}", selector, style.String())}, nil
}

var (
	// cssStringPattern matches a single- or double-quoted CSS string.
	cssStringPattern = regexp.MustCompile(
		`"([^"\r\n\f\\]|\\[\s\S])*"|` + // Double-quoted string literal
			`'([^'\r\n\f\\]|\\[\s\S])*'`) // Single-quoted string literal

	// invalidCSSSelectorRune matches a rune that is not allowed in a CSS3
	// selector that does not contain string literals.
	// See https://w3.org/TR/css3-selectors/#selectors.
	invalidCSSSelectorRune = regexp.MustCompile(`[^-_a-zA-Z0-9#.:* ,>+~[\]()=^$|]`)
)

// hasBalancedBrackets returns whether s has balanced () and [] brackets.
func hasBalancedBrackets(s string) bool {
	stack := list.New()
	for i := 0; i < len(s); i++ {
		c := s[i]
		if expected, ok := matchingBrackets[c]; ok {
			e := stack.Back()
			if e == nil {
				return false
			}
			// Skip success check for this type assertion since it is trivial to
			// see that only bytes are pushed onto this stack.
			if v := e.Value.(byte); v != expected {
				return false
			}
			stack.Remove(e)
			continue
		}
		for _, openBracket := range matchingBrackets {
			if c == openBracket {
				stack.PushBack(c)
				break
			}
		}
	}
	return stack.Len() == 0
}

// matchingBrackets[x] is the opening bracket that matches closing bracket x.
var matchingBrackets = map[byte]byte{
	')': '(',
	']': '[',
}

// String returns the string form of the StyleSheet.
func (s StyleSheet) String() string {
	return s.str
}
