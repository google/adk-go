// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package safehtml

import (
	"fmt"
	"regexp"
)

// A Identifier is an immutable string-like type that is safe to use in HTML
// contexts as an identifier for HTML elements. For example, it is unsafe to
// insert an untrusted string into a
//
//   <img name="..."></img>
//
// context since the string may be controlled by an attacker who can assign it
// a value that masks existing DOM properties (i.e. DOM Clobbering). An
// attacker may also be able to force legitimate Javascript code, which uses
// document.getElementsByName(...) to read DOM elements, to refer to this
// element. This may lead to unintended side effects, particularly if that
// element contains attacker-controlled data. It is, however, safe to use an
// Identifier in this context since its value is known to be partially or fully
// under application control.
//
// In order to ensure that an attacker cannot influence the Identifier value,
// an Identifier can only be instantiated from a compile-time constant string
// literal prefix.
//
// Note that Identifier is Go-specific and therefore does not have a Proto form
// for cross-language use.
type Identifier struct {
	// We declare a Identifier not as a string but as a struct wrapping a string
	// to prevent construction of Identifier values through string conversion.
	str string
}

// To minimize the risk of parsing errors, Identifier values must start with an
// alphabetical rune, and comprise of only alphanumeric, '-', and '_' runes.

// startsWithAlphabetPattern matches strings that start with an alphabetical rune.
var startsWithAlphabetPattern = regexp.MustCompile(`^[a-zA-Z]`)

// onlyAlphanumericsOrHyphenPattern matches strings that only contain alphanumeric,
// '-' and '_' runes.
var onlyAlphanumericsOrHyphenPattern = regexp.MustCompile(`^[-_a-zA-Z0-9]*$`)

// IdentifierFromConstant constructs an Identifier with its underlying identifier
// set to the given string value, which must be an untyped string constant. It
// panics if value does not start with an alphabetic rune or contains any
// non-alphanumeric runes other than '-' and '_'.
func IdentifierFromConstant(value stringConstant) Identifier {
	if !startsWithAlphabetPattern.MatchString(string(value)) ||
		!onlyAlphanumericsOrHyphenPattern.MatchString(string(value)) {
		panic(fmt.Sprintf("invalid identifier %q", string(value)))
	}
	return Identifier{string(value)}
}

// IdentifierFromConstantPrefix constructs an Identifier with its underlying string
// set to the string formed by joining prefix, which must be an untyped string
// constant, and value with a hyphen. It panics if prefix or value contain any
// non-alphanumeric runes other than '-' and '_', or if prefix does not start with
// an alphabetic rune.
func IdentifierFromConstantPrefix(prefix stringConstant, value string) Identifier {
	prefixString := string(prefix)
	if !startsWithAlphabetPattern.MatchString(string(prefix)) ||
		!onlyAlphanumericsOrHyphenPattern.MatchString(string(prefix)) {
		panic(fmt.Sprintf("invalid prefix %q", string(prefix)))
	}
	if !onlyAlphanumericsOrHyphenPattern.MatchString(value) {
		panic(fmt.Sprintf("value %q contains non-alphanumeric runes", value))
	}
	return Identifier{prefixString + "-" + value}
}

// String returns the string form of the Identifier.
func (i Identifier) String() string {
	return i.str
}
