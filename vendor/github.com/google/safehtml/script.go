// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package safehtml

import (
	"encoding/json"
	"fmt"
	"regexp"
)

// A Script is an immutable string-like type which represents JavaScript
// code and guarantees that its value, as a string, will not cause execution
// of unconstrained attacker controlled code (cross-site scripting) when
// evaluated as JavaScript in a browser.
//
// Script's string representation can safely be interpolated as the
// content of a script element within HTML, and can safely be passed to DOM
// properties and functions which expect JavaScript. In these cases, the Script
// string should not be escaped. Script's string representation can also be safely
// used as the value for on* attribute handlers in HTML, though the Script string
// must be escaped before such use.
//
// Note that the Script might contain text that is attacker-controlled but
// that text should have been interpolated with appropriate escaping,
// sanitization and/or validation into the right location in the script, such
// that it is highly constrained in its effect (for example, it had to match a
// set of allowed words).
//
// In order to ensure that an attacker cannot influence the Script
// value, a Script can only be instantiated from compile-time
// constant string literals or security-reviewed unchecked conversions,
// but never from arbitrary string values potentially representing untrusted
// user input.
type Script struct {
	// We declare a Script not as a string but as a struct wrapping a string
	// to prevent construction of Script values through string conversion.
	str string
}

// ScriptFromConstant constructs a Script with its underlying script set
// to the given script, which must be an untyped string constant.
//
// No runtime validation or sanitization is performed on script; being under
// application control, it is simply assumed to comply with the Script
// contract.
func ScriptFromConstant(script stringConstant) Script {
	return Script{string(script)}
}

// ScriptFromDataAndConstant constructs a Script of the form
//
//   var name = data; script
//
// where name is the supplied variable name, data is the supplied data value
// encoded as JSON using encoding/json.Marshal, and script is the supplied
// JavaScript statement or sequence of statements. The supplied name and script
// must both be untyped string constants. It returns an error if name is not a
// valid Javascript identifier or JSON encoding fails.
//
// No runtime validation or sanitization is performed on script; being under
// application control, it is simply assumed to comply with the Script
// contract.
func ScriptFromDataAndConstant(name stringConstant, data interface{}, script stringConstant) (Script, error) {
	if !jsIdentifierPattern.MatchString(string(name)) {
		return Script{}, fmt.Errorf("variable name %q is an invalid Javascript identifier", string(name))
	}
	json, err := json.Marshal(data)
	if err != nil {
		return Script{}, err
	}
	return Script{fmt.Sprintf("var %s = %s;\n%s", name, json, string(script))}, nil
}

// jsIdentifierPattern matches strings that are valid Javascript identifiers.
//
// This pattern accepts only a subset of valid identifiers defined in
// https://tc39.github.io/ecma262/#sec-names-and-keywords. In particular,
// it does not match identifiers that contain non-ASCII letters, Unicode
// escape sequences, and the Unicode format-control characters
// \u200C (zero-width non-joiner) and \u200D (zero-width joiner).
var jsIdentifierPattern = regexp.MustCompile(`^[$_a-zA-Z][$_a-zA-Z0-9]+$`)

// String returns the string form of the Script.
func (s Script) String() string {
	return s.str
}
