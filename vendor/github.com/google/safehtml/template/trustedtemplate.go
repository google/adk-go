// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package template

// A TrustedTemplate is an immutable string-like type containing a
// safehtml/template template body. It can be safely loaded as template
// text without the risk of untrusted template execution.
//
// In order to ensure that an attacker cannot influence the TrustedTemplate
// value, a TrustedTemplate can be instantiated only from untyped string constants,
// and never from arbitrary string values potentially representing untrusted user input.
//
type TrustedTemplate struct {
	// We declare a TrustedTemplate not as a string but as a struct wrapping a string
	// to prevent construction of TrustedTemplate values through string conversion.
	tmpl string
}

// MakeTrustedTemplate constructs a TrustedTemplate with its underlying
// tmpl set to the given tmpl, which must be an untyped string constant.
//
// No runtime validation or sanitization is performed on tmpl; being under
// application control, it is simply assumed to comply with the TrustedTemplate type
// contract.
func MakeTrustedTemplate(tmpl stringConstant) TrustedTemplate {
	return TrustedTemplate{string(tmpl)}
}

// String returns the string form of the TrustedTemplate.
func (t TrustedTemplate) String() string {
	return t.tmpl
}
