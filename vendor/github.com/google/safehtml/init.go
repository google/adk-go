// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package safehtml

import (
	"github.com/google/safehtml/internal/raw"
)

// stringConstant is an unexported string type. Users of this package cannot
// create values of this type except by passing an untyped string constant to
// functions which expect a stringConstant. This type should only be used in
// function and method parameters.
type stringConstant string

// The following functions are used by package uncheckedconversions
// (via package raw) to create safe HTML types from plain strings.

func htmlRaw(s string) HTML {
	return HTML{s}
}

func scriptRaw(s string) Script {
	return Script{s}
}

func style(s string) Style {
	return Style{s}
}

func styleSheetRaw(s string) StyleSheet {
	return StyleSheet{s}
}

func urlRaw(s string) URL {
	return URL{s}
}

func trustedResourceURLRaw(s string) TrustedResourceURL {
	return TrustedResourceURL{s}
}

func identifierRaw(s string) Identifier {
	return Identifier{s}
}

func init() {
	raw.HTML = htmlRaw
	raw.Script = scriptRaw
	raw.Style = style
	raw.StyleSheet = styleSheetRaw
	raw.URL = urlRaw
	raw.TrustedResourceURL = trustedResourceURLRaw
	raw.Identifier = identifierRaw
}
