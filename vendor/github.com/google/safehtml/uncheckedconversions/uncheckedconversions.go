// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

// Package uncheckedconversions provides functions to create values of package
// safehtml types from plain strings. Use of these functions could potentially
// result in instances of safe HTML types that violate their type contracts,
// and hence result in security vulnerabilties.
//
// Avoid use of the functions in this file whenever possible; instead prefer to
// create instances of package safehtml types using inherently safe builders or
// template systems.
//
// Example appropriate uses include:
//   * Wrapping the result of general-purpose or application-specific content
//     sanitizer libraries.
//   * Wrapping the result of rendering strictly contextually autoescaping
//     templates (assuming the template's autoescaping implementation is indeed
//     strict enough to support the type contract).
package uncheckedconversions

import (
	"github.com/google/safehtml/internal/raw"
	"github.com/google/safehtml"
)

var html = raw.HTML.(func(string) safehtml.HTML)
var script = raw.Script.(func(string) safehtml.Script)
var style = raw.Style.(func(string) safehtml.Style)
var styleSheet = raw.StyleSheet.(func(string) safehtml.StyleSheet)
var url = raw.URL.(func(string) safehtml.URL)
var trustedResourceURL = raw.TrustedResourceURL.(func(string) safehtml.TrustedResourceURL)
var identifier = raw.Identifier.(func(string) safehtml.Identifier)

// HTMLFromStringKnownToSatisfyTypeContract converts a string into a HTML.
//
func HTMLFromStringKnownToSatisfyTypeContract(s string) safehtml.HTML {
	return html(s)
}

// ScriptFromStringKnownToSatisfyTypeContract converts a string into a Script.
//
// Users of this function must ensure themselves that the string does not
// contain unsafe script. Note in particular that '<' is dangerous, even when
// inside JavaScript strings, and so should always be forbidden or JavaScript
// escaped in user controlled input. For example, if
// "</script><script>evil</script>" were interpolated inside a JavaScript
// string,it would break out of the context of the original script element and
// "evil" would execute. Also note that within an HTML script (raw text)
// element, HTML character references, such as "&lt;" are not allowed. See
// http://www.w3.org/TR/html5/scripting-1.html#restrictions-for-contents-of-script-elements.
func ScriptFromStringKnownToSatisfyTypeContract(s string) safehtml.Script {
	return script(s)
}

// StyleFromStringKnownToSatisfyTypeContract converts a string into a Style.
//
// Users of thie function must ensure themselves that the string:
//    * Does not contain unsafe CSS.
//    * Does not contain literal angle brackets. Otherwise, it could be unsafe to
//      place a Style into the contents of a <style> element where it can't be
//      HTML escaped (see http://www.w3.org/International/questions/qa-escapes).
//      For example, if the Style containing
//      "font: 'foo <style/><script>evil</script>'" was interpolated within a
//      <style> tag, it would then break out of the style context into HTML.
//    * Does not end in a property value or property name context.
//      For example, a value of "background:url(\"" or "font-" does not satisfy
//      the Style type contract. This rule is enforced to ensure composability:
//      concatenating two incomplete strings that themselves do not contain unsafe
//      CSS can result in an overall string that does. For example, if
//      "javascript:evil())\"" is appended to "background:url(\"", the resulting
//      string may result in the execution of a malicious script.
//
// The string may, however, contain literal single or double quotes (for example,
// in the "content" property). Therefore, the entire style string must be
// escaped when used in a style attribute.
//
// The following example values comply with Style's type contract:
//    width: 1em;
//    height:1em;
//    width: 1em;height: 1em;
//    background:url('http://url');
//
// In addition, the empty string is safe for use in a style attribute.
//
// The following example values do NOT comply with this type's contract:
//    background: red    --- missing a trailing semi-colon
//    background:        --- missing a value and a trailing semi-colon
//    1em                --- missing an attribute name, which provides context
//                           for the value
//
// See also http://www.w3.org/TR/css3-syntax/.
func StyleFromStringKnownToSatisfyTypeContract(s string) safehtml.Style {
	return style(s)
}

// StyleSheetFromStringKnownToSatisfyTypeContract converts a string into a StyleSheet.
//
// Users of this function must ensure themselves that the string does not
// contain unsafe script. Note in particular that '<' is dangerous, even when
// inside CSS strings, and so should always be forbidden or CSS-escaped in
// user controlled input. For example, if
// "</style><script>evil</script>" were interpolated inside a CSS string, it
// would break out of the context of the original style element and "evil" would
// execute. Also note that within an HTML style (raw text) element, HTML
// character references, such as "&lt;", are not allowed.See
// http://www.w3.org/TR/html5/scripting-1.html#restrictions-for-contents-of-script-elements
// (Similar considerations apply to the style element.)
func StyleSheetFromStringKnownToSatisfyTypeContract(s string) safehtml.StyleSheet {
	return styleSheet(s)
}

// URLFromStringKnownToSatisfyTypeContract converts a string into a URL.
//
func URLFromStringKnownToSatisfyTypeContract(s string) safehtml.URL {
	return url(s)
}

// TrustedResourceURLFromStringKnownToSatisfyTypeContract converts a string into a TrustedResourceURL.
//
func TrustedResourceURLFromStringKnownToSatisfyTypeContract(s string) safehtml.TrustedResourceURL {
	return trustedResourceURL(s)
}

// IdentifierFromStringKnownToSatisfyTypeContract converts a string into a Identifier.
//
func IdentifierFromStringKnownToSatisfyTypeContract(s string) safehtml.Identifier {
	return identifier(s)
}
