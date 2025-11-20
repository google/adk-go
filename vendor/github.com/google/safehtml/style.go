// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package safehtml

import (
	"bytes"
	"fmt"
	"regexp"
	"strings"
)

// A Style is an immutable string-like type which represents a sequence of CSS
// declarations (property_name1: property_value1; property_name2: property_value2; ...)
// and guarantees that its value will not cause untrusted script execution
// (cross-site scripting) when evaluated as CSS in a browser.
//
// Style's string representation can safely be:
//    * Interpolated as the content of a quoted HTML style attribute. However, the
//      Style string must be HTML-attribute-escaped before interpolation.
//    * Interpolated as the content of a {}-wrapped block within a StyleSheet.
//      '<' runes in the Style string must be CSS-escaped before interpolation.
//      The Style string is also guaranteed not to be able to introduce new
//      properties or elide existing ones.
//    * Interpolated as the content of a {}-wrapped block within an HTML <style>
//      element. '<' runes in the Style string must be CSS-escaped before interpolation.
//    * Assigned to the style property of a DOM node. The Style string should not
//      be escaped before being assigned to the property.
//
// In addition, values of this type are composable, that is, for any two Style
// values |style1| and |style2|, style1.style() + style2.style() is itself a
// value that satisfies the Style type constraint.
type Style struct {
	// We declare a Style not as a string but as a struct wrapping a string
	// to prevent construction of Style values through string conversion.
	str string
}

// StyleFromConstant constructs a Style with its underlying style set to the
// given style, which must be an untyped string constant, and panics if the
// style string does not pass basic syntax checks.
//
// Users of this function must ensure themselves that the style:
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
// The style may, however, contain literal single or double quotes (for example,
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
func StyleFromConstant(style stringConstant) Style {
	// TODO: implement UTF-8 interchange-validity checks and blocking of newlines
	// (including Unicode ones) and other whitespace characters (\t, \f) for Style and other safe types
	// in this package.
	if strings.ContainsAny(string(style), "<>") {
		panic(fmt.Sprintf("style string %q contains angle brackets", style))
	}
	if !strings.HasSuffix(string(style), ";") {
		panic(fmt.Sprintf("style string %q must end with ';'", style))
	}
	if !strings.Contains(string(style), ":") {
		panic(fmt.Sprintf("style string %q must contain at least one ':' to specify a property-value pair", style))
	}
	return Style{string(style)}
}

// String returns the string form of the Style.
func (s Style) String() string {
	return s.str
}

// StyleProperties contains property values for CSS properties whose names are
// the hyphen-separated form of the field names. These values will be validated
// by StyleFromProperties before being included in a Style.
//
// For example, BackgroundPosition contains the value for the
// "background-position" property, and Display contains the value for the "display"
// property.
//
type StyleProperties struct {
	// BackgroundImageURLs contains URL values for the background-image property.
	// These values val_1, val_2, ..., val_n will be passed through URLSanitized and CSS-escaped in
	// StyleFromProperties, then interpolated into to a comma-separated list of CSS URLs of the form
	//    url("val_1"), url("val_2"), ..., url("val_n")
	// See https://www.w3.org/TR/CSS2/syndata.html#value-def-uri and https://drafts.csswg.org/css-backgrounds-3/#layering.
	BackgroundImageURLs []string
	// FontFamily values are used, comma-separated, as the font-family property.
	//    * Names starting with a Latin alphabet runes and containing only Latin alphabets and hyphens will be included unquoted.
	//    * Names enclosed in double quote literals (e.g. `"21st Century"`) will be CSS-escaped without the outermost quotes,
	//      then included within double quotes.
	//    * All other names will be CSS-escaped, and included within double quotes.
	// See https://drafts.csswg.org/css-fonts-3/#font-family-prop.
	FontFamily []string
	// Display must consist of only ASCII alphabetic or '-' runes.
	// Non-conforming values will be replaced by InnocuousPropertyValue in
	// StyleFromProperties.
	Display string
	// The following values can only contain allowed runes, that is, alphanumerics,
	// space, tab, and the set [+-.!#%_/*]. In addition, comment markers "//", "/*",
	// and "*/" are disallowed. Non-conforming values will be replaced by
	// InnocuousPropertyValue in StyleFromProperties.
	BackgroundColor    string
	BackgroundPosition string
	BackgroundRepeat   string
	BackgroundSize     string
	Color              string
	Height             string
	Width              string
	Left               string
	Right              string
	Top                string
	Bottom             string
	FontWeight         string
	Padding            string
	// Note: this property might allow clickjacking, but the risk is limited without
	// the ability to set the position property to "absolute" or "fixed".
	ZIndex string
}

// identifierPattern matches a subset of valid <ident-token> values defined in
// https://www.w3.org/TR/css-syntax-3/#ident-token-diagram. This pattern matches all generic family name
// keywords defined in https://drafts.csswg.org/css-fonts-3/#family-name-value.
var identifierPattern = regexp.MustCompile(`^[a-zA-Z][-a-zA-Z]+$`)

// StyleFromProperties constructs a Style containining properties whose values
// are set in properties. The contents of the returned Style will be of the form
//    property_1:val_1;property2:val_2; ... ;property_n:val_n;
// This syntax is defined in https://www.w3.org/TR/css-style-attr/.
//
// All property values are validated and, if necessary, modified to ensure that their
// inclusion in a HTML style attribute does not result in untrusted script execution,
// the addition of new properties, or the removal of  existing properties. Please refer
// to the StyleProperties documentation for validation rules.
//
// The constructed Style is guaranteed to fulfill its type contract, but is not
// guaranteed to be semantically valid CSS.
func StyleFromProperties(properties StyleProperties) Style {
	// TODO: if this boilerplate code grows large, consider generating property names from Field names using reflection.
	var buf bytes.Buffer
	if len(properties.BackgroundImageURLs) > 0 {
		buf.WriteString("background-image:")
		for i, url := range properties.BackgroundImageURLs {
			if i > 0 {
				buf.WriteString(", ")
			}
			fmt.Fprintf(&buf, "url(\"%s\")", cssEscapeString(URLSanitized(url).String()))
		}
		buf.WriteString(";")
	}
	if len(properties.FontFamily) > 0 {
		buf.WriteString("font-family:")
		for i, name := range properties.FontFamily {
			if i > 0 {
				buf.WriteString(", ")
			}
			if identifierPattern.MatchString(name) {
				buf.WriteString(name)
				continue
			}
			unescaped := name
			if len(name) >= 3 && strings.HasPrefix(name, `"`) && strings.HasSuffix(name, `"`) {
				unescaped = name[1 : len(name)-1]
			}
			fmt.Fprintf(&buf, `"%s"`, cssEscapeString(unescaped))
		}
		buf.WriteByte(';')
	}
	if properties.Display != "" {
		fmt.Fprintf(&buf, "display:%s;", filter(properties.Display, safeEnumPropertyValuePattern))
	}
	if properties.BackgroundColor != "" {
		fmt.Fprintf(&buf, "background-color:%s;", filter(properties.BackgroundColor, safeRegularPropertyValuePattern))
	}
	if properties.BackgroundPosition != "" {
		fmt.Fprintf(&buf, "background-position:%s;", filter(properties.BackgroundPosition, safeRegularPropertyValuePattern))
	}
	if properties.BackgroundRepeat != "" {
		fmt.Fprintf(&buf, "background-repeat:%s;", filter(properties.BackgroundRepeat, safeRegularPropertyValuePattern))
	}
	if properties.BackgroundSize != "" {
		fmt.Fprintf(&buf, "background-size:%s;", filter(properties.BackgroundSize, safeRegularPropertyValuePattern))
	}
	if properties.Color != "" {
		fmt.Fprintf(&buf, "color:%s;", filter(properties.Color, safeRegularPropertyValuePattern))
	}
	if properties.Height != "" {
		fmt.Fprintf(&buf, "height:%s;", filter(properties.Height, safeRegularPropertyValuePattern))
	}
	if properties.Width != "" {
		fmt.Fprintf(&buf, "width:%s;", filter(properties.Width, safeRegularPropertyValuePattern))
	}
	if properties.Left != "" {
		fmt.Fprintf(&buf, "left:%s;", filter(properties.Left, safeRegularPropertyValuePattern))
	}
	if properties.Right != "" {
		fmt.Fprintf(&buf, "right:%s;", filter(properties.Right, safeRegularPropertyValuePattern))
	}
	if properties.Top != "" {
		fmt.Fprintf(&buf, "top:%s;", filter(properties.Top, safeRegularPropertyValuePattern))
	}
	if properties.Bottom != "" {
		fmt.Fprintf(&buf, "bottom:%s;", filter(properties.Bottom, safeRegularPropertyValuePattern))
	}
	if properties.FontWeight != "" {
		fmt.Fprintf(&buf, "font-weight:%s;", filter(properties.FontWeight, safeRegularPropertyValuePattern))
	}
	if properties.Padding != "" {
		fmt.Fprintf(&buf, "padding:%s;", filter(properties.Padding, safeRegularPropertyValuePattern))
	}
	if properties.ZIndex != "" {
		fmt.Fprintf(&buf, "z-index:%s;", filter(properties.ZIndex, safeRegularPropertyValuePattern))
	}

	return Style{buf.String()}
}

// InnocuousPropertyValue is an innocuous property generated by filter when its input unsafe.
const InnocuousPropertyValue = "zGoSafezInvalidPropertyValue"

// safeRegularPropertyValuePattern matches strings that are safe to use as property values.
// Specifically, it matches string where every '*' or '/' is followed by end-of-text or a safe rune
// (i.e. alphanumberics or runes in the set [+-.!#%_ \t]). This regex ensures that the following
// are disallowed:
//    * "/*" and "*/", which are CSS comment markers.
//    * "//", even though this is not a comment marker in the CSS specification. Disallowing
//      this string minimizes the chance that browser peculiarities or parsing bugs will allow
//      sanitization to be bypassed.
//    * '(' and ')', which can be used to call functions.
//    * ',', since it can be used to inject extra values into a property.
//    * Runes which could be matched on CSS error recovery of a previously malformed token, such as '@'
//      and ':'. See http://www.w3.org/TR/css3-syntax/#error-handling.
var safeRegularPropertyValuePattern = regexp.MustCompile(`^(?:[*/]?(?:[0-9a-zA-Z+-.!#%_ \t]|$))*$`)

// safeEnumPropertyValuePattern matches strings that are safe to use as enumerated property values.
// Specifically, it matches strings that contain only alphabetic and '-' runes.
var safeEnumPropertyValuePattern = regexp.MustCompile(`^[a-zA-Z-]*$`)

// filter returns value if it matches pattern. Otherwise, it returns InnocuousPropertyValue.
func filter(value string, pattern *regexp.Regexp) string {
	if !pattern.MatchString(value) {
		return InnocuousPropertyValue
	}
	return value
}

// cssEscapeString escapes s so that it is safe to put between "" to form a CSS <string-token>.
// See syntax at https://www.w3.org/TR/css-syntax-3/#string-token-diagram.
//
// On top of the escape sequences required in <string-token>, this function also escapes
// control runes to minimize the risk of these runes triggering browser-specific bugs.
func cssEscapeString(s string) string {
	var b bytes.Buffer
	b.Grow(len(s))
	// TODO: consider optmizations (e.g. ranging over bytes, batching writes of contiguous sequences of unescaped runes) if
	// performance becomes an issue.
	for _, c := range s {
		switch {
		case c == '\u0000':
			// Replace the NULL byte according to https://www.w3.org/TR/css-syntax-3/#input-preprocessing.
			// We take this extra precaution in case the user agent fails to handle NULL properly.
			b.WriteString("\uFFFD")
		case c == '<', // Prevents breaking out of a style element with `</style>`. Escape this in case the Style user forgets to.
			c == '"', c == '\\', // Must be CSS-escaped in <string-token>. U+000A line feed is handled in the next case.
			c <= '\u001F', c == '\u007F', // C0 control codes
			c >= '\u0080' && c <= '\u009F', // C1 control codes
			c == '\u2028', c == '\u2029':   // Unicode newline characters
			// See CSS escape sequence syntax at https://www.w3.org/TR/css-syntax-3/#escape-diagram.
			fmt.Fprintf(&b, "\\%06X", c)
		default:
			b.WriteRune(c)
		}
	}
	return b.String()
}
