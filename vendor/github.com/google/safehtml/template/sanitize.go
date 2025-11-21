// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"regexp"
	"strings"
)

// sanitizerForContext returns an ordered list of function names that will be called to
// sanitize data values found in the HTML context defined by c.
func sanitizerForContext(c context) ([]string, error) {
	switch c.state {
	case stateTag, stateAttrName, stateAfterName:
		return nil, fmt.Errorf("actions must not affect element or attribute names")
	case stateHTMLCmt:
		return []string{sanitizeHTMLCommentFuncName}, nil
	}
	if len(c.element.names) == 0 && c.element.name == "" && c.state == stateText {
		// Not in an HTML element.
		return []string{sanitizeHTMLFuncName}, nil
	}
	if c.attr.name != "" || len(c.attr.names) > 0 {
		// We are in an attribute value context.
		if c.delim != delimDoubleQuote && c.delim != delimSingleQuote {
			// TODO: consider disallowing single-quoted or unquoted attribute values completely, even in hardcoded template text.
			return nil, fmt.Errorf("unquoted attribute values disallowed")
		}
		return sanitizersForAttributeValue(c)
	}
	// Otherwise, we are in an element content context.
	elementContentSanitizer, err := sanitizerForElementContent(c)
	return appendIfNotEmpty([]string{}, elementContentSanitizer), err
}

// appendIfNotEmpty appends the given strings that are non-empty to the given slice.
func appendIfNotEmpty(slice []string, strings ...string) []string {
	for _, s := range strings {
		if s != "" {
			slice = append(slice, s)
		}
	}
	return slice
}

// sanitizersForAttributeValue returns a list of names of functions that will be
// called in order to sanitize data values found the HTML attribtue value context c.
func sanitizersForAttributeValue(c context) ([]string, error) {
	// Ensure that all combinations of element and attribute names for this context results
	// in the same attribute value sanitization context.
	var elems, attrs []string
	if len(c.element.names) == 0 {
		elems = []string{c.element.name}
	} else {
		elems = c.element.names
	}
	if len(c.attr.names) == 0 {
		attrs = []string{c.attr.name}
	} else {
		attrs = c.attr.names
	}
	var sc0 sanitizationContext
	var elem0, attr0 string
	for i, elem := range elems {
		for j, attr := range attrs {
			sc, err := sanitizationContextForAttrVal(elem, attr, c.linkRel)
			if err != nil {
				if len(elems) == 1 && len(attrs) == 1 {
					return nil, err
				}
				return nil, fmt.Errorf(`conditional branch with {element=%q, attribute=%q} results in sanitization error: %s`, elem, attr, err)
			}
			if i == 0 && j == 0 {
				sc0, elem0, attr0 = sc, elem, attr
				continue
			}
			if sc != sc0 {
				return nil, fmt.Errorf(
					`conditional branches end in different attribute value sanitization contexts: {element=%q, attribute=%q} has sanitization context %q, {element=%q, attribute=%q} has sanitization context %q`,
					elem0, attr0, sc0, elem, attr, sc)
			}
		}
	}
	if sc0.isEnum() && c.attr.value != "" {
		return nil, fmt.Errorf("partial substitutions are disallowed in the %q attribute value context of a %q element", c.attr.name, c.element.name)
	}
	if sc0 == sanitizationContextStyle && c.attr.value != "" {
		if err := validateDoesNotEndsWithCharRefPrefix(c.attr.value); err != nil {
			return nil, fmt.Errorf("action cannot be interpolated into the %q attribute value of this %q element: %s", c.attr.name, c.element.name, err)
		}
	}
	// ret is a stack of sanitizer names that will be built in reverse.
	var ret []string
	// All attribute values must be HTML-escaped at run time by sanitizeHTML to eliminate
	// any HTML markup that can cause the HTML parser to transition out of the attribute value state.
	// These attribute values will later be HTML-unescaped by the HTML parser in the browser.
	ret = append(ret, sanitizeHTMLFuncName)
	sanitizer := sc0.sanitizerName()
	if !sc0.isURLorTrustedResourceURL() {
		return reverse(appendIfNotEmpty(ret, sanitizer)), nil
	}
	urlAttrValPrefix := c.attr.value
	if urlAttrValPrefix == "" {
		// Attribute value prefixes in URL or TrustedResourceURL sanitization contexts
		// must sanitized and normalized.
		return reverse(appendIfNotEmpty(ret, normalizeURLFuncName, sanitizer)), nil
	}
	// Action occurs after a URL or TrustedResourceURL prefix.
	if c.attr.ambiguousValue {
		return nil, fmt.Errorf("actions must not occur after an ambiguous URL prefix in the %q attribute value context of a %q element", c.attr.name, c.element.name)
	}
	validator, ok := urlPrefixValidators[sc0]
	if !ok {
		return nil, fmt.Errorf("cannot validate attribute value prefix %q in the %q sanitization context", c.attr.value, sc0)
	}
	if err := validator(c.attr.value); err != nil {
		return nil, fmt.Errorf("action cannot be interpolated into the %q URL attribute value of this %q element: %s", c.attr.name, c.element.name, err)
	}
	switch {
	case sc0 == sanitizationContextTrustedResourceURL:
		// Untrusted data that occurs anywhere after TrustedResourceURL prefix must be query-escaped
		// to prevent the injection of any new path segments or URL components. Moreover, they must
		// not contain any ".." dot-segments.
		ret = append(ret, queryEscapeURLFuncName, validateTrustedResourceURLSubstitutionFuncName)
	case strings.ContainsAny(urlAttrValPrefix, "#?"):
		// For URLs, we only escape in the query or fragment part to prevent the injection of new query
		// parameters or fragments.
		ret = append(ret, queryEscapeURLFuncName)
	default:
		ret = append(ret, normalizeURLFuncName)
	}
	return reverse(ret), nil
}

// reverse reverses s and returns it.
func reverse(s []string) []string {
	for head, tail := 0, len(s)-1; head < tail; head, tail = head+1, tail-1 {
		s[head], s[tail] = s[tail], s[head]
	}
	return s
}

// sanitizationContextForAttrVal returns the sanitization context for attr when it
// appears within element.
func sanitizationContextForAttrVal(element, attr, linkRel string) (sanitizationContext, error) {
	if element == "link" && attr == "href" {
		// Special case: safehtml.URL values are allowed in a link element's href attribute if that element's
		// rel attribute possesses certain values.
		relVals := strings.Fields(linkRel)
		for _, val := range relVals {
			if urlLinkRelVals[val] {
				return sanitizationContextTrustedResourceURLOrURL, nil
			}
		}
	}
	if dataAttributeNamePattern.MatchString(attr) {
		// Special case: data-* attributes are specified by HTML5 to hold custom data private to
		// the page or application; they should not be interpreted by browsers. Therefore, no
		// sanitization is required for these attribute values.
		return sanitizationContextNone, nil
	}
	if sc, ok := elementSpecificAttrValSanitizationContext[attr][element]; ok {
		return sc, nil
	}
	sc, isAllowedAttr := globalAttrValSanitizationContext[attr]
	_, isAllowedElement := elementContentSanitizationContext[element]
	if isAllowedAttr && (isAllowedElement || allowedVoidElements[element]) {
		// Only sanitize attributes that appear in elements whose semantics are known.
		// Thes attributes might have different semantics in other standard or custom
		// elements that our sanitization policy does not handle correctly.
		return sc, nil
	}
	return 0, fmt.Errorf("actions must not occur in the %q attribute value context of a %q element", attr, element)
}

// dataAttributeNamePattern matches valid data attribute names.
// This pattern is conservative and matches only a subset of the valid names defined in
// https://html.spec.whatwg.org/multipage/dom.html#embedding-custom-non-visible-data-with-the-data-*-attributes
var dataAttributeNamePattern = regexp.MustCompile(`^data-[a-z_][-a-z0-9_]*$`)

// endsWithCharRefPrefixPattern matches strings that end in an incomplete
// HTML character reference.
//
// See https://html.spec.whatwg.org/multipage/syntax.html#character-references.
var endsWithCharRefPrefixPattern = regexp.MustCompile(
	`&(?:[[:alpha:]][[:alnum:]]*|#(?:[xX][[:xdigit:]]*|[[:digit:]]*))?$`)

// validateDoesNotEndsWithCharRefPrefix returns an error only if the given prefix ends
// with an incomplete HTML character reference.
func validateDoesNotEndsWithCharRefPrefix(prefix string) error {
	if endsWithCharRefPrefixPattern.MatchString(prefix) {
		return fmt.Errorf(`prefix %q ends with an incomplete HTML character reference; did you mean "&amp;" instead of "&"?`, prefix)
	}
	return nil
}

// sanitizerForElementContent returns the name of the function that will be called
// to sanitize data values found in the HTML element content context c.
func sanitizerForElementContent(c context) (string, error) {
	// Ensure that all other possible element names for this context result in the same
	// element content sanitization context.
	var elems []string
	if len(c.element.names) == 0 {
		elems = []string{c.element.name}
	} else {
		elems = c.element.names
	}
	var sc0 sanitizationContext
	var elem0 string
	for i, elem := range elems {
		var sc sanitizationContext
		var err error
		if elem == "" {
			// Special case: an empty element name represents a context outside of a HTML element.
			sc = sanitizationContextHTML
		} else {
			sc, err = sanitizationContextForElementContent(elem)
		}
		if err != nil {
			if len(elems) == 1 {
				return "", err
			}
			return "", fmt.Errorf(`conditional branch with element %q results in sanitization error: %s`, elem, err)
		}
		if i == 0 {
			sc0, elem0 = sc, elem
			continue
		}
		if sc != sc0 {
			return "",
				fmt.Errorf(`conditional branches end in different element content sanitization contexts: element %q has sanitization context %q, element %q has sanitization context %q`,
					elem0, sc0, elem, sc)
		}
	}
	return sc0.sanitizerName(), nil
}

// sanitizationContextForElementContent returns the element content sanitization context for the given element.
func sanitizationContextForElementContent(element string) (sanitizationContext, error) {
	sc, ok := elementContentSanitizationContext[element]
	if !ok {
		return 0, fmt.Errorf("actions must not occur in the element content context of a %q element", element)
	}
	return sc, nil
}

// sanitizeHTMLComment returns the empty string regardless of input.
// Comment content does not correspond to any parsed structure or
// human-readable content, so the simplest and most secure policy is to drop
// content interpolated into comments.
// This approach is equally valid whether or not static comment content is
// removed from the template.
func sanitizeHTMLComment(_ ...interface{}) string {
	return ""
}
