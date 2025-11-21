// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"html"
	"regexp"
	"strings"

	"github.com/google/safehtml/internal/safehtmlutil"
	"github.com/google/safehtml"
)

// urlPrefixValidators maps URL and TrustedResourceURL sanitization contexts to functions return an error
// if the given string is unsafe to use as a URL prefix in that sanitization context.
var urlPrefixValidators = map[sanitizationContext]func(string) error{
	sanitizationContextURL:                     validateURLPrefix,
	sanitizationContextTrustedResourceURLOrURL: validateURLPrefix,
	sanitizationContextTrustedResourceURL:      validateTrustedResourceURLPrefix,
}

// startsWithFullySpecifiedSchemePattern matches strings that have a fully-specified scheme component.
// See RFC 3986 Section 3.
var startsWithFullySpecifiedSchemePattern = regexp.MustCompile(
	`^[[:alpha:]](?:[[:alnum:]]|[+.-])*:`)

// validateURLPrefix validates if the given non-empty prefix is a safe safehtml.URL prefix.
//
// Prefixes are considered unsafe if they end in an incomplete HTML character reference
// or percent-encoding character triplet.
//
// If the prefix contains a fully-specified scheme component, it is considered safe only if
// it starts with an allowed scheme. See safehtml.URLSanitized for more details.
//
// Otherwise, the prefix is safe only if it contains '/', '?', or '#', since the presence of any
// of these runes ensures that this prefix, when combined with some arbitrary suffix, cannot be
// interpreted as a part of a scheme.
func validateURLPrefix(prefix string) error {
	decoded, err := decodeURLPrefix(prefix)
	if err != nil {
		return err
	}
	switch {
	case startsWithFullySpecifiedSchemePattern.MatchString(decoded):
		if safehtml.URLSanitized(decoded).String() != decoded {
			return fmt.Errorf("URL prefix %q contains an unsafe scheme", prefix)
		}
	case !strings.ContainsAny(decoded, "/?#"):
		// If the URL prefix does not already have a ':' scheme delimiter, and does not contain
		// '/', '?', or '#', any ':' following this prefix will be intepreted as a scheme
		// delimiter, causing this URL prefix to be interpreted as being part of a scheme.
		// e.g. `<a href="java{{ "script:" }}alert(1)>`
		return fmt.Errorf("URL prefix %q is unsafe; it might be interpreted as part of a scheme", prefix)
	}
	return nil
}

// validateTrustedResourceURLPrefix validates if the given non-empty prefix is a safe
// safehtml.TrustedResourceURL prefix.
//
// Prefixes are considered unsafe if they end in an incomplete HTML character reference
// or percent-encoding character triplet.
//
// See safehtmlutil.IsSafeTrustedResourceURLPrefix for details on how the prefix is validated.
func validateTrustedResourceURLPrefix(prefix string) error {
	decoded, err := decodeURLPrefix(prefix)
	if err != nil {
		return err
	}
	if !safehtmlutil.IsSafeTrustedResourceURLPrefix(decoded) {
		return fmt.Errorf("%q is a disallowed TrustedResourceURL prefix", prefix)
	}
	return nil
}

// endsWithPercentEncodingPrefixPattern matches strings that end in an incomplete
// URL percent encoding triplet.
//
// See https://tools.ietf.org/html/rfc3986#section-2.1.
var endsWithPercentEncodingPrefixPattern = regexp.MustCompile(
	`%[[:xdigit:]]?$`)

// containsWhitespaceOrControlPattern matches strings that contain ASCII whitespace
// or control characters.
var containsWhitespaceOrControlPattern = regexp.MustCompile(`[[:space:]]|[[:cntrl:]]`)

// decodeURLPrefix returns the given prefix after it has been HTML-unescaped.
// It returns an error if the prefix:
//    * ends in an incomplete HTML character reference before HTML-unescaping,
//    * ends in an incomplete percent-encoding character triplet after HTML-unescaping, or
//    * contains whitespace before or after HTML-unescaping.
func decodeURLPrefix(prefix string) (string, error) {
	if containsWhitespaceOrControlPattern.MatchString(prefix) {
		return "", fmt.Errorf("URL prefix %q contains whitespace or control characters", prefix)
	}
	if err := validateDoesNotEndsWithCharRefPrefix(prefix); err != nil {
		return "", fmt.Errorf("URL %s", err)
	}
	decoded := html.UnescapeString(prefix)
	// Check again for whitespace that might have previously been masked by a HTML reference,
	// such as in "javascript&NewLine;".
	if containsWhitespaceOrControlPattern.MatchString(decoded) {
		return "", fmt.Errorf("URL prefix %q contains whitespace or control characters", prefix)
	}
	if endsWithPercentEncodingPrefixPattern.MatchString(decoded) {
		return "", fmt.Errorf("URL prefix %q ends with an incomplete percent-encoding character triplet", prefix)
	}
	return decoded, nil
}

func validateTrustedResourceURLSubstitution(args ...interface{}) (string, error) {
	input := safehtmlutil.Stringify(args...)
	if safehtmlutil.URLContainsDoubleDotSegment(input) {
		// Reject substitutions containing the ".." dot-segment to prevent the final TrustedResourceURL from referencing
		// a resource higher up in the path name hierarchy than the path specified in the prefix.
		return "", fmt.Errorf(`cannot substitute %q after TrustedResourceURL prefix: ".." is disallowed`, input)
	}
	return input, nil
}
