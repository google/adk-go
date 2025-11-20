// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package safehtml

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"flag"
	"github.com/google/safehtml/internal/safehtmlutil"
)

// A TrustedResourceURL is an immutable string-like type referencing the
// application’s own, trusted resources. It can be used to safely load scripts,
// CSS and other sensitive resources without the risk of untrusted code execution.
// For example, it is unsafe to insert a plain string in a
//
//     <script src=“...”></script>
//
// context since the URL may originate from untrusted user input and the
// script it is pointing to may thus be controlled by an attacker. It is,
// however, safe to use a TrustedResourceURL since its value is known to never
// have left application control.
//
// In order to ensure that an attacker cannot influence the TrustedResourceURL
// value, a TrustedResourceURL can only be instantiated from compile-time
// constant string literals, command-line flags or a combination of the two,
// but never from arbitrary string values potentially representing untrusted user input.
//
// Additionally, TrustedResourceURLs can be serialized and passed along within
// the application via protocol buffers. It is the application’s responsibility
// to ensure that the protocol buffers originate from within the application
// itself and not from an external entity outside its trust domain.
//
// Note that TrustedResourceURLs can also use absolute paths (starting with '/')
// and relative paths. This allows the same binary to be used for different
// hosts without hard-coding the hostname in a string literal.
type TrustedResourceURL struct {
	// We declare a TrustedResourceURL not as a string but as a struct wrapping a string
	// to prevent construction of TrustedResourceURL values through string conversion.
	str string
}

// TrustedResourceURLWithParams constructs a new TrustedResourceURL with the
// given key-value pairs added as query parameters.
//
// Map entries with empty keys or values are ignored. The order of appended
// keys is guaranteed to be stable but may differ from the order in input.
func TrustedResourceURLWithParams(t TrustedResourceURL, params map[string]string) TrustedResourceURL {
	url := t.str
	var fragment string
	if i := strings.IndexByte(url, '#'); i != -1 {
		// The fragment identifier component will always appear at the end
		// of the URL after the query segment. It is therefore safe to
		// trim the fragment from the tail of the URL and re-append it after
		// all query parameters have been added.
		// See https://tools.ietf.org/html/rfc3986#appendix-A.
		fragment = url[i:]
		url = url[:i]
	}
	sep := "?"
	if i := strings.IndexRune(url, '?'); i != -1 {
		// The first "?" in a URL indicates the start of the query component.
		// See https://tools.ietf.org/html/rfc3986#section-3.4
		if i == len(url)-1 {
			sep = ""
		} else {
			sep = "&"
		}
	}
	stringParams := make([]string, 0, len(params))
	for k, v := range params {
		if k == "" || v == "" {
			continue
		}
		stringParam := safehtmlutil.QueryEscapeURL(k) + "=" + safehtmlutil.QueryEscapeURL(v)
		stringParams = append(stringParams, stringParam)
	}
	if len(stringParams) > 0 {
		sort.Strings(stringParams)
		url += sep + strings.Join(stringParams, "&")
	}
	return TrustedResourceURL{url + fragment}
}

// TrustedResourceURLFromConstant constructs a TrustedResourceURL with its underlying
// URL set to the given url, which must be an untyped string constant.
//
// No runtime validation or sanitization is performed on url; being under
// application control, it is simply assumed to comply with the TrustedResourceURL type
// contract.
func TrustedResourceURLFromConstant(url stringConstant) TrustedResourceURL {
	return TrustedResourceURL{string(url)}
}

// TrustedResourceURLFormatFromConstant constructs a TrustedResourceURL from a
// format string, which must be an untyped string constant, and string arguments.
//
// Arguments are specified as a map of labels, which must contain only alphanumeric
// and '_' runes, to string values. Each `%{<label>}` marker in the format string is
// replaced by the string value identified by <label> after it has been URL-escaped.
// Arguments that do not match any label in the format string are ignored.
//
// The format string must have a prefix of one of the following forms:
//    * `https://<origin>/`
//    * `//<origin>/`
//    * `/<pathStart>`
//    * `about:blank#`
//
// `<origin>` must contain only alphanumerics, '.', ':', '[', ']', or '-', and
// `<pathStart>` is any character except `/` and `\`.
func TrustedResourceURLFormatFromConstant(format stringConstant, args map[string]string) (TrustedResourceURL, error) {
	return trustedResourceURLFormat(string(format), args)
}

// TrustedResourceURLFormatFromFlag is a variant of TrustedResourceURLFormatFromConstant
// that constructs a TrustedResourceURL from a format string, which is given as a flag.Value,
// and string arguments.
//
// See TrustedResourceURLFormatFromConstant for more details about format
// string markers and validation.
func TrustedResourceURLFormatFromFlag(format flag.Value, args map[string]string) (TrustedResourceURL, error) {
	return trustedResourceURLFormat(fmt.Sprint(format.String()), args)
}

func trustedResourceURLFormat(format string, args map[string]string) (TrustedResourceURL, error) {
	if !safehtmlutil.IsSafeTrustedResourceURLPrefix(format) {
		return TrustedResourceURL{}, fmt.Errorf("%q is a disallowed TrustedResourceURL format string", format)
	}
	var err error
	ret := trustedResourceURLFormatMarkerPattern.ReplaceAllStringFunc(format, func(match string) string {
		argName := match[len("%{") : len(match)-len("}")]
		argVal, ok := args[argName]
		if !ok {
			if err == nil {
				// Report an error for the first missing argument.
				err = fmt.Errorf("expected argument named %q", argName)
			}
			return ""
		}
		if safehtmlutil.URLContainsDoubleDotSegment(argVal) {
			// Reject values containing the ".." dot-segment to prevent the final TrustedResourceURL from referencing
			// a resource higher up in the path name hierarchy than the path specified in the prefix.
			err = fmt.Errorf(`argument %q with value %q must not contain ".."`, argName, argVal)
			return ""
		}
		// QueryEscapeURL escapes some non-reserved characters in the path
		// segment (e.g. '/' and '?') in order to prevent the injection of any new path
		// segments or URL components.
		return safehtmlutil.QueryEscapeURL(argVal)
	})
	return TrustedResourceURL{ret}, err
}

// trustedResourceURLFormatMarkerPattern matches markers in TrustedResourceURLFormat
// format strings.
var trustedResourceURLFormatMarkerPattern = regexp.MustCompile(`%{[[:word:]]+}`)

// TrustedResourceURLFromFlag returns a TrustedResourceURL containing the string
// representation of the retrieved value of the flag.
//
// In a server setting, flags are part of the application's deployment
// configuration and are hence considered application-controlled.
func TrustedResourceURLFromFlag(value flag.Value) TrustedResourceURL {
	return TrustedResourceURL{fmt.Sprint(value.String())}
}

// String returns the string form of the TrustedResourceURL.
func (t TrustedResourceURL) String() string {
	return t.str
}

// TrustedResourceURLAppend URL-escapes a string and appends it to the TrustedResourceURL.
//
// This function can only be used if the TrustedResourceURL has a prefix of one of the following
// forms:
//    * `https://<origin>/`
//    * `//<origin>/`
//    * `/<pathStart>`
//    * `about:blank#`
//
// `<origin>` must contain only alphanumerics, '.', ':', '[', ']', or '-', and
// `<pathStart>` is any character except `/` and `\`.
func TrustedResourceURLAppend(t TrustedResourceURL, s string) (TrustedResourceURL, error) {
	if !safehtmlutil.IsSafeTrustedResourceURLPrefix(t.str) {
		return TrustedResourceURL{}, fmt.Errorf("cannot append to TrustedResourceURL %q because it has an unsafe prefix", t)
	}
	return TrustedResourceURL{t.str + safehtmlutil.QueryEscapeURL(s)}, nil
}
