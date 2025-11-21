// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package template

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"flag"
)

// A TrustedSource is an immutable string-like type referencing
// trusted template files under application control. It can be passed to
// template-parsing functions and methods to safely load templates
// without the risk of untrusted template execution.
//
// In order to ensure that an attacker cannot influence the TrustedSource
// value, a TrustedSource can be instantiated only from untyped string
// constants, command-line flags, and other application-controlled strings, but
// never from arbitrary string values potentially representing untrusted user input.
//
// Note that TrustedSource's constructors cannot truly guarantee that the
// templates it references are not attacker-controlled; it can guarantee only that
// the path to the template itself is under application control. Users of these
// constructors must ensure themselves that TrustedSource never references
// attacker-controlled files or directories that contain such files.
type TrustedSource struct {
	// We declare a TrustedSource not as a string but as a struct wrapping a string
	// to prevent construction of TrustedSource values through string conversion.
	src string
}

// TrustedSourceFromConstant constructs a TrustedSource with its underlying
// src set to the given src, which must be an untyped string constant.
//
// No runtime validation or sanitization is performed on src; being under
// application control, it is simply assumed to comply with the TrustedSource type
// contract.
func TrustedSourceFromConstant(src stringConstant) TrustedSource {
	return TrustedSource{string(src)}
}

// TrustedSourceFromConstantDir constructs a TrustedSource calling path/filepath.Join on
// an application-controlled directory path, which must be an untyped string constant,
// a TrustedSource, and a dynamic filename. It returns an error if filename contains
// filepath or list separators, since this might cause the resulting path to reference a
// file outside of the given directory.
//
// dir or src may be empty if either of these path segments are not required.
func TrustedSourceFromConstantDir(dir stringConstant, src TrustedSource, filename string) (TrustedSource, error) {
	if i := strings.IndexAny(filename, string([]rune{filepath.Separator, filepath.ListSeparator})); i != -1 {
		return TrustedSource{}, fmt.Errorf("filename %q must not contain the separator %q", filename, filename[i])
	}
	if filename == ".." {
		return TrustedSource{}, fmt.Errorf("filename must not be the special name %q", filename)
	}
	return TrustedSource{filepath.Join(string(dir), src.String(), filename)}, nil
}

// TrustedSourceJoin is a wrapper around path/filepath.Join that returns a
// TrustedSource formed by joining the given path elements into a single path,
// adding an OS-specific path separator if necessary.
func TrustedSourceJoin(elem ...TrustedSource) TrustedSource {
	return TrustedSource{filepath.Join(trustedSourcesToStrings(elem)...)}
}

// TrustedSourceFromFlag returns a TrustedSource containing the string
// representation of the retrieved value of the flag.
//
// In a server setting, flags are part of the application's deployment
// configuration and are hence considered application-controlled.
func TrustedSourceFromFlag(value flag.Value) TrustedSource {
	return TrustedSource{fmt.Sprint(value.String())}
}

// TrustedSourceFromEnvVar is a wrapper around os.Getenv that
// returns a TrustedSource containing the value of the environment variable
// named by the key. It returns the value, which will be empty if the variable
// is not present. To distinguish between an empty value and an unset value,
// use os.LookupEnv.
//
// In a server setting, environment variables are part of the application's
// deployment configuration and are hence considered application-controlled.
func TrustedSourceFromEnvVar(key stringConstant) TrustedSource {
	return TrustedSource{os.Getenv(string(key))}
}

// String returns the string form of the TrustedSource.
func (t TrustedSource) String() string {
	return t.src
}

func trustedSourcesToStrings(paths []TrustedSource) []string {
	ret := make([]string, 0, len(paths))
	for _, p := range paths {
		ret = append(ret, p.String())
	}
	return ret
}
