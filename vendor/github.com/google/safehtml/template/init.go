// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package template

import (
	"github.com/google/safehtml/internal/template/raw"
)

// The following functions are used by package uncheckedconversions
// (via package raw) to create TrustedSource and TrustedTemplate values
// from plain strings.

func trustedSourceRaw(s string) TrustedSource {
	return TrustedSource{s}
}

func trustedTemplateRaw(s string) TrustedTemplate {
	return TrustedTemplate{s}
}

func init() {
	raw.TrustedSource = trustedSourceRaw
	raw.TrustedTemplate = trustedTemplateRaw
}
