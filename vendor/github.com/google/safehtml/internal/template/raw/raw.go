// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

// Package raw provides a coordination point for package safehtml/template and
// package safehtml/template/uncheckedconversions. raw must be imported only by
// these two packages.
package raw

// TrustedSource is the raw constructor for a template.TrustedSource.
var TrustedSource interface{}

// TrustedTemplate is the raw constructor for a template.TrustedTemplate.
var TrustedTemplate interface{}
